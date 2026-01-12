"""Text extraction tool for prose-embedded financial data"""
import re
import time
from typing import Any

from src.models.tool_schemas import (
    TextExtractParams,
    TextConstantParams,
    TextExtractionResult,
    TextExtractionResponse,
)
from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


class TextTool:
    """
    Extract numeric values from unstructured financial text (pre_text and post_text).
    
    NEW APPROACH: Single-stage LLM extraction with verification
    - LLM receives structured params from planner (keywords, year, unit, value_context)
    - LLM extracts value with reasoning and evidence text
    - Python verification checks value exists in evidence
    - Handles complex patterns like "respectively" lists naturally
    
    Use cases:
    - Extract values mentioned in prose but not in tables
    - Extract from complex list structures ("(i) X for $Y (ii) Z for $W, respectively")
    - Extract year-specific values from narrative text
    """
    
    def __init__(self, tracker = None) -> None:
        # Common financial units and their multipliers
        self.unit_multipliers = {
            "million": 1_000_000,
            "millions": 1_000_000,
            "billion": 1_000_000_000,
            "billions": 1_000_000_000,
            "thousand": 1_000,
            "thousands": 1_000,
        }

        # LLM client will be initialized lazily when needed (allows testing without API key)
        self._llm_client = None
        self.model = Config.OPENAI_MODEL_SELECTOR  # Use selector model for structured extraction
        self.tracker = tracker
    
    @property
    def llm_client(self):
        """Lazy initialization of LLM client"""
        if self._llm_client is None:
            from src.services.llm_client import get_llm_client
            self._llm_client = get_llm_client()
        return self._llm_client
    
    async def execute(
        self,
        action: str,
        params: dict[str, Any],
        pre_text: str,
        post_text: str,
        question: str | None = None,
    ) -> float | dict[str, Any]:
        """
        Execute text extraction operation.
        
        Args:
            action: One of 'extract_numeric', 'extract_constant'
            params: Action-specific parameters with structured guidance:
                - keywords: List of keywords from question analysis
                - year: Year mentioned in question (if any)
                - unit: Expected unit (million, billion, thousand, none)
                - value_context: What the value represents (from planner)
            pre_text: Text before table
            post_text: Text after table
            question: Current question for context
        
        Returns:
            dict with {"value": float} for executor consumption
        """
        if action == "extract_numeric":
            value = await self._extract_numeric(
                params=params,
                pre_text=pre_text,
                post_text=post_text,
                question=question,
            )
            return {"value": value}
        elif action == "extract_constant":
            return await self._extract_constant(
                TextConstantParams(**params),
                pre_text,
                post_text
            )
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _extract_numeric(
        self,
        params: dict[str, Any],
        pre_text: str,
        post_text: str,
        question: str | None = None,
    ) -> float:
        """
        Extract numeric value using direct LLM extraction with verification.
        
        NEW APPROACH:
        1. Receive structured params from planner (keywords, year, unit, value_context)
        2. LLM extracts value with reasoning and evidence text
        3. Python verification checks value exists in evidence
        
        Args:
            params: Extraction parameters with:
                - keywords: List of keywords to guide extraction
                - year: Year filter (optional)
                - unit: Expected unit
                - value_context: What the value represents
                - text_source: "pre_text" or "post_text"
            pre_text: Text before table
            post_text: Text after table
            question: Current question for context (required)
            
        Returns:
            Extracted numeric value (float)
            
        Raises:
            ValueError: If extraction fails or verification fails
        """
        if not question:
            raise ValueError("text_tool.extract_numeric requires the current question")
        
        # Extract params with defaults
        keywords = params.get("keywords", [])
        year = params.get("year")
        unit = params.get("unit", "million")
        value_context = params.get("value_context", "")
        text_source = params.get("text_source", "post_text")
        
        # Select the appropriate text
        source_text = post_text if text_source == "post_text" else pre_text
        
        if not source_text or not source_text.strip():
            raise ValueError(f"No text available in {text_source}")
        
        logger.info(f"Extracting from {text_source} with keywords: {keywords}, year: {year}, unit: {unit}")
        
        # Call LLM for extraction
        extraction_response = await self._llm_extract_value(
            question=question,
            text=source_text,
            keywords=keywords,
            year=year,
            unit=unit,
            value_context=value_context,
        )
        
        logger.info(f"LLM extracted value: {extraction_response.value}")
        logger.info(f"LLM reasoning: {extraction_response.reasoning}")
        logger.info(f"LLM evidence: {extraction_response.evidence_text[:200]}...")
        
        # Verify extraction to prevent hallucination
        verified_value = self._verify_extraction(
            claimed_value=extraction_response.value,
            evidence_text=extraction_response.evidence_text,
            full_text=source_text,
            unit=unit,
        )
        
        logger.info(f"Verification passed. Final value: {verified_value}")
        
        return verified_value
    
    async def _llm_extract_value(
        self,
        question: str,
        text: str,
        keywords: list[str],
        year: str | None,
        unit: str,
        value_context: str,
    ) -> TextExtractionResponse:
        """
        Use LLM to directly extract numeric value from text.
        
        Args:
            question: Current question
            text: Full text to search (pre_text or post_text)
            keywords: Keywords from planner to guide extraction
            year: Year filter (if mentioned in question)
            unit: Expected unit (million, billion, thousand, none)
            value_context: What the value represents (from planner)
            
        Returns:
            TextExtractionResponse with value, reasoning, and evidence_text
        """
        if self.llm_client is None:
            raise ValueError("text_tool requires OPENAI_API_KEY for extraction")
        
        # Build system prompt
        system_prompt = """You are a precise financial value extraction assistant. Your task is to extract a specific numeric value from the provided text.

CRITICAL INSTRUCTIONS:
1. Extract ONLY the raw numeric value mentioned in the text - DO NOT compute, calculate, or derive answers
2. Extract the numeric value in the expected unit (e.g., if unit is "million", extract 6.0 from "$ 6.0 million")
3. Provide brief reasoning explaining how you identified this value
4. Provide the exact text snippet (verbatim quote) containing this value as evidence

CRITICAL RESTRICTION - EXTRACTION ONLY:
- Your role is EXTRACTION ONLY - find and return the numeric value as stated in the text
- DO NOT perform calculations, ratios, percentages, or any mathematical operations
- DO NOT answer the question - just extract the requested value
- If the value_context asks for "purchase price", extract the purchase price amount from text, NOT any calculated percentage
- If the value_context asks for "net tangible assets", extract that amount from text, NOT any calculated ratio
- Extract each value independently - do not compute relationships between values

IMPORTANT:
- The evidence_text MUST be copied EXACTLY from the source text (verbatim quote)
- Do NOT paraphrase or modify the evidence text
- The evidence text will be verified against the source to prevent hallucination
- Extract the raw numeric value in the specified unit (e.g., 6.0 for "$ 6.0 million" when unit is "million")

HANDLING COMPLEX PATTERNS:
- For "respectively" patterns like "(i) A for $X (ii) B for $Y (iii) C for $Z, respectively", use the keywords and context to select the correct value
- The full text is provided - do not require sentence boundaries
- Use year filters and keywords to disambiguate when multiple values are present

EXAMPLE 1 - Correct Extraction:
Question: "what was the total cash for towers acquisitions in 2005?"
Text: "... the company used cash to acquire a total of (i) 293 towers for $ 44.0 million (ii) 84 towers for $ 14.3 million and (iii) 30 towers for approximately $ 6.0 million in cash, respectively ..."
Keywords: ["towers", "cash", "acquisitions"]
Year: "2005"
Unit: "million"
Value Context: "total cash for 30 towers acquisition"

Response:
{
  "value": 6.0,
  "reasoning": "The text lists three acquisitions with 'respectively': (iii) shows 30 towers for $ 6.0 million. This matches the '30 towers' mentioned in the context.",
  "evidence_text": "30 towers for approximately $ 6.0 million in cash"
}

EXAMPLE 2 - Correct Extraction (No Computation):
Question: "what percentage did the total of net tangible assets acquired represent in relation to the purchase price?"
Text: "... acquired corphealth, inc. for cash consideration of approximately $ 54.2 million ... net tangible assets acquired of $ 6.0 million ..."
Keywords: ["purchase price", "corphealth"]
Year: "2005"
Unit: "thousand"
Value Context: "purchase price paid for CorpHealth, Inc. on December 20, 2005"

CORRECT Response:
{
  "value": 54200.0,
  "reasoning": "The text states 'cash consideration of approximately $ 54.2 million' for the CorpHealth acquisition. The requested unit is 'thousand', so $54.2 million = 54,200 thousand.",
  "evidence_text": "acquired corphealth , inc. , or corphealth , a behavioral health care management company , for cash consideration of approximately $ 54.2 million"
}

WRONG Response (computing answer instead of extracting):
{
  "value": 11.07,  // WRONG - this is a calculated percentage, not a value from text
  "reasoning": "Percentage = (6.0 / 54.2) * 100 = 11.07%"  // WRONG - don't compute
}"""

        # Build user payload - emphasize value_context over question
        user_payload = {
            "extraction_task": f"Extract the numeric value for: {value_context}",
            "text": text[:4000],  # Limit text length for token efficiency
            "guidance": {
                "keywords": keywords,
                "year": year,
                "unit": unit,
                "value_context": value_context,
            },
            "note": "The question is provided for reference only. Extract the raw value described in value_context, do NOT compute an answer to the question.",
            "question": question  # Keep for context but emphasize it's reference only
        }
        
        llm_start = time.time()
        response = await self.llm_client.parse_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_payload)},
            ],
            response_format=TextExtractionResponse,
            model=self.model,
        )
        llm_latency = (time.time() - llm_start) * 1000
        
        # Log LLM call to tracker
        if self.tracker and hasattr(response, 'usage') and response.usage:
            self.tracker.log_llm_call(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency_ms=llm_latency,
                model=response.model if hasattr(response, 'model') else self.model,
                purpose="text_extraction"
            )
        
        return response.choices[0].message.parsed
    
    def _verify_extraction(
        self,
        claimed_value: float,
        evidence_text: str,
        full_text: str,
        unit: str,
    ) -> float:
        """
        Verify that the claimed value exists in the evidence text to prevent hallucination.
        
        Verification steps:
        1. Check that evidence_text exists in full_text (verbatim)
        2. Generate all possible text formats of the claimed value
        3. Check if any format appears in the evidence_text
        
        Args:
            claimed_value: Value claimed by LLM
            evidence_text: Text snippet provided as evidence
            full_text: Full source text
            unit: Expected unit
            
        Returns:
            Verified value (same as claimed_value if verification passes)
            
        Raises:
            ValueError: If verification fails
        """
        # Step 1: Verify evidence_text exists in full_text
        if evidence_text.lower() not in full_text.lower():
            # Try with normalized whitespace
            normalized_full = re.sub(r'\s+', ' ', full_text.lower())
            normalized_evidence = re.sub(r'\s+', ' ', evidence_text.lower())
            if normalized_evidence not in normalized_full:
                raise ValueError(
                    f"Evidence text not found in source. This indicates LLM hallucination. "
                    f"Evidence: '{evidence_text[:100]}...'"
                )
        
        # Step 2: Generate all possible formats of the value
        value_formats = self._generate_value_formats(claimed_value, unit)
        
        # Step 3: Check if any format appears in evidence_text
        normalized_evidence = re.sub(r'\s+', ' ', evidence_text.lower())
        
        for fmt in value_formats:
            if fmt.lower() in normalized_evidence:
                logger.info(f"Verification passed: Found '{fmt}' in evidence")
                return claimed_value
        
        # Step 4: Check if the numeric value appears in evidence (extract all numbers and check)
        # The claimed_value is at "display scale" (e.g., 6.0 for "6.0 million")
        # The text also shows display scale (e.g., "6.0 million" in text)
        # So we compare pattern_value (from text) directly with claimed_value (from LLM)
        numeric_patterns = re.findall(r'\d+(?:\.\d+)?', normalized_evidence)
        for pattern_value_str in numeric_patterns:
            try:
                pattern_value = float(pattern_value_str)
                
                # Direct comparison at display scale (both should be at same scale)
                # Example: Text has "6.0 million", LLM extracts 6.0, pattern finds 6.0
                # All three are at display scale, so direct comparison works
                if abs(pattern_value - claimed_value) / max(abs(claimed_value), 1) < 0.01:
                    logger.info(f"Found numeric value in evidence: {pattern_value} ≈ {claimed_value}")
                    return claimed_value
                
                # Also check with rounding tolerance (handle cases like 6.0 vs 6)
                if abs(round(pattern_value, 1) - round(claimed_value, 1)) < 0.01:
                    logger.info(f"Found numeric value in evidence (rounded): {pattern_value} ≈ {claimed_value}")
                    return claimed_value
                    
            except (ValueError, ZeroDivisionError):
                continue
        
        # If no match found, log warning and reject
        logger.warning(
            f"Could not verify value {claimed_value} in evidence text. "
            f"Tried formats: {value_formats[:5]}... "
            f"Evidence: '{evidence_text[:100]}...'"
        )
        
        raise ValueError(
            f"Verification failed: Value {claimed_value} not found in evidence text. "
            f"Evidence: '{evidence_text}'"
        )
    
    def _generate_value_formats(self, value: float, unit: str) -> list[str]:
        """
        Generate all possible text formats of a numeric value for verification.
        
        Examples for value=6.0, unit="million":
        - "6.0"
        - "6"
        - "$6.0"
        - "$ 6.0"
        - "$6"
        - "$ 6"
        - "6.0 million"
        - "$6.0 million"
        - "$ 6.0 million"
        - etc.
        
        Args:
            value: Numeric value
            unit: Unit (million, billion, thousand, or none)
            
        Returns:
            List of possible text formats
        """
        formats = []
        
        # Base number formats
        value_float = f"{value}"
        value_int = f"{int(value)}" if value == int(value) else value_float
        
        # Add various formats
        base_formats = [
            value_float,
            value_int,
        ]
        
        # Add with dollar signs
        for base in base_formats:
            formats.extend([
                base,
                f"${base}",
                f"$ {base}",
                f"({base})",  # Negative format
                f"(${base})",
                f"($ {base})",
            ])
        
        # Add with units if not "none"
        if unit != "none" and unit:
            for base in base_formats:
                unit_singular = unit.rstrip('s')  # "millions" -> "million"
                formats.extend([
                    f"{base} {unit}",
                    f"{base} {unit_singular}",
                    f"${base} {unit}",
                    f"$ {base} {unit}",
                    f"${base} {unit_singular}",
                    f"$ {base} {unit_singular}",
                    f"{base}{unit}",  # No space
                    f"${base}{unit}",
                ])
        
        # Add with commas for large numbers
        if abs(value) >= 1000:
            value_with_commas = f"{value:,.1f}".rstrip('0').rstrip('.')
            formats.append(value_with_commas)
            formats.append(f"${value_with_commas}")
            formats.append(f"$ {value_with_commas}")
        
        return formats
    
    async def _extract_constant(
        self,
        params: TextConstantParams,
        pre_text: str,
        post_text: str
    ) -> float | dict[str, Any]:
        """
        Extract constant value using regex pattern matching.
        
        IMPORTANT: Searches both pre_text and post_text with equal priority.
        
        Args:
            params: Constant extraction parameters
            pre_text: Text before table
            post_text: Text after table
            
        Returns:
            Extracted value (float) or TextExtractionResult dict
        """
        # Pattern mapping for common constants
        constant_patterns = {
            "initial_investment": [
                r'initial\s+investment\s+of\s+\$\s*(?P<value>[\d,]+(?:\.\d+)?)',
                r'assumes\s+\$\s*(?P<value>[\d,]+(?:\.\d+)?)\s+(?:was\s+)?invested',
                r'assumes\s+an?\s+initial\s+investment\s+of\s+\$\s*(?P<value>[\d,]+(?:\.\d+)?)',
            ],
            "base_value": [
                r'base\s+(?:year\s+)?value\s+of\s+\$\s*(?P<value>[\d,]+(?:\.\d+)?)',
                r'normalized\s+to\s+\$?\s*(?P<value>[\d,]+(?:\.\d+)?)',
            ],
            "total_shares": [
                r'total\s+of\s+(?P<value>[\d,]+(?:\.\d+)?)\s+(?:million\s+)?shares',
            ],
        }
        
        patterns = constant_patterns.get(params.pattern, [])
        if not patterns:
            raise ValueError(f"Unknown constant pattern: {params.pattern}")
        
        # Search both texts
        texts = [
            ("pre_text", pre_text),
            ("post_text", post_text),
        ]
        
        for source, text in texts:
            if not text:
                continue
                
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value_str = match.group('value')
                    value = self._parse_numeric_value(value_str)
                    
                    # Get context
                    start_context = max(0, match.start() - 50)
                    end_context = min(len(text), match.end() + 50)
                    matched_text = text[start_context:end_context].strip()
                    
                    logger.info(
                        f"Extracted constant {value} for pattern '{params.pattern}' from {source}"
                    )
                    
                    return TextExtractionResult(
                        value=value,
                        matched_text=matched_text,
                        source=source,  # type: ignore
                        keywords=[params.pattern],
                        confidence=1.0
                    ).model_dump()
        
        raise ValueError(
            f"Could not find constant for pattern '{params.pattern}' "
            f"in pre_text or post_text"
        )
    
    def _parse_numeric_value(self, value_str: str) -> float:
        """Parse numeric string to float, handling commas."""
        cleaned = value_str.replace(',', '').replace(' ', '')
        return float(cleaned)