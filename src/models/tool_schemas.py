"""Pydantic models for tool inputs and outputs (text tool only)"""
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class TextExtractParams(BaseModel):
    """Parameters for extracting numeric values from prose text"""
    model_config = ConfigDict(extra="forbid")
    
    text_source: Literal["pre_text", "post_text"] = Field(
        description="Which text to search: pre_text (before table) or post_text (after table)"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Optional keywords/hints (not required). Used only for scoring, not for hard matching."
    )
    unit: str = Field(
        default="million",
        description="Expected unit (million, billion, thousand, or 'none')"
    )
    year: Optional[str] = Field(
        default=None,
        description="Optional year filter to disambiguate (e.g., '2017')"
    )
    allow_negative: bool = Field(
        default=True,
        description="Whether to allow negative values"
    )
    normalize_units: bool = Field(
        default=False,
        description="Whether to normalize units (e.g., convert 'million' to raw number)"
    )
    max_candidates: int = Field(
        default=12,
        description="Maximum number of candidate facts to consider (higher = more robust but more expensive for selection)"
    )
    context_window: int = Field(
        default=300,
        description="Number of characters to include around each extracted mention as evidence/snippet"
    )


class FactCandidate(BaseModel):
    """A grounded numeric candidate extracted from prose."""
    id: str
    value: float  # Normalized value (e.g., 243000000 for "$243 million")
    raw_value: float  # Raw value without unit multiplier (e.g., 243)
    unit: str  # Unit found (e.g., "million", "billion", "none")
    span_start: int
    span_end: int
    snippet: str
    context_tags: list[str] = []  # Pattern tags for semantic understanding
    source: str = "unknown"  # Text source: "pre_text" or "post_text"


class FactSelectionRequest(BaseModel):
    """Constrained selection request for an LLM: pick ONE candidate id, do not compute."""
    question: str
    candidates: list[FactCandidate]
    instruction: str = "Select the single best matching candidate id. Do not compute. Output only the selected id."


class FactSelectionResult(BaseModel):
    """Constrained selection response."""
    selected_id: str


class TextConstantParams(BaseModel):
    """Parameters for extracting common constants from prose"""
    model_config = ConfigDict(extra="forbid")
    
    text_source: Literal["pre_text", "post_text"] = Field(
        description="Which text to search"
    )
    pattern: Literal["initial_investment", "base_value", "total_shares"] = Field(
        description="Type of constant to extract"
    )


class TextToolParams(BaseModel):
    """Generic parameters for text tool; used in structured plans."""
    model_config = ConfigDict(extra="forbid")
    
    # Common to all actions
    text_source: Literal["pre_text", "post_text"] = "post_text"
    
    # Used by action="extract_numeric"
    keywords: Optional[list[str]] = None
    unit: str = "million"
    year: Optional[str] = None
    value_context: Optional[str] = None  # NEW: What the value represents
    allow_negative: bool = True
    normalize_units: bool = False
    
    # Used by action="extract_constant"
    pattern: Optional[Literal["initial_investment", "base_value", "total_shares"]] = None


class TextExtractionResult(BaseModel):
    """Result from text extraction with evidence to prevent hallucination"""
    value: float = Field(description="Extracted numeric value")
    matched_text: str = Field(description="The actual text that was matched (evidence)")
    source: str = Field(description="Text source (pre_text or post_text)")
    keywords: list[str] = Field(description="Keywords used for extraction")
    confidence: float = Field(description="Confidence score (1.0 = single match, <1.0 = multiple matches)")


class TextSpan(BaseModel):
    """A text span (sentence or snippet) from pre_text or post_text"""
    span_id: str = Field(description="Unique identifier for this span (e.g., 'pre_s0', 'post_s5')")
    text: str = Field(description="The text content of this span")
    source: Literal["pre_text", "post_text"] = Field(description="Which source this span comes from")
    span_start: int = Field(description="Character position where this span starts in the original text")
    span_end: int = Field(description="Character position where this span ends in the original text")


class SpanExtractionResponse(BaseModel):
    """LLM response for span-based extraction (DEPRECATED - use TextExtractionResponse)"""
    model_config = ConfigDict(extra="forbid")
    
    value_identifier: str = Field(
        description="Description of what value this represents (e.g., 'drawn amount from credit facility in 2016')"
    )
    search_span_id: str = Field(
        description="The span_id that contains the answer (e.g., 'pre_s0', 'post_s5')"
    )
    raw_value_text: str = Field(
        description="The raw text string containing the numeric value (e.g., '$243 million', '$1.6 billion')"
    )


class TextExtractionResponse(BaseModel):
    """LLM response for direct text extraction with verification"""
    model_config = ConfigDict(extra="forbid")
    
    value: float = Field(
        description="The extracted numeric value in the expected unit (e.g., 6.0 for '$6.0 million')"
    )
    reasoning: str = Field(
        description="Brief explanation of how you identified this value in context"
    )
    evidence_text: str = Field(
        description="The exact text snippet from the source that contains this value (must be verbatim quote for verification)"
    )