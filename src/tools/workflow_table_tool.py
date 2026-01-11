"""Workflow table tool with fuzzy matching using RapidFuzz"""
import re
from typing import Any
from rapidfuzz import process, fuzz
from src.models.workflow_schema import ExtractTableParams
from src.models.exceptions import TableExtractionError
from src.logger import get_logger

logger = get_logger(__name__)


class WorkflowTableTool:
    """
    Table tool for workflow-based extraction.
    
    Uses fuzzy matching with RapidFuzz for robust row/column matching.
    Supports normalized table format: table[year][metric] -> value
    """
    
    def __init__(self) -> None:
        self.similarity_threshold = 85  # RapidFuzz uses 0-100 scale
    
    async def extract_value(
        self,
        params: ExtractTableParams,
        table: dict[str, dict[str, float | str]]
    ) -> float:
        """
        Extract single numeric value from table using fuzzy matching.
        
        Args:
            params: ExtractTableParams with row_query, col_query
            table: Normalized table structure (table[year][metric])
        
        Returns:
            Extracted numeric value
            
        Raises:
            TableExtractionError: If matching fails or value not found
        """
        if not table:
            raise TableExtractionError(
                "Empty table provided",
                details={"params": params.model_dump()}
            )
        
        # Match column (year) first
        available_columns = list(table.keys())
        col_match = self._fuzzy_match(
            query=params.col_query,
            choices=available_columns,
            context="column"
        )
        
        logger.debug(f"Column match: '{params.col_query}' -> '{col_match}'")
        
        # Match row (metric) within the matched column
        available_rows = list(table[col_match].keys())
        row_match = self._fuzzy_match(
            query=params.row_query,
            choices=available_rows,
            context="row"
        )
        
        logger.debug(f"Row match: '{params.row_query}' -> '{row_match}'")
        
        # Extract value
        raw_value = table[col_match][row_match]
        logger.info(f"Extracted: table[{col_match}][{row_match}] = {raw_value}")
        
        # Convert to numeric
        numeric_value = self._to_numeric(raw_value)
        
        # Apply unit normalization if specified
        if params.unit_normalization:
            numeric_value = self._normalize_unit(
                value=numeric_value,
                unit=params.unit_normalization
            )
        
        return numeric_value
    
    def _fuzzy_match(
        self,
        query: str,
        choices: list[str],
        context: str = "key"
    ) -> str:
        """
        Fuzzy match query against choices using RapidFuzz.
        
        Strategy:
        1. Try exact match (case-insensitive)
        2. Try year normalization for numeric years
        3. Use RapidFuzz WRatio for fuzzy matching
        
        Args:
            query: Search query
            choices: Available options
            context: Context for error messages (row/column)
            
        Returns:
            Best matching choice
            
        Raises:
            TableExtractionError: If no confident match found
        """
        if not choices:
            raise TableExtractionError(
                f"No available {context}s to match against",
                details={"query": query}
            )
        
        query_lower = query.lower()
        
        # Step 1: Exact match (case-insensitive)
        for choice in choices:
            if choice.lower() == query_lower:
                logger.debug(f"Exact match: '{query}' -> '{choice}'")
                return choice
        
        # Step 2: Year normalization
        query_year = self._extract_year(query)
        if query_year:
            # Try exact year match first
            if query_year in choices:
                logger.debug(f"Year exact match: '{query}' -> '{query_year}'")
                return query_year
            
            # Match against normalized years from choices
            for choice in choices:
                choice_year = self._extract_year(choice)
                if choice_year == query_year:
                    logger.debug(f"Year normalized match: '{query}' (year={query_year}) -> '{choice}'")
                    return choice
        
        # Step 3: Fuzzy matching with RapidFuzz
        result = process.extractOne(
            query=query,
            choices=choices,
            scorer=fuzz.WRatio,
            score_cutoff=self.similarity_threshold
        )
        
        if result is None:
            raise TableExtractionError(
                f"No confident {context} match found for '{query}'",
                details={
                    "query": query,
                    "available": choices[:10],
                    "threshold": self.similarity_threshold,
                    "note": "Try using more specific keywords or exact names"
                }
            )
        
        matched_choice, score, _ = result
        logger.debug(f"Fuzzy match: '{query}' -> '{matched_choice}' (score: {score:.1f})")
        
        return matched_choice
    
    def _extract_year(self, text: str) -> str | None:
        """
        Extract 4-digit year from text.
        
        Handles:
        - "2014" -> "2014"
        - "FY2014" -> "2014"
        - "Year ended June 30, 2014" -> "2014"
        
        Returns:
            Extracted year as string, or None if not found
        """
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        if years:
            return years[0]
        return None
    
    def _to_numeric(self, value: Any) -> float:
        """
        Convert value to numeric float.
        
        Handles:
        - Already numeric: 145.2 -> 145.2
        - String with commas: "1,234.56" -> 1234.56
        - String with units: "$145.2M" -> 145.2
        - Percentage: "12.5%" -> 12.5
        
        Args:
            value: Raw value from table
            
        Returns:
            Numeric float value
            
        Raises:
            TableExtractionError: If value cannot be converted
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common formatting
            cleaned = value.strip()
            # Remove currency symbols and units
            cleaned = re.sub(r'[$£€¥]', '', cleaned)
            # Remove commas
            cleaned = cleaned.replace(',', '')
            # Remove percentage sign (but keep the number)
            cleaned = cleaned.replace('%', '')
            # Remove parentheses (negative numbers)
            if '(' in cleaned and ')' in cleaned:
                cleaned = cleaned.replace('(', '-').replace(')', '')
            
            try:
                return float(cleaned)
            except ValueError:
                raise TableExtractionError(
                    f"Cannot convert value to numeric: '{value}'",
                    details={"raw_value": value, "cleaned_value": cleaned}
                )
        
        raise TableExtractionError(
            f"Unsupported value type: {type(value)}",
            details={"value": value}
        )
    
    def _normalize_unit(self, value: float, unit: str) -> float:
        """
        Normalize value based on expected unit.
        
        Args:
            value: Numeric value
            unit: Expected unit (informational only)
            
        Returns:
            Value unchanged
        """
        # Return value as-is - no normalization needed
        # The unit parameter is just informational/documentation
        if unit and unit.lower() not in ["none", "null", ""]:
            logger.debug(f"Unit '{unit}' noted, returning value as-is: {value}")
        return value
