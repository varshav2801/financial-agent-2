"""Utility to normalize table structure to standard format"""
import re
from typing import Any
from src.logger import get_logger

logger = get_logger(__name__)

YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')


def _is_year_key(key: str) -> bool:
    """Check if a key looks like a year (e.g., '2008', '2009', 'Year ended June 30, 2009')"""
    key_str = str(key)
    # Check for 4-digit year pattern
    if YEAR_PATTERN.search(key_str):
        return True
    # Short numeric strings (3-4 digits) that are likely years
    key_stripped = key_str.strip()
    if key_stripped.isdigit() and len(key_stripped) in [3, 4]:
        year_val = int(key_stripped)
        if 1900 <= year_val <= 2100:
            return True
    return False


def _detect_table_orientation(table: dict[str, dict[str, Any]]) -> tuple[bool, list[str], list[str]]:
    """
    Detect table orientation by checking if top-level keys are years or metrics.
    
    Returns:
        (is_inverted, years, metrics)
        - is_inverted: True if table[metric][year], False if table[year][metric]
        - years: List of year keys
        - metrics: List of metric keys
    """
    if not table:
        return False, [], []
    
    top_level_keys = list(table.keys())
    first_value = next(iter(table.values()))
    
    if not isinstance(first_value, dict):
        # Not a nested dict, can't determine
        return False, top_level_keys, []
    
    nested_keys = list(first_value.keys())
    
    # Count how many top-level keys look like years
    top_year_count = sum(1 for key in top_level_keys if _is_year_key(key))
    # Count how many nested keys look like years
    nested_year_count = sum(1 for key in nested_keys if _is_year_key(key))
    
    # If top-level keys are mostly years, table is in standard format
    # If nested keys are mostly years, table is inverted
    if top_year_count > nested_year_count:
        # Standard format: table[year][metric]
        years = top_level_keys
        metrics = nested_keys
        logger.debug(f"Detected standard table format: {top_year_count} year keys at top level")
        return False, years, metrics
    elif nested_year_count > top_year_count:
        # Inverted format: table[metric][year]
        years = nested_keys
        metrics = top_level_keys
        logger.debug(f"Detected inverted table format: {nested_year_count} year keys at nested level")
        return True, years, metrics
    else:
        # Ambiguous - default to standard format (assume top-level are years)
        logger.warning(f"Table orientation ambiguous (top_year_count={top_year_count}, nested_year_count={nested_year_count}), defaulting to standard format")
        return False, top_level_keys, nested_keys


def normalize_table(table: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Normalize table to standard format: table[year][metric]
    
    Args:
        table: Table dict that may be in either format
        
    Returns:
        Normalized table in format table[year][metric]
    """
    if not table:
        return {}
    
    is_inverted, years, metrics = _detect_table_orientation(table)
    
    if not is_inverted:
        # Already in standard format
        logger.debug("Table is already in standard format (table[year][metric])")
        return table
    
    # Table is inverted - transpose it
    logger.info(f"Normalizing inverted table structure from table[metric][year] to table[year][metric] (years: {len(years)}, metrics: {len(metrics)})")
    normalized: dict[str, dict[str, Any]] = {}
    
    # Transpose: table[metric][year] -> table[year][metric]
    for metric_key in metrics:
        if metric_key not in table:
            continue
        metric_data = table[metric_key]
        if not isinstance(metric_data, dict):
            continue
            
        for year_key in years:
            if year_key not in metric_data:
                continue
            if year_key not in normalized:
                normalized[year_key] = {}
            normalized[year_key][metric_key] = metric_data[year_key]
    
    logger.info(f"Successfully normalized table: {len(years)} years, {len(metrics)} metrics")
    return normalized

