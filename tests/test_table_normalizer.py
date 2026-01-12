"""Unit tests for table normalization utilities"""
import pytest
from src.utils.table_normalizer import (
    normalize_table,
    _is_year_key,
    _detect_table_orientation
)


class TestYearDetection:
    """Test year key detection"""
    
    def test_simple_year(self):
        """Simple 4-digit year"""
        assert _is_year_key("2017")
        assert _is_year_key("2008")
        assert _is_year_key("1999")
    
    def test_year_in_string(self):
        """Year embedded in string"""
        assert _is_year_key("Year ended 2017")
        assert _is_year_key("December 31, 2016")
        assert _is_year_key("FY 2015")
    
    def test_non_year(self):
        """Non-year strings"""
        assert not _is_year_key("Revenue")
        assert not _is_year_key("Net Income")
        assert not _is_year_key("Q1")
    
    def test_edge_cases(self):
        """Edge cases"""
        assert not _is_year_key("1850")  # Too old
        assert not _is_year_key("2200")  # Too far in future
        assert not _is_year_key("99")    # Too short


class TestTableOrientation:
    """Test table orientation detection"""
    
    def test_year_oriented_table(self):
        """Table with years as top-level keys (standard format)"""
        table = {
            "2017": {"Revenue": 1000, "Net Income": 100},
            "2016": {"Revenue": 900, "Net Income": 90}
        }
        is_inverted, years, metrics = _detect_table_orientation(table)
        assert not is_inverted  # Standard format is NOT inverted
        assert "2017" in years
        assert "2016" in years
        assert "Revenue" in metrics
        assert "Net Income" in metrics
    
    def test_metric_oriented_table(self):
        """Table with metrics as top-level keys (inverted format)"""
        table = {
            "Revenue": {"2017": 1000, "2016": 900},
            "Net Income": {"2017": 100, "2016": 90}
        }
        is_inverted, years, metrics = _detect_table_orientation(table)
        assert is_inverted  # Metric-oriented IS inverted
        assert "2017" in years
        assert "2016" in years
        assert "Revenue" in metrics
        assert "Net Income" in metrics


class TestTableNormalization:
    """Test complete table normalization"""
    
    def test_already_normalized(self):
        """Already normalized table (year -> metric)"""
        table = {
            "2017": {"Revenue": 1000.0, "Net Income": 100.0},
            "2016": {"Revenue": 900.0, "Net Income": 90.0}
        }
        normalized = normalize_table(table)
        assert normalized == table
    
    def test_transpose_metric_oriented(self):
        """Transpose metric-oriented table"""
        table = {
            "Revenue": {"2017": 1000, "2016": 900},
            "Net Income": {"2017": 100, "2016": 90}
        }
        normalized = normalize_table(table)
        
        # Should be transposed to year -> metric
        assert "2017" in normalized
        assert "2016" in normalized
        assert normalized["2017"]["Revenue"] == 1000
        assert normalized["2016"]["Net Income"] == 90
    
    def test_handles_numeric_values(self):
        """Handles various numeric formats"""
        table = {
            "2017": {
                "Revenue": 1000,
                "Net Income": 100.5,
                "Loss": -50
            }
        }
        normalized = normalize_table(table)
        assert normalized["2017"]["Revenue"] == 1000.0
        assert normalized["2017"]["Net Income"] == 100.5
        assert normalized["2017"]["Loss"] == -50.0
    
    def test_handles_string_numbers(self):
        """Preserves string representations as-is (no type conversion)"""
        table = {
            "2017": {
                "Revenue": "1000",
                "Net Income": "100.5"
            }
        }
        normalized = normalize_table(table)
        # normalize_table doesn't convert types, just transposes if needed
        assert normalized["2017"]["Revenue"] == "1000"
        assert normalized["2017"]["Net Income"] == "100.5"
    
    def test_preserves_non_numeric(self):
        """Preserves all values including non-numeric (no filtering)"""
        table = {
            "2017": {
                "Revenue": 1000,
                "Description": "Strong year",  # Non-numeric
                "Net Income": 100
            }
        }
        normalized = normalize_table(table)
        # normalize_table doesn't filter values, just transposes if needed
        assert "Description" in normalized["2017"]
        assert normalized["2017"]["Description"] == "Strong year"
        assert "Revenue" in normalized["2017"]
    
    def test_empty_table(self):
        """Handles empty table"""
        normalized = normalize_table({})
        assert normalized == {}
    
    def test_complex_year_formats(self):
        """Handles complex year formats in keys"""
        table = {
            "Year Ended December 31, 2017": {"Revenue": 1000},
            "Year Ended December 31, 2016": {"Revenue": 900}
        }
        normalized = normalize_table(table)
        
        # Should extract years from complex strings
        assert any("2017" in key for key in normalized.keys())
        assert any("2016" in key for key in normalized.keys())
