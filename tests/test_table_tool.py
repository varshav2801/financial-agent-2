"""Unit tests for WorkflowTableTool"""
import pytest
from src.tools.workflow_table_tool import WorkflowTableTool
from src.models.workflow_schema import ExtractTableParams
from src.models.exceptions import TableExtractionError


class TestWorkflowTableTool:
    """Test WorkflowTableTool extraction logic"""
    
    def setup_method(self):
        """Setup table tool instance"""
        self.tool = WorkflowTableTool()
    
    @pytest.mark.asyncio
    async def test_exact_match_extraction(self, sample_table):
        """Exact match extraction works"""
        params = ExtractTableParams(
            row_query="Net Income",
            col_query="2017"
        )
        value = await self.tool.extract_value(params, sample_table)
        assert value == 1245.0
    
    @pytest.mark.asyncio
    async def test_fuzzy_match_row(self, sample_table):
        """Fuzzy matching on row names"""
        params = ExtractTableParams(
            row_query="net income",  # Lowercase
            col_query="2017"
        )
        value = await self.tool.extract_value(params, sample_table)
        assert value == 1245.0
    
    @pytest.mark.asyncio
    async def test_fuzzy_match_column(self, sample_table):
        """Fuzzy matching on column names"""
        params = ExtractTableParams(
            row_query="Revenue",
            col_query="2016"
        )
        value = await self.tool.extract_value(params, sample_table)
        assert value == 4800.0
    
    @pytest.mark.asyncio
    async def test_similar_name_matching(self, sample_table):
        """Matches similar names"""
        params = ExtractTableParams(
            row_query="operating expense",  # Close to "Operating Expenses"
            col_query="2017"
        )
        value = await self.tool.extract_value(params, sample_table)
        assert value == 3500.0
    
    @pytest.mark.asyncio
    async def test_empty_table_fails(self):
        """Empty table raises error"""
        params = ExtractTableParams(
            row_query="Revenue",
            col_query="2017"
        )
        with pytest.raises(TableExtractionError, match="Empty table"):
            await self.tool.extract_value(params, {})
    
    @pytest.mark.asyncio
    async def test_missing_column_fails(self, sample_table):
        """Missing column raises error"""
        params = ExtractTableParams(
            row_query="Revenue",
            col_query="2020"  # Doesn't exist
        )
        with pytest.raises(TableExtractionError, match="No confident column match found"):
            await self.tool.extract_value(params, sample_table)
    
    @pytest.mark.asyncio
    async def test_missing_row_fails(self, sample_table):
        """Missing row raises error"""
        params = ExtractTableParams(
            row_query="Nonexistent Metric",
            col_query="2017"
        )
        with pytest.raises(TableExtractionError, match="No confident row match found"):
            await self.tool.extract_value(params, sample_table)
    
    @pytest.mark.asyncio
    async def test_year_format_normalization(self):
        """Handles different year formats"""
        table = {
            "December 31, 2017": {"Revenue": 1000.0},
            "December 31, 2016": {"Revenue": 900.0}
        }
        params = ExtractTableParams(
            row_query="Revenue",
            col_query="2017"  # Simple year
        )
        value = await self.tool.extract_value(params, table)
        assert value == 1000.0
    
    @pytest.mark.asyncio
    async def test_similarity_threshold(self, sample_table):
        """Respects similarity threshold"""
        # This should fail because "XYZ" is too different from any row
        params = ExtractTableParams(
            row_query="XYZ",
            col_query="2017"
        )
        with pytest.raises(TableExtractionError):
            await self.tool.extract_value(params, sample_table)
    
    def test_fuzzy_match_internal(self):
        """Test internal fuzzy matching logic"""
        candidates = ["Net Income", "Revenue", "Operating Expenses"]
        query = "net income"
        
        match = self.tool._fuzzy_match(query, candidates)
        assert match == "Net Income"
    
    def test_fuzzy_match_no_match(self):
        """Test fuzzy matching with no good match"""
        candidates = ["Net Income", "Revenue"]
        query = "XYZ Completely Different"
        
        with pytest.raises(TableExtractionError):
            self.tool._fuzzy_match(query, candidates)
