"""Integration tests for WorkflowExecutor"""
import pytest
from src.agent.workflow_executor import WorkflowExecutor
from src.models.workflow_schema import (
    WorkflowPlan,
    WorkflowStep,
    ExtractTableParams,
    Operand
)


class TestWorkflowExecutor:
    """Test WorkflowExecutor execution logic"""
    
    def setup_method(self):
        """Setup executor instance"""
        self.executor = WorkflowExecutor()
    
    @pytest.mark.asyncio
    async def test_simple_extraction(self, simple_extraction_plan, sample_document):
        """Execute simple extraction plan"""
        result = await self.executor.execute(
            plan=simple_extraction_plan,
            document=sample_document,
            previous_answers={}
        )
        
        assert result.success
        assert result.final_value == 1245.0
        assert len(result.step_results) == 1
        assert 1 in result.step_results  # step_results is a dict[int, float]
    
    @pytest.mark.asyncio
    async def test_computation_plan(self, computation_plan, sample_document):
        """Execute plan with computation"""
        result = await self.executor.execute(
            plan=computation_plan,
            document=sample_document,
            previous_answers={}
        )
        
        assert result.success
        # Percentage change from 1180 to 1245
        expected = ((1245 - 1180) / 1180) * 100
        assert abs(result.final_value - expected) < 0.01
        assert len(result.step_results) == 3
    
    @pytest.mark.asyncio
    async def test_addition_operation(self, sample_document):
        """Test addition operation"""
        plan = WorkflowPlan(
            thought_process="Add two revenue values",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="extract_value",
                    source="table",
                    table_params=ExtractTableParams(
                        row_query="Revenue",
                        col_query="2017"
                    )
                ),
                WorkflowStep(
                    step_id=2,
                    tool="extract_value",
                    source="table",
                    table_params=ExtractTableParams(
                        row_query="Revenue",
                        col_query="2016"
                    )
                ),
                WorkflowStep(
                    step_id=3,
                    tool="compute",
                    operation="add",
                    operands=[
                        Operand(type="reference", step_ref=1),
                        Operand(type="reference", step_ref=2)
                    ]
                )
            ]
        )
        
        result = await self.executor.execute(plan, sample_document, {})
        assert result.success
        assert result.final_value == 5000.0 + 4800.0
    
    @pytest.mark.asyncio
    async def test_subtraction_operation(self, sample_document):
        """Test subtraction operation"""
        plan = WorkflowPlan(
            thought_process="Calculate difference in revenue",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="extract_value",
                    source="table",
                    table_params=ExtractTableParams(
                        row_query="Revenue",
                        col_query="2017"
                    )
                ),
                WorkflowStep(
                    step_id=2,
                    tool="extract_value",
                    source="table",
                    table_params=ExtractTableParams(
                        row_query="Revenue",
                        col_query="2016"
                    )
                ),
                WorkflowStep(
                    step_id=3,
                    tool="compute",
                    operation="subtract",
                    operands=[
                        Operand(type="reference", step_ref=1),
                        Operand(type="reference", step_ref=2)
                    ]
                )
            ]
        )
        
        result = await self.executor.execute(plan, sample_document, {})
        assert result.success
        assert result.final_value == 5000.0 - 4800.0
    
    @pytest.mark.asyncio
    async def test_multiplication_operation(self, sample_document):
        """Test multiplication operation"""
        plan = WorkflowPlan(
            thought_process="Multiply revenue by factor",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="extract_value",
                    source="table",
                    table_params=ExtractTableParams(
                        row_query="Revenue",
                        col_query="2017"
                    )
                ),
                WorkflowStep(
                    step_id=2,
                    tool="compute",
                    operation="multiply",
                    operands=[
                        Operand(type="reference", step_ref=1),
                        Operand(type="literal", value=2.0)
                    ]
                )
            ]
        )
        
        result = await self.executor.execute(plan, sample_document, {})
        assert result.success
        assert result.final_value == 5000.0 * 2.0
    
    @pytest.mark.asyncio
    async def test_division_operation(self, sample_document):
        """Test division operation"""
        plan = WorkflowPlan(
            thought_process="Divide revenue by constant",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="extract_value",
                    source="table",
                    table_params=ExtractTableParams(
                        row_query="Revenue",
                        col_query="2017"
                    )
                ),
                WorkflowStep(
                    step_id=2,
                    tool="compute",
                    operation="divide",
                    operands=[
                        Operand(type="reference", step_ref=1),
                        Operand(type="literal", value=100.0)
                    ]
                )
            ]
        )
        
        result = await self.executor.execute(plan, sample_document, {})
        assert result.success
        assert result.final_value == 5000.0 / 100.0
    
    @pytest.mark.asyncio
    async def test_percentage_operation(self, sample_document):
        """Test percentage operation (part/whole * 100)"""
        plan = WorkflowPlan(
            thought_process="Calculate percentage",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="compute",
                    operation="percentage",
                    operands=[
                        Operand(type="literal", value=50.0),
                        Operand(type="literal", value=100.0)
                    ]
                )
            ]
        )
        
        result = await self.executor.execute(plan, sample_document, {})
        assert result.success
        assert result.final_value == 50.0  # (50/100)*100 = 50%
    
    @pytest.mark.asyncio
    async def test_literal_operands(self, sample_document):
        """Test computation with only literal operands"""
        plan = WorkflowPlan(
            thought_process="Add two constants",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="compute",
                    operation="add",
                    operands=[
                        Operand(type="literal", value=100.0),
                        Operand(type="literal", value=50.0)
                    ]
                )
            ]
        )
        
        result = await self.executor.execute(plan, sample_document, {})
        assert result.success
        assert result.final_value == 150.0
    
    @pytest.mark.asyncio
    async def test_conversation_history_reference(self, sample_document):
        """Test referencing conversation history with negative indices"""
        previous_answers = {
            "prev_0": {"value": 100.0, "entity": "test", "operation": "extract"}
        }
        
        plan = WorkflowPlan(
            thought_process="Add to previous result",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="compute",
                    operation="add",
                    operands=[
                        Operand(type="reference", step_ref=-1),  # prev_0
                        Operand(type="literal", value=50.0)
                    ]
                )
            ]
        )
        
        result = await self.executor.execute(plan, sample_document, previous_answers)
        assert result.success
        assert result.final_value == 150.0
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, sample_document):
        """Test that memory persists across steps"""
        plan = WorkflowPlan(
            thought_process="Multi-step with memory",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="extract_value",
                    source="table",
                    table_params=ExtractTableParams(
                        row_query="Net Income",
                        col_query="2017"
                    )
                ),
                WorkflowStep(
                    step_id=2,
                    tool="compute",
                    operation="multiply",
                    operands=[
                        Operand(type="reference", step_ref=1),
                        Operand(type="literal", value=2.0)
                    ]
                ),
                WorkflowStep(
                    step_id=3,
                    tool="compute",
                    operation="add",
                    operands=[
                        Operand(type="reference", step_ref=1),
                        Operand(type="reference", step_ref=2)
                    ]
                )
            ]
        )
        
        result = await self.executor.execute(plan, sample_document, {})
        assert result.success
        # Step 1: 1245, Step 2: 1245*2 = 2490, Step 3: 1245 + 2490 = 3735
        assert result.final_value == 3735.0
    
    @pytest.mark.asyncio
    async def test_execution_metadata(self, simple_extraction_plan, sample_document):
        """Test execution metadata is captured"""
        result = await self.executor.execute(
            simple_extraction_plan,
            sample_document,
            {}
        )
        
        assert len(result.step_results) == 1
        assert result.execution_time_ms > 0
        assert result.success
