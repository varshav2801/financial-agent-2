"""Unit tests for WorkflowValidator"""
import pytest
from src.agent.workflow_validator import WorkflowValidator
from src.models.workflow_schema import (
    WorkflowPlan,
    WorkflowStep,
    ExtractTableParams,
    Operand
)


class TestWorkflowValidator:
    """Test WorkflowValidator logic"""
    
    def setup_method(self):
        """Setup validator instance"""
        self.validator = WorkflowValidator()
    
    def test_valid_simple_plan(self, simple_extraction_plan):
        """Valid simple extraction plan passes validation"""
        result = self.validator.validate(simple_extraction_plan)
        assert result.is_valid
        assert len(result.critiques) == 0
    
    def test_valid_computation_plan(self, computation_plan):
        """Valid computation plan passes validation"""
        result = self.validator.validate(computation_plan)
        assert result.is_valid
        assert len(result.critiques) == 0
    
    def test_empty_plan_fails(self):
        """Empty plan fails at Pydantic model validation"""
        from pydantic import ValidationError
        # Pydantic enforces min_length=1 at construction
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            WorkflowPlan(
                thought_process="Empty plan",
                steps=[]
            )
    
    def test_missing_thought_process_fails(self):
        """Missing thought_process fails validation"""
        plan = WorkflowPlan(
            thought_process="",
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="extract_value",
                    source="table",
                    table_params=ExtractTableParams(
                        row_query="Revenue",
                        col_query="2017"
                    )
                )
            ]
        )
        result = self.validator.validate(plan)
        assert not result.is_valid
        assert any(c.issue_type == "MissingThoughtProcess" for c in result.critiques)
    
    def test_forward_reference_fails(self, invalid_forward_reference_plan):
        """Forward reference fails validation"""
        result = self.validator.validate(invalid_forward_reference_plan)
        assert not result.is_valid
        assert any(c.issue_type == "ForwardReference" for c in result.critiques)
    
    def test_non_sequential_step_ids_fails(self):
        """Non-sequential step IDs fail at Pydantic validation"""
        from pydantic import ValidationError
        # Pydantic validator checks sequential IDs at construction
        with pytest.raises(ValidationError, match="must be sequential"):
            WorkflowPlan(
                thought_process="Plan with skipped step IDs",
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
                        step_id=3,  # Skipped 2
                        tool="extract_value",
                        source="table",
                        table_params=ExtractTableParams(
                            row_query="Net Income",
                            col_query="2017"
                        )
                    )
                ]
            )
    
    def test_invalid_operand_count_unary(self):
        """Unary operation with wrong operand count fails"""
        plan = WorkflowPlan(
            thought_process="Invalid unary operation",
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
                    operation="percentage",
                    operands=[
                        Operand(type="reference", step_ref=1),
                        Operand(type="literal", value=100.0)  # Should only have 1
                    ]
                )
            ]
        )
        result = self.validator.validate(plan)
        assert not result.is_valid
        assert any(c.issue_type == "InvalidOperandCount" for c in result.critiques)
    
    def test_invalid_operand_count_binary(self):
        """Binary operation with wrong operand count fails"""
        plan = WorkflowPlan(
            thought_process="Invalid binary operation",
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
                    operation="add",
                    operands=[
                        Operand(type="reference", step_ref=1)  # Should have 2
                    ]
                )
            ]
        )
        result = self.validator.validate(plan)
        assert not result.is_valid
        assert any(c.issue_type == "InvalidOperandCount" for c in result.critiques)
    
    def test_invalid_reference_step(self):
        """Reference to non-existent step fails"""
        plan = WorkflowPlan(
            thought_process="Invalid reference",
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
                    operation="add",
                    operands=[
                        Operand(type="reference", step_ref=1),
                        Operand(type="reference", step_ref=99)  # Doesn't exist
                    ]
                )
            ]
        )
        result = self.validator.validate(plan)
        assert not result.is_valid
        assert any(c.issue_type == "InvalidReference" for c in result.critiques)
    
    def test_negative_reference_allowed(self):
        """Negative references (conversation history) are allowed"""
        plan = WorkflowPlan(
            thought_process="Using conversation history",
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
        result = self.validator.validate(plan)
        # This should pass validator (executor will handle history availability)
        assert result.is_valid
    
    def test_missing_source_for_extraction(self):
        """Extraction without source fails at Pydantic validation"""
        from pydantic import ValidationError
        # Pydantic enforces required source at construction
        with pytest.raises(ValidationError, match="source is required"):
            WorkflowPlan(
                thought_process="Missing source",
                steps=[
                    WorkflowStep(
                        step_id=1,
                        tool="extract_value",
                        source=None,  # Missing
                        table_params=ExtractTableParams(
                            row_query="Revenue",
                            col_query="2017"
                        )
                    )
                ]
            )
    
    def test_critique_structure(self, invalid_forward_reference_plan):
        """Critiques have proper structure"""
        result = self.validator.validate(invalid_forward_reference_plan)
        assert not result.is_valid
        
        critique = result.critiques[0]
        assert critique.step_id is not None
        assert critique.issue_type != ""
        assert critique.reason != ""
        assert critique.fix_suggestion != ""
