"""Unit tests for Pydantic models and validation"""
import pytest
from pydantic import ValidationError
from src.models.workflow_schema import (
    WorkflowPlan,
    WorkflowStep,
    Operand,
    ExtractTableParams,
    ExtractTextParams,
    StepCritique,
    ValidationResult
)


class TestOperandModel:
    """Test Operand model validation"""
    
    def test_reference_operand_requires_step_ref(self):
        """Reference operand must have step_ref"""
        with pytest.raises(ValidationError, match="step_ref is required"):
            Operand(type="reference", step_ref=None)
    
    def test_literal_operand_requires_value(self):
        """Literal operand must have value"""
        with pytest.raises(ValidationError, match="value is required"):
            Operand(type="literal", value=None)
    
    def test_valid_reference_operand(self):
        """Valid reference operand creation"""
        op = Operand(type="reference", step_ref=1)
        assert op.type == "reference"
        assert op.step_ref == 1
        assert op.value is None
    
    def test_valid_literal_operand(self):
        """Valid literal operand creation"""
        op = Operand(type="literal", value=100.0)
        assert op.type == "literal"
        assert op.value == 100.0
        assert op.step_ref is None
    
    def test_negative_step_ref_for_history(self):
        """Negative step_ref for conversation history"""
        op = Operand(type="reference", step_ref=-1)
        assert op.step_ref == -1


class TestExtractTableParams:
    """Test ExtractTableParams model"""
    
    def test_valid_table_params(self):
        """Valid table extraction parameters"""
        params = ExtractTableParams(
            row_query="Net Income",
            col_query="2017"
        )
        assert params.row_query == "Net Income"
        assert params.col_query == "2017"
        assert params.table_id == "main"  # default
    
    def test_with_unit_normalization(self):
        """Table params with unit normalization"""
        params = ExtractTableParams(
            row_query="Revenue",
            col_query="2016",
            unit_normalization="million"
        )
        assert params.unit_normalization == "million"
    
    def test_missing_required_fields(self):
        """Missing required fields raises error"""
        with pytest.raises(ValidationError):
            ExtractTableParams(row_query="Net Income")


class TestExtractTextParams:
    """Test ExtractTextParams model"""
    
    def test_valid_text_params(self):
        """Valid text extraction parameters"""
        params = ExtractTextParams(
            context_window="pre_text",
            search_keywords=["debt", "refinancing", "charges"]
        )
        assert params.context_window == "pre_text"
        assert len(params.search_keywords) == 3
    
    def test_empty_keywords_fails(self):
        """Empty keywords list fails validation"""
        with pytest.raises(ValidationError):
            ExtractTextParams(
                context_window="post_text",
                search_keywords=[]
            )


class TestWorkflowStep:
    """Test WorkflowStep model"""
    
    def test_valid_extraction_step(self):
        """Valid extraction step"""
        step = WorkflowStep(
            step_id=1,
            tool="extract_value",
            source="table",
            table_params=ExtractTableParams(
                row_query="Revenue",
                col_query="2017"
            )
        )
        assert step.step_id == 1
        assert step.tool == "extract_value"
        assert step.source == "table"
        assert step.table_params is not None
    
    def test_valid_compute_step(self):
        """Valid computation step"""
        step = WorkflowStep(
            step_id=2,
            tool="compute",
            operation="add",
            operands=[
                Operand(type="reference", step_ref=1),
                Operand(type="literal", value=50.0)
            ]
        )
        assert step.tool == "compute"
        assert step.operation == "add"
        assert len(step.operands) == 2


class TestWorkflowPlan:
    """Test WorkflowPlan model"""
    
    def test_valid_plan(self, simple_extraction_plan):
        """Valid workflow plan"""
        assert simple_extraction_plan.thought_process != ""
        assert len(simple_extraction_plan.steps) == 1
    
    def test_empty_steps_allowed_by_model(self):
        """Model enforces at least one step via min_length"""
        with pytest.raises(ValidationError, match="at least 1 item"):
            WorkflowPlan(
                thought_process="Empty plan",
                steps=[]
            )
    
    def test_multiple_steps(self, computation_plan):
        """Plan with multiple steps"""
        assert len(computation_plan.steps) == 3
        assert computation_plan.steps[0].step_id == 1
        assert computation_plan.steps[-1].step_id == 3


class TestStepCritique:
    """Test StepCritique model"""
    
    def test_valid_critique(self):
        """Valid step critique"""
        critique = StepCritique(
            step_id=3,
            issue_type="ForwardReference",
            reason="Step 3 references Step 5 which doesn't exist yet.",
            fix_suggestion="Reorder steps or change reference to existing step."
        )
        assert critique.step_id == 3
        assert critique.issue_type == "ForwardReference"
    
    def test_critique_without_step_id(self):
        """Critique for plan-level issues (no step_id)"""
        critique = StepCritique(
            step_id=None,
            issue_type="EmptyPlan",
            reason="Plan has no steps",
            fix_suggestion="Add at least one step"
        )
        assert critique.step_id is None


class TestValidationResult:
    """Test ValidationResult model"""
    
    def test_valid_result(self):
        """Valid validation result"""
        result = ValidationResult(
            is_valid=False,
            confidence=0.8,
            critiques=[
                StepCritique(
                    step_id=2,
                    issue_type="InvalidOperandCount",
                    reason="Operation requires 2 operands",
                    fix_suggestion="Add missing operand"
                )
            ],
            issues=["Invalid operand count in step 2"],
            failed_steps=[2],
            repair_instructions="Fix operand count in step 2",
            is_hallucination_likely=False
        )
        assert not result.is_valid
        assert len(result.critiques) == 1
        assert result.failed_steps == [2]
    
    def test_successful_validation(self):
        """Successful validation result"""
        result = ValidationResult(
            is_valid=True,
            confidence=1.0,
            critiques=[],
            issues=[],
            failed_steps=[],
            repair_instructions="",
            is_hallucination_likely=False
        )
        assert result.is_valid
        assert len(result.critiques) == 0
