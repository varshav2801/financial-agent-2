"""Pytest configuration and shared fixtures"""
import pytest
from src.models.workflow_schema import (
    WorkflowPlan,
    WorkflowStep,
    ExtractTableParams,
    ExtractTextParams,
    Operand
)
from src.models.dataset import Document


@pytest.fixture
def sample_table():
    """Sample normalized table structure"""
    return {
        "2017": {
            "Net Income": 1245.0,
            "Revenue": 5000.0,
            "Operating Expenses": 3500.0
        },
        "2016": {
            "Net Income": 1180.0,
            "Revenue": 4800.0,
            "Operating Expenses": 3400.0
        },
        "2015": {
            "Net Income": 1050.0,
            "Revenue": 4500.0,
            "Operating Expenses": 3200.0
        }
    }


@pytest.fixture
def sample_document(sample_table):
    """Sample document with table and text"""
    return Document(
        id="test_doc_1",
        pre_text="The company showed strong performance in fiscal year 2017. Total debt refinancing charges amounted to $155.8 million.",
        post_text="Management expects continued growth in the coming quarters. The company maintained letters of credit totaling $127.1 million.",
        table=sample_table,
        qa_pairs=[]
    )


@pytest.fixture
def simple_extraction_plan():
    """Simple plan with single extraction step"""
    return WorkflowPlan(
        thought_process="Extract Net Income for 2017 from the financial table.",
        steps=[
            WorkflowStep(
                step_id=1,
                tool="extract_value",
                source="table",
                table_params=ExtractTableParams(
                    row_query="Net Income",
                    col_query="2017"
                )
            )
        ]
    )


@pytest.fixture
def computation_plan():
    """Plan with extraction and computation steps"""
    return WorkflowPlan(
        thought_process="Calculate percentage change in Net Income from 2016 to 2017.",
        steps=[
            WorkflowStep(
                step_id=1,
                tool="extract_value",
                source="table",
                table_params=ExtractTableParams(
                    row_query="Net Income",
                    col_query="2016"
                )
            ),
            WorkflowStep(
                step_id=2,
                tool="extract_value",
                source="table",
                table_params=ExtractTableParams(
                    row_query="Net Income",
                    col_query="2017"
                )
            ),
            WorkflowStep(
                step_id=3,
                tool="compute",
                operation="percentage_change",
                operands=[
                    Operand(type="reference", step_ref=1),
                    Operand(type="reference", step_ref=2)
                ]
            )
        ]
    )


@pytest.fixture
def invalid_forward_reference_plan():
    """Plan with forward reference error"""
    return WorkflowPlan(
        thought_process="Invalid plan referencing future step.",
        steps=[
            WorkflowStep(
                step_id=1,
                tool="compute",
                operation="add",
                operands=[
                    Operand(type="reference", step_ref=2),  # Forward reference
                    Operand(type="literal", value=100.0)
                ]
            ),
            WorkflowStep(
                step_id=2,
                tool="extract_value",
                source="table",
                table_params=ExtractTableParams(
                    row_query="Revenue",
                    col_query="2017"
                )
            )
        ]
    )
