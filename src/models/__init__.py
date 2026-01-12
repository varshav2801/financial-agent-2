"""Models package for ConvFinQA agent."""

from src.models.dataset import Document, ConvFinQARecord
from src.models.workflow_schema import WorkflowPlan, WorkflowStep, StepCritique
from src.models.tool_schemas import (
    TextExtractParams,
    TextConstantParams,
    TextExtractionResponse,
    TextExtractionResult,
)
from src.models.exceptions import (
    ToolExecutionError,
    TableExtractionError,
    TextExtractionError,
    MathOperationError,
    PlanExecutionError,
    StepExecutionError,
)

__all__ = [
    # Dataset models
    "Document",
    "ConvFinQARecord",
    # Workflow models
    "WorkflowPlan",
    "WorkflowStep",
    "StepCritique",
    # Tool schema models
    "TextExtractParams",
    "TextConstantParams",
    "TextExtractionResponse",
    "TextExtractionResult",
    # Exception classes
    "ToolExecutionError",
    "TableExtractionError",
    "TextExtractionError",
    "MathOperationError",
    "PlanExecutionError",
    "StepExecutionError",
]
