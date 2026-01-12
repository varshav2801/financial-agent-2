"""Workflow execution schema models for sequential plan execution"""
from typing import Literal, Optional, Union, List
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Operand(BaseModel):
    """Operand for compute operations
    
    Supports two types:
    - reference: Points to a previous step result (via step_ref)
    - literal: Constant numeric value
    """
    model_config = ConfigDict(extra="forbid")
    
    type: Literal["reference", "literal"]
    step_ref: Optional[int] = Field(
        None, 
        description="Step ID to reference (required for type=reference). Use negative values for conversation history: -1=prev_0, -2=prev_1"
    )
    value: Optional[float] = Field(
        None, 
        description="Literal numeric value (required for type=literal)"
    )
    
    @field_validator("step_ref")
    @classmethod
    def validate_reference_operand(cls, v, info):
        """Ensure step_ref is provided when type is reference"""
        if info.data.get("type") == "reference" and v is None:
            raise ValueError("step_ref is required when type='reference'")
        return v
    
    @field_validator("value")
    @classmethod
    def validate_literal_operand(cls, v, info):
        """Ensure value is provided when type is literal"""
        if info.data.get("type") == "literal" and v is None:
            raise ValueError("value is required when type='literal'")
        return v


class ExtractTableParams(BaseModel):
    """Parameters for table extraction
    
    Uses fuzzy matching to find the best matching row and column
    """
    model_config = ConfigDict(extra="forbid")
    
    table_id: str = Field(
        default="main",
        description="Table identifier"
    )
    row_query: str = Field(
        description="Row name to find (fuzzy matched against available rows)"
    )
    col_query: str = Field(
        description="Column name to find (fuzzy matched against available columns/years)"
    )
    unit_normalization: Optional[str] = Field(
        None,
        description="Expected unit for normalization (million/billion/thousand)"
    )


class ExtractTextParams(BaseModel):
    """Parameters for text extraction
    
    Extracts numeric values from prose using LLM with keyword hints
    """
    model_config = ConfigDict(extra="forbid")
    
    context_window: Literal["pre_text", "post_text"] = Field(
        description="Which text section to search (both are searched automatically)"
    )
    search_keywords: List[str] = Field(
        description="2-4 semantic keywords to help locate the value (filter stopwords)",
        min_length=1
    )
    year: Optional[str] = Field(
        None,
        description="Year mentioned in the context (helps disambiguate multiple values)"
    )
    unit: str = Field(
        default="million",
        description="Expected unit of the value (million/billion/thousand/none)"
    )
    value_context: Optional[str] = Field(
        None,
        description="Brief description of what the value represents (helps with 'respectively' lists)"
    )


class WorkflowStep(BaseModel):
    """Unified workflow step supporting both extraction and computation
    
    Tool type determines which fields are required:
    - extract_value: requires source and extract_params
    - compute: requires operation and operands
    """
    model_config = ConfigDict(extra="forbid")
    
    step_id: int = Field(
        description="Unique sequential step identifier (starts from 1)"
    )
    tool: Literal["extract_value", "compute"] = Field(
        description="Tool to use: 'extract_value' for data extraction, 'compute' for arithmetic"
    )
    
    # Extract fields (used when tool="extract_value")
    source: Optional[Literal["table", "text"]] = Field(
        None,
        description="Data source for extraction: 'table' or 'text'"
    )
    table_params: Optional[ExtractTableParams] = Field(
        None,
        description="Table extraction parameters (used when source='table')"
    )
    text_params: Optional[ExtractTextParams] = Field(
        None,
        description="Text extraction parameters (used when source='text')"
    )
    
    # Compute fields (used when tool="compute")
    operation: Optional[Literal[
        "add",
        "subtract", 
        "multiply",
        "divide",
        "percentage",
        "percentage_change"
    ]] = Field(
        None,
        description="Arithmetic operation (required when tool='compute')"
    )
    operands: Optional[List[Operand]] = Field(
        None,
        description="List of operands for computation (1-2 items, required when tool='compute')"
    )
    
    @field_validator("source")
    @classmethod
    def validate_extract_source(cls, v, info):
        """Ensure source is provided for extract_value tool"""
        if info.data.get("tool") == "extract_value" and v is None:
            raise ValueError("source is required when tool='extract_value'")
        return v
    
    @field_validator("operation")
    @classmethod
    def validate_compute_operation(cls, v, info):
        """Ensure operation is provided for compute tool"""
        if info.data.get("tool") == "compute" and v is None:
            raise ValueError("operation is required when tool='compute'")
        return v
    
    @field_validator("operands")
    @classmethod
    def validate_compute_operands(cls, v, info):
        """Ensure operands are provided and valid for compute tool"""
        if info.data.get("tool") == "compute":
            if v is None:
                raise ValueError("operands are required when tool='compute'")
            if not (1 <= len(v) <= 2):
                raise ValueError("operands must have 1-2 items")
        return v


class WorkflowPlan(BaseModel):
    """Sequential workflow plan
    
    Represents a complete execution plan as a sequence of extract and compute steps
    """
    model_config = ConfigDict(extra="forbid")
    
    thought_process: str = Field(
        description="Internal reasoning explaining the plan strategy"
    )
    steps: List[WorkflowStep] = Field(
        description="Sequential list of execution steps",
        min_length=1
    )
    
    @field_validator("steps")
    @classmethod
    def validate_step_ids_sequential(cls, v):
        """Ensure step_ids are sequential starting from 1"""
        if not v:
            return v
        
        step_ids = [step.step_id for step in v]
        expected_ids = list(range(1, len(v) + 1))
        
        if step_ids != expected_ids:
            raise ValueError(
                f"Step IDs must be sequential starting from 1. Got {step_ids}, expected {expected_ids}"
            )
        
        return v


class WorkflowResult(BaseModel):
    """Result from workflow execution"""
    model_config = ConfigDict(extra="forbid")
    
    final_value: float = Field(
        description="Final computed value from the workflow"
    )
    step_results: dict[int, float] = Field(
        description="Memory dictionary mapping step_id to result value"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    success: bool = Field(
        description="Whether execution completed successfully"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if execution failed"
    )


class StepCritique(BaseModel):
    """Structured critique for a workflow step validation issue"""
    model_config = ConfigDict(extra="forbid")
    
    step_id: Optional[int] = Field(
        None,
        description="Step ID where the issue was found (None for plan-level issues)"
    )
    issue_type: str = Field(
        description="Type of issue: ReferenceError, ForwardReference, MissingParams, InvalidOperandCount, EmptyPlan, etc."
    )
    reason: str = Field(
        description="Why this is an issue - detailed explanation of what's wrong"
    )
    fix_suggestion: str = Field(
        description="How to fix the issue - actionable steps to resolve it"
    )


class ValidationResult(BaseModel):
    """Result of plan validation"""
    model_config = ConfigDict(extra="forbid")
    
    is_valid: bool = Field(description="Whether the plan is valid")
    confidence: float = Field(description="Confidence in validation (0.0-1.0)")
    issues: List[str] = Field(
        default_factory=list,
        description="Legacy: List of validation issues as strings (deprecated, use critiques)"
    )
    critiques: List[StepCritique] = Field(
        default_factory=list,
        description="Structured validation critiques with step_id, issue_type, reason, and fix_suggestion"
    )
    failed_steps: List[int] = Field(default_factory=list, description="Step IDs that failed validation")
    failure_type: Optional[str] = Field(None, description="Type of validation failure")
    repair_instructions: Optional[str] = Field(None, description="Instructions for repairing the plan")
    is_hallucination_likely: bool = Field(default=False, description="Whether hallucination is likely")
