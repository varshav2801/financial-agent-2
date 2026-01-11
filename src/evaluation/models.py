"""
Data models for evaluation tracking
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class ToolCallLog(BaseModel):
    """Log entry for a single tool call"""
    tool_name: str
    params: dict[str, Any]
    result: Any
    latency_ms: float
    success: bool
    error: str | None = None


class LLMCallLog(BaseModel):
    """Log entry for a single LLM API call"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str = "gpt-4o-mini"
    purpose: str | None = None  # e.g., "plan_generation", "text_extraction"


class ValidatorLog(BaseModel):
    """Log entry for validator execution"""
    valid: bool
    confidence: float
    failed_steps: list[str]  # Changed from list[int] to support unified plan string step_ids
    issues: list[str]
    retry_number: int


class TurnMetrics(BaseModel):
    """Metrics for a single question turn"""
    turn_idx: int
    question: str
    expected_answer: float | str
    actual_answer: float | str | None
    
    # Execution metrics
    turn_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    turn_llm_calls: int = 0
    turn_tool_calls: int = 0
    turn_latency_ms: float = 0.0
    
    # Plan metrics
    plan_type: str | None = None  # "table_lookup" or "multi_step_table_query"
    plan_steps: int = 0
    plan_complexity: float = 0.0
    
    # Validator metrics
    validator_retries: int = 0
    validator_valid: bool = False
    validator_confidence: float = 0.0
    validator_logs: list[ValidatorLog] = Field(default_factory=list)
    
    # Tool usage
    tools_used: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    llm_calls: list[LLMCallLog] = Field(default_factory=list)
    
    # Accuracy
    numerical_match: bool = False  # Binary: |P - T| < epsilon (exact match)
    financial_match: bool = False  # Consulting: |P - T| / |T| <= 0.01 (1% relative)
    soft_match: bool = False  # Entity & Logic: Forgives units, scaling, signage
    
    # Reasoning quality
    logic_recall: float | None = None  # (Shared Ops) / (Total Ops) vs ground truth
    ground_truth_program: str | None = None  # turn_program from dataset
    operations_per_turn: int = 0  # Number of compute operations in plan
    
    # Error tracking
    error_occurred: bool = False
    error_type: str | None = None
    error_message: str | None = None
    error_context: dict[str, Any] = Field(default_factory=dict)  # Additional error details
    
    # Timing
    timestamp_start: datetime = Field(default_factory=datetime.now)
    timestamp_end: datetime | None = None


class TraceRecord(BaseModel):
    """Complete trace data for one conversation"""
    trace_id: str
    conversation_id: str
    conversation_type: str
    num_turns: int
    
    # Aggregate metrics
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    
    avg_plan_steps: float = 0.0
    total_validator_retries: int = 0
    
    # Accuracy
    numerical_accuracy: float = 0.0  # Binary: Exact match proportion
    financial_accuracy: float = 0.0  # Consulting: 1% relative tolerance proportion
    soft_match_accuracy: float = 0.0  # Entity & Logic: Forgiving match proportion
    
    # Reasoning quality
    avg_logic_recall: float = 0.0  # Average reasoning trace score
    avg_operations_per_turn: float = 0.0
    
    # Errors
    error_occurred: bool = False
    error_type: str | None = None
    error_message: str | None = None
    failed_turn: int | None = None
    
    # Turn-level data
    turns: list[TurnMetrics] = Field(default_factory=list)
    
    # Timing
    timestamp_start: datetime = Field(default_factory=datetime.now)
    timestamp_end: datetime | None = None
    
    # Metadata
    has_type2_question: bool = False
    has_duplicate_columns: bool = False
    has_non_numeric_values: bool = False


class EvaluationSummary(BaseModel):
    """Aggregate statistics for entire evaluation run"""
    run_id: str
    timestamp: datetime
    
    # Sample info
    total_conversations: int
    total_turns: int
    sample_size: int
    random_seed: int
    
    # Accuracy metrics
    numerical_accuracy: float  # Binary: Exact match
    financial_accuracy: float  # Consulting: 1% relative tolerance
    soft_match_accuracy: float  # Entity & Logic: Forgiving match
    conversation_level_accuracy: float  # All turns correct
    
    # Reasoning quality
    avg_logic_recall: float  # Average reasoning trace score
    cost_per_correct_answer: float  # Total tokens / successful conversations
    
    # Complexity analysis
    accuracy_by_turn_number: dict[int, float]  # Track "Complexity Ceiling"
    avg_operations_by_turn: dict[int, float]  # Operations per turn number
    
    # Performance metrics
    avg_tokens_per_turn: float
    avg_latency_per_turn: float
    avg_llm_calls_per_turn: float
    avg_tool_calls_per_turn: float
    avg_plan_steps: float
    
    # Error analysis
    error_rate: float
    error_types: dict[str, int]
    
    # Validation stats
    avg_validator_retries: float
    avg_validator_confidence: float
    
    # Tool usage
    tool_usage_counts: dict[str, int]
    tool_success_rates: dict[str, float]
    
    # Conversation type breakdown
    accuracy_by_type: dict[str, float]


