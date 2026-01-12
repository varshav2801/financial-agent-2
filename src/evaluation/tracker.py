"""
Metrics tracking for evaluation
"""

import uuid
import time
import re
from datetime import datetime
from src.evaluation.models import (
    TraceRecord,
    TurnMetrics,
    ToolCallLog,
    LLMCallLog,
    ValidatorLog
)
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track all metrics during agent execution
    """
    
    def __init__(self):
        self.current_trace: TraceRecord | None = None
        self.current_turn: TurnMetrics | None = None
        self.turn_start_time: float = 0.0
    
    def start_trace(self, conversation_id: str, conversation_type: str, num_turns: int, features: dict) -> str:
        """
        Initialize a new trace
        
        Returns:
            trace_id: UUID for this trace
        """
        trace_id = str(uuid.uuid4())
        
        self.current_trace = TraceRecord(
            trace_id=trace_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            num_turns=num_turns,
            has_type2_question=features.get("has_type2_question", False),
            has_duplicate_columns=features.get("has_duplicate_columns", False),
            has_non_numeric_values=features.get("has_non_numeric_values", False),
            timestamp_start=datetime.now(),
        )
        
        logger.debug(f"Started trace {trace_id} for conversation {conversation_id}")
        return trace_id
    
    def start_turn(self, turn_idx: int, question: str, expected_answer: float | str, ground_truth_program: str | None = None):
        """Initialize metrics for a new turn"""
        self.current_turn = TurnMetrics(
            turn_idx=turn_idx,
            question=question,
            expected_answer=expected_answer,
            actual_answer=None,
            ground_truth_program=ground_truth_program,
            timestamp_start=datetime.now(),
        )
        self.turn_start_time = time.perf_counter()
        
        logger.debug(f"Started turn {turn_idx}: {question[:50]}...")
    
    def log_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        model: str = "gpt-4o-mini",
        purpose: str | None = None
    ):
        """Track an LLM API call"""
        if self.current_turn is None:
            # Silently ignore - this can happen during planning between turns
            return
        
        call_log = LLMCallLog(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            model=model,
            purpose=purpose,
        )
        
        self.current_turn.llm_calls.append(call_log)
        self.current_turn.turn_llm_calls += 1
        self.current_turn.turn_tokens += call_log.total_tokens
        self.current_turn.prompt_tokens += prompt_tokens
        self.current_turn.completion_tokens += completion_tokens
    
    def log_tool_call(
        self,
        tool_name: str,
        params: dict,
        result: any,
        latency_ms: float,
        success: bool = True,
        error: str | None = None
    ):
        """Track a tool call"""
        if self.current_turn is None:
            # Silently ignore - this can happen during planning between turns
            return
        
        tool_log = ToolCallLog(
            tool_name=tool_name,
            params=params,
            result=result,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )
        
        self.current_turn.tool_calls.append(tool_log)
        self.current_turn.turn_tool_calls += 1
        
        if tool_name not in self.current_turn.tools_used:
            self.current_turn.tools_used.append(tool_name)
    
    def log_validator_result(
        self,
        valid: bool,
        confidence: float,
        failed_steps: list[str],  # Unified plans use string step_ids
        issues: list[str],
        retry_number: int = 0
    ):
        """Track validator execution"""
        if self.current_turn is None:
            logger.warning("log_validator_result called without active turn")
            return
        
        validator_log = ValidatorLog(
            valid=valid,
            confidence=confidence,
            failed_steps=failed_steps,
            issues=issues,
            retry_number=retry_number,
        )
        
        self.current_turn.validator_logs.append(validator_log)
        
        # Update current validator state
        self.current_turn.validator_valid = valid
        self.current_turn.validator_confidence = confidence
        
        if retry_number > 0:
            self.current_turn.validator_retries = retry_number
    
    def log_plan(self, num_steps: int, complexity: float, plan_type: str | None = None, plan_object: any = None):
        """Track plan characteristics and calculate operations"""
        if self.current_turn is None:
            logger.warning("log_plan called without active turn")
            return
        
        self.current_turn.plan_type = plan_type
        self.current_turn.plan_steps = num_steps
        self.current_turn.plan_complexity = complexity
        
        # Count compute operations
        if plan_object and hasattr(plan_object, 'steps'):
            compute_ops = sum(1 for step in plan_object.steps if step.tool == "compute")
            self.current_turn.operations_per_turn = compute_ops
            
            # Calculate logic recall if we have ground truth program
            if self.current_turn.ground_truth_program:
                self.current_turn.logic_recall = self._calculate_logic_recall(
                    plan_object, 
                    self.current_turn.ground_truth_program
                )
    
    def _calculate_logic_recall(self, plan_object, ground_truth_program: str) -> float:
        """
        Calculate Logic Recall: (Shared Ops) / (Total Ground Truth Ops)
        
        Matches workflow plan operations against ground truth turn_program.
        Handles:
        - Extract-only (no operation in turn_program)
        - Basic operations: add, subtract, divide, multiply
        - Percentage calculations (divide followed by multiply/format)
        """
        if not ground_truth_program or not hasattr(plan_object, 'steps'):
            return 0.0
        
        # Parse ground truth operations
        gt_ops = self._parse_ground_truth_operations(ground_truth_program)
        if not gt_ops:
            # No operations means extract-only
            # Check if plan is also extract-only
            plan_ops = [step.operation for step in plan_object.steps if step.tool == "compute" and step.operation]
            return 1.0 if len(plan_ops) == 0 else 0.0
        
        # Extract plan operations
        plan_ops = []
        for step in plan_object.steps:
            if step.tool == "compute" and step.operation:
                plan_ops.append(step.operation)
        
        # Find shared operations (order-insensitive)
        shared_ops = 0
        gt_ops_copy = gt_ops.copy()
        
        for plan_op in plan_ops:
            if plan_op in gt_ops_copy:
                shared_ops += 1
                gt_ops_copy.remove(plan_op)  # Remove to handle duplicates correctly
        
        # Calculate recall: shared / total_ground_truth
        logic_recall = shared_ops / len(gt_ops) if len(gt_ops) > 0 else 0.0
        
        logger.debug(f"Logic Recall: {logic_recall:.2f} (shared={shared_ops}, gt_total={len(gt_ops)}, plan_ops={plan_ops}, gt_ops={gt_ops})")
        
        return logic_recall
    
    def _parse_ground_truth_operations(self, program: str) -> list[str]:
        """
        Parse ground truth turn_program to extract operations.
        
        Examples:
        - "4241" -> []  (extract only)
        - "subtract(8181, 20454)" -> ["subtract"]
        - "subtract(8181, 20454), divide(#0, 20454)" -> ["subtract", "divide"]
        - "add(4241, 2441)" -> ["add"]
        """
        operations = []
        
        # Check for common operations
        op_patterns = ["add", "subtract", "divide", "multiply", "exp"]
        
        for op in op_patterns:
            # Count occurrences of operation
            count = program.count(f"{op}(")
            operations.extend([op] * count)
        
        return operations
    
    def log_error(self, error_type: str, error_message: str, error_context: dict | None = None):
        """Log an error with additional context for analysis"""
        if self.current_turn:
            self.current_turn.error_occurred = True
            self.current_turn.error_type = error_type
            self.current_turn.error_message = error_message
            self.current_turn.error_context = error_context or {}
    
    def finalize_turn(self, actual_answer: float | str | None):
        """Complete turn and calculate metrics"""
        if self.current_turn is None:
            logger.warning("finalize_turn called without active turn")
            return
        
        # Record answer
        self.current_turn.actual_answer = actual_answer
        
        # Calculate latency
        turn_end_time = time.perf_counter()
        self.current_turn.turn_latency_ms = (turn_end_time - self.turn_start_time) * 1000
        self.current_turn.timestamp_end = datetime.now()
        
        # Calculate accuracy
        if actual_answer is not None:
            accuracy = self.calculate_accuracy(
                self.current_turn.expected_answer,
                actual_answer
            )
            self.current_turn.numerical_match = accuracy["numerical"]
            self.current_turn.financial_match = accuracy["financial"]
            self.current_turn.soft_match = accuracy["soft_match"]
        
        # Add to trace
        if self.current_trace:
            self.current_trace.turns.append(self.current_turn)
        
        logger.debug(f"Finalized turn {self.current_turn.turn_idx}")
    
    def calculate_accuracy(self, expected: float | str, actual: float | str) -> dict:
        """
        Calculate three accuracy metrics:
        
        1. Numerical Match (Binary): |P - T| < epsilon (exact match within floating-point tolerance)
        2. Financial Match (Consulting): Enhanced to handle:
           - Percentage format flexibility: 10.1% = 0.101 (decimal)
           - Rounding tolerance: 15.686... ≈ 15.7%
           - Unit agnostic: formatting doesn't matter
        3. Soft Match (Entity & Logic): Forgives units, scaling, and signage
           - Unit mismatches: 0.12 = 12% = 1,000,000 = 1M
           - Signage: -500 = (500)
           - Scaling: 125.0 vs 1.25 (digits identical but decimal shifted)
        
        Returns:
            dict with accuracy flags: numerical, financial, soft_match
        """
        # Convert to comparable format
        try:
            exp_val = self._normalize_value(expected)
            act_val = self._normalize_value(actual)
            
            abs_diff = abs(exp_val - act_val)
            epsilon = 1e-5  # Near-zero constant for floating-point comparison
            
            # 1. Numerical Accuracy (Binary): Exact match within epsilon
            numerical_match = abs_diff < epsilon
            
            # 2. Financial Accuracy (Enhanced): Format-agnostic with rounding tolerance
            financial_match = self._calculate_financial_match(exp_val, act_val, expected, actual, epsilon)
            
            # 3. Soft Match (Entity & Logic): Forgiving match
            soft_match = self._calculate_soft_match(exp_val, act_val, expected, actual)
            
            return {
                "numerical": numerical_match,
                "financial": financial_match,
                "soft_match": soft_match
            }
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Accuracy calculation error: {e}")
            return {
                "numerical": False,
                "financial": False,
                "soft_match": False
            }
    
    def _calculate_financial_match(
        self,
        exp_val: float,
        act_val: float,
        expected_raw: float | str,
        actual_raw: float | str,
        epsilon: float
    ) -> bool:
        """
        Enhanced financial accuracy check that handles:
        1. Percentage format flexibility (10.1% = 0.101)
        2. Rounding tolerance (15.686... ≈ 15.7%)
        3. Standard 1% relative error tolerance
        """
        abs_diff = abs(exp_val - act_val)
        
        # Direct match
        if abs_diff < epsilon:
            return True
        
        # Standard 1% relative error tolerance
        if exp_val != 0:
            relative_error = abs_diff / abs(exp_val)
            if relative_error <= 0.01:
                return True
        elif abs_diff < epsilon:
            return True
        
        # Handle percentage format flexibility: 10.1 vs 0.101
        # Check if one is ~100x the other (percentage vs decimal)
        # This handles: expected="11%" (0.11) vs actual=10.548 (percentage form)
        if exp_val != 0:
            # exp_val * 100 converts decimal to percentage for comparison
            scaled_exp = exp_val * 100
            if abs(scaled_exp - act_val) / max(abs(scaled_exp), abs(act_val)) <= 0.01:
                return True
            # Also check with rounding at percentage scale
            if abs(round(scaled_exp, 0) - round(act_val, 0)) < epsilon:
                return True
            if abs(round(scaled_exp, 1) - round(act_val, 1)) < epsilon:
                return True
                
        if act_val != 0:
            # act_val * 100 converts decimal to percentage for comparison
            scaled_act = act_val * 100
            if abs(exp_val - scaled_act) / max(abs(exp_val), abs(scaled_act)) <= 0.01:
                return True
            # Also check with rounding at percentage scale
            if abs(round(exp_val, 0) - round(scaled_act, 0)) < epsilon:
                return True
            if abs(round(exp_val, 1) - round(scaled_act, 1)) < epsilon:
                return True
        
        # Rounding tolerance at the same scale
        # This handles cases like 15.686274509803921 ≈ 15.7
        try:
            # Round to 0 decimal places (integers)
            exp_rounded = round(exp_val, 0)
            act_rounded = round(act_val, 0)
            if abs(exp_rounded - act_rounded) < epsilon:
                return True
            
            # Round both to 1 decimal place
            exp_rounded = round(exp_val, 1)
            act_rounded = round(act_val, 1)
            if abs(exp_rounded - act_rounded) < epsilon:
                return True
            
            # Also check rounding to 2 decimal places
            exp_rounded = round(exp_val, 2)
            act_rounded = round(act_val, 2)
            if abs(exp_rounded - act_rounded) < epsilon:
                return True
        except:
            pass
        
        # Check if the difference is just due to rounding precision
        # If the relative difference is within 0.5% (reasonable rounding error)
        if exp_val != 0:
            if abs_diff / abs(exp_val) <= 0.005:  # 0.5% tolerance for rounding
                return True
        
        return False
    
    def _calculate_soft_match(
        self,
        exp_val: float,
        act_val: float,
        expected_raw: float | str,
        actual_raw: float | str
    ) -> bool:
        """
        Soft Match: Forgives units, scaling, and signage
        
        Forgiveness rules:
        1. Unit mismatches: 0.12 = 12% = 0.0012 (scaling by 100 or 0.01)
        2. Signage: -500 = 500 (absolute values match)
        3. Scaling: If digits are identical but decimal shifted
        """
        epsilon = 1e-5
        abs_diff = abs(exp_val - act_val)
        
        # Already matches exactly
        if abs_diff < epsilon:
            return True
        
        # Check financial tolerance (1% relative)
        if exp_val != 0:
            if abs_diff / abs(exp_val) <= 0.01:
                return True
        elif abs_diff < epsilon:
            return True
        
        # Forgive signage: Check if absolute values match
        if abs(abs(exp_val) - abs(act_val)) < epsilon:
            return True
        
        # Forgive unit mismatches: Check scaling by 100 (percentage conversion)
        if abs(exp_val * 100 - act_val) < 0.1:
            return True
        if abs(exp_val - act_val * 100) < 0.1:
            return True
        
        # Forgive unit mismatches: Check scaling by 0.01
        if abs(exp_val * 0.01 - act_val) < epsilon:
            return True
        if abs(exp_val - act_val * 0.01) < epsilon:
            return True
        
        # Check if one has % sign and forgive with relative tolerance
        exp_has_pct = "%" in str(expected_raw)
        act_has_pct = "%" in str(actual_raw)
        
        if exp_has_pct != act_has_pct:
            # One has %, other doesn't - be more lenient
            if exp_val != 0 and abs_diff / abs(exp_val) <= 0.05:  # 5% tolerance for format mismatch
                return True
        
        return False
    
    def _normalize_value(self, value: float | str) -> float:
        """
        Convert value to float, handling percentages and formatting.
        
        Normalization rules:
        - Remove formatting: commas, dollar signs
        - Convert percentages to decimal: "10%" -> 0.1
        - Return raw float for everything else
        
        The comparison logic in _calculate_financial_match and _calculate_soft_match
        handles scale mismatches (e.g., 0.1 vs 10.0 for percentage format differences).
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        # Remove common formatting
        value_str = str(value).strip().replace(",", "").replace("$", "")
        
        # Handle percentage: convert to decimal
        if "%" in value_str:
            value_str = value_str.replace("%", "")
            return float(value_str) / 100
        
        return float(value_str)
    
    def finalize_trace(self) -> TraceRecord:
        """Complete trace and calculate aggregate metrics"""
        if self.current_trace is None:
            raise ValueError("No active trace to finalize")
        
        self.current_trace.timestamp_end = datetime.now()
        
        # Aggregate metrics from all turns
        for turn in self.current_trace.turns:
            self.current_trace.total_tokens += turn.turn_tokens
            self.current_trace.total_prompt_tokens += turn.prompt_tokens
            self.current_trace.total_completion_tokens += turn.completion_tokens
            self.current_trace.total_llm_calls += turn.turn_llm_calls
            self.current_trace.total_tool_calls += turn.turn_tool_calls
            self.current_trace.total_latency_ms += turn.turn_latency_ms
            self.current_trace.total_validator_retries += turn.validator_retries
            
            # Track first error
            if turn.error_occurred and not self.current_trace.error_occurred:
                self.current_trace.error_occurred = True
                self.current_trace.error_type = turn.error_type
                self.current_trace.error_message = turn.error_message
                self.current_trace.failed_turn = turn.turn_idx
        
        # Calculate averages
        if self.current_trace.num_turns > 0:
            self.current_trace.avg_plan_steps = sum(
                t.plan_steps for t in self.current_trace.turns
            ) / self.current_trace.num_turns
            
            # Calculate accuracy rates
            numerical_correct = sum(1 for t in self.current_trace.turns if t.numerical_match)
            financial_correct = sum(1 for t in self.current_trace.turns if t.financial_match)
            soft_match_correct = sum(1 for t in self.current_trace.turns if t.soft_match)
            
            self.current_trace.numerical_accuracy = numerical_correct / self.current_trace.num_turns
            self.current_trace.financial_accuracy = financial_correct / self.current_trace.num_turns
            self.current_trace.soft_match_accuracy = soft_match_correct / self.current_trace.num_turns
            
            # Calculate average logic recall (only for turns with ground truth)
            logic_recalls = [t.logic_recall for t in self.current_trace.turns if t.logic_recall is not None]
            self.current_trace.avg_logic_recall = sum(logic_recalls) / len(logic_recalls) if logic_recalls else 0.0
            
            # Calculate average operations per turn
            self.current_trace.avg_operations_per_turn = sum(
                t.operations_per_turn for t in self.current_trace.turns
            ) / self.current_trace.num_turns
        
        logger.info(f"Finalized trace {self.current_trace.trace_id}: "
                   f"numerical={self.current_trace.numerical_accuracy:.2%}, "
                   f"financial={self.current_trace.financial_accuracy:.2%}, "
                   f"soft_match={self.current_trace.soft_match_accuracy:.2%}, "
                   f"logic_recall={self.current_trace.avg_logic_recall:.2%}")
        
        trace = self.current_trace
        self.current_trace = None
        self.current_turn = None
        
        return trace

