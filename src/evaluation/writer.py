"""
Result writing for evaluation outputs
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from src.evaluation.models import TraceRecord, EvaluationSummary
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ResultWriter:
    """
    Write evaluation results to CSV and JSON files
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.traces: list[TraceRecord] = []
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "traces").mkdir(exist_ok=True)
        
        logger.info(f"Results will be written to: {self.output_dir}")
    
    def write_trace_json(self, trace: TraceRecord):
        """Write detailed JSON log for a single trace"""
        trace_path = self.output_dir / "traces" / f"{trace.trace_id}.json"
        
        with open(trace_path, "w") as f:
            json.dump(trace.model_dump(), f, indent=2, default=str)
        
        self.traces.append(trace)
    
    def write_summary_csv(self):
        """Write high-level summary CSV"""
        summary_path = self.output_dir / "summary.csv"
        
        if not self.traces:
            logger.warning("No traces to write to summary CSV")
            return
        
        logger.info(f"Writing summary CSV with {len(self.traces)} conversations")
        
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "trace_id",
                "conversation_id",
                "conversation_type",
                "num_turns",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_llm_calls",
                "total_tool_calls",
                "total_latency_ms",
                "avg_plan_steps",
                "total_validator_retries",
                "numerical_accuracy",
                "financial_accuracy",
                "soft_match_accuracy",
                "avg_logic_recall",
                "avg_operations_per_turn",
                "error_occurred",
                "error_type",
                "error_message",
                "failed_turn",
                "timestamp_start",
                "timestamp_end",
                "has_type2_question",
                "has_non_numeric_values",
            ])
            
            # Data rows
            for trace in self.traces:
                writer.writerow([
                    trace.trace_id,
                    trace.conversation_id,
                    trace.conversation_type,
                    trace.num_turns,
                    trace.total_tokens,
                    trace.total_prompt_tokens,
                    trace.total_completion_tokens,
                    trace.total_llm_calls,
                    trace.total_tool_calls,
                    trace.total_latency_ms,
                    trace.avg_plan_steps,
                    trace.total_validator_retries,
                    trace.numerical_accuracy,
                    trace.financial_accuracy,
                    trace.soft_match_accuracy,
                    trace.avg_logic_recall,
                    trace.avg_operations_per_turn,
                    trace.error_occurred,
                    trace.error_type or "",
                    trace.error_message or "",
                    trace.failed_turn or "",
                    trace.timestamp_start,
                    trace.timestamp_end,
                    trace.has_type2_question,
                    trace.has_non_numeric_values,
                ])
        
        logger.info(f"Summary CSV written to: {summary_path}")
    
    def write_turn_level_csv(self):
        """Write turn-by-turn detailed metrics CSV"""
        turns_path = self.output_dir / "turns.csv"
        
        if not self.traces:
            logger.warning("No traces to write to turns CSV")
            return
        
        logger.info("Writing turn-level CSV")
        
        with open(turns_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "trace_id",
                "conversation_id",
                "turn_idx",
                "question",
                "expected_answer",
                "actual_answer",
                "turn_tokens",
                "prompt_tokens",
                "completion_tokens",
                "turn_llm_calls",
                "turn_tool_calls",
                "turn_latency_ms",
                "plan_type",
                "plan_steps",
                "plan_complexity",
                "validator_retries",
                "validator_valid",
                "validator_confidence",
                "tools_used",
                "numerical_match",
                "financial_match",
                "soft_match",
                "logic_recall",
                "operations_per_turn",
                "ground_truth_program",
                "error_occurred",
                "error_type",
                "error_message",
            ])
            
            # Data rows
            for trace in self.traces:
                for turn in trace.turns:
                    writer.writerow([
                        trace.trace_id,
                        trace.conversation_id,
                        turn.turn_idx,
                        turn.question,
                        turn.expected_answer,
                        turn.actual_answer or "",
                        turn.turn_tokens,
                        turn.prompt_tokens,
                        turn.completion_tokens,
                        turn.turn_llm_calls,
                        turn.turn_tool_calls,
                        turn.turn_latency_ms,
                        turn.plan_type or "",
                        turn.plan_steps,
                        turn.plan_complexity,
                        turn.validator_retries,
                        turn.validator_valid,
                        turn.validator_confidence,
                        ",".join(turn.tools_used),
                        turn.numerical_match,
                        turn.financial_match,
                        turn.soft_match,
                        turn.logic_recall or "",
                        turn.operations_per_turn,
                        turn.ground_truth_program or "",
                        turn.error_occurred,
                        turn.error_type or "",
                        turn.error_message or "",
                    ])
        
        logger.info(f"Turn-level CSV written to: {turns_path}")
    
    def write_validation_log(self):
        """Write all validator responses to separate CSV"""
        validation_path = self.output_dir / "validation.csv"
        
        if not self.traces:
            logger.warning("No traces to write to validation log")
            return
        
        logger.info("Writing validation log CSV")
        
        with open(validation_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "trace_id",
                "conversation_id",
                "turn_idx",
                "retry_number",
                "valid",
                "confidence",
                "failed_steps",
                "issues",
            ])
            
            # Data rows
            for trace in self.traces:
                for turn in trace.turns:
                    for val_log in turn.validator_logs:
                        writer.writerow([
                            trace.trace_id,
                            trace.conversation_id,
                            turn.turn_idx,
                            val_log.retry_number,
                            val_log.valid,
                            val_log.confidence,
                            ",".join(map(str, val_log.failed_steps)),
                            "; ".join(val_log.issues),
                        ])
        
        logger.info(f"Validation log written to: {validation_path}")
    
    def write_error_log(self):
        """Write detailed error log for error mode analysis"""
        error_path = self.output_dir / "errors.csv"
        
        if not self.traces:
            logger.warning("No traces to write to error log")
            return
        
        logger.info("Writing error log CSV")
        
        with open(error_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "trace_id",
                "conversation_id",
                "turn_idx",
                "question",
                "expected_answer",
                "error_type",
                "error_message",
                "has_type2_question",
                "has_non_numeric_values",
                "plan_steps",
                "operations_per_turn",
                "error_context",
            ])
            
            # Data rows - only failed turns
            for trace in self.traces:
                for turn in trace.turns:
                    if turn.error_occurred:
                        writer.writerow([
                            trace.trace_id,
                            trace.conversation_id,
                            turn.turn_idx,
                            turn.question,
                            turn.expected_answer,
                            turn.error_type or "",
                            turn.error_message or "",
                            trace.has_type2_question,
                            trace.has_non_numeric_values,
                            turn.plan_steps,
                            turn.operations_per_turn,
                            str(turn.error_context),
                        ])
        
        logger.info(f"Error log written to: {error_path}")
    
    def generate_summary_statistics(self, run_id: str, sample_size: int, random_seed: int) -> EvaluationSummary:
        """Generate aggregate statistics from all traces"""
        if not self.traces:
            raise ValueError("No traces available for statistics")
        
        logger.info("Generating summary statistics")
        
        total_turns = sum(trace.num_turns for trace in self.traces)
        
        # Accuracy metrics
        numerical_correct_turns = sum(
            sum(1 for t in trace.turns if t.numerical_match)
            for trace in self.traces
        )
        financial_correct_turns = sum(
            sum(1 for t in trace.turns if t.financial_match)
            for trace in self.traces
        )
        soft_match_turns = sum(
            sum(1 for t in trace.turns if t.soft_match)
            for trace in self.traces
        )
        conversation_level_correct = sum(
            1 for trace in self.traces
            if trace.numerical_accuracy == 1.0
        )
        
        # Reasoning quality metrics
        all_logic_recalls = [
            turn.logic_recall
            for trace in self.traces
            for turn in trace.turns
            if turn.logic_recall is not None
        ]
        avg_logic_recall = sum(all_logic_recalls) / len(all_logic_recalls) if all_logic_recalls else 0.0
        
        # Cost per correct answer: Total tokens / successful conversations
        successful_conversations = sum(1 for trace in self.traces if trace.numerical_accuracy == 1.0)
        total_tokens_all = sum(trace.total_tokens for trace in self.traces)
        cost_per_correct = total_tokens_all / successful_conversations if successful_conversations > 0 else 0.0
        
        # Accuracy by turn number
        accuracy_by_turn = {}
        operations_by_turn = {}
        turns_by_number = defaultdict(lambda: {"correct": 0, "total": 0, "operations": []})
        
        for trace in self.traces:
            for turn in trace.turns:
                turn_num = turn.turn_idx
                turns_by_number[turn_num]["total"] += 1
                turns_by_number[turn_num]["operations"].append(turn.operations_per_turn)
                if turn.numerical_match:
                    turns_by_number[turn_num]["correct"] += 1
        
        for turn_num, stats in turns_by_number.items():
            accuracy_by_turn[turn_num] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            operations_by_turn[turn_num] = sum(stats["operations"]) / len(stats["operations"]) if stats["operations"] else 0.0
        
        # Performance metrics
        avg_tokens = sum(trace.total_tokens for trace in self.traces) / total_turns
        avg_latency = sum(trace.total_latency_ms for trace in self.traces) / total_turns
        avg_llm_calls = sum(trace.total_llm_calls for trace in self.traces) / total_turns
        avg_tool_calls = sum(trace.total_tool_calls for trace in self.traces) / total_turns
        avg_plan_steps = sum(trace.avg_plan_steps for trace in self.traces) / len(self.traces)
        
        # Error analysis
        errors = [trace for trace in self.traces if trace.error_occurred]
        error_rate = len(errors) / len(self.traces)
        
        error_types = defaultdict(int)
        for trace in errors:
            if trace.error_type:
                error_types[trace.error_type] += 1
        
        # Validation stats
        total_retries = sum(trace.total_validator_retries for trace in self.traces)
        avg_retries = total_retries / len(self.traces)
        
        all_confidences = [
            turn.validator_confidence
            for trace in self.traces
            for turn in trace.turns
            if turn.validator_confidence > 0
        ]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        # Tool usage
        tool_counts = defaultdict(int)
        tool_successes = defaultdict(int)
        tool_totals = defaultdict(int)
        
        for trace in self.traces:
            for turn in trace.turns:
                for tool in turn.tools_used:
                    tool_counts[tool] += 1
                for tool_call in turn.tool_calls:
                    tool_totals[tool_call.tool_name] += 1
                    if tool_call.success:
                        tool_successes[tool_call.tool_name] += 1
        
        tool_success_rates = {
            tool: tool_successes[tool] / tool_totals[tool]
            for tool in tool_totals
        }
        
        # Accuracy by conversation type
        type_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
        for trace in self.traces:
            type_accuracy[trace.conversation_type]["total"] += 1
            if trace.numerical_accuracy == 1.0:
                type_accuracy[trace.conversation_type]["correct"] += 1
        
        accuracy_by_type = {
            conv_type: stats["correct"] / stats["total"]
            for conv_type, stats in type_accuracy.items()
        }
        
        summary = EvaluationSummary(
            run_id=run_id,
            timestamp=datetime.now(),
            total_conversations=len(self.traces),
            total_turns=total_turns,
            sample_size=sample_size,
            random_seed=random_seed,
            numerical_accuracy=numerical_correct_turns / total_turns,
            financial_accuracy=financial_correct_turns / total_turns,
            soft_match_accuracy=soft_match_turns / total_turns,
            conversation_level_accuracy=conversation_level_correct / len(self.traces),
            avg_logic_recall=avg_logic_recall,
            cost_per_correct_answer=cost_per_correct,
            accuracy_by_turn_number=accuracy_by_turn,
            avg_operations_by_turn=operations_by_turn,
            avg_tokens_per_turn=avg_tokens,
            avg_latency_per_turn=avg_latency,
            avg_llm_calls_per_turn=avg_llm_calls,
            avg_tool_calls_per_turn=avg_tool_calls,
            avg_plan_steps=avg_plan_steps,
            error_rate=error_rate,
            error_types=dict(error_types),
            avg_validator_retries=avg_retries,
            avg_validator_confidence=avg_confidence,
            tool_usage_counts=dict(tool_counts),
            tool_success_rates=tool_success_rates,
            accuracy_by_type=accuracy_by_type,
        )
        
        # Write summary statistics to JSON
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)
        
        logger.info(f"Statistics written to: {stats_path}")
        
        return summary


