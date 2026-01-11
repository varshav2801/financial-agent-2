"""Workflow executor with register pattern for sequential step execution"""
import time
from typing import TYPE_CHECKING
from src.models.workflow_schema import (
    WorkflowPlan,
    WorkflowResult,
    WorkflowStep,
    Operand,
    ExtractTableParams,
    ExtractTextParams
)
from src.models.dataset import Document
from src.models.exceptions import (
    StepExecutionError,
    TableExtractionError,
    TextExtractionError,
    MathOperationError
)
from src.tools.workflow_table_tool import WorkflowTableTool
from src.tools.text_tool import TextTool
from src.utils.table_normalizer import normalize_table
from src.logger import get_logger

if TYPE_CHECKING:
    from src.evaluation.tracker import MetricsTracker

logger = get_logger(__name__)


class WorkflowExecutor:
    """
    Execute workflow plans using register pattern.
    
    Key concepts:
    - Memory: Dictionary mapping step_id -> result value
    - Sequential: Steps execute in order, no dependency analysis needed
    - Register pattern: Each step stores result in memory, later steps reference by ID
    """
    
    def __init__(self, tracker: "MetricsTracker | None" = None) -> None:
        self.tracker = tracker
        self.table_tool = WorkflowTableTool()
        self.text_tool = TextTool(tracker=tracker)
        self.memory: dict[int, float] = {}
    
    async def execute(
        self,
        plan: WorkflowPlan,
        document: Document,
        previous_answers: dict[str, float | dict],
        current_question: str | None = None
    ) -> WorkflowResult:
        """
        Execute workflow plan sequentially.
        
        Args:
            plan: WorkflowPlan with sequential steps
            document: Document with table and text
            previous_answers: Previous turn answers (for conversation history)
            current_question: Current question for context
            
        Returns:
            WorkflowResult with final value and execution details
        """
        start_time = time.time()
        
        try:
            # Reset memory for new execution
            self.memory = {}
            
            # Pre-populate memory with conversation history (negative indices)
            # prev_0 (most recent) = -1, prev_1 = -2, etc.
            for idx, (key, value) in enumerate(previous_answers.items()):
                negative_idx = -(idx + 1)
                # Extract numeric value from metadata dict or use directly if float
                numeric_value = value['value'] if isinstance(value, dict) else value
                self.memory[negative_idx] = float(numeric_value)
                logger.debug(f"Loaded history: memory[{negative_idx}] = {numeric_value} (from {key})")
            
            # Normalize table to standard format
            normalized_table = normalize_table(document.table)
            
            # Execute each step sequentially
            for step in plan.steps:
                try:
                    if step.tool == "extract_value":
                        result = await self._execute_extract(
                            step=step,
                            table=normalized_table,
                            pre_text=document.pre_text,
                            post_text=document.post_text,
                            question=current_question
                        )
                    elif step.tool == "compute":
                        result = await self._execute_compute(step)
                    else:
                        raise StepExecutionError(
                            f"Unknown tool: {step.tool}",
                            step_id=step.step_id
                        )
                    
                    # Store result in memory
                    self.memory[step.step_id] = result
                    logger.info(f"Step {step.step_id} completed: {result}")
                    
                except Exception as e:
                    raise StepExecutionError(
                        f"Step {step.step_id} failed: {str(e)}",
                        step_id=step.step_id,
                        original_error=e
                    ) from e
            
            # Final result is the last step's output
            final_value = self.memory[plan.steps[-1].step_id]
            execution_time_ms = (time.time() - start_time) * 1000
            
            return WorkflowResult(
                final_value=final_value,
                step_results=self.memory.copy(),
                execution_time_ms=execution_time_ms,
                success=True,
                error=None
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Workflow execution failed: {e}")
            
            # Log error to tracker
            if self.tracker:
                error_context = {
                    "execution_time_ms": execution_time_ms,
                    "completed_steps": list(self.memory.keys()),
                    "num_plan_steps": len(plan.steps),
                    "stage": "workflow_execution"
                }
                
                # Add step-specific context if it's a StepExecutionError
                if isinstance(e, StepExecutionError):
                    error_context["failed_step_id"] = e.step_id if hasattr(e, 'step_id') else None
                
                self.tracker.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_context=error_context
                )
            
            return WorkflowResult(
                final_value=0.0,
                step_results=self.memory.copy(),
                execution_time_ms=execution_time_ms,
                success=False,
                error=str(e)
            )
    
    async def _execute_extract(
        self,
        step: WorkflowStep,
        table: dict[str, dict[str, float | str]],
        pre_text: str,
        post_text: str,
        question: str | None
    ) -> float:
        """
        Execute extraction step.
        
        Args:
            step: WorkflowStep with source and extraction params
            table: Normalized table data
            pre_text: Pre-table text
            post_text: Post-table text
            question: Current question for context
            
        Returns:
            Extracted numeric value
        """
        if step.source == "table":
            if step.table_params is None:
                raise TableExtractionError(
                    "table_params is required for table extraction",
                    tool_name="workflow_table_tool",
                    details={"step_id": step.step_id}
                )
            
            tool_start = time.time()
            try:
                result = await self.table_tool.extract_value(
                    params=step.table_params,
                    table=table
                )
                tool_latency = (time.time() - tool_start) * 1000
                
                # Log tool call
                if self.tracker:
                    self.tracker.log_tool_call(
                        tool_name="workflow_table_tool",
                        params=step.table_params.model_dump(),
                        result=result,
                        latency_ms=tool_latency,
                        success=True
                    )
                
                return result
            except Exception as e:
                tool_latency = (time.time() - tool_start) * 1000
                if self.tracker:
                    self.tracker.log_tool_call(
                        tool_name="workflow_table_tool",
                        params=step.table_params.model_dump(),
                        result=None,
                        latency_ms=tool_latency,
                        success=False,
                        error=str(e)
                    )
                raise
        
        elif step.source == "text":
            if step.text_params is None:
                raise TextExtractionError(
                    "text_params is required for text extraction",
                    tool_name="text_tool",
                    details={"step_id": step.step_id}
                )
            
            # Convert ExtractTextParams to dict for text_tool compatibility
            params_dict = {
                "context_window": step.text_params.context_window,
                "keywords": step.text_params.search_keywords,
                "year": step.text_params.year,
                "unit": step.text_params.unit,
                "value_context": step.text_params.value_context
            }
            
            tool_start = time.time()
            try:
                result = await self.text_tool.execute(
                    action="extract_numeric",
                    params=params_dict,
                    pre_text=pre_text,
                    post_text=post_text,
                    question=question
                )
                tool_latency = (time.time() - tool_start) * 1000
                
                # text_tool.execute returns {"value": float}
                value = float(result["value"]) if isinstance(result, dict) and "value" in result else float(result)
                
                # Log tool call
                if self.tracker:
                    self.tracker.log_tool_call(
                        tool_name="text_tool",
                        params=params_dict,
                        result=value,
                        latency_ms=tool_latency,
                        success=True
                    )
                
                return value
            except Exception as e:
                tool_latency = (time.time() - tool_start) * 1000
                if self.tracker:
                    self.tracker.log_tool_call(
                        tool_name="text_tool",
                        params=params_dict,
                        result=None,
                        latency_ms=tool_latency,
                        success=False,
                        error=str(e)
                    )
                raise
        
        else:
            raise StepExecutionError(
                f"Unknown extraction source: {step.source}",
                step_id=step.step_id
            )
    
    async def _execute_compute(self, step: WorkflowStep) -> float:
        """
        Execute computation step.
        
        Args:
            step: WorkflowStep with operation and operands
            
        Returns:
            Computed result
        """
        if step.operation is None or step.operands is None:
            raise MathOperationError(
                "operation and operands are required for compute",
                operation="unknown",
                details={"step_id": step.step_id}
            )
        
        compute_start = time.time()
        try:
            # Resolve operands to values
            values = [self._resolve_operand(op, step.step_id) for op in step.operands]
            
            # Apply operation
            result = self._apply_operation(step.operation, values, step.step_id)
            compute_latency = (time.time() - compute_start) * 1000
            
            # Log tool call
            if self.tracker:
                self.tracker.log_tool_call(
                    tool_name="compute",
                    params={
                        "operation": step.operation,
                        "operands": [op.model_dump() for op in step.operands],
                        "values": values
                    },
                    result=result,
                    latency_ms=compute_latency,
                    success=True
                )
            
            return result
        except Exception as e:
            compute_latency = (time.time() - compute_start) * 1000
            if self.tracker:
                self.tracker.log_tool_call(
                    tool_name="compute",
                    params={
                        "operation": step.operation if step.operation else "unknown",
                        "operands": [op.model_dump() for op in step.operands] if step.operands else []
                    },
                    result=None,
                    latency_ms=compute_latency,
                    success=False,
                    error=str(e)
                )
            raise
    
    def _resolve_operand(self, operand: Operand, current_step_id: int) -> float:
        """
        Resolve operand to numeric value.
        
        Args:
            operand: Operand (reference or literal)
            current_step_id: Current step ID (for error messages)
            
        Returns:
            Numeric value
            
        Raises:
            StepExecutionError: If reference not found or forward reference
        """
        if operand.type == "reference":
            if operand.step_ref is None:
                raise StepExecutionError(
                    f"Reference operand missing step_ref",
                    step_id=current_step_id
                )
            
            # Check for forward reference
            if operand.step_ref >= current_step_id:
                raise StepExecutionError(
                    f"Forward reference detected: step {current_step_id} references step {operand.step_ref}",
                    step_id=current_step_id
                )
            
            # Check if reference exists in memory
            if operand.step_ref not in self.memory:
                raise StepExecutionError(
                    f"Reference step_ref={operand.step_ref} not found in memory. Available: {list(self.memory.keys())}",
                    step_id=current_step_id
                )
            
            return self.memory[operand.step_ref]
        
        elif operand.type == "literal":
            if operand.value is None:
                raise StepExecutionError(
                    f"Literal operand missing value",
                    step_id=current_step_id
                )
            return operand.value
        
        else:
            raise StepExecutionError(
                f"Unknown operand type: {operand.type}",
                step_id=current_step_id
            )
    
    def _apply_operation(
        self,
        operation: str,
        values: list[float],
        step_id: int
    ) -> float:
        """
        Apply arithmetic operation to values.
        
        Args:
            operation: Operation name
            values: List of numeric values (1-2 items)
            step_id: Current step ID (for error messages)
            
        Returns:
            Result of operation
            
        Raises:
            MathOperationError: If operation fails
        """
        try:
            if operation == "add":
                if len(values) < 1:
                    raise MathOperationError(
                        "Add requires at least 1 operand",
                        operation=operation,
                        inputs=values
                    )
                return sum(values)
            
            elif operation == "subtract":
                if len(values) != 2:
                    raise MathOperationError(
                        "Subtract requires exactly 2 operands",
                        operation=operation,
                        inputs=values
                    )
                return values[0] - values[1]
            
            elif operation == "multiply":
                if len(values) != 2:
                    raise MathOperationError(
                        "Multiply requires exactly 2 operands",
                        operation=operation,
                        inputs=values
                    )
                return values[0] * values[1]
            
            elif operation == "divide":
                if len(values) != 2:
                    raise MathOperationError(
                        "Divide requires exactly 2 operands",
                        operation=operation,
                        inputs=values
                    )
                if values[1] == 0:
                    raise MathOperationError(
                        "Division by zero",
                        operation=operation,
                        inputs=values
                    )
                return values[0] / values[1]
            
            elif operation == "percentage":
                if len(values) != 2:
                    raise MathOperationError(
                        "Percentage requires exactly 2 operands (part, whole)",
                        operation=operation,
                        inputs=values
                    )
                if values[1] == 0:
                    raise MathOperationError(
                        "Percentage with zero denominator",
                        operation=operation,
                        inputs=values
                    )
                return (values[0] / values[1]) * 100
            
            elif operation == "percentage_change":
                if len(values) != 2:
                    raise MathOperationError(
                        "Percentage change requires exactly 2 operands (old, new)",
                        operation=operation,
                        inputs=values
                    )
                if values[0] == 0:
                    raise MathOperationError(
                        "Percentage change with zero baseline",
                        operation=operation,
                        inputs=values
                    )
                return ((values[1] - values[0]) / values[0]) * 100
            
            else:
                raise MathOperationError(
                    f"Unknown operation: {operation}",
                    operation=operation,
                    inputs=values
                )
                
        except MathOperationError:
            raise
        except Exception as e:
            raise MathOperationError(
                f"Operation '{operation}' failed: {str(e)}",
                operation=operation,
                inputs=values
            ) from e
