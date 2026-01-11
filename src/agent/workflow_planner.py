"""Workflow planner for generating WorkflowPlan using structured outputs"""
import time
from typing import TYPE_CHECKING
from pydantic import ValidationError
from src.models.workflow_schema import WorkflowPlan
from src.models.dataset import Document
from src.prompts.workflow_planner import (
    WORKFLOW_PLANNER_SYSTEM_PROMPT,
    WORKFLOW_PLANNER_USER_TEMPLATE
)
from src.services.llm_client import get_llm_client
from src.utils.table_normalizer import normalize_table
from src.logger import get_logger

if TYPE_CHECKING:
    from src.evaluation.tracker import MetricsTracker

logger = get_logger(__name__)


class WorkflowPlanner:
    """
    Generate WorkflowPlan using LLM structured outputs.
    
    Key differences from legacy Planner:
    - Generates WorkflowPlan (not ExecutionPlan)
    - Uses comprehensive prompt with 9 critical patterns
    - Cleaner schema with no empty fields
    - Register pattern references (step_ref instead of field names)
    """
    
    def __init__(self, llm_client=None, tracker: "MetricsTracker | None" = None) -> None:
        self.llm_client = llm_client if llm_client is not None else get_llm_client()
        self.tracker = tracker
    
    async def create_plan(
        self,
        question: str,
        document: Document,
        previous_answers: dict[str, float | dict],
        conversation_history: list[dict] | None = None
    ) -> WorkflowPlan:
        """
        Create workflow plan using LLM with structured output.
        
        Args:
            question: User question to answer
            document: Document with table and text
            previous_answers: Previous turn answers for multi-turn context
            conversation_history: Optional conversation history
            
        Returns:
            WorkflowPlan with sequential extract and compute steps
        """
        start_time = time.time()
        
        # Normalize table to standard format (table[year][metric])
        normalized_table = normalize_table(document.table)
        
        table_years = list(normalized_table.keys())
        # Get metrics from first row
        table_metrics = list(next(iter(normalized_table.values())).keys()) if normalized_table else []
        
        # Extract document context (truncate to avoid overwhelming the LLM)
        pre_text = document.pre_text[:800] if document.pre_text else "None"
        post_text = document.post_text[:800] if document.post_text else "None"
        
        # Format previous answers with metadata for prompt
        if previous_answers:
            prev_list = []
            for key, ans in previous_answers.items():
                if isinstance(ans, dict):
                    # New format with metadata
                    prev_list.append(
                        f"{key}: {ans['value']} (entity: {ans['entity']}, operation: {ans['operation']})"
                    )
                else:
                    # Legacy format (just value)
                    prev_list.append(f"{key}: {ans}")
            prev_answers_str = "\n".join(prev_list)
        else:
            prev_answers_str = "None"
        
        user_message = WORKFLOW_PLANNER_USER_TEMPLATE.format(
            question=question,
            table_years=table_years,
            table_metrics=table_metrics,
            pre_text=pre_text,
            post_text=post_text,
            previous_answers=prev_answers_str
        )
        
        logger.info(f"Generating workflow plan for question: {question}")
        logger.debug(f"Available years: {table_years}")
        logger.debug(f"Available metrics: {len(table_metrics)} metrics")
        
        try:
            # Use structured output to get WorkflowPlan
            logger.debug("Calling LLM for plan generation...")
            response = await self.llm_client.parse_completion(
                messages=[
                    {"role": "system", "content": WORKFLOW_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                response_format=WorkflowPlan
            )
            logger.debug("LLM response received")
            
            plan = response.choices[0].message.parsed
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Plan generated: {len(plan.steps)} steps in {elapsed_ms:.1f}ms"
            )
            
            # Log LLM call to tracker
            if self.tracker and hasattr(response, 'usage') and response.usage:
                self.tracker.log_llm_call(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=elapsed_ms,
                    model=response.model if hasattr(response, 'model') else "unknown",
                    purpose="plan_generation"
                )
            
            return plan
            
        except ValidationError as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Plan validation failed: {e}")
            
            # Log error to tracker
            if self.tracker:
                self.tracker.log_error(
                    error_type="ValidationError",
                    error_message=f"Invalid plan generated: {e}",
                    error_context={
                        "question": question[:100],
                        "elapsed_ms": elapsed_ms,
                        "stage": "plan_generation_validation"
                    }
                )
            
            raise ValueError(f"Invalid plan generated: {e}") from e
        
        except TimeoutError as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Plan generation timed out after {elapsed_ms:.1f}ms: {e}")
            
            # Log error to tracker
            if self.tracker:
                self.tracker.log_error(
                    error_type="TimeoutError",
                    error_message=f"LLM request timed out after {elapsed_ms/1000:.1f}s",
                    error_context={
                        "question": question[:100],
                        "elapsed_ms": elapsed_ms,
                        "stage": "plan_generation_llm_call"
                    }
                )
            
            raise RuntimeError(f"LLM request timed out after {elapsed_ms/1000:.1f}s") from e
        
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Plan generation failed after {elapsed_ms:.1f}ms: {e}")
            
            # Log error to tracker
            if self.tracker:
                self.tracker.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    error_context={
                        "question": question[:100],
                        "elapsed_ms": elapsed_ms,
                        "stage": "plan_generation"
                    }
                )
            
            raise RuntimeError(f"Failed to generate plan: {e}") from e
