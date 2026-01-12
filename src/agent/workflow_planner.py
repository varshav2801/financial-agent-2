"""Workflow planner for generating WorkflowPlan using structured outputs"""
import time
from typing import TYPE_CHECKING, Optional
from pydantic import ValidationError
from src.models.workflow_schema import WorkflowPlan, StepCritique
from src.models.dataset import Document
from src.prompts.workflow_planner import (
    WORKFLOW_PLANNER_SYSTEM_PROMPT,
    WORKFLOW_PLANNER_USER_TEMPLATE
)
from src.services.llm_client import get_llm_client
from src.utils.table_normalizer import normalize_table
from src.config import Config
from src.logger import get_logger

if TYPE_CHECKING:
    from src.evaluation.tracker import MetricsTracker
    from src.agent.workflow_validator import WorkflowValidator

logger = get_logger(__name__)


class WorkflowPlanner:
    """
    Generate WorkflowPlan using LLM structured outputs.
    
    Key differences from legacy Planner:
    - Generates WorkflowPlan (not ExecutionPlan)
    - Uses comprehensive prompt with 9 critical patterns
    - Register pattern references
    """
    
    def __init__(
        self, 
        llm_client=None, 
        tracker: "MetricsTracker | None" = None,
        validator: "WorkflowValidator | None" = None,
        enable_validation: bool = True
    ) -> None:
        self.llm_client = llm_client if llm_client is not None else get_llm_client()
        self.tracker = tracker
        self.validator = validator
        self.enable_validation = enable_validation and Config.ENABLE_PLAN_VALIDATION
    
    async def create_plan(
        self,
        question: str,
        document: Document,
        previous_answers: dict[str, float | dict],
        conversation_history: list[dict] | None = None,
        error_feedback: Optional[str] = None
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
        
        # Construct base user message first
        base_user_message = WORKFLOW_PLANNER_USER_TEMPLATE.format(
            question=question,
            table_years=table_years,
            table_metrics=table_metrics,
            pre_text=pre_text,
            post_text=post_text,
            previous_answers=prev_answers_str
        )
        
        # Inject refinement block before final instruction if validation failed
        if error_feedback:
            refinement_block = f"""====================
REFINEMENT REQUIRED: FIX PREVIOUS ERRORS
====================
Your previous plan was INVALID. You MUST correct the following issues:

{error_feedback}

Ensure the new plan resolves these specific failures while maintaining logical soundness.
====================

"""
            logger.warning(f"Plan refinement requested due to validation errors")
            # Inject before the final "Generate a WorkflowPlan" instruction
            final_instruction = "Generate a WorkflowPlan to answer this question."
            user_message = base_user_message.replace(
                final_instruction,
                refinement_block + final_instruction
            )
        else:
            user_message = base_user_message
        
        logger.info(f"Generating workflow plan for question: {question}" + 
                   (" (refinement attempt)" if error_feedback else ""))
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
            
            # Validate response structure
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Invalid LLM response: no choices returned")
            
            if not response.choices[0].message:
                raise ValueError("Invalid LLM response: message is None")
            
            if not hasattr(response.choices[0].message, 'parsed') or response.choices[0].message.parsed is None:
                raise ValueError("Invalid LLM response: parsed content is None")
            
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
    
    async def create_plan_with_validation(
        self,
        question: str,
        document: Document,
        previous_answers: dict[str, float | dict],
        conversation_history: list[dict] | None = None,
        max_retries: Optional[int] = None
    ) -> WorkflowPlan:
        """
        Create workflow plan with validation and refinement loop.
        
        If validation fails, sends critiques back to LLM for refinement.
        Retries up to max_retries times before giving up.
        
        Args:
            question: User question to answer
            document: Document with table and text
            previous_answers: Previous turn answers for multi-turn context
            conversation_history: Optional conversation history
            max_retries: Maximum refinement attempts (defaults to Config.MAX_PLAN_REFINEMENT_RETRIES)
            
        Returns:
            Valid WorkflowPlan
            
        Raises:
            RuntimeError: If unable to generate valid plan after max retries
        """
        if not self.enable_validation or not self.validator:
            # Validation disabled, return plan directly
            return await self.create_plan(question, document, previous_answers, conversation_history)
        
        max_retries = max_retries or Config.MAX_PLAN_REFINEMENT_RETRIES
        error_feedback = None
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Generate plan (with optional error feedback from previous attempt)
                plan = await self.create_plan(
                    question=question,
                    document=document,
                    previous_answers=previous_answers,
                    conversation_history=conversation_history,
                    error_feedback=error_feedback
                )
                
                # Validate the plan
                validation_result = self.validator.validate(plan)
                
                if validation_result.is_valid:
                    if attempt > 0:
                        logger.info(f"✓ Plan validation passed after {attempt} refinement(s)")
                    else:
                        logger.info("✓ Plan validation passed on first attempt")
                    return plan
                
                # Validation failed - prepare feedback for next iteration
                logger.warning(f"✗ Plan validation failed (attempt {attempt + 1}/{max_retries + 1}): "
                              f"{len(validation_result.critiques)} issues found")
                
                # Format critiques into feedback message
                error_feedback = self._format_validation_feedback(validation_result.critiques)
                
                # Log validation failure to tracker
                if self.tracker:
                    self.tracker.log_error(
                        error_type="PlanValidationError",
                        error_message=f"Plan validation failed with {len(validation_result.critiques)} issues",
                        error_context={
                            "question": question[:100],
                            "attempt": attempt + 1,
                            "failed_steps": validation_result.failed_steps,
                            "failure_type": validation_result.failure_type,
                            "critique_count": len(validation_result.critiques)
                        }
                    )
                
                if attempt >= max_retries:
                    # Max retries reached
                    logger.error(f"✗ Failed to generate valid plan after {max_retries + 1} attempts")
                    raise RuntimeError(
                        f"Unable to generate valid plan after {max_retries + 1} attempts. "
                        f"Last validation issues: {validation_result.issues[:3]}"
                    )
                
                # Continue to next attempt
                logger.info(f"→ Attempting refinement {attempt + 1}/{max_retries}...")
                
            except (ValidationError, ValueError) as e:
                # Pydantic validation error (malformed response)
                logger.error(f"Plan generation error (attempt {attempt + 1}): {e}")
                
                if attempt >= max_retries:
                    raise RuntimeError(f"Unable to generate valid plan after {max_retries + 1} attempts: {e}") from e
                
                # Provide generic feedback for malformed responses
                error_feedback = f"""Your previous response failed to parse correctly:
{str(e)[:300]}

Please ensure your response follows the exact WorkflowPlan schema format."""
                
                continue
        
        # Should never reach here
        raise RuntimeError("Unexpected error in plan validation loop")
    
    def _format_validation_feedback(self, critiques: list[StepCritique]) -> str:
        """
        Format validation critiques as strict corrective constraints.
        
        Uses the Iterative Prompt-Refinement framework's format for maximum correction effectiveness.
        
        Args:
            critiques: List of StepCritique objects
            
        Returns:
            Formatted feedback string with critique details
        """
        feedback_lines = []
        
        # Group critiques by step
        plan_level = [c for c in critiques if c.step_id is None]
        step_level = [c for c in critiques if c.step_id is not None]
        
        if plan_level:
            feedback_lines.append("[PLAN-LEVEL CRITIQUES]")
            for critique in plan_level:
                feedback_lines.append(f"- Issue: {critique.issue_type}")
                feedback_lines.append(f"  Reason: {critique.reason}")
                feedback_lines.append(f"  Fix: {critique.fix_suggestion}")
                feedback_lines.append("")
        
        if step_level:
            feedback_lines.append("[STEP-LEVEL CRITIQUES]")
            for critique in step_level:
                feedback_lines.append(f"- Issue: {critique.issue_type} at Step {critique.step_id}")
                feedback_lines.append(f"  Reason: {critique.reason}")
                feedback_lines.append(f"  Fix: {critique.fix_suggestion}")
                feedback_lines.append("")
        
        return "\n".join(feedback_lines)
