"""
Financial Agent - Main interface for the ConvFinQA agent
Wraps planner, executor, and validator into a single interface
"""

import asyncio
import time
from typing import Dict, Any, Optional
from src.models.dataset import Document, ConvFinQARecord
from src.agent.workflow_planner import WorkflowPlanner
from src.agent.workflow_executor import WorkflowExecutor
from src.agent.workflow_validator import WorkflowValidator
from src.agent.result_verifier import ResultVerifier
from src.services.llm_client import get_llm_client
from src.evaluation.tracker import MetricsTracker
from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


class TurnResult:
    """Result from a single dialogue turn"""
    def __init__(
        self,
        success: bool,
        answer: Optional[float | str],
        expected: Optional[str],
        question: str,
        turn_idx: int,
        execution_time_ms: float,
        plan: Optional[Dict] = None,
        step_results: Optional[Dict] = None,
        error: Optional[str] = None,
        # Accuracy metrics
        numerical_match: bool = False,
        financial_match: bool = False,
        soft_match: bool = False,
        # Token metrics
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ):
        self.success = success
        self.answer = answer
        self.expected = expected
        self.question = question
        self.turn_idx = turn_idx
        self.execution_time_ms = execution_time_ms
        self.plan = plan
        self.step_results = step_results
        self.error = error
        self.numerical_match = numerical_match
        self.financial_match = financial_match
        self.soft_match = soft_match
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
    
    @property
    def numerical_accuracy(self) -> float:
        """Numerical accuracy (0.0 or 1.0)"""
        return 1.0 if self.numerical_match else 0.0
    
    @property
    def financial_accuracy(self) -> float:
        """Financial accuracy (0.0 or 1.0)"""
        return 1.0 if self.financial_match else 0.0
    
    @property
    def soft_match_accuracy(self) -> float:
        """Soft match accuracy (0.0 or 1.0)"""
        return 1.0 if self.soft_match else 0.0
    
    @property
    def response_time_ms(self) -> float:
        """Response time in milliseconds (alias for execution_time_ms)"""
        return self.execution_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'success': self.success,
            'answer': self.answer,
            'expected': self.expected,
            'question': self.question,
            'turn': self.turn_idx + 1,
            'execution_time_ms': self.execution_time_ms,
            'plan': self.plan,
            'step_results': self.step_results,
            'error': self.error,
            'numerical_match': self.numerical_match,
            'financial_match': self.financial_match,
            'soft_match': self.soft_match,
            'tokens': {
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'total_tokens': self.total_tokens
            }
        }


class ConversationResult:
    """Result from a multi-turn conversation"""
    def __init__(self, record_id: str, model: str, features: Dict):
        self.record_id = record_id
        self.model = model
        self.features = features
        self.turns: list[TurnResult] = []
    
    def add_turn(self, turn: TurnResult):
        """Add a turn result"""
        self.turns.append(turn)
    
    @property
    def turn_results(self) -> list[TurnResult]:
        """Alias for turns (compatibility)"""
        return self.turns
    
    @property
    def numerical_accuracy(self) -> float:
        """Numerical accuracy across all turns"""
        if not self.turns:
            return 0.0
        return sum(1 for t in self.turns if t.numerical_match) / len(self.turns)
    
    @property
    def financial_accuracy(self) -> float:
        """Financial accuracy across all turns"""
        if not self.turns:
            return 0.0
        return sum(1 for t in self.turns if t.financial_match) / len(self.turns)
    
    @property
    def soft_match_accuracy(self) -> float:
        """Soft match accuracy across all turns"""
        if not self.turns:
            return 0.0
        return sum(1 for t in self.turns if t.soft_match) / len(self.turns)
    
    @property
    def all_correct(self) -> bool:
        """Whether all turns were correct (financial accuracy)"""
        if not self.turns:
            return False
        return all(t.financial_match for t in self.turns)
    
    @property
    def correct_turns(self) -> int:
        """Number of correct turns (financial accuracy)"""
        return sum(1 for t in self.turns if t.financial_match)
    
    @property
    def total_turns(self) -> int:
        """Total number of turns"""
        return len(self.turns)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used across all turns"""
        return sum(t.total_tokens for t in self.turns)
    
    @property
    def avg_tokens_per_turn(self) -> float:
        """Average tokens per turn"""
        if not self.turns:
            return 0.0
        return self.total_tokens / len(self.turns)
    
    @property
    def total_response_time_ms(self) -> float:
        """Total response time in milliseconds"""
        return sum(t.execution_time_ms for t in self.turns)
    
    @property
    def avg_response_time_ms(self) -> float:
        """Average response time per turn in milliseconds"""
        if not self.turns:
            return 0.0
        return self.total_response_time_ms / len(self.turns)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with aggregate metrics"""
        num_turns = len(self.turns)
        
        # Count correct by accuracy type
        numerical_correct = sum(1 for t in self.turns if t.numerical_match)
        financial_correct = sum(1 for t in self.turns if t.financial_match)
        soft_match_correct = sum(1 for t in self.turns if t.soft_match)
        
        # Aggregate tokens and timing
        total_tokens = sum(t.total_tokens for t in self.turns)
        total_prompt_tokens = sum(t.prompt_tokens for t in self.turns)
        total_completion_tokens = sum(t.completion_tokens for t in self.turns)
        total_response_time_ms = sum(t.execution_time_ms for t in self.turns)
        
        return {
            'example_id': self.record_id,
            'model': self.model,
            'features': self.features,
            'num_turns': num_turns,
            'turns': [t.to_dict() for t in self.turns],
            
            # Accuracy metrics
            'numerical_accuracy': numerical_correct / num_turns if num_turns > 0 else 0,
            'financial_accuracy': financial_correct / num_turns if num_turns > 0 else 0,
            'soft_match_accuracy': soft_match_correct / num_turns if num_turns > 0 else 0,
            
            # Correct turns (x out of y)
            'correct_turns': f"{financial_correct}/{num_turns}",
            'numerical_correct_turns': f"{numerical_correct}/{num_turns}",
            'soft_match_correct_turns': f"{soft_match_correct}/{num_turns}",
            
            # Token usage
            'total_tokens': total_tokens,
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'avg_tokens_per_turn': total_tokens / num_turns if num_turns > 0 else 0,
            
            # Response time
            'total_response_time_ms': total_response_time_ms,
            'avg_response_time_ms': total_response_time_ms / num_turns if num_turns > 0 else 0,
            
            # Legacy compatibility
            'conversation_accuracy': financial_correct / num_turns if num_turns > 0 else 0,
            'all_correct': financial_correct == num_turns
        }


class FinancialAgent:
    """
    Main agent interface for ConvFinQA
    
    Provides a clean interface to:
    - Run single turns
    - Run full conversations
    - Track metrics automatically
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        enable_validation: bool = False,
        enable_judge: bool = False
    ):
        """
        Initialize the agent
        
        Args:
            model_name: Model to use (e.g., 'gpt-4o', 'gpt-4o-mini')
            enable_validation: Whether to enable plan validation
            enable_judge: Whether to enable post-execution judge audit
        """
        self.model_name = model_name
        self.enable_validation = enable_validation
        self.enable_judge = enable_judge and Config.ENABLE_EXECUTION_JUDGE
        
        # Initialize components (lazy initialization)
        self._llm_client = None
        self._planner = None
        self._executor = None
        self._validator = None
        self._judge = None
        self._tracker = None
    
    @property
    def tracker(self) -> Optional[MetricsTracker]:
        """Expose tracker for external access"""
        return self._tracker
    
    def _initialize_components(self):
        """Lazy initialization of components"""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
            self._tracker = MetricsTracker()
            
            # Initialize validator first if validation is enabled
            if self.enable_validation:
                self._validator = WorkflowValidator()
            else:
                self._validator = None
            
            # Initialize judge if enabled
            if self.enable_judge:
                self._judge = ResultVerifier(self._llm_client)
                logger.info("Execution judge enabled (post-execution audit)")
            else:
                self._judge = None
            
            # Pass validator to planner
            self._planner = WorkflowPlanner(
                self._llm_client, 
                tracker=self._tracker,
                validator=self._validator,
                enable_validation=self.enable_validation
            )
            self._executor = WorkflowExecutor(tracker=self._tracker)
    
    async def run_turn(
        self,
        question: str,
        document: Document,
        previous_answers: Dict[str, Dict],
        expected_answer: Optional[str] = None,
        turn_idx: int = 0
    ) -> TurnResult:
        """
        Run a single dialogue turn with optional judge-based refinement.
        
        Flow:
        1. Generate plan (with optional validation)
        2. Execute plan
        3. If judge enabled: Audit execution for grounding errors
        4. If audit fails: Retry with judge critiques (max retries)
        5. Return final result
        
        Args:
            question: User's question
            document: Financial document
            previous_answers: Previous Q&A context
            expected_answer: Ground truth answer (optional)
            turn_idx: Turn index (0-based)
            
        Returns:
            TurnResult with answer and metrics
        """
        self._initialize_components()
        
        start_time = time.time()
        
        try:
            # Start turn tracking
            self._tracker.start_turn(turn_idx, question, expected_answer or "", ground_truth_program=None)
            
            # Retry loop for judge-based refinement
            max_retries = Config.MAX_JUDGE_REFINEMENT_RETRIES if self.enable_judge else 1
            judge_critiques = None
            
            for attempt in range(max_retries):
                # Generate plan with validation and refinement if enabled
                if self.enable_validation:
                    plan = await self._planner.create_plan_with_validation(
                        question=question,
                        document=document,
                        previous_answers=previous_answers,
                        error_feedback=self._planner._format_validation_feedback(judge_critiques) if judge_critiques else None
                    )
                else:
                    plan = await self._planner.create_plan(
                        question=question,
                        document=document,
                        previous_answers=previous_answers,
                        error_feedback=self._planner._format_validation_feedback(judge_critiques) if judge_critiques else None
                    )
                
                # Log plan for tracking
                self._tracker.log_plan(
                    num_steps=len(plan.steps),
                    complexity=0.0,
                    plan_type="workflow",
                    plan_object=plan
                )
                
                # Execute plan
                result = await self._executor.execute(
                    plan=plan,
                    document=document,
                    previous_answers=previous_answers,
                    current_question=question
                )
                
                answer_value = result.final_value if result.success else None
                
                # Judge audit if enabled and execution succeeded
                if self.enable_judge and result.success and answer_value is not None:
                    logger.info(f"Judge auditing execution (attempt {attempt + 1}/{max_retries})")
                    
                    audit = await self._judge.evaluate(
                        question=question,
                        plan=plan,
                        execution_trace=self._executor.memory,
                        document=document,
                        previous_answers=previous_answers
                    )
                    
                    # Use judge's should_retry method to check if retry is needed
                    if not self._judge.should_retry(audit, Config.JUDGE_CONFIDENCE_THRESHOLD):
                        if audit.is_valid:
                            logger.info("✓ Judge audit passed: Execution is semantically grounded")
                        else:
                            logger.info(
                                f"Judge found issues but confidence below threshold "
                                f"({audit.confidence_score}% < {Config.JUDGE_CONFIDENCE_THRESHOLD}%), accepting result"
                            )
                        break  # Accept result
                    
                    # Judge says retry with high confidence - retry if attempts remain
                    if attempt < max_retries - 1:
                        judge_critiques = self._judge.convert_to_critiques(audit)
                        logger.warning(
                            f"Judge audit failed with high confidence ({audit.confidence_score}%), "
                            f"retrying with critiques (attempt {attempt + 2}/{max_retries})"
                        )
                        continue  # Retry with judge feedback
                    else:
                        logger.warning(
                            f"Judge audit failed but max retries reached ({max_retries}), "
                            "returning final result anyway"
                        )
                        break
                else:
                    # No judge or execution failed - exit retry loop
                    break
            
            # Finalize turn to calculate accuracies
            self._tracker.finalize_turn(answer_value)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Get metrics from tracker - turn is now in the trace
            turn_metrics = None
            if self._tracker.current_trace and self._tracker.current_trace.turns:
                turn_metrics = self._tracker.current_trace.turns[-1]  # Get last turn added
            
            return TurnResult(
                success=result.success,
                answer=answer_value,
                expected=expected_answer,
                question=question,
                turn_idx=turn_idx,
                execution_time_ms=execution_time,
                plan=plan.model_dump(),
                step_results=result.step_results,
                error=result.error if not result.success else None,
                numerical_match=turn_metrics.numerical_match if turn_metrics else False,
                financial_match=turn_metrics.financial_match if turn_metrics else False,
                soft_match=turn_metrics.soft_match if turn_metrics else False,
                prompt_tokens=turn_metrics.prompt_tokens if turn_metrics else 0,
                completion_tokens=turn_metrics.completion_tokens if turn_metrics else 0,
                total_tokens=turn_metrics.turn_tokens if turn_metrics else 0,
            )
            
        except Exception as e:
            # Log error in tracker
            error_context = {
                "turn_idx": turn_idx,
                "question": question
            }
            self._tracker.log_error(type(e).__name__, str(e), error_context)
            self._tracker.finalize_turn(None)
            
            execution_time = (time.time() - start_time) * 1000
            
            return TurnResult(
                success=False,
                answer=None,
                expected=expected_answer,
                question=question,
                turn_idx=turn_idx,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    async def run_conversation(
        self,
        record: ConvFinQARecord,
        model_name: Optional[str] = None
    ) -> ConversationResult:
        """
        Run a full multi-turn conversation
        
        Args:
            record: ConvFinQA record with questions and document
            model_name: Model to use (overrides instance model)
            
        Returns:
            ConversationResult with all turns and aggregate metrics
        """
        # Initialize components first
        self._initialize_components()
        
        # Use provided model or instance model
        model = model_name or self.model_name or "unknown"
        
        result = ConversationResult(
            record_id=record.id,
            model=model,
            features={
                'num_dialogue_turns': record.features.num_dialogue_turns,
                'has_type2_question': record.features.has_type2_question,
                'has_duplicate_columns': record.features.has_duplicate_columns,
                'has_non_numeric_values': record.features.has_non_numeric_values
            }
        )
        
        # Start trace for this conversation
        conversation_type = self._get_conversation_type(record)
        self._tracker.start_trace(
            conversation_id=record.id,
            conversation_type=conversation_type,
            num_turns=record.features.num_dialogue_turns,
            features={
                'has_type2_question': record.features.has_type2_question,
                'has_duplicate_columns': record.features.has_duplicate_columns,
                'has_non_numeric_values': record.features.has_non_numeric_values
            }
        )
        
        previous_answers: Dict[str, Dict] = {}
        
        for turn_idx, question in enumerate(record.dialogue.conv_questions):
            expected = record.dialogue.conv_answers[turn_idx]
            
            # Run turn
            turn_result = await self.run_turn(
                question=question,
                document=record.doc,
                previous_answers=previous_answers,
                expected_answer=expected,
                turn_idx=turn_idx
            )
            
            result.add_turn(turn_result)
            
            # Update previous answers (match main flow pattern)
            if turn_result.success and turn_result.answer is not None:
                # Extract entity and operation from plan for tracking
                # CRITICAL: Track LAST extraction entity (most recent context)
                # and FINAL operation (what the turn computed)
                entity = "unknown"
                operation = "unknown"
                if turn_result.plan:
                    steps = turn_result.plan.get('steps', [])
                    
                    # Track the LAST extraction entity (most recent in plan)
                    for step in steps:
                        if step.get('tool') == 'extract_value' and step.get('source') == 'table':
                            table_params = step.get('table_params', {})
                            extracted_entity = table_params.get('row_query', 'unknown')
                            logger.debug(f"Turn {turn_idx}, Step {step.get('step_id')}: Found extraction entity='{extracted_entity}'")
                            entity = extracted_entity
                            # Don't break - keep going to find the last extraction
                    
                    # Track the FINAL operation (last step's operation)
                    if steps:
                        last_step = steps[-1]
                        if last_step.get('tool') == 'compute':
                            operation = last_step.get('operation', 'unknown')
                        elif last_step.get('tool') == 'extract_value':
                            operation = 'extraction'
                    
                    logger.info(f"Turn {turn_idx} final metadata: entity='{entity}', operation='{operation}'")
                
                previous_answers[f"prev_{turn_idx}"] = {
                    "value": turn_result.answer,
                    "entity": entity,
                    "operation": operation,
                    "question": question[:60] + "..." if len(question) > 60 else question
                }
            
            # Rate limiting between turns
            if turn_idx < len(record.dialogue.conv_questions) - 1:
                await asyncio.sleep(1)
        
        return result
    
    def _get_conversation_type(self, record: ConvFinQARecord) -> str:
        """Get conversation type string for stratification"""
        num_turns = record.features.num_dialogue_turns
        has_type2 = record.features.has_type2_question
        
        if num_turns <= 2:
            turn_group = "2turn"
        elif num_turns == 3:
            turn_group = "3turn"
        else:
            turn_group = "4+turn"
        
        return f"{turn_group}_type2{has_type2}"
