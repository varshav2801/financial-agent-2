"""
Financial Agent V2 - LangGraph-native implementation
Wraps existing components in a LangGraph workflow for declarative state management
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional, Dict, Any, Literal
import operator
import time

from src.models.dataset import Document, ConvFinQARecord
from src.models.workflow_schema import WorkflowPlan, WorkflowResult
from src.agent.workflow_planner import WorkflowPlanner
from src.agent.workflow_executor import WorkflowExecutor
from src.agent.workflow_validator import WorkflowValidator
from src.agent.result_verifier import ResultVerifier
from src.services.llm_client import get_llm_client
from src.evaluation.tracker import MetricsTracker
from src.config import Config
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# STATE DEFINITION - LangGraph state that flows through the workflow
# ============================================================================

class AgentState(TypedDict):
    """
    State passed between nodes in the LangGraph workflow.

    Uses Annotated with operator.add for automatic accumulation across retries.
    """
    # Input context
    question: str
    document: Document
    previous_answers: Dict[str, Dict]
    expected_answer: Optional[str]
    turn_idx: int

    # Workflow artifacts
    plan: Optional[WorkflowPlan]
    execution_result: Optional[WorkflowResult]

    # Validation & refinement
    validation_critiques: Annotated[list, operator.add]  # Accumulates critiques
    judge_critiques: Annotated[list, operator.add]  # Accumulates judge feedback

    # Retry management
    validation_retry_count: int
    judge_retry_count: int

    # Final output
    final_answer: Optional[float]
    success: bool
    error: Optional[str]

    # Metrics tracking (accumulated across retries)
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    execution_time_ms: float


# ============================================================================
# NODE FUNCTIONS - Each wraps an existing component
# ============================================================================

async def plan_generation_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate a workflow plan using the existing WorkflowPlanner.

    LangGraph Pattern: Nodes receive state and return partial state updates.
    """
    logger.info(f"[PLAN_NODE] Generating plan for question: {state['question'][:50]}...")

    # Initialize planner (reuse existing component)
    planner = WorkflowPlanner(tracker=MetricsTracker())

    # Check if we have validation feedback from previous iteration
    error_feedback = None
    if state.get("validation_critiques"):
        error_feedback = format_validation_feedback(state["validation_critiques"])

    # Generate plan using existing logic
    plan = await planner.create_plan(
        question=state["question"],
        document=state["document"],
        previous_answers=state["previous_answers"],
        error_feedback=error_feedback
    )

    # Extract metrics from planner's tracker
    prompt_tokens = 0
    completion_tokens = 0
    if planner.tracker and planner.tracker.current_trace and planner.tracker.current_trace.turns:
        last_turn = planner.tracker.current_trace.turns[-1]
        prompt_tokens = last_turn.prompt_tokens
        completion_tokens = last_turn.completion_tokens

    return {
        "plan": plan,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


async def validation_node(state: AgentState) -> Dict[str, Any]:
    """
    Validate the generated plan using the existing WorkflowValidator.

    LangGraph Pattern: Validation nodes check invariants and return critiques.
    """
    logger.info("[VALIDATION_NODE] Validating plan structure...")

    validator = WorkflowValidator()
    is_valid, critiques = validator.validate(state["plan"])

    if is_valid:
        logger.info("✓ Plan validation passed")
        return {"validation_critiques": []}  # Empty list signals success
    else:
        logger.warning(f"✗ Plan validation failed with {len(critiques)} issues")
        return {
            "validation_critiques": critiques,
            "validation_retry_count": state.get("validation_retry_count", 0) + 1
        }


async def execution_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute the validated plan using the existing WorkflowExecutor.

    LangGraph Pattern: Execution nodes perform side effects and update state.
    """
    logger.info("[EXECUTION_NODE] Executing workflow plan...")

    executor = WorkflowExecutor(tracker=MetricsTracker())

    result = await executor.execute(
        plan=state["plan"],
        document=state["document"],
        previous_answers=state["previous_answers"],
        current_question=state["question"]
    )

    return {
        "execution_result": result,
        "final_answer": result.final_value if result.success else None,
        "success": result.success,
        "error": result.error if not result.success else None
    }


async def judge_audit_node(state: AgentState) -> Dict[str, Any]:
    """
    Perform semantic audit using the existing ResultVerifier.

    LangGraph Pattern: Judge nodes provide feedback for potential refinement.
    """
    logger.info("[JUDGE_NODE] Auditing execution for semantic grounding...")

    llm_client = get_llm_client()
    judge = ResultVerifier(llm_client)

    audit = await judge.evaluate(
        question=state["question"],
        plan=state["plan"],
        execution_trace={},  # Could pass executor.memory here
        document=state["document"],
        previous_answers=state["previous_answers"]
    )

    if audit.is_valid or audit.confidence_score < 80:
        logger.info(f"✓ Judge audit passed (confidence: {audit.confidence_score}%)")
        return {"judge_critiques": []}
    else:
        logger.warning(f"✗ Judge audit failed (confidence: {audit.confidence_score}%)")
        critiques = judge.convert_to_critiques(audit)
        return {
            "judge_critiques": critiques,
            "judge_retry_count": state.get("judge_retry_count", 0) + 1
        }


# ============================================================================
# CONDITIONAL EDGES - Route state through the graph based on conditions
# ============================================================================

def should_validate(state: AgentState) -> Literal["validate", "execute"]:
    """
    Decide whether to validate the plan or skip to execution.

    LangGraph Pattern: Conditional edges return the name of the next node.
    """
    # Skip validation if disabled globally
    if not Config.ENABLE_PLAN_VALIDATION:
        return "execute"
    return "validate"


def validation_router(state: AgentState) -> Literal["plan", "execute", END]:
    """
    Route based on validation results.

    LangGraph Pattern: Use state to determine routing.
    """
    MAX_VALIDATION_RETRIES = 3

    # Check if validation passed (empty critiques list)
    if not state.get("validation_critiques"):
        return "execute"

    # Check retry limit
    if state.get("validation_retry_count", 0) >= MAX_VALIDATION_RETRIES:
        logger.error("Max validation retries reached, terminating")
        return END

    # Retry with feedback
    return "plan"


def should_judge(state: AgentState) -> Literal["judge", END]:
    """
    Decide whether to audit execution with judge.
    """
    # Skip judge if disabled or execution failed
    if not Config.ENABLE_EXECUTION_JUDGE or not state.get("success"):
        return END

    return "judge"


def judge_router(state: AgentState) -> Literal["plan", END]:
    """
    Route based on judge audit results.
    """
    MAX_JUDGE_RETRIES = 2

    # Check if judge passed (empty critiques)
    if not state.get("judge_critiques"):
        return END

    # Check retry limit
    if state.get("judge_retry_count", 0) >= MAX_JUDGE_RETRIES:
        logger.warning("Max judge retries reached, accepting result")
        return END

    # Retry with judge feedback
    logger.info("Retrying plan generation with judge feedback")
    return "plan"


# ============================================================================
# GRAPH CONSTRUCTION - Build the state machine
# ============================================================================

def create_financial_agent_graph() -> StateGraph:
    """
    Construct the LangGraph workflow for financial question answering.

    Graph Structure:

        START
          ↓
        [plan_generation] ──────────────────┐
          ↓                                  │
        [validation] ← (retry with feedback)┘
          ↓
        [execution]
          ↓
        [judge_audit] ← (retry if needed)
          ↓
        END

    LangGraph Features Used:
    - Conditional edges for dynamic routing
    - State accumulation (validation_critiques, tokens)
    - Retry loops with max attempts
    """
    # Initialize graph with state schema
    workflow = StateGraph(AgentState)

    # Add nodes (each performs one transformation)
    workflow.add_node("plan", plan_generation_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("judge", judge_audit_node)

    # Define entry point
    workflow.set_entry_point("plan")

    # Add edges with routing logic
    workflow.add_conditional_edges(
        "plan",
        should_validate,
        {
            "validate": "validate",
            "execute": "execute"
        }
    )

    workflow.add_conditional_edges(
        "validate",
        validation_router,
        {
            "plan": "plan",      # Retry with feedback
            "execute": "execute", # Validation passed
            END: END             # Max retries exceeded
        }
    )

    workflow.add_conditional_edges(
        "execute",
        should_judge,
        {
            "judge": "judge",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "judge",
        judge_router,
        {
            "plan": "plan",  # Retry with judge feedback
            END: END         # Judge passed or max retries
        }
    )

    return workflow.compile()


# ============================================================================
# AGENT INTERFACE - Main entry point with same interface as FinancialAgent
# ============================================================================

class FinancialAgentV2:
    """
    LangGraph-native implementation of FinancialAgent.

    Key Differences from FinancialAgent:
    - State management handled by LangGraph
    - Retry logic encoded in graph structure
    - Declarative control flow instead of imperative loops
    - Same external interface for compatibility

    Same Interface as FinancialAgent:
    - run_turn() -> TurnResult
    - run_conversation() -> ConversationResult
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

        # Compile the LangGraph workflow
        self.graph = create_financial_agent_graph()
        logger.info("FinancialAgentV2 initialized with LangGraph workflow")

    async def run_turn(
        self,
        question: str,
        document: Document,
        previous_answers: Dict[str, Dict],
        expected_answer: Optional[str] = None,
        turn_idx: int = 0
    ) -> "TurnResult":  # Forward reference to avoid circular import
        """
        Run a single dialogue turn through the LangGraph workflow.

        Same interface as FinancialAgent.run_turn()

        Args:
            question: User's question
            document: Financial document
            previous_answers: Previous Q&A context
            expected_answer: Ground truth answer (optional)
            turn_idx: Turn index (0-based)

        Returns:
            TurnResult with answer and metrics
        """
        start_time = time.time()

        try:
            # Initialize state
            initial_state: AgentState = {
                "question": question,
                "document": document,
                "previous_answers": previous_answers,
                "expected_answer": expected_answer,
                "turn_idx": turn_idx,
                "plan": None,
                "execution_result": None,
                "validation_critiques": [],
                "judge_critiques": [],
                "validation_retry_count": 0,
                "judge_retry_count": 0,
                "final_answer": None,
                "success": False,
                "error": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "execution_time_ms": 0.0
            }

            # Execute graph
            logger.info(f"Starting LangGraph execution for turn {turn_idx}")
            final_state = await self.graph.ainvoke(initial_state)

            execution_time_ms = (time.time() - start_time) * 1000

            # Convert LangGraph state to TurnResult (same format as FinancialAgent)
            return self._convert_to_turn_result(final_state, execution_time_ms)

        except Exception as e:
            logger.error(f"Error in LangGraph execution: {e}")
            execution_time_ms = (time.time() - start_time) * 1000

            # Return error result
            from .agent import TurnResult  # Import here to avoid circular import
            return TurnResult(
                success=False,
                answer=None,
                expected=expected_answer,
                question=question,
                turn_idx=turn_idx,
                execution_time_ms=execution_time_ms,
                error=str(e)
            )

    async def run_conversation(
        self,
        record: ConvFinQARecord,
        model_name: Optional[str] = None
    ) -> "ConversationResult":  # Forward reference
        """
        Run a full multi-turn conversation through LangGraph.

        Same interface as FinancialAgent.run_conversation()

        Args:
            record: ConvFinQA record with questions and document
            model_name: Model to use (overrides instance model)

        Returns:
            ConversationResult with all turns and aggregate metrics
        """
        # Use provided model or instance model
        model = model_name or self.model_name or "unknown"

        from .agent import ConversationResult  # Import here to avoid circular import
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

        previous_answers: Dict[str, Dict] = {}

        for turn_idx, question in enumerate(record.dialogue.conv_questions):
            turn_result = await self.run_turn(
                question=question,
                document=record.doc,
                previous_answers=previous_answers,
                expected_answer=record.dialogue.conv_answers[turn_idx] if turn_idx < len(record.dialogue.conv_answers) else None,
                turn_idx=turn_idx
            )

            result.add_turn(turn_result)

            # Update previous answers for next turn (if successful)
            if turn_result.success and turn_result.answer is not None:
                previous_answers[f"prev_{turn_idx}"] = {
                    "value": turn_result.answer,
                    "entity": getattr(turn_result, 'entity', 'unknown'),
                    "operation": getattr(turn_result, 'operation', 'extraction')
                }

        return result

    def _convert_to_turn_result(self, state: AgentState, execution_time_ms: float) -> "TurnResult":
        """
        Convert LangGraph final state to TurnResult format (compatible with FinancialAgent).
        """
        from .agent import TurnResult  # Import here to avoid circular import

        # Calculate accuracy metrics
        numerical_match = self._calculate_numerical_match(state.get("final_answer"), state.get("expected_answer"))
        financial_match = self._calculate_financial_match(state.get("final_answer"), state.get("expected_answer"))
        soft_match = self._calculate_soft_match(state.get("final_answer"), state.get("expected_answer"))

        return TurnResult(
            success=state.get("success", False),
            answer=state.get("final_answer"),
            expected=state.get("expected_answer"),
            question=state["question"],
            turn_idx=state.get("turn_idx", 0),
            execution_time_ms=execution_time_ms,
            plan=state.get("plan").model_dump() if state.get("plan") else None,
            step_results=state.get("execution_result").model_dump() if state.get("execution_result") else None,
            error=state.get("error"),
            numerical_match=numerical_match,
            financial_match=financial_match,
            soft_match=soft_match,
            prompt_tokens=state.get("prompt_tokens", 0),
            completion_tokens=state.get("completion_tokens", 0),
            total_tokens=state.get("prompt_tokens", 0) + state.get("completion_tokens", 0)
        )

    def _calculate_numerical_match(self, answer: Optional[float], expected: Optional[str]) -> bool:
        """Calculate numerical accuracy (exact match after normalization)"""
        if answer is None or expected is None:
            return False
        try:
            expected_float = float(expected.replace(',', '').replace('$', '').replace('%', ''))
            return abs(answer - expected_float) < 0.01  # Allow small floating point differences
        except (ValueError, AttributeError):
            return False

    def _calculate_financial_match(self, answer: Optional[float], expected: Optional[str]) -> bool:
        """Calculate financial accuracy (handles millions/billions scaling)"""
        if answer is None or expected is None:
            return False
        try:
            expected_float = float(expected.replace(',', '').replace('$', '').replace('%', ''))
            # Check for exact match or scaled match (million vs billion)
            return (
                abs(answer - expected_float) < 0.01 or
                abs(answer - expected_float / 1000) < 0.01 or  # million -> billion
                abs(answer - expected_float * 1000) < 0.01     # billion -> million
            )
        except (ValueError, AttributeError):
            return False

    def _calculate_soft_match(self, answer: Optional[float], expected: Optional[str]) -> bool:
        """Calculate soft match (rounded to reasonable precision)"""
        if answer is None or expected is None:
            return False
        try:
            expected_float = float(expected.replace(',', '').replace('$', '').replace('%', ''))
            # Allow 1% relative difference or 0.1 absolute difference
            return (
                abs(answer - expected_float) / max(abs(expected_float), 1) < 0.01 or
                abs(answer - expected_float) < 0.1
            )
        except (ValueError, AttributeError):
            return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_validation_feedback(critiques: list) -> str:
    """Format validation critiques for planner refinement."""
    if not critiques:
        return ""

    feedback_lines = ["Previous plan had validation errors:"]
    for critique in critiques[-3:]:  # Last 3 critiques to avoid token explosion
        feedback_lines.append(f"- {critique.get('issue_type', 'Unknown')}: {critique.get('reason', 'No reason provided')}")
    return "\n".join(feedback_lines)