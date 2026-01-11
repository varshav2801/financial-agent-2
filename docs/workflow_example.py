"""Example runner for workflow-based financial agent"""
import asyncio
from src.models.dataset import Document
from src.agent.workflow_planner import WorkflowPlanner
from src.agent.workflow_executor import WorkflowExecutor
from src.agent.workflow_validator import WorkflowValidator
from src.logger import get_logger

logger = get_logger(__name__)


async def run_workflow_example():
    """
    Example demonstrating workflow-based execution.
    
    This shows how to:
    1. Create a WorkflowPlanner
    2. Generate a WorkflowPlan
    3. Validate the plan
    4. Execute the plan with WorkflowExecutor
    """
    
    # Sample document
    document = Document(
        id="example_001",
        pre_text="Company X operates in technology sector.",
        post_text="The company acquired 30 towers for $6.0 million in 2005.",
        table={
            "2014": {
                "revenue": 145.2,
                "cost": 85.3,
                "net income": 59.9,
                "assets": 320.5
            },
            "2013": {
                "revenue": 132.8,
                "cost": 78.1,
                "net income": 54.7,
                "assets": 298.3
            },
            "2012": {
                "revenue": 118.5,
                "cost": 71.2,
                "net income": 47.3,
                "assets": 275.1
            }
        },
        questions=[]
    )
    
    # Question to answer
    question = "What was the change in revenue from 2013 to 2014?"
    
    logger.info(f"Question: {question}")
    logger.info("="*60)
    
    # Step 1: Create planner
    planner = WorkflowPlanner()
    logger.info("Created WorkflowPlanner")
    
    # Step 2: Generate plan
    logger.info("Generating plan...")
    plan = await planner.create_plan(
        question=question,
        document=document,
        previous_answers={},
        conversation_history=None
    )
    
    logger.info(f"Plan generated with {len(plan.steps)} steps")
    logger.info(f"Thought process: {plan.thought_process}")
    
    # Print plan steps
    for i, step in enumerate(plan.steps, 1):
        logger.info(f"Step {i}: {step.tool} - {step.model_dump_json(indent=2)}")
    
    # Step 3: Validate plan
    validator = WorkflowValidator()
    validation_result = validator.validate(plan)
    
    if not validation_result.is_valid:
        logger.error("Plan validation failed!")
        for issue in validation_result.issues:
            logger.error(f"  - {issue}")
        return
    
    logger.info("Plan validation passed")
    
    # Step 4: Execute plan
    executor = WorkflowExecutor()
    logger.info("Executing plan...")
    
    result = await executor.execute(
        plan=plan,
        document=document,
        previous_answers={},
        current_question=question
    )
    
    # Display results
    logger.info("="*60)
    if result.success:
        logger.info(f"SUCCESS: Final answer = {result.final_value}")
        logger.info(f"Execution time: {result.execution_time_ms:.1f}ms")
        logger.info("Step results:")
        for step_id, value in result.step_results.items():
            logger.info(f"  Step {step_id}: {value}")
    else:
        logger.error(f"FAILED: {result.error}")
        logger.info(f"Partial results: {result.step_results}")


async def run_multi_turn_example():
    """Example of multi-turn conversation using workflow system"""
    
    document = Document(
        id="example_002",
        pre_text="Annual financial report.",
        post_text="",
        table={
            "2014": {"revenue": 145.2, "cost": 85.3},
            "2013": {"revenue": 132.8, "cost": 78.1}
        },
        questions=[]
    )
    
    planner = WorkflowPlanner()
    executor = WorkflowExecutor()
    previous_answers = {}
    
    # Turn 1
    question1 = "What is revenue in 2014?"
    logger.info(f"Turn 1: {question1}")
    
    plan1 = await planner.create_plan(question1, document, previous_answers)
    result1 = await executor.execute(plan1, document, previous_answers, question1)
    
    logger.info(f"Answer 1: {result1.final_value}")
    previous_answers["prev_1"] = result1.final_value
    
    # Turn 2
    question2 = "What is revenue in 2013?"
    logger.info(f"Turn 2: {question2}")
    
    # Update previous answers (shift indices)
    previous_answers_turn2 = {"prev_0": result1.final_value}
    
    plan2 = await planner.create_plan(question2, document, previous_answers_turn2)
    result2 = await executor.execute(plan2, document, previous_answers_turn2, question2)
    
    logger.info(f"Answer 2: {result2.final_value}")
    
    # Turn 3 - reference previous answers
    question3 = "What is the difference?"
    logger.info(f"Turn 3: {question3}")
    
    # Update previous answers
    previous_answers_turn3 = {
        "prev_0": result2.final_value,  # Most recent
        "prev_1": result1.final_value   # Second most recent
    }
    
    plan3 = await planner.create_plan(question3, document, previous_answers_turn3)
    result3 = await executor.execute(plan3, document, previous_answers_turn3, question3)
    
    logger.info(f"Answer 3: {result3.final_value}")
    logger.info("="*60)
    logger.info("Multi-turn conversation completed successfully!")


def main():
    """Main entry point"""
    logger.info("Starting Workflow-based Financial Agent Examples")
    logger.info("="*60)
    
    # Run simple example
    asyncio.run(run_workflow_example())
    
    logger.info("\n" + "="*60)
    logger.info("Multi-turn Example")
    logger.info("="*60)
    
    # Run multi-turn example
    asyncio.run(run_multi_turn_example())


if __name__ == "__main__":
    main()
