"""
Evaluation runner - orchestrates the entire evaluation process
"""

import asyncio
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging

from src.evaluation.sampler import StratifiedSampler
from src.evaluation.tracker import MetricsTracker
from src.evaluation.writer import ResultWriter
from src.evaluation.models import EvaluationSummary
from src.agent.workflow_planner import WorkflowPlanner
from src.agent.workflow_executor import WorkflowExecutor
from src.agent.workflow_validator import WorkflowValidator
from src.services.llm_client import get_llm_client
from src.utils.year_context import infer_year_context, extract_metric_context

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Run batch evaluation with comprehensive metrics tracking
    """
    
    def __init__(self, output_dir: Path, enable_validation: bool = True):
        self.output_dir = Path(output_dir)
        self.enable_validation = enable_validation
        self.sampler = StratifiedSampler()
        self.writer = ResultWriter(output_dir)
    
    async def run_evaluation(
        self,
        sample_size: int = 100,
        random_seed: int = 42,
    ) -> EvaluationSummary:
        """
        Main evaluation loop
        
        Args:
            sample_size: Total number of conversations to evaluate
            random_seed: Random seed for reproducibility
            
        Returns:
            EvaluationSummary with aggregate statistics
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting evaluation run {run_id}")
        logger.info(f"Sample size: {sample_size}, Random seed: {random_seed}, Validation: {'ENABLED' if self.enable_validation else 'DISABLED'}")
        
        # Sample conversations
        logger.info("Sampling conversations...")
        conversations = self.sampler.sample_conversations(
            sample_size=sample_size,
            random_seed=random_seed
        )
        
        logger.info(f"Sampled {len(conversations)} conversations")
        
        # Save configuration
        config_path = self.output_dir / "config.json"
        import json
        with open(config_path, "w") as f:
            json.dump({
                "run_id": run_id,
                "sample_size": sample_size,
                "random_seed": random_seed,
                "actual_sample_size": len(conversations),
                "enable_validation": self.enable_validation,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        
        # Process conversations with progress bar
        logger.info("Running agent on conversations...")
        
        for record in tqdm(conversations, desc="Evaluating conversations"):
            try:
                await self._run_conversation(record)
            except Exception as e:
                logger.error(f"Error processing conversation {record.id}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next conversation
        
        # Write all results
        logger.info("Writing results...")
        self.writer.write_summary_csv()
        self.writer.write_turn_level_csv()
        self.writer.write_validation_log()
        self.writer.write_error_log()
        
        # Generate statistics
        logger.info("Generating summary statistics...")
        summary = self.writer.generate_summary_statistics(
            run_id=run_id,
            sample_size=sample_size,
            random_seed=random_seed
        )
        
        logger.info("Evaluation complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return summary
    
    async def _run_conversation(self, record):
        """Run a single conversation and track all metrics"""
        # Initialize agent with tracker for metrics collection
        from src.agent.agent import FinancialAgent
        import os
        
        model_name = os.getenv('MODEL_NAME', 'gpt-4o')
        agent = FinancialAgent(model_name=model_name, enable_validation=self.enable_validation)
        
        try:
            # Run conversation through agent
            result = await agent.run_conversation(record)
            
            # Write trace using agent's tracker
            if agent.tracker:
                trace = agent.tracker.finalize_trace()
                self.writer.write_trace_json(trace)
        except Exception as e:
            logger.error(f"Error processing conversation {record.id}: {e}")
            # Create minimal error trace
            tracker = MetricsTracker()
            trace_id = tracker.start_trace(
                conversation_id=record.id,
                conversation_type=self._get_conversation_type(record),
                num_turns=record.features.num_dialogue_turns,
                features={
                    "has_type2_question": record.features.has_type2_question,
                    "has_duplicate_columns": record.features.has_duplicate_columns,
                    "has_non_numeric_values": record.features.has_non_numeric_values,
                }
            )
            error_context = {
                "conversation_id": record.id,
                "has_type2_question": record.features.has_type2_question,
                "has_non_numeric_values": record.features.has_non_numeric_values,
            }
            tracker.log_error(type(e).__name__, str(e), error_context)
            trace = tracker.finalize_trace()
            self.writer.write_trace_json(trace)
    
    def _get_conversation_type(self, record) -> str:
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

