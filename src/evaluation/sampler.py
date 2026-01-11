"""
Stratified sampling for balanced evaluation
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from src.models.dataset import ConvFinQARecord
import logging

logger = logging.getLogger(__name__)


class StratifiedSampler:
    """
    Sample conversations with equal representation across conversation types
    """
    
    def __init__(self, dataset_path: str | None = None):
        if dataset_path is None:
            # Default to project root / data / convfinqa_dataset.json
            project_root = Path(__file__).parent.parent.parent
            dataset_path = project_root / "data" / "convfinqa_dataset.json"
        self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
    
    def sample_conversations(
        self,
        sample_size: int,
        random_seed: int = 42
    ) -> list[ConvFinQARecord]:
        """
        Sample conversations with stratification by:
        - Number of dialogue turns
        - has_type2_question
        
        Args:
            sample_size: Total number of conversations to sample
            random_seed: Random seed for reproducibility
            
        Returns:
            List of ConvFinQARecord objects
        """
        random.seed(random_seed)
        
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Load and group dataset
        strata = self._group_by_strata()
        
        if not strata:
            raise ValueError(f"No conversations found in dataset at {self.dataset_path}")
        
        logger.info(f"Found {len(strata)} conversation types:")
        for key, records in strata.items():
            logger.info(f"  - {key}: {len(records)} conversations")
        
        # Calculate samples per stratum
        num_strata = len(strata)
        samples_per_stratum = sample_size // num_strata
        remainder = sample_size % num_strata
        
        logger.info(f"Sampling {samples_per_stratum} conversations from each stratum")
        
        # Sample from each stratum
        sampled = []
        for idx, (stratum_key, records) in enumerate(sorted(strata.items())):
            # Add one extra sample to some strata to handle remainder
            n_samples = samples_per_stratum + (1 if idx < remainder else 0)
            
            # Random sample from this stratum
            if len(records) < n_samples:
                logger.warning(f"Stratum '{stratum_key}' has only {len(records)} records, "
                             f"requested {n_samples}. Taking all.")
                stratum_sample = records
            else:
                stratum_sample = random.sample(records, n_samples)
            
            sampled.extend(stratum_sample)
            logger.info(f"  Sampled {len(stratum_sample)} from '{stratum_key}'")
        
        # Shuffle final sample to mix strata
        random.shuffle(sampled)
        
        logger.info(f"Total sampled: {len(sampled)} conversations")
        return sampled
    
    def _group_by_strata(self) -> dict[str, list[ConvFinQARecord]]:
        """
        Load dataset and group by conversation characteristics
        
        Strata keys format: "{num_turns}turn_type2{has_type2}"
        Example: "2turn_type2True", "4turn_type2False"
        """
        strata = defaultdict(list)
        valid_records = 0
        errors = defaultdict(int)
        
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Load entire JSON file (structured format with "train" key)
        with open(self.dataset_path, "r") as f:
            dataset = json.load(f)
        
        # Access the train split
        train_records = dataset.get("train", [])
        logger.info(f"Found {len(train_records)} records in 'train' split")
        
        # Process each record
        for idx, data in enumerate(train_records):
            try:
                record = ConvFinQARecord(**data)
                
                # Create stratum key
                num_turns = record.features.num_dialogue_turns
                has_type2 = record.features.has_type2_question
                
                # Group conversations by turn count (2, 3, 4+)
                if num_turns <= 2:
                    turn_group = "2turn"
                elif num_turns == 3:
                    turn_group = "3turn"
                else:
                    turn_group = "4+turn"
                
                stratum_key = f"{turn_group}_type2{has_type2}"
                strata[stratum_key].append(record)
                valid_records += 1
                
            except Exception as e:
                error_type = type(e).__name__
                errors[error_type] += 1
                if errors[error_type] <= 3:  # Log first 3 of each error type
                    logger.warning(f"Record {idx}: {error_type}: {e}")
        
        logger.info(f"Dataset loading complete: {valid_records}/{len(train_records)} valid records")
        if errors:
            logger.warning(f"Errors encountered: {dict(errors)}")
        
        return dict(strata)
    
    def get_dataset_statistics(self) -> dict:
        """Get statistics about the dataset distribution"""
        strata = self._group_by_strata()
        
        total = sum(len(records) for records in strata.values())
        
        stats = {
            "total_conversations": total,
            "num_strata": len(strata),
            "strata_distribution": {
                key: {
                    "count": len(records),
                    "percentage": len(records) / total * 100
                }
                for key, records in sorted(strata.items())
            }
        }
        
        return stats

