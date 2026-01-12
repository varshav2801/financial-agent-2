"""Data loader for ConvFinQA dataset"""
import json
from pathlib import Path
from src.models.dataset import ConvFinQARecord
from src.logger import get_logger

logger = get_logger(__name__)


def load_dataset(split: str = "train") -> list[ConvFinQARecord]:
    """
    Load ConvFinQA dataset.
    
    Args:
        split: Dataset split ('train', 'dev', or 'test')
    
    Returns:
        List of ConvFinQARecord objects
    """
    data_path = Path(__file__).parent.parent.parent / "data" / "convfinqa_dataset.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    logger.info(f"Loading {split} dataset from {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    if split not in data:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(data.keys())}")

    records = [ConvFinQARecord(**record) for record in data[split]]
    logger.info(f"Loaded {len(records)} records from {split} split")

    return records


def load_record(record_id: str) -> ConvFinQARecord:
    """
    Load a specific record by ID.
    
    Args:
        record_id: Record identifier
    
    Returns:
        ConvFinQARecord object
    """
    # Load dataset once to check available splits
    data_path = Path(__file__).parent.parent.parent / "data" / "convfinqa_dataset.json"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    available_splits = list(data.keys())
    logger.info(f"Searching for record {record_id} in splits: {available_splits}")
    
    # Try all available splits
    for split in available_splits:
        records = load_dataset(split)
        for record in records:
            if record.id == record_id:
                logger.info(f"Found record {record_id} in {split} split")
                return record

    raise ValueError(f"Record with ID '{record_id}' not found in any split ({', '.join(available_splits)})")
