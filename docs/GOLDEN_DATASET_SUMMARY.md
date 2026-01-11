# Golden Dataset - Quick Summary

## What Was Created

### 1. Golden Dataset (`data/golden_dataset.json`)
- **12 carefully balanced examples** from ConvFinQA dataset
- Covers diverse dialogue complexities and question types
- Perfect for model comparison and evaluation

### 2. Evaluation Script (`evaluate_golden_dataset.py`)
- Automated testing across 4 state-of-the-art models
- Comprehensive metrics and comparison reports
- Production-ready with progress bars and error handling

## Dataset Balance

| Category | Count | Features |
|----------|-------|----------|
| **1-2 turns** | 4 | 1 of each feature combination |
| **3-4 turns** | 4 | 1 of each feature combination |
| **5+ turns** | 4 | 1 of each feature combination |

**Feature Combinations (each turn category has 1):**
- ✓ Simple questions, numeric only
- ✓ Simple questions, with text
- ✓ Multi-hop reasoning, numeric only  
- ✓ Multi-hop reasoning, with text

## Quick Start

```bash
# 1. Verify golden dataset exists
ls -lh version3/data/golden_dataset.json

# 2. Run evaluation on all 4 models
cd version3
python evaluate_golden_dataset.py

# 3. View results
cat golden_eval_results/comparison_report.json
```

## Models Tested

1. **gpt-4o** - OpenAI flagship
2. **gpt-4o-mini** - OpenAI efficient  
3. **o3-mini** - OpenAI reasoning
4. **claude-3-5-sonnet-20241022** - Anthropic Claude 3.5

## Output Files

```
golden_eval_results/
├── comparison_report.json          # ← START HERE
├── gpt-4o/results.json
├── gpt-4o-mini/results.json
├── o3-mini/results.json
└── claude-3-5-sonnet-20241022/results.json
```

## Key Metrics

The evaluation tracks:

✓ **Mean Accuracy** - Average correctness across all turns
✓ **Perfect Rate** - % of conversations with all turns correct
✓ **By Complexity** - Performance on 1-2, 3-4, 5+ turn dialogues
✓ **By Question Type** - Simple vs multi-hop reasoning
✓ **By Data Type** - Numeric vs text handling

## Example Output

```
Overall Performance:
Model                               Mean Accuracy    Perfect Rate
--------------------------------------------------------------------------------
gpt-4o                                     95.83%          83.33%
gpt-4o-mini                                91.67%          75.00%
o3-mini                                    97.22%          91.67%
claude-3-5-sonnet-20241022                 93.06%          75.00%
```

## Documentation

- **Full Guide**: See `GOLDEN_DATASET_EVALUATION.md`
- **Dataset Details**: See `data/golden_dataset.json` metadata section
- **Examples**: All 12 examples with full context in the JSON

## Time & Cost Estimate

- **Time**: ~20-30 minutes per model
- **Total**: ~90-120 minutes for all 4 models
- **Cost**: ~$2-4 total across all models

## Next Steps

After running evaluation:

1. Open `golden_eval_results/comparison_report.json`
2. Identify best-performing model for your use case
3. Examine specific model results for error patterns
4. Use insights to optimize your agent configuration
