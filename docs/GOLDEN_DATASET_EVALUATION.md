# Golden Dataset Model Evaluation

## Overview

This evaluation framework tests the financial agent across multiple LLM models using a carefully curated "golden dataset" of 12 examples. The dataset is balanced across:

- **Dialogue complexity**: 4 examples each with 1-2, 3-4, and 5+ turns
- **Question types**: Mix of type2 (multi-hop reasoning) and simple questions
- **Data types**: Mix of numeric-only and non-numeric value handling

## Golden Dataset

Located at: `data/golden_dataset.json`

### Distribution

**By Dialogue Turns:**
- 1-2 turns: 4 examples
- 3-4 turns: 4 examples  
- 5+ turns: 4 examples

**By Features (each turn category has 1 of each):**
- `has_type2_question=false`, `has_non_numeric_values=false`
- `has_type2_question=false`, `has_non_numeric_values=true`
- `has_type2_question=true`, `has_non_numeric_values=false`
- `has_type2_question=true`, `has_non_numeric_values=true`

### Selected Examples

```
1.  Double_MAS/2012/page_92.pdf          | turns=2 | type2=False | non_numeric=False
2.  Single_MRO/2009/page_127.pdf-1       | turns=2 | type2=False | non_numeric=True
3.  Double_AES/2011/page_230.pdf         | turns=2 | type2=True  | non_numeric=False
4.  Double_AES/2011/page_131.pdf         | turns=1 | type2=True  | non_numeric=True
5.  Single_JKHY/2009/page_28.pdf-3       | turns=4 | type2=False | non_numeric=False
6.  Single_RE/2013/page_40.pdf-1         | turns=3 | type2=False | non_numeric=True
7.  Double_AES/2016/page_98.pdf          | turns=4 | type2=True  | non_numeric=False
8.  Double_PNC/2014/page_99.pdf          | turns=4 | type2=True  | non_numeric=True
9.  Single_UPS/2009/page_33.pdf-2        | turns=6 | type2=False | non_numeric=False
10. Single_CE/2010/page_134.pdf-2        | turns=5 | type2=False | non_numeric=True
11. Double_UPS/2009/page_33.pdf          | turns=7 | type2=True  | non_numeric=False
12. Double_ETR/2016/page_403.pdf         | turns=8 | type2=True  | non_numeric=True
```

## Models Evaluated

The evaluation script tests 4 state-of-the-art models:

1. **gpt-5.2** - OpenAI's flagship model
2. **gpt-5-mini** - OpenAI's efficient model
3. **o3** - OpenAI's reasoning model
4. **gpt4o** - baseline

## Running the Evaluation

### Basic Usage

```bash
cd version3
python evaluate_golden_dataset.py
```

This will:
1. Load the golden dataset
2. Evaluate each of the 4 models on all 12 examples
3. Generate detailed results and comparison reports
4. Save outputs to `golden_eval_results/`

### Custom Options

```bash
# Evaluate specific models only
python evaluate_golden_dataset.py --models gpt-4o claude-3-5-sonnet-20241022

# Use custom dataset path
python evaluate_golden_dataset.py --golden-dataset path/to/dataset.json

# Save to custom output directory
python evaluate_golden_dataset.py --output-dir my_eval_results

# Combine options
python evaluate_golden_dataset.py \
  --models gpt-4o gpt-4o-mini \
  --output-dir eval_$(date +%Y%m%d) \
  --golden-dataset data/golden_dataset.json
```

## Output Structure

```
golden_eval_results/
├── comparison_report.json          # Cross-model comparison
├── gpt-4o/
│   └── results.json               # Detailed results for gpt-4o
├── gpt-4o-mini/
│   └── results.json               # Detailed results for gpt-4o-mini
├── o3-mini/
│   └── results.json               # Detailed results for o3-mini
└── claude-3-5-sonnet-20241022/
    └── results.json               # Detailed results for Claude
```

### Results Format

Each model's `results.json` contains:

```json
[
  {
    "example_id": "Double_MAS/2012/page_92.pdf",
    "model": "gpt-4o",
    "features": {
      "num_dialogue_turns": 2,
      "has_type2_question": false,
      "has_non_numeric_values": false
    },
    "num_turns": 2,
    "conversation_accuracy": 1.0,
    "all_correct": true,
    "turns": [
      {
        "turn": 1,
        "question": "...",
        "answer": 123.45,
        "expected": "123.45",
        "correct": true,
        "accuracy": 1.0,
        "execution_time_ms": 1234.5,
        "plan": {...},
        "result": {...}
      },
      ...
    ]
  },
  ...
]
```

### Comparison Report

The `comparison_report.json` includes:

- Overall performance metrics per model
- Performance breakdown by dialogue turns (1-2, 3-4, 5+)
- Performance breakdown by question type (type2 vs simple)
- Performance breakdown by data type (numeric vs non-numeric)

## Evaluation Metrics

### Conversation-Level Metrics

- **Mean Accuracy**: Average accuracy across all dialogue turns
- **Perfect Rate**: Percentage of conversations where all turns are correct
- **Execution Time**: Time taken for each turn

### Turn-Level Metrics

- **Accuracy**: Binary (1.0 if correct within 0.01 tolerance, 0.0 otherwise)
- **Answer**: Model's final answer
- **Expected**: Ground truth answer
- **Correct**: Boolean indicating if answer matches expected

## Sample Output

```
═══════════════════════════════════════════════════════════════════════════════
                    GOLDEN DATASET MODEL EVALUATION
═══════════════════════════════════════════════════════════════════════════════

Dataset: data/golden_dataset.json
Output: golden_eval_results
Models: gpt-4o, gpt-4o-mini, o3-mini, claude-3-5-sonnet-20241022

Loaded 12 examples from golden dataset

════════════════════════════════════════════════════════════════════════════════
Evaluating model: gpt-4o
════════════════════════════════════════════════════════════════════════════════

  [1/12] Double_MAS/2012/page_92.pdf: Accuracy=100.00%, Turns=2
  [2/12] Single_MRO/2009/page_127.pdf-1: Accuracy=100.00%, Turns=2
  ...

✓ Saved results to golden_eval_results/gpt-4o/results.json

════════════════════════════════════════════════════════════════════════════════
MODEL COMPARISON REPORT
════════════════════════════════════════════════════════════════════════════════

Overall Performance:
Model                               Mean Accuracy    Perfect Rate
--------------------------------------------------------------------------------
gpt-4o                                     95.83%          83.33%
gpt-4o-mini                                91.67%          75.00%
o3-mini                                    97.22%          91.67%
claude-3-5-sonnet-20241022                 93.06%          75.00%


Performance by Dialogue Turns:
Model                                1-2 turns   3-4 turns    5+ turns
--------------------------------------------------------------------------------
gpt-4o                                  100.00%      95.83%      91.67%
gpt-4o-mini                              95.83%      91.67%      87.50%
o3-mini                                 100.00%      97.92%      93.75%
claude-3-5-sonnet-20241022               97.92%      91.67%      89.58%
...
```

## Requirements

Ensure you have the required dependencies:

```bash
uv pip install -r pyproject.toml
```

Required environment variables:
- `OPENAI_API_KEY` - For OpenAI models (gpt-4o, gpt-4o-mini, o3-mini)
- `ANTHROPIC_API_KEY` - For Claude models

## Tips

1. **Rate Limiting**: The script includes 1-second delays between turns to avoid rate limits
2. **Cost**: Each evaluation run costs approximately:
   - gpt-4o: ~$0.50-1.00
   - gpt-4o-mini: ~$0.05-0.10
   - o3-mini: ~$0.30-0.60
   - claude-3-5-sonnet: ~$0.40-0.80

3. **Time**: Full evaluation takes ~20-30 minutes per model (12 examples × multiple turns)

4. **Debugging**: Check individual model results in their respective directories for detailed execution traces

## Interpreting Results

### High Performance Indicators
- Mean accuracy > 95%
- Perfect rate > 80%
- Consistent performance across turn categories

### Areas to Investigate
- Significant drops in 5+ turn conversations (context tracking issues)
- Poor performance on type2 questions (multi-hop reasoning issues)
- Issues with non-numeric values (text parsing problems)

## Next Steps

After running evaluation:

1. Review `comparison_report.json` for model rankings
2. Examine individual model results for error patterns
3. Identify which models perform best on specific question types
4. Use insights to optimize model selection or prompting strategies
