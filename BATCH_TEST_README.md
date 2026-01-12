# Batch Test Script

A script to batch test the financial agent on randomly selected examples, exactly mimicking the behavior of the `chat` command.

## Usage

```bash
# Test 5 random examples with default settings
uv run python run_batch_test.py -n 5

# Test 10 examples with a specific seed for reproducibility
uv run python run_batch_test.py -n 10 --seed 123

# Test with a specific model
uv run python run_batch_test.py -n 20 --model claude-3-5-sonnet-20241022

# Specify custom output directory
uv run python run_batch_test.py -n 15 --output-dir my_test_results

# Use all options together
uv run python run_batch_test.py -n 25 --seed 456 --model gpt-4o --output-dir custom_results
```

## Options

- `-n, --num-examples`: **[Required]** Number of examples to randomly select and test
- `--seed`: Random seed for reproducibility (default: 42)
- `--model`: Model to use (default: gpt-4o)
- `--output-dir`: Output directory (default: `batch_test_results_<model>_<timestamp>`)
- `--dataset`: Path to dataset (default: `data/convfinqa_dataset.json`)

## Output Format

The script saves results in a format matching the `o3_eval_results` structure:

### Files Created

1. **`results.json`**: Complete turn-by-turn results for all examples
   - Includes plan, step results, tokens, timing for each turn
   - Matches the format in `evaluation_results/o3_eval_results/o3/results.json`

2. **`metadata.json`**: Test run metadata
   - Timestamp, model, seed, sample size
   - List of all example IDs tested

3. **`aggregate_metrics.json`**: Summary statistics
   - Overall accuracy metrics
   - Token and timing averages
   - Perfect conversation rate

### Results Format

Each example in `results.json` includes:

```json
{
  "example_id": "Double_UPS/2009/page_33.pdf",
  "model": "gpt-4o",
  "features": {
    "num_dialogue_turns": 7,
    "has_type2_question": true,
    "has_duplicate_columns": false,
    "has_non_numeric_values": false
  },
  "num_turns": 7,
  "turns": [
    {
      "success": true,
      "answer": -8.94,
      "expected": "-8.94",
      "question": "what was the fluctuation...",
      "turn": 1,
      "execution_time_ms": 2500.5,
      "plan": { /* full plan with thought_process and steps */ },
      "step_results": { "1": 91.06, "2": 100.0, "3": -8.94 },
      "error": null,
      "numerical_match": true,
      "financial_match": true,
      "soft_match": true,
      "tokens": {
        "prompt_tokens": 5000,
        "completion_tokens": 500,
        "total_tokens": 5500
      }
    }
    /* ... more turns ... */
  ],
  "numerical_accuracy": 1.0,
  "financial_accuracy": 1.0,
  "soft_match_accuracy": 1.0,
  "all_correct": true,
  "correct_turns": 7,
  "total_turns": 7,
  "total_tokens": 38500,
  "total_response_time_ms": 17500.0,
  "avg_tokens_per_turn": 5500.0,
  "avg_response_time_ms": 2500.0
}
```

## Behavior

- **Random Selection**: Uses Python's `random.sample()` with specified seed for reproducibility
- **Sequential Execution**: Runs examples one at a time (not parallel) to avoid rate limiting
- **Rate Limiting**: 1-second delay between examples
- **Chat-like Output**: Displays each turn's question, answer, and correctness during execution
- **Error Handling**: Captures and logs errors without stopping the batch
- **Exact Agent Behavior**: Uses `FinancialAgent.run_conversation()` exactly like the `chat` command

## Example Output

```
═══════════════════════════════════════════════════════════════════════════════
                         BATCH TEST RUNNER                           
═══════════════════════════════════════════════════════════════════════════════

Dataset: data/convfinqa_dataset.json
Sample Size: 5
Seed: 42
Model: gpt-4o
Output Dir: batch_test_results_gpt_4o_20260112_123045

Loading dataset from data/convfinqa_dataset.json...
Found 3892 total examples
Selected 5 examples (seed=42)
✓ Saved metadata to batch_test_results_gpt_4o_20260112_123045/metadata.json

Initializing agent with model: gpt-4o...
✓ Agent initialized

================================================================================
Example 1/5: Double_UPS/2009/page_33.pdf
================================================================================

Loaded record with 7 conversation turns

Turn 1/7: what was the fluctuation of the performance price...
  Answer: -8.94
  Expected: -8.94
  ✓ Correct! (Numerical)

...

Summary:
  Financial Accuracy: 100.00%
  Correct Turns: 7/7
  Total Tokens: 38500
  Total Time: 17500.0ms

================================================================================
✓ BATCH TEST COMPLETE
================================================================================

Results saved to: batch_test_results_gpt_4o_20260112_123045/results.json

Aggregate Metrics:
  Successful: 5/5
  Financial Accuracy: 95.20%
  Numerical Accuracy: 92.40%
  Perfect Conversations: 3 (60.0%)
  Avg Tokens/Example: 42500.0
  Avg Time/Example: 18250.5ms

✓ Saved aggregate metrics to batch_test_results_gpt_4o_20260112_123045/aggregate_metrics.json
```

## Reproducibility

Using the same seed will always select the same examples:

```bash
# These will test the exact same examples
uv run python run_batch_test.py -n 10 --seed 42
uv run python run_batch_test.py -n 10 --seed 42
```

## Comparison with Other Models

You can easily compare different models on the same examples:

```bash
# Test GPT-4o
uv run python run_batch_test.py -n 20 --seed 42 --model gpt-4o --output-dir results_gpt4o

# Test Claude on same examples
uv run python run_batch_test.py -n 20 --seed 42 --model claude-3-5-sonnet-20241022 --output-dir results_claude

# Compare results
diff results_gpt4o/results.json results_claude/results.json
```
