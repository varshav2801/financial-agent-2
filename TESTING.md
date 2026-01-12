# Test Scripts Documentation

This directory contains test scripts for evaluating the financial agent implementation.

## Scripts Overview

### 1. `batch_test.py` - Random Sample Testing

Tests the agent on a random sample from the dataset with configurable parameters.

**Features:**
- Random sampling with seed for reproducibility
- Configurable sample size
- Comprehensive metrics logging
- Support for any model

**Usage:**

```bash
# Basic usage (20 examples, seed 42, gpt-4o)
uv run python batch_test.py

# Custom sample size and seed
uv run python batch_test.py --sample-size 50 --seed 123

# Test with different model
uv run python batch_test.py --model claude-3-5-sonnet-20241022

# Specify custom output directory
uv run python batch_test.py --output-dir my_test_results

# Full example
uv run python batch_test.py \
    --dataset data/convfinqa_dataset.json \
    --sample-size 30 \
    --seed 456 \
    --model gpt-4o-mini \
    --output-dir batch_test_gpt4o_mini
```

**Arguments:**
- `--dataset`: Path to dataset JSON file (default: `data/convfinqa_dataset.json`)
- `--sample-size`: Number of examples to sample (default: 20)
- `--seed`: Random seed for reproducibility (default: 42)
- `--model`: Model to evaluate (default: `gpt-4o`)
- `--output-dir`: Directory to save results (default: auto-generated)

**Output Files:**
- `sample_metadata.json`: Information about the sample (IDs, seed, etc.)
- `detailed_results.json`: Full results for each example
- `metrics.json`: Comprehensive metrics and aggregations

**Metrics Tracked:**
- Numerical, Financial, and Soft Match Accuracy
- Perfect conversation rate
- Token usage (total and per-turn averages)
- Response times (total and per-turn averages)
- Performance by dialogue turns (1-2, 3-4, 5+)
- Performance by features (type2 questions, non-numeric values, duplicate columns)

---

### 2. `quick_test.py` - Specific Examples Testing

Tests the agent on specific examples, ideal for development and debugging.

**Features:**
- Test specific example IDs
- Load examples from file (like `test_examples_20_good.txt`)
- Filter by complexity (simple/medium/complex)
- Detailed per-turn results
- Quick feedback loop

**Usage:**

```bash
# Test all examples from file
uv run python quick_test.py

# Test only simple examples (1-2 turns)
uv run python quick_test.py --simple

# Test only medium examples (3-4 turns)
uv run python quick_test.py --medium

# Test only complex examples (4-5 turns or type2)
uv run python quick_test.py --complex

# Test specific examples by ID
uv run python quick_test.py --examples \
    "Single_JPM/2013/page_104.pdf-2" \
    "Single_AAPL/2002/page_23.pdf-1"

# Test with different model
uv run python quick_test.py --model o3-mini --simple

# Use custom examples file
uv run python quick_test.py --examples-file my_examples.txt

# Full example
uv run python quick_test.py \
    --dataset data/convfinqa_dataset.json \
    --examples-file ../test_examples_20_good.txt \
    --model claude-3-5-sonnet-20241022 \
    --output-dir quick_test_claude \
    --medium
```

**Arguments:**
- `--dataset`: Path to dataset JSON file (default: `data/convfinqa_dataset.json`)
- `--examples-file`: Path to file with example IDs (default: `../test_examples_20_good.txt`)
- `--examples`: Specific example IDs to test (overrides `--examples-file`)
- `--model`: Model to evaluate (default: `gpt-4o`)
- `--output-dir`: Directory to save results (default: auto-generated)
- `--simple`: Only test simple examples (1-2 turns)
- `--medium`: Only test medium examples (3-4 turns)
- `--complex`: Only test complex examples (4-5 turns or type2)

**Output Files:**
- `test_metadata.json`: Information about the test (IDs, missing examples, etc.)
- `detailed_results.json`: Full results for each example including per-turn details
- `metrics.json`: Overall metrics summary

**Example File Format:**
```
# Comments start with #
# Example IDs, one per line

# SIMPLE (1-2 turns)
Single_BKR/2017/page_105.pdf-2
Single_L/2016/page_62.pdf-1

# MEDIUM (3-4 turns)
Single_JPM/2013/page_104.pdf-2
Single_JKHY/2009/page_28.pdf-3

# COMPLEX (4-5 turns or type2)
Double_AES/2016/page_98.pdf
Single_HII/2015/page_120.pdf-1
```

---

### 3. `evaluate_model_eval_dataset.py` - Model Comparison

Evaluates multiple models on a balanced test dataset for comprehensive comparison.

**Usage:**

```bash
# Evaluate multiple models
uv run python evaluate_model_eval_dataset.py \
    --models gpt-4o gpt-4o-mini o3-mini

# Use custom dataset
uv run python evaluate_model_eval_dataset.py \
    --model-eval-dataset data/my_eval_dataset.json \
    --models claude-3-5-sonnet-20241022 gpt-4o
```

---

## Common Workflows

### Quick Development Testing
```bash
# Test 5 simple examples quickly
uv run python quick_test.py --simple --examples \
    "Single_BKR/2017/page_105.pdf-2" \
    "Single_L/2016/page_62.pdf-1" \
    "Single_JPM/2009/page_206.pdf-3"
```

### Reproducible Performance Testing
```bash
# Test with fixed seed for reproducibility
uv run python batch_test.py --sample-size 50 --seed 42 --model gpt-4o
```

### Complexity-Specific Testing
```bash
# Test only complex examples to debug difficult cases
uv run python quick_test.py --complex --model gpt-4o
```

### Model Comparison
```bash
# Compare two models on same random sample
uv run python batch_test.py --sample-size 30 --seed 100 --model gpt-4o
uv run python batch_test.py --sample-size 30 --seed 100 --model gpt-4o-mini
```

---

## Understanding Metrics

### Accuracy Metrics
- **Numerical Accuracy**: Exact numerical match
- **Financial Accuracy**: Within 1% tolerance (standard for financial data)
- **Soft Match Accuracy**: Flexible matching (handles formatting variations)

### Efficiency Metrics
- **Tokens per Turn**: Average tokens used per dialogue turn
- **Response Time**: Average milliseconds per turn
- **Perfect Rate**: Percentage of conversations with all turns correct

### Feature-Based Analysis
- **By Dialogue Turns**: Performance on short (1-2), medium (3-4), and long (5+) conversations
- **By Type2 Question**: Performance on complex reasoning questions
- **By Non-Numeric Values**: Performance when tables contain text/mixed data

---

## Tips

1. **Start Small**: Use `quick_test.py --simple` first to validate basic functionality
2. **Use Seeds**: Always use `--seed` in `batch_test.py` for reproducible experiments
3. **Check Errors**: Review `detailed_results.json` for examples that failed or timed out
4. **Compare Apples to Apples**: Use same seed when comparing models with `batch_test.py`
5. **Debug with Quick Test**: Use `quick_test.py` with specific IDs to debug individual failures

---

## Troubleshooting

### "Dataset file not found"
- Ensure you're running from the `version3` directory
- Check the dataset path: `data/convfinqa_dataset.json` should exist

### "No examples found in dataset"
- Verify the JSON structure has `examples` or `train` key
- Check JSON is valid with `python -m json.tool data/convfinqa_dataset.json`

### "Example ID not found"
- Verify the example ID exists in the dataset
- Check for typos or extra whitespace in example IDs

### "Agent initialization failed"
- Check API keys are set in environment variables
- Verify model name is correct

### Timeouts
- Default timeout is 5 minutes per example
- Some complex examples may need longer
- Check for infinite loops or errors in agent code
