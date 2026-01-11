# Evaluation Metrics Quick Reference

## Three Core Accuracy Metrics

| Metric | Threshold | Description | Purpose |
|--------|-----------|-------------|---------|
| **Numerical** | \|P - T\| < 10⁻⁵ | Binary: Exact match | Scientific pass/fail bar |
| **Financial** | \|P - T\| / \|T\| ≤ 0.01 | 1% relative tolerance | Consulting standard (rounding forgiveness) |
| **Soft Match** | Varies | Forgiving match | Entity & logic understanding (units, scaling, sign) |

### 1. Numerical Accuracy (The "Binary" Metric)
**Formula**: Match = 1 if |P - T| < ε, else 0 (where ε = 10⁻⁵)

**Purpose**: Measures if system extracted exact digits from report

**Usage**: Strict pass/fail - no tolerance

### 2. Financial Accuracy (The "Consulting" Metric)  
**Formula**: Match = 1 if |P - T| / |T| ≤ 0.01, else 0

**Purpose**: Recognizes financial correctness is relative to rounding conventions

**Usage**: Most important metric for human analysts - forgives 12.5 vs 12.51

### 3. Soft Match (The "Entity & Logic" Metric)
**What it forgives**:
- **Unit Mismatches**: 0.12 = 12% = 1,000,000 = 1M
- **Signage**: -500 = (500)
- **Scaling**: 125.0 vs 1.25 (digits identical, decimal shifted)

**Purpose**: Debug tool - High soft match + Low numerical = Unit/Normalization bug

**Usage**: Evaluates intent and magnitude even with format differences

## Logic Recall (Reasoning Trace Score)

**Formula**: `Shared Operations / Total Ground Truth Operations`

**Interpretation**:
- 1.0 (100%) = Perfect reasoning match
- 0.5-0.99 = Partial reasoning match  
- 0.0 = No reasoning overlap

**Example**:
```
Ground Truth: subtract(8181, 20454), divide(#0, 20454)
GT Operations: ["subtract", "divide"]

Your Plan: subtract then divide
Your Operations: ["subtract", "divide"]
Logic Recall: 2/2 = 1.0 ✓

Your Plan: only subtract
Your Operations: ["subtract"]
Logic Recall: 1/2 = 0.5 ⚠️
```

## Key Metrics to Monitor

### Accuracy Hierarchy
1. **Numerical Accuracy** - Binary metric (exact match)
2. **Financial Accuracy** - Consulting standard (1% tolerance)  
3. **Soft Match** - Forgiving metric (units/scaling/sign)

### Reasoning Quality
- **Logic Recall** - Did model use correct operations?
- **Operations Per Turn** - Complexity of plans
- **Accuracy vs Turn Number** - Performance degradation

### Cost Efficiency  
- **Cost Per Correct Answer** - Tokens / Successful conversations
- **Avg Tokens Per Turn** - Token efficiency

## CSV Outputs

### summary.csv
Per-conversation aggregate metrics
```csv
trace_id,conversation_id,num_turns,numerical_accuracy,financial_accuracy,
soft_match_accuracy,avg_logic_recall,avg_operations_per_turn,total_tokens,...
```

### turns.csv
Per-turn detailed metrics
```csv
trace_id,conversation_id,turn_idx,question,expected_answer,actual_answer,
numerical_match,financial_match,soft_match,logic_recall,operations_per_turn,
ground_truth_program,...
```

### errors.csv (NEW)
Failed turns with context
```csv
trace_id,conversation_id,turn_idx,question,error_type,error_message,
has_type2_question,has_non_numeric_values,operations_per_turn,...
```

### statistics.json
Aggregate statistics
```json
{
  "numerical_accuracy": 0.82,
  "financial_accuracy": 0.87,
  "soft_match_accuracy": 0.92,
  "avg_logic_recall": 0.75,
  "cost_per_correct_answer": 12500.5,
  "accuracy_by_turn_number": {"0": 0.92, "1": 0.85, "2": 0.78},
  "avg_operations_by_turn": {"0": 0.5, "1": 1.2, "2": 1.8}
}
```

## Analysis Patterns

### 1. Identify Primary Issue

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Low numerical, High financial | Small rounding differences | Acceptable - within 1% |
| Low numerical, High soft_match | Unit/scaling issues | Fix normalization |
| Low financial, High soft_match | Sign or magnitude errors | Review operand order |
| Low logic recall | Wrong operations | Review prompt patterns |

### 2. Find Complexity Ceiling

```python
# Load statistics
import json
with open("test_eval/statistics.json") as f:
    stats = json.load(f)

# Check where accuracy drops
for turn, acc in stats["accuracy_by_turn_number"].items():
    ops = stats["avg_operations_by_turn"][turn]
    print(f"Turn {turn}: {acc:.1%} accuracy, {ops:.1f} avg operations")
```

### 3. Error Mode Analysis

```python
import pandas as pd

errors = pd.read_csv("test_eval/errors.csv")

# Most common errors
print(errors["error_type"].value_counts())

# Errors by complexity
print(errors.groupby("operations_per_turn")["error_type"].value_counts())

# Errors by question type
print(errors.groupby("has_type2_question").size())
```

### 4. Cost Optimization

```python
turns = pd.read_csv("test_eval/turns.csv")

# Token efficiency by accuracy
correct_turns = turns[turns["numerical_match"] == True]
incorrect_turns = turns[turns["numerical_match"] == False]

print(f"Avg tokens (correct): {correct_turns['turn_tokens'].mean():.0f}")
print(f"Avg tokens (incorrect): {incorrect_turns['turn_tokens'].mean():.0f}")
```

## Quick Wins

### Improve Accuracy
1. Check **soft_match** vs **numerical** gap → Unit/scaling bug (digits right, format wrong)
2. Check **financial** vs **numerical** gap → Rounding precision (acceptable if financial is high)
3. High **soft_match** but low **financial** → Logic error (wrong magnitude)

### Improve Reasoning
1. Low **logic recall** → Review prompt patterns
2. High **operations_per_turn** but low accuracy → Simplify plans
3. **Accuracy drops** with turn number → Improve context tracking

### Reduce Cost
1. High **cost_per_correct_answer** → Optimize prompt length
2. Many **validator_retries** → Improve plan validation
3. High **avg_tokens_per_turn** → Reduce context window

## Feature Flags from Dataset

- **has_type2_question**: Reasoning-heavy questions (harder)
- **has_non_numeric_values**: Tables with text entries
- **has_duplicate_columns**: Tables with repeated column names

Use these to stratify analysis:
```python
turns = pd.read_csv("test_eval/turns.csv")
summary = pd.read_csv("test_eval/summary.csv")

# Accuracy by question type
summary.groupby("has_type2_question")["numerical_accuracy"].mean()

# Logic recall by complexity
turns.groupby("has_type2_question")["logic_recall"].mean()
```

## Command Reference

### Run Evaluation
```bash
# Small test (10 samples)
uv run main evaluate --sample-size 10

# Medium test (100 samples)
uv run main evaluate --sample-size 100

# Full evaluation (1000+ samples)
uv run main evaluate --sample-size 1000
```

### Results Location
All outputs saved to: `test_eval/` (or specified output dir)
- `summary.csv` - Conversation-level metrics
- `turns.csv` - Turn-level metrics  
- `validation.csv` - Validator logs
- `errors.csv` - **NEW** Error analysis
- `statistics.json` - Aggregate stats
- `traces/*.json` - Individual traces

## Interpretation Guide

### Numerical Accuracy (Binary Metric)
- **>95%**: Excellent - extracting exact values
- **85-95%**: Good - minor floating-point differences
- **70-85%**: Moderate - rounding issues
- **<70%**: Poor - significant extraction errors

### Financial Accuracy (Consulting Metric)
- **>90%**: Excellent - meets professional standards
- **80-90%**: Good - acceptable for analysis
- **70-80%**: Moderate - needs improvement
- **<70%**: Poor - unacceptable for financial work

### Soft Match (Debug Metric)
- **High soft + Low financial**: Model has logic error (wrong calculation)
- **High soft + Low numerical**: Model has formatting error (right magnitude, wrong format)
- **Low soft**: Model fundamentally misunderstands the question

### Logic Recall
- **>90%**: Model understands reasoning well
- **75-90%**: Good reasoning, minor gaps
- **50-75%**: Partial reasoning understanding
- **<50%**: Poor reasoning match

### Cost Per Correct Answer
- **<10K tokens**: Very efficient
- **10-20K tokens**: Efficient
- **20-30K tokens**: Moderate efficiency
- **>30K tokens**: Inefficient, optimize prompts

### Complexity Ceiling
- Accuracy should be **>80%** for turns 0-1
- Acceptable drop of **5-10%** by turn 2-3
- Drop of **>15%** indicates context degradation issue
