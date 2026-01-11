# Evaluation Metrics Update

## Overview
Updated the evaluation system to provide comprehensive accuracy metrics, reasoning quality analysis, and error tracking for detailed performance analysis.

## New Accuracy Metrics

### 1. Soft Numerical Match (±1% Tolerance)
- **Purpose**: Accepts values within ±1% tolerance to account for rounding differences
- **Example**: 33.3 vs 33.33 would match
- **Implementation**: `abs_diff <= abs(exp_val) * 0.01`

### 2. Unit-Agnostic Accuracy
- **Purpose**: Forgives scaling differences to identify if model understood magnitude
- **Examples**: 
  - 0.10 vs 10 vs 10% (all match)
  - Handles common percentage representation issues
- **Implementation**: Checks scaling by 100 and 0.01

### 3. Sign-Agnostic Accuracy  
- **Purpose**: Forgives sign if absolute values match
- **Example**: -33.3 vs 33.3 would match
- **Use Case**: Identifies when model got magnitude right but sign wrong

### 4. General Match (Existing, Enhanced)
- **Purpose**: Flexible matching with percentage format conversions
- **Handles**: Various formats and representations

### 5. Numerical Match (Existing, Baseline)
- **Purpose**: Strict accuracy within 1% relative error OR absolute difference < 0.001
- **Use Case**: Primary accuracy metric for model performance

## Reasoning Quality Metrics

### Logic Recall (Reasoning Trace Score)
**Formula**: `(Shared Operations) / (Total Ground Truth Operations)`

**Purpose**: Measures if the model understood the correct reasoning steps

**Methodology**:
1. Parse ground truth `turn_program` from dataset
2. Extract operations from workflow plan (add, subtract, divide, multiply)
3. Calculate overlap between plan operations and ground truth operations
4. Handles special cases:
   - Extract-only (no operations): Checks if plan is also extract-only
   - Percentage calculations: Matches divide operations
   - Multi-step reasoning: Compares operation sequences

**Example**:
```
Ground Truth: "subtract(8181, 20454), divide(#0, 20454)"
Operations: ["subtract", "divide"]

Plan Operations: ["subtract", "divide"]
Logic Recall: 2/2 = 1.0 (100%)

Plan Operations: ["subtract"]
Logic Recall: 1/2 = 0.5 (50%)
```

### Operations Per Turn
- Tracks number of compute operations in each turn's plan
- Correlates with accuracy to identify "Complexity Ceiling"
- Helps understand where model struggles with multi-step reasoning

## Complexity Analysis

### Accuracy vs Turn Number
- Tracks accuracy for each turn position (0, 1, 2, 3...)
- Identifies if model performance degrades in later conversation turns
- **Output**: `accuracy_by_turn_number` dict in statistics.json

### Average Operations by Turn Number
- Tracks average operations for each turn position
- Correlates with accuracy to find complexity limits
- **Output**: `avg_operations_by_turn` dict in statistics.json

## Cost Analysis

### Cost Per Correct Answer
**Formula**: `Total Tokens (Input + Output) / Number of Successful Conversations`

**Purpose**: Measures efficiency of correct answers

**Use Case**: 
- Optimization metric for prompt engineering
- Compare different model configurations
- Budget planning for production deployment

## Enhanced Error Logging

### Error Context Tracking
Each error now includes:
- `turn_idx`: Which turn failed
- `question`: The question that caused error
- `has_type2_question`: Boolean from dataset features
- `has_non_numeric_values`: Boolean from dataset features  
- `conversation_id`: Trace back to original conversation
- `plan_steps`: Complexity at failure
- `operations_per_turn`: Operations attempted

### Error CSV Output
New `errors.csv` file contains all failed turns with full context for error mode analysis.

**Columns**:
- trace_id, conversation_id, turn_idx
- question, expected_answer
- error_type, error_message
- has_type2_question, has_non_numeric_values
- plan_steps, operations_per_turn
- error_context (full dict)

## Dataset Features Tracking

### Per-Turn Feature Tracking
Now tracks from dataset:
- `has_type2_question`: Type 2 questions (reasoning-heavy)
- `has_non_numeric_values`: Non-numeric table entries
- `ground_truth_program`: Expected operation sequence

### Per-Conversation Feature Tracking  
Includes in trace:
- `has_type2_question`
- `has_duplicate_columns`
- `has_non_numeric_values`

## Output Files Updated

### 1. summary.csv
**New Columns**:
- `soft_match_accuracy`
- `unit_agnostic_accuracy`
- `sign_agnostic_accuracy`
- `avg_logic_recall`
- `avg_operations_per_turn`
- `has_non_numeric_values`

### 2. turns.csv  
**New Columns**:
- `soft_match`
- `unit_agnostic_match`
- `sign_agnostic_match`
- `logic_recall`
- `operations_per_turn`
- `ground_truth_program`

### 3. errors.csv (NEW)
Dedicated error analysis file with full context for each failure.

### 4. statistics.json
**New Fields**:
```json
{
  "soft_match_accuracy": 0.85,
  "unit_agnostic_accuracy": 0.90,
  "sign_agnostic_accuracy": 0.87,
  "avg_logic_recall": 0.75,
  "cost_per_correct_answer": 12500.5,
  "accuracy_by_turn_number": {
    "0": 0.92,
    "1": 0.85,
    "2": 0.78,
    "3": 0.72
  },
  "avg_operations_by_turn": {
    "0": 0.5,
    "1": 1.2,
    "2": 1.8,
    "3": 2.1
  }
}
```

## Usage

Run evaluation as before:
```bash
uv run main evaluate --sample-size 100
```

**Output will include**:
1. `summary.csv` - Per-conversation metrics with new accuracy types
2. `turns.csv` - Per-turn details with logic recall
3. `validation.csv` - Validator logs (existing)
4. `errors.csv` - **NEW** Error analysis file
5. `statistics.json` - Aggregate statistics with all new metrics
6. `traces/*.json` - Individual conversation traces

## Analysis Workflows

### 1. Identify Accuracy Issues
Compare accuracy metrics to understand failure modes:
- Low numerical + High unit-agnostic → Scaling issues
- Low numerical + High sign-agnostic → Sign/direction confusion
- Low numerical + High soft-match → Rounding precision issues

### 2. Reasoning Quality Analysis
```python
import pandas as pd

turns = pd.read_csv("turns.csv")

# Correlation between operations and accuracy
turns.groupby("operations_per_turn")["numerical_match"].mean()

# Logic recall by turn number
turns.groupby("turn_idx")["logic_recall"].mean()
```

### 3. Error Mode Analysis
```python
errors = pd.read_csv("errors.csv")

# Most common error types
errors["error_type"].value_counts()

# Errors by question type
errors.groupby("has_type2_question")["error_type"].value_counts()

# Errors by complexity
errors.groupby("operations_per_turn")["error_type"].value_counts()
```

### 4. Complexity Ceiling Analysis
```python
import json

with open("statistics.json") as f:
    stats = json.load(f)

# Plot accuracy vs turn number
import matplotlib.pyplot as plt
turn_nums = sorted(stats["accuracy_by_turn_number"].keys())
accuracies = [stats["accuracy_by_turn_number"][t] for t in turn_nums]
operations = [stats["avg_operations_by_turn"][t] for t in turn_nums]

plt.plot(turn_nums, accuracies, label="Accuracy")
plt.plot(turn_nums, operations, label="Avg Operations")
plt.legend()
plt.show()
```

## Implementation Details

### Files Modified
1. **src/evaluation/models.py**: Added new metric fields to TurnMetrics, TraceRecord, EvaluationSummary
2. **src/evaluation/tracker.py**: 
   - Enhanced `calculate_accuracy()` with 5 accuracy types
   - Added `_calculate_logic_recall()` method
   - Added `_parse_ground_truth_operations()` method
   - Enhanced error logging with context
3. **src/evaluation/runner.py**: 
   - Extract `turn_program` from dataset
   - Pass ground truth to tracker
   - Enhanced error context
4. **src/evaluation/writer.py**:
   - Updated CSV headers and data rows
   - Added `write_error_log()` method
   - Enhanced `generate_summary_statistics()`

### Key Algorithm: Logic Recall

```python
def _calculate_logic_recall(plan_object, ground_truth_program):
    # Parse ground truth operations
    gt_ops = extract_operations(ground_truth_program)
    
    # Extract plan operations  
    plan_ops = [step.operation for step in plan_object.steps 
                if step.tool == "compute"]
    
    # Calculate overlap (order-insensitive)
    shared_ops = count_shared_operations(plan_ops, gt_ops)
    
    # Return recall
    return shared_ops / len(gt_ops) if gt_ops else 1.0
```

## Future Enhancements

Potential additions:
1. **Precision metric**: Complement to recall
2. **Operation order scoring**: Penalize incorrect operation sequences
3. **Partial credit**: Score based on partial correctness
4. **Entity tracking accuracy**: Verify correct entities extracted
5. **Semantic similarity**: Compare natural language reasoning
