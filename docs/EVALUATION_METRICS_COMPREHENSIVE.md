# Comprehensive Evaluation Metrics

## Overview
The golden dataset evaluation now tracks **all required metrics** for comprehensive model comparison.

## Metrics Tracked

### 1. **Accuracy Metrics** (Three Types)

#### Numerical Accuracy (Binary)
- **Definition**: Exact match within floating-point tolerance (ε = 1e-5)
- **Formula**: `|predicted - expected| < ε`
- **Use Case**: Strictest accuracy for exact matches
- **Reported As**: Percentage (0-100%)

#### Financial Accuracy (Consulting Standard) ⭐ PRIMARY METRIC
- **Definition**: Within 1% relative error tolerance
- **Formula**: `|predicted - expected| / |expected| ≤ 0.01`
- **Use Case**: Industry standard for financial calculations
- **Reported As**: Percentage (0-100%)
- **Note**: Used for "Correct Turns" counting

#### Soft Match (Entity & Logic)
- **Definition**: Forgiving match that handles:
  - Unit mismatches: `0.12 = 12% = 0.0012`
  - Signage: `-500 = 500`
  - Scaling: `125.0 vs 1.25` (digits identical)
- **Use Case**: Captures logically correct answers with formatting differences
- **Reported As**: Percentage (0-100%)

### 2. **Correct Turns** ✓
- **Format**: `"x/y"` (e.g., `"3/4"`)
- **Definition**: Number of correct turns out of total turns
- **Based On**: Financial accuracy (primary metric)
- **Variations Tracked**:
  - `correct_turns`: Based on financial accuracy
  - `numerical_correct_turns`: Based on numerical accuracy
  - `soft_match_correct_turns`: Based on soft match

### 3. **Token Usage** 📊
Tracked per turn and aggregated:
- **prompt_tokens**: Input tokens consumed
- **completion_tokens**: Output tokens generated
- **total_tokens**: Sum of prompt + completion
- **avg_tokens_per_turn**: Average across all turns
- **avg_tokens_per_example**: Average per conversation

### 4. **Response Time** ⏱️
Measured in milliseconds:
- **execution_time_ms**: Total time per turn
- **plan_time_ms**: Time for planning phase
- **exec_time_ms**: Time for execution phase
- **avg_response_time_ms_per_turn**: Average per turn
- **avg_response_time_ms_per_example**: Average per conversation

## Report Structure

### Example-Level Results
Each example includes:
```json
{
  "example_id": "Single_JKHY/2009/page_28.pdf-3",
  "model": "gpt-4o",
  "num_turns": 4,
  
  "numerical_accuracy": 0.75,
  "financial_accuracy": 1.0,
  "soft_match_accuracy": 1.0,
  
  "correct_turns": "4/4",
  "numerical_correct_turns": "3/4",
  "soft_match_correct_turns": "4/4",
  
  "total_tokens": 2456,
  "avg_tokens_per_turn": 614.0,
  
  "total_response_time_ms": 3240.5,
  "avg_response_time_ms": 810.1,
  
  "turns": [...]
}
```

### Aggregate Metrics
Calculated across all examples with breakdowns by:
- **Overall**: All examples combined
- **By Turns**: `1-2`, `3-4`, `5+` dialogue turns
- **By Type2 Question**: `true` vs `false`
- **By Non-Numeric Values**: `true` vs `false`

## Comparison Report Output

### 1. Overall Performance - Accuracy Metrics
```
Model                          Numerical   Financial  Soft Match  Perfect Rate
-------------------------------------------------------------------------------------------------
gpt-4o                            85.42%      91.67%      95.83%        75.00%
gpt-4o-mini                       79.17%      87.50%      93.75%        66.67%
o3-mini                           81.25%      89.58%      94.79%        70.83%
claude-3-5-sonnet-20241022        87.50%      93.75%      97.92%        83.33%
```

### 2. Overall Performance - Efficiency Metrics
```
Model                          Avg Tokens/Turn  Avg Response (ms)
-------------------------------------------------------------------------------------------------
gpt-4o                                   645.2             892.3
gpt-4o-mini                              512.8             623.1
o3-mini                                  578.4             745.6
claude-3-5-sonnet-20241022               721.5             856.7
```

### 3. Performance by Dialogue Turns (Financial Accuracy)
```
Model                           1-2 turns   3-4 turns    5+ turns
-------------------------------------------------------------------------------------------------
gpt-4o                             95.83%      91.67%      87.50%
gpt-4o-mini                        91.67%      87.50%      83.33%
...
```

### 4. Performance by Type2 Question (Financial Accuracy)
```
Model                             No Type2   Has Type2
-------------------------------------------------------------------------------------------------
gpt-4o                              93.75%      89.58%
...
```

### 5. Performance by Non-Numeric Values (Financial Accuracy)
```
Model                         No Non-Numeric  Has Non-Numeric
-------------------------------------------------------------------------------------------------
gpt-4o                                95.83%           87.50%
...
```

## Key Improvements

✅ **All Required Metrics Tracked**:
- ✓ Numerical accuracy (binary)
- ✓ Financial accuracy (1% tolerance)
- ✓ Soft match (forgiving)
- ✓ Correct turns (x/y format)
- ✓ Token usage (per turn and total)
- ✓ Response time (breakdown by phase)

✅ **Comprehensive Breakdowns**:
- By dialogue complexity (turn count)
- By question type (type2)
- By value type (non-numeric)

✅ **Industry-Standard Metrics**:
- Primary metric: Financial accuracy (consulting standard)
- Token efficiency for cost analysis
- Response time for latency analysis

## Usage

Run evaluation:
```bash
cd version3
python evaluate_golden_dataset.py \
  --golden-dataset data/golden_dataset.json \
  --output-dir golden_eval_results \
  --models gpt-4o gpt-4o-mini o3-mini claude-3-5-sonnet-20241022
```

Results saved to:
- `golden_eval_results/[model_name]/results.json` - Detailed per-example results
- `golden_eval_results/comparison_report.json` - Aggregate comparison data
- Console output - Formatted comparison tables

## Interpretation Guide

### Financial Accuracy (Primary)
- **>95%**: Excellent performance
- **90-95%**: Good performance
- **85-90%**: Acceptable performance
- **<85%**: Needs improvement

### Token Efficiency
- Lower is better for cost optimization
- Compare across models for same accuracy level

### Response Time
- Lower is better for user experience
- Consider trade-off with accuracy

### Perfect Rate
- Percentage of conversations with all turns correct
- Critical for multi-turn reasoning evaluation
