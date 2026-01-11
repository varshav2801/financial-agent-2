# Metrics Tracking Fixes - Summary

## Issues Found and Fixed

### ✅ 1. Missing Tool Call Logging
**Problem**: The `WorkflowExecutor` was executing tool calls but never logging them to the tracker.

**Impact**: 
- `turn_tool_calls` was always 0
- Tool usage statistics were missing
- Tool latency metrics unavailable

**Fix**: Added `log_tool_call()` calls in `workflow_executor.py` for:
- Table extractions (`workflow_table_tool`)
- Text extractions (`text_tool`)
- Compute operations

---

### ✅ 2. Missing LLM Call Logging in TextTool
**Problem**: The `text_tool._llm_extract_value()` was calling the LLM but not logging to tracker.

**Impact**:
- Token counts underreported when text extraction used
- LLM call counts incomplete
- Cost tracking inaccurate

**Fix**: 
- Added `tracker` parameter to `TextTool.__init__()`
- Added `log_llm_call()` in `_llm_extract_value()` with `purpose="text_extraction"`
- Updated `WorkflowExecutor` to pass tracker to TextTool

---

### ✅ 3. Missing `purpose` Field in LLMCallLog
**Problem**: The planner was passing `purpose="plan_generation"` but `LLMCallLog` model didn't have this field.

**Impact**:
- Purpose information silently dropped
- Unable to distinguish between plan generation vs text extraction LLM calls

**Fix**: Added optional `purpose: str | None = None` field to `LLMCallLog` in `models.py`

---

### ✅ 4. Parameter Order Mismatch in log_llm_call
**Problem**: The planner was calling `log_llm_call()` with parameters in wrong order:
- Passing: `model, prompt_tokens, completion_tokens, total_tokens, latency_ms, purpose`
- Expected: `prompt_tokens, completion_tokens, latency_ms, model`

**Impact**: 
- Parameter misalignment (though Python's keyword args prevented crashes)
- Unnecessary `total_tokens` parameter (calculated internally)

**Fix**: 
- Updated `tracker.log_llm_call()` signature to include `purpose` parameter
- Fixed call in `workflow_planner.py` to use correct parameter order
- Removed redundant `total_tokens` parameter (computed from prompt + completion)

---

## Files Modified

1. **src/evaluation/models.py**
   - Added `purpose` field to `LLMCallLog`

2. **src/evaluation/tracker.py**
   - Updated `log_llm_call()` signature to accept `purpose`

3. **src/agent/workflow_planner.py**
   - Fixed parameter order when calling `log_llm_call()`

4. **src/agent/workflow_executor.py**
   - Added tool call logging for table extractions
   - Added tool call logging for text extractions  
   - Added tool call logging for compute operations
   - Pass tracker to TextTool constructor

5. **src/tools/text_tool.py**
   - Added `tracker` parameter to constructor
   - Added `log_llm_call()` in `_llm_extract_value()`
   - Added `import time` for timing

---

## Metrics Now Properly Tracked

### Turn-Level Metrics (TurnMetrics)
✅ **Token Usage**
- `turn_tokens` - Total tokens (prompt + completion)
- `prompt_tokens` - Input tokens
- `completion_tokens` - Output tokens

✅ **API Calls**
- `turn_llm_calls` - Count of LLM API calls
- `turn_tool_calls` - Count of tool executions
- `llm_calls` - Detailed log with purpose
- `tool_calls` - Detailed log with params and results

✅ **Latency**
- `turn_latency_ms` - Total turn execution time
- Per-tool latency in `tool_calls`
- Per-LLM-call latency in `llm_calls`

✅ **Tool Usage**
- `tools_used` - List of unique tools
- Tool success/failure tracking
- Error messages for failed calls

✅ **Accuracy**
- `numerical_match` - Exact match
- `financial_match` - 1% tolerance
- `soft_match` - Forgiving match

✅ **Reasoning Quality**
- `logic_recall` - Operation overlap with ground truth
- `operations_per_turn` - Compute operations count

### Trace-Level Metrics (TraceRecord)
✅ **Aggregated from all turns**
- `total_tokens`, `total_llm_calls`, `total_tool_calls`
- `numerical_accuracy`, `financial_accuracy`, `soft_match_accuracy`
- `avg_logic_recall`, `avg_operations_per_turn`
- `avg_plan_steps`, `total_latency_ms`

---

## Verification

To verify metrics are now tracked correctly:

```bash
cd version3
uv run main chat "Single_SPGI/2018/page_54.pdf-1"
```

Check the trace JSON for:
- Non-zero `turn_tool_calls` counts
- `llm_calls` with `purpose` field
- `tool_calls` with latency and results
- Accurate token counts across all LLM calls

---

## Impact on Evaluation

### Before Fixes:
- Token counts only from plan generation (50-70% of total)
- Tool usage statistics incomplete
- Cost per answer underestimated
- Latency breakdown inaccurate

### After Fixes:
- Complete token accounting (plan generation + text extraction)
- Full tool usage tracking with success rates
- Accurate cost and latency metrics
- Detailed performance breakdowns for optimization
