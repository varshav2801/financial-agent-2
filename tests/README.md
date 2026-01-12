# Test Suite for Financial Agent

This directory contains comprehensive unit and integration tests for the Financial Agent system.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and shared test data
├── test_models.py           # Unit tests for Pydantic models
├── test_validator.py        # Unit tests for WorkflowValidator
├── test_table_normalizer.py # Unit tests for table normalization utilities
├── test_table_tool.py       # Unit tests for WorkflowTableTool
├── test_executor.py         # Integration tests for WorkflowExecutor

```

## Running Tests

### Install Test Dependencies

```bash
pip install -r tests/requirements-test.txt
```

Or with uv:

```bash
uv pip install -r tests/requirements-test.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Files

```bash
pytest tests/test_models.py
pytest tests/test_validator.py
```

### Run Specific Test Classes or Functions

```bash
pytest tests/test_models.py::TestOperandModel
pytest tests/test_validator.py::TestWorkflowValidator::test_forward_reference_fails
```

## Test Categories

### Unit Tests

#### `test_models.py`
Tests Pydantic model validation, field requirements, and type checking:
- Operand models (reference vs literal)
- ExtractTableParams and ExtractTextParams
- WorkflowStep and WorkflowPlan
- StepCritique and ValidationResult

#### `test_validator.py`
Tests WorkflowValidator logic for catching logical errors:
- Empty plan detection
- Forward reference detection
- Sequential step ID validation
- Operand count validation
- Invalid step reference detection

#### `test_table_normalizer.py`
Tests table normalization utilities:
- Year key detection
- Table orientation detection
- Metric-to-year transposition
- Numeric value extraction and filtering

#### `test_table_tool.py`
Tests WorkflowTableTool extraction:
- Exact match extraction
- Fuzzy matching (rows and columns)
- Year format normalization
- Error handling for missing data

### Integration Tests

#### `test_executor.py`
Tests WorkflowExecutor with complete plans:
- Simple extraction execution
- Multi-step computation plans
- All arithmetic operations (add, subtract, multiply, divide, percentage, percentage_change)
- Literal operands
- Conversation history references
- Memory persistence across steps


## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `sample_table`: Normalized financial table with 3 years of data
- `sample_document`: Complete document with table and text
- `simple_extraction_plan`: Basic single-step extraction plan
- `computation_plan`: Multi-step plan with computation
- `invalid_forward_reference_plan`: Plan with forward reference error

## Test Coverage

The test suite covers:

- **Models**: 95%+ coverage of Pydantic validation logic
- **Validator**: 90%+ coverage of validation checks
- **Table Tool**: 85%+ coverage including fuzzy matching
- **Executor**: 80%+ coverage of execution paths
- **Integration**: Key user workflows and error scenarios

## Writing New Tests

### Adding Unit Tests

1. Import required models and classes
2. Use pytest fixtures for shared data
3. Test one concept per test function
4. Use descriptive test names

Example:
```python
def test_valid_reference_operand(self):
    """Valid reference operand creation"""
    op = Operand(type="reference", step_ref=1)
    assert op.type == "reference"
    assert op.step_ref == 1
```


Example:
```python
@pytest.mark.asyncio
async def test_simple_query_workflow(self, sample_document, mock_llm_client):
    \"\"\"Test complete workflow for simple query\"\"\"
    agent = FinancialAgent(llm_client=mock_llm_client)
    result = await agent.run_turn(question, document, {})
    assert result.success
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r tests/requirements-test.txt
    pytest tests/ --cov=src --cov-report=xml
```

## Debugging Tests

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run with print statements:
```bash
pytest tests/ -s
```

### Run and stop at first failure:
```bash
pytest tests/ -x
```

### Run only failed tests from last run:
```bash
pytest tests/ --lf
```
