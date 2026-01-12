# FinQA Assistant - ConvFinQA Solution

This solution implements a **Modular Planner-Executor** architecture for multi-turn financial question answering on the ConvFinQA dataset. The system decouples semantic understanding (LLM-based planning) from mathematical execution (deterministic Python), achieving:

- **Zero Arithmetic Errors**: All calculations performed by symbolic Python engine
- **Full Auditability**: Complete trace from answer to source document
- **Cost Efficiency**: Optimized token usage via structured planning
- **Production-Ready**: Type-safe schemas, comprehensive testing, error handling

The architecture features:
- **Workflow Planner**: Converts natural language to structured JSON execution plans
- **Workflow Validator**: Ensures logical consistency before execution
- **Workflow Executor**: Deterministically executes plans with conversation memory
- **Result Verifier**: Post-execution semantic validation
- **Specialized Tools**: Fuzzy table extraction + LLM-guided text extraction

**For detailed implementation, architecture rationale, evaluation results, and future work, please see [REPORT.md](REPORT.md).**

---

## Quick Start

### Prerequisites
- Python 3.12+
- [UV environment manager](https://docs.astral.sh/uv/getting-started/installation/)
- OpenAI API key

### Setup
1. Clone this repository
2. Install dependencies using UV:

```bash
# Install UV package manager
brew install uv

# Install all dependencies
uv sync
```

3. Set up your OpenAI API key:

**Option 1: Environment Variable (Recommended)**
```bash
# Set for current session
export OPENAI_API_KEY="your-api-key-here"

**Option 2: .env File**
```bash
# Create a .env file in the project root (version3/)
echo 'OPENAI_API_KEY=your-api-key-here' > .env

# The system will automatically load it from .env
```

---

## Usage

### Interactive Chat Interface

Test a single conversation from the ConvFinQA dataset:

```bash
# Basic usage - chat with a specific conversation
uv run python src/main.py chat <example_id>

# Example with default settings
uv run python src/main.py chat Single_JKHY/2009/page_28.pdf-3

# With custom model
uv run python src/main.py chat Single_JKHY/2009/page_28.pdf-3 --model gpt-5-mini

# Enable validator and verifier (validation + result verification)
uv run python src/main.py chat Single_JKHY/2009/page_28.pdf-3 --validator --verifier

# Disable verifier (faster, but less accuracy checking)
uv run python src/main.py chat Single_JKHY/2009/page_28.pdf-3 --no-verifier
```

**Available Options:**
- `--model`: Model name (default: `gpt-5-mini`)
- `--validator` / `--no-validator`: Enable/disable plan validation (default: disabled)
- `--verifier` / `--no-verifier`: Enable/disable result verification (default: enabled)

**Output includes:**
- Document context (tables, pre-text, post-text)
- Multi-turn conversation with generated plans
- Step-by-step execution traces
- Accuracy indicators (numerical match, financial match, soft match)
- Per-turn metrics (tokens, execution time)

### Batch Evaluation

Evaluate on multiple conversations with comprehensive metrics:

```bash
# Evaluate on N random conversations
uv run python src/main.py evaluate --records <N>

# Example: Test 20 conversations with default model
uv run python src/main.py evaluate --records 20

# With custom model and validation/verification settings
uv run python src/main.py evaluate --records 10 --model gpt-5-mini --validator --verifier

# Fast evaluation (no validator, no verifier)
uv run python src/main.py evaluate --records 5 --model gpt-5-mini --no-validator --no-verifier
```

**Available Options:**
- `--records`: Number of conversations to test (default: 20)
- `--model`: Model name (default: `gpt-5-mini`)
- `--validator` / `--no-validator`: Enable/disable plan validation (default: disabled)
- `--verifier` / `--no-verifier`: Enable/disable result verification (default: enabled)

**Output Structure:**

Results saved to `batch_test_results_<model>_<timestamp>/`
- `results.json`: Per-turn detailed results with plans, traces, and accuracy
- `aggregate_metrics.json`: Overall metrics (accuracy, latency, tokens)
- `metadata.json`: Evaluation configuration

## Documentation

**For comprehensive documentation, please see [REPORT.md](REPORT.md)** which includes:
- Architecture rationale and design decisions
- Detailed component implementation
- Evaluation methodology and results
- Error analysis and future work

Additional resources:
- **[dataset.md](dataset.md)**: ConvFinQA dataset format and structure
- **[tests/README.md](tests/README.md)**: Testing strategy and coverage

---
See [REPORT.md](REPORT.md) for detailed error analysis and qualitative results.

---

## License & Contact

**Author**: Varsha Venkatesh 2025

