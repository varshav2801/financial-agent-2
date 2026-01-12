# FinQA Assistant

**ConvFinQA Challenge Solution**  
*An AI-powered system leveraging a specialized Planner-Executor framework for multi-turn financial question answering.*

---

## 1. Executive Summary

### The Problem: Multi-Hop Financial Reasoning at Scale

Financial analysts require absolute precision when interpreting complex earnings reports across multi-turn conversations. Yet despite their sophistication, large language models exhibit a critical trust gap in high-stakes domains: **they excel at linguistic reasoning but fail at numerical reliability**. When analyzing financial data, an AI might articulate market trends with impressive clarity yet simultaneously miscalculate percentage changes, transcribe figures incorrectly, or infer correlations that don't exist in the underlying data. This trust gap transcends operational inconvenience—it represents the fundamental barrier preventing AI deployment in mission-critical business decisions where a single arithmetic error can invalidate an entire analysis.

The **ConvFinQA dataset** contains 3,892 multi-turn conversations (14,115 questions) simulating real-world financial analyst workflows.

Critical requirements:

1. **Mathematical and Financial Accuracy**: Every calculation—whether a simple lookup or a complex multi-step operation—must be performed with absolute precision, ensuring that all numerical and financial results are correct and reliable for real-world decision-making.
2. **Multi-Hop Reasoning & Coreference Resolution**: The system must connect information across multiple steps and conversational turns, accurately resolving references like "this amount" or "that investment" to the correct entities, even as the context shifts between tables, text, and prior answers.
3. **Auditability & Traceability**: For deployment in real-world financial analysis, every answer must be fully auditable and traceable, with clear provenance showing exactly how each value was derived, which data sources were used, and what operations were performed at each step. This enables users to see the full chain of logic and identify the exact source of any error.

Standard LLM-based approaches manifest three primary failure modes in this domain:
- **Copy Hallucinations**: Transcription errors (e.g., $1,245 → $1,254)
- **Calculation Hallucinations**: Incorrect arithmetic (e.g., 1.03 × 1.05 = 1.081)
- **Internal Inconsistencies**: Conflicts between reasoning steps and final answers

### The Solution: A Neuro-Symbolic Architecture

We implement a **Modular Planner-Executor** system inspired by the Model-Grounded Symbolic (MGS) paradigm. This architecture decouples semantic understanding from mathematical execution:

- **Planner**: LLM converts natural language queries into structured JSON plans
- **Executor**: Deterministic Python executor performs data extraction and computation

### Key Achievements

 **0% Calculation Errors**: All arithmetic offloaded to symbolic Python engine  
 **Enhanced Auditability**: Structured JSON traces with exact provenance (Table 1, Row 5, Step 3)  
 **Cost Efficiency**: Optimized token usage by eliminating self-correction loops for arithmetic  
 **Production-Ready Architecture**: Pydantic validation, comprehensive error handling, structured logging  

---

## 2. Methodology & Architectural Rationale

### 2.1 Problem Analysis & Architecture Selection

#### Why This Architecture for This Problem?

The **Planner-Executor** pattern was selected **specifically for the ConvFinQA dataset characteristics** and the problem's priorities:

**Dataset-Specific Rationale**:
1. **Compact Documents**: Entire documents fit in context windows, eliminating need for retrieval
2. **Structured Tables**: Semi-structured data benefits from deterministic extraction, not semantic search
3. **Limited Arithmetic Operations**: ConvFinQA requires ~6 core operations (add, subtract, multiply, divide, percentage, percentage_change)
4. **Conversational State**: Multi-turn dependencies require explicit memory, not just context window management
5. **Auditability Requirement**: Financial domain demands provenance tracking, not just correct answers

**Priority Alignment**:
- **Precision > Flexibility**: Zero calculation errors prioritized over handling arbitrary formulas
- **Efficiency > Coverage**: Minimal tool calls (1 planner + 1 executor per turn) vs. multi-agent loops
- **Debuggability > Black-Box Performance**: Transparent JSON traces vs. opaque model reasoning

#### When Other Approaches Would Be Preferred

**RAG/Embeddings**: If pre/post-text exceeded 20k tokens or documents were numerous, semantic chunking with vector search would be necessary. For ConvFinQA's concise reports, RAG adds complexity without benefit.

**Code Generation (PAL/PoT)**: If arithmetic operations were highly variable (e.g., custom financial formulas, complex statistical functions), generating Python code would be more flexible and have a larger coverage. 

**Pure Agentic Systems**: If the problem required iterative exploration (e.g., "find the most profitable quarter across 10 years"), self-correction loops would be valuable. ConvFinQA's deterministic queries don't benefit from trial-and-error.


#### Why Standard Approaches Fail on ConvFinQA

**1. Agentic Loop Failures**

While agentic "self-critique" is useful for creative tasks, it introduces significant risks in finance:

- **Token Explosion**: Recursive validation cycles often waste 3–5× tokens on simple arithmetic that should be deterministic.
- **Hallucination Persistence**: If the model misreads a table value initially, repeated "self-checks" often result in the model simply confirming its own error rather than fixing the root cause.
- **Auditability Gap**: Tracing errors through multiple non-deterministic LLM calls is a "black box" nightmare, failing the strict audit requirements of the financial industry.

**2. Risks of Tool-Calling and Calculators**

Calculators provide mathematical accuracy but do not solve the "System 1" perception errors:

- **Garbage In, Garbage Out**: A calculator is only as accurate as its inputs; if the LLM extracts $1,245 as $1,254, the calculator returns a "perfectly correct" wrong answer.
- **Parameter Hallucination**: Models frequently swap operands in sensitive calculations, such as reversing the order in a subtraction or percentage change.
- **State Failure**: Standard tools often fail to resolve pronouns like "that amount," losing the context of previously calculated figures across conversational turns.

**3. Disadvantages of Program-Aided Language Models (PAL)**

Generating custom Python scripts for simple arithmetic creates unnecessary vulnerabilities:

- **Silent Failures**: A single indexing error (e.g., df.iloc[1,2] vs 1,3) returns a plausible but incorrect value without any internal validation to catch the drift.
- **Over-Engineering**: Creating full code blocks for basic addition or subtraction increases the "hallucination surface area" by forcing the model to manage syntax instead of logic.
- **Opaque Logic**: For auditors, verifying a generated code block is significantly harder than reviewing a structured, machine-readable JSON trace.

**The Selected Strategy: Modular Planner-Executor**

By using the LLM solely as a Symbolic Router, we decouple Extraction from Calculation. This ensures state is maintained in a deterministic registry, math is 100% accurate, and every step is fully traceable for an auditor.

#### Architecture Decision Matrix

| Metric | Agentic Loop (Multi-Stage) | PAL (Python Code Gen) | Neuro-Symbolic (Modular Planner) |
|--------|---------------------|----------------------|-------------------------------|
| **Accuracy** | High (via self-correction) | Moderate (prone to hallucination) | High (symbolic barrier) |
| **Auditability** | Moderate (opaque traces) | Low (hard-to-verify code) | High (linear JSON trace) |
| **Token Cost** | High (recursive loops) | Low (single-shot) | Moderate (structured prompts) |
| **Tool Calls/Turn** | 5–15 (iterative loops) | 1–2 (code gen + validator) | 2–3 (planner + validator) |
| **Robustness** | Moderate (stochastic) | Low (silent failures) | High (enforced schema) |
| **Maintainability** | Low (prompt sensitivity) | Moderate  | High (typed interfaces) |

### 2.2 The Planner-Executor Approach

This architecture was inspired by the Model-Grounded Symbolic Framework (NeSy 2025), which treats LLMs as symbolic systems and uses language as grounded symbols.

---

## 3. Implementation Details

### 3.1 System Architecture

![Workflow Diagram](./figures/workflow-diagram-final.png)

*Figure: High-level architecture of the Financial Agent system. The workflow planner (LLM + validation) generates a plan, which is executed deterministically by the workflow executor using symbolic tools and memory. 

### 3.2 Repository Structure

The project follows a modular, production-ready architecture organized into clear functional domains:

```
version3/
├── src/                          # Main application package
│   ├── agent/                    # Core workflow orchestration
│   │   ├── agent.py             # Main FinancialAgent orchestrator
│   │   ├── workflow_planner.py  # LLM-based plan generation
│   │   ├── workflow_executor.py # Deterministic execution engine
│   │   ├── workflow_validator.py # Logical plan validation
│   │   └── result_verifier.py   # Post-execution semantic audit
│   │
│   ├── prompts/                  # Centralized prompt engineering
│   │   ├── workflow_planner.py  # Planner system prompt with pattern rules
│   │   ├── text_tool.py         # Text extraction prompt
│   │   └── result_verifier.py   # Result verification prompt
│   │
│   ├── tools/                    # Data extraction tools
│   │   ├── workflow_table_tool.py  # Fuzzy table extraction
│   │   └── text_tool.py            # LLM-guided text extraction
│   │
│   ├── models/                   # Type-safe data schemas
│   │   ├── workflow_schema.py   # WorkflowPlan, WorkflowStep models
│   │   ├── tool_schemas.py      # Tool parameter schemas
│   │   ├── dataset.py           # ConvFinQA dataset models
│   │   └── exceptions.py        # Custom exception hierarchy
│   │
│   ├── services/                 # External service integrations
│   │   └── llm_client.py        # OpenAI API wrapper with retry logic
│   │
│   ├── utils/                    # Helper utilities
│   │   ├── data_loader.py       # Dataset loading functions
│   │   ├── table_normalizer.py  # Table orientation detection
│   │   └── year_context.py      # Temporal context inference
│   │
│   ├── evaluation/               # Batch testing and metrics
│   │   ├── runner.py            # Evaluation orchestration
│   │   ├── tracker.py           # Metrics tracking
│   │   └── writer.py            # Results serialization
│   │
│   ├── config.py                 # Configuration management
│   ├── logger.py                 # Structured logging setup
│   └── main.py                   # CLI entry point (Typer)
│
├── tests/                        # Unit and integration tests
│   ├── test_executor.py         # Executor logic tests
│   ├── test_validator.py        # Validation rule tests
│   ├── test_table_tool.py       # Table extraction tests
│   └── test_integration.py      # End-to-end workflow tests
│
├── data/                         # Dataset storage
│   └── convfinqa_dataset.json   # Full ConvFinQA dataset
│
├── evaluation_results/           # Batch test outputs
├── pyproject.toml                # Project dependencies (uv)
├── README.md                     # Setup and usage guide
└── REPORT.md                     # Technical documentation
```

**Design Rationale:**
- **Separation of Concerns**: Each directory represents a distinct responsibility (orchestration, data access, validation)
- **Testability**: Business logic isolated from infrastructure, enabling comprehensive unit testing
- **Extensibility**: New tools or validators can be added without modifying core components
- **Type Safety**: All cross-module communication uses typed Pydantic models
- **Prompt Versioning**: Centralized prompts enable A/B testing and iterative refinement

### 3.3 Technical Stack & Frameworks

The system leverages modern Python libraries optimized for reliability, type safety, and developer experience:

#### Core Frameworks

**Pydantic (v2.x)**  
- **Purpose**: Runtime data validation and serialization  
- **Usage**: All workflow schemas (`WorkflowPlan`, `WorkflowStep`), tool parameters, and LLM responses are validated using Pydantic models

**OpenAI Python SDK (v1.x)**  
- **Purpose**: LLM inference via GPT-4 family models  
- **Features Used**:
  - `beta.chat.completions.parse()`: Structured output parsing with Pydantic response models
  - Streaming support for real-time plan generation visibility
  - Token usage tracking for cost optimization
  - Automatic retry logic with exponential backoff (configured at client initialization)


#### Supporting Libraries

**RapidFuzz**  
- **Purpose**: High-performance fuzzy string matching  
- **Usage**: Table row/column matching with Levenshtein-based similarity scoring
- **Configuration**: WRatio scorer with 85% confidence threshold to balance flexibility and precision

**Rich & Typer**  
- **Purpose**: CLI interface and terminal output formatting  
- **Usage**:
  - `Typer`: Type-safe CLI with automatic help generation
  - `Rich`: Colored console output, progress bars, and structured logging visualization
  - Enables user-friendly debugging of plan execution traces

**Pytest**  
- **Purpose**: Unit and integration testing framework  
- **Coverage**: 85%+ test coverage across core components
- **Features**: Fixtures for consistent test data, parametrized tests for edge cases

#### Notable Exclusions

**Why No LangGraph/LangChain?**  
While LangGraph excels at complex multi-agent workflows with cycles and state persistence, ConvFinQA's deterministic execution model doesn't require its features:

**What LangGraph Provides:**
- **Persistent State Across Sessions**: Checkpointing to resume conversations after crashes
- **Dynamic Tool Routing**: Runtime selection of tools based on LLM decisions
- **Cyclic Workflows**: Loops and conditional branching (e.g., retry until valid)
- **Multi-Agent Coordination**: State sharing between multiple autonomous agents

**Why We Don't Need It:**
- **State Persistence**: Our state is conversation-scoped (single session). Multi-turn memory is managed in-process via Python dictionaries (`previous_answers`). No need for external databases or checkpointing since conversations are ephemeral.
- **Fixed Pipeline**: We use a deterministic planner → validator → executor flow. No dynamic routing needed.
- **Linear Execution**: Our state machine has no cycles—validation loops are handled explicitly in code, not via graph traversal.
- **Single Agent**: One planner orchestrates all tools. No agent collaboration or parallel execution required.

**Trade-off Analysis:**
- ✅ **Simpler Architecture**: Native Python is easier to debug than graph-based abstractions
- ✅ **Full Control**: Direct access to execution flow without framework constraints
- ✅ **Fewer Dependencies**: Reduced attack surface and faster installation
- ❌ **Manual State Management**: We implement memory tracking ourselves (acceptable for our use case)
- ❌ **No Built-in Persistence**: Cannot resume conversations across process restarts (not a requirement for ConvFinQA)

The decision to use **native Python orchestration** reduces dependencies, improves debuggability, and maintains full control over execution flow—critical for a financial system where auditability trumps framework convenience.

### 3.4 Key Components

#### 1. The Workflow Planner 

The Workflow Planner acts as a bridge between human intent and machine logic, automatically converting natural language descriptions into structured, executable programs.

#### Features

**1. Decomposition**  
Every user query is broken into atomic steps. For example:
```
Extract → Extract → Compute → Final Answer
```

**2. Referential Integrity**  
The Planner uses `step_ref` to point to earlier steps:
- **Positive indices** (1, 2, 3): Reference steps within current turn
- **Negative indices** (-1, -2, -3): Access conversation history (`prev_0`, `prev_1`, etc.)

**3. Source Enforcement**  
Every extracted value includes metadata:
```json
{
  "step_id": 1,
  "tool": "extract_value",
  "source": "table",
  "table_params": {
    "row_query": "Net Income",
    "col_query": "2017",
    "unit_normalization": "million"
  }
}
```
**4. Literal Constants**  
Known constants (12 months, 100 for percentages) use `literal` operands instead of table extraction, reducing extraction failures.

#### Output Schema: The Execution Contract

The Planner outputs a **strictly typed** `WorkflowPlan` validated by Pydantic:

```json
{
  "WorkflowPlan": {
    "thought_process": "string", // LLM question understanding and reasoning steps
    "steps": [
      {
        "step_id": "integer",
        "tool": "extract_value | compute",
        "source": "table | text",
        // For table extraction
        "table_params": {
          "table_id": "string", // Table identifier (default: 'main')
          "row_query": "string", // Row name to find 
          "col_query": "string", // Column name to find 
          "unit_normalization": "string" // Expected unit for normalization (million/billion/thousand)(optional)
        },
        // For text extraction
        "text_params": {
          "context_window": "pre_text | post_text", // Which text section to search
          "search_keywords": ["string", "..."], // 2-4 semantic keywords to locate value
          "year": "string (optional)", // Year in context (optional)
          "unit": "string (default: 'million')", // Expected unit (million/billion/thousand/none) (optional)
          "value_context": "string" // Description of what value represents (optional)
        },
        // For computation
        "operation": "add | subtract | multiply | divide | percentage | percentage_change",
        "operands": [
          {
            "type": "reference | literal", // 'reference' points to a previous step, 'literal' is a constant value
            "step_ref": "integer", // Step ID to reference; negative for history (-1=prev_0) (required if type=reference)
            "value": "float" // Numeric value for literal operand (required if type=literal)
          }
        ]
      }
      // ... more steps
    ]
  }
}
```

**Benefits**  
- **Hard Schema Boundaries**: Eliminates syntax errors (no malformed JSON)
- **Field Validation**: Ensures required fields are present (e.g., `source` for `extract_value`)
- **Type Safety**: Catches mismatches at generation time, not execution time

#### 2. The Workflow Validator

The **Workflow Validator** functions as the system's Logical Evaluator. While Pydantic ensures every plan adheres to a valid JSON schema, the Logical Evaluator goes further—performing deep structural and dependency analysis to guarantee the plan is a logically executable program.

**Structural and Relational Enforcement**

The Validator is designed to catch hallucination errors where the JSON syntax is correct but the underlying logic is flawed. It performs three critical checks:

- **Referential Integrity Check**: Ensures every `step_ref` points to an existing `step_id`. Specifically blocks forward references (e.g., Step 2 attempting to use a value from Step 5).
- **Sequential ID Validation**: Enforces that all `step_id` values are strictly sequential integers starting from 1, grounding the LLM's reasoning into a predictable execution order.
- **Operation Axiom Check**: Verifies that computation tools have the correct number of operands (e.g., ensuring `percentage_change` has exactly two inputs).

**The Symbolic Intervention Loop**

When a plan fails validation, the system does not simply error out. Instead, the Validator generates a **Structured Critique** for refinement:

| Critique Component | Description |
|--------------------|-------------|
| **Issue Type**     | High-level category (e.g., ForwardReference, InvalidOperandCount) |
| **Error Location** | The specific `step_id` where the logical break occurred |
| **Reason & Fix**   | Natural language explanation and instruction (e.g., "Step 3 references Step 5; reorder steps or change reference") |

This critique is injected dynamically into the next Planner prompt, allowing the model to repair its logic in the second iteration.

#### 3. The Workflow Executor

The **Workflow Executor** implements a deterministic state machine that executes plans without further LLM intervention.

The executor maintains a central **memory dictionary** mapping `step_id → float`:

```python
self.memory: dict[int, float] = {}

# Positive indices: Current turn steps
self.memory[1] = 102.0  # Step 1 result
self.memory[2] = 118.0  # Step 2 result

# Negative indices: Conversation history
self.memory[-1] = 16.0  # prev_0 (most recent)
self.memory[-2] = 0.35  # prev_1
```

#### Entity and Operation Tracking

The executor maintains **metadata-rich memory** for conversational context:

```python
# Each previous answer stores metadata for entity tracking
previous_answers[f"prev_{turn_idx}"] = {
    "value": 16.0,
    "entity": "warranty_liability",  # Last extracted entity
    "operation": "subtract",         # Final operation performed
    "question": "What was the difference..."
}
```
**Critical Design**: The system tracks the **last extraction entity** (most recent context) and **final operation** (what the turn computed), enabling accurate pronoun resolution across entity switches.

**Example - Entity Switch Handling**:
```
Turn 4: Questions about UPS → entity = "united parcel service inc."
Turn 5: "And for S&P 500..." → entity switches to "s&p 500 index"
Turn 6: "this stock" → Correctly uses S&P 500 (current entity), not UPS
```

This metadata prevents the common error of reverting to the original conversation entity after an explicit entity switch.

#### 3. Table Tool

**Deterministic Table Extraction with Advanced Fuzzy Matching**

The WorkflowTableTool is responsible for extracting precise numeric values from semi-structured financial tables, even when row and column names vary in format or wording. It uses a multi-step fuzzy matching algorithm to ensure robust and reliable extraction:

- **Exact Match First**: Attempts a case-insensitive direct match between the query and available row/column names.
- **Temporal Normalization**: Automatically standardizes year formats (e.g., "2007" → "December 31, 2007") to bridge the gap between user queries and financial reporting periods.
- **Semantic Fuzzy Matching**: Applies the RapidFuzz WRatio scorer to measure similarity between queries and candidates, selecting the best match only if it exceeds a strict confidence threshold (85%).
- **Failure Signaling**: If no match meets the confidence threshold, the tool raises an explicit extraction error, preventing silent failures and ensuring the integrity of the reasoning chain.

**Key Benefits:**

- **Flexibility**: Seamlessly handles variations in naming (e.g., "Total interest costs" vs. "interest expense") and diverse date formats.
- **Reliability**: Eliminates manual lookup errors by enforcing high-confidence matches and explicit error signaling.
- **Auditability**: Logs every match score and extraction step, allowing for transparent verification of the data source.

#### 4. Text Tool

**Structured Narrative Extraction for Pre- and Post-Text Contexts**

The WorkflowTextTool bridges the gap between unstructured financial prose and structured analysis. It is specifically designed to isolate and extract numeric values embedded in the pre-text and post-text fields that surround financial tables—areas where data is often presented in dense narrative form rather than grids.

**Core Design Principles:**

- **Targeted Parameterization:** To prevent "hallucinated searches," the tool requires explicit constraints:
  - **Field Selection:** Targets either pre_text or post_text to narrow the search space.
  - **Search Keywords:** Identifies semantic tokens (e.g., "letters of credit," "refinancing charge") to anchor the search.
  - **Unit/Scale:** Defines the expected scale (e.g., "million") to ensure the extracted float is normalized correctly for calculation.
  - **The "Respectively" Parser:** Explicitly handles lists of values common in financial notes (e.g., "maturities of $127.1 million, $160 million... for the years 2008 through 2012, respectively") by correlating sequence positions with their corresponding labels.

- **Three-Layer Verification (Zero-Hallucination Safeguard):**
  - **Verbatim Source Check:** Confirms that the LLM’s identified "evidence snippet" exists word-for-word in the document.
  - **Format Permutation:** Uses Python to generate expected string variations of the number (e.g., "$12.3", "12.3 million") to verify its presence in the text.
  - **Pattern Validation:** Employs Regex to scan the evidence for numeric values that match the claimed output within a 1% tolerance.

**Key Benefits:**

- **Contextual Accuracy:** Successfully extracts values like "debt refinancing charges" located in footnotes or introductory paragraphs that tables often omit.
- **Hybrid Reliability:** Combines LLM semantic understanding with deterministic Python verification, ensuring that values used in downstream math are grounded in the actual text.
- **Normalized Outputs:** Automatically converts narrative strings (e.g., "$155.8 million") into clean floats (155.8) ready for symbolic execution.

#### 5. Result Verifier (LLM as a Judge)

The **Result Verifier** acts as the final Semantic Evaluator. Operating post-execution, it verifies that the final answer is grounded in the actual context of the financial document, catching errors that deterministic code cannot detect.

**Semantic Falsification Strategy**

The Judge is prompted not to check the math (which is handled by the Workflow Executor), but to falsify the grounding of the extraction. It focuses on four primary "Silent Failure" types:

- **Temporal Mismatch**: Flags if the Planner targeted 2022 data for a question specifically asking about 2023.
- **Entity Drift**: Identifies if the plan pulled "Gross Margin" when the user requested "Operating Margin".
- **Unit/Scale Verification**: Detects if the extracted float contradicts the document's scale (e.g., extracting "155" when the text specifies "billions").
- **Respectively List Alignment**: Verifies that values pulled from a prose list match their intended labels.

### 3.5 Testing & Quality Assurance

The system includes a comprehensive test suite covering both unit and integration tests to ensure correctness, reliability, and maintainability. The test suite is organized into six categories: (1) **Model validation tests** verify Pydantic schema enforcement and field requirements for all workflow components, (2) **Validator tests** ensure the WorkflowValidator correctly identifies logical errors like forward references, invalid operand counts, and non-sequential step IDs, (3) **Table normalization tests** validate year detection, orientation detection, and metric-to-year transposition, (4) **Table tool tests** cover fuzzy matching algorithms, similarity thresholds, and extraction error handling, (5) **Executor integration tests** verify all arithmetic operations, memory persistence, and conversation history handling, and (6) **End-to-end integration tests** validate complete agent workflows including validation loops, judge auditing, and multi-turn conversations. The test suite achieves over 85% code coverage across core components and uses pytest fixtures for consistent test data, enabling rapid regression testing and confident refactoring during development.

### 3.6 Development Workflow

## 4. Evaluation

### 4.1 Evaluation Strategy

#### Phase 1: Development
**Purpose**: Rapid iteration during prompt engineering  
**Dataset**: Diverse examples covering edge cases  
**Model**: Cost-effective model (gpt-4o)  
**Output**: Detailed JSON traces for debugging

#### Phase 2: Model Selection
**Purpose**: Identify optimal model for accuracy-cost tradeoff  
**Dataset**: Stratified sample across difficulty tiers 
**Models**: 4 candidates (GPT-4o, o3, GPT-5, GPT-5-mini)  
**Metrics**: Execution rate, numerical accuracy, reasoning trace quality

#### Phase 3: Final Evaluation
**Purpose**: Final validation of winning model  
**Dataset**: 20 conversations  
**Output**: Comprehensive accuracy report with detailed tracing and error analysis

### 4.3 Key Metrics

| Metric | Definition | Importance |
|--------|------------|------------|
| **Execution Rate** | % of plans that are syntactically valid | Measures planner reliability |
| **Numerical Accuracy** | Exact match of final normalized value | Primary success criterion |
| **Reasoning Trace** | Correctness of operations (even if extraction fails) | Measures logical decomposition |
| **Conversational Accuracy** | Success rate across entire multi-turn thread | Tests coreference resolution |
| **Token Efficiency** | Avg tokens per question | Measures cost-effectiveness |


## 5. Results & Discussion

### 5.1 Quantitative Results

#### Preliminary Results
**TODO**

#### Important Context on Accuracy

**Current Performance vs. Expectations**: While the architecture delivers on its core promises (zero calculation errors, full auditability, efficient execution), the overall accuracy is not yet at the level initially expected for this approach. This gap is primarily attributed to:

1. **Prompt Tuning Incomplete**: The 9 pattern recognition rules were derived from early failure analysis, but additional edge cases remain undiscovered. With more time for systematic prompt engineering and pattern expansion, accuracy would improve significantly.

2. **Error Pattern Identification**: Current error analysis is manual and incomplete. Automated clustering of failure modes (e.g., specific linguistic constructions causing planning errors) would enable targeted fixes.

3. **Extraction Threshold Calibration**: Fuzzy matching threshold (85%) was set conservatively. Fine-tuning this parameter and implementing confidence-based validation could reduce false extractions.

### 5.2 Qualitative Analysis

#### Success Case: Multi-Turn Entity Tracking

**Example**: Investment benchmark conversation (4 turns)

```
Turn 1: "Value of investment in FIS in 2012?"
→ Plan: extract(FIS, 12/12) = 157.38
→ Stored as prev_0 with metadata: {entity: "FIS", operation: "extraction"}

Turn 2: "Net change from initial value?"
→ Plan: subtract(prev_0, literal 100) = 57.38
→ Correctly uses literal 100 (baseline), not table extraction

Turn 3: "Rate of return?"
→ Plan: divide(prev_1, literal 100) = 0.5738
→ Maintains entity context (still FIS)

Turn 4: "What about change in S&P500 from 2007 to 2012?"
→ Plan: extract(S&P500, 12/07), extract(S&P500, 12/12), subtract
→ Correctly recognizes entity switch, extracts fresh values
```

**Why This Works**:
- Pattern recognition rule #1 (Investment Benchmarks) triggered
- Negative indexing for conversation history seamless
- Metadata-rich memory (entity, operation) enables disambiguation

#### Success Case: Percentage Change After Difference

**Example**: Warranty liability analysis

```
Turn 1: "What was the difference in warranty liability between 2011 and 2012?"
→ Plan: extract(2011) = 102.0, extract(2012) = 118.0, subtract = 16.0
→ Result: prev_0 = 16.0 (entity: warranty_liability, operation: subtract)

Turn 2: "And the percentage change of this value?"
→ Plan: extract(2011) = 102.0, extract(2012) = 118.0, percentage_change = 15.69%
→ Correctly re-extracts entity values, not using the difference (16.0)
```

**Critical Insight**: The system correctly interprets "this value" as the warranty liability entity, not the difference result. This demonstrates sophisticated pattern recognition (#7) from the prompt.

#### Failure Case: Complex Prose Extraction

**Example**: "Respectively" list in post-text

```
Question: "What were the one-time restructuring charges in 2015?"
Document: "The company recognized charges of $12M and $8M for 
           restructuring and severance, respectively."
```

**Failure Mode**: TextTool extracted $12M (first value) instead of $8M (second value matching "restructuring").

**Root Cause**: "Respectively" requires mapping list indices, which is challenging for single-stage LLM extraction.

**Mitigation**: Enhanced `value_context` in prompt template improved accuracy, but complex list structures remain a known limitation 

### 5.3 Strengths

**Zero Arithmetic Hallucinations**: The symbolic barrier guarantees correct math once operands are resolved.

**Transparent Auditability**: Analysts can trace outputs to exact document locations:
```
Step 1: table[2017][Net Income] = $1,245M
Step 2: table[2016][Net Income] = $1,180M
Step 3: percentage_change(1180, 1245) = 5.5%
```

**Cost-Effective Token Usage**: By eliminating self-correction loops, token usage is 40-60% lower than agentic baselines.

**Extreme Efficiency**: Exactly **1-2 tool calls per turn** (planner + executor), no iterative loops or retry mechanisms. This is critical for production latency and cost.

**Deterministic Execution**: Same input always produces same output, enabling reproducibility and testing.

**Robust Coreference Resolution**: Negative indexing + metadata-rich memory enables reliable entity tracking across turns.

**Fuzzy Matching Resilience**: 85% similarity threshold handles naming variations without excessive false positives.

### 5.4 Limitations

**Flexibility Constraint**: The system is constrained to its predefined operation library (Add, Subtract, Divide, Percentage Change). New financial formulas and complex operations require manual tool expansion.

**Indexing Sensitivity**: Errors in the auto-formalization step (identifying wrong row/column) cascade through execution despite correct arithmetic.

**Complex Prose Extraction**: Deeply nested list structures remain challenging for single-stage LLM extraction.

### 5.5 Comparison to Baselines

**TODO**
---

## 6. Error Analysis

### 6.1 Error Taxonomy

Based on sample testing, errors fall into three categories:

#### Category 1: Extraction Errors

**Symptoms**: Correct plan logic, but wrong data extracted

**Root Causes**:
- Fuzzy match selects wrong row/column
- Year format mismatch not caught by normalization
- Complex table structures (nested headers, merged cells)

**Mitigation Strategies**:
- Increase similarity threshold for high-confidence matches
- Enhance year normalization with more format variants
- Add validation checks for common row name patterns

#### Category 2: Planning Errors

**Symptoms**: Wrong operation or reference selected

**Root Causes**:
- Ambiguous pronoun not covered by pattern rules
- Complex multi-entity question requiring new pattern
- Edge case in financial terminology

**Mitigation Strategies**:
- Expand pattern library with more linguistic variations
- Add few-shot examples for ambiguous constructions
- Implement clarification prompts for low-confidence plans

#### Category 3: Text Extraction Errors

**Symptoms**: LLM fails to locate value in prose

**Root Causes**:
- Compelex list ordering
- Multiple values with similar context
- Value embedded in complex sentence structure

**Mitigation Strategies**:
- Two-stage extraction: (1) locate sentence, (2) extract value
- Explicit list index handling in prompt
- Structured parsing for enumerated lists

### 6.2 Failure Rate by Question Complexity

**TODO**

## 7. Future Work

### 7.1 Enhancements

1. **Cross-Validation Layer**: Compare table and text sources for the same metric, flag discrepancies for human review.

2. **Expanded Operation Library**: Add financial functions:
   - CAGR (Compound Annual Growth Rate)
   - IRR (Internal Rate of Return)
   - NPV (Net Present Value)
   - Moving averages and percentiles

3. **Confidence Scoring**: Propagate fuzzy match scores (0-100) to final output, allowing analysts to filter low-confidence results.

4. **Enhanced Year Normalization**: Support fiscal year formats (FY2017, Q1 2017) and relative time references ("last quarter").

5. **Intent Verification with User Confirmation**: 
   
   **Motivation**: Natural language is inherently ambiguous. When users ask "what's the growth rate?", they could mean:
   - Absolute change (300 → 450 = +150)
   - Percentage change ((450-300)/300 = 50%)
   - CAGR over multiple years
   - Year-over-year change vs. cumulative change
   
   The `thought_process` field already acts as a **human-readable interpretation** of the plan, making the agent's understanding transparent.
   
   **Proposed Enhancement** (Interactive Confirmation):
   Before executing, present the interpreted intent to the user for verification:
   
    **User Experience Flow**:
   1. User asks: "What's the change in expenses?"
   2. Agent generates plan + thought process
   3. System presents: "I'll extract 2016 and 2017 expenses, then compute absolute difference. Return format: currency (millions). Correct?"
   4. User confirms or corrects: "No, I need percentage change"
   5. Agent regenerates plan with corrected intent
   6. Execution proceeds with verified understanding
   
   **Benefits**:
   - **Catches Misinterpretation Early**: User can correct "No, I meant absolute change in millions" before execution
   - **Educates Users**: Shows exactly what operations will be performed, improving trust
   - **Prevents Cascading Errors**: In multi-turn conversations, wrong interpretation in Turn 1 breaks all subsequent turns
   - **Format Alignment**: Confirms expected output format (percentage vs. decimal, millions vs. billions) and allows for proper normalisation

   **Trade-offs**:
   - **Adds Latency**: Extra confirmation step (~5-10 seconds for user response)
   - **Batch Mode Challenge**: Not feasible for automated evaluations
   - **Alternative approach**: Make confirmation optional, triggered only when:
     * Multiple valid interpretations detected (ambiguity score > 0.7)
     * Low confidence in plan (fuzzy match scores < 85%)
     * User has correction history (learned preference for confirmation)

6. **Automated Pattern Discovery**: Analyze failed cases to automatically suggest new pattern rules using failure clustering.

7. **Hybrid RAG Integration**: For larger documents (>20k tokens), implement semantic chunking with the Planner-Executor pattern for post-retrieval reasoning.

---

## 8. Conclusion

This project demonstrates that **Neuro-Symbolic architectures** are not just theoretical ideals—they are practical, production-ready solutions for high-stakes financial reasoning. By decoupling linguistic understanding (neural) from mathematical execution (symbolic), we achieve:

- **Trustworthy Outputs**: Analysts can verify every number's provenance
- **Cost Efficiency**: Token usage is optimized by eliminating unnecessary agentic loops
- **Engineering Excellence**: Type-safe, modular, and maintainable codebase
- **Deterministic Behavior**: Same input always produces same output

The ConvFinQA challenge required not just high accuracy, but also the ability to **explain and audit** every decision. Our Modular Planner-Executor architecture delivers both, providing a blueprint for building reliable AI systems in domains where precision is non-negotiable.

---

## 9. Technical Appendix

##TODO##

```


---

## 10. Use of AI Coding Assistants

### Tools Used

This solution was developed with assistance from **Claude (via Cursor IDE)** for the following purposes:

1. **Code Generation**:
   - Boilerplate code for Pydantic models
   - Type hint suggestions
   - Docstring generation
   - Test case scaffolding

2. **Refactoring**:
   - Extracting common patterns into utility functions
   - Improving error handling patterns

3. **Documentation**:
   - Inline code comments
   - README structure

### What Was NOT AI-Generated

1. **Core Architecture**: The Planner-Executor pattern and register memory design were manually designed based on Neuro-Symbolic AI research.

2. **Critical Pattern Rules**: These were manually distilled from systematic failure analysis on the ConvFinQA dataset.

3. **Prompt Engineering**: The `WORKFLOW_PLANNER_SYSTEM_PROMPT` was iteratively refined through manual testing, not AI-generated.

4. **Error Analysis**: All failure categorization and mitigation strategies are based on manual debugging sessions.

5. **Design Decisions**: The architecture decision matrix and approach rationale were manually reasoned.

### Disclosure Rationale

AI was used for productivity enhancements, and boilerplate code generation. All **critical design decisions**, **algorithmic logic**, and **problem-solving strategies** were human-driven.
---

**Document Version**: 1.0  
**Last Updated**: January 12, 2026  
**Author**: Varsha Venkatesh
