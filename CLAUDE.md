# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyGGB is a geometry problem-solving system that uses multimodal LLMs to generate geometric constructions from Chinese text descriptions. The system uses a custom Domain-Specific Language (DSL) for constructing geometry and a ReAct agent pattern for iterative problem-solving.

## Core Architecture

### 1. DSL Execution Pipeline

**Key Flow**: Problem Text → ReAct Agent → DSL Code → Construction → Validation

```
src/core/random_constr.py (Construction class)
├── parse_command() - Parses DSL lines into Command objects
│   ├── Evaluates math expressions (100*cos(45°))
│   ├── Handles literals, variables, and references
│   └── Priority: existing objects > expressions > numbers
├── run_commands() - Executes all commands sequentially
│   └── Enhanced error messages with line numbers and diagnostics
└── generate() - Creates geometric construction with retry logic
```

**Critical Design Decision**: Mathematical expressions are evaluated at parse time, not runtime. The parser converts `100*cos(45°)` into a float element before command execution.

### 2. ReAct Agent Pattern

**Location**: `src/agent/react_agent.py`

The agent follows a strict loop:
1. **Thought**: Reason about the problem
2. **Action**: One of `generate_dsl`, `modify_dsl`, or `final_answer`
3. **Observation**: DSL execution result or validation feedback

**Memory System** (`src/agent/agent_memory.py`): Stores all thoughts, actions, and observations for context building across iterations.

**Error Hint System**: Pattern-matches error messages to provide targeted hints from `prompts/hints/error_*.txt` files.

### 3. DSL Validation

**Location**: `src/dsl/dsl_validator.py`

**Hybrid Approach**:
- **Explicit validation**: Polygons, circles, labeled points must exist as objects
- **Implicit inference**: Segments and lines are inferred if their endpoints exist

**Scoring**:
- Object Score (30%): Percentage of required objects found
- Condition Score (70%): Percentage of geometric conditions satisfied
- Total Score must be ≥90% on both for success

### 4. Multimodal Interface

**Location**: `src/interfaces/multimodal_interface.py`

Abstracts different vision LLM APIs:
- **OpenAI** (GPT-4o, GPT-4V): Standard chat completions with image_url
- **GPT-5**: Uses `input_text`/`input_image` format with `responses.create()`
- **Anthropic** (Claude): Base64 image in content array
- **vLLM**: OpenAI-compatible with custom base_url

**Image Format**: Always base64-encoded PNG with `data:image/png;base64,` prefix.

## DSL Language Critical Rules

### Parser Priority (parse_command in random_constr.py)

When parsing input labels, this exact order is followed:
1. Check if it's an existing element in `element_dict` → use that element
2. Check if it contains operators/trig functions → evaluate as expression
3. Check if it's a simple trig function → evaluate
4. Check if it has angle notation (°, rad) → parse as angle
5. Try to parse as number → create numeric element
6. Fail with clear KeyError

**Why this matters**: A label "100" could be a point named "100" OR the number 100. The parser always checks for existing objects first to avoid ambiguity.

### Polygon Command is FORBIDDEN

**Critical**: Agents must NEVER use `polygon` command. All polygons must be constructed with explicit segments:

```
# ❌ WRONG
polygon : A B C -> triangle AB BC CA

# ✅ CORRECT
segment : A B -> AB
segment : B C -> BC
segment : C A -> CA
```

This is enforced through prompts but not code. The `polygon` command still exists for backward compatibility.

### Error Message Architecture

Error messages follow a structured format (implemented in `run_commands()` and `load()`):

```
Error at line {N}: `{original_line}`

❌ {ERROR_TYPE}: {details}
   {helpful explanation}

💡 TIP: {actionable advice}
```

Specific error types:
- **KeyError**: Undefined element or unknown command (with input types)
- **AssertionError**: Output count mismatch (command returned X but expected Y)
- **ValueError**: Invalid syntax, missing arrows, duplicate labels

## Development Commands

### Running Benchmarks

```bash
# Single problem with verbose output
python run_agent_benchmark.py --problem-id 0 --model gpt-4o --verbose

# Batch evaluation (10 problems)
python run_agent_benchmark.py --batch --model gpt-4o --limit 10

# Full benchmark with specific range
python run_agent_benchmark.py --batch --start 0 --end 100 --model gpt-4o

# Multi-model comparison
./run_multi_model_benchmark.sh
```

### Dataset Creation

```bash
# Parallel processing (4 workers)
./run_parallel_dataset.sh

# Custom parallel settings (8 workers, 5000 files, 625 per batch)
./run_parallel_dataset.sh 8 5000 625

# Single-threaded with range
python scripts/create_dataset.py --start 0 --end 1000

# Merge JSONL files into final dataset
python scripts/create_dataset.py --merge
```

### Testing DSL Code

```bash
# Interactive viewer (navigate with arrow keys)
python preview.py

# Direct DSL execution
python -c "from src.dsl.dsl_executor import DSLExecutor; exec = DSLExecutor(); exec.execute('point : 0 0 -> A')"
```

### Analyzing Results

```bash
python scripts/analyze_benchmark_results.py --results-dir agent_logs/
```

## Prompt System Architecture

**Location**: `prompts/`

- `system_prompt.txt`: Core DSL syntax and agent instructions
- `dsl_guidelines.txt`: Comprehensive DSL reference
- `examples.txt`: Working DSL examples
- `hints/error_*.txt`: Context-specific error recovery hints

**Critical Rule in Prompts**: Rule #3 explicitly forbids `polygon` command usage. This must be maintained across all prompt files.

**Math Expression Feature**: Recently added (Dec 2025). Agents can now use `100*cos(45°)` directly in DSL instead of pre-computing values. This is documented in all prompt files with examples.

**Error Handling Strategy** (Dec 2025): Added comprehensive error handling section to system_prompt.txt that teaches agents to:
1. Analyze error messages systematically (line numbers, error types, hints)
2. Observe rendered images to understand what was actually constructed
3. Choose appropriate fix strategies (specific fix vs major revision)
4. Learn from previous attempts to avoid repeating mistakes
5. Use images as primary feedback for construction validation

## Code Modification Guidelines

### When Modifying DSL Parser (random_constr.py)

1. **Maintain parse order**: The priority of existing objects > expressions > numbers is critical
2. **Line tracking**: All Command objects need `line_number` and `original_line` for error messages
3. **Test expression edge cases**: Empty strings, malformed expressions, scientific notation
4. **Update error messages**: Keep the structured format with emoji indicators

### When Modifying ReAct Agent (react_agent.py)

1. **Preserve memory**: AgentMemory must accumulate all steps for context
2. **Image handling**: Always check `use_vision` flag before adding images
3. **Logging**: Use AgentLogger for session logs, don't rely on print()
4. **Error hints**: Update `ErrorHintManager.ERROR_PATTERNS` when adding new error types

### When Modifying Prompts

1. **Update all related files**: system_prompt.txt, dsl_guidelines.txt, examples.txt
2. **Include examples**: Every new feature needs working examples
3. **Test with agent**: Run actual benchmarks to verify prompt effectiveness
4. **Maintain consistency**: All files must agree on syntax and rules

### When Adding Commands (commands.py)

1. **Type signatures**: Commands use type-based dispatch via `command_types_name()`
2. **Return tuples**: Always return tuple even for single output
3. **Document output count**: Critical for `polygon` and `intersect` which vary
4. **Add to dsl_guidelines.txt**: Document the new command immediately

## Testing Strategy

**No unit tests exist**. Testing is done through:

1. **Benchmark runs**: Run on 10-100 problems to validate changes
2. **Manual DSL execution**: Create test DSL files and verify they execute
3. **Prompt iteration**: Monitor agent logs to see if agents use features correctly

**Before committing**:
```bash
# Test basic execution
python -c "from src.core.random_constr import Construction; c = Construction(); c.load('test.txt'); c.generate()"

# Test on small benchmark
python run_agent_benchmark.py --batch --limit 5 --model gpt-4o
```

## Common Pitfalls

1. **Don't use `assert`**: Replace with explicit `if` checks and clear error messages
2. **Image format for GPT-5**: String URL, not object (different from GPT-4o)
3. **Math expressions**: Remember they're evaluated at parse time, not runtime
4. **Element labels**: Can be any string including numbers; check dict first
5. **Vision flag**: Some models don't need rendered images; respect `use_vision=False`

## Configuration Files

- `.env`: API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- `benchmark_config.yaml`: Default model and benchmark settings
- `requirements.txt`: Minimal dependencies (numpy, matplotlib, openai, dotenv)

## Output Directories

All generated relative to project root:
- `agent_logs/`: ReAct session logs and metrics
- `agent_images/`: Rendered construction images
- `ground_truth/`: Parsed benchmark datasets
- `data/`: Source problem files (GeoQA3)
