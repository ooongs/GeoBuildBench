# ReAct Multimodal Agent for Geometry Problems

A multimodal LLM agent that solves Chinese geometry problems using ReAct (Reasoning and Acting) pattern with visual feedback.

## 🎯 Overview

The agent:
1. Reads geometry problems from benchmark dataset
2. Generates DSL code to construct geometric figures
3. Executes DSL and renders images
4. Observes visual output and error messages
5. Iteratively refines solutions using multimodal feedback
6. Validates against problem requirements

## 🏗️ Architecture

```
Problem → ReAct Agent → DSL Code → Executor → Image
              ↑                                  ↓
              └────── Observation ───────────────┘
```

### Components

- **ReActAgent**: Core reasoning and action loop
- **DSLExecutor**: Safe DSL execution and image rendering
- **MultimodalInterface**: Vision LLM integration (GPT-4o, Claude)
- **AgentMemory**: Conversation history and learning
- **DSLValidator**: Validation against benchmark requirements

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install openai anthropic pillow matplotlib python-dotenv

# Set API key in .env
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. Run Single Problem

```bash
# Solve a specific problem
python run_agent_benchmark.py --problem-id 1 --model gpt-4o --verbose

# Debug mode with detailed output
python run_agent_benchmark.py --problem-id 1 --debug
```

### 3. Run Batch Evaluation

```bash
# Evaluate on first 5 problems
python run_agent_benchmark.py --batch --limit 5 --model gpt-4o

# Full evaluation
python run_agent_benchmark.py --batch --output full_results.json
```

## 📝 Usage Examples

### Command Line Interface

```bash
# Basic usage
python run_agent_benchmark.py --problem-id 1 --model gpt-4o

# With custom settings
python run_agent_benchmark.py \
    --problem-id 1 \
    --model gpt-4o \
    --max-iter 15 \
    --verbose \
    --output results.json

# Batch with limit
python run_agent_benchmark.py \
    --batch \
    --limit 10 \
    --model gpt-4o-mini \
    --output batch_results.json

# Different models
python run_agent_benchmark.py --problem-id 1 --model claude-3-5-sonnet-20241022
python run_agent_benchmark.py --problem-id 1 --model gpt-4o-mini
```

### Python API

```python
from react_agent import ReActAgent
from benchmark_dataset import BenchmarkDataset

# Initialize agent
agent = ReActAgent(
    model="gpt-4o",
    max_iterations=10,
    save_images=True,
    verbose=True
)

# Load problem
dataset = BenchmarkDataset("benchmark_geoqa3.json")
problem = dataset.get_problem("1")

# Solve
results = agent.solve(problem)

print(f"Success: {results['success']}")
print(f"Iterations: {results['iterations']}")
print(f"Final DSL:\n{results['final_dsl']}")
```

## 🔧 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--problem-id ID` | Solve specific problem | - |
| `--batch` | Run on multiple problems | False |
| `--dataset PATH` | Benchmark dataset path | benchmark_geoqa3.json |
| `--limit N` | Max problems (batch mode) | None (all) |
| `--model NAME` | LLM model | gpt-4o |
| `--max-iter N` | Max iterations per problem | 10 |
| `--output FILE` | Output results file | agent_results.json |
| `--no-save-images` | Don't save images | False |
| `--verbose` | Detailed logs | False |
| `--debug` | Debug mode | False |

## 🤖 Supported Models

| Model | Provider | Speed | Cost | Quality |
|-------|----------|-------|------|---------|
| gpt-4o | OpenAI | ★★★ | $$ | ★★★★ |
| gpt-4o-mini | OpenAI | ★★★★ | $ | ★★★ |
| gpt-4-vision-preview | OpenAI | ★★ | $$$ | ★★★★★ |
| claude-3-5-sonnet-20241022 | Anthropic | ★★★ | $$ | ★★★★ |

**Recommended**: `gpt-4o` (best balance of speed, cost, and quality)

## 📊 Output Format

### Single Problem Result

```json
{
  "problem_id": "1",
  "success": true,
  "iterations": 5,
  "final_dsl": "point :  -> A\n...",
  "summary": {
    "total_steps": 5,
    "successful_executions": 4,
    "duration_seconds": 45.2
  },
  "memory_path": "agent_images/1_memory.json"
}
```

### Batch Evaluation Report

```json
{
  "timestamp": "2025-11-23T...",
  "model": "gpt-4o",
  "total_problems": 10,
  "successful": 7,
  "failed": 3,
  "success_rate": 0.7,
  "average_iterations": 6.2,
  "results": [...]
}
```

## 🎓 How It Works

### ReAct Loop

The agent follows this pattern:

```
1. Thought: "I need to create parallel lines AB and CD"
2. Action: generate_dsl
   ```
   point :  -> A
   point :  -> B
   line : A B -> line_AB
   ...
   ```
3. Observation: [Image + Error messages]
4. Reflection: Check if construction is correct
5. Repeat until solved or max iterations
```

### Multimodal Observation

The agent can:
- **See** rendered geometry images
- **Read** error messages
- **Understand** spatial relationships visually
- **Detect** missing or incorrect objects
- **Refine** DSL based on visual feedback

### Self-Correction

The agent learns from:
- Syntax errors → Fix DSL syntax
- Missing objects → Add required constructions
- Visual discrepancies → Adjust geometric relationships
- Validation failures → Refine to meet conditions

## 📁 Files Created

### Core System
- `react_agent.py` - ReAct agent implementation
- `dsl_executor.py` - DSL execution and rendering
- `multimodal_interface.py` - Vision LLM integration
- `agent_memory.py` - Memory management
- `run_agent_benchmark.py` - Main evaluation script

### Prompts
- `prompts/system_prompt.txt` - Agent role and capabilities
- `prompts/react_template.txt` - ReAct reasoning format
- `prompts/dsl_guidelines.txt` - DSL syntax reference

### Generated
- `agent_images/` - Rendered images and memories
- `agent_results.json` - Evaluation results
- `{problem_id}_memory.json` - Per-problem memory

## 🔍 Debugging

### View Agent Reasoning

```bash
python run_agent_benchmark.py --problem-id 1 --verbose
```

This shows:
- Each thought and action
- DSL execution results
- Error messages
- Iteration progress

### Inspect Memory

```python
from agent_memory import AgentMemory

# Load saved memory
memory = AgentMemory.load_from_file("agent_images/1_memory.json")

# View steps
for step in memory.steps:
    print(f"Iteration {step.iteration}:")
    print(f"  Thought: {step.thought.content}")
    print(f"  Action: {step.action.action_type}")
    print(f"  Success: {step.observation.success}")
```

### View Generated Images

Images are saved in `agent_images/` directory:
- `{problem_id}_iter{N}.png` - Image from iteration N
- `{problem_id}_memory.json` - Complete reasoning trace

## 💰 Cost Estimation

### Per Problem (approx)

| Model | Avg Cost | Tokens Used |
|-------|----------|-------------|
| gpt-4o | $0.03-0.10 | 5K-15K |
| gpt-4o-mini | $0.01-0.03 | 5K-15K |
| claude-3-5-sonnet | $0.05-0.15 | 5K-15K |

### 100 Problems

| Model | Est. Cost |
|-------|-----------|
| gpt-4o | $3-10 |
| gpt-4o-mini | $1-3 |
| claude-3-5-sonnet | $5-15 |

*Costs vary based on problem complexity and iterations*

## 🎯 Performance Metrics

### Target Performance

- **Phase 1 (Basic)**: 60% success rate
- **Phase 2 (Visual)**: 75% success rate  
- **Phase 3 (Advanced)**: 85% success rate

### Evaluation Metrics

- **Construction Success**: DSL executes without errors
- **Visual Correctness**: Image matches problem
- **Benchmark Pass**: Meets verification conditions
- **Efficiency**: Iterations to solution

## 🐛 Troubleshooting

### API Key Errors

```
ERROR: OPENAI_API_KEY not found
```
→ Set API key in `.env` file

### Import Errors

```
ModuleNotFoundError: No module named 'openai'
```
→ Install dependencies: `pip install openai anthropic`

### DSL Execution Failures

Check:
1. DSL syntax is correct
2. All objects defined before use
3. Ends with `prove` statement
4. No circular dependencies

### Low Success Rate

Try:
- Increase max iterations: `--max-iter 15`
- Use more capable model: `--model gpt-4-vision-preview`
- Enable verbose mode to debug: `--verbose`

## 📚 Examples

### Example 1: Simple Triangle

```python
from react_agent import ReActAgent
from benchmark_dataset import BenchmarkProblem, RequiredObjects

# Define problem
problem = BenchmarkProblem(
    id="triangle",
    subject="Create a triangle ABC",
    required_objects=RequiredObjects(
        points=["A", "B", "C"],
        polygons=[["A", "B", "C"]]
    ),
    verification_conditions=[]
)

# Solve
agent = ReActAgent()
results = agent.solve(problem)
```

### Example 2: Parallel Lines

Problem: "AB∥CD, line EF intersects them"

The agent will:
1. Create points A, B, C, D
2. Create lines AB and CD (ensuring parallel)
3. Create line EF
4. Find intersections E and F
5. Validate construction

## 🔗 Integration

The agent integrates with existing benchmark system:

```python
from benchmark_dataset import BenchmarkDataset
from dsl_validator import DSLValidator
from react_agent import ReActAgent

# Load problems
dataset = BenchmarkDataset("benchmark_geoqa3.json")

# Solve with agent
agent = ReActAgent()
results = agent.solve(dataset[0])

# Validate result
validator = DSLValidator()
validation = validator.validate(results['final_dsl'], dataset[0])
```

## 📖 Further Reading

- ReAct paper: https://arxiv.org/abs/2210.03629
- DSL documentation: See `DSL_PIPELINE_EXPLANATION.md`
- Benchmark system: See `BENCHMARK_README.md`

## 🎉 Quick Test

```bash
# Test the agent on a simple problem
python run_agent_benchmark.py --problem-id 0 --model gpt-4o --verbose
```

This will solve problem 0 and show detailed reasoning!

