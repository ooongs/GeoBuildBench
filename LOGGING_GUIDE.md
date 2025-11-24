# Agent Logging Guide

## Overview

The ReAct agent now has comprehensive logging that saves all outputs, DSL code, and images for every session.

## What Gets Logged

### 1. Session Log Files
Location: `agent_logs/sessions/{session_id}.log`

Contains:
- Problem description
- Each iteration with:
  - Agent's thought process
  - Action taken (DSL code)
  - Execution result (success/failure)
  - Error messages if failed
- Final validation results
- Session summary

### 2. DSL Code Files
Location: `agent_logs/dsl_code/{session_id}_iter{N}.txt`

- Separate file for each iteration's DSL code
- Clean, executable DSL without comments
- Can be run independently for debugging

### 3. Images
Location: `agent_logs/images/{session_id}_iter{N}.png`

- Rendered geometry image from each successful execution
- Saved as PNG files
- Can be viewed to see agent's progress

### 4. Session Summary
Location: `agent_logs/sessions/{session_id}_summary.json`

Contains:
- Session metadata
- Paths to all generated files
- Quick overview in JSON format

## Comment Handling

The DSL executor now handles both comment styles:

### Supported Comment Styles

```dsl
# Hash comments (Python/Shell style)
point :  -> A  # Inline hash comment

// Slash comments (C++/Java style)  
point :  -> B  // Inline slash comment
```

Both are automatically stripped before execution!

## Usage Examples

### View Session Log

```bash
# Find your session
ls agent_logs/sessions/

# View the log
cat agent_logs/sessions/0_20251123_161928.log

# Or use less for scrolling
less agent_logs/sessions/0_20251123_161928.log
```

### View Generated Images

```bash
# List all images
ls agent_logs/images/

# Open an image (macOS)
open agent_logs/images/0_20251123_161928_iter1.png

# Or use any image viewer
```

### View DSL Code

```bash
# List DSL files
ls agent_logs/dsl_code/

# View DSL from iteration 1
cat agent_logs/dsl_code/0_20251123_161928_iter1.txt

# Test the DSL directly
python -c "from random_constr import Construction; c = Construction(); c.load('agent_logs/dsl_code/0_20251123_161928_iter1.txt'); c.generate(); print('Success!')"
```

### View Session Summary

```bash
# Pretty print JSON summary
cat agent_logs/sessions/0_20251123_161928_summary.json | python -m json.tool
```

## Log Structure

```
agent_logs/
├── sessions/
│   ├── {problem_id}_{timestamp}.log          # Main log file
│   └── {problem_id}_{timestamp}_summary.json # Summary JSON
├── dsl_code/
│   ├── {session_id}_iter1.txt               # DSL from iteration 1
│   ├── {session_id}_iter2.txt               # DSL from iteration 2
│   └── ...
└── images/
    ├── {session_id}_iter1.png               # Image from iteration 1
    ├── {session_id}_iter2.png               # Image from iteration 2
    └── ...
```

## Example Log Output

```
================================================================================
REACT AGENT SESSION
================================================================================
Problem ID: 0
Start Time: 2025-11-23T16:19:28.123456
Problem: 如图,在△ABC中,已知∠A=80°,∠B=60°,D、E 分别在 AB、AC 上,DE∥BC...
================================================================================

================================================================================
ITERATION 1
================================================================================
Time: 2025-11-23T16:19:30.456789

THOUGHT:
--------------------------------------------------------------------------------
I need to construct triangle ABC with angles A=80° and B=60°...

ACTION: generate_dsl
--------------------------------------------------------------------------------
DSL CODE:
```
point :  -> A
point :  -> B
point :  -> C
polygon : A B C -> triangle c a b
...
```

EXECUTION RESULT:
--------------------------------------------------------------------------------
✓ SUCCESS
Image saved: agent_images/0_iter1.png
Image copied to: agent_logs/images/0_20251123_161928_iter1.png
```

## Programmatic Access

### Load and Analyze Logs

```python
import json
from agent_logger import AgentLogger

# Initialize logger
logger = AgentLogger(log_dir="agent_logs")

# Get all sessions
sessions = logger.get_all_sessions()
print(f"Total sessions: {len(sessions)}")

# Load a session summary
with open("agent_logs/sessions/0_20251123_161928_summary.json") as f:
    summary = json.load(f)

print(f"Session: {summary['session_id']}")
print(f"Iterations: {summary['iterations']}")
print(f"Images: {len(summary['images'])}")
```

## Cleanup

To clean old logs:

```bash
# Remove all logs older than 7 days
find agent_logs/sessions/ -name "*.log" -mtime +7 -delete
find agent_logs/images/ -name "*.png" -mtime +7 -delete
find agent_logs/dsl_code/ -name "*.txt" -mtime +7 -delete

# Or remove all logs
rm -rf agent_logs/
```

## Configuration

Control logging behavior:

```python
from react_agent import ReActAgent

# With full logging and image saving
agent = ReActAgent(
    model="gpt-4o",
    save_images=True,      # Save images
    log_dir="agent_logs",  # Log directory
    verbose=True           # Print to console
)

# Minimal logging (no images)
agent = ReActAgent(
    model="gpt-4o",
    save_images=False,     # Don't save images
    log_dir="agent_logs",
    verbose=False          # Quiet mode
)
```

## Tips

1. **Session IDs** are in format: `{problem_id}_{timestamp}`
2. **Images are only saved** when DSL execution succeeds
3. **DSL files** have comments stripped but are otherwise identical to what LLM generated
4. **Log files** are plain text - easy to grep/search
5. **Summary JSON** files provide quick metadata access

## Analysis Examples

### Count Success Rate

```bash
# Count successful vs failed iterations
grep "✓ SUCCESS" agent_logs/sessions/*.log | wc -l
grep "✗ FAILED" agent_logs/sessions/*.log | wc -l
```

### Find Common Errors

```bash
# Find most common errors
grep "Error:" agent_logs/sessions/*.log | sort | uniq -c | sort -rn | head -10
```

### Check Average Iterations

```bash
# Count iterations per session
grep "ITERATION" agent_logs/sessions/*.log | wc -l
```

## Troubleshooting

### No images saved

Check that `save_images=True` and DSL executed successfully.

### Log file empty

Agent may have crashed before completing. Check terminal output.

### Large log directory

Old sessions accumulate. Use cleanup script or rotate logs.

## See Also

- `AGENT_README.md` - Full agent documentation
- `agent_logger.py` - Logger implementation
- `react_agent.py` - Agent that uses logging


