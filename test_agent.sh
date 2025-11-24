#!/bin/bash
# Test script for ReAct Multimodal Agent

echo "=================================="
echo "ReAct Agent Test Suite"
echo "=================================="
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ] && [ ! -f .env ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not found"
    echo "Set it in .env file to test the agent"
    echo ""
fi

# Test 1: DSL Executor
echo "Test 1: DSL Executor..."
python -c "from dsl_executor import DSLExecutor; print('✓ DSL Executor imported')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ DSL Executor test passed"
else
    echo "✗ DSL Executor test failed"
fi
echo ""

# Test 2: Multimodal Interface
echo "Test 2: Multimodal Interface..."
python -c "from multimodal_interface import MultimodalInterface; print('✓ Multimodal Interface imported')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Multimodal Interface test passed"
else
    echo "✗ Multimodal Interface test failed"
fi
echo ""

# Test 3: Agent Memory
echo "Test 3: Agent Memory..."
python -c "from agent_memory import AgentMemory; print('✓ Agent Memory imported')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Agent Memory test passed"
else
    echo "✗ Agent Memory test failed"
fi
echo ""

# Test 4: ReAct Agent
echo "Test 4: ReAct Agent..."
python -c "from react_agent import ReActAgent; print('✓ ReAct Agent imported')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ ReAct Agent test passed"
else
    echo "✗ ReAct Agent test failed"
fi
echo ""

# Test 5: Run Agent Benchmark (if API key available)
if [ -n "$OPENAI_API_KEY" ] || [ -f .env ]; then
    echo "Test 5: Running agent on test problem..."
    echo "(This will use API and may take 1-2 minutes)"
    echo ""
    
    # Run on first problem with debug mode
    python run_agent_benchmark.py --problem-id 0 --model gpt-4o --max-iter 5 --debug 2>&1 | head -50
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Agent execution test passed"
    else
        echo ""
        echo "✗ Agent execution test failed"
    fi
else
    echo "Test 5: Skipped (no API key)"
    echo "Set OPENAI_API_KEY in .env to run full test"
fi

echo ""
echo "=================================="
echo "Test Suite Complete"
echo "=================================="
echo ""
echo "To run the agent:"
echo "  python run_agent_benchmark.py --problem-id 0 --model gpt-4o --verbose"
echo ""
echo "See AGENT_README.md for full documentation"
echo "=================================="

