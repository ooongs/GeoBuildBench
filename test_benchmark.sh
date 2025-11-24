#!/bin/bash
# Quick test script for Geometry LLM Benchmark System

echo "=================================="
echo "Geometry LLM Benchmark System Test"
echo "=================================="
echo ""

# Test 1: Convert sample GeoQA3 problems
echo "Test 1: Converting GeoQA3 sample problems..."
python convert_geoqa_to_benchmark.py --sample
echo ""

# Test 2: Evaluate true benchmarks
echo "Test 2: Evaluating ggb-benchmark/true (should all pass)..."
python benchmark_evaluator.py --mode benchmark \
    --dsl-dir ggb-benchmark/true \
    --expected true \
    --output report_true.json | tail -20
echo ""

# Test 3: Evaluate false benchmarks
echo "Test 3: Evaluating ggb-benchmark/false (should all fail)..."
python benchmark_evaluator.py --mode benchmark \
    --dsl-dir ggb-benchmark/false \
    --expected false \
    --output report_false.json
echo ""

echo "=================================="
echo "All tests completed!"
echo ""
echo "Generated files:"
echo "  - benchmark_geoqa3.json (sample dataset)"
echo "  - report_true.json (true benchmark report)"
echo "  - report_false.json (false benchmark report)"
echo ""
echo "See BENCHMARK_README.md for detailed documentation"
echo "=================================="

