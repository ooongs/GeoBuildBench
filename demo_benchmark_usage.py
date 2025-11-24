#!/usr/bin/env python3
"""
Demo: Complete workflow for Geometry LLM Benchmark System
Shows how to use all components together.
"""

from problem_parser import ProblemParser
from benchmark_dataset import (
    BenchmarkDataset, BenchmarkProblem, RequiredObjects,
    VerificationCondition, ConditionBuilder
)
from dsl_validator import DSLValidator


def demo_1_parse_problem():
    """Demo 1: Parse a Chinese geometry problem."""
    print("="*70)
    print("DEMO 1: Parse Chinese Geometry Problem")
    print("="*70)
    
    parser = ProblemParser()
    
    problem_text = "如图,AB∥CD,直线EF交AB于点E,交CD于点F,EG平分∠BEF,交CD于点G,∠EFG=50°,则∠EGF等于()"
    
    result = parser.parse_problem(problem_text, problem_id="demo_1")
    
    print(f"\nProblem: {problem_text}")
    print(f"\nExtracted Points: {result['required_objects']['points']}")
    print(f"Extracted Segments: {result['required_objects']['segments']}")
    print(f"Extracted Lines: {result['required_objects']['lines']}")
    print(f"\nConditions:")
    for i, cond in enumerate(result['verification_conditions'], 1):
        print(f"  {i}. {cond['type']}: {cond}")
    print()


def demo_2_create_benchmark():
    """Demo 2: Create a benchmark problem manually."""
    print("="*70)
    print("DEMO 2: Create Benchmark Problem Manually")
    print("="*70)
    
    # Define required objects
    required_objects = RequiredObjects(
        points=["A", "B", "C"],
        segments=[["A", "B"], ["B", "C"], ["A", "C"]],
        lines=[],
        circles=[],
        polygons=[["A", "B", "C"]]
    )
    
    # Define verification conditions
    conditions = [
        VerificationCondition.from_dict(
            ConditionBuilder.not_collinear(["A", "B", "C"])
        ),
        VerificationCondition.from_dict(
            ConditionBuilder.angle_value(["A", "B", "C"], 60.0)
        ),
    ]
    
    # Create problem
    problem = BenchmarkProblem(
        id="demo_triangle",
        subject="Triangle ABC with angle ABC = 60°",
        required_objects=required_objects,
        verification_conditions=conditions,
        metadata={"difficulty": "easy"}
    )
    
    print(f"\nCreated problem: {problem.id}")
    print(f"Subject: {problem.subject}")
    print(f"Required points: {problem.required_objects.points}")
    print(f"Conditions: {len(problem.verification_conditions)}")
    print()


def demo_3_validate_dsl():
    """Demo 3: Validate a DSL file."""
    print("="*70)
    print("DEMO 3: Validate DSL File")
    print("="*70)
    
    # Use an existing DSL file
    dsl_file = "ggb-benchmark/true/regular-triangle.txt"
    
    # Create a problem that matches this DSL
    required_objects = RequiredObjects(
        points=["A", "B", "C"],
        segments=[["A", "B"], ["B", "C"], ["C", "A"]],
        lines=[],
        circles=[],
        polygons=[["A", "B", "C"]]
    )
    
    conditions = [
        VerificationCondition.from_dict(
            ConditionBuilder.not_collinear(["A", "B", "C"])
        ),
        VerificationCondition.from_dict(
            ConditionBuilder.segment_equality(["A", "B"], ["B", "C"])
        ),
        VerificationCondition.from_dict(
            ConditionBuilder.segment_equality(["B", "C"], ["C", "A"])
        ),
    ]
    
    problem = BenchmarkProblem(
        id="regular_triangle",
        subject="Regular triangle ABC",
        required_objects=required_objects,
        verification_conditions=conditions
    )
    
    # Validate
    validator = DSLValidator()
    result = validator.validate(dsl_file, problem)
    
    print(f"\nDSL File: {dsl_file}")
    print(f"Validation Result: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    print(f"Object Score: {result.object_score:.2%}")
    print(f"Condition Score: {result.condition_score:.2%}")
    print(f"Total Score: {result.total_score:.2%}")
    
    if result.missing_objects['points']:
        print(f"Missing points: {result.missing_objects['points']}")
    
    if result.failed_conditions:
        print(f"Failed conditions: {len(result.failed_conditions)}")
        for cond in result.failed_conditions:
            print(f"  - {cond['type']}")
    print()


def demo_4_dataset_operations():
    """Demo 4: Dataset operations."""
    print("="*70)
    print("DEMO 4: Dataset Operations")
    print("="*70)
    
    # Create dataset
    dataset = BenchmarkDataset()
    
    # Add some problems
    for i in range(3):
        required_objects = RequiredObjects(
            points=["A", "B", "C"],
            segments=[],
            lines=[],
            circles=[],
            polygons=[]
        )
        
        problem = BenchmarkProblem(
            id=f"demo_{i}",
            subject=f"Demo problem {i}",
            required_objects=required_objects,
            verification_conditions=[]
        )
        dataset.add_problem(problem)
    
    print(f"\nDataset size: {len(dataset)} problems")
    print("Problems:")
    for problem in dataset:
        print(f"  - {problem.id}: {problem.subject}")
    
    # Save and load
    output_file = "demo_dataset.json"
    dataset.save(output_file)
    print(f"\nSaved to: {output_file}")
    
    loaded = BenchmarkDataset(output_file)
    print(f"Loaded dataset: {len(loaded)} problems")
    print()
    
    # Clean up
    import os
    if os.path.exists(output_file):
        os.remove(output_file)


def demo_5_all_condition_types():
    """Demo 5: Show all available condition types."""
    print("="*70)
    print("DEMO 5: All Available Condition Types")
    print("="*70)
    
    conditions = [
        ("Parallel Lines", ConditionBuilder.parallel(["A", "B"], ["C", "D"])),
        ("Perpendicular", ConditionBuilder.perpendicular(["A", "B"], ["C", "D"])),
        ("Angle Value", ConditionBuilder.angle_value(["A", "B", "C"], 60.0)),
        ("Angle Equality", ConditionBuilder.angle_equality(["A", "B", "C"], ["D", "E", "F"])),
        ("Segment Equality", ConditionBuilder.segment_equality(["A", "B"], ["C", "D"])),
        ("Collinear", ConditionBuilder.collinear(["A", "B", "C"])),
        ("Not Collinear", ConditionBuilder.not_collinear(["A", "B", "C"])),
        ("Concyclic", ConditionBuilder.concyclic(["A", "B", "C", "D"])),
        ("Concurrent Lines", ConditionBuilder.concurrent([["A", "B"], ["C", "D"], ["E", "F"]])),
        ("Point on Line", ConditionBuilder.point_on_line("P", ["A", "B"])),
        ("Point on Circle", ConditionBuilder.point_on_circle("P", "O")),
        ("Angle Bisector", ConditionBuilder.angle_bisector(["E", "G"], ["B", "E", "F"])),
    ]
    
    print("\nSupported Condition Types:")
    print("-" * 70)
    for i, (name, condition) in enumerate(conditions, 1):
        print(f"{i:2d}. {name:20s} type='{condition['type']}'")
    
    print(f"\nTotal: {len(conditions)} condition types supported")
    print()


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("GEOMETRY LLM BENCHMARK SYSTEM - COMPLETE DEMO")
    print("="*70)
    print()
    
    demo_1_parse_problem()
    demo_2_create_benchmark()
    demo_3_validate_dsl()
    demo_4_dataset_operations()
    demo_5_all_condition_types()
    
    print("="*70)
    print("All demos completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Read BENCHMARK_README.md for detailed documentation")
    print("2. Run test_benchmark.sh for comprehensive testing")
    print("3. Try: python benchmark_evaluator.py --help")
    print("="*70)


if __name__ == "__main__":
    main()

