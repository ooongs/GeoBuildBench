# Geometry LLM Benchmark System

A comprehensive benchmark system for evaluating LLM-generated geometric constructions using DSL (Domain-Specific Language).

## Overview

This system enables evaluation of LLM agents' ability to generate valid geometric constructions by:
1. **Parsing** Chinese geometry problems to extract requirements
2. **Validating** generated DSL against required objects and conditions
3. **Scoring** constructions with detailed metrics and reports

## Architecture

The system consists of four main components:

### 1. Problem Parser (`problem_parser.py`)
Extracts geometric requirements from Chinese problem text:
- Required objects (points, lines, segments, circles, polygons)
- Verification conditions (parallel, perpendicular, angles, etc.)
- Supports both LLM-based and rule-based parsing

### 2. Dataset Management (`benchmark_dataset.py`)
Structured JSON format for benchmark datasets:
- `BenchmarkProblem`: Single problem with requirements and conditions
- `BenchmarkDataset`: Collection of problems with load/save utilities
- `ConditionBuilder`: Helper for creating verification conditions

### 3. DSL Validator (`dsl_validator.py`)
Core validation engine:
- Parses and executes DSL files
- Checks for required geometric objects
- Verifies geometric conditions using existing `commands.py` functions
- Returns detailed scores and error messages

### 4. Benchmark Evaluator (`benchmark_evaluator.py`)
Main evaluation script:
- Evaluates single or multiple DSL files
- Generates comprehensive reports
- Supports two modes: dataset (detailed) and benchmark (pass/fail)

## Installation

The system uses the existing pyggb infrastructure. No additional dependencies needed.

## Quick Start

### 1. Convert GeoQA3 Problems to Benchmark Format

```bash
# Convert first 10 problems as sample
python convert_geoqa_to_benchmark.py --sample

# Convert all problems
python convert_geoqa_to_benchmark.py --input-dir data-5/GeoQA3/json --output benchmark_geoqa3.json
```

### 2. Evaluate Existing Benchmarks (Pass/Fail Mode)

```bash
# Test "true" benchmarks (should pass)
python benchmark_evaluator.py --mode benchmark \
    --dsl-dir ggb-benchmark/true \
    --expected true \
    --output report_true.json

# Test "false" benchmarks (should fail)
python benchmark_evaluator.py --mode benchmark \
    --dsl-dir ggb-benchmark/false \
    --expected false \
    --output report_false.json
```

### 3. Evaluate with Dataset (Detailed Validation)

```bash
# Assuming you have DSL files named by problem ID
python benchmark_evaluator.py --mode dataset \
    --dsl-dir path/to/dsl/files \
    --dataset benchmark_geoqa3.json \
    --output detailed_report.json
```

## Dataset Format

### BenchmarkProblem JSON Structure

```json
{
  "id": "1",
  "subject": "如图,AB∥CD,直线EF交AB于点E,交CD于点F,EG平分∠BEF,交CD于点G,∠EFG=50°,则∠EGF等于()",
  "required_objects": {
    "points": ["A", "B", "C", "D", "E", "F", "G"],
    "segments": [["A", "B"], ["C", "D"], ["E", "F"]],
    "lines": [["A", "B"], ["C", "D"], ["E", "F"]],
    "circles": [],
    "polygons": []
  },
  "verification_conditions": [
    {
      "type": "parallel",
      "objects": [["A", "B"], ["C", "D"]]
    },
    {
      "type": "angle_value",
      "points": [["E", "F", "G"]],
      "value": 50.0,
      "tolerance": 1.0
    },
    {
      "type": "angle_bisector",
      "line": ["E", "G"],
      "angle_points": ["B", "E", "F"]
    },
    {
      "type": "point_on_line",
      "point": "G",
      "line": ["C", "D"]
    }
  ],
  "metadata": {
    "source": "GeoQA3",
    "difficulty": "medium"
  }
}
```

## Verification Condition Types

The system supports comprehensive geometric verification:

### Line Relationships
- `parallel`: Two lines are parallel
- `perpendicular`: Two lines are perpendicular
- `concurrent`: Three lines meet at a point

### Angles
- `angle_value`: Angle has specific value (in degrees)
- `angle_equality`: Two angles are equal
- `angle_bisector`: Line bisects an angle

### Segments
- `segment_equality`: Two segments have equal length

### Point Relationships
- `collinear`: Points lie on the same line
- `not_collinear`: Points do NOT lie on same line (for valid triangles)
- `concyclic`: Four points lie on the same circle
- `point_on_line`: Point lies on a line
- `point_on_circle`: Point lies on a circle

## Creating Verification Conditions

Use the `ConditionBuilder` helper class:

```python
from benchmark_dataset import ConditionBuilder

# Parallel lines
ConditionBuilder.parallel(["A", "B"], ["C", "D"])

# Angle with specific value
ConditionBuilder.angle_value(["A", "B", "C"], value=60.0, tolerance=1.0)

# Angle equality
ConditionBuilder.angle_equality(["A", "B", "C"], ["D", "E", "F"])

# Perpendicular lines
ConditionBuilder.perpendicular(["A", "B"], ["C", "D"])

# Collinearity
ConditionBuilder.collinear(["A", "B", "C"])

# Non-collinearity (for triangles)
ConditionBuilder.not_collinear(["A", "B", "C"])

# Angle bisector
ConditionBuilder.angle_bisector(["E", "G"], ["B", "E", "F"])
```

## Evaluation Reports

### Report Structure

```json
{
  "timestamp": "2025-11-23T15:16:12.982572",
  "dataset_name": "ggb-benchmark/true",
  "total_problems": 241,
  "evaluated_problems": 241,
  "successful_problems": 231,
  "success_rate": 0.9585,
  "average_object_score": 0.9585,
  "average_condition_score": 0.9585,
  "average_total_score": 0.9585,
  "results": [...],
  "summary": {
    "score_distribution": {...},
    "common_failures": {...}
  }
}
```

### Scoring

- **Object Score** (30% weight): Percentage of required objects present
- **Condition Score** (70% weight): Percentage of conditions satisfied
- **Total Score**: Weighted average
- **Success**: Total score ≥ 90%

## Test Results

### ggb-benchmark/true (241 problems)
- **Accuracy**: 95.85%
- **Passed**: 231/241
- Most failures due to numerical precision or random construction issues

### ggb-benchmark/false (14 problems)
- **Accuracy**: 100%
- **Passed**: 14/14
- All false theorems correctly identified

## Example: Validating a DSL File

```python
from dsl_validator import DSLValidator
from benchmark_dataset import BenchmarkProblem, RequiredObjects, ConditionBuilder, VerificationCondition

# Create validator
validator = DSLValidator()

# Define problem requirements
required_objects = RequiredObjects(
    points=["A", "B", "C", "D"],
    segments=[["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
    lines=[],
    circles=[],
    polygons=[["A", "B", "C", "D"]]
)

conditions = [
    VerificationCondition.from_dict(ConditionBuilder.parallel(["A", "B"], ["C", "D"])),
    VerificationCondition.from_dict(ConditionBuilder.parallel(["B", "C"], ["D", "A"])),
]

problem = BenchmarkProblem(
    id="test_parallelogram",
    subject="四边形ABCD是平行四边形",
    required_objects=required_objects,
    verification_conditions=conditions
)

# Validate DSL file
result = validator.validate("path/to/dsl/file.txt", problem)

print(f"Success: {result.success}")
print(f"Object Score: {result.object_score:.2%}")
print(f"Condition Score: {result.condition_score:.2%}")
print(f"Total Score: {result.total_score:.2%}")
```

## Extending the System

### Adding New Condition Types

1. Add condition type to `ConditionBuilder` in `benchmark_dataset.py`
2. Implement checker method in `DSLValidator._check_condition()`
3. Use existing functions from `commands.py` for verification

Example:

```python
# In benchmark_dataset.py
@staticmethod
def my_new_condition(param1, param2):
    return {
        "type": "my_new_condition",
        "param1": param1,
        "param2": param2
    }

# In dsl_validator.py
def _check_my_new_condition(self, data: Dict) -> Dict[str, Any]:
    # Implementation using commands.py functions
    param1 = data.get("param1")
    param2 = data.get("param2")
    # ... validation logic ...
    return {
        "passed": True/False,
        "message": "Description of result"
    }
```

## Files Created

- `problem_parser.py`: Chinese problem text parser
- `benchmark_dataset.py`: Dataset structure and management
- `dsl_validator.py`: DSL validation engine
- `benchmark_evaluator.py`: Main evaluation script
- `convert_geoqa_to_benchmark.py`: GeoQA3 conversion utility
- `BENCHMARK_README.md`: This documentation

## Integration with Existing System

The benchmark system integrates seamlessly with existing pyggb components:
- Uses `random_constr.Construction` for DSL parsing
- Leverages `commands.py` functions for geometric verification
- Works with existing `geo_types.py` classes
- Compatible with all existing DSL files in `ggb-benchmark/`

## Future Enhancements

Potential improvements:
1. LLM API integration for automatic problem parsing
2. Visual diff tool to compare expected vs actual constructions
3. More sophisticated scoring metrics
4. Support for inequality conditions (>, <, ≥, ≤)
5. Batch evaluation with parallel processing
6. Web interface for interactive evaluation

## Citation

If you use this benchmark system, please cite:
```
Geometry LLM Benchmark System for pyggb
https://github.com/yourusername/pyggb
```

## License

Same as the main pyggb project.

