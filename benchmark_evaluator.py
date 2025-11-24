#!/usr/bin/env python3
"""
Benchmark Evaluator
Main script for running benchmarks and generating reports.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from benchmark_dataset import BenchmarkDataset, BenchmarkProblem
from dsl_validator import DSLValidator, ValidationResult


@dataclass
class EvaluationReport:
    """Complete evaluation report for a benchmark run."""
    timestamp: str
    dataset_name: str
    total_problems: int
    evaluated_problems: int
    successful_problems: int
    average_object_score: float
    average_condition_score: float
    average_total_score: float
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "total_problems": self.total_problems,
            "evaluated_problems": self.evaluated_problems,
            "successful_problems": self.successful_problems,
            "average_object_score": self.average_object_score,
            "average_condition_score": self.average_condition_score,
            "average_total_score": self.average_total_score,
            "success_rate": self.successful_problems / self.evaluated_problems if self.evaluated_problems > 0 else 0,
            "results": self.results,
            "summary": self.summary
        }
    
    def save(self, output_file: str):
        """Save report to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "="*70)
        print("BENCHMARK EVALUATION REPORT")
        print("="*70)
        print(f"Dataset: {self.dataset_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Problems: {self.total_problems}")
        print(f"Evaluated: {self.evaluated_problems}")
        print(f"Successful: {self.successful_problems} ({100*self.successful_problems/self.evaluated_problems:.1f}%)")
        print("-"*70)
        print(f"Average Object Score: {100*self.average_object_score:.2f}%")
        print(f"Average Condition Score: {100*self.average_condition_score:.2f}%")
        print(f"Average Total Score: {100*self.average_total_score:.2f}%")
        print("="*70)
        
        # Print per-problem summary
        if len(self.results) > 0 and "problem_id" in self.results[0]:
            # Dataset mode - detailed results
            print("\nPer-Problem Results:")
            print("-"*70)
            for result in self.results:
                problem_id = result["problem_id"]
                success = "✓" if result["validation"]["success"] else "✗"
                total_score = result["validation"]["total_score"]
                print(f"{success} {problem_id:20s} Score: {100*total_score:.1f}%")
                if not result["validation"]["success"]:
                    error = result["validation"].get("error_message")
                    if error:
                        print(f"  Error: {error}")
                    else:
                        obj_score = result["validation"]["object_score"]
                        cond_score = result["validation"]["condition_score"]
                        print(f"  Object: {100*obj_score:.1f}%, Condition: {100*cond_score:.1f}%")
        elif len(self.results) > 0 and "file" in self.results[0]:
            # Benchmark mode - show failures only
            print("\nFailed Problems:")
            print("-"*70)
            for result in self.results:
                if not result["correct"]:
                    file = result["file"]
                    expected = "SUCCESS" if result["expected"] else "FAILURE"
                    actual = "SUCCESS" if result["actual"] else "FAILURE" if result["actual"] is not None else "ERROR"
                    error = result.get("error", "")
                    print(f"✗ {file:40s} Expected: {expected}, Got: {actual}")
                    if error:
                        print(f"  Error: {error}")


class BenchmarkEvaluator:
    """Evaluate LLM-generated DSL against benchmark dataset."""
    
    def __init__(self, validator: Optional[DSLValidator] = None):
        """
        Initialize evaluator.
        
        Args:
            validator: DSLValidator instance (creates new one if None)
        """
        self.validator = validator or DSLValidator()
    
    def evaluate_single(self, dsl_file: str, problem: BenchmarkProblem, 
                       max_attempts: int = 100) -> Dict[str, Any]:
        """
        Evaluate a single DSL file against a problem.
        
        Args:
            dsl_file: Path to DSL file
            problem: BenchmarkProblem to evaluate against
            max_attempts: Max attempts for construction generation
            
        Returns:
            Dictionary with problem info and validation result
        """
        result = self.validator.validate(dsl_file, problem, max_attempts)
        
        return {
            "problem_id": problem.id,
            "problem_subject": problem.subject,
            "dsl_file": dsl_file,
            "validation": result.to_dict()
        }
    
    def evaluate_dataset(self, dsl_directory: str, dataset: BenchmarkDataset,
                        max_attempts: int = 100, verbose: bool = True) -> EvaluationReport:
        """
        Evaluate all DSL files in a directory against a dataset.
        
        Args:
            dsl_directory: Directory containing DSL files (named by problem ID)
            dataset: BenchmarkDataset to evaluate against
            max_attempts: Max attempts for each construction
            verbose: Print progress messages
            
        Returns:
            EvaluationReport with all results
        """
        results = []
        
        total_object_score = 0.0
        total_condition_score = 0.0
        total_score = 0.0
        successful_count = 0
        evaluated_count = 0
        
        for problem in dataset:
            # Find corresponding DSL file
            dsl_file = self._find_dsl_file(dsl_directory, problem.id)
            
            if dsl_file is None:
                if verbose:
                    print(f"⚠ Warning: No DSL file found for problem {problem.id}")
                continue
            
            if verbose:
                print(f"Evaluating {problem.id}...", end=" ")
            
            # Evaluate
            result = self.evaluate_single(dsl_file, problem, max_attempts)
            results.append(result)
            
            # Update statistics
            validation = result["validation"]
            if validation.get("error_message") is None:
                evaluated_count += 1
                total_object_score += validation["object_score"]
                total_condition_score += validation["condition_score"]
                total_score += validation["total_score"]
                
                if validation["success"]:
                    successful_count += 1
                    if verbose:
                        print("✓")
                else:
                    if verbose:
                        print(f"✗ (score: {100*validation['total_score']:.1f}%)")
            else:
                if verbose:
                    print(f"✗ ERROR: {validation['error_message']}")
        
        # Calculate averages
        avg_object_score = total_object_score / evaluated_count if evaluated_count > 0 else 0.0
        avg_condition_score = total_condition_score / evaluated_count if evaluated_count > 0 else 0.0
        avg_total_score = total_score / evaluated_count if evaluated_count > 0 else 0.0
        
        # Create summary
        summary = {
            "score_distribution": self._calculate_score_distribution(results),
            "common_failures": self._analyze_common_failures(results)
        }
        
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            dataset_name=dsl_directory,
            total_problems=len(dataset),
            evaluated_problems=evaluated_count,
            successful_problems=successful_count,
            average_object_score=avg_object_score,
            average_condition_score=avg_condition_score,
            average_total_score=avg_total_score,
            results=results,
            summary=summary
        )
        
        return report
    
    def evaluate_benchmark_folder(self, benchmark_folder: str, 
                                  expected_result: bool = True,
                                  max_attempts: int = 100,
                                  verbose: bool = True) -> EvaluationReport:
        """
        Evaluate a benchmark folder (like ggb-benchmark/true or /false).
        
        Args:
            benchmark_folder: Path to folder with .txt DSL files
            expected_result: True if all should pass, False if all should fail
            max_attempts: Max attempts for construction generation
            verbose: Print progress messages
            
        Returns:
            EvaluationReport
        """
        if not os.path.isdir(benchmark_folder):
            raise ValueError(f"Not a directory: {benchmark_folder}")
        
        results = []
        total_correct = 0
        total_files = 0
        
        # Get all .txt files
        txt_files = [f for f in os.listdir(benchmark_folder) if f.endswith('.txt')]
        
        if verbose:
            print(f"\nEvaluating {len(txt_files)} DSL files in {benchmark_folder}")
            print(f"Expected result: {'SUCCESS' if expected_result else 'FAILURE'}")
            print("-"*70)
        
        for filename in sorted(txt_files):
            filepath = os.path.join(benchmark_folder, filename)
            total_files += 1
            
            if verbose:
                print(f"{filename:40s} ", end="")
            
            try:
                # Load and execute DSL
                from random_constr import Construction
                construction = Construction()
                construction.load(filepath)
                construction.generate(require_theorem=False, max_attempts=max_attempts)
                
                # Check if construction succeeded
                actual_result = True
                
                # If there's a prove statement, check its result
                if construction.to_prove is not None:
                    actual_result = construction.to_prove.data.b
                
                # Compare with expected
                is_correct = (actual_result == expected_result)
                
                if is_correct:
                    total_correct += 1
                    if verbose:
                        print("✓ CORRECT")
                else:
                    if verbose:
                        print(f"✗ WRONG (got {'SUCCESS' if actual_result else 'FAILURE'})")
                
                results.append({
                    "file": filename,
                    "expected": bool(expected_result),
                    "actual": bool(actual_result),
                    "correct": bool(is_correct),
                    "error": None
                })
                
            except Exception as e:
                if verbose:
                    print(f"✗ ERROR: {str(e)}")
                
                results.append({
                    "file": filename,
                    "expected": bool(expected_result),
                    "actual": None,
                    "correct": False,
                    "error": str(e)
                })
        
        # Create report
        accuracy = total_correct / total_files if total_files > 0 else 0.0
        
        if verbose:
            print("-"*70)
            print(f"Accuracy: {total_correct}/{total_files} ({100*accuracy:.1f}%)")
            print("="*70)
        
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            dataset_name=benchmark_folder,
            total_problems=total_files,
            evaluated_problems=total_files,
            successful_problems=total_correct,
            average_object_score=accuracy,
            average_condition_score=accuracy,
            average_total_score=accuracy,
            results=results,
            summary={
                "expected_result": expected_result,
                "accuracy": accuracy
            }
        )
        
        return report
    
    def _find_dsl_file(self, directory: str, problem_id: str) -> Optional[str]:
        """Find DSL file for a problem ID."""
        # Try common filename patterns
        patterns = [
            f"{problem_id}.txt",
            f"{problem_id}.dsl",
            f"problem_{problem_id}.txt",
            f"problem_{problem_id}.dsl"
        ]
        
        for pattern in patterns:
            filepath = os.path.join(directory, pattern)
            if os.path.exists(filepath):
                return filepath
        
        return None
    
    def _calculate_score_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of scores."""
        distribution = {
            "0-20": 0,
            "20-40": 0,
            "40-60": 0,
            "60-80": 0,
            "80-100": 0
        }
        
        for result in results:
            validation = result.get("validation", {})
            if validation.get("error_message") is None:
                score = validation.get("total_score", 0) * 100
                
                if score < 20:
                    distribution["0-20"] += 1
                elif score < 40:
                    distribution["20-40"] += 1
                elif score < 60:
                    distribution["40-60"] += 1
                elif score < 80:
                    distribution["60-80"] += 1
                else:
                    distribution["80-100"] += 1
        
        return distribution
    
    def _analyze_common_failures(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze common failure patterns."""
        missing_object_types = {}
        failed_condition_types = {}
        
        for result in results:
            validation = result.get("validation", {})
            
            # Analyze missing objects
            missing = validation.get("missing_objects", {})
            for obj_type, obj_list in missing.items():
                if len(obj_list) > 0:
                    missing_object_types[obj_type] = missing_object_types.get(obj_type, 0) + 1
            
            # Analyze failed conditions
            failed_conds = validation.get("failed_conditions", [])
            for cond in failed_conds:
                cond_type = cond.get("type", "unknown")
                failed_condition_types[cond_type] = failed_condition_types.get(cond_type, 0) + 1
        
        return {
            "missing_object_types": missing_object_types,
            "failed_condition_types": failed_condition_types
        }


def main():
    """Command-line interface for benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate geometry DSL benchmark")
    
    parser.add_argument("--mode", choices=["dataset", "benchmark"], default="benchmark",
                       help="Evaluation mode: dataset (with JSON) or benchmark (simple pass/fail)")
    
    parser.add_argument("--dsl-dir", type=str,
                       help="Directory containing DSL files")
    
    parser.add_argument("--dataset", type=str,
                       help="Path to benchmark dataset JSON")
    
    parser.add_argument("--expected", choices=["true", "false"], default="true",
                       help="Expected result for benchmark mode")
    
    parser.add_argument("--output", type=str, default="benchmark_report.json",
                       help="Output report file")
    
    parser.add_argument("--max-attempts", type=int, default=100,
                       help="Maximum attempts for construction generation")
    
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print verbose output")
    
    args = parser.parse_args()
    
    evaluator = BenchmarkEvaluator()
    
    if args.mode == "dataset":
        # Evaluate with dataset
        if not args.dsl_dir or not args.dataset:
            parser.error("--dsl-dir and --dataset required for dataset mode")
        
        dataset = BenchmarkDataset(args.dataset)
        report = evaluator.evaluate_dataset(
            args.dsl_dir, 
            dataset, 
            max_attempts=args.max_attempts,
            verbose=args.verbose
        )
    else:
        # Evaluate benchmark folder (true/false)
        if not args.dsl_dir:
            parser.error("--dsl-dir required for benchmark mode")
        
        expected = (args.expected == "true")
        report = evaluator.evaluate_benchmark_folder(
            args.dsl_dir,
            expected_result=expected,
            max_attempts=args.max_attempts,
            verbose=args.verbose
        )
    
    # Print summary
    report.print_summary()
    
    # Save report
    report.save(args.output)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()

