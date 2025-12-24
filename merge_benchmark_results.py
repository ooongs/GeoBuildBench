#!/usr/bin/env python3
"""
Merge two benchmark result files.

Usage:
    python merge_benchmark_results.py \
        --original benchmark_results/run_20251219_160111/results_gpt-5.1.json \
        --resume agent_logs/gpt-5.1_vision_20251219_160112/result.json \
        --output merged_results.json
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Set
from run_agent_benchmark import DetailedMetrics


def merge_benchmark_results(original_file: str, resume_file: str, output_file: str, verbose: bool = False):
    """
    Merge original and resume benchmark results.

    Args:
        original_file: Path to original (interrupted) results file
        resume_file: Path to resume results file (retried problems only)
        output_file: Path to save merged results
        verbose: Print detailed progress
    """
    print(f"📊 Merging benchmark results...")
    print(f"   Original: {original_file}")
    print(f"   Resume:   {resume_file}")
    print(f"   Output:   {output_file}")
    print()

    # 1. Load both files
    with open(original_file, 'r', encoding='utf-8') as f:
        original = json.load(f)

    with open(resume_file, 'r', encoding='utf-8') as f:
        resume = json.load(f)

    # 2. Get problem IDs from each file
    original_ids = {r.get("problem_id") for r in original.get("results", []) if r.get("problem_id")}
    resume_ids = {r.get("problem_id") for r in resume.get("results", []) if r.get("problem_id")}

    print(f"📈 Original file: {len(original_ids)} problems")
    print(f"🔄 Resume file: {len(resume_ids)} problems")
    print(f"🆕 New/retried: {len(resume_ids - original_ids)} problems")
    print(f"♻️  Replaced: {len(resume_ids & original_ids)} problems")
    print()

    # 3. Merge results array (problem_id as key)
    result_dict = {}

    # Add all original results first
    for result in original.get("results", []):
        pid = result.get("problem_id")
        if pid:
            result_dict[pid] = result

    # Overwrite with resume results (newer data)
    for result in resume.get("results", []):
        pid = result.get("problem_id")
        if pid:
            result_dict[pid] = result
            if verbose:
                status = "✅" if result.get("success") else "❌"
                print(f"  {status} Problem {pid}: {result.get('iterations', 0)} iterations")

    # 4. Merge problem_details
    details_dict = {}

    for detail in original.get("problem_details", []):
        pid = detail.get("problem_id")
        if pid:
            details_dict[pid] = detail

    for detail in resume.get("problem_details", []):
        pid = detail.get("problem_id")
        if pid:
            details_dict[pid] = detail

    # 5. Merge validation_errors
    validation_errors_dict = {}

    for err in original.get("validation_errors", {}).get("errors", []):
        pid = err.get("problem_id")
        if pid:
            if pid not in validation_errors_dict:
                validation_errors_dict[pid] = []
            validation_errors_dict[pid].append(err)

    for err in resume.get("validation_errors", {}).get("errors", []):
        pid = err.get("problem_id")
        if pid:
            # Replace validation errors for retried problems
            if pid in resume_ids:
                validation_errors_dict[pid] = [err]
            elif pid not in validation_errors_dict:
                validation_errors_dict[pid] = [err]
            else:
                validation_errors_dict[pid].append(err)

    # 6. Convert back to lists
    merged_results = list(result_dict.values())
    merged_details = list(details_dict.values())
    merged_val_errors = [err for errs in validation_errors_dict.values() for err in errs]

    print(f"📦 Merged totals:")
    print(f"   Results: {len(merged_results)}")
    print(f"   Details: {len(merged_details)}")
    print(f"   Validation errors: {len(merged_val_errors)}")
    print()

    # 7. Recalculate all metrics from merged data
    print(f"🔢 Recalculating metrics from merged data...")
    merged_metrics = DetailedMetrics()

    for result in merged_results:
        # Try to load memory_data if available
        memory_data = None
        if result.get("memory_path") and os.path.exists(result["memory_path"]):
            try:
                with open(result["memory_path"], 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load memory for {result.get('problem_id')}: {e}")

        # Handle skipped problems
        if result.get("skipped"):
            merged_metrics.add_skipped_problem(
                problem_id=result["problem_id"],
                reason=result.get("skip_reason", "Unknown"),
                error_types=result.get("validation_result", {}).get("dataset_error_types", [])
            )
        else:
            merged_metrics.add_problem_result(result, memory_data)

    merged_summary = merged_metrics.get_summary()

    # 8. Build final merged report
    merged_report = {
        "metadata": {
            **original.get("metadata", {}),
            "merged_at": datetime.now().isoformat(),
            "original_file": original_file,
            "resume_file": resume_file,
            "original_problems": len(original_ids),
            "resume_problems": len(resume_ids),
            "retried_problems": sorted(list(resume_ids & original_ids)),
            "new_problems": sorted(list(resume_ids - original_ids)),
            "total_problems_evaluated": len(merged_results),
        },
        "metrics": merged_summary,
        "detailed_analysis": {
            "success_rate": {
                "value": merged_summary["success_rate"],
                "successful": merged_summary["successful_problems"],
                "failed": merged_summary["failed_problems"],
                "skipped": merged_summary["skipped_problems"],
                "total": merged_summary["total_problems"],
                "note": "Success rate is calculated excluding skipped problems"
            },
            "step_analysis": {
                "average_steps_to_success": merged_summary["average_success_steps"],
                "min_steps_to_success": merged_summary["min_success_steps"],
                "max_steps_to_success": merged_summary["max_success_steps"],
                "step_distribution": merged_summary["success_step_distribution"]
            },
            "hallucination_analysis": {
                "total_hallucinations": merged_summary["total_hallucinations"],
                "average_per_problem": merged_summary["average_hallucinations_per_problem"],
                "average_recovery_steps": merged_summary["average_hallucination_recovery_steps"],
                "error_type_distribution": merged_summary["hallucination_error_types"]
            },
            "object_analysis": {
                "total_missing_objects": merged_summary["total_missing_objects"],
                "missing_by_type": merged_summary["missing_objects_by_type"]
            },
            "condition_analysis": {
                "total_failed_conditions": merged_summary["total_failed_conditions"],
                "failed_by_type": merged_summary["failed_conditions_by_type"]
            },
            "validation_error_analysis": {
                "total_errors": merged_summary["validation_errors_count"],
                "errors_by_type": merged_summary["validation_errors_by_type"],
                "unknown_condition_types": merged_summary["unknown_condition_types"]
            }
        },
        "results": merged_results,
        "problem_details": merged_details,
        "validation_errors": {
            "summary": {
                "total_errors": len(merged_val_errors),
                "unique_problems_with_errors": len(validation_errors_dict)
            },
            "errors": merged_val_errors
        }
    }

    # Preserve validation_error_log_file if it exists
    if "validation_error_log_file" in original:
        merged_report["validation_error_log_file"] = original["validation_error_log_file"]
    if "validation_error_log_file" in resume:
        merged_report["validation_error_log_file"] = resume["validation_error_log_file"]

    # 9. Save merged results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_report, f, ensure_ascii=False, indent=2)

    print(f"✅ Merged results saved to: {output_file}")
    print()
    print(f"📊 Final Statistics:")
    print(f"   Total Problems: {merged_summary['total_problems']}")
    print(f"   Successful: {merged_summary['successful_problems']} ({merged_summary['success_rate']:.1%})")
    print(f"   Failed: {merged_summary['failed_problems']}")
    print(f"   Skipped: {merged_summary['skipped_problems']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge original and resume benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python merge_benchmark_results.py \\
    --original benchmark_results/run_20251219_160111/results_gpt-5.1.json \\
    --resume agent_logs/gpt-5.1_vision_20251219_160112/result.json \\
    --output merged_results.json
        """
    )

    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original (interrupted) results file"
    )

    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Path to resume results file (retried problems only)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged results"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.original):
        print(f"❌ Error: Original file not found: {args.original}")
        return 1

    if not os.path.exists(args.resume):
        print(f"❌ Error: Resume file not found: {args.resume}")
        return 1

    # Merge results
    try:
        merge_benchmark_results(
            original_file=args.original,
            resume_file=args.resume,
            output_file=args.output,
            verbose=args.verbose
        )
        return 0
    except Exception as e:
        print(f"❌ Error merging results: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
