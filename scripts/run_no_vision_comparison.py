#!/usr/bin/env python3
"""Run no-vision benchmark on sampled problems and compare with existing vision results.

This script:
1. Reads a run_info.json file from a previous vision-enabled benchmark run
2. Filters problems with iterations >= 2 (excludes first-attempt successes)
3. Samples problems with stratification by success/failure (default 70% success, 30% failure)
4. Runs no-vision benchmark on sampled problems only
5. Compares results with the existing vision results
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "geoqa3_dataset.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run no-vision benchmark and compare with existing vision results"
    )
    parser.add_argument(
        "--vision-run-info",
        type=Path,
        default=REPO_ROOT / "agent_logs" / "gpt-5.1_vision_20251219_160112" / "run_info.json",
        help="Path to run_info.json from previous vision-enabled benchmark run",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Original dataset file (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="Model to test (default: gpt-5.1)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Total number of problems to sample (default: 100)",
    )
    parser.add_argument(
        "--success-ratio",
        type=float,
        default=0.7,
        help="Ratio of successful problems to sample (default: 0.7)",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=2,
        help="Minimum iterations to include (default: 2, excludes first-attempt successes)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum reasoning iterations per problem",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark_results",
        help="Directory to store comparison outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose through to the benchmark runner",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Disable saving intermediate images during benchmarking",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    return parser.parse_args()


def load_run_info(run_info_path: Path) -> Dict:
    """Load and parse run_info.json."""
    print(f"Loading vision run info from: {run_info_path}")
    with open(run_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_and_stratified_sample(
    problems: List[Dict],
    n: int,
    success_ratio: float,
    min_iterations: int,
    seed: int = 42,
) -> tuple[List[str], Dict[str, Dict]]:
    """Filter by iterations and sample problems with stratification by success/failure.

    Args:
        problems: List of problem dicts with 'problem_id', 'success', and 'iterations' fields
        n: Total number of problems to sample
        success_ratio: Ratio of successful problems (e.g., 0.7 for 70%)
        min_iterations: Minimum iterations to include
        seed: Random seed for reproducibility

    Returns:
        Tuple of (problem_ids, problem_details_dict)
    """
    random.seed(seed)

    # Filter by iterations
    filtered_problems = [p for p in problems if p.get("iterations", 0) >= min_iterations]

    print(f"\nFiltering by iterations:")
    print(f"  Total problems in vision run: {len(problems)}")
    print(f"  Problems with iterations >= {min_iterations}: {len(filtered_problems)}")

    if not filtered_problems:
        raise ValueError(f"No problems found with iterations >= {min_iterations}")

    # Separate by success/failure
    successful = [p for p in filtered_problems if p.get("success", False)]
    failed = [p for p in filtered_problems if not p.get("success", False)]

    print(f"\nFiltered dataset statistics:")
    print(f"  Successful: {len(successful)} ({len(successful)/len(filtered_problems)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(filtered_problems)*100:.1f}%)")

    # Calculate sample sizes
    n_success = int(n * success_ratio)
    n_fail = n - n_success

    # Adjust if we don't have enough problems in either category
    if n_success > len(successful):
        print(f"  Warning: Requested {n_success} successful problems but only {len(successful)} available")
        n_success = len(successful)
        n_fail = min(n - n_success, len(failed))

    if n_fail > len(failed):
        print(f"  Warning: Requested {n_fail} failed problems but only {len(failed)} available")
        n_fail = len(failed)
        n_success = min(n - n_fail, len(successful))

    actual_total = n_success + n_fail
    print(f"\nSampling strategy:")
    print(f"  Target: {n} problems ({success_ratio*100:.0f}% success, {(1-success_ratio)*100:.0f}% failure)")
    print(f"  Actual: {n_success} successful + {n_fail} failed = {actual_total} total")

    # Sample
    sampled_success = random.sample(successful, n_success) if n_success > 0 else []
    sampled_failed = random.sample(failed, n_fail) if n_fail > 0 else []

    # Combine and shuffle
    sampled = sampled_success + sampled_failed
    random.shuffle(sampled)

    problem_ids = [p["problem_id"] for p in sampled]
    problem_details = {p["problem_id"]: p for p in sampled}

    print(f"\nSampled {len(problem_ids)} problems:")
    print(f"  Success: {len(sampled_success)}")
    print(f"  Failed: {len(sampled_failed)}")
    print(f"  Problem IDs (first 20): {sorted(problem_ids)[:20]}")
    if len(problem_ids) > 20:
        print(f"  ... and {len(problem_ids) - 20} more")

    return problem_ids, problem_details


def create_filtered_dataset(
    original_dataset_path: Path,
    problem_ids: List[str],
    output_path: Path,
) -> None:
    """Create a new dataset containing only the specified problems.

    Args:
        original_dataset_path: Path to the original dataset JSON
        problem_ids: List of problem IDs to include
        output_path: Path to write the filtered dataset
    """
    print(f"\nCreating filtered dataset at: {output_path}")

    with open(original_dataset_path, "r", encoding="utf-8") as f:
        original = json.load(f)

    # Filter to only include sampled problems
    problem_id_set = set(problem_ids)

    # Handle both list format and dict with "problems" key
    if isinstance(original, list):
        filtered = [p for p in original if str(p.get("id", "")) in problem_id_set]
    else:
        original_list = original.get("problems", original)
        filtered = [p for p in original_list if str(p.get("id", "")) in problem_id_set]

    print(f"  Filtered dataset: {len(filtered)} problems")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write filtered dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"  Filtered dataset saved to: {output_path}")


def run_no_vision_benchmark(
    dataset_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    """Run the benchmark with --no-vision flag."""
    benchmark_script = REPO_ROOT / "run_agent_benchmark.py"

    cmd = [
        sys.executable,
        str(benchmark_script),
        "--batch",
        "--model", args.model,
        "--dataset", str(dataset_path),
        "--max-iter", str(args.max_iter),
        "--output", str(output_path),
        "--no-vision",
    ]

    if args.verbose:
        cmd.append("--verbose")
    if args.no_save_images:
        cmd.append("--no-save-images")

    print(f"\nRunning no-vision benchmark:")
    print(f"  Command: {' '.join(cmd)}")
    print()

    env = os.environ.copy()
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def compare_results(
    vision_details: Dict[str, Dict],
    no_vision_result_path: Path,
    output_dir: Path,
    model_name: str,
) -> None:
    """Compare vision and no-vision results and generate reports."""
    print(f"\nComparing vision vs no-vision results...")

    # Load no-vision results
    with open(no_vision_result_path, "r", encoding="utf-8") as f:
        no_vision_data = json.load(f)

    no_vision_problems = {p["problem_id"]: p for p in no_vision_data.get("problems", [])}

    # Prepare comparison data
    comparisons = []

    for problem_id in vision_details:
        vision_result = vision_details[problem_id]
        no_vision_result = no_vision_problems.get(problem_id, {})

        comparison = {
            "problem_id": problem_id,
            "vision": {
                "success": vision_result.get("success", False),
                "iterations": vision_result.get("iterations", 0),
            },
            "no_vision": {
                "success": no_vision_result.get("success", False),
                "iterations": no_vision_result.get("iterations", 0),
            },
            "delta_success": (no_vision_result.get("success", False) - vision_result.get("success", False)),
            "delta_iterations": (no_vision_result.get("iterations", 0) - vision_result.get("iterations", 0)),
        }
        comparisons.append(comparison)

    # Calculate aggregate metrics
    total = len(comparisons)
    vision_success_count = sum(1 for c in comparisons if c["vision"]["success"])
    no_vision_success_count = sum(1 for c in comparisons if c["no_vision"]["success"])

    both_success = sum(1 for c in comparisons if c["vision"]["success"] and c["no_vision"]["success"])
    only_vision_success = sum(1 for c in comparisons if c["vision"]["success"] and not c["no_vision"]["success"])
    only_no_vision_success = sum(1 for c in comparisons if not c["vision"]["success"] and c["no_vision"]["success"])
    both_fail = sum(1 for c in comparisons if not c["vision"]["success"] and not c["no_vision"]["success"])

    vision_avg_iterations = sum(c["vision"]["iterations"] for c in comparisons) / total if total > 0 else 0
    no_vision_avg_iterations = sum(c["no_vision"]["iterations"] for c in comparisons) / total if total > 0 else 0

    # Write summary report
    summary_path = output_dir / f"vision_comparison_{model_name.replace('/', '__')}.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("Vision vs No-Vision Comparison Report")
    lines.append("=" * 80)
    lines.append(f"Model: {model_name}")
    lines.append(f"Total problems evaluated: {total}")
    lines.append(f"Filtered criteria: iterations >= 2 (excludes first-attempt successes)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("Success Rate Comparison")
    lines.append("-" * 80)
    lines.append(f"Vision-enabled success rate:    {vision_success_count}/{total} ({vision_success_count/total*100:.1f}%)")
    lines.append(f"No-vision success rate:         {no_vision_success_count}/{total} ({no_vision_success_count/total*100:.1f}%)")
    lines.append(f"Difference:                     {(no_vision_success_count - vision_success_count)/total*100:+.1f} pp")
    lines.append("")
    lines.append("-" * 80)
    lines.append("Success Pattern Breakdown")
    lines.append("-" * 80)
    lines.append(f"Both succeeded:                 {both_success} ({both_success/total*100:.1f}%)")
    lines.append(f"Only vision succeeded:          {only_vision_success} ({only_vision_success/total*100:.1f}%)")
    lines.append(f"Only no-vision succeeded:       {only_no_vision_success} ({only_no_vision_success/total*100:.1f}%)")
    lines.append(f"Both failed:                    {both_fail} ({both_fail/total*100:.1f}%)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("Iteration Count Comparison")
    lines.append("-" * 80)
    lines.append(f"Vision average iterations:      {vision_avg_iterations:.2f}")
    lines.append(f"No-vision average iterations:   {no_vision_avg_iterations:.2f}")
    lines.append(f"Difference:                     {no_vision_avg_iterations - vision_avg_iterations:+.2f}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("Per-Problem Comparison (showing cases where results differ)")
    lines.append("-" * 80)
    lines.append(f"{'Problem ID':<12} {'Vision':<15} {'No-Vision':<15} {'Change':<20}")
    lines.append("-" * 80)

    for c in sorted(comparisons, key=lambda x: (not x["vision"]["success"], x["problem_id"])):
        if c["vision"]["success"] != c["no_vision"]["success"] or abs(c["delta_iterations"]) > 0:
            vision_str = f"{'✓' if c['vision']['success'] else '✗'} ({c['vision']['iterations']} iter)"
            no_vision_str = f"{'✓' if c['no_vision']['success'] else '✗'} ({c['no_vision']['iterations']} iter)"

            if c["delta_success"] > 0:
                change = "Improved (no-vision)"
            elif c["delta_success"] < 0:
                change = "Degraded (no-vision)"
            elif c["delta_iterations"] > 0:
                change = f"+{c['delta_iterations']} iterations"
            elif c["delta_iterations"] < 0:
                change = f"{c['delta_iterations']} iterations"
            else:
                change = "No change"

            lines.append(f"{c['problem_id']:<12} {vision_str:<15} {no_vision_str:<15} {change:<20}")

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n{summary_text}")
    print(f"\nSummary written to: {summary_path}")

    # Write detailed JSON comparison
    json_path = output_dir / f"vision_comparison_{model_name.replace('/', '__')}.json"
    comparison_data = {
        "model": model_name,
        "total_problems": total,
        "summary": {
            "vision_success_rate": vision_success_count / total if total > 0 else 0,
            "no_vision_success_rate": no_vision_success_count / total if total > 0 else 0,
            "both_success": both_success,
            "only_vision_success": only_vision_success,
            "only_no_vision_success": only_no_vision_success,
            "both_fail": both_fail,
            "vision_avg_iterations": vision_avg_iterations,
            "no_vision_avg_iterations": no_vision_avg_iterations,
        },
        "comparisons": comparisons,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)

    print(f"Detailed JSON comparison written to: {json_path}")


def main() -> int:
    args = parse_args()

    # Load vision run info
    vision_run_info = load_run_info(args.vision_run_info)
    vision_problems = vision_run_info.get("problems", [])

    if not vision_problems:
        print("Error: No problems found in vision run_info.json")
        return 1

    # Filter and sample problems
    sampled_ids, vision_details = filter_and_stratified_sample(
        vision_problems,
        args.n,
        args.success_ratio,
        args.min_iterations,
        args.seed,
    )

    if not sampled_ids:
        print("Error: No problems sampled")
        return 1

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"no_vision_comparison_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create filtered dataset
    temp_dataset = run_dir / "sampled_problems.json"
    create_filtered_dataset(args.dataset, sampled_ids, temp_dataset)

    # Run no-vision benchmark
    no_vision_output = run_dir / f"results_{args.model.replace('/', '__')}_no_vision.json"

    try:
        run_no_vision_benchmark(temp_dataset, no_vision_output, args)
    except subprocess.CalledProcessError as e:
        print(f"\nError running no-vision benchmark: {e}")
        return 1

    # Compare results
    compare_results(vision_details, no_vision_output, run_dir, args.model)

    print(f"\nAll artifacts written to: {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
