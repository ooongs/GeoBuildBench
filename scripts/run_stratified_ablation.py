#!/usr/bin/env python3
"""Run vision ablation on a stratified sample from previous run results.

This script:
1. Reads a run_info.json file from a previous benchmark run
2. Samples problems with stratification by success/failure (default 70% success, 30% failure)
3. Creates a temporary dataset with only the sampled problems
4. Runs vision ablation on the sampled dataset
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "geoqa3_dataset.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run vision ablation on stratified sample from previous results"
    )
    parser.add_argument(
        "--run-info",
        type=Path,
        default=REPO_ROOT / "agent_logs" / "gpt-5.1_vision_20251219_160112" / "run_info.json",
        help="Path to run_info.json from previous benchmark run",
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
        "--max-iter",
        type=int,
        default=5,
        help="Maximum reasoning iterations per problem",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark_results",
        help="Directory to store ablation outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose through to the vision ablation script",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Disable saving intermediate images during benchmarking",
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Run only the vision-enabled pass (skip no-vision)",
    )
    parser.add_argument(
        "--no-vision-only",
        action="store_true",
        help="Run only the no-vision pass (skip vision-enabled)",
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
    print(f"Loading run info from: {run_info_path}")
    with open(run_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_sample(
    problems: List[Dict],
    n: int,
    success_ratio: float,
    seed: int = 42,
) -> List[str]:
    """Sample problems with stratification by success/failure.

    Args:
        problems: List of problem dicts with 'problem_id' and 'success' fields
        n: Total number of problems to sample
        success_ratio: Ratio of successful problems (e.g., 0.7 for 70%)
        seed: Random seed for reproducibility

    Returns:
        List of problem IDs to test
    """
    random.seed(seed)

    # Separate by success/failure
    successful = [p for p in problems if p.get("success", False)]
    failed = [p for p in problems if not p.get("success", False)]

    print(f"\nDataset statistics:")
    print(f"  Total problems: {len(problems)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(problems)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(problems)*100:.1f}%)")

    # Calculate sample sizes
    n_success = int(n * success_ratio)
    n_fail = n - n_success

    # Adjust if we don't have enough problems in either category
    if n_success > len(successful):
        print(f"  Warning: Requested {n_success} successful problems but only {len(successful)} available")
        n_success = len(successful)
        n_fail = n - n_success

    if n_fail > len(failed):
        print(f"  Warning: Requested {n_fail} failed problems but only {len(failed)} available")
        n_fail = len(failed)
        n_success = n - n_fail

    print(f"\nSampling strategy:")
    print(f"  Target: {n} problems ({success_ratio*100:.0f}% success, {(1-success_ratio)*100:.0f}% failure)")
    print(f"  Actual: {n_success} successful + {n_fail} failed = {n_success + n_fail} total")

    # Sample
    sampled_success = random.sample(successful, n_success)
    sampled_failed = random.sample(failed, n_fail)

    # Combine and shuffle
    sampled = sampled_success + sampled_failed
    random.shuffle(sampled)

    problem_ids = [p["problem_id"] for p in sampled]

    print(f"\nSampled {len(problem_ids)} problems:")
    print(f"  Success: {len(sampled_success)}")
    print(f"  Failed: {len(sampled_failed)}")
    print(f"  Problem IDs: {sorted(problem_ids)[:10]}..." if len(problem_ids) > 10 else f"  Problem IDs: {sorted(problem_ids)}")

    return problem_ids


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
    print(f"  Problem IDs: {problem_id_set}")
    filtered = [p for p in original.get("problems", []) if str(p.get("id", "")) in problem_id_set]

    print(f"  Original dataset: {len(original)} problems")
    print(f"  Filtered dataset: {len(filtered)} problems")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write filtered dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"  Filtered dataset saved to: {output_path}")


def run_vision_ablation(
    dataset_path: Path,
    args: argparse.Namespace,
) -> None:
    """Run the vision_ablation.py script on the filtered dataset."""
    ablation_script = REPO_ROOT / "scripts" / "vision_ablation.py"

    cmd = [
        sys.executable,
        str(ablation_script),
        "--model", args.model,
        "--dataset", str(dataset_path),
        "--max-iter", str(args.max_iter),
        "--output-dir", str(args.output_dir),
    ]

    if args.verbose:
        cmd.append("--verbose")
    if args.no_save_images:
        cmd.append("--no-save-images")
    if args.vision_only:
        cmd.append("--vision-only")
    if args.no_vision_only:
        cmd.append("--no-vision-only")

    print(f"\nRunning vision ablation:")
    print(f"  Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    args = parse_args()

    # Load run info
    run_info = load_run_info(args.run_info)
    problems = run_info.get("problems", [])

    if not problems:
        print("Error: No problems found in run_info.json")
        return 1

    # Sample problems
    sampled_ids = stratified_sample(
        problems,
        args.n,
        args.success_ratio,
        args.seed,
    )

    # Create filtered dataset
    temp_dataset = REPO_ROOT / "data" / "temp_stratified_sample.json"
    create_filtered_dataset(args.dataset, sampled_ids, temp_dataset)

    try:
        # Run vision ablation
        run_vision_ablation(temp_dataset, args)
    finally:
        # Clean up temporary dataset
        if temp_dataset.exists():
            print(f"\nCleaning up temporary dataset: {temp_dataset}")
            temp_dataset.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
