#!/usr/bin/env python3
"""
Rerun API-failed problems and rebuild ablation results.

Given an ablation run directory (containing ablation.log and result JSONs),
this script:
  1) Finds problems that failed due to API issues (timeouts, rate limits,
     upstream provider errors, invalid/empty responses) using both the
     ablation log and the saved results JSON files.
  2) Re-runs only those problems for the corresponding model/mode using
     run_agent_benchmark.py --problem-id, logging all reruns to rerun.log.
  3) Merges rerun results with the original reports and recomputes metrics.
  4) Writes merged per-model reports (JSON) alongside rerun.log and regenerates
     analyzer outputs (text/csv/json + summaries) into a summary/ subdirectory.

Usage:
  python scripts/rerun_api_failures.py --run-dir benchmark_results/vision_ablation_20251213_200810

Optional flags:
  --dataset <path>       Dataset to use (default: data/geoqa3_dataset.json)
  --max-iter N           Max iterations per problem (default: 5)
  --output-dir <path>    Output directory for merged reports
  --no-save-images       Do not save intermediate images on rerun
  --verbose              Print rerun commands
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import select

# Allow imports from project root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

API_ERROR_KEYWORDS = [
    "request timed out",
    "timed out",
    "429",
    "rate-limit",
    "rate limit",
    "temporarily rate-limited",
    "provider returned error",
    "bad gateway",
    "upstream",
    "context deadline exceeded",
    "no instances available",
    "connection reset",
    "connection aborted",
    "503",
    "expecting value",
    "nonetype",
]


def build_env_for_runs() -> Dict[str, str]:
    """Build environment with OpenRouter fallbacks for OpenAI-compatible client."""
    env = os.environ.copy()
    openrouter_key = env.get("OPENROUTER_API_KEY")
    openrouter_base = env.get("OPENROUTER_API_BASE")
    if openrouter_key and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = openrouter_key
    if openrouter_base:
        env.setdefault("OPENAI_API_BASE", openrouter_base)
        env.setdefault("OPENROUTER_API_BASE", openrouter_base)
    return env


def run_command(
    cmd: List[str],
    log_path: Path,
    env: Dict[str, str],
    allow_error: bool = False,
    timeout_seconds: Optional[float] = None,
) -> int:
    """Run a command, streaming stdout/stderr to console and rerun log. Returns exit code."""
    printable_cmd = " ".join(cmd)
    print(f"\n$ {printable_cmd}")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {printable_cmd}\n")
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        start_time = time.monotonic()
        stdout_stream = process.stdout

        def _timed_out() -> bool:
            return (
                timeout_seconds is not None
                and (time.monotonic() - start_time) > timeout_seconds
            )

        while True:
            ready, _, _ = select.select([stdout_stream], [], [], 0.5)
            if ready:
                line = stdout_stream.readline()
                if not line:
                    break
                print(line, end="")
                log_file.write(line)

            if process.poll() is not None and not ready:
                break

            if _timed_out():
                timeout_msg = f"[warn] Command exceeded {timeout_seconds}s; terminating."
                print(timeout_msg)
                log_file.write(timeout_msg + "\n")
                process.kill()
                break

        process.wait()
        rc = process.returncode
        if rc != 0:
            warn = f"[warn] Command exited with {rc}"
            print(warn)
            log_file.write(warn + "\n")
            if not allow_error:
                raise subprocess.CalledProcessError(rc, cmd)
        return rc


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _is_api_error(msg: str, error_keywords: List[str]) -> bool:
    """Check if an error message looks like an API/backend failure."""
    lowered = msg.lower()
    return any(k in lowered for k in error_keywords)


def parse_failures_from_results(
    run_dir: Path, error_keywords: List[str]
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Inspect saved results_*.json files to find problems that failed due to API issues.

    Returns:
        {model: {"with_vision": set(problem_ids), "no_vision": set(problem_ids)}}
    """
    failures: Dict[str, Dict[str, Set[str]]] = {}
    for path in sorted(run_dir.glob("results_*_vision.json")):
        try:
            report = load_results(path)
        except Exception:
            continue

        metadata = report.get("metadata", {})
        model = (
            metadata.get("model")
            or path.name.split("results_", 1)[-1].rsplit("_", 2)[0]
        )
        vision_mode = metadata.get("vision_mode") or (
            "with_vision" if metadata.get("use_vision") else "no_vision"
        )

        for res in report.get("results", []):
            if res.get("success"):
                continue
            error_msg = res.get("error")
            if not error_msg:
                continue
            if _is_api_error(str(error_msg), error_keywords):
                pid = str(res.get("problem_id"))
                failures.setdefault(model, {}).setdefault(vision_mode, set()).add(pid)

    return failures


def merge_failure_maps(
    *maps: Dict[str, Dict[str, Set[str]]]
) -> Dict[str, Dict[str, Set[str]]]:
    """Merge multiple {model: {mode: set(ids)}} dicts."""
    merged: Dict[str, Dict[str, Set[str]]] = {}
    for mapping in maps:
        for model, modes in mapping.items():
            for mode, ids in modes.items():
                merged.setdefault(model, {}).setdefault(mode, set()).update(ids)
    return merged


def parse_failures_from_log(
    log_path: Path, error_keywords: List[str]
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Parse ablation.log to find problems that failed at Iteration 1 due to API issues.

    Returns:
        {model: {"with_vision": set(problem_ids), "no_vision": set(problem_ids)}}
    """
    failures: Dict[str, Dict[str, Set[str]]] = {}
    if not log_path.exists():
        return failures

    current_model: Optional[str] = None
    current_mode: Optional[str] = None
    current_problem: Optional[str] = None
    current_iteration: Optional[int] = None

    mode_from_line = {
        "Vision Mode: ✓ Enabled": "with_vision",
        "Vision Mode: ✗ Disabled": "no_vision",
    }

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = _strip_ansi(raw_line).strip()

            # Detect model from banner
            m_model = re.search(r"=== Benchmarking (.+) ===", line)
            if m_model:
                current_model = m_model.group(1).strip()
                current_mode = None
                current_problem = None
                current_iteration = None
                continue

            # Detect model from command line invocation
            if "run_agent_benchmark.py" in line and "--model" in line:
                m_cmd_model = re.search(r"--model\s+(\S+)", line)
                if m_cmd_model:
                    current_model = m_cmd_model.group(1).strip().strip("'\"")
                current_mode = None
                current_problem = None
                current_iteration = None
                continue

            for key, val in mode_from_line.items():
                if key in line:
                    current_mode = val
                    current_problem = None
                    current_iteration = None
                    break

            m_prob = re.search(r"\[\d+\/\d+\]\s+Problem\s+([^\s]+)", line)
            if m_prob:
                current_problem = m_prob.group(1)
                # Assume iteration 1 unless overwritten
                current_iteration = 1
                continue

            m_iter = re.search(r"--- Iteration\s+(\d+)\/\d+\s+---", line)
            if m_iter:
                current_iteration = int(m_iter.group(1))
                continue

            if (
                current_model
                and current_mode
                and current_problem
                and "ERROR:" in line
                and any(k in line.lower() for k in error_keywords)
            ):
                if current_iteration is None or current_iteration == 1:
                    failures.setdefault(current_model, {}).setdefault(
                        current_mode, set()
                    ).add(current_problem)

    return failures


def load_results(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def rerun_problem(
    model: str,
    problem_id: str,
    dataset: Path,
    max_iter: int,
    use_vision: bool,
    save_images: bool,
    workdir: Path,
    log_path: Path,
    env: Dict[str, str],
    verbose: bool,
    timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    """Run a single problem via run_agent_benchmark.py and return the result dict."""
    tmp_out = (
        workdir / f"tmp_{model.replace('/','__').replace(':','__')}_{problem_id}.json"
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_agent_benchmark.py"),
        "--problem-id",
        problem_id,
        "--dataset",
        str(dataset),
        "--model",
        model,
        "--max-iter",
        str(max_iter),
        "--output",
        str(tmp_out),
    ]
    if verbose:
        cmd.append("--verbose")
    if not save_images:
        cmd.append("--no-save-images")
    if not use_vision:
        cmd.append("--no-vision")

    rc = run_command(
        cmd, log_path, env, allow_error=True, timeout_seconds=timeout_seconds
    )
    if not tmp_out.exists():
        raise RuntimeError(
            f"Rerun produced no output file for problem {problem_id} (rc={rc})"
        )

    data = load_results(tmp_out)
    tmp_out.unlink(missing_ok=True)
    return data.get("result") or data  # single-problem output uses "result"


def recompute_report(
    combined_results: List[Dict[str, Any]],
    model: str,
    dataset: Path,
    max_iter: int,
    use_vision: bool,
    run_id: str,
    base_metadata: Optional[Dict[str, Any]] = None,
    validation_errors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recompute metrics and build a report matching the batch structure."""
    from run_agent_benchmark import (
        DetailedMetrics,
    )  # Local import to avoid heavy deps on parse

    metrics = DetailedMetrics()
    metrics.reset()

    for res in combined_results:
        if res.get("skipped"):
            metrics.add_skipped_problem(
                res.get("problem_id", "unknown"),
                res.get("skip_reason", "Skipped"),
                res.get("dataset_error_types", []),
            )
            continue

        memory_data = None
        mem_path = res.get("memory_path")
        if mem_path and Path(mem_path).exists():
            try:
                with open(mem_path, "r", encoding="utf-8") as f:
                    memory_data = json.load(f)
            except Exception:
                memory_data = None

        metrics.add_problem_result(res, memory_data)

    summary = metrics.get_summary()
    vision_mode = "with_vision" if use_vision else "no_vision"

    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "dataset": str(dataset),
            "max_iterations": max_iter,
            "use_vision": use_vision,
            "vision_mode": vision_mode,
            "run_id": run_id,
        }
    )

    report = {
        "metadata": metadata,
        "metrics": summary,
        "detailed_analysis": {
            "success_rate": {
                "value": summary["success_rate"],
                "successful": summary["successful_problems"],
                "failed": summary["failed_problems"],
                "skipped": summary["skipped_problems"],
                "total": summary["total_problems"],
                "note": "Success rate excludes skipped problems",
            },
            "step_analysis": {
                "average_steps_to_success": summary["average_success_steps"],
                "min_steps_to_success": summary["min_success_steps"],
                "max_steps_to_success": summary["max_success_steps"],
                "step_distribution": summary["success_step_distribution"],
            },
            "hallucination_analysis": {
                "total_hallucinations": summary["total_hallucinations"],
                "average_per_problem": summary["average_hallucinations_per_problem"],
                "average_recovery_steps": summary[
                    "average_hallucination_recovery_steps"
                ],
                "error_type_distribution": summary["hallucination_error_types"],
            },
            "object_analysis": {
                "total_missing_objects": summary["total_missing_objects"],
                "missing_by_type": summary["missing_objects_by_type"],
            },
            "condition_analysis": {
                "total_failed_conditions": summary["total_failed_conditions"],
                "failed_by_type": summary["failed_conditions_by_type"],
            },
            "validation_error_analysis": {
                "total_errors": summary["validation_errors_count"],
                "errors_by_type": summary["validation_errors_by_type"],
                "unknown_condition_types": summary["unknown_condition_types"],
            },
        },
        "results": combined_results,
        "problem_details": metrics.problem_details,
    }
    if validation_errors is not None:
        report["validation_errors"] = validation_errors
    return report


def merge_results(
    orig_report: Dict[str, Any],
    rerun_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Replace results for rerun problem_ids and return combined list.

    rerun_results: {problem_id: result_dict}
    """
    combined = []
    for res in orig_report.get("results", []):
        pid = str(res.get("problem_id"))
        if pid in rerun_results:
            combined.append(rerun_results[pid])
        else:
            combined.append(res)
    # Add any rerun results missing from original
    for pid, res in rerun_results.items():
        if not any(str(r.get("problem_id")) == pid for r in combined):
            combined.append(res)
    return combined


def run_analyzer(
    with_path: Path,
    no_path: Path,
    out_dir: Path,
    log_path: Path,
    env: Dict[str, str],
    model_safe: str,
) -> None:
    """Run analyze_benchmark_results in all formats and log output."""
    analyzer = REPO_ROOT / "scripts" / "analyze_benchmark_results.py"
    base = out_dir
    base.mkdir(parents=True, exist_ok=True)

    base_name = f"vision_ablation_report_{model_safe}"
    run_command(
        [
            sys.executable,
            str(analyzer),
            str(with_path),
            str(no_path),
            "--format",
            "text",
            "--output",
            str(base / f"{base_name}.txt"),
            "--per-problem",
        ],
        log_path,
        env,
    )
    run_command(
        [
            sys.executable,
            str(analyzer),
            str(with_path),
            str(no_path),
            "--format",
            "csv",
            "--output",
            str(base / f"{base_name}.csv"),
        ],
        log_path,
        env,
    )
    run_command(
        [
            sys.executable,
            str(analyzer),
            str(with_path),
            str(no_path),
            "--format",
            "json",
            "--output",
            str(base / f"{base_name}.json"),
        ],
        log_path,
        env,
    )


def collect_metrics(result_path: Path) -> Dict[str, float]:
    """Extract key metrics for summarization."""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    return {
        "success_rate": metrics.get("success_rate", 0.0),
        "average_success_steps": metrics.get("average_success_steps", 0.0),
        "average_hallucinations_per_problem": metrics.get(
            "average_hallucinations_per_problem", 0.0
        ),
        "total_missing_objects": metrics.get("total_missing_objects", 0),
        "total_failed_conditions": metrics.get("total_failed_conditions", 0),
        "total_problems": metrics.get("total_problems", 0),
        "successful_problems": metrics.get("successful_problems", 0),
        "failed_problems": metrics.get("failed_problems", 0),
        "skipped_problems": metrics.get("skipped_problems", 0),
    }


def write_summary(
    with_vision: Dict[str, float],
    no_vision: Dict[str, float],
    summary_path: Path,
) -> None:
    """Write a simple text summary comparing vision vs no-vision."""

    def pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    lines = []
    lines.append("Vision Ablation Summary")
    lines.append("-" * 72)
    lines.append(f"Results file: {summary_path.parent}")
    lines.append(
        f"Problems evaluated (vision):   {with_vision['total_problems']} "
        f"(skipped: {with_vision['skipped_problems']})"
    )
    lines.append(
        f"Problems evaluated (no vision): {no_vision['total_problems']} "
        f"(skipped: {no_vision['skipped_problems']})"
    )
    lines.append("")
    header = f"{'Metric':<32}{'With vision':>15}{'No vision':>15}{'Diff':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(
        f"{'Success rate':<32}"
        f"{pct(with_vision['success_rate']):>15}"
        f"{pct(no_vision['success_rate']):>15}"
        f"{(with_vision['success_rate'] - no_vision['success_rate']) * 100:>11.1f} pp"
    )
    lines.append(
        f"{'Avg steps to success':<32}"
        f"{with_vision['average_success_steps']:>15.2f}"
        f"{no_vision['average_success_steps']:>15.2f}"
        f"{with_vision['average_success_steps'] - no_vision['average_success_steps']:>12.2f}"
    )
    lines.append(
        f"{'Avg hallucinations/problem':<32}"
        f"{with_vision['average_hallucinations_per_problem']:>15.2f}"
        f"{no_vision['average_hallucinations_per_problem']:>15.2f}"
        f"{with_vision['average_hallucinations_per_problem'] - no_vision['average_hallucinations_per_problem']:>12.2f}"
    )
    lines.append(
        f"{'Total missing objects':<32}"
        f"{with_vision['total_missing_objects']:>15d}"
        f"{no_vision['total_missing_objects']:>15d}"
        f"{with_vision['total_missing_objects'] - no_vision['total_missing_objects']:>12d}"
    )
    lines.append(
        f"{'Total failed conditions':<32}"
        f"{with_vision['total_failed_conditions']:>15d}"
        f"{no_vision['total_failed_conditions']:>15d}"
        f"{with_vision['total_failed_conditions'] - no_vision['total_failed_conditions']:>12d}"
    )

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text + "\n")


def write_single_summary(
    metrics: Dict[str, float], summary_path: Path, label: str
) -> None:
    """Write summary when only one mode was run."""

    def pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    lines = []
    lines.append(f"{label} Summary")
    lines.append("-" * 72)
    lines.append(f"Results file: {summary_path.parent}")
    lines.append(
        f"Problems evaluated: {metrics['total_problems']} (skipped: {metrics['skipped_problems']})"
    )
    lines.append("")
    lines.append(f"Success rate: {pct(metrics['success_rate'])}")
    lines.append(f"Avg steps to success: {metrics['average_success_steps']:.2f}")
    lines.append(
        f"Avg hallucinations/problem: {metrics['average_hallucinations_per_problem']:.2f}"
    )
    lines.append(f"Total missing objects: {metrics['total_missing_objects']}")
    lines.append(f"Total failed conditions: {metrics['total_failed_conditions']}")

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun API failures and rebuild ablation results"
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path, help="Original ablation run directory"
    )
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "data" / "geoqa3_dataset.json"
    )
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <run-dir>_rerun_<ts>)",
    )
    parser.add_argument(
        "--no-save-images", action="store_true", help="Do not save images during reruns"
    )
    parser.add_argument("--verbose", action="store_true", help="Print rerun commands")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Kill a single rerun if it exceeds this many seconds (default: 900)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    orig_log_path = run_dir / "ablation.log"
    if not orig_log_path.exists():
        raise SystemExit(f"ablation.log not found in {run_dir}")

    env = build_env_for_runs()

    log_failures = parse_failures_from_log(orig_log_path, API_ERROR_KEYWORDS)
    result_failures = parse_failures_from_results(run_dir, API_ERROR_KEYWORDS)
    failures = merge_failure_maps(log_failures, result_failures)
    if not failures:
        print(
            "No API-related failures found; copying originals and regenerating summaries."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or run_dir.parent / f"{run_dir.name}_rerun_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rerun_log = out_dir / "rerun.log"
    rerun_log.touch()

    # Seed rerun directory with original results so files exist before updates
    for orig_path in run_dir.glob("results_*_vision.json"):
        dest_path = out_dir / orig_path.name
        if not dest_path.exists():
            shutil.copy(orig_path, dest_path)

    # Copy original artifacts for reference
    shutil.copy(orig_log_path, out_dir / "ablation.log.orig")

    with open(rerun_log, "a", encoding="utf-8") as log_file:
        log_file.write(f"Rerun started at {datetime.now().isoformat()}\n")
        if failures:
            log_file.write(f"Detected failures for {len(failures)} model(s).\n")
        else:
            log_file.write("No failures detected; no reruns performed.\n")

    for model, modes in failures.items():
        model_safe = model.replace("/", "__").replace(":", "__")
        for vision_mode, pids in modes.items():
            if not pids:
                continue

            use_vision = vision_mode == "with_vision"
            print(
                f"\nModel: {model} | Mode: {vision_mode} | Rerunning {len(pids)} problems"
            )
            with open(rerun_log, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n[{model} | {vision_mode}] Rerunning {len(pids)} problems: {sorted(pids)}\n"
                )

            src_file = run_dir / f"results_{model_safe}_{vision_mode}.json"
            out_file = out_dir / src_file.name
            if not src_file.exists() and not out_file.exists():
                msg = f"Skipping: source results not found: {src_file}"
                print(f"  {msg}")
                with open(rerun_log, "a", encoding="utf-8") as log_file:
                    log_file.write(msg + "\n")
                continue

            if not out_file.exists() and src_file.exists():
                shutil.copy(src_file, out_file)

            orig_report = load_results(out_file if out_file.exists() else src_file)

            rerun_map: Dict[str, Dict[str, Any]] = {}
            run_id = f"{model_safe}_{vision_mode}_patched_{timestamp}"

            def write_incremental_report() -> None:
                merged_results = merge_results(orig_report, rerun_map)
                updated_report = recompute_report(
                    merged_results,
                    model=model,
                    dataset=args.dataset,
                    max_iter=args.max_iter,
                    use_vision=use_vision,
                    run_id=run_id,
                    base_metadata=orig_report.get("metadata"),
                    validation_errors=orig_report.get("validation_errors"),
                )
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(updated_report, f, ensure_ascii=False, indent=2)

            # Ensure a file exists before reruns and keep it updated per-problem
            write_incremental_report()

            for pid in sorted(
                pids, key=lambda x: int(x) if str(x).isdigit() else str(x)
            ):
                try:
                    res = rerun_problem(
                        model=model,
                        problem_id=str(pid),
                        dataset=args.dataset,
                        max_iter=args.max_iter,
                        use_vision=use_vision,
                        save_images=not args.no_save_images,
                        workdir=out_dir,
                        log_path=rerun_log,
                        env=env,
                        verbose=args.verbose,
                        timeout_seconds=args.timeout_seconds,
                    )
                    rerun_map[str(pid)] = res
                    write_incremental_report()
                except Exception as exc:
                    msg = f"Rerun failed for problem {pid}: {exc}. Keeping original result."
                    print(f"  {msg}")
                    with open(rerun_log, "a", encoding="utf-8") as log_file:
                        log_file.write(msg + "\n")

    # Build summary outputs under summary/
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    model_files: Dict[str, Dict[str, Path]] = {}
    for path in sorted(out_dir.glob("results_*_vision.json")):
        try:
            report = load_results(path)
        except Exception:
            continue
        metadata = report.get("metadata", {})
        model = metadata.get("model")
        if not model:
            continue
        vision_mode = metadata.get("vision_mode") or (
            "with_vision" if metadata.get("use_vision") else "no_vision"
        )
        model_files.setdefault(model, {})[vision_mode] = path

    for model, modes in model_files.items():
        model_safe = model.replace("/", "__").replace(":", "__")
        with_path = modes.get("with_vision")
        no_path = modes.get("no_vision")
        if with_path and no_path:
            run_analyzer(with_path, no_path, summary_dir, rerun_log, env, model_safe)
            metrics_with = collect_metrics(with_path)
            metrics_without = collect_metrics(no_path)
            write_summary(
                metrics_with,
                metrics_without,
                summary_dir / f"vision_ablation_summary_{model_safe}.txt",
            )
        elif with_path:
            metrics_with = collect_metrics(with_path)
            write_single_summary(
                metrics_with,
                summary_dir / f"vision_ablation_summary_{model_safe}.txt",
                "Vision-only",
            )
        elif no_path:
            metrics_without = collect_metrics(no_path)
            write_single_summary(
                metrics_without,
                summary_dir / f"vision_ablation_summary_{model_safe}.txt",
                "No-vision-only",
            )

    print(f"\nPatched results and summaries written to: {out_dir}")
    print(f"Summary directory: {summary_dir}")
    print(f"Rerun log: {rerun_log}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user; partial results preserved.")
        sys.exit(130)
