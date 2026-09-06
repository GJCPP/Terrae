#!/usr/bin/env python3

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import csv
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark_common import REPO_ROOT, TRAINING_SEMANTICS_FINGERPRINT, _environment_flag
from benchmark_focus_extract import (
    SUMMARY_FIELDNAMES,
    SOUNDNESS_SUMMARY_FIELDNAMES,
    TERRAE_SUMMARY_FIELDNAMES,
    ZKBOOST_SUMMARY_FIELDNAMES,
    collect_checkpoint_summary_rows,
    collect_summary_rows,
    complete_case_summary_rows,
    write_summary_csv,
    write_soundness_summary_csv,
    write_terrae_summary_csv,
    write_zkboost_summary_csv,
)
from benchmark_standard import _prepare_build


FOCUS_SCRIPT = REPO_ROOT / "tools" / "python_trace_export" / "benchmark_focus_extract.py"
MEMORY_WRAPPER = REPO_ROOT / "tools" / "measure_peak_memory.py"
DEFAULT_TIMEOUT_SECONDS = 7200.0
DEFAULT_JOBS = 1


@dataclass(frozen=True)
class PlanCase:
    order: int
    case_id: str
    phase: str
    dataset: str
    N: int
    T: int
    d: int
    bins: int
    b: int
    routes: str
    optional: bool = False
    gradient_mode: str = "first-order-rss"
    learning_rate_numerator: int = 4
    learning_rate_denominator: int = 1
    split_gain_threshold: int = 1


def _is_terminal_status(status: str) -> bool:
    return status in {"completed", "timed_out", "unsupported"}


_MEMORY_GUARD_STATUSES = {
    "swap_detected",
    "oom_detected",
    "memory_pressure_detected",
    "rss_limit_detected",
}

_STOPPABLE_STAGE_SELECTORS = (
    "terrae-training",
    "zkboost-full-training",
    "zkboost-inference",
)


def _stage_selector(stage: str) -> str | None:
    if stage.startswith("terrae-training"):
        return "terrae-training"
    if stage.startswith("zkboost-full-training"):
        return "zkboost-full-training"
    if stage.startswith("zkboost-inference"):
        return "zkboost-inference"
    return None


def _guarded_stage_selectors(case_output_dir: Path) -> list[str]:
    selectors: set[str] = set()
    for path in case_output_dir.rglob("*_memory.json"):
        measurement = _read_json_object(path)
        if measurement.get("status") not in _MEMORY_GUARD_STATUSES:
            continue
        selector = _stage_selector(path.name)
        if selector is not None:
            selectors.add(selector)
    return sorted(selectors)


def _blocked_stage_selectors(
    blocked_stages: dict[tuple[str, str, str], tuple[int, str]],
    case: PlanCase,
) -> list[str]:
    return [
        stage
        for (dataset, routes, stage), (failed_t, _) in blocked_stages.items()
        if dataset == case.dataset and routes == case.routes and case.T > failed_t
    ]


def _message_indicates_timeout(message: str) -> bool:
    return "timed out after " in message


def _message_indicates_unsupported(message: str) -> bool:
    return "too large for size_t-based range tables" in message


def _default_plan() -> list[PlanCase]:
    return [
        PlanCase(1, "dry_bean_smoke_N256_T1_d2_b256_bins8", "calibration", "dry-bean", 256, 1, 2, 8, 256, "terrae+zkboost"),
        PlanCase(2, "dry_bean_base_N512_T1_d2_b512_bins8", "calibration", "dry-bean", 512, 1, 2, 8, 512, "terrae+zkboost"),
        PlanCase(3, "dry_bean_depth_check_N512_T1_d3_b512_bins8", "calibration", "dry-bean", 512, 1, 3, 8, 512, "terrae+zkboost"),
        PlanCase(4, "dry_bean_T2_N512_d2_b512_bins8", "T-scale", "dry-bean", 512, 2, 2, 8, 512, "terrae+zkboost"),
        PlanCase(5, "dry_bean_T4_N512_d2_b512_bins8", "T-scale", "dry-bean", 512, 4, 2, 8, 512, "terrae+zkboost"),
        PlanCase(6, "dry_bean_T8_N512_d2_b512_bins8", "T-scale", "dry-bean", 512, 8, 2, 8, 512, "terrae+zkboost", True),
        PlanCase(7, "dry_bean_depth3_N512_T2_b512_bins8", "depth-scale", "dry-bean", 512, 2, 3, 8, 512, "terrae+zkboost"),
        PlanCase(8, "dry_bean_depth4_N512_T2_b512_bins8", "depth-scale", "dry-bean", 512, 2, 4, 8, 512, "terrae+zkboost", True),
        PlanCase(9, "dry_bean_N1024_T2_d3_b1024_bins8", "N-scale", "dry-bean", 1024, 2, 3, 8, 1024, "terrae+zkboost", True),
        PlanCase(10, "dry_bean_infer_b128_N512_T2_d3_bins8", "inference-scale", "dry-bean", 512, 2, 3, 8, 128, "terrae+zkboost"),
        PlanCase(11, "dry_bean_infer_b256_N512_T2_d3_bins8", "inference-scale", "dry-bean", 512, 2, 3, 8, 256, "terrae+zkboost"),
        PlanCase(12, "dry_bean_infer_b512_N512_T2_d3_bins8", "inference-scale", "dry-bean", 512, 2, 3, 8, 512, "terrae+zkboost"),
        # The current trace cropper rejects b whose power-of-two padding exceeds N.
        # Keep the intended b=1024 point explicit in the design doc, but do not run it
        # by default until the runner supports exporting a larger inference batch than
        # the training subset.
    ]


def _load_cases(path: Path | None) -> list[PlanCase]:
    if path is None:
        return _default_plan()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("case json root must be a list")
    cases: list[PlanCase] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case = PlanCase(
            order=int(item.get("order", index + 1)),
            case_id=str(item["case_id"]),
            phase=str(item.get("phase", "custom")),
            dataset=str(item.get("dataset", "dry-bean")),
            N=int(item["N"]),
            T=int(item["T"]),
            d=int(item["d"]),
            bins=int(item.get("bins", item.get("bin_count", 8))),
            b=int(item.get("b", item["N"])),
            routes=str(item.get("routes", "terrae+zkboost")),
            optional=bool(item.get("optional", False)),
            gradient_mode=str(item.get("loss_mode", item.get("gradient_mode", "first-order-rss"))),
            learning_rate_numerator=int(item.get("learning_rate_numerator", 4)),
            learning_rate_denominator=int(item.get("learning_rate_denominator", 1)),
            split_gain_threshold=int(item.get("split_gain_threshold", 1)),
        )
        if case.routes not in {"terrae", "zkboost", "terrae+zkboost"}:
            raise ValueError(
                f"{case.case_id}: routes must be terrae, zkboost, or terrae+zkboost"
            )
        if case.gradient_mode not in {"first-order-rss", "second-order-xgboost"}:
            raise ValueError(
                f"{case.case_id}: loss_mode must be first-order-rss or second-order-xgboost"
            )
        if case.learning_rate_numerator <= 0 or case.learning_rate_denominator <= 0:
            raise ValueError(f"{case.case_id}: learning-rate numerator and denominator must be positive")
        if case.split_gain_threshold < 0:
            raise ValueError(f"{case.case_id}: split_gain_threshold must be nonnegative")
        cases.append(case)
    return sorted(cases, key=lambda case: case.order)


def _safe_output_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return safe or "benchmark_plan"


def _default_output_dir(*, cases_json: Path | None, phase1: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if phase1:
        return REPO_ROOT / "out" / f"benchmark_phase_1_{timestamp}"
    label = "default_plan" if cases_json is None else _safe_output_name(cases_json.stem)
    return REPO_ROOT / "out" / f"{label}_{timestamp}"


def _normalize_output_dir(output_dir: Path | None, *, cases_json: Path | None, phase1: bool) -> Path:
    default_dir = _default_output_dir(cases_json=cases_json, phase1=phase1)
    if output_dir is None:
        return default_dir

    if output_dir.is_absolute():
        candidate = output_dir
    else:
        candidate = REPO_ROOT / output_dir

    if candidate.resolve() == (REPO_ROOT / "out").resolve():
        return default_dir
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a benchmark parameter plan with checkpoint resume. "
            "Plans may come from the built-in defaults or an external JSON case list."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Benchmark output directory. If omitted, the runner creates a new timestamped "
            "subdirectory under out/. If set to the bare out/ root, the runner also creates "
            "a timestamped subdirectory there automatically."
        ),
    )
    parser.add_argument(
        "--cases-json",
        type=Path,
        help="Optional JSON list overriding the built-in plan cases.",
    )
    parser.add_argument(
        "--phase",
        action="append",
        help=(
            "Run only selected phase(s). Repeatable. Default: all phases in the loaded plan. "
            "Aliases: phase1/phase-1 -> calibration."
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help=(
            "Run only the selected case id(s). Repeatable and composable with phase/order "
            "filters. Unknown ids are rejected."
        ),
    )
    parser.add_argument(
        "--phase1",
        "--phase-1",
        action="store_true",
        help="Run the phase 1 calibration suite and default output to out/benchmark_phase_1.",
    )
    parser.add_argument("--start-order", type=int, default=1)
    parser.add_argument("--stop-order", type=int)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument(
        "--rerun-stage",
        action="append",
        choices=["terrae-training", "zkboost-full-training", "zkboost-inference"],
        default=[],
        help=(
            "Rerun the selected stage even for completed cases while reusing all other "
            "successful stage checkpoints. Repeatable."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true", default=False)
    parser.add_argument("--keep-going", action="store_false", dest="fail_fast")
    parser.add_argument(
        "--stop-subsequent-t-after-failure",
        action="store_true",
        help=(
            "Run cases sequentially and, after a proof stage fails, times out, or hits "
            "a recorded limit, skip only that stage at larger T for the same dataset "
            "and route. Independent proof stages still run."
        ),
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build-release")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--loss-mode",
        "--gradient-mode",
        dest="gradient_mode",
        choices=["first-order-rss", "second-order-xgboost"],
        help="Override the per-case training loss; cases default to first-order-rss.",
    )
    parser.add_argument(
        "--srs-cache-dir",
        type=Path,
        help="Directory shared by per-case runners for reusable KZG SRS files.",
    )
    parser.add_argument("--initial-srs-deg1-plus-1", type=int, default=1 << 17)
    parser.add_argument("--initial-srs-deg2-plus-1", type=int, default=1 << 17)
    parser.add_argument(
        "--assume-srs-deg1-plus-1",
        type=int,
        help=(
            "Skip per-case SRS requirement queries and assume this G1 SRS size. "
            "Use only when a matching cached SRS is known to cover every selected case."
        ),
    )
    parser.add_argument(
        "--assume-srs-deg2-plus-1",
        type=int,
        help=(
            "Skip per-case SRS requirement queries and assume this G2 SRS size. "
            "Use only with --assume-srs-deg1-plus-1."
        ),
    )
    parser.add_argument(
        "--max-srs-deg1-plus-1",
        type=int,
        help=(
            "Optional hard cap for automatic SRS retry growth on the G1 side. "
            "Default: unbounded."
        ),
    )
    parser.add_argument(
        "--max-srs-deg2-plus-1",
        type=int,
        help=(
            "Optional hard cap for automatic SRS retry growth on the G2 side. "
            "Default: unbounded."
        ),
    )
    timeout_group = parser.add_mutually_exclusive_group()
    timeout_group.add_argument(
        "--timeout-seconds",
        type=float,
        dest="timeout_seconds",
        help=f"Per-stage timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS:g}.",
    )
    timeout_group.add_argument(
        "--no-timeout",
        action="store_const",
        const=None,
        dest="timeout_seconds",
        help="Disable build and per-stage wall-clock time limits.",
    )
    parser.set_defaults(timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--record-timed-out-stage",
        action="append",
        choices=["zkboost-full-training"],
        default=[],
        help="Record a timeout from an earlier run without executing that stage again.",
    )
    parser.add_argument(
        "--reuse-existing-stages",
        action="store_true",
        help="Reuse successful stage payloads from each case's existing workflow report.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="Number of benchmark case workers to use. Default: 1.",
    )
    parser.add_argument("--state-file", type=Path)
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.stop_subsequent_t_after_failure and args.jobs != 1:
        raise ValueError("--stop-subsequent-t-after-failure requires --jobs 1")
    if args.rerun_stage and not args.reuse_existing_stages:
        raise ValueError("--rerun-stage requires --reuse-existing-stages")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if (args.assume_srs_deg1_plus_1 is None) != (args.assume_srs_deg2_plus_1 is None):
        raise ValueError("--assume-srs-deg1-plus-1 and --assume-srs-deg2-plus-1 must be provided together")
    if args.assume_srs_deg1_plus_1 is not None and (
        args.assume_srs_deg1_plus_1 <= 0 or args.assume_srs_deg2_plus_1 <= 0
    ):
        raise ValueError("assumed SRS sizes must be positive")
    if args.phase1 and not args.phase:
        args.phase = ["calibration"]
    args.phase = _normalize_phases(args.phase)
    args.output_dir = _normalize_output_dir(
        args.output_dir,
        cases_json=args.cases_json,
        phase1=args.phase1,
    )
    return args


def _normalize_phases(phases: list[str] | None) -> list[str] | None:
    if phases is None:
        return None
    aliases = {
        "phase1": "calibration",
        "phase-1": "calibration",
        "phase_1": "calibration",
    }
    return [aliases.get(phase, phase) for phase in phases]


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"cases": {}, "events": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _normalize_timeout_entries(state: dict[str, Any]) -> None:
    for entry in state.get("cases", {}).values():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "failed":
            continue
        message = str(entry.get("message", ""))
        if _message_indicates_timeout(message):
            entry["status"] = "timed_out"
            entry["timed_out_at"] = entry.get("failed_at") or datetime.now().isoformat(timespec="seconds")
            continue
        if _message_indicates_unsupported(message):
            entry["status"] = "unsupported"
            entry["unsupported_at"] = entry.get("failed_at") or datetime.now().isoformat(timespec="seconds")


def _case_filter(cases: list[PlanCase], args: argparse.Namespace) -> list[PlanCase]:
    selected: list[PlanCase] = []
    allowed_phases = set(args.phase or [])
    allowed_case_ids = set(args.case_ids or [])
    unknown_case_ids = allowed_case_ids - {case.case_id for case in cases}
    if unknown_case_ids:
        raise ValueError(
            "unknown --case-id value(s): " + ", ".join(sorted(unknown_case_ids))
        )
    for case in cases:
        if allowed_case_ids and case.case_id not in allowed_case_ids:
            continue
        if case.order < args.start_order:
            continue
        if args.stop_order is not None and case.order > args.stop_order:
            continue
        if allowed_phases and case.phase not in allowed_phases:
            continue
        if case.optional and not args.include_optional:
            continue
        selected.append(case)
    return selected


def _run_case(
    case: PlanCase,
    args: argparse.Namespace,
    case_output_dir: Path,
    skipped_stages: tuple[str, ...] = (),
) -> dict[str, Any]:
    case_output_dir.mkdir(parents=True, exist_ok=True)
    case_log = case_output_dir / "case_runner.log"
    case_memory_json = case_output_dir / "case_memory_guard.json"
    command = [
        sys.executable,
        str(FOCUS_SCRIPT),
        "--output-dir",
        str(case_output_dir),
        "--summary-md-name",
        "focus_benchmark_summary.md",
        "--summary-csv-name",
        "focus_benchmark_summary.csv",
        "--build-dir",
        str(args.build_dir),
        "--routes",
        case.routes,
        "--loss-mode",
        args.gradient_mode or case.gradient_mode,
        "--dataset",
        case.dataset,
        "--learning-rate-numerator",
        str(case.learning_rate_numerator),
        "--learning-rate-denominator",
        str(case.learning_rate_denominator),
        "--split-gain-threshold",
        str(case.split_gain_threshold),
        "--N",
        str(case.N),
        "--T",
        str(case.T),
        "--d",
        str(case.d),
        "--b",
        str(case.b),
        "--bin-count",
        str(case.bins),
        "--case-id",
        case.case_id,
        "--srs-initial-deg1-plus-1",
        str(args.initial_srs_deg1_plus_1),
        "--srs-initial-deg2-plus-1",
        str(args.initial_srs_deg2_plus_1),
    ]
    if args.timeout_seconds is None:
        command.append("--no-timeout")
    else:
        command.extend(["--timeout-seconds", str(args.timeout_seconds)])
    if args.max_srs_deg1_plus_1 is not None:
        command.extend([
            "--max-srs-deg1-plus-1",
            str(args.max_srs_deg1_plus_1),
        ])
    if args.max_srs_deg2_plus_1 is not None:
        command.extend([
            "--max-srs-deg2-plus-1",
            str(args.max_srs_deg2_plus_1),
        ])
    if args.assume_srs_deg1_plus_1 is not None:
        command.extend([
            "--assume-srs-deg1-plus-1",
            str(args.assume_srs_deg1_plus_1),
            "--assume-srs-deg2-plus-1",
            str(args.assume_srs_deg2_plus_1),
        ])
    if args.srs_cache_dir is not None:
        command.extend([
            "--srs-cache-dir",
            str(args.srs_cache_dir),
        ])
    for stage in args.record_timed_out_stage:
        command.extend(["--record-timed-out-stage", stage])
    if args.reuse_existing_stages:
        command.append("--reuse-existing-stages")
    for stage in args.rerun_stage:
        command.extend(["--rerun-stage", stage])
    for stage in skipped_stages:
        command.extend(["--skip-stage", stage])
    if args.skip_build or args.jobs > 1:
        command.append("--skip-build")
    abort_on_swap = _environment_flag("BENCHMARK_ABORT_ON_SWAP", None)
    abort_on_oom = _environment_flag("BENCHMARK_ABORT_ON_OOM", None)
    memory_guard_options = [
        ("BENCHMARK_RSS_LIMIT_BYTES", "--rss-limit-bytes"),
        ("BENCHMARK_SWAP_LIMIT_BYTES", "--swap-limit-bytes"),
        ("BENCHMARK_SWAP_LIMIT_DURATION_SECONDS", "--swap-limit-duration-seconds"),
        ("BENCHMARK_MIN_MEM_AVAILABLE_BYTES", "--min-mem-available-bytes"),
        (
            "BENCHMARK_MIN_MEM_AVAILABLE_DURATION_SECONDS",
            "--min-mem-available-duration-seconds",
        ),
    ]
    memory_guard_args = [
        item
        for environment_name, option_name in memory_guard_options
        if (value := os.environ.get(environment_name, "").strip())
        for item in (option_name, value)
    ]
    if abort_on_swap or abort_on_oom or memory_guard_args:
        command = [
            sys.executable,
            str(MEMORY_WRAPPER),
            "--output-json",
            str(case_memory_json),
            *(["--abort-on-swap"] if abort_on_swap else []),
            *(["--abort-on-oom"] if abort_on_oom else []),
            *memory_guard_args,
            "--",
            *command,
        ]
    try:
        with case_log.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n[{datetime.now().isoformat(timespec='seconds')}] "
                f"start {case.case_id} on {threading.current_thread().name}\n"
            )
            log_handle.write("command: " + " ".join(command) + "\n")
            log_handle.flush()
            subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            log_handle.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"finish {case.case_id}\n"
            )
    except subprocess.CalledProcessError as error:
        if case_memory_json.exists():
            try:
                memory = json.loads(case_memory_json.read_text(encoding="utf-8"))
                if memory.get("status") in _MEMORY_GUARD_STATUSES:
                    message = (
                        f"{case.case_id}: case memory guard triggered"
                        f"; memory_status={memory.get('status')}"
                        f"; peak_rss_bytes={memory.get('peak_rss_bytes', 'unknown')}"
                        f"; peak_swap_bytes={memory.get('peak_swap_bytes', 'unknown')}"
                        f"; memory_guard_trigger={memory.get('memory_guard_trigger', 'unknown')}"
                        f"; measurement_json={case_memory_json}"
                    )
                    # This is an expected resource limit for the parameter point,
                    # not a harness failure.  Keep the measured peak in the
                    # aggregate report and proceed to the remaining cases.
                    return {
                        "status": "unsupported",
                        "message": message,
                        "worker": threading.current_thread().name,
                        "case_log": str(case_log),
                        "failed_stages": _guarded_stage_selectors(case_output_dir),
                    }
            except (OSError, ValueError, TypeError):
                pass
        workflow_report = case_output_dir / "workflow_benchmark_report.json"
        if workflow_report.exists():
            try:
                report = json.loads(workflow_report.read_text(encoding="utf-8"))
                failed_cases = [
                    item for item in report.get("cases", [])
                    if item.get("status") != "completed"
                ]
                if failed_cases:
                    failed = failed_cases[0]
                    failed_case = failed.get("case", {}).get("case_id", case.case_id)
                    failed_message = failed.get("error_message", "").strip()
                    if failed_message:
                        raise RuntimeError(
                            f"{failed_case}: {failed_message}"
                        ) from error
            except (OSError, ValueError, TypeError):
                pass
        raise RuntimeError(f"{case.case_id}: benchmark subprocess failed; see {case_log}") from error
    report_status = "completed"
    report_message = ""
    failed_stages: set[str] = set()
    workflow_report = case_output_dir / "workflow_benchmark_report.json"
    if workflow_report.exists():
        try:
            report = json.loads(workflow_report.read_text(encoding="utf-8"))
            for item in report.get("cases", []):
                case_payload = item.get("case", {})
                if case_payload.get("case_id", case.case_id) != case.case_id:
                    continue
                status = str(item.get("status", "completed"))
                message = str(item.get("error_message", "")).strip()
                stage_failures = item.get("benchmark_payload", {}).get("stage_failures", [])
                if isinstance(stage_failures, list):
                    for failure in stage_failures:
                        if not isinstance(failure, dict):
                            continue
                        if failure.get("status") == "skipped_after_failure":
                            continue
                        selector = _stage_selector(str(failure.get("stage", "")))
                        if selector is not None:
                            failed_stages.add(selector)
                if status == "failed" and _message_indicates_timeout(message):
                    status = "timed_out"
                if status == "failed" and _message_indicates_unsupported(message):
                    status = "unsupported"
                if status != "completed":
                    report_status = status
                    report_message = message
                break
        except (OSError, ValueError, TypeError):
            pass
    return {
        "status": report_status,
        "message": report_message,
        "worker": threading.current_thread().name,
        "case_log": str(case_log),
        "failed_stages": sorted(failed_stages),
    }


def _phase_order(cases: list[PlanCase]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for case in cases:
        if case.phase in seen:
            continue
        seen.add(case.phase)
        ordered.append(case.phase)
    return ordered


def _print_phase_banner(phase: str, cases: list[PlanCase], phase_index: int, phase_count: int) -> None:
    phase_cases = [case for case in cases if case.phase == phase]
    first_order = phase_cases[0].order
    last_order = phase_cases[-1].order
    print("", flush=True)
    print("=" * 72, flush=True)
    print(
        f"[benchmark-plan] Phase {phase_index}/{phase_count} `{phase}` starting: "
        f"{len(phase_cases)} case(s), order {first_order}-{last_order}",
        flush=True,
    )
    print("=" * 72, flush=True)


def _phase_progress(phase: str, selected_cases: list[PlanCase]) -> tuple[int, int]:
    ordered_phases = _phase_order(selected_cases)
    return ordered_phases.index(phase) + 1, len(ordered_phases)


def _case_progress(case: PlanCase, selected_cases: list[PlanCase]) -> tuple[int, int]:
    ordered_cases = sorted(selected_cases, key=lambda item: item.order)
    for index, item in enumerate(ordered_cases, start=1):
        if item.case_id == case.case_id:
            return index, len(ordered_cases)
    return 1, len(ordered_cases)


def _print_case_banner(case: PlanCase, selected_cases: list[PlanCase]) -> None:
    phase_index, phase_count = _phase_progress(case.phase, selected_cases)
    case_index, case_count = _case_progress(case, selected_cases)
    optional = " optional" if case.optional else ""
    print(
        f"[benchmark-plan] phase {phase_index}/{phase_count}, case {case_index}/{case_count}: "
        f"{case.case_id} "
        f"(dataset={case.dataset}, N={case.N}, T={case.T}, d={case.d}, b={case.b}, bins={case.bins}, "
        f"routes={case.routes}, loss={case.gradient_mode}{optional})",
        flush=True,
    )


def _terminal_case_count(selected_cases: list[PlanCase], state: dict[str, Any]) -> int:
    return sum(
        1
        for case in selected_cases
        if _is_terminal_status(state.get("cases", {}).get(case.case_id, {}).get("status", ""))
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _failure_peak_memory_bytes(entry: dict[str, Any], case_output_dir: Path) -> int | None:
    peaks: list[int] = []
    for match in re.finditer(r"peak_rss_bytes=(\d+)", str(entry.get("message", ""))):
        peaks.append(int(match.group(1)))
    measurement = _read_json_object(case_output_dir / "case_memory_guard.json")
    measured_peak = measurement.get("peak_rss_bytes")
    if isinstance(measured_peak, int):
        peaks.append(measured_peak)
    log_paths = {case_output_dir / "case_runner.log"}
    if entry.get("case_log"):
        log_paths.add(Path(entry["case_log"]))
    for log_path in log_paths:
        if not log_path.exists():
            continue
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in re.finditer(r"peak_rss_bytes[=:](?:\s*)?(\d+)", log_text):
            peaks.append(int(match.group(1)))
    return max(peaks) if peaks else None


def _case_summary_metadata(
    case: PlanCase,
    checkpoint: dict[str, Any],
    case_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    adaptation = checkpoint.get("dataset_adaptation", {})
    if not isinstance(adaptation, dict):
        adaptation = {}
    checkpoint_case = checkpoint.get("case", {})
    metadata: dict[str, Any] = {
        "case_id": case.case_id,
        "dataset": case.dataset,
        "N_requested": case.N,
        "N_used": adaptation.get("actual_training_original_samples", ""),
        "N_padded": adaptation.get("padded_training_samples", ""),
        "b_requested": case.b,
        "b_padded": adaptation.get("padded_inference_samples", ""),
        "T": case.T,
        "d": case.d,
        "bin_count": case.bins,
        "feature_count": adaptation.get("feature_count", ""),
        "real_class_count": adaptation.get("real_class_count", ""),
        "learning_rate": (
            f"{checkpoint_case.get('learning_rate_numerator', case.learning_rate_numerator)}/"
            f"{checkpoint_case.get('learning_rate_denominator', case.learning_rate_denominator)}"
        ),
    }
    for row in case_rows:
        for field in SUMMARY_FIELDNAMES:
            if metadata.get(field, "") == "" and row.get(field, "") != "":
                metadata[field] = row[field]
    return metadata


def _case_checkpoint_path(case_output_dir: Path, case_id: str) -> Path | None:
    preferred = case_output_dir / "cases" / case_id / "stage_checkpoint.json"
    if preferred.exists():
        return preferred
    candidates = sorted((case_output_dir / "cases").glob("*/stage_checkpoint.json"))
    return candidates[0] if len(candidates) == 1 else None


def _collect_case_rows(
    output_dir: Path,
    case: PlanCase,
    entry: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path]:
    case_status = str(entry.get("status", "not_run"))
    case_output_dir = Path(
        entry.get("output_dir")
        or output_dir / "cases" / f"{case.order:02d}_{case.case_id}"
    )
    failure_peak = (
        _failure_peak_memory_bytes(entry, case_output_dir)
        if case_status != "completed"
        else None
    )
    checkpoint_path = _case_checkpoint_path(case_output_dir, case.case_id)
    checkpoint = _read_json_object(checkpoint_path) if checkpoint_path is not None else {}

    case_rows: list[dict[str, Any]] = []
    workflow_path = Path(
        entry.get("workflow_json") or case_output_dir / "workflow_benchmark_report.json"
    )
    if workflow_path.exists():
        workflow_report = _read_json_object(workflow_path)
        if workflow_report:
            case_rows = collect_summary_rows(workflow_report)
    elif not checkpoint:
        summary_path = Path(
            entry.get("summary_csv") or case_output_dir / "focus_benchmark_summary.csv"
        )
        case_rows = _read_csv(summary_path)

    if checkpoint:
        metadata = _case_summary_metadata(case, checkpoint, case_rows)
        checkpoint_rows = collect_checkpoint_summary_rows(
            checkpoint,
            metadata=metadata,
            routes=case.routes,
            case_status=case_status,
            failure_peak_memory_bytes=failure_peak,
        )
        merged = {
            (str(row.get("route", "")), str(row.get("proof_stage", ""))): row
            for row in case_rows
        }
        for checkpoint_row in checkpoint_rows:
            key = (
                str(checkpoint_row.get("route", "")),
                str(checkpoint_row.get("proof_stage", "")),
            )
            existing = merged.get(key)
            if existing is None or (
                existing.get("stage_status") != "completed"
                and checkpoint_row.get("stage_status") == "completed"
            ):
                if existing is not None:
                    for field, value in existing.items():
                        if (
                            field not in {"case_status", "stage_status"}
                            and checkpoint_row.get(field, "") == ""
                            and value != ""
                        ):
                            checkpoint_row[field] = value
                merged[key] = checkpoint_row
        case_rows = list(merged.values())

    metadata = _case_summary_metadata(case, checkpoint, case_rows)
    case_rows = complete_case_summary_rows(
        case_rows,
        metadata=metadata,
        routes=case.routes,
        case_status=case_status,
        failure_peak_memory_bytes=failure_peak,
    )

    case_output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = case_output_dir / "focus_benchmark_summary.csv"
    write_summary_csv(summary_csv_path, case_rows)
    write_terrae_summary_csv(
        case_output_dir / "focus_benchmark_terrae_summary.csv",
        case_rows,
    )
    write_zkboost_summary_csv(
        case_output_dir / "focus_benchmark_zkboost_summary.csv",
        case_rows,
    )
    write_soundness_summary_csv(
        case_output_dir / "focus_benchmark_soundness_summary.csv",
        case_rows,
    )
    entry["summary_csv"] = str(summary_csv_path)
    entry["terrae_summary_csv"] = str(
        case_output_dir / "focus_benchmark_terrae_summary.csv"
    )
    entry["zkboost_summary_csv"] = str(
        case_output_dir / "focus_benchmark_zkboost_summary.csv"
    )
    entry["soundness_summary_csv"] = str(
        case_output_dir / "focus_benchmark_soundness_summary.csv"
    )
    return case_rows, case_output_dir


def _write_aggregate(
    output_dir: Path,
    cases: list[PlanCase],
    state: dict[str, Any],
) -> tuple[Path, Path, Path, Path, Path]:
    rows: list[dict[str, Any]] = []
    state_cases = state.get("cases", {})
    for case in sorted(cases, key=lambda item: (item.order, item.case_id)):
        entry = state_cases.get(case.case_id, {"status": "not_run"})
        case_rows, _ = _collect_case_rows(output_dir, case, entry)
        for row in case_rows:
            row["phase"] = case.phase
            row["case_order"] = str(case.order)
            rows.append(row)

    csv_path = output_dir / "summary.csv"
    fieldnames = ["case_order", "phase", *SUMMARY_FIELDNAMES]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    compatibility_csv_path = output_dir / "benchmark_plan_focus_summary.csv"
    if compatibility_csv_path != csv_path:
        compatibility_csv_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    terrae_csv_path = output_dir / "terrae_summary.csv"
    terrae_fieldnames = ["case_order", "phase", *TERRAE_SUMMARY_FIELDNAMES]
    with terrae_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=terrae_fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("route") != "Terrae":
                continue
            writer.writerow({field: row.get(field, "") for field in terrae_fieldnames})

    zkboost_csv_path = output_dir / "zkboost_summary.csv"
    zkboost_fieldnames = ["case_order", "phase", *ZKBOOST_SUMMARY_FIELDNAMES]
    with zkboost_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=zkboost_fieldnames)
        writer.writeheader()
        for row in rows:
            if not str(row.get("route", "")).startswith("CertXGB"):
                continue
            writer.writerow({field: row.get(field, "") for field in zkboost_fieldnames})

    soundness_csv_path = output_dir / "soundness_summary.csv"
    soundness_fieldnames = ["case_order", "phase", *SOUNDNESS_SUMMARY_FIELDNAMES]
    with soundness_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=soundness_fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("route") != "Terrae":
                continue
            writer.writerow({field: row.get(field, "") for field in soundness_fieldnames})

    completed_count = sum(
        1
        for case in cases
        if state.get("cases", {}).get(case.case_id, {}).get("status") == "completed"
    )
    timed_out_count = sum(
        1
        for case in cases
        if state.get("cases", {}).get(case.case_id, {}).get("status") == "timed_out"
    )
    unsupported_count = sum(
        1
        for case in cases
        if state.get("cases", {}).get(case.case_id, {}).get("status") == "unsupported"
    )
    failed_count = sum(
        1
        for case in cases
        if state.get("cases", {}).get(case.case_id, {}).get("status") == "failed"
    )
    pending_count = len(cases) - completed_count - timed_out_count - unsupported_count - failed_count

    md_path = output_dir / "summary.md"
    lines = [
        "# Benchmark Phase Summary",
        "",
        "## Run Summary",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| output_dir | `{output_dir}` |",
        f"| cases_json | `{state.get('cases_json', 'built-in default plan')}` |",
        f"| listed_cases | {len(cases)} |",
        f"| completed_cases | {completed_count} |",
        f"| timed_out_cases | {timed_out_count} |",
        f"| unsupported_cases | {unsupported_count} |",
        f"| failed_cases | {failed_count} |",
        f"| pending_cases | {pending_count} |",
        f"| state_file | `{state.get('state_file', output_dir / 'benchmark_plan_state.json')}` |",
        "",
        "## Cases",
        "",
        "| order | phase | case_id | status | output | message |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        entry = state.get("cases", {}).get(case.case_id, {})
        status = entry.get("status", "pending")
        output = entry.get("output_dir", "")
        message = str(entry.get("message", "")).replace("\n", " ")
        lines.append(f"| {case.order} | {case.phase} | `{case.case_id}` | {status} | `{output}` | {message} |")

    lines.extend([
        "",
        "## Focus Table",
        "",
        "| order | phase | case_id | route | proof_stage | N_used | b_padded | T | d | bins | accuracy | accuracy_split | accuracy_samples | learning_rate | proof_size_bytes | prover_time_s | verifier_time_s | peak_memory_bytes |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row.get('case_order', '')} | {row.get('phase', '')} | `{row.get('case_id', '')}` | "
            f"{row.get('route', '')} | {row.get('proof_stage', '')} | {row.get('N_used', '')} | "
            f"{row.get('b_padded', '')} | {row.get('T', '')} | {row.get('d', '')} | {row.get('bin_count', '')} | "
            f"`{row.get('accuracy', '')}` | `{row.get('accuracy_split', '')}` | "
            f"{row.get('accuracy_samples', '')} | `{row.get('learning_rate', '')}` | "
            f"{row.get('proof_size_bytes', '')} | {row.get('prover_time_s', '')} | "
            f"{row.get('verifier_time_s', '')} | {row.get('peak_memory_bytes', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    compatibility_md_path = output_dir / "benchmark_plan_focus_summary.md"
    if compatibility_md_path != md_path:
        compatibility_md_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return md_path, csv_path, terrae_csv_path, zkboost_csv_path, soundness_csv_path


def _case_output_dir(
    output_dir: Path,
    case: PlanCase,
    existing_entry: dict[str, Any] | None = None,
) -> Path:
    existing_output = (existing_entry or {}).get("output_dir")
    if existing_output:
        candidate = Path(str(existing_output))
        resolved_candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else (REPO_ROOT / candidate).resolve()
        )
        resolved_cases_root = (
            (output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir)
            / "cases"
        ).resolve()
        if resolved_candidate.is_relative_to(resolved_cases_root):
            return candidate
    return output_dir / "cases" / f"{case.order:02d}_{case.case_id}"


def _record_case_start(state: dict[str, Any], case: PlanCase, case_output_dir: Path) -> None:
    state["cases"][case.case_id] = {
        "status": "running",
        "case": asdict(case),
        "output_dir": str(case_output_dir),
        "case_log": str(case_output_dir / "case_runner.log"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def _case_skip_message(case: PlanCase, selected_cases: list[PlanCase], entry: dict[str, Any]) -> None:
    _print_case_banner(case, selected_cases)
    existing_status = entry.get("status", "completed")
    recorded_at = entry.get(
        (
            "timed_out_at"
            if existing_status == "timed_out"
            else ("unsupported_at" if existing_status == "unsupported" else "completed_at")
        ),
        "unknown",
    )
    existing_output = entry.get("output_dir", "")
    print(
        f"[benchmark-plan]   skip {existing_status} "
        f"(recorded_at={recorded_at}, output={existing_output})",
        flush=True,
    )


def _case_completion_update(case_output_dir: Path, case_result: dict[str, str]) -> dict[str, str]:
    case_status = case_result.get("status", "completed")
    update = {
        "status": case_status,
        "summary_md": str(case_output_dir / "focus_benchmark_summary.md"),
        "summary_csv": str(case_output_dir / "focus_benchmark_summary.csv"),
        "terrae_summary_csv": str(case_output_dir / "focus_benchmark_terrae_summary.csv"),
        "zkboost_summary_csv": str(case_output_dir / "focus_benchmark_zkboost_summary.csv"),
        "soundness_summary_csv": str(case_output_dir / "focus_benchmark_soundness_summary.csv"),
        "workflow_json": str(case_output_dir / "workflow_benchmark_report.json"),
    }
    if case_result.get("message"):
        update["message"] = case_result["message"]
    if case_result.get("worker"):
        update["worker"] = case_result["worker"]
    if case_result.get("case_log"):
        update["case_log"] = case_result["case_log"]
    if case_status == "timed_out":
        update["timed_out_at"] = datetime.now().isoformat(timespec="seconds")
    elif case_status == "unsupported":
        update["unsupported_at"] = datetime.now().isoformat(timespec="seconds")
    else:
        update["completed_at"] = datetime.now().isoformat(timespec="seconds")
    return update


def _case_failure_update(error: Exception) -> dict[str, str]:
    return {
        "status": "failed",
        "failed_at": datetime.now().isoformat(timespec="seconds"),
        "message": str(error),
    }


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_file or (args.output_dir / "benchmark_plan_state.json")
    all_cases = _load_cases(args.cases_json)
    if args.gradient_mode is not None:
        all_cases = [replace(case, gradient_mode=args.gradient_mode) for case in all_cases]
    selected_cases = _case_filter(all_cases, args)
    state = _read_state(state_file)
    state.setdefault("cases", {})
    previous_training_semantics = state.get("training_semantics_fingerprint")
    if previous_training_semantics != TRAINING_SEMANTICS_FINGERPRINT:
        for entry in state["cases"].values():
            if _is_terminal_status(str(entry.get("status", ""))):
                entry["status"] = "pending"
                entry["message"] = (
                    "invalidated by training-semantics change: "
                    f"{previous_training_semantics or 'legacy'} -> {TRAINING_SEMANTICS_FINGERPRINT}"
                )
    state["training_semantics_fingerprint"] = TRAINING_SEMANTICS_FINGERPRINT
    state.setdefault("events", [])
    _normalize_timeout_entries(state)
    state["plan"] = [asdict(case) for case in selected_cases]
    state["phases"] = _phase_order(selected_cases)
    state["output_dir"] = str(args.output_dir)
    state["state_file"] = str(state_file)
    state["cases_json"] = str(args.cases_json) if args.cases_json is not None else "built-in default plan"
    state["jobs"] = args.jobs
    state["timeout_seconds"] = args.timeout_seconds
    state["stop_subsequent_t_after_failure"] = args.stop_subsequent_t_after_failure
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_state(state_file, state)

    print(
        f"[benchmark-plan] cases source: {state['cases_json']}",
        flush=True,
    )
    print(
        f"[benchmark-plan] existing terminal cases in state: "
        f"{_terminal_case_count(selected_cases, state)}/{len(selected_cases)}",
        flush=True,
    )
    print(
        f"[benchmark-plan] jobs: {args.jobs}, timeout_seconds: "
        f"{'none' if args.timeout_seconds is None else args.timeout_seconds}",
        flush=True,
    )

    if not args.skip_build and selected_cases:
        include_zkboost = any(
            case.routes in {"zkboost", "terrae+zkboost"} for case in selected_cases
        )
        print(
            "[benchmark-plan] preparing shared build before worker launch",
            flush=True,
        )
        _prepare_build(
            args.build_dir,
            include_zkboost=include_zkboost,
            timeout_seconds=args.timeout_seconds,
        )

    status = "completed"
    message = "All selected cases completed."
    saw_timed_out = False
    try:
        ordered_phases = _phase_order(selected_cases)
        for phase_index, phase in enumerate(ordered_phases, start=1):
            _print_phase_banner(phase, selected_cases, phase_index, len(ordered_phases))
            state["events"].append(
                {
                    "event": "phase_start",
                    "phase": phase,
                    "at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        _write_state(state_file, state)

        pending_cases: list[PlanCase] = []
        for case in selected_cases:
            existing = state["cases"].get(case.case_id, {})
            if (
                _is_terminal_status(existing.get("status", ""))
                and not args.rerun_completed
                and not args.rerun_stage
            ):
                _case_skip_message(case, selected_cases, existing)
                continue
            pending_cases.append(case)

        next_case = 0
        blocked_stages: dict[tuple[str, str, str], tuple[int, str]] = {}
        running: dict[Future[dict[str, Any]], tuple[PlanCase, Path]] = {}
        with ThreadPoolExecutor(max_workers=args.jobs, thread_name_prefix="benchmark-case") as executor:
            while next_case < len(pending_cases) or running:
                while next_case < len(pending_cases) and len(running) < args.jobs and not (
                    status == "failed" and args.fail_fast
                ):
                    case = pending_cases[next_case]
                    next_case += 1
                    skipped_stages = (
                        _blocked_stage_selectors(blocked_stages, case)
                        if args.stop_subsequent_t_after_failure
                        else []
                    )
                    existing_entry = state["cases"].get(case.case_id, {})
                    case_output_dir = _case_output_dir(
                        args.output_dir,
                        case,
                        existing_entry,
                    )
                    _record_case_start(state, case, case_output_dir)
                    _write_state(state_file, state)
                    _print_case_banner(case, selected_cases)
                    print(
                        f"[benchmark-plan]   launched log={case_output_dir / 'case_runner.log'}",
                        flush=True,
                    )
                    if skipped_stages:
                        print(
                            f"[benchmark-plan]   skipping later-T stage(s): "
                            f"{', '.join(skipped_stages)}",
                            flush=True,
                        )
                    future = executor.submit(
                        _run_case,
                        case,
                        args,
                        case_output_dir,
                        tuple(skipped_stages),
                    )
                    running[future] = (case, case_output_dir)

                if not running:
                    break

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    case, case_output_dir = running.pop(future)
                    try:
                        case_result = future.result()
                        update = _case_completion_update(case_output_dir, case_result)
                        if update["status"] == "timed_out":
                            saw_timed_out = True
                        if args.stop_subsequent_t_after_failure:
                            for stage in case_result.get("failed_stages", []):
                                if stage in _STOPPABLE_STAGE_SELECTORS:
                                    blocked_stages[(case.dataset, case.routes, stage)] = (
                                        case.T,
                                        case.case_id,
                                    )
                        state["cases"][case.case_id].update(update)
                        _write_state(state_file, state)
                        print(
                            f"[benchmark-plan]   {case.case_id} {update['status']} "
                            f"(worker={update.get('worker', 'unknown')}, log={update.get('case_log', '')})",
                            flush=True,
                        )
                    except Exception as error:
                        state["cases"][case.case_id].update(_case_failure_update(error))
                        _write_state(state_file, state)
                        status = "failed"
                        message = f"Case {case.order} `{case.case_id}` failed: {error}"
                        print(f"[benchmark-plan] {message}", file=sys.stderr, flush=True)
                        if args.fail_fast:
                            print(
                                "[benchmark-plan] fail-fast is set; no new cases will be launched, "
                                "waiting for already-running cases to finish.",
                                flush=True,
                            )

        if any(entry.get("status") == "failed" for entry in state["cases"].values()):
            status = "failed"
            message = "Selected cases finished with at least one failure."
        elif (
            saw_timed_out or
            any(entry.get("status") == "timed_out" for entry in state["cases"].values()) or
            any(entry.get("status") == "unsupported" for entry in state["cases"].values())
        ):
            status = "completed"
            message = "Selected cases completed; some cases timed out or hit recorded limits."
    finally:
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        summary_md, summary_csv, terrae_summary_csv, zkboost_summary_csv, soundness_summary_csv = _write_aggregate(
            args.output_dir,
            all_cases,
            state,
        )
        _write_state(state_file, state)

    print(f"[benchmark-plan] status: {status}", flush=True)
    print(f"[benchmark-plan] markdown: {summary_md}", flush=True)
    print(f"[benchmark-plan] common csv: {summary_csv}", flush=True)
    print(f"[benchmark-plan] Terrae csv: {terrae_summary_csv}", flush=True)
    print(f"[benchmark-plan] CertXGB csv: {zkboost_summary_csv}", flush=True)
    print(f"[benchmark-plan] soundness csv: {soundness_summary_csv}", flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
