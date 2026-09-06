#!/usr/bin/env python3

import argparse
import fcntl
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_common import (
    MeasuredStepError,
    REPO_ROOT,
    TRAINING_SEMANTICS_FINGERPRINT,
    _parse_key_value_lines,
    _run_logged,
    _write_gbdt_proof_config,
)
from benchmark_standard import (
    _cost_statistics,
    _deduplicate_combined_profile_pcs_statement,
    _environment,
    _measured_step,
    _prepare_build,
    _safe_name,
    _write_markdown as _write_standard_markdown,
)
from train_fixed_point_dataset import _load_dataset
from validate import validate_inference_trace
from writer import INFERENCE_FORMAT_TAG, write_inference_trace
from schema import SemanticInferenceQueryBatch


@dataclass(frozen=True)
class WorkflowCase:
    case_id: str
    phase: str
    dataset: str
    training_samples: int | None
    gbdt_iterations: int
    max_depth: int
    bin_count: int
    inference_samples: int | None
    gradient_mode: str = "first-order-rss"
    learning_rate_numerator: int = 4
    learning_rate_denominator: int = 1
    split_gain_threshold: int = 1


def _is_timeout_error_message(message: str) -> bool:
    return "timed out after " in message


def _is_unsupported_error_message(message: str) -> bool:
    return "too large for size_t-based range tables" in message


def _status_from_error_message(message: str) -> str:
    if _is_timeout_error_message(message):
        return "timed_out"
    if _is_unsupported_error_message(message):
        return "unsupported"
    return "failed"


def _is_memory_guard_error(error: Exception) -> bool:
    return (
        isinstance(error, MeasuredStepError)
        and isinstance(error.measurement, dict)
        and error.measurement.get("status")
        in {"swap_detected", "oom_detected", "memory_pressure_detected", "rss_limit_detected"}
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _merge_stage_payloads(*payloads: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for route in ("terrae", "zkboost"):
            route_payload = payload.get(route)
            if not isinstance(route_payload, dict):
                continue
            merged.setdefault(route, {}).update(route_payload)
        stage_failures = payload.get("stage_failures")
        if isinstance(stage_failures, list):
            merged["stage_failures"] = [
                dict(item) for item in stage_failures if isinstance(item, dict)
            ]
        srs_requirement = payload.get("srs_requirement")
        if isinstance(srs_requirement, dict):
            merged["srs_requirement"] = dict(srs_requirement)
    return merged


def _is_terminal_status(status: str) -> bool:
    return status in {"completed", "timed_out", "unsupported"}


def _includes_terrae(routes: str) -> bool:
    return routes in {"terrae", "terrae+zkboost"}


def _includes_zkboost(routes: str) -> bool:
    return routes in {"zkboost", "terrae+zkboost"}


def _required_srs_query_routes(routes: str, rerun_stages: set[str]) -> str:
    if not rerun_stages or routes != "terrae+zkboost":
        return routes
    needs_terrae = any(stage.startswith("terrae-") for stage in rerun_stages)
    needs_zkboost = any(stage.startswith("zkboost-") for stage in rerun_stages)
    if needs_terrae and not needs_zkboost:
        return "terrae"
    if needs_zkboost and not needs_terrae:
        return "zkboost"
    return routes


def _merge_cached_srs_requirement(
    current: dict[str, Any],
    cached: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(cached, dict):
        return current
    cached_routes = cached.get("routes")
    if not isinstance(cached_routes, dict):
        return current
    routes = current.setdefault("routes", {})
    for route, metrics in cached_routes.items():
        if route not in routes and isinstance(metrics, dict):
            routes[route] = dict(metrics)
    current["deg1_plus_1"] = max(
        [1] + [
            int(metrics.get("required_srs_deg1_plus_1", 1))
            for metrics in routes.values()
            if isinstance(metrics, dict)
        ]
    )
    current["deg2_plus_1"] = max(
        [1] + [
            int(metrics.get("required_srs_deg2_plus_1", 1))
            for metrics in routes.values()
            if isinstance(metrics, dict)
        ]
    )
    return current


def _extract_relevant_error_message(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_timeout_error_message(line) or _is_unsupported_error_message(line):
            return line
    return text.strip()


def _summarize_exception(error: Exception) -> str:
    parts = [str(error)]
    if isinstance(error, subprocess.CalledProcessError):
        if error.output:
            parts.append(str(error.output))
        if error.stderr:
            parts.append(str(error.stderr))
    combined = "\n".join(part for part in parts if part)
    return _extract_relevant_error_message(combined) or str(error)


def _next_power_of_two(value: int) -> int:
    out = 1
    while out < value:
        out <<= 1
    return out


def _parse_metric_lines(stdout: str) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 3 or parts[0] != "metric":
            continue
        metrics[parts[1]] = int(parts[2])
    return metrics


def _ensure_srs(
    *,
    build_dir: Path,
    srs_cache_dir: Path,
    deg1_plus_1: int,
    deg2_plus_1: int,
    timeout_seconds: float | None,
) -> Path:
    srs_cache_dir.mkdir(parents=True, exist_ok=True)
    srs_path = srs_cache_dir / f"kzg_deg1_{deg1_plus_1}_deg2_{deg2_plus_1}.srs"
    lock_path = srs_cache_dir / f"{srs_path.name}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if srs_path.exists():
            return srs_path
        tmp_path = srs_cache_dir / f".{srs_path.name}.{os.getpid()}.tmp"
        try:
            _run_logged(
                f"generate-srs deg1+1={deg1_plus_1} deg2+1={deg2_plus_1}",
                [
                    str(build_dir / "examples" / "kzg_setup_helper"),
                    str(tmp_path),
                    str(deg1_plus_1),
                    str(deg2_plus_1),
                ],
                cwd=REPO_ROOT,
                timeout_seconds=timeout_seconds,
            )
            tmp_path.replace(srs_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return srs_path


def _query_case_srs_requirement(
    *,
    case: WorkflowCase,
    build_dir: Path,
    routes: str,
    training_trace: Path,
    inference_trace: Path,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    requirements: dict[str, dict[str, int]] = {}
    deg1_plus_1 = 1
    deg2_plus_1 = 1

    if _includes_terrae(routes):
        _, terrae_stdout = _run_logged(
            f"{case.case_id}: terrae-required-srs",
            [
                str(build_dir / "examples" / "example_training_then_inference_bundle"),
                "required-srs-from-split-traces",
                str(training_trace),
                str(inference_trace),
            ],
            cwd=REPO_ROOT,
            timeout_seconds=timeout_seconds,
        )
        terrae_metrics = _parse_metric_lines(terrae_stdout)
        requirements["terrae"] = terrae_metrics
        deg1_plus_1 = max(deg1_plus_1, terrae_metrics["required_srs_deg1_plus_1"])
        deg2_plus_1 = max(deg2_plus_1, terrae_metrics["required_srs_deg2_plus_1"])

    if _includes_zkboost(routes):
        _, zkboost_stdout = _run_logged(
            f"{case.case_id}: zkboost-required-srs",
            [
                str(build_dir / "examples" / "example_zkboost_shared_trace"),
                "required-srs-preprocessed",
                str(training_trace),
                str(inference_trace),
            ],
            cwd=REPO_ROOT,
            timeout_seconds=timeout_seconds,
        )
        zkboost_metrics = _parse_metric_lines(zkboost_stdout)
        requirements["zkboost"] = zkboost_metrics
        deg1_plus_1 = max(deg1_plus_1, zkboost_metrics["required_srs_deg1_plus_1"])
        deg2_plus_1 = max(deg2_plus_1, zkboost_metrics["required_srs_deg2_plus_1"])

    return {
        "deg1_plus_1": deg1_plus_1,
        "deg2_plus_1": deg2_plus_1,
        "routes": requirements,
    }


def _select_srs_size(
    *,
    route_requirement: dict[str, int],
    initial_deg1_plus_1: int,
    initial_deg2_plus_1: int,
    max_deg1_plus_1: int | None,
    max_deg2_plus_1: int | None,
    case_id: str,
    route_name: str,
) -> tuple[int, int]:
    deg1_plus_1 = max(
        initial_deg1_plus_1,
        _next_power_of_two(route_requirement["required_srs_deg1_plus_1"]),
    )
    deg2_plus_1 = max(
        initial_deg2_plus_1,
        _next_power_of_two(route_requirement["required_srs_deg2_plus_1"]),
    )
    if (
        (max_deg1_plus_1 is not None and deg1_plus_1 > max_deg1_plus_1) or
        (max_deg2_plus_1 is not None and deg2_plus_1 > max_deg2_plus_1)
    ):
        raise RuntimeError(
            f"{case_id}: {route_name} required SRS exceeds the configured cap "
            f"(deg1+1={deg1_plus_1}, deg2+1={deg2_plus_1}; "
            f"max deg1+1={max_deg1_plus_1}, deg2+1={max_deg2_plus_1})"
        )
    return deg1_plus_1, deg2_plus_1


def _require_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _load_cases_from_json(path: Path) -> list[WorkflowCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("cases json root must be a list")

    cases: list[WorkflowCase] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"cases[{index}] must be an object")
        dataset = item.get("dataset")
        if not isinstance(dataset, str):
            raise ValueError(f"cases[{index}].dataset must be a string")
        gbdt_iterations = _require_int(item.get("T"), f"cases[{index}].T")
        max_depth = _require_int(item.get("d"), f"cases[{index}].d")
        if max_depth <= 0:
            raise ValueError(f"cases[{index}].d must be at least one because the root has depth one")
        training_samples = item.get("N")
        if training_samples is not None:
            training_samples = _require_int(training_samples, f"cases[{index}].N")
        inference_samples = item.get("b")
        if inference_samples is not None:
            inference_samples = _require_int(inference_samples, f"cases[{index}].b")
        bin_count = item.get("bin_count", item.get("bins", 8))
        bin_count = _require_int(bin_count, f"cases[{index}].bin_count")
        gradient_mode = item.get("loss_mode", item.get("gradient_mode", "first-order-rss"))
        if gradient_mode not in {"first-order-rss", "second-order-xgboost"}:
            raise ValueError(
                f"cases[{index}].loss_mode must be first-order-rss or second-order-xgboost"
            )
        learning_rate_numerator = _require_int(
            item.get("learning_rate_numerator", 4),
            f"cases[{index}].learning_rate_numerator",
        )
        learning_rate_denominator = _require_int(
            item.get("learning_rate_denominator", 1),
            f"cases[{index}].learning_rate_denominator",
        )
        split_gain_threshold = _require_int(
            item.get("split_gain_threshold", 1),
            f"cases[{index}].split_gain_threshold",
        )
        if learning_rate_numerator <= 0 or learning_rate_denominator <= 0:
            raise ValueError(f"cases[{index}] learning-rate numerator and denominator must be positive")
        if split_gain_threshold < 0:
            raise ValueError(f"cases[{index}].split_gain_threshold must be nonnegative")
        case_id = item.get("case_id")
        phase = item.get("phase", "")
        if not isinstance(phase, str):
            raise ValueError(f"cases[{index}].phase must be a string when provided")
        if case_id is None:
            train_tag = training_samples if training_samples is not None else "full"
            infer_tag = inference_samples if inference_samples is not None else train_tag
            case_id = (
                f"{_safe_name(dataset)}_N{train_tag}_T{gbdt_iterations}"
                f"_d{max_depth}_b{infer_tag}_bins{bin_count}"
            )
        if not isinstance(case_id, str):
            raise ValueError(f"cases[{index}].case_id must be a string when provided")
        cases.append(
            WorkflowCase(
                case_id=case_id,
                phase=phase,
                dataset=dataset,
                training_samples=training_samples,
                gbdt_iterations=gbdt_iterations,
                max_depth=max_depth,
                bin_count=bin_count,
                inference_samples=inference_samples,
                gradient_mode=gradient_mode,
                learning_rate_numerator=learning_rate_numerator,
                learning_rate_denominator=learning_rate_denominator,
                split_gain_threshold=split_gain_threshold,
            )
        )
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run parameterized Terrae / CertXGB workflow benchmarks from explicit "
            "case inputs, with dataset adaptation and benchmark report generation."
        )
    )
    parser.add_argument("--cases-json", type=Path, help="JSON array of cases with fields dataset, N, T, d, b.")
    parser.add_argument("--case-id")
    parser.add_argument("--dataset", default="iris-binary")
    parser.add_argument("--training-samples", "--N", dest="training_samples", type=int)
    parser.add_argument("--gbdt-iterations", "--T", dest="gbdt_iterations", type=int, default=1)
    parser.add_argument("--max-depth", "--d", dest="max_depth", type=int, default=2)
    parser.add_argument("--inference-samples", "--b", dest="inference_samples", type=int)
    parser.add_argument("--bin-count", type=int, default=8)
    parser.add_argument(
        "--loss-mode",
        "--gradient-mode",
        dest="gradient_mode",
        choices=["first-order-rss", "second-order-xgboost"],
        default="first-order-rss",
        help=(
            "Training loss embedded in the public semantic trace and enforced by both Terrae and CertXGB. "
            "Default: first-order-rss."
        ),
    )
    parser.add_argument("--learning-rate-numerator", type=int, default=4)
    parser.add_argument("--learning-rate-denominator", type=int, default=1)
    parser.add_argument("--split-gain-threshold", type=int, default=1)
    parser.add_argument(
        "--routes",
        choices=["terrae", "zkboost", "terrae+zkboost"],
        default="terrae+zkboost",
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build-release")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "out" / "benchmarks" / "parameterized_workflow",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--srs-cache-dir",
        type=Path,
        help="Directory for reusable KZG SRS files. Default: OUTPUT_DIR/srs.",
    )
    parser.add_argument("--srs-initial-deg1-plus-1", type=int, default=1 << 17)
    parser.add_argument("--srs-initial-deg2-plus-1", type=int, default=1 << 17)
    parser.add_argument(
        "--assume-srs-deg1-plus-1",
        type=int,
        help=(
            "Skip Terrae/CertXGB required-SRS queries and assume this G1 SRS size. "
            "Use only when a matching cached SRS is known to cover the case."
        ),
    )
    parser.add_argument(
        "--assume-srs-deg2-plus-1",
        type=int,
        help=(
            "Skip Terrae/CertXGB required-SRS queries and assume this G2 SRS size. "
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
        help="Per-stage timeout in seconds. Default: 7200.",
    )
    timeout_group.add_argument(
        "--no-timeout",
        action="store_const",
        const=None,
        dest="timeout_seconds",
        help="Disable build and per-stage wall-clock time limits.",
    )
    parser.set_defaults(timeout_seconds=7200.0)
    parser.add_argument("--fail-fast", action="store_true")
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
        help="Reuse successful stage payloads from an existing workflow report in OUTPUT_DIR.",
    )
    parser.add_argument(
        "--rerun-stage",
        action="append",
        choices=["terrae-training", "zkboost-full-training", "zkboost-inference"],
        default=[],
        help=(
            "Ignore and replace the selected successful stage checkpoint while reusing "
            "all other stages. Repeatable."
        ),
    )
    parser.add_argument(
        "--skip-stage",
        action="append",
        choices=["terrae-training", "zkboost-full-training", "zkboost-inference"],
        default=[],
        help="Do not execute the selected proof stage; used for stage-specific continuation.",
    )
    args = parser.parse_args()
    if args.max_depth <= 0:
        raise ValueError("--max-depth/--d must be at least one because the root has depth one")
    if args.learning_rate_numerator <= 0 or args.learning_rate_denominator <= 0:
        raise ValueError("learning-rate numerator and denominator must be positive")
    if args.split_gain_threshold < 0:
        raise ValueError("--split-gain-threshold must be nonnegative")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.rerun_stage and not args.reuse_existing_stages:
        raise ValueError("--rerun-stage requires --reuse-existing-stages")
    if (args.assume_srs_deg1_plus_1 is None) != (args.assume_srs_deg2_plus_1 is None):
        raise ValueError("--assume-srs-deg1-plus-1 and --assume-srs-deg2-plus-1 must be provided together")
    if args.assume_srs_deg1_plus_1 is not None and (
        args.assume_srs_deg1_plus_1 <= 0 or args.assume_srs_deg2_plus_1 <= 0
    ):
        raise ValueError("assumed SRS sizes must be positive")

    if args.cases_json is None:
        train_tag = args.training_samples if args.training_samples is not None else "full"
        infer_tag = args.inference_samples if args.inference_samples is not None else train_tag
        args.cases = [
            WorkflowCase(
                case_id=(
                    args.case_id
                    or f"{_safe_name(args.dataset)}_N{train_tag}_T{args.gbdt_iterations}"
                    f"_d{args.max_depth}_b{infer_tag}_bins{args.bin_count}"
                ),
                phase="custom",
                dataset=args.dataset,
                training_samples=args.training_samples,
                gbdt_iterations=args.gbdt_iterations,
                max_depth=args.max_depth,
                bin_count=args.bin_count,
                inference_samples=args.inference_samples,
                gradient_mode=args.gradient_mode,
                learning_rate_numerator=args.learning_rate_numerator,
                learning_rate_denominator=args.learning_rate_denominator,
                split_gain_threshold=args.split_gain_threshold,
            )
        ]
    else:
        args.cases = _load_cases_from_json(args.cases_json)
    return args


def _preflight_case(case: WorkflowCase) -> dict[str, Any]:
    dataset = _load_dataset(case.dataset, case.bin_count)
    available_original_samples = dataset.original_query_count
    requested_training_samples = (
        available_original_samples if case.training_samples is None else case.training_samples
    )
    if requested_training_samples <= 0:
        raise ValueError(f"{case.case_id}: training sample count must be positive")
    actual_training_original_samples = min(requested_training_samples, available_original_samples)
    padded_training_samples = _next_power_of_two(actual_training_original_samples)

    requested_inference_samples = (
        actual_training_original_samples
        if case.inference_samples is None
        else case.inference_samples
    )
    if requested_inference_samples <= 0:
        raise ValueError(f"{case.case_id}: inference sample count must be positive")
    padded_inference_samples = _next_power_of_two(requested_inference_samples)
    if padded_inference_samples > padded_training_samples:
        raise ValueError(
            f"{case.case_id}: padded inference sample count {padded_inference_samples} exceeds "
            f"padded training sample count {padded_training_samples}"
        )

    return {
        "dataset_name": dataset.name,
        "available_original_samples": available_original_samples,
        "feature_count": dataset.feature_count,
        "real_class_count": dataset.class_count,
        "requested_training_samples": requested_training_samples,
        "actual_training_original_samples": actual_training_original_samples,
        "padded_training_samples": padded_training_samples,
        "requested_inference_samples": requested_inference_samples,
        "padded_inference_samples": padded_inference_samples,
        "inference_uses_padded_training_samples": (
            requested_inference_samples > actual_training_original_samples
        ),
    }


def _read_trace_fields(path: Path) -> tuple[str, dict[str, int], dict[str, list[int]]]:
    format_tag = ""
    scalars: dict[str, int] = {}
    vectors: dict[str, list[int]] = {}
    with path.open("r", encoding="ascii") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            kind = parts[0]
            if kind == "format":
                format_tag = parts[1]
                continue
            if kind == "scalar":
                scalars[parts[1]] = int(parts[2])
                continue
            if kind == "i64vec":
                name = parts[1]
                count = int(parts[2])
                values = [int(value) for value in parts[3:]]
                if len(values) != count:
                    raise ValueError(f"{path}: inconsistent i64vec length for {name}")
                vectors[name] = values
    return format_tag, scalars, vectors


def _crop_terrae_inference_trace(
    source_path: Path,
    target_path: Path,
    *,
    tree_count: int,
    requested_inference_samples: int,
) -> dict[str, int]:
    format_tag, scalars, vectors = _read_trace_fields(source_path)
    if format_tag != INFERENCE_FORMAT_TAG:
        raise ValueError(f"{source_path}: expected {INFERENCE_FORMAT_TAG}, got {format_tag or 'missing format'}")

    source_query_count = scalars["inference.query_count"]
    feature_count = scalars["inference.feature_count"]
    if source_query_count % tree_count != 0:
        raise ValueError(f"{source_path}: query_count {source_query_count} not divisible by tree_count {tree_count}")

    source_sample_count = source_query_count // tree_count
    padded_inference_samples = _next_power_of_two(requested_inference_samples)
    if padded_inference_samples > source_sample_count:
        raise ValueError(
            f"{source_path}: requested padded inference sample count {padded_inference_samples} exceeds "
            f"available sample count {source_sample_count}"
        )

    if padded_inference_samples == source_sample_count:
        return {
            "requested_inference_samples": requested_inference_samples,
            "padded_inference_samples": padded_inference_samples,
            "source_sample_count": source_sample_count,
        }

    source_query_tree_ids = vectors["inference.query_tree_ids"]
    source_lifted_inputs = vectors["inference.lifted_inputs"]
    cropped_query_tree_ids: list[int] = []
    cropped_lifted_inputs: list[int] = []
    source_tree_row_width = source_sample_count
    source_tree_input_width = source_sample_count * feature_count
    target_tree_input_width = padded_inference_samples * feature_count

    for tree in range(tree_count):
        row_start = tree * source_tree_row_width
        cropped_query_tree_ids.extend(
            source_query_tree_ids[row_start : row_start + padded_inference_samples]
        )
        input_start = tree * source_tree_input_width
        cropped_lifted_inputs.extend(
            source_lifted_inputs[input_start : input_start + target_tree_input_width]
        )

    cropped = SemanticInferenceQueryBatch(
        query_count=tree_count * padded_inference_samples,
        feature_count=feature_count,
        query_tree_ids=cropped_query_tree_ids,
        lifted_inputs=cropped_lifted_inputs,
    )
    validate_inference_trace(cropped, tree_count=tree_count)
    write_inference_trace(str(target_path), cropped)
    return {
        "requested_inference_samples": requested_inference_samples,
        "padded_inference_samples": padded_inference_samples,
        "source_sample_count": source_sample_count,
    }


def _report_environment(build_dir: Path, results: list[dict[str, Any]]) -> dict[str, str]:
    deg1 = max((result.get("final_srs_deg1_plus_1", 0) for result in results), default=0)
    deg2 = max((result.get("final_srs_deg2_plus_1", 0) for result in results), default=0)
    return _environment(build_dir, deg1, deg2)


def _write_case_standard_report(
    case_dir: Path,
    *,
    build_dir: Path,
    routes: str,
    result: dict[str, Any],
) -> None:
    report = {
        "config": {
            "routes": routes,
            "cost_statistics_schema_version": 3,
        },
        "environment": _environment(
            build_dir,
            result["final_srs_deg1_plus_1"],
            result["final_srs_deg2_plus_1"],
        ),
        "cases": [result["benchmark_payload"]],
    }
    json_path = case_dir / "benchmark_report.json"
    md_path = case_dir / "benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_standard_markdown(md_path, report)


def _write_summary_markdown(path: Path, *, environment: dict[str, str], routes: str, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Parameterized Workflow Benchmark",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in ["source_revision", "build_dir", "machine", "cpu", "logical_cpus", "os", "srs_deg1_plus_1", "srs_deg2_plus_1"]:
        lines.append(f"| {key} | `{environment[key]}` |")
    lines.extend([
        f"| route scope | `{routes}` |",
        "",
        "## Cases",
        "",
        "| case_id | status | dataset | loss | N_requested | N_used | N_padded | b_requested | b_padded | T | d | bins | final_deg1+1 | final_deg2+1 | standard_report |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for result in results:
        adaptation = result["dataset_adaptation"]
        report_path = result.get("standard_markdown")
        report_cell = f"`{report_path}`" if report_path is not None else "unavailable"
        lines.append(
            f"| `{result['case']['case_id']}` | {result['status']} | `{result['case']['dataset']}` | "
            f"`{result['case']['gradient_mode']}` | {adaptation['requested_training_samples']} | {adaptation['actual_training_original_samples']} | "
            f"{adaptation['padded_training_samples']} | {adaptation['requested_inference_samples']} | "
            f"{adaptation['padded_inference_samples']} | {result['case']['gbdt_iterations']} | "
            f"{result['case']['max_depth']} | {result['case']['bin_count']} | "
            f"{result.get('final_srs_deg1_plus_1', 'unavailable')} | "
            f"{result.get('final_srs_deg2_plus_1', 'unavailable')} | {report_cell} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `N_used` is the effective original training sample count after clamping to the dataset size.",
        "- `N_padded` and `b_padded` are the power-of-two counts actually used by the proof traces.",
        "- SRS files are cached by `(deg1+1, deg2+1)` under the workflow output directory and reused across cases.",
        "- Within a case, Terrae and CertXGB route branches are run sequentially when both routes are requested.",
        "- Case directories with any completed proof stage contain a standard `benchmark_report.json` and `benchmark_report.md` for detailed proof and memory metrics.",
    ])
    non_completed = [result for result in results if result["status"] != "completed"]
    if non_completed:
        lines.extend([
            "",
            "## Incomplete Cases",
            "",
            "| case_id | status | message |",
            "| --- | --- | --- |",
        ])
        for result in non_completed:
            message = result.get("error_message", "").replace("\n", " ")
            lines.append(f"| `{result['case']['case_id']}` | {result['status']} | {message} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_case_once(
    case: WorkflowCase,
    *,
    case_dir: Path,
    build_dir: Path,
    srs_cache_dir: Path,
    routes: str,
    srs_deg1_plus_1: int,
    srs_deg2_plus_1: int,
    max_srs_deg1_plus_1: int | None,
    max_srs_deg2_plus_1: int | None,
    assume_srs_deg1_plus_1: int | None,
    assume_srs_deg2_plus_1: int | None,
    recorded_timed_out_stages: set[str],
    rerun_stages: set[str],
    skipped_stages: set[str],
    reused_benchmark_payload: dict[str, Any] | None,
    dataset_adaptation: dict[str, Any],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    prefix = (
        f"{_safe_name(case.dataset)}_T{case.gbdt_iterations}_d{case.max_depth}_bins{case.bin_count}"
        f"_{_safe_name(case.gradient_mode)}"
    )
    if case.training_samples is not None:
        prefix += f"_N{case.training_samples}"
    if case.inference_samples is not None:
        prefix += f"_b{case.inference_samples}"

    training_trace = case_dir / f"{prefix}_training.trace"
    inference_trace_full = case_dir / f"{prefix}_inference_full.trace"
    proof_config = case_dir / f"{prefix}_proof_config.json"

    _write_gbdt_proof_config(
        proof_config,
        dataset_name=case.dataset,
        iterations=case.gbdt_iterations,
        max_depth=case.max_depth,
        bin_count=case.bin_count,
        gradient_mode=case.gradient_mode,
        learning_rate_numerator=case.learning_rate_numerator,
        learning_rate_denominator=case.learning_rate_denominator,
        split_gain_threshold=case.split_gain_threshold,
    )

    train_elapsed, train_stdout = _run_logged(
        f"{case.case_id}: train-export-gbdt",
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "python_trace_export" / "train_fixed_point_dataset.py"),
            "--config",
            str(proof_config),
            "--dataset",
            case.dataset,
            "--iterations",
            str(case.gbdt_iterations),
            "--max-depth",
            str(case.max_depth),
            "--bin-count",
            str(case.bin_count),
            *(
                ["--leakage-free-evaluation-training-split"]
                if (
                    case.dataset in ("dry-bean", "optdigits-7class-train")
                    and case.training_samples is not None
                )
                else []
            ),
            *(["--max-original-queries", str(case.training_samples)] if case.training_samples is not None else []),
            "--training-trace",
            str(training_trace),
            "--inference-trace",
            str(inference_trace_full),
        ],
        cwd=REPO_ROOT,
        timeout_seconds=timeout_seconds,
    )
    train_fields = _parse_key_value_lines(train_stdout)
    training_tree_count = int(train_fields["tree_count"])
    requested_inference_samples = (
        case.inference_samples
        if case.inference_samples is not None
        else int(train_fields["original_samples"])
    )
    inference_trace = inference_trace_full
    inference_adaptation = _crop_terrae_inference_trace(
        inference_trace_full,
        case_dir / f"{prefix}_inference.trace",
        tree_count=training_tree_count,
        requested_inference_samples=requested_inference_samples,
    )
    if inference_adaptation["padded_inference_samples"] != inference_adaptation["source_sample_count"]:
        inference_trace = case_dir / f"{prefix}_inference.trace"
    reduced_inference_only = (
        inference_adaptation["padded_inference_samples"] != inference_adaptation["source_sample_count"]
    )

    previous_payload = reused_benchmark_payload or {}
    srs_query_routes = _required_srs_query_routes(routes, rerun_stages)

    if assume_srs_deg1_plus_1 is not None:
        terrae_srs_deg1_plus_1 = assume_srs_deg1_plus_1
        terrae_srs_deg2_plus_1 = assume_srs_deg2_plus_1
        zkboost_srs_deg1_plus_1 = assume_srs_deg1_plus_1
        zkboost_srs_deg2_plus_1 = assume_srs_deg2_plus_1
        assumed_route_requirement = {
            "required_srs_deg1_plus_1": assume_srs_deg1_plus_1,
            "required_srs_deg2_plus_1": assume_srs_deg2_plus_1,
        }
        srs_requirement = {
            "assumed": True,
            "deg1_plus_1": assume_srs_deg1_plus_1,
            "deg2_plus_1": assume_srs_deg2_plus_1,
            "routes": {
                "terrae": dict(assumed_route_requirement),
            },
        }
        if _includes_zkboost(routes):
            srs_requirement["routes"]["zkboost"] = dict(assumed_route_requirement)
        if not _includes_terrae(routes):
            del srs_requirement["routes"]["terrae"]
        print(
            f"[workflow] {case.case_id}: skipping required-SRS queries; "
            f"assuming SRS deg1+1={assume_srs_deg1_plus_1} "
            f"deg2+1={assume_srs_deg2_plus_1}",
            flush=True,
        )
    else:
        srs_requirement = _query_case_srs_requirement(
            case=case,
            build_dir=build_dir,
            routes=srs_query_routes,
            training_trace=training_trace,
            inference_trace=inference_trace,
            timeout_seconds=timeout_seconds,
        )
        srs_requirement = _merge_cached_srs_requirement(
            srs_requirement,
            previous_payload.get("srs_requirement"),
        )
        shared_requirement = {
            "required_srs_deg1_plus_1": srs_requirement["deg1_plus_1"],
            "required_srs_deg2_plus_1": srs_requirement["deg2_plus_1"],
        }
        terrae_srs_deg1_plus_1, terrae_srs_deg2_plus_1 = _select_srs_size(
            route_requirement=shared_requirement,
            initial_deg1_plus_1=srs_deg1_plus_1,
            initial_deg2_plus_1=srs_deg2_plus_1,
            max_deg1_plus_1=max_srs_deg1_plus_1,
            max_deg2_plus_1=max_srs_deg2_plus_1,
            case_id=case.case_id,
            route_name="shared-terrae-zkboost",
        )
        zkboost_srs_deg1_plus_1 = terrae_srs_deg1_plus_1
        zkboost_srs_deg2_plus_1 = terrae_srs_deg2_plus_1
    selected_route_names = []
    if _includes_terrae(routes):
        selected_route_names.append("Terrae")
    if _includes_zkboost(routes):
        selected_route_names.append("CertXGB")
    print(
        f"[workflow] {case.case_id}: using {' + '.join(selected_route_names)} SRS "
        f"deg1+1={terrae_srs_deg1_plus_1} deg2+1={terrae_srs_deg2_plus_1}",
        flush=True,
    )
    shared_srs_path = _ensure_srs(
        build_dir=build_dir,
        srs_cache_dir=srs_cache_dir,
        deg1_plus_1=terrae_srs_deg1_plus_1,
        deg2_plus_1=terrae_srs_deg2_plus_1,
        timeout_seconds=timeout_seconds,
    )
    terrae_srs_path = shared_srs_path
    zkboost_srs_path = shared_srs_path
    # Let the C++ loaders decode only the per-stage required SRS prefix.
    # Users can still set TERRAE_SRS_PREFIX_DECODE_FLOOR explicitly when they
    # want to force a larger decoded prefix for profiling or debugging.
    benchmark_env = dict(os.environ)
    failure_stage_aliases = {
        "terrae-training": {"terrae-training", "terrae-training-prove", "terrae-training-verify"},
        "zkboost-full-training": {"zkboost-full-training"},
        "zkboost-inference": {"zkboost-inference"},
    }
    selected_failure_stages = {
        stage
        for selector in rerun_stages
        for stage in failure_stage_aliases.get(selector, {selector})
    }
    selected_skipped_stages = {
        stage
        for selector in skipped_stages
        for stage in failure_stage_aliases.get(selector, {selector})
    }
    prior_stage_failures = previous_payload.get("stage_failures", [])
    stage_failures: list[dict[str, str]] = [
        dict(item)
        for item in prior_stage_failures
        if isinstance(item, dict) and item.get("stage") not in selected_failure_stages
    ] if rerun_stages and isinstance(prior_stage_failures, list) else []
    for selector in sorted(skipped_stages):
        stage_failures.append({
            "stage": selector,
            "status": "skipped_after_failure",
            "message": "skipped because the same proof stage failed at a smaller T",
        })

    def stage_selected_for_rerun(stage: str) -> bool:
        return any(
            stage in failure_stage_aliases.get(selector, {selector})
            for selector in rerun_stages
        )

    def should_execute_stage(stage: str) -> bool:
        return (
            stage not in selected_skipped_stages
            and (not rerun_stages or stage in rerun_stages)
        )
    stage_checkpoint_path = case_dir / "stage_checkpoint.json"
    stage_checkpoint: dict[str, Any] = {
        "schema_version": 2,
        "training_semantics_fingerprint": TRAINING_SEMANTICS_FINGERPRINT,
        "case": case.__dict__,
        "routes": routes,
        "dataset_adaptation": dataset_adaptation,
        "train_export": {
            "elapsed_seconds": train_elapsed,
            "fields": train_fields,
        },
        "terrae": dict(previous_payload.get("terrae", {}))
        if isinstance(previous_payload.get("terrae"), dict) else {},
        "zkboost": dict(previous_payload.get("zkboost", {}))
        if isinstance(previous_payload.get("zkboost"), dict) else {},
    }
    if "terrae-training" in rerun_stages:
        stage_checkpoint["terrae"].pop("training_core_memory", None)
        stage_checkpoint["terrae"].pop("training_memory", None)
        stage_checkpoint["terrae"].pop("training_prover_step", None)
        stage_checkpoint["terrae"].pop("training_verifier_step", None)
        print(
            f"[workflow] {case.case_id}: invalidating completed terrae-training",
            flush=True,
        )
    rerun_zkboost_keys = {
        "zkboost-full-training": "training",
        "zkboost-inference": "inference",
    }
    for stage, key in rerun_zkboost_keys.items():
        if stage not in rerun_stages:
            continue
        stage_checkpoint["zkboost"].pop(key, None)
        print(
            f"[workflow] {case.case_id}: invalidating completed {stage}",
            flush=True,
        )
    _write_json_atomic(stage_checkpoint_path, stage_checkpoint)

    def checkpoint_stage(route: str, key: str, value: dict[str, Any] | None) -> None:
        if not isinstance(value, dict) or not value:
            return
        stage_checkpoint.setdefault(route, {})[key] = value
        _write_json_atomic(stage_checkpoint_path, stage_checkpoint)

    def reuse_stage(route: str, key: str, stage: str) -> dict[str, Any] | None:
        if stage_selected_for_rerun(stage):
            return None
        route_payload = previous_payload.get(route)
        if not isinstance(route_payload, dict):
            return None
        value = route_payload.get(key)
        if not isinstance(value, dict) or not value:
            return None
        print(f"[workflow] {case.case_id}: reusing completed {stage}", flush=True)
        return value

    def record_prior_timeout(stage: str) -> None:
        message = "recorded timeout from a previous run; stage was not rerun"
        stage_failures.append({
            "stage": stage,
            "status": "timed_out",
            "message": message,
        })
        print(f"[workflow] {case.case_id}: {stage} timed_out: {message}", flush=True)

    def run_optional_stage(stage: str, callback: Any) -> dict[str, Any] | None:
        try:
            return callback()
        except Exception as error:
            if _is_memory_guard_error(error):
                raise
            message = _summarize_exception(error)
            status = _status_from_error_message(message)
            stage_failures.append({
                "stage": stage,
                "status": status,
                "message": message,
            })
            print(
                f"[workflow] {case.case_id}: {stage} {status}: {message}",
                flush=True,
            )
            return None

    def combine_prover_and_verifier_steps(prover: dict[str, Any], verifier: dict[str, Any]) -> dict[str, Any]:
        combined_metrics = dict(prover.get("metrics", {}))
        for name, value in verifier.get("metrics", {}).items():
            if name.startswith("cost_") and name in combined_metrics:
                # Unprefixed cost metrics describe one proof statement or an already
                # aggregated operation total.  The verifier emits the same statement
                # again, so adding it would double claims and distinct points.
                continue
            combined_metrics[name] = value
        combined_timings = dict(prover.get("timings", {}))
        combined_timings.update(verifier.get("timings", {}))
        combined_fields = dict(prover.get("fields", {}))
        combined_fields.update(verifier.get("fields", {}))
        return {
            "elapsed_seconds": prover.get("elapsed_seconds", 0.0) + verifier.get("elapsed_seconds", 0.0),
            "stdout": prover.get("stdout", "") + "\n" + verifier.get("stdout", ""),
            "metrics": combined_metrics,
            "cost_statistics": _cost_statistics(combined_metrics),
            "timings": combined_timings,
            "fields": combined_fields,
            "memory": prover.get("memory", {}),
            "prover_step": prover,
            "verifier_step": verifier,
        }

    def combine_terrae_training_with_well_formedness(
        training_core: dict[str, Any],
        well_formedness: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(training_core, dict) or not training_core:
            return {}
        if not isinstance(well_formedness, dict) or not well_formedness:
            return training_core
        if int(training_core.get("metrics", {}).get("training_includes_well_formedness", 0)) == 1:
            return training_core
        combined_metrics = dict(training_core.get("metrics", {}))
        for name, value in well_formedness.get("metrics", {}).items():
            if not name.startswith("cost_") or not isinstance(value, int):
                continue
            if name.endswith(("required_srs_deg1_plus_1", "required_srs_deg2_plus_1")):
                combined_metrics[name] = max(int(combined_metrics.get(name, 0)), value)
            else:
                combined_metrics[name] = int(combined_metrics.get(name, 0)) + value
        training_proof_bytes = int(training_core.get("metrics", {}).get("training_proof_bytes", 0))
        well_formedness_proof_bytes = int(
            well_formedness.get("metrics", {}).get("well_formedness_proof_bytes", 0)
        )
        combined_metrics["training_with_well_formedness_proof_bytes"] = (
            training_proof_bytes + well_formedness_proof_bytes
        )
        training_peak = int(
            training_core.get("metrics", {}).get(
                "proof_stage_peak_rss_bytes",
                training_core.get("memory", {}).get("peak_rss_bytes", 0),
            )
        )
        well_formedness_peak = int(
            well_formedness.get("metrics", {}).get(
                "proof_stage_peak_rss_bytes",
                well_formedness.get("memory", {}).get("peak_rss_bytes", 0),
            )
        )
        combined_metrics["proof_stage_peak_rss_bytes"] = max(
            training_peak,
            well_formedness_peak,
        )
        combined_timings = dict(training_core.get("timings", {}))
        combined_timings["prove_training_with_well_formedness"] = float(
            training_core.get("timings", {}).get("prove_training", 0.0)
        ) + float(well_formedness.get("timings", {}).get("prove_well_formedness", 0.0))
        combined_timings["verify_training_with_well_formedness_online"] = float(
            training_core.get("timings", {}).get("verify_training_online", 0.0)
        ) + float(
            well_formedness.get("timings", {}).get("verify_well_formedness_online", 0.0)
        )
        training_memory_peak = int(training_core.get("memory", {}).get("peak_rss_bytes", 0))
        well_formedness_memory_peak = int(
            well_formedness.get("memory", {}).get("peak_rss_bytes", 0)
        )
        combined_memory = dict(
            well_formedness.get("memory", {})
            if well_formedness_memory_peak > training_memory_peak
            else training_core.get("memory", {})
        )
        combined_memory["peak_rss_bytes"] = max(
            training_memory_peak,
            well_formedness_memory_peak,
        )
        return {
            "elapsed_seconds": float(training_core.get("elapsed_seconds", 0.0))
            + float(well_formedness.get("elapsed_seconds", 0.0)),
            "stdout": training_core.get("stdout", "") + "\n" + well_formedness.get("stdout", ""),
            "metrics": combined_metrics,
            "cost_statistics": _cost_statistics(combined_metrics),
            "timings": combined_timings,
            "fields": {
                **well_formedness.get("fields", {}),
                **training_core.get("fields", {}),
            },
            "memory": combined_memory,
            "training_core_step": training_core,
            "well_formedness_step": well_formedness,
        }

    def run_terrae_reusable_proof_stage(
        *,
        proof_stage: str,
        prove_mode: str,
        verify_mode: str,
        proof_path: Path,
    ) -> dict[str, Any]:
        prover_key = f"{proof_stage}_prover_step"
        verifier_key = f"{proof_stage}_verifier_step"
        prover = reuse_stage("terrae", prover_key, f"terrae-{proof_stage}-prove")
        if prover is None:
            prover = run_optional_stage(
                f"terrae-{proof_stage}-prove",
                lambda: _measured_step(
                    f"{case.case_id}: terrae-{proof_stage}-prove",
                    [
                        str(build_dir / "examples" / "example_training_then_inference_bundle"),
                        prove_mode,
                        str(terrae_srs_path),
                        str(training_trace),
                        str(inference_trace),
                        str(proof_path),
                    ],
                    cwd=REPO_ROOT,
                    output_dir=case_dir,
                    prefix=prefix,
                    timeout_seconds=timeout_seconds,
                    env=benchmark_env,
                ),
            )
            checkpoint_stage("terrae", prover_key, prover)
        if prover is None:
            return {}
        verifier = reuse_stage("terrae", verifier_key, f"terrae-{proof_stage}-verify")
        if verifier is None:
            verifier = run_optional_stage(
                f"terrae-{proof_stage}-verify",
                lambda: _measured_step(
                    f"{case.case_id}: terrae-{proof_stage}-verify",
                    [
                        str(build_dir / "examples" / "example_training_then_inference_bundle"),
                        verify_mode,
                        str(terrae_srs_path),
                        str(training_trace),
                        str(inference_trace),
                        str(proof_path),
                    ],
                    cwd=REPO_ROOT,
                    output_dir=case_dir,
                    prefix=prefix,
                    timeout_seconds=timeout_seconds,
                    env=benchmark_env,
                ),
            )
            checkpoint_stage("terrae", verifier_key, verifier)
        if verifier is None:
            return combine_prover_and_verifier_steps(prover, {})
        return combine_prover_and_verifier_steps(prover, verifier)

    def run_terrae_branch() -> dict[str, Any]:
        terrae_commit = reuse_stage("terrae", "commit_forest", "terrae-commit-forest")
        if terrae_commit is None and should_execute_stage("terrae-commit-forest"):
            terrae_commit = run_optional_stage(
                "terrae-commit-forest",
                lambda: _measured_step(
                    f"{case.case_id}: terrae-commit-forest",
                    [
                        str(build_dir / "examples" / "example_training_then_inference_bundle"),
                        "profile-commit-forest-from-split-traces",
                        str(terrae_srs_path),
                        str(training_trace),
                        str(inference_trace),
                    ],
                    cwd=REPO_ROOT,
                    output_dir=case_dir,
                    prefix=prefix,
                    timeout_seconds=timeout_seconds,
                    env=benchmark_env,
                ),
            )
            checkpoint_stage("terrae", "commit_forest", terrae_commit)

        terrae_classification = reuse_stage(
            "terrae", "classification_memory", "terrae-classification"
        )
        if terrae_classification is None and should_execute_stage("terrae-classification"):
            terrae_classification = run_terrae_reusable_proof_stage(
                proof_stage="classification",
                prove_mode="prove-classification-from-split-traces",
                verify_mode="profile-classification-verifier-from-split-traces",
                proof_path=case_dir / f"{prefix}_terrae_classification.proof",
            )
            checkpoint_stage("terrae", "classification_memory", terrae_classification)

        result = {}
        if terrae_commit is not None:
            result["commit_forest"] = terrae_commit
        if terrae_classification:
            result["classification_memory"] = terrae_classification
        if reduced_inference_only:
            return result

        terrae_well_formedness = reuse_stage(
            "terrae", "well_formedness_memory", "terrae-well-formedness"
        )
        if terrae_well_formedness is None and should_execute_stage("terrae-well-formedness"):
            terrae_well_formedness = run_optional_stage(
                "terrae-well-formedness",
                lambda: _measured_step(
                    f"{case.case_id}: terrae-well-formedness-memory",
                    [
                        str(build_dir / "examples" / "example_training_then_inference_bundle"),
                        "profile-well-formedness-from-split-traces",
                        str(terrae_srs_path),
                        str(training_trace),
                        str(inference_trace),
                    ],
                    cwd=REPO_ROOT,
                    output_dir=case_dir,
                    prefix=prefix,
                    timeout_seconds=timeout_seconds,
                    env=benchmark_env,
                ),
            )
        if terrae_well_formedness is not None:
            _deduplicate_combined_profile_pcs_statement(
                terrae_well_formedness,
                prover_scope="prover_well_formedness",
            )
            checkpoint_stage("terrae", "well_formedness_memory", terrae_well_formedness)

        terrae_training_core = reuse_stage(
            "terrae", "training_core_memory", "terrae-training"
        )
        if terrae_training_core is None and should_execute_stage("terrae-training"):
            terrae_training_core = run_terrae_reusable_proof_stage(
                proof_stage="training",
                prove_mode="prove-training-from-split-traces",
                verify_mode="profile-training-verifier-from-split-traces",
                proof_path=case_dir / f"{prefix}_terrae_training.proof",
            )
            checkpoint_stage("terrae", "training_core_memory", terrae_training_core)
        terrae_training = combine_terrae_training_with_well_formedness(
            terrae_training_core,
            terrae_well_formedness,
        )
        checkpoint_stage("terrae", "training_memory", terrae_training)

        if terrae_well_formedness is not None:
            result["well_formedness_memory"] = terrae_well_formedness
        if terrae_training:
            result["training_memory"] = terrae_training
        return result

    def run_zkboost_branch() -> dict[str, Any]:
        inference_mode = "profile-inference-preprocessed"
        zkboost_commit = reuse_stage("zkboost", "commit_forest", "zkboost-commit-forest")
        if zkboost_commit is None and should_execute_stage("zkboost-commit-forest"):
            zkboost_commit = run_optional_stage(
                "zkboost-commit-forest",
                lambda: _measured_step(
                    f"{case.case_id}: zkboost-commit-forest",
                    [
                        str(build_dir / "examples" / "example_zkboost_shared_trace"),
                        "profile-commit-forest",
                        str(zkboost_srs_path),
                        str(training_trace),
                    ],
                    cwd=REPO_ROOT,
                    output_dir=case_dir,
                    prefix=prefix,
                    timeout_seconds=timeout_seconds,
                    env=benchmark_env,
                ),
            )
            checkpoint_stage("zkboost", "commit_forest", zkboost_commit)

        zkboost_inference = reuse_stage("zkboost", "inference", "zkboost-inference")
        if zkboost_inference is None and should_execute_stage("zkboost-inference"):
            zkboost_inference = run_optional_stage(
                "zkboost-inference",
                lambda: _measured_step(
                    f"{case.case_id}: zkboost-inference",
                    [
                        str(build_dir / "examples" / "example_zkboost_shared_trace"),
                        inference_mode,
                        str(zkboost_srs_path),
                        str(training_trace),
                        str(inference_trace),
                    ],
                    cwd=REPO_ROOT,
                    output_dir=case_dir,
                    prefix=prefix,
                    timeout_seconds=timeout_seconds,
                    env=benchmark_env,
                ),
            )
            checkpoint_stage("zkboost", "inference", zkboost_inference)

        result = {}
        if zkboost_commit is not None:
            result["commit_forest"] = zkboost_commit
        if zkboost_inference is not None:
            result["inference"] = zkboost_inference
        if reduced_inference_only:
            return result

        zkboost_training = reuse_stage("zkboost", "training", "zkboost-full-training")
        if zkboost_training is None and should_execute_stage("zkboost-full-training"):
            if "zkboost-full-training" in recorded_timed_out_stages:
                record_prior_timeout("zkboost-full-training")
            else:
                zkboost_training = run_optional_stage(
                    "zkboost-full-training",
                    lambda: _measured_step(
                        f"{case.case_id}: zkboost-full-training",
                        [
                            str(build_dir / "examples" / "example_zkboost_shared_trace"),
                            "profile-cert-gbdt",
                            str(zkboost_srs_path),
                            str(training_trace),
                            str(inference_trace),
                        ],
                        cwd=REPO_ROOT,
                        output_dir=case_dir,
                        prefix=prefix,
                        timeout_seconds=timeout_seconds,
                        env=benchmark_env,
                    ),
                )
                checkpoint_stage("zkboost", "training", zkboost_training)
        if zkboost_training is not None:
            result["training"] = zkboost_training
        return result

    terrae = run_terrae_branch() if _includes_terrae(routes) else {}
    zkboost = run_zkboost_branch() if _includes_zkboost(routes) else None

    benchmark_payload = {
        "training_semantics_fingerprint": TRAINING_SEMANTICS_FINGERPRINT,
        "case": {
            "case_id": case.case_id,
            "phase": case.phase,
            "dataset": case.dataset,
            "gbdt_iterations": case.gbdt_iterations,
            "max_depth": case.max_depth,
            "bin_count": case.bin_count,
            "loss_mode": case.gradient_mode,
            "max_original_queries": case.training_samples,
            "requested_inference_samples": requested_inference_samples,
            "padded_inference_samples": inference_adaptation["padded_inference_samples"],
            "reduced_inference_only": reduced_inference_only,
        },
        "paths": {
            "srs": str(terrae_srs_path),
            "terrae_srs": str(terrae_srs_path) if _includes_terrae(routes) else None,
            "zkboost_srs": str(zkboost_srs_path) if _includes_zkboost(routes) else None,
            "training_trace": str(training_trace),
            "inference_trace": str(inference_trace),
            "bundle": None,
            "proof_config": str(proof_config),
        },
        "train_export": {
            "elapsed_seconds": train_elapsed,
            "fields": train_fields,
        },
        "srs_requirement": srs_requirement,
        "stage_failures": stage_failures,
        "terrae": terrae,
        "zkboost": zkboost,
    }

    return {
        "benchmark_payload": benchmark_payload,
        "train_fields": train_fields,
        "inference_adaptation": inference_adaptation,
        "final_srs_deg1_plus_1": max(terrae_srs_deg1_plus_1, zkboost_srs_deg1_plus_1),
        "final_srs_deg2_plus_1": max(terrae_srs_deg2_plus_1, zkboost_srs_deg2_plus_1),
        "srs_requirement": srs_requirement,
        "stage_failures": stage_failures,
    }


def _run_case_with_retries(
    case: WorkflowCase,
    *,
    case_dir: Path,
    build_dir: Path,
    srs_cache_dir: Path,
    routes: str,
    initial_deg1_plus_1: int,
    initial_deg2_plus_1: int,
    max_deg1_plus_1: int | None,
    max_deg2_plus_1: int | None,
    assume_srs_deg1_plus_1: int | None,
    assume_srs_deg2_plus_1: int | None,
    recorded_timed_out_stages: set[str],
    rerun_stages: set[str],
    skipped_stages: set[str],
    reused_benchmark_payload: dict[str, Any] | None,
    dataset_adaptation: dict[str, Any],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    payload = _run_case_once(
        case,
        case_dir=case_dir,
        build_dir=build_dir,
        srs_cache_dir=srs_cache_dir,
        routes=routes,
        srs_deg1_plus_1=initial_deg1_plus_1,
        srs_deg2_plus_1=initial_deg2_plus_1,
        max_srs_deg1_plus_1=max_deg1_plus_1,
        max_srs_deg2_plus_1=max_deg2_plus_1,
        assume_srs_deg1_plus_1=assume_srs_deg1_plus_1,
        assume_srs_deg2_plus_1=assume_srs_deg2_plus_1,
        recorded_timed_out_stages=recorded_timed_out_stages,
        rerun_stages=rerun_stages,
        skipped_stages=skipped_stages,
        reused_benchmark_payload=reused_benchmark_payload,
        dataset_adaptation=dataset_adaptation,
        timeout_seconds=timeout_seconds,
    )
    return payload


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        _prepare_build(
            args.build_dir,
            include_zkboost=_includes_zkboost(args.routes),
            timeout_seconds=args.timeout_seconds,
        )

    results: list[dict[str, Any]] = []
    srs_cache_dir = args.srs_cache_dir if args.srs_cache_dir is not None else args.output_dir / "srs"
    existing_payloads: dict[str, dict[str, Any]] = {}
    existing_report_path = args.output_dir / "workflow_benchmark_report.json"
    if args.reuse_existing_stages and existing_report_path.exists():
        existing_report = json.loads(existing_report_path.read_text(encoding="utf-8"))
        existing_payloads = {
            str(item["case"]["case_id"]): item["benchmark_payload"]
            for item in existing_report.get("cases", [])
            if isinstance(item, dict)
            and isinstance(item.get("benchmark_payload"), dict)
            and item["benchmark_payload"].get("training_semantics_fingerprint")
            == TRAINING_SEMANTICS_FINGERPRINT
        }
    for case in args.cases:
        case_dir = args.output_dir / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        dataset_adaptation = _preflight_case(case)
        result: dict[str, Any] = {
            "case": case.__dict__,
            "dataset_adaptation": dataset_adaptation,
            "status": "failed",
            "error_message": "",
        }
        reused_payload = existing_payloads.get(case.case_id)
        if args.reuse_existing_stages:
            stage_checkpoint = _read_json_object(case_dir / "stage_checkpoint.json")
            if stage_checkpoint.get("training_semantics_fingerprint") != TRAINING_SEMANTICS_FINGERPRINT:
                stage_checkpoint = {}
            reused_payload = _merge_stage_payloads(reused_payload, stage_checkpoint)
        try:
            payload = _run_case_with_retries(
                case,
                case_dir=case_dir,
                build_dir=args.build_dir,
                srs_cache_dir=srs_cache_dir,
                routes=args.routes,
                initial_deg1_plus_1=args.srs_initial_deg1_plus_1,
                initial_deg2_plus_1=args.srs_initial_deg2_plus_1,
                max_deg1_plus_1=args.max_srs_deg1_plus_1,
                max_deg2_plus_1=args.max_srs_deg2_plus_1,
                assume_srs_deg1_plus_1=args.assume_srs_deg1_plus_1,
                assume_srs_deg2_plus_1=args.assume_srs_deg2_plus_1,
                recorded_timed_out_stages=set(args.record_timed_out_stage),
                rerun_stages=set(args.rerun_stage),
                skipped_stages=set(args.skip_stage),
                reused_benchmark_payload=reused_payload,
                dataset_adaptation=dataset_adaptation,
                timeout_seconds=args.timeout_seconds,
            )
            result.update(payload)
            stage_failures = payload.get("stage_failures", [])
            if stage_failures:
                failure_statuses = {str(item.get("status", "failed")) for item in stage_failures}
                if "failed" in failure_statuses:
                    result["status"] = "failed"
                elif "timed_out" in failure_statuses:
                    result["status"] = "timed_out"
                else:
                    result["status"] = "unsupported"
                result["error_message"] = "; ".join(
                    f"{item.get('stage', 'stage')}: {item.get('message', '')}"
                    for item in stage_failures
                )
            else:
                result["status"] = "completed"
            _write_case_standard_report(
                case_dir,
                build_dir=args.build_dir,
                routes=args.routes,
                result=result,
            )
            result["standard_json"] = str(case_dir / "benchmark_report.json")
            result["standard_markdown"] = str(case_dir / "benchmark_report.md")
        except Exception as error:  # pragma: no cover - outer workflow aggregation
            result["error_message"] = _summarize_exception(error)
            result["status"] = _status_from_error_message(result["error_message"])
            stage_checkpoint = _read_json_object(case_dir / "stage_checkpoint.json")
            if (
                stage_checkpoint.get("training_semantics_fingerprint")
                == TRAINING_SEMANTICS_FINGERPRINT
                and (stage_checkpoint.get("terrae") or stage_checkpoint.get("zkboost"))
            ):
                result["benchmark_payload"] = {
                    "training_semantics_fingerprint": TRAINING_SEMANTICS_FINGERPRINT,
                    "train_export": stage_checkpoint.get("train_export", {}),
                    "terrae": stage_checkpoint.get("terrae", {}),
                    "zkboost": stage_checkpoint.get("zkboost", {}),
                }
            if args.fail_fast:
                results.append(result)
                break
        results.append(result)

    summary_environment = _report_environment(args.build_dir, results)
    summary = {
        "config": {
            "routes": args.routes,
            "cost_statistics_schema_version": 3,
            "training_semantics_fingerprint": TRAINING_SEMANTICS_FINGERPRINT,
            "build_dir": str(args.build_dir),
            "srs_initial_deg1_plus_1": args.srs_initial_deg1_plus_1,
            "srs_initial_deg2_plus_1": args.srs_initial_deg2_plus_1,
            "max_srs_deg1_plus_1": args.max_srs_deg1_plus_1,
            "max_srs_deg2_plus_1": args.max_srs_deg2_plus_1,
            "timeout_seconds": args.timeout_seconds,
            "srs_cache_dir": str(srs_cache_dir),
            "reuse_existing_stages": args.reuse_existing_stages,
            "record_timed_out_stages": args.record_timed_out_stage,
        },
        "environment": summary_environment,
        "cases": results,
    }
    summary_json = args.output_dir / "workflow_benchmark_report.json"
    summary_md = args.output_dir / "workflow_benchmark_report.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_markdown(summary_md, environment=summary_environment, routes=args.routes, results=results)
    print("Parameterized workflow benchmark complete", flush=True)
    print(f"  json: {summary_json}", flush=True)
    print(f"  markdown: {summary_md}", flush=True)
    return 0 if all(_is_terminal_status(result["status"]) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
