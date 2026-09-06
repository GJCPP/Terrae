#!/usr/bin/env python3
"""Report exact Appendix K soundness metrics without proof generation or SRS loading.

For one split-trace pair, invoke the native metadata-only verifier lowering and
write the three Terrae soundness rows. With --cases-json, first export the
canonical traces for every Terrae case in a benchmark plan, then count them.
No KZG SRS, proof generation, proof serialization, or pairing is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from benchmark_common import REPO_ROOT, _write_gbdt_proof_config
from benchmark_focus_extract import SOUNDNESS_METRIC_FIELDS, SOUNDNESS_SUMMARY_FIELDNAMES
from benchmark_parameterized_workflow import (
    WorkflowCase,
    _crop_terrae_inference_trace,
    _preflight_case,
)


DERIVATION_KIND = "exact_count_only"
EXTRA_FIELDNAMES = ["derivation_kind", "srs_loaded", "proof_generated", "notes"]
SCALAR_RE = re.compile(r"^scalar\s+(\S+)\s+(\d+)\s*$")
METRIC_RE = re.compile(r"^metric\s+(\S+)\s+(\d+)\s*$")
STAGES = (
    ("well_formedness", "well-formedness", "well-formedness"),
    ("training", "training", "training+well-formedness"),
    ("classification", "classification", "routing+score-aggregation+argmax"),
)


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
    gradient_mode: str = "first-order-rss"
    learning_rate_numerator: int = 4
    learning_rate_denominator: int = 1
    split_gain_threshold: int = 1


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _read_scalars(paths: Iterable[Path]) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="ascii") as handle:
            for line in handle:
                match = SCALAR_RE.match(line)
                if match:
                    name, value = match.groups()
                    parsed = int(value)
                    if name in out and out[name] != parsed:
                        raise ValueError(f"conflicting scalar {name} across traces")
                    out[name] = parsed
    required = {
        "training.query_count",
        "training.iteration_count",
        "training.class_count",
        "training.max_depth",
        "training.bin_count",
        "inference.query_count",
    }
    missing = sorted(required - out.keys())
    if missing:
        raise ValueError("trace is missing scalar(s): " + ", ".join(missing))
    return out


def _run_native(binary: Path, training_trace: Path, inference_trace: Path) -> dict[str, int]:
    completed = subprocess.run(
        [
            str(binary),
            "soundness-from-split-traces",
            str(training_trace),
            str(inference_trace),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    metrics: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        match = METRIC_RE.match(line)
        if match:
            metrics[match.group(1)] = int(match.group(2))
    return metrics


def _rows_from_native(
    metrics: dict[str, int],
    scalars: dict[str, int],
    *,
    case_id: str,
    dataset: str,
    source_revision: str,
    metadata: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    metadata = metadata or {}
    tree_count = scalars["training.iteration_count"] * scalars["training.class_count"]
    rows: list[dict[str, Any]] = []
    for native_stage, proof_stage, proof_scope in STAGES:
        row: dict[str, Any] = {
            "case_id": case_id,
            "proof_stage": proof_stage,
            "proof_scope": proof_scope,
            "case_status": "completed",
            "stage_status": "exact-count-only",
            "dataset": dataset,
            "source_revision": source_revision,
            "cost_statistics_schema_version": 3,
            "N_used": metadata.get("N_used", scalars["training.query_count"]),
            "N_padded": metadata.get("N_padded", scalars["training.query_count"]),
            "b_padded": metadata.get(
                "b_padded", scalars["inference.query_count"] // tree_count
            ),
            "T": metadata.get("T", scalars["training.iteration_count"]),
            "d": metadata.get("d", scalars["training.max_depth"]),
            "bin_count": metadata.get("bin_count", scalars["training.bin_count"]),
            "derivation_kind": DERIVATION_KIND,
            "srs_loaded": "no",
            "proof_generated": "no",
            "notes": "exact protocol-lowering counters; deterministic placeholder proof values; KZG finalization disabled",
        }
        for csv_name, canonical_metric in SOUNDNESS_METRIC_FIELDS.items():
            suffix = canonical_metric.removeprefix("soundness_session_")
            native_name = f"soundness_exact_{native_stage}_{suffix}"
            if native_name not in metrics:
                raise RuntimeError(f"native counter omitted metric: {native_name}")
            row[csv_name] = metrics[native_name]
        rows.append(row)
    return rows


def count_trace_pair(
    binary: Path,
    training_trace: Path,
    inference_trace: Path,
    *,
    case_id: str,
    dataset: str,
    source_revision: str,
    metadata: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    scalars = _read_scalars((training_trace, inference_trace))
    metrics = _run_native(binary, training_trace, inference_trace)
    return _rows_from_native(
        metrics,
        scalars,
        case_id=case_id,
        dataset=dataset,
        source_revision=source_revision,
        metadata=metadata,
    )


def _load_plan(path: Path) -> list[PlanCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("case JSON root must be a list")
    cases: list[PlanCase] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"cases[{index}] must be an object")
        routes = str(item.get("routes", "terrae+zkboost"))
        if routes not in {"terrae", "zkboost", "terrae+zkboost"}:
            raise ValueError(f"{item.get('case_id', index)}: invalid routes")
        cases.append(
            PlanCase(
                order=int(item.get("order", index + 1)),
                case_id=str(item["case_id"]),
                phase=str(item.get("phase", "custom")),
                dataset=str(item.get("dataset", "dry-bean")),
                N=int(item["N"]),
                T=int(item["T"]),
                d=int(item["d"]),
                bins=int(item.get("bins", item.get("bin_count", 8))),
                b=int(item.get("b", item["N"])),
                routes=routes,
                gradient_mode=str(item.get("loss_mode", item.get("gradient_mode", "first-order-rss"))),
                learning_rate_numerator=int(item.get("learning_rate_numerator", 4)),
                learning_rate_denominator=int(item.get("learning_rate_denominator", 1)),
                split_gain_threshold=int(item.get("split_gain_threshold", 1)),
            )
        )
    return sorted(cases, key=lambda case: case.order)


def _workflow_case(case: PlanCase) -> WorkflowCase:
    return WorkflowCase(
        case_id=case.case_id,
        phase=case.phase,
        dataset=case.dataset,
        training_samples=case.N,
        gbdt_iterations=case.T,
        max_depth=case.d,
        bin_count=case.bins,
        inference_samples=case.b,
        gradient_mode=case.gradient_mode,
        learning_rate_numerator=case.learning_rate_numerator,
        learning_rate_denominator=case.learning_rate_denominator,
        split_gain_threshold=case.split_gain_threshold,
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _export_and_count_case(
    case: PlanCase,
    *,
    binary: Path,
    output_dir: Path,
    source_revision: str,
) -> tuple[int, str, list[dict[str, Any]], dict[str, Any]]:
    workflow_case = _workflow_case(case)
    adaptation = _preflight_case(workflow_case)
    case_dir = output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{_safe_name(case.dataset)}_T{case.T}_d{case.d}_bins{case.bins}"
        f"_{_safe_name(case.gradient_mode)}_N{case.N}_b{case.b}"
    )
    proof_config = case_dir / f"{prefix}_proof_config.json"
    training_trace = case_dir / f"{prefix}_training.trace"
    inference_full = case_dir / f"{prefix}_inference_full.trace"
    inference_trace = case_dir / f"{prefix}_inference.trace"
    native_log = case_dir / "soundness_native.log"

    _write_gbdt_proof_config(
        proof_config,
        dataset_name=case.dataset,
        iterations=case.T,
        max_depth=case.d,
        bin_count=case.bins,
        gradient_mode=case.gradient_mode,
        learning_rate_numerator=case.learning_rate_numerator,
        learning_rate_denominator=case.learning_rate_denominator,
        split_gain_threshold=case.split_gain_threshold,
    )
    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/python_trace_export/train_fixed_point_dataset.py"),
        "--config",
        str(proof_config),
        "--dataset",
        case.dataset,
        "--iterations",
        str(case.T),
        "--max-depth",
        str(case.d),
        "--bin-count",
        str(case.bins),
    ]
    if case.dataset in ("dry-bean", "optdigits-7class-train"):
        train_cmd.append("--leakage-free-evaluation-training-split")
    train_cmd += [
        "--max-original-queries",
        str(case.N),
        "--training-trace",
        str(training_trace),
        "--inference-trace",
        str(inference_full),
    ]
    train = subprocess.run(train_cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    (case_dir / "train_export.log").write_text(
        train.stdout + train.stderr, encoding="utf-8"
    )

    scalars = _read_scalars((training_trace, inference_full))
    tree_count = scalars["training.iteration_count"] * scalars["training.class_count"]
    crop = _crop_terrae_inference_trace(
        inference_full,
        inference_trace,
        tree_count=tree_count,
        requested_inference_samples=case.b,
    )
    selected_inference = (
        inference_full
        if crop["padded_inference_samples"] == crop["source_sample_count"]
        else inference_trace
    )
    native = subprocess.run(
        [
            str(binary),
            "soundness-from-split-traces",
            str(training_trace),
            str(selected_inference),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    native_log.write_text(native.stdout + native.stderr, encoding="utf-8")
    metrics = {
        match.group(1): int(match.group(2))
        for line in native.stdout.splitlines()
        if (match := METRIC_RE.match(line))
    }
    selected_scalars = _read_scalars((training_trace, selected_inference))
    rows = _rows_from_native(
        metrics,
        selected_scalars,
        case_id=case.case_id,
        dataset=case.dataset,
        source_revision=source_revision,
        metadata={
            "N_used": int(adaptation["actual_training_original_samples"]),
            "N_padded": int(adaptation["padded_training_samples"]),
            "b_padded": int(crop["padded_inference_samples"]),
            "T": case.T,
            "d": case.d,
            "bin_count": case.bins,
        },
    )
    manifest = {
        "case_id": case.case_id,
        "training_trace": str(training_trace),
        "inference_trace": str(selected_inference),
        "native_log": str(native_log),
        "status": "completed",
    }
    return case.order, case.case_id, rows, manifest


def _write_csv(handle: TextIO, rows: list[dict[str, Any]]) -> None:
    fields = [*SOUNDNESS_SUMMARY_FIELDNAMES, *EXTRA_FIELDNAMES]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})


def run_plan(
    cases_json: Path,
    *,
    binary: Path,
    output_dir: Path,
    jobs: int,
    source_revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = [case for case in _load_plan(cases_json) if case.routes in {"terrae", "terrae+zkboost"}]
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[tuple[int, str, list[dict[str, Any]], dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="soundness") as pool:
        futures = {
            pool.submit(
                _export_and_count_case,
                case,
                binary=binary,
                output_dir=output_dir,
                source_revision=source_revision,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            result = future.result()
            completed.append(result)
            print(f"[soundness] completed {case.case_id}", flush=True)
    completed.sort(key=lambda item: item[0])
    rows = [row for _, _, case_rows, _ in completed for row in case_rows]
    manifest = [entry for _, _, _, entry in completed]
    return rows, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_trace", type=Path, nargs="?")
    parser.add_argument("inference_trace", type=Path, nargs="?")
    parser.add_argument("--cases-json", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO_ROOT / "build/examples/example_training_then_inference_bundle",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--source-revision", default="")
    args = parser.parse_args(argv)
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.cases_json:
        if args.training_trace or args.inference_trace:
            parser.error("positional traces cannot be combined with --cases-json")
        if args.output_dir is None:
            parser.error("--output-dir is required with --cases-json")
    elif not args.training_trace or not args.inference_trace:
        parser.error("provide both traces, or use --cases-json with --output-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    revision = args.source_revision or _git_revision()
    try:
        if args.cases_json:
            rows, manifest = run_plan(
                args.cases_json,
                binary=args.binary,
                output_dir=args.output_dir,
                jobs=args.jobs,
                source_revision=revision,
            )
            output = args.output or (args.output_dir / "soundness_metrics_exact.csv")
            with output.open("w", encoding="utf-8", newline="") as handle:
                _write_csv(handle, rows)
            manifest_path = args.output_dir / "soundness_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "derivation_kind": DERIVATION_KIND,
                        "srs_loaded": False,
                        "proof_generated": False,
                        "cases_json": str(args.cases_json),
                        "cases": manifest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"soundness metrics: {output}")
            print(f"soundness manifest: {manifest_path}")
            return 0

        rows = count_trace_pair(
            args.binary,
            args.training_trace,
            args.inference_trace,
            case_id=args.case_id or args.training_trace.stem,
            dataset=args.dataset,
            source_revision=revision,
        )
        output = args.output or Path("soundness_metrics_exact.csv")
        if str(output) == "-":
            _write_csv(sys.stdout, rows)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                _write_csv(handle, rows)
            print(f"soundness metrics: {output}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"soundness-metrics error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
