#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_common import (
    REPO_ROOT,
    _parse_key_value_lines,
    _parse_metrics_and_timings,
    _run_logged,
    _run_measured_logged,
    _write_gbdt_proof_config,
)
from source_revision import source_revision


@dataclass(frozen=True)
class StandardCase:
    case_id: str
    dataset: str
    gbdt_iterations: int
    max_depth: int
    bin_count: int
    max_original_queries: int | None


SMALL_CASES = [
    StandardCase("iris_q64_d2_b2", "iris-binary", 1, 2, 2, 64),
    StandardCase("iris_q128_d2_b8", "iris-binary", 1, 2, 8, 128),
    StandardCase("wine_q128_d2_b8", "wine-binary", 1, 2, 8, 128),
]

DEFAULT_TIMEOUT_SECONDS = 7200.0


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _run_text(cmd: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def _source_revision(repo_root: Path = REPO_ROOT) -> str:
    return source_revision(repo_root)


def _environment(build_dir: Path, srs_deg1_plus_1: int, srs_deg2_plus_1: int) -> dict[str, str]:
    cpu_model = "unknown"
    logical_cpus = "unknown"
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
        logical_cpus = str(sum(1 for line in cpuinfo.splitlines() if line.startswith("processor")))
    except OSError:
        pass
    return {
        "source_revision": _source_revision(),
        "build_dir": str(build_dir),
        "machine": _run_text(["hostname"], cwd=REPO_ROOT),
        "cpu": cpu_model,
        "logical_cpus": logical_cpus,
        "os": _run_text(["uname", "-srmo"], cwd=REPO_ROOT),
        "srs_deg1_plus_1": str(srs_deg1_plus_1),
        "srs_deg2_plus_1": str(srs_deg2_plus_1),
    }


def _prepare_build(build_dir: Path, *, include_zkboost: bool, timeout_seconds: float | None = None) -> None:
    configure_command = [
        "cmake",
        "-S",
        ".",
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=native -mtune=native",
    ]
    mcl_source_dir = os.environ.get("TERRAE_MCL_SOURCE_DIR", "").strip()
    if mcl_source_dir:
        configure_command.append(f"-DTERRAE_MCL_SOURCE_DIR={Path(mcl_source_dir).expanduser().resolve()}")
    _run_logged(
        "configure",
        configure_command,
        cwd=REPO_ROOT,
        timeout_seconds=timeout_seconds,
    )
    targets = [
        "example_training_then_inference_bundle",
        "kzg_setup_helper",
    ]
    if include_zkboost:
        targets.append("example_zkboost_shared_trace")
    _run_logged(
        "build",
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            *targets,
        ],
        cwd=REPO_ROOT,
        timeout_seconds=timeout_seconds,
    )


def _metrics(stdout: str) -> tuple[dict[str, int], dict[str, float], dict[str, str]]:
    metrics, timings = _parse_metrics_and_timings(stdout)
    fields = _parse_key_value_lines(stdout)
    return metrics, timings, fields


def _cost_statistics(metrics: dict[str, int]) -> dict[str, int]:
    return {
        name[len("cost_"):]: value
        for name, value in metrics.items()
        if name.startswith("cost_")
    }


COMMON_PCS_STATEMENT_METRICS = {
    "polynomial_point_claim_count",
}
COMMON_PCS_OPERATION_METRICS = {
    "pairing_count",
}
COMMITTED_DATA_METRICS = {
    "polynomial_commitment_operation_count",
    "committed_polynomial_coefficient_count",
}


PAPER_NATIVE_METRICS = {"zkboost_plonk_active_gate_row_count"}


def _is_common_pcs_backend_metric(name: str) -> bool:
    return (
        name in COMMON_PCS_STATEMENT_METRICS
        or name in COMMON_PCS_OPERATION_METRICS
        or name in COMMITTED_DATA_METRICS
        or name in PAPER_NATIVE_METRICS
    )


def _common_pcs_backend_statistics(payload: dict[str, Any]) -> dict[str, int]:
    output = {
        name: value
        for name, value in payload.get("cost_statistics", {}).items()
        if _is_common_pcs_backend_metric(name)
    }
    return output


def _deduplicate_combined_profile_pcs_statement(
    payload: dict[str, Any],
    *,
    prover_scope: str,
) -> None:
    """Keep one copy of PCS statement/provenance counters in a prove+verify profile.

    Some legacy profile modes execute proving and verification in one process.
    The profiler consequently emits the same proof statement once under each
    phase scope and adds both copies to the unscoped total. Backend operations
    are genuinely phase-specific and remain untouched.
    """
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return
    scope_prefix = f"{prover_scope}_"
    for scoped_name, value in list(_cost_statistics(metrics).items()):
        if not scoped_name.startswith(scope_prefix):
            continue
        unscoped_name = scoped_name[len(scope_prefix):]
        if unscoped_name in COMMON_PCS_STATEMENT_METRICS:
            metrics[f"cost_{unscoped_name}"] = value
    payload["cost_statistics"] = _cost_statistics(metrics)


def _committed_data_statistics(payload: dict[str, Any]) -> dict[str, int]:
    return {
        name: value
        for name, value in payload.get("cost_statistics", {}).items()
        if any(name == metric or name.endswith("_" + metric) for metric in COMMITTED_DATA_METRICS)
    }


def _native_system_statistics(payload: dict[str, Any]) -> dict[str, int]:
    value = payload.get("cost_statistics", {}).get("zkboost_plonk_active_gate_row_count")
    return {"zkboost_plonk_active_gate_row_count": value} if isinstance(value, int) else {}


def _workload_statistics(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, int]:
    metrics = payload.get("metrics", {})
    fields = result.get("train_export", {}).get("fields", {})
    aliases = {
        "record_count_padded": ("record_count_padded",),
        "tree_count": ("tree_count",),
        "max_depth": ("max_depth", "forest_max_depth"),
        "feature_count": ("feature_count",),
        "bin_count": ("bin_count",),
        "class_count": ("class_count",),
        "node_count": ("active_node_count", "node_count"),
        "active_node_count": ("active_node_count", "node_count"),
        "allocated_node_slot_count": ("allocated_node_slot_count",),
        "internal_node_count": ("internal_node_count", "split_node_count"),
        "leaf_node_count": ("semantic_leaf_node_count", "leaf_node_count"),
        "candidate_count": ("candidate_count",),
        "inference_sample_count": ("inference_sample_count", "sample_count"),
        "checked_path_count": ("checked_path_count", "checked_paths", "query_count"),
    }
    output: dict[str, int] = {}
    for output_name, input_names in aliases.items():
        for input_name in input_names:
            value = metrics.get(input_name)
            if isinstance(value, int):
                output[f"workload_{output_name}"] = value
                break
    for output_name, field_name in [
        ("record_count_used", "original_samples"),
        ("record_count_padded", "padded_samples"),
        ("tree_count", "tree_count"),
        ("max_depth", "max_depth"),
        ("feature_count", "feature_count"),
        ("class_count", "trace_class_count"),
        ("node_count", "active_node_count"),
        ("active_node_count", "active_node_count"),
        ("allocated_node_slot_count", "allocated_node_slot_count"),
    ]:
        value = fields.get(field_name)
        if value not in (None, ""):
            output[f"workload_{output_name}"] = int(value)
    return output


def _measured_step(
    step: str,
    command: list[str],
    *,
    cwd: Path,
    output_dir: Path,
    prefix: str,
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    elapsed, stdout, memory = _run_measured_logged(
        step,
        command,
        cwd=cwd,
        output_dir=output_dir,
        prefix=prefix,
        env=env,
        timeout_seconds=timeout_seconds,
    )
    metrics, timings, fields = _metrics(stdout)
    return {
        "elapsed_seconds": elapsed,
        "stdout": stdout,
        "metrics": metrics,
        "cost_statistics": _cost_statistics(metrics),
        "timings": timings,
        "fields": fields,
        "memory": memory,
    }


def _run_case(
    case: StandardCase,
    *,
    build_dir: Path,
    output_dir: Path,
    srs_deg1_plus_1: int,
    srs_deg2_plus_1: int,
    include_zkboost: bool,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    prefix = f"{_safe_name(case.dataset)}_gbdt_iter{case.gbdt_iterations}_depth{case.max_depth}_bins{case.bin_count}"
    if case.max_original_queries is not None:
        prefix += f"_q{case.max_original_queries}"

    srs_path = output_dir / f"{prefix}.srs"
    training_trace = output_dir / f"{prefix}_gbdt_training.trace"
    inference_trace = output_dir / f"{prefix}_gbdt_inference.trace"
    proof_config = output_dir / f"{prefix}_gbdt_proof_config.json"

    _write_gbdt_proof_config(
        proof_config,
        dataset_name=case.dataset,
        iterations=case.gbdt_iterations,
        max_depth=case.max_depth,
        bin_count=case.bin_count,
    )

    _run_logged(
        f"{case.case_id}: generate-srs",
        [
            str(build_dir / "examples" / "kzg_setup_helper"),
            str(srs_path),
            str(srs_deg1_plus_1),
            str(srs_deg2_plus_1),
        ],
        cwd=REPO_ROOT,
        timeout_seconds=timeout_seconds,
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
                    and case.max_original_queries is not None
                )
                else []
            ),
            *(["--max-original-queries", str(case.max_original_queries)] if case.max_original_queries is not None else []),
            "--training-trace",
            str(training_trace),
            "--inference-trace",
            str(inference_trace),
        ],
        cwd=REPO_ROOT,
        timeout_seconds=timeout_seconds,
    )

    def run_terrae_branch() -> dict[str, Any]:
        terrae_commit = _measured_step(
            f"{case.case_id}: terrae-commit-forest",
            [
                str(build_dir / "examples" / "example_training_then_inference_bundle"),
                "profile-commit-forest-from-split-traces",
                str(srs_path),
                str(training_trace),
                str(inference_trace),
            ],
            cwd=REPO_ROOT,
            output_dir=output_dir,
            prefix=prefix,
            timeout_seconds=timeout_seconds,
        )
        terrae_well_formedness = _measured_step(
            f"{case.case_id}: terrae-well-formedness-memory",
            [
                str(build_dir / "examples" / "example_training_then_inference_bundle"),
                "profile-well-formedness-from-split-traces",
                str(srs_path),
                str(training_trace),
                str(inference_trace),
            ],
            cwd=REPO_ROOT,
            output_dir=output_dir,
            prefix=prefix,
            timeout_seconds=timeout_seconds,
        )
        terrae_training = _measured_step(
            f"{case.case_id}: terrae-training-memory",
            [
                str(build_dir / "examples" / "example_training_then_inference_bundle"),
                "profile-training-from-split-traces",
                str(srs_path),
                str(training_trace),
                str(inference_trace),
            ],
            cwd=REPO_ROOT,
            output_dir=output_dir,
            prefix=prefix,
            timeout_seconds=timeout_seconds,
        )
        terrae_inference = _measured_step(
            f"{case.case_id}: terrae-inference-memory",
            [
                str(build_dir / "examples" / "example_training_then_inference_bundle"),
                "profile-inference-from-split-traces",
                str(srs_path),
                str(training_trace),
                str(inference_trace),
            ],
            cwd=REPO_ROOT,
            output_dir=output_dir,
            prefix=prefix,
            timeout_seconds=timeout_seconds,
        )
        return {
            "commit_forest": terrae_commit,
            "well_formedness_memory": terrae_well_formedness,
            "training_memory": terrae_training,
            "inference_memory": terrae_inference,
        }

    def run_zkboost_branch() -> dict[str, Any]:
        zkboost_commit = _measured_step(
            f"{case.case_id}: zkboost-commit-forest",
            [
                str(build_dir / "examples" / "example_zkboost_shared_trace"),
                "profile-commit-forest",
                str(srs_path),
                str(training_trace),
            ],
            cwd=REPO_ROOT,
            output_dir=output_dir,
            prefix=prefix,
            timeout_seconds=timeout_seconds,
        )
        zkboost_inference = _measured_step(
            f"{case.case_id}: zkboost-inference",
            [
                str(build_dir / "examples" / "example_zkboost_shared_trace"),
                "profile-inference",
                str(srs_path),
                str(training_trace),
                str(inference_trace),
            ],
            cwd=REPO_ROOT,
            output_dir=output_dir,
            prefix=prefix,
            timeout_seconds=timeout_seconds,
        )
        zkboost_training = _measured_step(
            f"{case.case_id}: zkboost-full-training",
            [
                str(build_dir / "examples" / "example_zkboost_shared_trace"),
                "profile-cert-gbdt",
                str(srs_path),
                str(training_trace),
                str(inference_trace),
            ],
            cwd=REPO_ROOT,
            output_dir=output_dir,
            prefix=prefix,
            timeout_seconds=timeout_seconds,
        )
        return {
            "commit_forest": zkboost_commit,
            "inference": zkboost_inference,
            "training": zkboost_training,
        }

    if include_zkboost:
        terrae = run_terrae_branch()
        zkboost: dict[str, Any] | None = run_zkboost_branch()
    else:
        terrae = run_terrae_branch()
        zkboost = None

    return {
        "case": case.__dict__,
        "paths": {
            "srs": str(srs_path),
            "training_trace": str(training_trace),
            "inference_trace": str(inference_trace),
            "bundle": None,
            "proof_config": str(proof_config),
        },
        "train_export": {
            "elapsed_seconds": train_elapsed,
            "fields": _parse_key_value_lines(train_stdout),
        },
        "terrae": terrae,
        "zkboost": zkboost,
    }


def _fmt_s(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "unavailable"


def _fmt_i(value: Any) -> str:
    return str(value) if isinstance(value, int) else "unavailable"


def _peak(step: dict[str, Any]) -> int | None:
    stage_value = step.get("metrics", {}).get("proof_stage_peak_rss_bytes")
    if isinstance(stage_value, int):
        return stage_value
    value = step.get("memory", {}).get("peak_rss_bytes")
    return value if isinstance(value, int) else None


def _mib(bytes_value: int | None) -> str:
    if bytes_value is None:
        return "unavailable"
    return f"{bytes_value / (1024 * 1024):.1f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    env = report["environment"]
    cases = report["cases"]
    include_terrae = report["config"]["routes"] in {"terrae", "terrae+zkboost"}
    include_zkboost = report["config"]["routes"] in {"zkboost", "terrae+zkboost"}
    lines: list[str] = [
        "# Standard Benchmark Report",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in ["source_revision", "build_dir", "machine", "cpu", "logical_cpus", "os", "srs_deg1_plus_1", "srs_deg2_plus_1"]:
        lines.append(f"| {key} | `{env[key]}` |")
    lines.extend([
        "| timing evidence | `single completed run per case` |",
        "| memory evidence | `external /proc process-tree peak RSS, with Terrae proof rows preferring stage-local RSS when available` |",
        f"| route scope | `{report['config']['routes']}` |",
        f"| cost statistics schema | `{report['config'].get('cost_statistics_schema_version', 'unversioned')}` |",
        "",
        "## Case Shape",
        "",
        "| case_id | route | dataset | samples | features | bins | trees | depth | active nodes | leaves | checked_paths | comparability |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for result in cases:
        case = result["case"]
        terrae = result["terrae"]
        terrae_commit = terrae.get("commit_forest")
        terrae_training = terrae.get("training_memory")
        terrae_inference = (
            result["terrae"].get("inference_memory")
            or result["terrae"].get("classification_memory")
        )
        g_metrics: dict[str, Any] = {}
        for payload in [terrae_commit, terrae_training, terrae_inference, terrae.get("full", {})]:
            if payload is not None:
                g_metrics.update(payload.get("metrics", {}))
        zki = result["zkboost"].get("inference") if include_zkboost and result["zkboost"] is not None else None
        z_metrics = zki["metrics"] if zki is not None else {}
        checked_paths = z_metrics.get(
            "checked_paths",
            terrae_inference["metrics"].get("query_count", 0) if terrae_inference is not None else 0,
        )
        samples = (
            result["train_export"]["fields"].get("original_samples")
            or case["max_original_queries"]
            or z_metrics.get("sample_count")
            or "full"
        )
        features = g_metrics.get("feature_count", z_metrics.get("feature_count", 0))
        bins = case["bin_count"]
        # Native proof metrics expose padded trace slots as `node_count`.
        # Prefer the exporter field, which counts active dataset-class nodes.
        active_nodes = (
            result["train_export"]["fields"].get("active_node_count")
            or g_metrics.get("active_node_count")
            or z_metrics.get("active_node_count")
            or g_metrics.get("node_count", z_metrics.get("node_count", 0))
        )
        if include_terrae:
            lines.append(
                f"| `{case['case_id']}` | Terrae | `{case['dataset']}` | {samples} | {features} | {bins} | "
                f"{g_metrics.get('tree_count', 0)} | {case['max_depth']} | {active_nodes} | "
                f"{g_metrics.get('leaf_node_count', z_metrics.get('leaf_node_count', 0))} | {checked_paths} | shared trace |"
            )
        if include_zkboost:
            lines.append(
                f"| `{case['case_id']}` | CertXGB shared | `{case['dataset']}` | {samples} | {features} | {bins} | "
                f"{z_metrics.get('tree_count', 0)} | {case['max_depth']} | {active_nodes} | "
                f"{g_metrics.get('leaf_node_count', z_metrics.get('leaf_node_count', 0))} | {checked_paths} | shared trace |"
            )

    lines.extend([
        "",
        "## Proof Costs",
        "",
        "| case_id | route | proof_stage | proof_bytes | prover_time_s | verifier_time_s | peak_memory_bytes | query_count | verified |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for result in cases:
        case = result["case"]
        full = result["terrae"].get("full", {})
        wm = result["terrae"].get("well_formedness_memory")
        tm = result["terrae"].get("training_memory")
        im = (
            result["terrae"].get("inference_memory")
            or result["terrae"].get("classification_memory")
        )
        zki = result["zkboost"].get("inference") if include_zkboost and result["zkboost"] is not None else None
        zkt = result["zkboost"].get("training") if include_zkboost and result["zkboost"] is not None else None
        query_count = (
            zki["metrics"].get(
                "checked_paths",
                im["metrics"].get("query_count", case["max_original_queries"] or 0) if im is not None else case["max_original_queries"] or 0,
            )
            if zki is not None
            else (im["metrics"].get("query_count", case["max_original_queries"] or 0) if im is not None else case["max_original_queries"] or 0)
        )
        if wm is not None:
            wm_verify_time = wm['timings'].get(
                'verify_well_formedness_online',
                full.get('timings', {}).get('verify_well_formedness_online'),
            )
            lines.append(
                f"| `{case['case_id']}` | Terrae | well-formedness | {wm['metrics'].get('well_formedness_proof_bytes', full.get('metrics', {}).get('well_formedness_proof_bytes', 0))} | "
                f"{_fmt_s(wm['timings'].get('prove_well_formedness', full.get('timings', {}).get('prove_well_formedness')))} | "
                f"{_fmt_s(wm_verify_time)} | "
                f"{_fmt_i(_peak(wm))} | not applicable | {'yes' if wm_verify_time is not None else 'unavailable'} |"
            )
        if tm is not None:
            tm_verify_time = (
                tm['timings'].get('verify_training_with_well_formedness_online')
                or tm['timings'].get('verify_training_online', full.get('timings', {}).get('verify_training_online'))
            )
            lines.append(
                f"| `{case['case_id']}` | Terrae | training | {tm['metrics'].get('training_with_well_formedness_proof_bytes') or tm['metrics'].get('training_proof_bytes', full.get('metrics', {}).get('training_proof_bytes', 0))} | "
                f"{_fmt_s(tm['timings'].get('prove_training_with_well_formedness') or tm['timings'].get('prove_training', full.get('timings', {}).get('prove_training')))} | "
                f"{_fmt_s(tm_verify_time)} | "
                f"{_fmt_i(_peak(tm))} | not applicable | {'yes' if tm_verify_time is not None else 'unavailable'} |"
            )
        if im is not None:
            im_verify_time = (
                im['timings'].get('verify_inference_online')
                or im['timings'].get('verify_classification_online')
                or full.get('timings', {}).get('verify_inference_online')
            )
            lines.append(
                f"| `{case['case_id']}` | Terrae | inference routing+sum+argmax | {im['metrics'].get('inference_proof_bytes') or im['metrics'].get('classification_proof_bytes', full.get('metrics', {}).get('inference_proof_bytes', 0))} | "
                f"{_fmt_s(im['timings'].get('prove_inference') or im['timings'].get('prove_classification', full.get('timings', {}).get('prove_inference')))} | "
                f"{_fmt_s(im_verify_time)} | "
                f"{_fmt_i(_peak(im))} | {query_count} | {'yes' if im_verify_time is not None else 'unavailable'} |"
            )
        if include_zkboost and zkt is not None:
            lines.append(
                f"| `{case['case_id']}` | CertXGB shared | training | {zkt['metrics'].get('proof_bytes', 0)} | "
                f"{_fmt_s(zkt['timings'].get('prove'))} | {_fmt_s(zkt['timings'].get('verify'))} | "
                f"{_fmt_i(_peak(zkt))} | not applicable | yes |"
            )
        if include_zkboost and zki is not None:
            lines.append(
                f"| `{case['case_id']}` | CertXGB shared | inference routing+sum+argmax | {zki['metrics'].get('proof_bytes', 0)} | "
                f"{_fmt_s(zki['timings'].get('prove'))} | {_fmt_s(zki['timings'].get('verify'))} | "
                f"{_fmt_i(_peak(zki))} | {query_count} | yes |"
            )

    lines.extend([
        "",
        "## Stage Memory",
        "",
        "| case_id | route | measured_stage | peak_rss_bytes | peak_rss_mib | samples | memory_json |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for result in cases:
        case = result["case"]
        stage_rows = []
        if "commit_forest" in result["terrae"]:
            stage_rows.append(("Terrae", "commit-forest", result["terrae"]["commit_forest"]))
        if "full" in result["terrae"]:
            stage_rows.insert(0, ("Terrae", "full-profile", result["terrae"]["full"]))
        if "well_formedness_memory" in result["terrae"]:
            stage_rows.append(("Terrae", "well-formedness", result["terrae"]["well_formedness_memory"]))
        if "training_memory" in result["terrae"]:
            stage_rows.append(("Terrae", "training", result["terrae"]["training_memory"]))
        if "inference_memory" in result["terrae"]:
            stage_rows.append(("Terrae", "inference routing+sum+argmax", result["terrae"]["inference_memory"]))
        elif "classification_memory" in result["terrae"]:
            stage_rows.append(("Terrae", "inference routing+sum+argmax", result["terrae"]["classification_memory"]))
        if include_zkboost and result["zkboost"] is not None:
            if "commit_forest" in result["zkboost"]:
                stage_rows.append(("CertXGB shared", "commit-forest", result["zkboost"]["commit_forest"]))
            if "training" in result["zkboost"]:
                stage_rows.append(("CertXGB shared", "training", result["zkboost"]["training"]))
            if "inference" in result["zkboost"]:
                stage_rows.append(("CertXGB shared", "inference routing+sum+argmax", result["zkboost"]["inference"]))
        for route, stage, payload in stage_rows:
            peak = _peak(payload)
            memory = payload["memory"]
            lines.append(
                f"| `{case['case_id']}` | {route} | {stage} | {_fmt_i(peak)} | {_mib(peak)} | "
                f"{_fmt_i(memory.get('samples'))} | `{memory.get('measurement_json', '')}` |"
            )

    statistics_rows: list[tuple[str, str, str, dict[str, Any], dict[str, Any]]] = []
    for result in cases:
        case_id = result["case"]["case_id"]
        terrae = result["terrae"]
        if terrae.get("well_formedness_memory") is not None:
            statistics_rows.append((case_id, "Terrae", "well-formedness", terrae["well_formedness_memory"], result))
        if terrae.get("training_memory") is not None:
            statistics_rows.append((case_id, "Terrae", "training", terrae["training_memory"], result))
        if terrae.get("inference_memory") is not None:
            statistics_rows.append((case_id, "Terrae", "inference routing+sum+argmax", terrae["inference_memory"], result))
        elif terrae.get("classification_memory") is not None:
            statistics_rows.append((case_id, "Terrae", "inference routing+sum+argmax", terrae["classification_memory"], result))
        if include_zkboost and result.get("zkboost") is not None:
            zkboost = result["zkboost"]
            if zkboost.get("training") is not None:
                statistics_rows.append((case_id, "CertXGB shared", "training", zkboost["training"], result))
            if zkboost.get("inference") is not None:
                statistics_rows.append((case_id, "CertXGB shared", "inference routing+sum+argmax", zkboost["inference"], result))

    lines.extend([
        "",
        "## Common PCS / Backend Statistics",
        "",
        "Only paper-used counters are included here: KZG coefficients, polynomial commitments, polynomial-point claims, verifier pairings, and CertXGB active PLONK gates. Prover/verifier copies of the same proof statement are not added together.",
        "",
        "These counters were collected during the same measured stage execution as the proof-cost rows above; no proof or verification pass was repeated for statistics.",
        "",
        "| case_id | route | stage | metric | value |",
        "| --- | --- | --- | --- | ---: |",
    ])
    for case_id, route, stage, payload, _ in statistics_rows:
        for name, value in sorted(_common_pcs_backend_statistics(payload).items()):
            lines.append(f"| `{case_id}` | {route} | {stage} | `{name}` | {value} |")

    lines.extend([
        "",
        "## Accuracy And Digest",
        "",
        "| case_id | route | accuracy | prediction_digest |",
        "| --- | --- | --- | --- |",
    ])
    for result in cases:
        case = result["case"]
        train_fields = result["train_export"]["fields"]
        if include_terrae:
            lines.append(
                f"| `{case['case_id']}` | Terrae | `{train_fields.get('inference_accuracy', '')}` | unavailable |"
            )
        if include_zkboost and result["zkboost"] is not None:
            zki = result["zkboost"].get("inference")
            if zki is not None:
                lines.append(
                    f"| `{case['case_id']}` | CertXGB shared inference | `{zki['fields'].get('shared_accuracy', '')}` | "
                    f"`{zki['fields'].get('shared_prediction_digest', '')}` |"
                )

    stage_failures: list[tuple[str, dict[str, Any]]] = []
    for result in cases:
        for failure in result.get("stage_failures", []):
            stage_failures.append((result["case"]["case_id"], failure))
    if stage_failures:
        lines.extend([
            "",
            "## Stage Failures",
            "",
            "| case_id | stage | status | message |",
            "| --- | --- | --- | --- |",
        ])
        for case_id, failure in stage_failures:
            message = str(failure.get("message", "")).replace("\n", " ")
            lines.append(
                f"| `{case_id}` | `{failure.get('stage', '')}` | "
                f"{failure.get('status', '')} | {message} |"
            )

    lines.extend([
        "",
        "## Notes",
        "",
        f"- Route scope: `{report['config']['routes']}`.",
        "- Terrae and CertXGB route branches are run sequentially when both routes are requested.",
        "- Terrae stage-profile commands emit proof size, prover time, verifier time, and peak memory together; the deprecated full-profile duplicate pass is not run by this benchmark runner.",
        "- Terrae proof-stage rows prefer in-process stage-local peak RSS metrics when available. The external memory JSON still records the full profiled command boundary.",
        "- Terrae training rows combine core training and well-formedness from the same run: proof bytes and prover/verifier times are added, while peak memory is the maximum of the two sequential stages; the separate well-formedness row remains available for breakdown.",
        "- Terrae inference rows run the exported-forest classification workflow: route/fetch leaf scores, sum scores across trees, and prove the class argmax.",
        "- CertXGB shared training rows use `training_core` scope and exclude final class prediction/argmax; inference rows include routing/fetch, score aggregation, and tie-broken class argmax in the reported proof bytes and prover/verifier totals.",
        "- Constraint/PCS counters are collected inside those same measured stage commands; the runner does not launch an additional proof or verifier pass for them.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cases_from_args(args: argparse.Namespace) -> list[StandardCase]:
    if args.preset == "small":
        return SMALL_CASES
    return [
        StandardCase(
            args.case_id or f"{_safe_name(args.dataset)}_q{args.max_original_queries or 'full'}_d{args.max_depth}_b{args.bin_count}",
            args.dataset,
            args.gbdt_iterations,
            args.max_depth,
            args.bin_count,
            args.max_original_queries,
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standard Terrae/CertXGB benchmark and write JSON/Markdown reports.")
    parser.add_argument("--preset", choices=["single", "small"], default="single")
    parser.add_argument("--routes", choices=["terrae", "terrae+zkboost"], default="terrae+zkboost")
    parser.add_argument("--case-id")
    parser.add_argument("--dataset", default="iris-binary")
    parser.add_argument("--gbdt-iterations", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--bin-count", type=int, default=2)
    parser.add_argument("--max-original-queries", type=int)
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build-release")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "benchmarks" / "standard")
    parser.add_argument("--srs-deg1-plus-1", type=int, default=131072)
    parser.add_argument("--srs-deg2-plus-1", type=int, default=131072)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.max_depth <= 0:
        raise ValueError("--max-depth must be at least one because the root has depth one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        _prepare_build(
            args.build_dir,
            include_zkboost=(args.routes == "terrae+zkboost"),
            timeout_seconds=args.timeout_seconds,
        )

    cases = _cases_from_args(args)
    results = [
        _run_case(
            case,
            build_dir=args.build_dir,
            output_dir=args.output_dir,
            srs_deg1_plus_1=args.srs_deg1_plus_1,
            srs_deg2_plus_1=args.srs_deg2_plus_1,
            include_zkboost=(args.routes == "terrae+zkboost"),
            timeout_seconds=args.timeout_seconds,
        )
        for case in cases
    ]
    report = {
        "config": {
            "routes": args.routes,
            "timeout_seconds": args.timeout_seconds,
            "cost_statistics_schema_version": 3,
        },
        "environment": _environment(args.build_dir, args.srs_deg1_plus_1, args.srs_deg2_plus_1),
        "cases": results,
    }
    json_path = args.output_dir / "benchmark_report.json"
    md_path = args.output_dir / "benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(md_path, report)
    print("Standard benchmark complete", flush=True)
    print(f"  json: {json_path}", flush=True)
    print(f"  markdown: {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
