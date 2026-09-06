#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_common import REPO_ROOT


WORKFLOW_SCRIPT = REPO_ROOT / "tools" / "python_trace_export" / "benchmark_parameterized_workflow.py"


BASE_SUMMARY_FIELDNAMES = [
    "case_id",
    "route",
    "proof_stage",
    "proof_scope",
    "case_status",
    "stage_status",
    "dataset",
    "source_revision",
    "cost_statistics_schema_version",
    "N_requested",
    "N_used",
    "N_padded",
    "b_requested",
    "b_padded",
    "T",
    "d",
    "bin_count",
    "accuracy",
    "accuracy_split",
    "accuracy_samples",
    "learning_rate",
    "active_node_count",
    "allocated_node_slot_count",
    "proof_size_bytes",
    "prover_time_s",
    "verifier_time_s",
    "peak_memory_bytes",
    "memory_source",
]

COMMITTED_DATA_AND_PCS_FIELDNAMES = [
    "committed_polynomial_coefficient_count",
    "polynomial_commitment_operation_count",
    "polynomial_point_claim_count",
    "pairing_count",
]

# The paper reports CertXGB's active PLONK gates as its sole native metric.
ZKBOOST_PAPER_FIELDNAMES = [
    "zkboost_plonk_active_gate_row_count",
]

STATISTICS_FIELDNAMES = (
    COMMITTED_DATA_AND_PCS_FIELDNAMES
    + ZKBOOST_PAPER_FIELDNAMES
)
SUMMARY_FIELDNAMES = BASE_SUMMARY_FIELDNAMES + COMMITTED_DATA_AND_PCS_FIELDNAMES
TERRAE_SUMMARY_FIELDNAMES = BASE_SUMMARY_FIELDNAMES
ZKBOOST_SUMMARY_FIELDNAMES = BASE_SUMMARY_FIELDNAMES + ZKBOOST_PAPER_FIELDNAMES

SOUNDNESS_METRIC_FIELDS = {
    "N_comm": "soundness_session_commitment_count",
    "m_zero": "soundness_session_zero_claims",
    "m_sum": "soundness_session_sum_batched_claims",
    "sum_direct_openings": "soundness_session_sum_direct_opening_claims",
    "H": "soundness_session_histogram_checks",
    "N_lookup_logical": "soundness_session_logical_lookup_instances",
    "N_lookup_materialized": "soundness_session_materialized_lookup_instances",
    "E": "soundness_session_evaluated_identity_count",
    "domain_bound_2_to_L": "soundness_session_domain_bound",
    "L": "soundness_session_domain_log2",
    "constraint_degree": "soundness_session_constraint_degree",
    "zero_batch_coefficient": "soundness_session_zero_batch_coefficient",
    "sum_batch_coefficient": "soundness_session_sum_batch_coefficient",
    "histogram_count_coefficient": "soundness_session_histogram_checks",
    "evaluation_coefficient": "soundness_session_evaluation_coefficient",
    "lookup_compression_coefficient": "soundness_session_lookup_compression_coefficient",
    "c_pi": "soundness_session_c_pi",
    "c_pi_times_2_to_L": "soundness_session_round_by_round_numerator",
    "quotient_degree_bound_exclusive": "soundness_session_quotient_degree_bound_exclusive",
    "tested_identity_degree_bound_exclusive": "soundness_session_tested_identity_degree_bound_exclusive",
    "histogram_exact_degree_sum": "soundness_session_histogram_coefficient",
    "zero_max_domain_size": "soundness_session_zero_max_domain_size",
    "sum_batched_max_domain_size": "soundness_session_sum_batched_max_domain_size",
    "histogram_max_address_space": "soundness_session_histogram_max_address_space",
    "logical_lookup_max_query_rows": "soundness_session_logical_lookup_max_query_rows",
    "logical_lookup_max_table_rows": "soundness_session_logical_lookup_max_table_rows",
    "materialized_lookup_max_query_rows": "soundness_session_materialized_lookup_max_query_rows",
    "materialized_lookup_max_table_rows": "soundness_session_materialized_lookup_max_table_rows",
    "deterministic_root_uniqueness_checks": "soundness_session_deterministic_root_uniqueness_checks",
}

SOUNDNESS_SUMMARY_FIELDNAMES = [
    "case_id",
    "proof_stage",
    "proof_scope",
    "case_status",
    "stage_status",
    "dataset",
    "source_revision",
    "cost_statistics_schema_version",
    "N_used",
    "N_padded",
    "b_padded",
    "T",
    "d",
    "bin_count",
    *SOUNDNESS_METRIC_FIELDS.keys(),
]

EXPECTED_STAGE_SPECS = [
    {
        "route": "Terrae",
        "proof_stage": "well-formedness",
        "proof_scope": "well-formedness",
        "route_selector": "terrae",
        "execution_order": 1,
    },
    {
        "route": "Terrae",
        "proof_stage": "training",
        "proof_scope": "training+well-formedness",
        "route_selector": "terrae",
        "execution_order": 2,
    },
    {
        "route": "Terrae",
        "proof_stage": "classification",
        "proof_scope": "routing+score-aggregation+argmax",
        "route_selector": "terrae",
        "execution_order": 0,
    },
    {
        "route": "CertXGB shared",
        "proof_stage": "training",
        "proof_scope": "training+forest-well-formedness",
        "route_selector": "zkboost",
        "execution_order": 4,
    },
    {
        "route": "CertXGB shared",
        "proof_stage": "inference",
        "proof_scope": "routing+score-aggregation+argmax",
        "route_selector": "zkboost",
        "execution_order": 3,
    },
]


def _is_terminal_status(status: str) -> bool:
    return status in {"completed", "timed_out", "unsupported"}


def _stage_peak(payload: dict[str, Any]) -> tuple[int | None, str]:
    stage_peak = payload.get("metrics", {}).get("proof_stage_peak_rss_bytes")
    if isinstance(stage_peak, int):
        return stage_peak, "proof-stage-local-rss"
    process_peak = payload.get("memory", {}).get("peak_rss_bytes")
    if isinstance(process_peak, int):
        return process_peak, "stage-process-peak"
    return None, "unavailable"


def _metric(payload: dict[str, Any], name: str, fallback: dict[str, Any] | None = None) -> Any:
    value = payload.get("metrics", {}).get(name)
    if value is not None or fallback is None:
        return value
    return fallback.get("metrics", {}).get(name)


def _timing(payload: dict[str, Any], name: str, fallback: dict[str, Any] | None = None) -> Any:
    value = payload.get("timings", {}).get(name)
    if value is not None or fallback is None:
        return value
    return fallback.get("timings", {}).get(name)


def _accuracy(case_result: dict[str, Any]) -> str:
    fields = case_result["benchmark_payload"].get("train_export", {}).get("fields", {})
    return str(fields.get("accuracy") or "")


def _accuracy_split(case_result: dict[str, Any]) -> str:
    fields = case_result["benchmark_payload"].get("train_export", {}).get("fields", {})
    return str(fields.get("accuracy_split") or "")


def _accuracy_samples(case_result: dict[str, Any]) -> str:
    fields = case_result["benchmark_payload"].get("train_export", {}).get("fields", {})
    return str(fields.get("accuracy_samples") or "")


def _learning_rate(case_result: dict[str, Any]) -> str:
    fields = case_result["benchmark_payload"].get("train_export", {}).get("fields", {})
    return str(fields.get("learning_rate") or "")


def _summary_statistics(payload: dict[str, Any], case_result: dict[str, Any]) -> dict[str, Any]:
    statistics = payload.get("cost_statistics", {})
    metrics = payload.get("metrics", {})
    output = {
        name: statistics[name]
        for name in STATISTICS_FIELDNAMES
        if isinstance(statistics.get(name), (int, float))
    }
    for output_name, metric_name in SOUNDNESS_METRIC_FIELDS.items():
        value = statistics.get(metric_name)
        if isinstance(value, (int, float)):
            output[output_name] = value
    adaptation = case_result.get("dataset_adaptation", {})
    output["feature_count"] = adaptation.get("feature_count")
    output["real_class_count"] = adaptation.get("real_class_count")
    workload_aliases = {
        "padded_class_count": ("class_count",),
        "tree_count": ("tree_count",),
        "node_count": ("active_node_count", "node_count"),
        "active_node_count": ("active_node_count", "node_count"),
        "allocated_node_slot_count": ("allocated_node_slot_count",),
        "internal_node_count": ("internal_node_count", "split_node_count"),
        "leaf_node_count": ("leaf_node_count", "semantic_leaf_node_count"),
        "candidate_count": ("candidate_count",),
        "checked_path_count": ("checked_path_count", "checked_paths", "query_count"),
    }
    for output_name, aliases in workload_aliases.items():
        value = statistics.get(f"workload_{output_name}")
        if not isinstance(value, int):
            for alias in aliases:
                candidate = metrics.get(alias)
                if isinstance(candidate, int):
                    value = candidate
                    break
        if isinstance(value, int):
            output[output_name] = value
    return output


def _extract_terrae_rows(case_result: dict[str, Any]) -> list[dict[str, Any]]:
    case = case_result["case"]
    payload = case_result["benchmark_payload"]
    terrae = payload["terrae"]
    full = terrae.get("full", {})
    well_formedness_memory = terrae.get("well_formedness_memory")
    training_memory = terrae.get("training_memory")
    classification_memory = terrae.get("classification_memory")

    common = {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "N_requested": case_result["dataset_adaptation"]["requested_training_samples"],
        "N_used": case_result["dataset_adaptation"]["actual_training_original_samples"],
        "N_padded": case_result["dataset_adaptation"]["padded_training_samples"],
        "b_requested": case_result["dataset_adaptation"]["requested_inference_samples"],
        "b_padded": case_result["dataset_adaptation"]["padded_inference_samples"],
        "T": case["gbdt_iterations"],
        "d": case["max_depth"],
        "bin_count": case["bin_count"],
        "route": "Terrae",
        "accuracy": _accuracy(case_result),
        "accuracy_split": _accuracy_split(case_result),
        "accuracy_samples": _accuracy_samples(case_result),
        "learning_rate": _learning_rate(case_result),
    }
    rows: list[dict[str, Any]] = []
    if well_formedness_memory is not None:
        well_peak, well_peak_source = _stage_peak(well_formedness_memory)
        rows.append(
            {
                **common,
                **_summary_statistics(well_formedness_memory, case_result),
                "proof_stage": "well-formedness",
                "proof_scope": "well-formedness",
                "proof_size_bytes": _metric(well_formedness_memory, "well_formedness_proof_bytes", full),
                "prover_time_s": _timing(well_formedness_memory, "prove_well_formedness", full),
                "verifier_time_s": _timing(well_formedness_memory, "verify_well_formedness_online", full),
                "peak_memory_bytes": well_peak,
                "memory_source": well_peak_source,
            }
        )
    if training_memory is not None:
        training_peak, training_peak_source = _stage_peak(training_memory)
        rows.append(
            {
                **common,
                **_summary_statistics(training_memory, case_result),
                "proof_stage": "training",
                "proof_scope": "training+well-formedness",
                "proof_size_bytes": (
                    _metric(training_memory, "training_with_well_formedness_proof_bytes")
                    or _metric(training_memory, "training_proof_bytes", full)
                ),
                "prover_time_s": (
                    _timing(training_memory, "prove_training_with_well_formedness")
                    or _timing(training_memory, "prove_training", full)
                ),
                "verifier_time_s": (
                    _timing(training_memory, "verify_training_with_well_formedness_online")
                    or _timing(training_memory, "verify_training_online", full)
                ),
                "peak_memory_bytes": training_peak,
                "memory_source": training_peak_source,
            }
        )
    if classification_memory is not None:
        classification_peak, classification_peak_source = _stage_peak(classification_memory)
        rows.append(
            {
                **common,
                **_summary_statistics(classification_memory, case_result),
                "proof_stage": "classification",
                "proof_scope": "routing+score-aggregation+argmax",
                "proof_size_bytes": _metric(classification_memory, "classification_proof_bytes", full),
                "prover_time_s": _timing(classification_memory, "prove_classification", full),
                "verifier_time_s": _timing(classification_memory, "verify_classification_online", full),
                "peak_memory_bytes": classification_peak,
                "memory_source": classification_peak_source,
            }
        )
    return rows


def _extract_zkboost_rows(case_result: dict[str, Any]) -> list[dict[str, Any]]:
    payload = case_result["benchmark_payload"]
    zkboost = payload.get("zkboost")
    if zkboost is None:
        return []

    case = case_result["case"]
    common = {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "N_requested": case_result["dataset_adaptation"]["requested_training_samples"],
        "N_used": case_result["dataset_adaptation"]["actual_training_original_samples"],
        "N_padded": case_result["dataset_adaptation"]["padded_training_samples"],
        "b_requested": case_result["dataset_adaptation"]["requested_inference_samples"],
        "b_padded": case_result["dataset_adaptation"]["padded_inference_samples"],
        "T": case["gbdt_iterations"],
        "d": case["max_depth"],
        "bin_count": case["bin_count"],
        "route": "CertXGB shared",
        "accuracy": _accuracy(case_result),
        "accuracy_split": _accuracy_split(case_result),
        "accuracy_samples": _accuracy_samples(case_result),
        "learning_rate": _learning_rate(case_result),
    }
    rows: list[dict[str, Any]] = []
    training = zkboost.get("training")
    if training is not None:
        rows.append(
            {
                **common,
                **_summary_statistics(training, case_result),
                "proof_stage": "training",
                "proof_scope": "training+forest-well-formedness",
                "proof_size_bytes": training["metrics"].get("proof_bytes"),
                "prover_time_s": training["timings"].get("prove"),
                "verifier_time_s": training["timings"].get("verify"),
                "peak_memory_bytes": _stage_peak(training)[0],
                "memory_source": _stage_peak(training)[1],
            }
        )
    inference = zkboost.get("inference")
    if inference is not None:
        rows.append(
            {
                **common,
                **_summary_statistics(inference, case_result),
                "proof_stage": "inference",
                "proof_scope": "routing+score-aggregation+argmax",
                "proof_size_bytes": inference["metrics"].get("proof_bytes"),
                "prover_time_s": inference["timings"].get("prove"),
                "verifier_time_s": inference["timings"].get("verify"),
                "peak_memory_bytes": _stage_peak(inference)[0],
                "memory_source": _stage_peak(inference)[1],
            }
        )
    return rows


def _selected_stage_specs(routes: str) -> list[dict[str, Any]]:
    selectors = {
        "terrae": {"terrae"},
        "zkboost": {"zkboost"},
        "terrae+zkboost": {"terrae", "zkboost"},
    }.get(routes, set())
    return [spec for spec in EXPECTED_STAGE_SPECS if spec["route_selector"] in selectors]


def complete_case_summary_rows(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    routes: str,
    case_status: str,
    failure_peak_memory_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Add explicit rows for proof stages that did not produce a result."""
    specs = _selected_stage_specs(routes)
    rows_by_key = {
        (str(row.get("route", "")), str(row.get("proof_stage", ""))): row
        for row in rows
    }

    common_fields = [
        "case_id",
        "dataset",
        "source_revision",
        "cost_statistics_schema_version",
        "N_requested",
        "N_used",
        "N_padded",
        "b_requested",
        "b_padded",
        "T",
        "d",
        "bin_count",
                    "accuracy",
        "accuracy_split",
        "accuracy_samples",
        "learning_rate",
    ]
    for row in rows:
        row["case_status"] = case_status
        row.setdefault("stage_status", "completed")
        for field in common_fields:
            if row.get(field, "") == "" and metadata.get(field, "") != "":
                row[field] = metadata[field]

    for spec in specs:
        key = (spec["route"], spec["proof_stage"])
        if key in rows_by_key:
            continue
        row = {field: metadata.get(field, "") for field in common_fields}
        row.update({
            "route": spec["route"],
            "proof_stage": spec["proof_stage"],
            "proof_scope": spec["proof_scope"],
            "case_status": case_status,
            "stage_status": "not_run",
            "proof_size_bytes": "",
            "prover_time_s": "",
            "verifier_time_s": "",
            "peak_memory_bytes": "",
            "memory_source": "unavailable",
        })
        rows.append(row)
        rows_by_key[key] = row

    incomplete_specs = sorted(
        (
            spec for spec in specs
            if rows_by_key[(spec["route"], spec["proof_stage"])].get("stage_status")
            != "completed"
        ),
        key=lambda spec: int(spec["execution_order"]),
    )
    if incomplete_specs and case_status != "skipped_after_failure":
        failed_spec = incomplete_specs[0]
        failed_key = (failed_spec["route"], failed_spec["proof_stage"])
        failed_row = rows_by_key[failed_key]
        failed_row["stage_status"] = (
            case_status if case_status != "completed" else "missing"
        )
        if failure_peak_memory_bytes is not None:
            failed_row["peak_memory_bytes"] = failure_peak_memory_bytes
            failed_row["memory_source"] = "case-memory-guard-peak"
        for spec in incomplete_specs[1:]:
            row = rows_by_key[(spec["route"], spec["proof_stage"])]
            row["stage_status"] = "not_run"

    summary_order = {
        (spec["route"], spec["proof_stage"]): index
        for index, spec in enumerate(specs)
    }
    return sorted(
        rows,
        key=lambda row: summary_order.get(
            (str(row.get("route", "")), str(row.get("proof_stage", ""))),
            len(summary_order),
        ),
    )


def _first_checkpoint_metric(stage_checkpoint: dict[str, Any], *names: str) -> Any:
    for route in ("terrae", "zkboost"):
        route_payload = stage_checkpoint.get(route, {})
        if not isinstance(route_payload, dict):
            continue
        for payload in route_payload.values():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics", {})
            for name in names:
                if metrics.get(name) is not None:
                    return metrics[name]
    return ""


def collect_checkpoint_summary_rows(
    stage_checkpoint: dict[str, Any],
    *,
    metadata: dict[str, Any],
    routes: str,
    case_status: str,
    failure_peak_memory_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Recover completed proof stages from an interrupted workflow checkpoint."""
    checkpoint_case = stage_checkpoint.get("case", {})
    case = {
        "case_id": checkpoint_case.get("case_id", metadata.get("case_id", "")),
        "dataset": checkpoint_case.get("dataset", metadata.get("dataset", "")),
        "gbdt_iterations": checkpoint_case.get("gbdt_iterations", metadata.get("T", 0)),
        "max_depth": checkpoint_case.get("max_depth", metadata.get("d", 0)),
        "bin_count": checkpoint_case.get("bin_count", metadata.get("bin_count", 0)),
    }
    adaptation = dict(stage_checkpoint.get("dataset_adaptation", {}))
    adaptation.setdefault("requested_training_samples", metadata.get("N_requested", ""))
    adaptation.setdefault("actual_training_original_samples", metadata.get("N_used", ""))
    adaptation.setdefault(
        "padded_training_samples",
        metadata.get("N_padded") or _first_checkpoint_metric(stage_checkpoint, "record_count_padded"),
    )
    adaptation.setdefault("requested_inference_samples", metadata.get("b_requested", ""))
    adaptation.setdefault(
        "padded_inference_samples",
        metadata.get("b_padded")
        or _first_checkpoint_metric(stage_checkpoint, "inference_sample_count", "sample_count"),
    )
    adaptation.setdefault(
            metadata.get("feature_count") or _first_checkpoint_metric(stage_checkpoint, "feature_count"),
    )
    adaptation.setdefault("real_class_count", metadata.get("real_class_count", ""))
    benchmark_payload = {
        "train_export": stage_checkpoint.get("train_export", {}),
        "terrae": stage_checkpoint.get("terrae", {}),
        "zkboost": stage_checkpoint.get("zkboost", {}),
    }
    case_result = {
        "case": case,
        "dataset_adaptation": adaptation,
        "benchmark_payload": benchmark_payload,
    }
    rows = _extract_terrae_rows(case_result) + _extract_zkboost_rows(case_result)
    for row in rows:
        row["source_revision"] = metadata.get("source_revision", "")
        row["cost_statistics_schema_version"] = metadata.get(
            "cost_statistics_schema_version", ""
        )
    recovered_metadata = dict(metadata)
    for row in rows:
        for field in BASE_SUMMARY_FIELDNAMES:
            if recovered_metadata.get(field, "") == "" and row.get(field, "") != "":
                recovered_metadata[field] = row[field]
    return complete_case_summary_rows(
        rows,
        metadata=recovered_metadata,
        routes=routes,
        case_status=case_status,
        failure_peak_memory_bytes=failure_peak_memory_bytes,
    )


def collect_summary_rows(workflow_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_revision = workflow_report.get("environment", {}).get("source_revision", "")
    schema_version = workflow_report.get("config", {}).get("cost_statistics_schema_version", "")
    routes = str(workflow_report.get("config", {}).get("routes", "terrae+zkboost"))
    for case_result in workflow_report["cases"]:
        case = case_result["case"]
        adaptation = case_result.get("dataset_adaptation", {})
        case_status = str(case_result.get("status", "completed"))
        case_rows: list[dict[str, Any]] = []
        if "benchmark_payload" in case_result:
            case_rows = _extract_terrae_rows(case_result) + _extract_zkboost_rows(case_result)
            for row in case_rows:
                row["source_revision"] = source_revision
                row["cost_statistics_schema_version"] = schema_version
        metadata = {
            "case_id": case.get("case_id", ""),
            "dataset": case.get("dataset", ""),
            "source_revision": source_revision,
            "cost_statistics_schema_version": schema_version,
            "N_requested": adaptation.get("requested_training_samples", ""),
            "N_used": adaptation.get("actual_training_original_samples", ""),
            "N_padded": adaptation.get("padded_training_samples", ""),
            "b_requested": adaptation.get("requested_inference_samples", ""),
            "b_padded": adaptation.get("padded_inference_samples", ""),
            "T": case.get("gbdt_iterations", ""),
            "d": case.get("max_depth", ""),
            "bin_count": case.get("bin_count", ""),
            "feature_count": adaptation.get("feature_count", ""),
            "real_class_count": adaptation.get("real_class_count", ""),
        }
        for row in case_rows:
            for field in BASE_SUMMARY_FIELDNAMES:
                if metadata.get(field, "") == "" and row.get(field, "") != "":
                    metadata[field] = row[field]
        rows.extend(
            complete_case_summary_rows(
                case_rows,
                metadata=metadata,
                routes=routes,
                case_status=case_status,
            )
        )
    return rows


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "unavailable"


def _fmt_int(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return "unavailable"


def _write_markdown(path: Path, *, workflow_report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    env = workflow_report["environment"]
    routes = workflow_report["config"]["routes"]
    lines = [
        "# Focus Benchmark Summary",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in ["source_revision", "build_dir", "machine", "cpu", "logical_cpus", "os", "srs_deg1_plus_1", "srs_deg2_plus_1"]:
        lines.append(f"| {key} | `{env[key]}` |")
    lines.extend([
        f"| route scope | `{routes}` |",
        "",
        "## Focus Table",
        "",
        "| case_id | route | proof_stage | dataset | N_used | N_padded | b_padded | T | d | bins | active nodes | allocated slots | accuracy | accuracy_split | accuracy_samples | learning_rate | proof_size_bytes | prover_time_s | verifier_time_s | peak_memory_bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | {row['route']} | {row['proof_stage']} | `{row['dataset']}` | "
            f"{row['N_used']} | {row['N_padded']} | {row['b_padded']} | {row['T']} | {row['d']} | "
            f"{row['bin_count']} | {_fmt_int(row.get('active_node_count'))} | "
            f"{_fmt_int(row.get('allocated_node_slot_count'))} | `{row.get('accuracy', '')}` | `{row.get('accuracy_split', '')}` | "
            f"{row.get('accuracy_samples', '')} | `{row.get('learning_rate', '')}` | "
            f"{_fmt_int(row['proof_size_bytes'])} | {_fmt_float(row['prover_time_s'])} | "
            f"{_fmt_float(row['verifier_time_s'])} | {_fmt_int(row['peak_memory_bytes'])} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `prover_time_s` and `verifier_time_s` are extracted from the proof-stage timings themselves and do not include SRS loading.",
        "- The workflow runner collects proof metrics and memory in the same stage pass; it does not run Terrae's deprecated full-profile pass for a second proof batch.",
        "- Terrae `peak_memory_bytes` uses the stage-local proof RSS metric when available; otherwise the extractor falls back to the external process peak for the profiled command.",
        "- The Terrae training row includes core training plus well-formedness; the separate well-formedness row is a component breakdown and must not be added again.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows, SUMMARY_FIELDNAMES)


def write_terrae_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    terrae_rows = [row for row in rows if row.get("route") == "Terrae"]
    _write_csv(path, terrae_rows, TERRAE_SUMMARY_FIELDNAMES)


def write_zkboost_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    zkboost_rows = [row for row in rows if str(row.get("route", "")).startswith("CertXGB")]
    _write_csv(path, zkboost_rows, ZKBOOST_SUMMARY_FIELDNAMES)


def write_soundness_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    terrae_rows = [row for row in rows if row.get("route") == "Terrae"]
    _write_csv(path, terrae_rows, SOUNDNESS_SUMMARY_FIELDNAMES)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the parameterized workflow benchmark runner and extract the focused "
            "proof-size / prove-time / verify-time / memory tables into Markdown and CSV."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-md-name", default="focus_benchmark_summary.md")
    parser.add_argument("--summary-csv-name", default="focus_benchmark_summary.csv")
    parser.add_argument(
        "--terrae-summary-csv-name",
        default="focus_benchmark_terrae_summary.csv",
    )
    parser.add_argument(
        "--zkboost-summary-csv-name",
        default="focus_benchmark_zkboost_summary.csv",
    )
    parser.add_argument(
        "--soundness-summary-csv-name",
        default="focus_benchmark_soundness_summary.csv",
    )
    parser.add_argument("--reuse-existing-report", type=Path)
    args, workflow_args = parser.parse_known_args()
    return args, workflow_args


def main() -> int:
    args, workflow_args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    workflow_report_path = (
        args.reuse_existing_report
        if args.reuse_existing_report is not None
        else args.output_dir / "workflow_benchmark_report.json"
    )

    workflow_returncode = 0
    if args.reuse_existing_report is None:
        forwarded = list(workflow_args)
        if "--output-dir" in forwarded:
            raise ValueError("do not pass --output-dir through workflow args; use the wrapper's --output-dir")
        command = [
            sys.executable,
            str(WORKFLOW_SCRIPT),
            "--output-dir",
            str(args.output_dir),
            *forwarded,
        ]
        try:
            subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        except subprocess.CalledProcessError as error:
            workflow_returncode = error.returncode

    if not workflow_report_path.exists():
        raise FileNotFoundError(f"workflow report not found: {workflow_report_path}")

    workflow_report = json.loads(workflow_report_path.read_text(encoding="utf-8"))
    rows = collect_summary_rows(workflow_report)

    summary_md = args.output_dir / args.summary_md_name
    summary_csv = args.output_dir / args.summary_csv_name
    terrae_summary_csv = args.output_dir / args.terrae_summary_csv_name
    zkboost_summary_csv = args.output_dir / args.zkboost_summary_csv_name
    soundness_summary_csv = args.output_dir / args.soundness_summary_csv_name
    _write_markdown(summary_md, workflow_report=workflow_report, rows=rows)
    write_summary_csv(summary_csv, rows)
    write_terrae_summary_csv(terrae_summary_csv, rows)
    write_zkboost_summary_csv(zkboost_summary_csv, rows)
    write_soundness_summary_csv(soundness_summary_csv, rows)

    print("Focus benchmark extraction complete", flush=True)
    print(f"  markdown: {summary_md}", flush=True)
    print(f"  common csv: {summary_csv}", flush=True)
    print(f"  Terrae csv: {terrae_summary_csv}", flush=True)
    print(f"  CertXGB csv: {zkboost_summary_csv}", flush=True)
    print(f"  soundness csv: {soundness_summary_csv}", flush=True)

    if workflow_returncode != 0:
        return workflow_returncode
    if any(not _is_terminal_status(case["status"]) for case in workflow_report["cases"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
