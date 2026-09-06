#!/usr/bin/env python3

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SEMANTICS_FINGERPRINT = "fixed-point-public-loss-mode-first-order-rss-lr-4-root-depth-one-v5"


class MeasuredStepError(RuntimeError):
    def __init__(self, step: str, cause: Exception, measurement: dict[str, Any] | None) -> None:
        self.step = step
        self.cause = cause
        self.measurement = measurement
        details = str(cause)
        if measurement:
            details += (
                f"; peak_rss_bytes={measurement.get('peak_rss_bytes', 'unknown')}"
                f"; peak_swap_bytes={measurement.get('peak_swap_bytes', 'unknown')}"
                f"; memory_status={measurement.get('status', 'unknown')}"
                f"; memory_guard_trigger={measurement.get('memory_guard_trigger', 'unknown')}"
                f"; measurement_json={measurement.get('measurement_json', 'unknown')}"
            )
        super().__init__(f"{step}: measured command failed: {details}")


def _is_preprocessing_timing_name(name: str) -> bool:
    lowered = name.lower()
    suppressed_names = {
        "assemble_exported_inference_plan",
        "build_committed_forest_table",
        "build_inference_public_table",
        "commit_committed_forest_public_inputs",
        "commit_forest",
        "export_leaf_table",
        "prepare_inference_witness",
    }
    return (
        lowered in suppressed_names
        or lowered.startswith("assemble_plan_")
        or ("prepare_" in lowered and "verifier_bundle" in lowered)
        or "preprocess" in lowered
        or "preprocessing" in lowered
        or "preprocessed" in lowered
    )


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> tuple[float, str]:
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout_seconds,
    )
    elapsed = time.perf_counter() - start
    return elapsed, completed.stdout


def _run_logged(step: str,
                cmd: list[str],
                *,
                cwd: Path,
                env: dict[str, str] | None = None,
                timeout_seconds: float | None = None) -> tuple[float, str]:
    print(f"[benchmark] {step}", flush=True)
    try:
        elapsed, stdout = _run(cmd, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{step}: timed out after {timeout_seconds:.1f}s") from error
    except subprocess.CalledProcessError as error:
        if timeout_seconds is not None and error.returncode == 124:
            raise RuntimeError(f"{step}: timed out after {timeout_seconds:.1f}s") from error
        raise
    print(f"[benchmark]   done in {elapsed:.3f}s", flush=True)
    return elapsed, stdout


def _safe_step_name(step: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", step).strip("_")


def _environment_flag(name: str, env: dict[str, str] | None) -> bool:
    source = env if env is not None else os.environ
    value = source.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _environment_value(name: str, env: dict[str, str] | None) -> str | None:
    source = env if env is not None else os.environ
    value = source.get(name, "").strip()
    return value or None


def _run_measured_logged(step: str,
                         cmd: list[str],
                         *,
                         cwd: Path,
                         output_dir: Path,
                         prefix: str,
                         env: dict[str, str] | None = None,
                         timeout_seconds: float | None = None) -> tuple[float, str, dict[str, Any]]:
    safe_step = _safe_step_name(step)
    measurement_json = output_dir / f"{prefix}_{safe_step}_memory.json"
    stdout_log = output_dir / f"{prefix}_{safe_step}.stdout.log"
    stderr_log = output_dir / f"{prefix}_{safe_step}.stderr.log"
    measured_cmd = [
        "python3",
        str(REPO_ROOT / "tools" / "measure_peak_memory.py"),
        "--output-json",
        str(measurement_json),
        "--stdout-log",
        str(stdout_log),
        "--stderr-log",
        str(stderr_log),
        "--suppress-preprocessing-timings",
        *(
            ["--rss-limit-bytes", value]
            if (value := _environment_value("BENCHMARK_RSS_LIMIT_BYTES", env))
            else []
        ),
        *(["--abort-on-swap"] if _environment_flag("BENCHMARK_ABORT_ON_SWAP", env) else []),
        *(["--abort-on-oom"] if _environment_flag("BENCHMARK_ABORT_ON_OOM", env) else []),
        *(
            ["--swap-limit-bytes", value]
            if (value := _environment_value("BENCHMARK_SWAP_LIMIT_BYTES", env))
            else []
        ),
        *(
            ["--swap-limit-duration-seconds", value]
            if (value := _environment_value("BENCHMARK_SWAP_LIMIT_DURATION_SECONDS", env))
            else []
        ),
        *(
            ["--min-mem-available-bytes", value]
            if (value := _environment_value("BENCHMARK_MIN_MEM_AVAILABLE_BYTES", env))
            else []
        ),
        *(
            ["--min-mem-available-duration-seconds", value]
            if (
                value := _environment_value(
                    "BENCHMARK_MIN_MEM_AVAILABLE_DURATION_SECONDS", env
                )
            )
            else []
        ),
        *(["--timeout-seconds", str(timeout_seconds)] if timeout_seconds is not None else []),
        "--",
        *cmd,
    ]
    try:
        elapsed, stdout = _run_logged(
            step,
            measured_cmd,
            cwd=cwd,
            env=env,
            timeout_seconds=None if timeout_seconds is None else timeout_seconds + 30.0,
        )
    except Exception as error:
        measurement: dict[str, Any] | None = None
        if measurement_json.exists():
            try:
                raw_measurement = json.loads(measurement_json.read_text(encoding="utf-8"))
                if isinstance(raw_measurement, dict):
                    measurement = raw_measurement
                    measurement["measurement_json"] = str(measurement_json)
            except (OSError, ValueError, TypeError):
                pass
        raise MeasuredStepError(step, error, measurement) from error
    measurement = json.loads(measurement_json.read_text(encoding="utf-8"))
    measurement["measurement_json"] = str(measurement_json)
    measurement["stdout_log"] = str(stdout_log)
    measurement["stderr_log"] = str(stderr_log)
    measurement["wrapper_elapsed_seconds"] = elapsed
    return elapsed, stdout, measurement


def _parse_metrics_and_timings(stdout: str) -> tuple[dict[str, int], dict[str, float]]:
    metrics: dict[str, int] = {}
    timings: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        kind, name, value = parts
        if kind == "metric":
            metrics[name] = int(value)
        elif kind == "timing":
            if _is_preprocessing_timing_name(name):
                continue
            timings[name] = float(value)
    return metrics, timings


def _parse_key_value_lines(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        if "preprocess" in normalized_key or "preprocessed" in normalized_key:
            continue
        out[normalized_key] = value.strip()
    return out


def _write_gbdt_proof_config(
    path: Path,
    *,
    dataset_name: str,
    iterations: int,
    max_depth: int,
    bin_count: int,
    gradient_mode: str = "first-order-rss",
    learning_rate_numerator: int = 4,
    learning_rate_denominator: int = 1,
    split_gain_threshold: int = 1,
) -> None:
    # The public learning rate is included in both training proof backends.
    # LR=4 is the retained practical setting for the first-order RSS trainer.
    config = {
        "dataset": {"name": dataset_name},
        "model": {
            "iterations": iterations,
            "max_depth": max_depth,
            "bin_count": bin_count,
            "gradient_mode": gradient_mode,
            "learning_rate_numerator": learning_rate_numerator,
            "learning_rate_denominator": learning_rate_denominator,
        },
        "regularization": {
            "l2_regularization": 2,
            "split_penalty": 0,
            "split_gain_threshold": split_gain_threshold,
        },
        "quantization": {
            "gain_fractional_bits": 2,
            "leaf_fractional_bits": 4,
            "score_fractional_bits": 4,
            "gradient_fractional_bits": 6,
            "hessian_fractional_bits": 8,
        },
    }
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
