#!/usr/bin/env python3

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_benchmark_parameter_plan as plan_runner

from benchmark_focus_extract import complete_case_summary_rows
from run_benchmark_parameter_plan import (
    PlanCase,
    _blocked_stage_selectors,
    _failure_peak_memory_bytes,
    _run_case,
    _write_aggregate,
)


class BenchmarkFailureSummaryTest(unittest.TestCase):
    def test_stage_failure_blocks_only_the_same_later_stage(self) -> None:
        case = PlanCase(1, "later", "plan", "dry-bean", 512, 4, 2, 8, 512, "zkboost")
        blocked = {
            ("dry-bean", "zkboost", "zkboost-full-training"): (2, "T2"),
        }
        self.assertEqual(
            _blocked_stage_selectors(blocked, case),
            ["zkboost-full-training"],
        )

    def test_failed_stage_keeps_peak_and_leaves_unavailable_metrics_blank(self) -> None:
        inference = {
            "case_id": "case",
            "route": "CertXGB shared",
            "proof_stage": "inference",
            "proof_scope": "routing+score-aggregation+argmax",
            "prover_time_s": 12.5,
            "verifier_time_s": 0.1,
            "proof_size_bytes": 128,
            "peak_memory_bytes": 2048,
        }
        rows = complete_case_summary_rows(
            [inference],
            metadata={"case_id": "case", "dataset": "dataset", "N_requested": 512, "learning_rate": "4/1"},
            routes="zkboost",
            case_status="failed",
            failure_peak_memory_bytes=8192,
        )
        by_stage = {row["proof_stage"]: row for row in rows}
        training = by_stage["training"]
        self.assertEqual(training["stage_status"], "failed")
        self.assertEqual(training["prover_time_s"], "")
        self.assertEqual(training["verifier_time_s"], "")
        self.assertEqual(training["proof_size_bytes"], "")
        self.assertEqual(training["peak_memory_bytes"], 8192)
        self.assertEqual(training["memory_source"], "case-memory-guard-peak")
        self.assertEqual(by_stage["inference"]["stage_status"], "completed")
        self.assertEqual(by_stage["inference"]["case_status"], "failed")
        self.assertEqual(by_stage["inference"]["learning_rate"], "4/1")
        self.assertNotIn("failure_reason", training)

    def test_plan_rows_exist_even_when_case_was_never_run(self) -> None:
        case = PlanCase(
            order=1,
            case_id="planned",
            phase="plan",
            dataset="dry-bean",
            N=512,
            T=1,
            d=2,
            bins=8,
            b=512,
            routes="terrae+zkboost",
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _write_aggregate(output_dir, [case], {"cases": {}})
            with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with (output_dir / "soundness_summary.csv").open(newline="", encoding="utf-8") as handle:
                soundness_rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 5)
        self.assertEqual({row["case_status"] for row in rows}, {"not_run"})
        self.assertEqual({row["stage_status"] for row in rows}, {"not_run"})
        self.assertTrue(all(row["prover_time_s"] == "" for row in rows))
        self.assertTrue(all(row["proof_size_bytes"] == "" for row in rows))
        self.assertNotIn("failure_reason", rows[0])
        self.assertEqual(len(soundness_rows), 3)
        self.assertTrue(all(row["N_comm"] == "" for row in soundness_rows))

    def test_later_t_runs_inference_when_only_training_is_blocked(self) -> None:
        cases = [
            {
                "order": 1, "case_id": "T2", "phase": "zkboost", "dataset": "dry-bean",
                "N": 512, "T": 2, "d": 2, "bins": 8, "b": 512, "routes": "zkboost",
            },
            {
                "order": 2, "case_id": "T4", "phase": "zkboost", "dataset": "dry-bean",
                "N": 512, "T": 4, "d": 2, "bins": 8, "b": 512, "routes": "zkboost",
            },
        ]
        calls = []

        def run_case(case, args, output_dir, skipped_stages=()):
            calls.append((case.case_id, tuple(skipped_stages)))
            return {
                "status": "unsupported",
                "message": "training unavailable",
                "worker": "test",
                "case_log": str(output_dir / "case.log"),
                "failed_stages": ["zkboost-full-training"] if case.case_id == "T2" else [],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.json"
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            argv = [
                "runner", "--cases-json", str(cases_path), "--output-dir", str(root / "out"),
                "--skip-build", "--no-timeout", "--jobs", "1", "--stop-subsequent-t-after-failure",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(plan_runner, "_run_case", side_effect=run_case):
                self.assertEqual(plan_runner.main(), 0)

        self.assertEqual(calls, [("T2", ()), ("T4", ("zkboost-full-training",))])

    def test_unsupported_stage_keeps_memory_guard_peak(self) -> None:
        rows = complete_case_summary_rows(
            [],
            metadata={"case_id": "case", "dataset": "dataset"},
            routes="terrae",
            case_status="unsupported",
            failure_peak_memory_bytes=8192,
        )
        by_stage = {row["proof_stage"]: row for row in rows}
        failed_stage = by_stage["classification"]
        self.assertEqual(failed_stage["stage_status"], "unsupported")
        self.assertEqual(failed_stage["peak_memory_bytes"], 8192)
        self.assertEqual(failed_stage["memory_source"], "case-memory-guard-peak")
        self.assertEqual(failed_stage["prover_time_s"], "")

    def test_memory_guard_is_an_unsupported_case(self) -> None:
        case = PlanCase(
            order=1,
            case_id="guarded",
            phase="plan",
            dataset="dry-bean",
            N=512,
            T=1,
            d=2,
            bins=8,
            b=512,
            routes="terrae",
        )
        args = SimpleNamespace(
            gradient_mode=None,
            timeout_seconds=None,
            max_srs_deg1_plus_1=None,
            max_srs_deg2_plus_1=None,
            assume_srs_deg1_plus_1=None,
            assume_srs_deg2_plus_1=None,
            srs_cache_dir=None,
            initial_srs_deg1_plus_1=1024,
            initial_srs_deg2_plus_1=1024,
            record_timed_out_stage=[],
            reuse_existing_stages=False,
            rerun_stage=[],
            skip_build=True,
            jobs=1,
            build_dir=Path("build"),
        )
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory)
            (case_dir / "case_memory_guard.json").write_text(
                json.dumps({"status": "rss_limit_detected", "peak_rss_bytes": 8192}),
                encoding="utf-8",
            )
            with (
                mock.patch.dict(os.environ, {"BENCHMARK_RSS_LIMIT_BYTES": "1"}),
                mock.patch(
                    "run_benchmark_parameter_plan.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, ["benchmark"]),
                ),
            ):
                result = _run_case(case, args, case_dir)
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("peak_rss_bytes=8192", result["message"])

    def test_failure_peak_uses_maximum_across_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "case_memory_guard.json").write_text(
                json.dumps({"peak_rss_bytes": 200}),
                encoding="utf-8",
            )
            log_path = output_dir / "case_runner.log"
            log_path.write_text(
                "first peak_rss_bytes=300\nretry peak_rss_bytes=100\n",
                encoding="utf-8",
            )
            entry = {
                "message": "failed; peak_rss_bytes=250",
                "case_log": str(log_path),
            }
            self.assertEqual(_failure_peak_memory_bytes(entry, output_dir), 300)


if __name__ == "__main__":
    unittest.main()
