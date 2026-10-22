#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_focus_extract import (
    SOUNDNESS_SUMMARY_FIELDNAMES,
    _summary_statistics,
    write_soundness_summary_csv,
)


class SoundnessSummaryTest(unittest.TestCase):
    def test_trainer_active_nodes_override_native_allocated_slots(self) -> None:
        payload = {
            "cost_statistics": {"workload_active_node_count": 32},
            "metrics": {"node_count": 32},
        }
        case_result = {
            "dataset_adaptation": {},
            "benchmark_payload": {
                "train_export": {
                    "fields": {
                        "active_node_count": "21",
                        "allocated_node_slot_count": "32",
                    }
                }
            },
        }
        result = _summary_statistics(payload, case_result)
        self.assertEqual(result["node_count"], 21)
        self.assertEqual(result["active_node_count"], 21)
        self.assertEqual(result["allocated_node_slot_count"], 32)

    def test_maps_session_metrics_to_appendix_k_columns(self) -> None:
        payload = {
            "cost_statistics": {
                "soundness_session_commitment_count": 25,
                "soundness_session_zero_claims": 32,
                "soundness_session_sum_batched_claims": 0,
                "soundness_session_histogram_checks": 0,
                "soundness_session_logical_lookup_instances": 4,
                "soundness_session_materialized_lookup_instances": 2,
                "soundness_session_evaluated_identity_count": 1,
                "soundness_session_domain_bound": 128,
                "soundness_session_domain_log2": 7,
                "soundness_session_c_pi": 43,
                "soundness_session_round_by_round_numerator": 5504,
            },
            "metrics": {},
        }
        result = _summary_statistics(payload, {"dataset_adaptation": {}})
        self.assertEqual(result["N_comm"], 25)
        self.assertEqual(result["m_zero"], 32)
        self.assertEqual(result["N_lookup_logical"], 4)
        self.assertEqual(result["N_lookup_materialized"], 2)
        self.assertEqual(result["c_pi"], 43)
        self.assertEqual(result["c_pi_times_2_to_L"], 5504)

    def test_dedicated_csv_contains_only_terrae_rows(self) -> None:
        rows = [
            {
                "case_id": "case",
                "route": "Terrae",
                "proof_stage": "classification",
                "N_comm": 25,
                "m_zero": 32,
                "c_pi": 43,
            },
            {
                "case_id": "case",
                "route": "CertXGB shared",
                "proof_stage": "inference",
                "N_comm": 99,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "soundness.csv"
            write_soundness_summary_csv(path, rows)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                output = list(reader)
                self.assertEqual(reader.fieldnames, SOUNDNESS_SUMMARY_FIELDNAMES)

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["case_id"], "case")
        self.assertEqual(output[0]["N_comm"], "25")
        self.assertEqual(output[0]["c_pi"], "43")


if __name__ == "__main__":
    unittest.main()
