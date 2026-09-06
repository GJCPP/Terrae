import json
import tempfile
import unittest
from pathlib import Path

from benchmark_focus_extract import SOUNDNESS_METRIC_FIELDS
from report_soundness_from_traces import (
    DERIVATION_KIND,
    _load_plan,
    _rows_from_native,
)


class ReportSoundnessFromTracesTest(unittest.TestCase):
    def setUp(self):
        self.scalars = {
            "training.query_count": 512,
            "training.iteration_count": 1,
            "training.class_count": 8,
            "training.max_depth": 2,
            "training.bin_count": 8,
            "inference.query_count": 4096,
        }
        self.metrics = {}
        for native_stage in ("well_formedness", "training", "classification"):
            for index, canonical in enumerate(SOUNDNESS_METRIC_FIELDS.values(), start=1):
                suffix = canonical.removeprefix("soundness_session_")
                self.metrics[f"soundness_exact_{native_stage}_{suffix}"] = index

    def test_builds_three_exact_rows(self):
        rows = _rows_from_native(
            self.metrics,
            self.scalars,
            case_id="dry-bean-N512-T1",
            dataset="dry-bean",
            source_revision="deadbeef",
        )
        self.assertEqual(
            [row["proof_stage"] for row in rows],
            ["well-formedness", "training", "classification"],
        )
        self.assertEqual(rows[0]["b_padded"], 512)
        self.assertTrue(all(row["derivation_kind"] == DERIVATION_KIND for row in rows))
        self.assertTrue(all(row["stage_status"] == "exact-count-only" for row in rows))
        self.assertTrue(all(row["srs_loaded"] == "no" for row in rows))
        self.assertTrue(all(row["proof_generated"] == "no" for row in rows))

    def test_requires_every_native_metric(self):
        del self.metrics["soundness_exact_training_commitment_count"]
        with self.assertRaisesRegex(RuntimeError, "omitted metric"):
            _rows_from_native(
                self.metrics,
                self.scalars,
                case_id="case",
                dataset="dataset",
                source_revision="deadbeef",
            )

    def test_plan_loader_preserves_layout_and_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "order": 2,
                            "case_id": "b",
                            "dataset": "dry-bean",
                            "N": 1024,
                            "T": 4,
                            "d": 3,
                            "bins": 8,
                            "b": 1024,
                            "routes": "terrae",
                            "loss_mode": "first-order-rss",
                        },
                        {
                            "order": 1,
                            "case_id": "a",
                            "dataset": "optdigits-7class-train",
                            "N": 512,
                            "T": 1,
                            "d": 2,
                            "routes": "zkboost",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            cases = _load_plan(path)
        self.assertEqual([case.case_id for case in cases], ["a", "b"])
        self.assertEqual(cases[1].T, 4)
        self.assertEqual(cases[1].routes, "terrae")


if __name__ == "__main__":
    unittest.main()
