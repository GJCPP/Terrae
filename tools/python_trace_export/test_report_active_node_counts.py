#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from report_active_node_counts import _find_trace, _parse_trace_counts


def _write_trace(path: Path, tree_ids: list[int]) -> None:
    length = len(tree_ids)
    path.write_text(
        "\n".join(
            [
                "scalar training.class_count 4",
                "scalar training.tree_count 4",
                f"scalar training.node_count {length}",
                f"i64vec training.split_node_mask {length} "
                + " ".join(["1", "0"] * (length // 2)),
                f"i64vec training.node_leaf_mask {length} "
                + " ".join(["0", "1"] * (length // 2)),
                f"i64vec training.node_tree_ids {length} "
                + " ".join(str(value) for value in tree_ids),
            ]
        )
        + "\n",
        encoding="ascii",
    )


class ActiveNodeCountTest(unittest.TestCase):
    def test_zero_and_one_based_tree_ids_give_the_same_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_trace = root / "old.trace"
            new_trace = root / "new.trace"
            _write_trace(old_trace, [0, 0, 1, 1, 2, 2, 3, 3])
            _write_trace(new_trace, [1, 1, 2, 2, 3, 3, 4, 4])

            old = _parse_trace_counts(old_trace, real_class_count=3)
            new = _parse_trace_counts(new_trace, real_class_count=3)

        self.assertEqual(old["tree_id_base"], 0)
        self.assertEqual(new["tree_id_base"], 1)
        self.assertEqual(old["active_trace_node_count"], 8)
        self.assertEqual(new["active_trace_node_count"], 8)
        self.assertEqual(old["active_node_count"], 6)
        self.assertEqual(new["active_node_count"], 6)

    def test_ambiguous_search_does_not_silently_choose_a_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("first", "second"):
                trace = root / name / "case" / "run" / "training.trace"
                trace.parent.mkdir(parents=True)
                trace.touch()
            with self.assertRaisesRegex(ValueError, "multiple training traces"):
                _find_trace([root], "case")


if __name__ == "__main__":
    unittest.main()
