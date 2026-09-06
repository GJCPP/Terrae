#!/usr/bin/env python3
"""Extract active model-node counts from retained benchmark training traces.

This is intentionally trace-only: it never invokes a benchmark or a prover.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from train_fixed_point_dataset import _load_dataset


FIELDNAMES = ["case_id", "active_node_count"]


def _parse_trace_counts(path: Path, real_class_count: int) -> dict[str, int]:
    scalars: dict[str, int] = {}
    vectors: dict[str, list[int]] = {}
    needed_vectors = {
        "training.split_node_mask",
        "training.node_leaf_mask",
        "training.node_tree_ids",
    }
    with path.open(encoding="ascii") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0] == "scalar":
                scalars[parts[1]] = int(parts[2])
            elif parts[0] == "i64vec" and parts[1] in needed_vectors:
                vectors[parts[1]] = [int(value) for value in parts[3:]]
    trace_class_count = scalars["training.class_count"]
    allocated = scalars["training.node_count"]
    split = vectors["training.split_node_mask"]
    leaf = vectors["training.node_leaf_mask"]
    tree_ids = vectors["training.node_tree_ids"]
    if not (len(split) == len(leaf) == len(tree_ids) == allocated):
        raise ValueError(f"invalid node vectors in {path}")
    active = [
        tree_id
        for split_mask, leaf_mask, tree_id in zip(split, leaf, tree_ids)
        if split_mask or leaf_mask
    ]
    active_node_count = sum((tree_id - 1) % trace_class_count < real_class_count for tree_id in active)
    return {
        "trace_class_count": trace_class_count,
        "allocated_node_slot_count": allocated,
        "active_trace_node_count": len(active),
        "active_node_count": active_node_count,
    }


def _find_trace(search_dirs: list[Path], case_id: str) -> Path | None:
    matches: list[Path] = []
    for directory in search_dirs:
        matches.extend(directory.glob(f"**/{case_id}/**/*training.trace"))
    return min(matches) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", type=Path, action="append", required=True)
    parser.add_argument("--search-dir", type=Path, action="append", default=[Path("out")])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    class_counts: dict[tuple[str, int], int] = {}
    rows: list[dict[str, object]] = []
    for cases_path in args.cases_json:
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        for case in cases:
            dataset = str(case["dataset"])
            bins = int(case["bins"])
            cache_key = (dataset, bins)
            if cache_key not in class_counts:
                class_counts[cache_key] = _load_dataset(dataset, bins).class_count
            trace = _find_trace(args.search_dir, str(case["case_id"]))
            if trace is None:
                raise FileNotFoundError(f"training trace not found for {case['case_id']}")
            counts = _parse_trace_counts(trace, class_counts[cache_key])
            rows.append({
                "case_id": case["case_id"],
                "active_node_count": counts["active_node_count"],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
