#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "benchmark",
    "case_id",
    "route",
    "n",
    "m",
    "proof_n",
    "proof_m",
    "ram_trace_domain",
    "proof_size_bytes",
    "prover_time_s",
    "verifier_time_s",
    "accepted",
    "consumed",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert histogram benchmark TSV output into summary.csv and summary.md."
    )
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-csv-name", default="summary.csv")
    parser.add_argument("--summary-md-name", default="summary.md")
    return parser.parse_args()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if reader.fieldnames is None:
        raise ValueError(f"{path}: missing TSV header")
    required = {
        "route",
        "n",
        "m",
        "proof_n",
        "proof_m",
        "ram_trace_domain",
        "proof_bytes",
        "prover_seconds",
        "verifier_seconds",
        "accepted",
        "consumed",
    }
    missing = sorted(required.difference(reader.fieldnames))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
    return rows


def _case_id(row: dict[str, str]) -> str:
    return f"{row['route']}_n{row['n']}_m{row['m']}"


def _normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "benchmark": "histogram",
                "case_id": _case_id(row),
                "route": row["route"],
                "n": row["n"],
                "m": row["m"],
                "proof_n": row["proof_n"],
                "proof_m": row["proof_m"],
                "ram_trace_domain": row["ram_trace_domain"],
                "proof_size_bytes": row["proof_bytes"],
                "prover_time_s": row["prover_seconds"],
                "verifier_time_s": row["verifier_seconds"],
                "accepted": row["accepted"],
                "consumed": row["consumed"],
            }
        )
    return normalized


def _read_log_summary(path: Path | None) -> list[tuple[str, str]]:
    if path is None or not path.exists():
        return []
    fields: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.split("\t")
        if not parts:
            continue
        if parts[0] == "histogram_scale_config":
            for index in range(1, len(parts) - 1, 2):
                fields.append((parts[index], parts[index + 1]))
        elif parts[0] == "histogram_scale_srs_cache" and len(parts) >= 2:
            fields.append(("srs_cache", parts[1]))
    return fields


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def _write_markdown(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    input_tsv: Path,
    log_path: Path | None,
    log_fields: list[tuple[str, str]],
) -> None:
    accepted_rows = sum(1 for row in rows if row["accepted"] == "1" and row["consumed"] == "1")
    lines = [
        "# Histogram Benchmark Summary",
        "",
        "## Run Summary",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| input_tsv | `{input_tsv}` |",
        f"| log | `{log_path if log_path is not None else ''}` |",
        f"| rows | {len(rows)} |",
        f"| accepted_and_consumed_rows | {accepted_rows} |",
    ]
    for key, value in log_fields:
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Focus Table",
            "",
            "| case_id | route | n | m | proof_n | proof_m | ram_trace_domain | proof_size_bytes | prover_time_s | verifier_time_s | accepted | consumed |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | {row['route']} | {row['n']} | {row['m']} | "
            f"{row['proof_n']} | {row['proof_m']} | {row['ram_trace_domain']} | "
            f"{row['proof_size_bytes']} | {row['prover_time_s']} | {row['verifier_time_s']} | "
            f"{row['accepted']} | {row['consumed']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _normalize_rows(_read_tsv(args.input_tsv))
    csv_path = args.output_dir / args.summary_csv_name
    md_path = args.output_dir / args.summary_md_name
    _write_csv(csv_path, rows)
    _write_markdown(
        md_path,
        rows=rows,
        input_tsv=args.input_tsv,
        log_path=args.log,
        log_fields=_read_log_summary(args.log),
    )
    print(f"Histogram summary CSV: {csv_path}")
    print(f"Histogram summary Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
