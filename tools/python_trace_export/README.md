# Python Trace Export

This directory contains the Python tooling used by the release GBDT benchmark:
dataset download helpers, fixed-point histogram-tree training, semantic trace
export, benchmark runners, and report extraction.

The Python tools use only the standard library. No `pip install` step is
required.

## Release Entrypoints

The paper/rebuttal benchmark has one default entrypoint:

```bash
make benchmark
```

The selected practical comparison has a separate entrypoint:

```bash
make benchmark-practical
```

It fixes `N=b=1024`, current `d=3`, `bins=8`, first-order RSS loss, and learning
rate `4/1`, then runs `T=1,2,4,8` for Dry Bean and Optdigits-7. All Terrae
cases run before all
CertXGB cases. If a proof stage fails, swaps, or encounters OOM at a given
`T`, only that stage is skipped at larger `T` values for the same dataset and
route; the independent training or inference stage continues. These skipped
stages remain resumable and can be explicitly rerun later.

The current trainer and benchmark runner interpret `d` as a one-based maximum
node depth: every root has depth 1 and every active leaf has depth at most
`d`. The earliest benchmark plans instead used legacy depth 1 for a stump
(one root split). The checked-in plans therefore use current `d = legacy d + 1`;
for example, the legacy anchor `d=1` is now executed with current `d=2`.

The default `make benchmark` runs `benchmark_rebuttal_full.json`: nine unique anchor/scaling settings on
both Dry Bean and Optdigits restricted to seven classes. All nine Dry Bean
cases run first, followed by all nine Optdigits cases. Every dataset/setting
case runs both Terrae and CertXGB, for 18 runner cases and 36 route-level
results. It uses a single worker, a Release build under `build-release/`, no
wall-clock timeout, and writes under `out/benchmark/` by default.

The entrypoint is resumable. Re-running the same command skips completed cases
using `benchmark_plan_state.json`, and `--reuse-existing-stages` reuses
successfully checkpointed stages inside an interrupted or failed case. Use a
new `BENCHMARK_OUTPUT_DIR` when intentionally starting a new measurement run.
Case state is keyed by `case_id`, and an existing case's recorded output
directory is reused even if the plan order changes, so reordered plans can
resume the original per-stage checkpoints safely.

Logs and checkpoints include:

- `out/benchmark/benchmark.log`: append-only top-level build/dataset/runner log;
- `out/benchmark/benchmark_plan_state.json`: atomic case-level checkpoint;
- `out/benchmark/cases/*/case_runner.log`: complete per-case runner log;
- `out/benchmark/cases/*/cases/*/stage_checkpoint.json`: atomic successful-stage checkpoint;
- per-stage stdout/stderr logs and `*_memory.json` peak-RSS records;
- per-case `benchmark_report.json`/`.md`, focused CSV/Markdown, and top-level
  `summary.csv`/`.md`.

The memory wrapper atomically refreshes each `*_memory.json` while a stage is
running and finalizes it when the child succeeds, fails, times out, or receives
SIGINT/SIGTERM. A SIGKILL or machine failure cannot run cleanup, but the most
recent periodic snapshot remains on disk.

The full `make benchmark` entrypoint also enables process-tree memory guards.
The wrapper terminates a stage when its process-tree RSS exceeds 180 GiB and
saves an `rss_limit_detected` record. Swap usage is measured but does not trigger
termination. Linux cgroup-v2 `memory.events` is sampled for `oom`/`oom_kill`;
`/proc/vmstat` is used as a fallback when cgroup events are unavailable. An OOM
event is saved as `oom_detected`. A second case-level wrapper covers SRS preparation, trace export,
and all nested proof stages, writing `case_memory_guard.json`. Either guard marks the current case unavailable; the parameter-plan runner records it and continues with the next case. Successfully checkpointed stages remain reusable on a later run.
Set `BENCHMARK_MEMORY_GUARD_ENV=` only when intentionally disabling both guards.

The reported Terrae training total includes its well-formedness proof. Proof
bytes and prover/verifier times are added; because the two stages execute
sequentially, combined peak memory is their maximum. The raw core-training and
well-formedness stages remain separately available. CertXGB training includes
its bound forest well-formedness component, and both inference routes include
routing, score aggregation, and tie-broken argmax.

SRS handling is automatic. Each case queries the Terrae and CertXGB required
KZG degrees, chooses a shared power-of-two SRS large enough for both, and uses
that same SRS file for the two systems. Files are cached as
`out/srs/gbdt/kzg_deg1_<n>_deg2_<m>.srs`; generation is protected by a file
lock and an atomic temporary-file rename. Existing matching files are reused.
The C++ stage loaders decode only the prefix required by that stage. Required-SRS
query and setup generation are preprocessing and are not included in reported
proof-stage prover/verifier time or peak memory.

The installation smoke entrypoint is:

```bash
make benchmark-smoke
```

For paper measurements, use `make benchmark-paper` or the canonical individual
commands `make benchmark`, `make benchmark-practical`, and
`make benchmark-histogram`.

## Dataset Preparation

The root Makefile wraps the dataset downloader:

```bash
make datasets
make benchmark-smoke
```

`make datasets` downloads all retained reference datasets into
`data/datasets/raw/`. The smoke benchmark downloads only the Iris dataset when
needed.

The downloader is:

```bash
DATASET_SET=all bash tools/python_trace_export/download_reference_datasets.sh
DATASET_SET=smoke bash tools/python_trace_export/download_reference_datasets.sh
```

Set `FORCE_DATASETS=1` to refresh existing files.

## Training And Trace Export

Generate split training and inference traces with:

```bash
python3 tools/python_trace_export/train_fixed_point_dataset.py \
  --dataset wine \
  --iterations 2 \
  --max-depth 2 \
  --bin-count 16 \
  --training-trace /tmp/wine_training.trace \
  --inference-trace /tmp/wine_inference.trace
```

Supported dataset names are:

- `iris`
- `wine`
- `wdbc`
- `optdigits-train`
- `optdigits-7class-train`
- `dry-bean`
- `iris-binary`
- `wine-binary`
- `breast-cancer-binary`
- `optdigits-parity-train`

The trainer defaults to `--gradient-mode first-order-rss` (the main-paper
Equations (2.3)--(2.9): negative gradients, RSS split reduction, and
mean-residual leaves). Use `--gradient-mode second-order-xgboost` to retain the
Appendix G.1 objective. The selected loss is stored as public `training.loss_mode`
in the semantic training trace and is enforced by both the Terrae and CertXGB
training routes. The parameterized workflow accepts the equivalent
`--loss-mode {first-order-rss,second-order-xgboost}` option (and JSON cases may
set `loss_mode`); its default is also `first-order-rss`.

## Leakage-Free Benchmark Accuracy

The practical benchmark splits raw data before preprocessing, chooses exactly
`N` training samples with deterministic proportional stratified sampling, and
fits uniform-bin minima and maxima only on those samples. Dry Bean uses a seeded
stratified 80/20 holdout; Optdigits uses the official test split restricted to
labels 0--6. The training subset preserves the class proportions of its source pool instead of
using a class-balanced prefix. Test values outside the training range are clamped
to the first or last bin.

For Dry Bean and Optdigits-7, the retained proof benchmark invokes the trace
exporter with the same clean proportional training-subset and training-only
quantization semantics, so the proved forest matches the model used for the
accuracy evaluation. The benchmark
reports `accuracy` by applying that exact trained forest to the held-out test
split in plaintext: a seeded stratified 20% holdout for Dry Bean and the official
test split restricted to labels 0--6 for Optdigits. The held-out rows are never
included in the training or inference proof traces. `accuracy_split` and
`accuracy_samples` make the evaluation population explicit in every CSV row.
The benchmark training-semantics fingerprint prevents reuse of checkpoints
produced with the legacy class-balanced prefix.

## Parameterized Benchmark Runner

The Makefile calls the parameterized runner underneath:

```bash
make benchmark             # scaling tables
make benchmark-practical   # practical-setting table
make benchmark-histogram   # histogram tables
make benchmark-paper-summary
```

The underlying GBDT runner remains `run_benchmark_parameter_plan.py`; its two
paper plans are `benchmark_rebuttal_full.json` and
`benchmark_practical_N1024_d3_T124.json`.

The runner produces top-level `summary.csv`, `terrae_summary.csv`,
`zkboost_summary.csv`, and `summary.md` reports, plus per-case workflow
artifacts under `cases/`. Each CSV row is one paper proof stage:

- `summary.csv` is the cross-system table. Besides the setting and performance
  fields (time, proof size, peak memory, and practical-setting accuracy), it
  contains exactly the four shared structural metrics reported in the paper:
  KZG-processed coefficients, polynomial commitments, polynomial--point
  claims, and verifier pairings.
- `terrae_summary.csv` contains only Terrae performance rows.
- `zkboost_summary.csv` contains only CertXGB performance rows plus its active
  PLONK-gate count, the sole system-native metric reported in the paper.

The runner intentionally does not collect or export relation-family, opening
provenance, MSM, SRS, padding, or other implementation-debug statistics.
Every CSV also includes failed, timed-out, and
unsupported cases. `case_status` records the outcome of the whole case, while
`stage_status` distinguishes completed stages from the failed or unstarted
stage. The case JSON plan determines the complete row set, including cases that
were never started. Metrics that were not produced remain empty; for a memory-guard failure,
`peak_memory_bytes` records the largest observed RSS across all attempts and
`memory_source` is `case-memory-guard-peak`. During aggregate generation, the
runner regenerates all three per-case CSV files from workflow reports and stage
checkpoints, so a resumed run also upgrades older outputs without rerunning
proofs.

The retained plan files are:

- `benchmark_rebuttal_full.json` (scaling tables)
- `benchmark_practical_N1024_d3_T124.json` (practical table)
- `benchmark_smoke.json` (installation smoke check)

## Module Map

- `schema.py`: high-level semantic trace records.
- `writer.py`: text trace writers consumed by the C++ workflows.
- `validate.py`: shape, encoding, index, mask, selector, and compatibility checks.
- `extractor.py`: lowers trainer-facing objects into the semantic trace schema.
- `fixed_point_trainer.py`: proof-compatible fixed-point histogram trainer.
- `datasets.py` and `zkboost/datasets.py`: local dataset loading and binning.
- `train_fixed_point_dataset.py`: CLI for training and trace export.
- `benchmark_common.py`: shared build, profile, report, and trace helpers.
- `benchmark_parameterized_workflow.py`: single-case workflow runner.
- `benchmark_focus_extract.py`: focused report extraction helper.
- `run_benchmark_parameter_plan.py`: multi-case benchmark plan runner.
- `benchmark_standard.py`: retained manual runner for small ad hoc comparisons.
- `download_reference_datasets.sh`: local reference dataset downloader.
