# Benchmarks

The canonical commands for reproducing the paper are:

```bash
make benchmark-paper
```

This sequentially runs the scaling, practical-setting, and histogram
benchmarks, then invokes `make benchmark-paper-summary`. The latter writes a
single reader-facing directory, by default `out/benchmark_paper/`, containing:

- `summary.csv`: all GBDT cross-system rows;
- `terrae_summary.csv`: Terrae-native GBDT statistics;
- `zkboost_summary.csv`: CertXGB-native GBDT statistics;
- `soundness_summary.csv`: Appendix K soundness ledgers for Terrae proofs;
- `histogram_summary.csv`: histogram microbenchmark rows;
- `manifest.json`: source directories and row counts.

All benchmark stages are resumable. To run or resume an individual family:

```bash
make benchmark             # Dry Bean + Optdigit* scaling tables
make benchmark-practical   # N=1024, d=3, T=1/2/4/8 practical table
make benchmark-histogram   # histogram microbenchmark tables
make benchmark-paper-summary # assemble already completed outputs only
```

`make benchmark` uses `benchmark_rebuttal_full.json`; `make
benchmark-practical` uses `benchmark_practical_N1024_d3_T124.json`. Both use
one process at a time, reuse compatible SRS files from `out/srs/gbdt/`, resume
completed stages, record ordinary failures, and record RSS/OOM-guarded points
as unavailable while continuing the remaining cases. They default to the
180-GiB RSS plus OOM guard.

For a quick installation check:

```bash
make benchmark-smoke
```

When there is not enough time to rerun proofs, the exact soundness-only
command reproduces Appendix K counters from a Terrae trace pair:

```bash
python3 tools/python_trace_export/report_soundness_from_traces.py \
  path/to/training.trace path/to/inference.trace \
  --case-id CASE_ID --dataset DATASET \
  --output soundness_metrics_exact.csv
```

A complete parameter plan can be exported and counted with:

```bash
python3 tools/python_trace_export/report_soundness_from_traces.py \
  --cases-json tools/python_trace_export/configs/benchmark_rebuttal_full.json \
  --output-dir out/soundness_rebuttal --jobs 2
```

The native `soundness-from-split-traces` mode executes the same deterministic
protocol-lowering/finalization schedule as verification, using metadata-only
commitments and placeholder proof values. It does not load an SRS, run KZG
commitment/opening/pairing work, generate a proof, or change the normal
prover/verifier path. Rows are marked `derivation_kind=exact_count_only`.

This produces separate smoke summaries under `out/benchmark_gbdt_smoke/` and
`out/benchmark_histogram_smoke/`; they are not paper measurements.

## GBDT Benchmark

The GBDT benchmark compares:

- Terrae committed-forest GBDT proofs;
- CertXGB shared GBDT semantic-trace proofs.

The top-level reports in both the scaling and practical output directories are:

```text
summary.csv
terrae_summary.csv
zkboost_summary.csv
soundness_summary.csv
summary.md
```

Each case also retains its workflow report and peak-memory record under
`cases/`. SRS setup/loading diagnostics are logged separately from proof timing.

## Histogram Benchmark

The histogram benchmark compares:

- Terrae `hist_agg`;
- the retained CertXGB RAM-style histogram primitive.

Smoke run:

```bash
make benchmark-histogram-smoke
```

Default output:

```text
out/benchmark_histogram_smoke/summary.csv
out/benchmark_histogram_smoke/summary.md
```

The smoke SRS cache is shared across output directories by default:

```text
out/srs/histogram/kzg_srs_smoke.params
```

The smoke grid is intentionally small:

```text
(n, m) = (128, 64), (256, 128)
```

The first smoke run may spend around a minute generating the local SRS cache.
Later runs reuse `out/srs/histogram/kzg_srs_smoke.params`.

The raw histogram binary output is retained next to the summaries:

```text
out/benchmark_histogram_smoke/histogram.tsv
out/benchmark_histogram_smoke/run.log
```

Full run:

```bash
make benchmark-histogram
```

Default output:

```text
out/benchmark_histogram/summary.csv
out/benchmark_histogram/summary.md
```

The full histogram target computes the minimum required SRS size from the
selected histogram cases and route, then caches it as:

```text
out/srs/histogram/kzg_srs_<size>.params
```

An explicit `SRS_SIZE` acts as a lower bound; the script raises it when the
selected workload requires a larger SRS. Prefer the Makefile target for release
measurements unless you intentionally override the environment.

The raw TSV and stderr log are retained as:

```text
out/benchmark_histogram/histogram.tsv
out/benchmark_histogram/run.log
```

Override smoke or full settings with environment assignments:

```bash
make benchmark-histogram-smoke \
  HISTOGRAM_SMOKE_ENV='BUILD_DIR=build SKIP_BUILD=1 ROUTE=ram REPEATS=3 HISTOGRAM_CASES="128:64 256:128" SRS_CACHE=out/srs/histogram/custom_smoke.params'

make benchmark-histogram \
  HISTOGRAM_ENV='BUILD_DIR=build SKIP_BUILD=1 ROUTE=both REPEATS=1 SRS_CACHE=out/srs/histogram/custom.params'
```

The underlying script is:

```bash
bash tools/run_histogram_scale_benchmark.sh
```

It supports these environment variables:

- `BUILD_DIR`
- `JOBS` (build parallelism for the histogram benchmark binary; default `1`)
- `ROUTE`
- `REPEATS`
- `SRS_SIZE`
- `SRS_PATH`
- `SRS_CACHE`
- `SKIP_BUILD`
- `HISTOGRAM_CASES`
- `PROOF_N`
- `PROOF_M`
- `PROOF_DIR`
- `VERIFIER_ONLY`

## Interpreting Results

Both benchmark families write top-level `summary.csv` and `summary.md` files in
their output directory. GBDT also keeps per-case workflow reports under
`cases/`. Histogram also keeps the raw TSV and stderr log as `histogram.tsv` and
`run.log`. SRS generation/loading and setup diagnostics are logged separately
from per-row proof timings where the underlying benchmark exposes that split.

Smoke targets are intended to verify installation and workflow wiring. They are
not full measurements. Use the default benchmark targets or explicit case files
for full tables.

If you change `SRS_SIZE`, `PROOF_N`, or `PROOF_M` for histogram runs, also use a
matching `SRS_CACHE` path. A cache generated for a smaller SRS cannot satisfy a
larger benchmark.
