# Terrae

Terrae is a zero-knowledge proof system for the training and inference of gradient boosted decision trees (GBDT). This artifact contains the benchmarks for the Terrae GBDT proofs and the histogram aggregation comparison against the retained ZK-RAM and CertXGB weighted-LogUp baselines.

## Requirements

Software requirements:

- Linux (native Linux or WSL2)
- CMake 3.14 or newer
- a C++17 compiler such as `g++` or `clang++`
- Python 3.10 or newer
- Git, `curl`, and `unzip`
- Boost development headers, including Multiprecision (`libboost-dev`)
- GMP and GMPXX development headers and libraries (`libgmp-dev`)

CMake fetches [herumi/mcl](https://github.com/herumi/mcl) at its pinned revision by default. Set `TERRAE_MCL_SOURCE_DIR` to an existing local mcl source tree to avoid that fetch. The benchmark targets download their reference datasets from UCI, so the first run normally requires network access.

The complete paper benchmark is intended for a server-class machine. We recommend at least 192 GiB of RAM and 50 GiB of free disk space. The default runner limits each proof process to 180 GiB RSS; this is a safety limit, not a guarantee that every case will consume that much memory. Cached SRS files can occupy tens of gigabytes, and a complete run may take hours or days.

## Environment Check

Check the required tools, then run the small installation benchmark:

```bash
make doctor
make benchmark-smoke
```

The first command reports missing tools and datasets. The smoke target builds the project, downloads its small reference dataset, and runs both the GBDT and histogram workflows. Its reports are written to:

```text
out/benchmark_gbdt_smoke/summary.csv
out/benchmark_gbdt_smoke/summary.md
out/benchmark_histogram_smoke/summary.csv
out/benchmark_histogram_smoke/summary.md
```

Smoke measurements are installation checks and are not paper measurements.

## Paper Benchmarks

Run every benchmark used for the paper and assemble the combined reports with:

```bash
make benchmark-paper
```

This runs the following benchmark families sequentially:

```bash
make benchmark             # Dry Bean and Optdigits scaling benchmark
make benchmark-practical   # N=1024, d=3, T=1/2/4/8 practical benchmark
make benchmark-histogram   # histogram aggregation benchmark
```

The GBDT benchmark families are resumable: completed stages and compatible SRS files under `out/srs/` are reused. If the families have already completed, assemble their existing results without rerunning proofs:

```bash
make benchmark-paper-summary
```

The combined reader-facing output is written to `out/benchmark_paper/` and contains:

```text
summary.csv
terrae_summary.csv
zkboost_summary.csv
soundness_summary.csv
histogram_summary.csv
manifest.json
```

The individual outputs and logs are stored in `out/benchmark/`, `out/benchmark_practical/`, and `out/benchmark_histogram/`. For optional case selection, stage-specific reruns, and benchmark parameter overrides, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Estimated Setup and Runtime

On our dual-Xeon Gold 6326 server (64 logical CPUs, about 188 GiB RAM),
the following time is consumed:

| Stage | Estimated time |
| --- | --- |
| Initial build | 10-30 minutes, plus downloads |
| Smoke tests after building | 2-5 minutes |
| Full histogram benchmark | 1-2 hours |
| GBDT scaling | About 2 days |
| Practical settings | About 1 day |
| Complete paper benchmark | 3-5 days |

These are planning estimates based on historical runs, some of which included
failed or skipped stages. GBDT timings assume cached SRS files; first-time large
SRS generation was not timed separately, which could require additional hours. Cached runs
still incur SRS loading costs. Paper prover times exclude SRS preparation and
are shorter than end-to-end command runtimes.

## Citation

Terrae has been accepted to the ACM SIGSAC Conference on Computer and Communications Security (CCS) 2026. If you use this artifact, please cite:

```bibtex
@inproceedings{gao2026terrae,
  author    = {Jiacheng Gao and Wenjie Qu and Yuan Zhang and Sheng Zhong and Jiaheng Zhang},
  title     = {Zero-Knowledge Proofs for Gradient Boosted Decision Trees},
  booktitle = {Proceedings of the 2026 ACM SIGSAC Conference on Computer and Communications Security},
  year      = {2026},
  doi       = {10.1145/3830454.3846711},
  url       = {https://doi.org/10.1145/3830454.3846711}
}
```

Conference Paper: https://doi.org/10.1145/3830454.3846711

Full Version: https://eprint.iacr.org/2026/907

## License

See [LICENSE](LICENSE).
