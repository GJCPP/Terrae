# Examples

The examples are source-level documentation for the current
`ProofRuntime + ProofProgram` API and the release benchmark binaries. They are
built by the root CMake project and link against the shared `terrae_core`
library, so protocol implementations live centrally under `src/`.

Build the examples with:

```bash
make build
```

Run a specific example from the repository root after building:

```bash
./build/examples/example_custom_program
```

## CMake Targets

The active example and helper targets are:

- `kzg_setup_helper`
- `example_custom_program`
- `example_private_table_lookups`
- `example_all_protocols`
- `example_forest_training_proof`
- `example_forest_committed_inference`
- `example_training_then_inference`
- `example_training_then_inference_bundle`
- `example_zkboost_shared_trace`
- `benchmark_histogram_primitives`

## Application API Examples

`example_custom_program.cpp` is the best starting point for composing a custom
proof sequence. It shows one `ProofProgram` binding witness commitments, public
commitments, and several protocol checks into one proof session.

`example_private_table_lookups.cpp` demonstrates split prover/verifier
execution for lookups against a private table, including external SRS loading
and proof bundle serialization:

```bash
make srs
./build/examples/example_private_table_lookups prover build/kzg_srs.params /tmp/private_lookup.bundle
./build/examples/example_private_table_lookups verifier build/kzg_srs.params /tmp/private_lookup.bundle
```

`example_all_protocols.cpp` is the broadest integration reference. It exercises
many runtime-facing protocols in a single prover/verifier bundle flow:

```bash
make srs
./build/examples/example_all_protocols prover build/kzg_srs.params /tmp/all_protocols.bundle
./build/examples/example_all_protocols verifier build/kzg_srs.params /tmp/all_protocols.bundle
```

## GBDT Examples

`example_forest_training_proof.cpp` builds a small semantic training trace,
derives the committed-forest training witness bundle, proves it through the
training workflow, and verifies the resulting artifact.

`example_training_then_inference.cpp` demonstrates the full application flow:
training proof, trained-leaf export, and one batched exported-forest inference
proof.

`example_training_then_inference_bundle.cpp` is the scriptable file-based
variant. It can write demo cases, validate semantic traces, convert split
training/inference traces into proof inputs, and run prover/verifier or profile
modes. Useful commands include:

```bash
./build/examples/example_training_then_inference_bundle describe-formats
./build/examples/example_training_then_inference_bundle describe-trace-format
./build/examples/example_training_then_inference_bundle describe-trace-contract
./build/examples/example_training_then_inference_bundle required-srs-from-split-traces /tmp/training.trace /tmp/inference.trace
./build/examples/example_training_then_inference_bundle prover-from-split-traces build/kzg_srs.params /tmp/training.trace /tmp/inference.trace /tmp/terrae.bundle
./build/examples/example_training_then_inference_bundle verifier-from-split-traces build/kzg_srs.params /tmp/training.trace /tmp/inference.trace /tmp/terrae.bundle
```

The benchmark runner uses its profile modes to collect stage-level timing and
memory rows for committed-forest preprocessing, standalone well-formedness,
training, and inference.

`example_forest_committed_inference.cpp` proves committed-forest inference
directly against a verifier-side forest commitment. It supports a built-in demo,
a synthetic scale profile, and trace-driven mode:

```bash
./build/examples/example_forest_committed_inference
./build/examples/example_forest_committed_inference profile-synthetic-scale
./build/examples/example_forest_committed_inference build/kzg_srs.params /tmp/training.trace /tmp/inference.trace
```

## CertXGB Example

`example_zkboost_shared_trace.cpp` exposes the retained CertXGB comparison
route over the same semantic GBDT traces used by Terrae:

```bash
./build/examples/example_zkboost_shared_trace profile-commit-forest build/kzg_srs.params /tmp/training.trace
./build/examples/example_zkboost_shared_trace profile-validate-forest build/kzg_srs.params /tmp/training.trace /tmp/inference.trace
./build/examples/example_zkboost_shared_trace profile-inference build/kzg_srs.params /tmp/training.trace /tmp/inference.trace
./build/examples/example_zkboost_shared_trace profile-inference-preprocessed build/kzg_srs.params /tmp/training.trace /tmp/inference.trace
./build/examples/example_zkboost_shared_trace required-srs-preprocessed /tmp/training.trace /tmp/inference.trace
./build/examples/example_zkboost_shared_trace profile-synthetic-inference-scale
./build/examples/example_zkboost_shared_trace profile-cert-gbdt build/kzg_srs.params /tmp/training.trace /tmp/inference.trace
```

## Histogram Benchmark Binary

`benchmark_histogram_primitives.cpp` compares Terrae `hist_agg`, the retained
CertXGB RAM-style histogram primitive, and CertXGB's weighted-LogUp histogram
reduction. Use `--route all` to run all three routes.

```text
s[i] = sum_{j: index[j] = i} val[j]
```

Run the standalone binary directly for small local checks:

```bash
./build/examples/benchmark_histogram_primitives --route all --repeats 3
```

The binary accepts:

```text
--route both|all|terrae|ram|weighted-logup
--repeats N
--srs PATH
--srs-size N
--srs-cache PATH
--proof-n N --proof-m M
--proof-dir DIR
--verifier-only
--case n:m
```

For release histogram runs, use the wrapper through the Makefile. It writes raw
TSV plus summarized `summary.csv` and `summary.md` files:

```bash
make benchmark-histogram
make benchmark-histogram-smoke
```

## KZG SRS Helper

`kzg_setup_helper.cpp` generates KZG public parameters on disk. The Makefile
wraps it as:

```bash
make srs
```

By default this writes `build/kzg_srs.params` with `131072` G1 and G2 powers.
Override the path or size with `SRS_PATH`, `SRS_DEG1_PLUS_1`, and
`SRS_DEG2_PLUS_1`.
