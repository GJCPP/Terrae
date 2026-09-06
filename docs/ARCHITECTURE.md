# Architecture

Terrae is a research implementation of zero-knowledge proofs for gradient
boosted decision tree (GBDT) training and inference. The repository is organized
around the proof statements evaluated in the paper:

- well-formedness of a committed forest;
- correctness of GBDT training against a committed dataset and public training
  procedure;
- correctness of batched inference against a committed forest;
- histogram aggregation benchmarks used to isolate the main aggregation
  primitive.

The implementation separates semantic trace generation from proof execution.
Python tooling trains fixed-point histogram-tree models and exports semantic
traces. The C++ proof system validates those traces, derives algebraic
witnesses, commits to the relevant polynomials, and runs the proof workflows.

## Repository Map

```text
include/
  algebra/       field, polynomial, subgroup, matrix, and fixed-point helpers
  commitments/   transcript, proof session, artifact, and KZG APIs
  protocols/     reusable algebraic proof protocols
  plonk/         retained PLONK-style circuit and proof components
  forest/
    structure/          committed-forest structure and legality checks
    public_inference/   forest inference statements and prediction checks
    training/           training, export, and committed-inference workflows
  zkboost/       retained CertXGB comparison route
src/
  algebra/
  commitments/
  protocols/
  plonk/
  forest/
    structure/
    public_inference/
    training/
  zkboost/
examples/        runnable proof and benchmark binaries
tests/           GoogleTest coverage for protocol and workflow behavior
tools/           benchmark runners, dataset tooling, and trace export scripts
```

The shared C++ library target is `terrae_core`. It contains the common protocol
implementation used by tests, examples, and benchmark binaries.

## Proof Stack

The code follows a reduction-based structure. High-level GBDT statements are
translated into polynomial relations and lookup statements, then discharged by
lower-level protocols over KZG commitments.

The main layers are:

- `algebra`: field arithmetic, polynomial evaluation, Lagrange-basis helpers,
  matrix utilities, and fixed-point encodings used by the GBDT trace.
- `commitments`: KZG public parameters, polynomial commitments, Fiat-Shamir
  transcript state, serialized proof artifacts, and prover/verifier transport.
- `protocols`: reusable algebraic checks such as sum checks, zero checks,
  table lookups, concatenated/CQ lookups, routing, shuffle, argmax, range, and
  histogram aggregation.
- `forest`: application-level reductions from GBDT structure, training, and
  inference statements into the reusable protocol layer.

`ProofRuntime` owns the execution mode, KZG context, transcript, and proof
session. `ProofProgram` is the composition layer that binds public polynomial
commitments, witness commitments, verifier artifacts, and protocol checks into
one proof artifact. Verifier execution consumes the serialized artifact and the
public commitments rather than the prover's witness polynomials.

## Commitment And Transcript Model

Proof objects are represented by `ProofArtifact` and transported through
`ProofSession`. Values that must affect Fiat-Shamir challenges are exchanged
through transcript-visible operations. Values that are serialized only for
prover/verifier synchronization are transported without becoming new transcript
inputs.

Most high-level checks use `virCommitment` expressions to represent symbolic
linear combinations of committed polynomials. This keeps application statements
close to their algebraic form: workflow code binds named committed columns, and
protocol code reduces those expressions to openings, sum checks, lookup checks,
or zero checks.
