#pragma once

#include "kzg.hpp"
#include "polynomial.hpp"
#include "protocol_lifecycle.hpp"
#include "table_lookup.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

// Sums a flattened cube along the middle dimension.
// Example for cube[2][2][2]:
//   outer 0: [[a, b], [c, d]]
//   outer 1: [[e, f], [g, h]]
// returns [[a + c, b + d], [e + g, f + h]].
poly cube_middle_sum_polynomial(const poly& cube,
                                size_t outer_count,
                                size_t middle_count,
                                size_t inner_count);

// Segmented prefix sums along the middle dimension.
// Example for cube[2][3][1] with values [[a], [b], [c], [d], [e], [f]]:
// returns [[a], [a + b], [a + b + c], [d], [d + e], [d + e + f]].
poly cube_middle_prefix_sum_polynomial(const poly& cube,
                                       size_t outer_count,
                                       size_t middle_count,
                                       size_t inner_count);

// Metadata for one cube-sum check, including prover-generated witnesses.
class cubeSumCheckRecord {
public:
    size_t outer_count = 0;
    size_t middle_count = 0;
    size_t inner_count = 0;
    kzgCommitment prefix_commitment;
    kzgCommitment boundary_selector_commitment;
};

// Shared dependencies for the cube-sum protocol.
class cubeSumParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    tableLookupProtocol* table_lookup = nullptr;
    size_t check_count = 0;
};

// Verifier state for batched cube-sum checks.
// Besides ordinary check records, this keeps direct opening checks for
// one-point anchor constraints.
class cubeSumVerifier {
public:
    void add_check(const virCommitment& cube, const virCommitment& sum, const cubeSumCheckRecord& record);
    void add_direct_opening_check(Fr opened_value, Fr expected_value);
    bool finalize() const;

    std::vector<virCommitment> cubes;
    std::vector<virCommitment> sums;
    std::vector<cubeSumCheckRecord> checks;
    std::vector<std::pair<Fr, Fr>> direct_opening_checks;
    const cubeSumParam* params = nullptr;
};

// Prover state for batched cube-sum checks.
class cubeSumProver {
public:
    cubeSumCheckRecord add_check(const virCommitment& cube,
                                 size_t outer_count,
                                 size_t middle_count,
                                 size_t inner_count);

    std::vector<cubeSumCheckRecord> checks;
    const cubeSumParam* params = nullptr;
};

// Proves that sum[i][k] = sum_j cube[i][j][k].
// The implementation treats the cube as an (outer_count * middle_count) x inner_count
// matrix, proves a segmented prefix sum over each length-middle_count block, and
// then uses table lookup to show that sum is copied from the last row of each block.
class cubeSumProtocol : protected ProtocolLifecycle {
public:
    explicit cubeSumProtocol(tableLookupProtocol& table_lookup)
        : table_lookup_dependency(&table_lookup)
    {
    }

    ~cubeSumProtocol() noexcept
    {
        auto_finalize("cubeSumProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        params.table_lookup = table_lookup_dependency;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    void add_check(const virCommitment& cube,
                   const virCommitment& sum,
                   size_t outer_count,
                   size_t middle_count,
                   size_t inner_count);

    bool finalize();

protected:
    cubeSumParam params;
    cubeSumProver prover;
    cubeSumVerifier verifier;
    tableLookupProtocol* table_lookup_dependency = nullptr;
};
