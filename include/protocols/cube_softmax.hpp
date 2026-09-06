#pragma once

#include <vector>

#include "cube_sum.hpp"
#include "exp_check.hpp"
#include "fixed_point.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "signed_range_check.hpp"
#include "table_lookup.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

// Replicates an outer_count x inner_count matrix across the cube's middle axis.
// Example for base[2][2] = [[a, b], [c, d]] and middle_count = 2:
// returns cube[2][2][2] = [[[a, b], [a, b]], [[c, d], [c, d]]].
poly cube_middle_replication_polynomial(const poly& base,
                                        size_t outer_count,
                                        size_t middle_count,
                                        size_t inner_count);

class cubeMiddleReplicationCheckRecord {
public:
    size_t outer_count = 0;
    size_t middle_count = 0;
    size_t inner_count = 0;
    kzgCommitment boundary_selector_commitment;
};

class cubeMiddleReplicationParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    tableLookupProtocol* table_lookup = nullptr;
    size_t check_count = 0;
};

class cubeMiddleReplicationVerifier {
public:
    void add_check(const virCommitment& base,
                   const virCommitment& lifted,
                   const cubeMiddleReplicationCheckRecord& record);
    void add_direct_opening_check(Fr opened_value, Fr expected_value);
    bool finalize() const;

    std::vector<virCommitment> bases;
    std::vector<virCommitment> lifted_values;
    std::vector<cubeMiddleReplicationCheckRecord> checks;
    std::vector<std::pair<Fr, Fr>> direct_opening_checks;
    const cubeMiddleReplicationParam* params = nullptr;
};

class cubeMiddleReplicationProver {
public:
    cubeMiddleReplicationCheckRecord add_check(size_t outer_count,
                                               size_t middle_count,
                                               size_t inner_count);

    std::vector<cubeMiddleReplicationCheckRecord> checks;
    const cubeMiddleReplicationParam* params = nullptr;
};

// Proves that lifted[i][j][k] = base[i][k] for every cube coordinate.
class cubeMiddleReplicationProtocol : protected ProtocolLifecycle {
public:
    ~cubeMiddleReplicationProtocol() noexcept
    {
        auto_finalize("cubeMiddleReplicationProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero, tableLookupProtocol& table_lookup)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.table_lookup = &table_lookup;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& base,
                   const virCommitment& lifted,
                   size_t outer_count,
                   size_t middle_count,
                   size_t inner_count);

    bool finalize();

private:
    cubeMiddleReplicationParam params_;
    cubeMiddleReplicationProver prover_;
    cubeMiddleReplicationVerifier verifier_;
};

class cubeMiddleSoftmaxCheckSpec {
public:
    size_t quantization_bits = 0;
    size_t outer_count = 0;
    size_t middle_count = 0;
    size_t inner_count = 0;
    bool uses_signed_score_encoding = false;
    bool uses_explicit_shift = false;
    bool uses_row_max_shift = false;
    fixedPointEncoding score_encoding;
};

class cubeMiddleSoftmaxCheckRecord {
public:
    size_t quantization_bits = 0;
    size_t outer_count = 0;
    size_t middle_count = 0;
    size_t inner_count = 0;
    bool uses_signed_score_encoding = false;
    bool uses_explicit_shift = false;
    bool uses_row_max_shift = false;
    fixedPointEncoding score_encoding;
    kzgCommitment exp_commitment;
    kzgCommitment denominator_commitment;
    kzgCommitment lifted_denominator_commitment;
    kzgCommitment lifted_shift_commitment;
};

class cubeMiddleSoftmaxParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    signedRangeCheckProtocol* signed_range_check = nullptr;
    class cubeMiddleArgmaxProtocol* cube_middle_argmax = nullptr;
    expCheckProtocol* exp_check = nullptr;
    cubeSumProtocol* cube_sum = nullptr;
    cubeMiddleReplicationProtocol* replication = nullptr;
};

class cubeMiddleSoftmaxVerifier {
public:
    void add_check(const virCommitment& scores,
                   const virCommitment& softmax,
                   const cubeMiddleSoftmaxCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> scores;
    std::vector<virCommitment> softmax_values;
    std::vector<cubeMiddleSoftmaxCheckRecord> checks;
    const cubeMiddleSoftmaxParam* params = nullptr;
};

class cubeMiddleSoftmaxProver {
public:
    cubeMiddleSoftmaxCheckRecord add_check(const virCommitment& scores,
                                           size_t quantization_bits,
                                           size_t outer_count,
                                           size_t middle_count,
                                           size_t inner_count);

    cubeMiddleSoftmaxCheckRecord add_check(const virCommitment& scores,
                                           const fixedPointEncoding& score_encoding,
                                           size_t outer_count,
                                           size_t middle_count,
                                           size_t inner_count);

    cubeMiddleSoftmaxCheckRecord add_shifted_check(const virCommitment& shifts,
                                                   const fixedPointEncoding& score_encoding,
                                                   size_t outer_count,
                                                   size_t middle_count,
                                                   size_t inner_count);

    std::vector<cubeMiddleSoftmaxCheckRecord> checks;
    const cubeMiddleSoftmaxParam* params = nullptr;
};

// Proves softmax[i][j][k] = exp(scores[i][j][k]) / sum_l exp(scores[i][l][k]).
class cubeMiddleSoftmaxProtocol : protected ProtocolLifecycle {
public:
    ~cubeMiddleSoftmaxProtocol() noexcept
    {
        auto_finalize("cubeMiddleSoftmaxProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero,
               signedRangeCheckProtocol& signed_range,
               class cubeMiddleArgmaxProtocol& cube_middle_argmax,
               expCheckProtocol& exp,
               cubeSumProtocol& cube_sum,
               cubeMiddleReplicationProtocol& replication)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.signed_range_check = &signed_range;
        params_.cube_middle_argmax = &cube_middle_argmax;
        params_.exp_check = &exp;
        params_.cube_sum = &cube_sum;
        params_.replication = &replication;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& scores,
                   const virCommitment& softmax,
                   size_t quantization_bits,
                   size_t outer_count,
                   size_t middle_count,
                   size_t inner_count);

    // Signed overload: scores are offset-binary encoded scaled integers, while the
    // softmax output still lives in the field as exp(score) / sum exp(score).
    void add_check(const virCommitment& scores,
                   const virCommitment& softmax,
                   const fixedPointEncoding& score_encoding,
                   size_t outer_count,
                   size_t middle_count,
                   size_t inner_count);

    // Shifted/stabilized softmax path:
    // - `scores` and `shifted_scores` are encoded signed fixed-point cubes
    // - `shifts` is an outer_count x inner_count encoded matrix replicated across the middle axis
    // - the protocol proves shifted_scores = scores - lift(shifts) + offset
    // Example: choosing shifts as per-row maxima keeps shifted_scores in [-M, 0].
    void add_shifted_check(const virCommitment& scores,
                           const virCommitment& shifted_scores,
                           const virCommitment& shifts,
                           const virCommitment& softmax,
                           const fixedPointEncoding& score_encoding,
                           size_t outer_count,
                           size_t middle_count,
                           size_t inner_count);

    // Row-max-derived shifted softmax adds a selector that proves shifts are the
    // middle-axis maxima of scores before applying the shifted softmax relation.
    void add_row_max_shifted_check(const virCommitment& scores,
                                   const virCommitment& shifted_scores,
                                   const virCommitment& shifts,
                                   const virCommitment& shift_selector,
                                   const virCommitment& softmax,
                                   const fixedPointEncoding& score_encoding,
                                   size_t outer_count,
                                   size_t middle_count,
                                   size_t inner_count);

    bool finalize();

private:
    cubeMiddleSoftmaxParam params_;
    cubeMiddleSoftmaxProver prover_;
    cubeMiddleSoftmaxVerifier verifier_;
    std::vector<virCommitment> scores_;
    std::vector<virCommitment> shifted_scores_;
    std::vector<virCommitment> shifts_;
    std::vector<virCommitment> shift_selectors_;
    std::vector<virCommitment> softmax_values_;
    std::vector<cubeMiddleSoftmaxCheckSpec> checks_;
};
