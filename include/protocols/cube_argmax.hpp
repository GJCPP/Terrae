#pragma once

#include "cube_softmax.hpp"
#include "cube_sum.hpp"
#include "fixed_point.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class cubeMiddleArgmaxCheckRecord {
public:
    size_t outer_count = 0;
    size_t middle_count = 0;
    size_t inner_count = 0;
    size_t bit_bound = 0;
    kzgCommitment lifted_max_commitment;
};

class cubeMiddleArgmaxParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    cubeSumProtocol* cube_sum = nullptr;
    cubeMiddleReplicationProtocol* replication = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

class cubeMiddleArgmaxVerifier {
public:
    void add_check(const virCommitment& scores,
                   const virCommitment& max_values,
                   const virCommitment& selector,
                   const cubeMiddleArgmaxCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> scores;
    std::vector<virCommitment> max_values;
    std::vector<virCommitment> selectors;
    std::vector<cubeMiddleArgmaxCheckRecord> checks;
    const cubeMiddleArgmaxParam* params = nullptr;
};

class cubeMiddleArgmaxProver {
public:
    cubeMiddleArgmaxCheckRecord add_check(const virCommitment& max_values,
                                          size_t outer_count,
                                          size_t middle_count,
                                          size_t inner_count,
                                          size_t bit_bound);

    std::vector<cubeMiddleArgmaxCheckRecord> checks;
    const cubeMiddleArgmaxParam* params = nullptr;
};

// Proves that max_values[outer][inner] is the maximum score across the middle axis.
//
// Statement:
// - selector is boolean on the full cube
// - selector sums to 1 across the middle axis for each (outer, inner)
// - cube_middle_sum(selector o scores) = max_values
// - max_values replicated across the middle axis dominates scores entry-wise
//
// Example for outer_count = 1, middle_count = 4, inner_count = 1:
//   scores    = [[[1], [5], [2], [4]]]
//   selector  = [[[0], [1], [0], [0]]]
//   max       = [[5]]
class cubeMiddleArgmaxProtocol : protected ProtocolLifecycle {
public:
    ~cubeMiddleArgmaxProtocol() noexcept
    {
        auto_finalize("cubeMiddleArgmaxProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero,
               cubeSumProtocol& cube_sum,
               cubeMiddleReplicationProtocol& replication,
               lessCheckProtocol& less_check)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.cube_sum = &cube_sum;
        params_.replication = &replication;
        params_.less_check = &less_check;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& scores,
                   const virCommitment& max_values,
                   const virCommitment& selector,
                   size_t outer_count,
                   size_t middle_count,
                   size_t inner_count,
                   size_t bit_bound);

    void add_check(const virCommitment& scores,
                   const virCommitment& max_values,
                   const virCommitment& lifted_max_values,
                   const virCommitment& selector,
                   size_t outer_count,
                   size_t middle_count,
                   size_t inner_count,
                   size_t bit_bound);

    bool finalize();

private:
    cubeMiddleArgmaxParam params_;
    cubeMiddleArgmaxProver prover_;
    cubeMiddleArgmaxVerifier verifier_;
};
