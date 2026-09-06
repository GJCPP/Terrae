#pragma once

#include <vector>

#include "index_routing.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class rootBoxRangeCheckRecord {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t full_upper_bound = 0;
    kzgCommitment root_lower_commitment;
    kzgCommitment root_upper_commitment;
};

class rootBoxRangeParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    indexRoutingProtocol* index_routing = nullptr;
};

class rootBoxRangeVerifier {
public:
    void add_check(const rootBoxRangeCheckRecord& record);
    bool finalize() const;

    std::vector<rootBoxRangeCheckRecord> checks;
    const rootBoxRangeParam* params = nullptr;
};

class rootBoxRangeProver {
public:
    rootBoxRangeCheckRecord add_check(const virCommitment& root_indices,
                                      const virCommitment& lower_bounds,
                                      const virCommitment& upper_bounds,
                                      size_t tree_count,
                                      size_t node_count,
                                      size_t feature_count,
                                      size_t full_upper_bound);

    std::vector<rootBoxRangeCheckRecord> checks;
    const rootBoxRangeParam* params = nullptr;
};

// Proves that the selected root node of each tree has the full feature range.
//
// Statement:
// - route `lower_bounds` through `root_indices` and prove the selected root lower
//   box is all zero
// - route `upper_bounds` through `root_indices` and prove the selected root upper
//   box is constantly `full_upper_bound`
//
// This matches the forest-commitment condition
// `Root * N^left = 0` and `Root * N^right = (B+1) * 1`.
class rootBoxRangeProtocol : protected ProtocolLifecycle {
public:
    ~rootBoxRangeProtocol() noexcept
    {
        auto_finalize("rootBoxRangeProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero_check, indexRoutingProtocol& index_routing)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero_check;
        params_.index_routing = &index_routing;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& root_indices,
                   const virCommitment& lower_bounds,
                   const virCommitment& upper_bounds,
                   size_t tree_count,
                   size_t node_count,
                   size_t feature_count,
                   size_t full_upper_bound);

    bool finalize();

private:
    rootBoxRangeParam params_;
    rootBoxRangeProver prover_;
    rootBoxRangeVerifier verifier_;
    std::vector<virCommitment> root_indices_;
    std::vector<virCommitment> lower_bounds_;
    std::vector<virCommitment> upper_bounds_;
};
