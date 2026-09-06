#pragma once

#include <vector>

#include "index_routing.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestDepthConsistencyCheckRecord {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    kzgCommitment root_depths_commitment;
    kzgCommitment left_child_depths_commitment;
    kzgCommitment right_child_depths_commitment;
};

class forestDepthConsistencyParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    indexRoutingProtocol* index_routing = nullptr;
};

class forestDepthConsistencyVerifier {
public:
    void add_check(const forestDepthConsistencyCheckRecord& record);
    bool finalize() const;

    std::vector<forestDepthConsistencyCheckRecord> checks;
    const forestDepthConsistencyParam* params = nullptr;
};

class forestDepthConsistencyProver {
public:
    forestDepthConsistencyCheckRecord add_check(const virCommitment& root_indices,
                                                const virCommitment& parent_mask,
                                                const virCommitment& left_child_indices,
                                                const virCommitment& right_child_indices,
                                                const virCommitment& depths,
                                                size_t tree_count,
                                                size_t node_count);

    std::vector<forestDepthConsistencyCheckRecord> checks;
    const forestDepthConsistencyParam* params = nullptr;
};

// Proves that one private depth witness is compatible with the committed forest:
// - every selected root has depth 1
// - every active split parent routes left/right children whose depths are
//   exactly parent_depth + 1
//
// Together with exact incoming-role coverage, this rules out disconnected
// cycles while keeping depth as a private witness column.
class forestDepthConsistencyProtocol : protected ProtocolLifecycle {
public:
    ~forestDepthConsistencyProtocol() noexcept
    {
        auto_finalize("forestDepthConsistencyProtocol", [this] { return finalize(); });
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
                   const virCommitment& parent_mask,
                   const virCommitment& left_child_indices,
                   const virCommitment& right_child_indices,
                   const virCommitment& depths,
                   size_t tree_count,
                   size_t node_count);

    bool finalize();

private:
    forestDepthConsistencyParam params_;
    forestDepthConsistencyProver prover_;
    forestDepthConsistencyVerifier verifier_;
    std::vector<virCommitment> root_indices_;
    std::vector<virCommitment> parent_masks_;
    std::vector<virCommitment> left_child_indices_;
    std::vector<virCommitment> right_child_indices_;
    std::vector<virCommitment> depths_;
};
