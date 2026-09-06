#pragma once

#include <vector>

#include "hist_agg.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class nodeRoleConsistencyCheckRecord {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    bool uses_mask = false;
    kzgCommitment root_occupancy_commitment;
    kzgCommitment left_child_occupancy_commitment;
    kzgCommitment right_child_occupancy_commitment;
    kzgCommitment total_occupancy_commitment;
};

class nodeRoleConsistencyParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    histAggProtocol* hist_agg = nullptr;
};

class nodeRoleConsistencyVerifier {
public:
    void add_check(const nodeRoleConsistencyCheckRecord& record);
    bool finalize() const;

    std::vector<nodeRoleConsistencyCheckRecord> checks;
    const nodeRoleConsistencyParam* params = nullptr;
};

class nodeRoleConsistencyProver {
public:
    nodeRoleConsistencyCheckRecord add_check(const virCommitment& root_indices,
                                            const virCommitment& node_leaf_mask,
                                            const virCommitment& left_child_indices,
                                            const virCommitment& right_child_indices,
                                            const virCommitment* parent_mask,
                                            size_t tree_count,
                                            size_t node_count);

    std::vector<nodeRoleConsistencyCheckRecord> checks;
    const nodeRoleConsistencyParam* params = nullptr;
};

// Proves that each node participates in at most one incoming structural role:
// either as a root, as a left child, or as a right child.
//
// Statement:
// - `root_indices[tree]` picks one root per tree
// - `left_child_indices[parent]` / `right_child_indices[parent]` pick child rows
// - the combined occupancy histogram
//     `root_occ + left_occ + right_occ`
//   is boolean, so no node can:
//   - be the root of two trees,
//   - be reused as a child by two parents,
//   - or be both a root and a child
//
// This is a thin uniqueness layer; it does not by itself prove tree membership
// or box inheritance, which are handled by sibling structure protocols.
class nodeRoleConsistencyProtocol : protected ProtocolLifecycle {
public:
    ~nodeRoleConsistencyProtocol() noexcept
    {
        auto_finalize("nodeRoleConsistencyProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero_check, histAggProtocol& hist_agg)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero_check;
        params_.hist_agg = &hist_agg;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& root_indices,
                   const virCommitment& node_leaf_mask,
                   const virCommitment& left_child_indices,
                   const virCommitment& right_child_indices,
                   size_t tree_count,
                   size_t node_count);

    void add_masked_check(const virCommitment& root_indices,
                          const virCommitment& node_leaf_mask,
                          const virCommitment& left_child_indices,
                          const virCommitment& right_child_indices,
                          const virCommitment& parent_mask,
                          size_t tree_count,
                          size_t node_count);

    bool finalize();

private:
    void add_check_internal(const virCommitment& root_indices,
                            const virCommitment& node_leaf_mask,
                            const virCommitment& left_child_indices,
                            const virCommitment& right_child_indices,
                            const virCommitment* parent_mask,
                            size_t tree_count,
                            size_t node_count);

    nodeRoleConsistencyParam params_;
    nodeRoleConsistencyProver prover_;
    nodeRoleConsistencyVerifier verifier_;
    std::vector<virCommitment> root_indices_;
    std::vector<virCommitment> node_leaf_masks_;
    std::vector<virCommitment> left_child_indices_;
    std::vector<virCommitment> right_child_indices_;
    std::vector<virCommitment> parent_masks_;
};
