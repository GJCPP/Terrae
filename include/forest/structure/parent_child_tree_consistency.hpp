#pragma once

#include <vector>

#include "index_routing.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class parentChildTreeConsistencyCheckRecord {
public:
    size_t node_count = 0;
    bool uses_mask = false;
    kzgCommitment left_child_tree_ids_commitment;
    kzgCommitment right_child_tree_ids_commitment;
};

class parentChildTreeConsistencyParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    indexRoutingProtocol* index_routing = nullptr;
};

class parentChildTreeConsistencyVerifier {
public:
    void add_check(const parentChildTreeConsistencyCheckRecord& record);
    bool finalize() const;

    std::vector<parentChildTreeConsistencyCheckRecord> checks;
    const parentChildTreeConsistencyParam* params = nullptr;
};

class parentChildTreeConsistencyProver {
public:
    parentChildTreeConsistencyCheckRecord add_check(const virCommitment& left_child_indices,
                                                    const virCommitment& right_child_indices,
                                                    const virCommitment& node_tree_ids,
                                                    const virCommitment* mask,
                                                    size_t node_count);

    std::vector<parentChildTreeConsistencyCheckRecord> checks;
    const parentChildTreeConsistencyParam* params = nullptr;
};

// Proves that the left/right child chosen for each active parent node belongs to
// the same tree as the parent itself.
//
// Unmasked statement:
// - `left_child_indices[parent]` and `right_child_indices[parent]` choose the
//   corresponding child rows
// - routing `node_tree_ids` through either index vector yields exactly `node_tree_ids`
//
// Masked statement:
// - active rows route one left and one right child
// - inactive rows route through dummy row 0 and are later zero-masked out
// - on active rows, routed child tree ids equal the parent tree id
//
// Example:
//   node_tree_ids = [0, 0, 0, 1, 1, 1]
//   mask          = [1, 0, 0, 1, 0, 0]
//   left_child    chooses nodes [1, -, -, 4, -, -]
//   right_child   chooses nodes [2, -, -, 5, -, -]
//
// Then the routed left/right tree-id vectors are both [0, 0, 0, 1, 0, 0],
// which match `mask * node_tree_ids`.
class parentChildTreeConsistencyProtocol : protected ProtocolLifecycle {
public:
    ~parentChildTreeConsistencyProtocol() noexcept
    {
        auto_finalize("parentChildTreeConsistencyProtocol", [this] { return finalize(); });
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

    void add_check(const virCommitment& left_child_indices,
                   const virCommitment& right_child_indices,
                   const virCommitment& node_tree_ids,
                   size_t node_count);

    void add_masked_check(const virCommitment& left_child_indices,
                          const virCommitment& right_child_indices,
                          const virCommitment& mask,
                          const virCommitment& node_tree_ids,
                          size_t node_count);

    bool finalize();

private:
    void add_check_internal(const virCommitment& left_child_indices,
                            const virCommitment& right_child_indices,
                            const virCommitment* mask,
                            const virCommitment& node_tree_ids,
                            size_t node_count);

    parentChildTreeConsistencyParam params_;
    parentChildTreeConsistencyProver prover_;
    parentChildTreeConsistencyVerifier verifier_;
    std::vector<virCommitment> left_child_indices_;
    std::vector<virCommitment> right_child_indices_;
    std::vector<virCommitment> masks_;
    std::vector<virCommitment> node_tree_ids_;
};
