#pragma once

#include <map>
#include <vector>

#include "hist_agg.hpp"
#include "index_routing.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "profiling/soundness_counter.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class rootSelectorConsistencyCheckRecord {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    kzgCommitment selected_tree_ids_commitment;
};

class rootSelectorConsistencyParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    histAggProtocol* hist_agg = nullptr;
    indexRoutingProtocol* index_routing = nullptr;
    terraeSoundnessCounter* soundness_counter = nullptr;
};

class rootSelectorConsistencyVerifier {
public:
    void add_check(const rootSelectorConsistencyCheckRecord& record);
    bool finalize() const;

    std::vector<rootSelectorConsistencyCheckRecord> checks;
    const rootSelectorConsistencyParam* params = nullptr;
};

class rootSelectorConsistencyProver {
public:
    rootSelectorConsistencyCheckRecord add_check(const virCommitment& root_indices,
                                                const virCommitment& node_tree_ids,
                                                size_t tree_count,
                                                size_t node_count);

    std::vector<rootSelectorConsistencyCheckRecord> checks;
    const rootSelectorConsistencyParam* params = nullptr;
};

// Proves that a root-index vector consistently chooses one root per tree.
//
// Statement:
// - `root_indices[tree]` is the chosen root node for that tree
// - the canonical routed tree ids are distinct, so one node cannot be selected
//   as the root of two trees
// - routing `node_tree_ids` through `root_indices` yields the canonical tree-id
//   vector `[0, 1, ..., tree_count - 1]`
//
// Example:
//   root_indices = [1, 3]
//   node_tree_ids = [0, 0, 1, 1]
//
// Then tree 0 chooses node 1, tree 1 chooses node 3, and the selected nodes
// indeed belong to trees 0 and 1 respectively.
class rootSelectorConsistencyProtocol : protected ProtocolLifecycle {
public:
    ~rootSelectorConsistencyProtocol() noexcept
    {
        auto_finalize("rootSelectorConsistencyProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero_check,
               histAggProtocol& hist_agg,
               indexRoutingProtocol& index_routing)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero_check;
        params_.hist_agg = &hist_agg;
        params_.index_routing = &index_routing;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void set_soundness_counter(terraeSoundnessCounter* counter) { params_.soundness_counter = counter; }

    void add_check(const virCommitment& root_indices,
                   const virCommitment& node_tree_ids,
                   size_t tree_count,
                   size_t node_count);

    void register_expected_tree_ids_commitment(size_t tree_count, const Group1& commitment);

    bool finalize();

private:
    rootSelectorConsistencyParam params_;
    rootSelectorConsistencyProver prover_;
    rootSelectorConsistencyVerifier verifier_;
    std::vector<virCommitment> root_indices_;
    std::vector<virCommitment> node_tree_ids_;
    std::map<size_t, Group1> expected_tree_ids_commitments_;
};
