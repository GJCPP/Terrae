#pragma once

#include "forest_structure.hpp"
#include "hist_agg.hpp"
#include "matrix_sum.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestSplitUpdateWitness {
public:
    virCommitment root_indices;
    virCommitment active_node_mask;
    virCommitment current_leaf_mask;
    virCommitment current_internal_mask;
    virCommitment next_leaf_mask;
    virCommitment next_internal_mask;
    virCommitment next_active_node_mask;
    virCommitment left_child_occupancy;
    virCommitment right_child_occupancy;

    virCommitment left_child_indices;
    virCommitment right_child_indices;
    virCommitment node_tree_ids;
    virCommitment selected_features;
    virCommitment selected_bins;
    virCommitment feature_selector;
    virCommitment lower_bounds;
    virCommitment upper_bounds;
    virCommitment left_lower;
    virCommitment left_upper;
    virCommitment right_lower;
    virCommitment right_upper;
    virCommitment thresholds;
};

class forestSplitUpdateCheckRecord {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t full_upper_bound = 0;
    size_t bit_bound = 0;
};

class forestSplitUpdateParam {
public:
    forestStructureProtocol* forest_structure = nullptr;
    histAggProtocol* hist_agg = nullptr;
    matrixRowSumProtocol* row_sum = nullptr;
    zeroProtocol* zero_check = nullptr;
};

// Proves one chosen-split-driven witness-side forest update step.
//
// The protocol proves:
// - the selected `(feature, bin)` split induces the supplied one-hot
//   `feature_selector` and `thresholds`,
// - the active parent rows satisfy the existing masked `forest_structure`
//   statement for the chosen child indices and child boxes,
// - `left_child_occupancy` / `right_child_occupancy` are the masked occupancy
//   histograms induced by the active parent set,
// - the next frontier equals the combined child occupancy,
// - active rows are consumed from the current leaf mask and added to the
//   internal mask,
// - and the new frontier rows become the next leaf rows.
//
// This is a one-step transition layer. It does not yet encode repeated updates
// or the global training-state layout.
class forestSplitUpdateProtocol : protected ProtocolLifecycle {
public:
    ~forestSplitUpdateProtocol() noexcept
    {
        auto_finalize("forestSplitUpdateProtocol", [this] { return finalize(); });
    }

    void setup(forestStructureProtocol& forest_structure,
               histAggProtocol& hist_agg,
               matrixRowSumProtocol& row_sum,
               zeroProtocol& zero_check)
    {
        params_.forest_structure = &forest_structure;
        params_.hist_agg = &hist_agg;
        params_.row_sum = &row_sum;
        params_.zero_check = &zero_check;
        mark_setup();
    }

    void add_check(const forestSplitUpdateWitness& witness,
                   size_t tree_count,
                   size_t node_count,
                   size_t feature_count,
                   size_t full_upper_bound,
                   size_t bit_bound);

    bool finalize();

private:
    forestSplitUpdateParam params_;
    std::vector<forestSplitUpdateWitness> witnesses_;
    std::vector<forestSplitUpdateCheckRecord> checks_;
};
