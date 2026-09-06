#pragma once

#include "zkboost/common/records.hpp"

#include <cstdint>
#include <vector>

class zkboostTrainingTrace {
public:
    zkboostBinaryTreeShape tree_shape;
    zkboostTrainingHyperparameters hyperparameters;
    zkboostFixedPointParameters fixed_point;

    size_t feature_count = 0;
    size_t query_count = 0;

    // Feature matrix is flattened row-major after equal-width pre-binning.
    // `raw_inputs` stores the pre-binning feature matrix; `binned_inputs`
    // remains the retained compatibility lane and should match
    // `prebin_bin_ids` when legacy PreBin witnesses are present.
    std::vector<std::int64_t> raw_inputs;
    std::vector<std::int64_t> labels;
    std::vector<std::int64_t> binned_inputs;
    std::int64_t base_score = 0;

    // Optional legacy PreBin witness lanes.
    std::vector<std::int64_t> prebin_feature_mins;
    std::vector<std::int64_t> prebin_feature_maxs;
    std::vector<std::int64_t> prebin_deltas;
    std::vector<std::int64_t> prebin_edges;
    std::vector<std::int64_t> prebin_bin_ids;
    size_t prebin_dummy_feature = 0;
    size_t prebin_dummy_bin = 0;
    std::int64_t prebin_dummy_threshold = 0;

    // Canonical full-tree representation with dummy-node pruning.
    std::vector<std::int64_t> root_indices;
    std::vector<std::int64_t> split_node_mask;
    std::vector<std::int64_t> left_child_indices;
    std::vector<std::int64_t> right_child_indices;
    std::vector<std::int64_t> node_split_features;
    std::vector<std::int64_t> node_threshold_bins;
    std::vector<std::int64_t> node_dummy_mask;
    std::vector<std::int64_t> node_leaf_mask;
    std::vector<std::int64_t> leaf_weights;

    // Flattened per-(tree, query) reached-leaf witness lane.
    std::vector<std::int64_t> leaf_indices;

    // Flattened per-(tree, query, step) path witness.
    // `path_node_indices` stores `max_depth + 1` node indices per path,
    // starting at the root and ending at the reached leaf. Once a path reaches
    // a leaf early, remaining entries must repeat that leaf.
    std::vector<std::int64_t> path_node_indices;
    // `path_decisions` stores `max_depth` branch bits per path, where
    // 0 = left and 1 = right. Once a path reaches a leaf early, remaining
    // entries must be 0.
    std::vector<std::int64_t> path_decisions;

    // Per-round numeric witness lanes used by ValidateLogit-style checking.
    // All are flattened as [round][query].
    std::vector<std::int64_t> round_start_scores;
    std::vector<std::int64_t> round_probabilities;
    std::vector<std::int64_t> round_gradients;
    std::vector<std::int64_t> round_hessians;

    // Flattened per-(tree, node, feature, bin) histogram lanes used by
    // InitHists-style checking.
    std::vector<std::int64_t> node_hist_counts;
    std::vector<std::int64_t> node_hist_gradients;
    std::vector<std::int64_t> node_hist_hessians;
};
