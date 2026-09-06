#pragma once

#include "zkboost/shared_trace/validate_histograms.hpp"

#include <cstddef>
#include <vector>

class zkboostSharedSplitSummary {
public:
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t candidate_count = 0;
    size_t split_node_count = 0;
};

class zkboostSharedSplitView {
public:
    zkboostSharedSplitSummary summary;

    std::vector<Fr> left_counts;
    std::vector<Fr> right_counts;
    std::vector<Fr> candidate_masks;
    std::vector<Fr> left_gradients;
    std::vector<Fr> right_gradients;
    std::vector<Fr> left_hessians;
    std::vector<Fr> right_hessians;
    std::vector<Fr> parent_gradients;
    std::vector<Fr> parent_hessians;
    std::vector<Fr> left_gradient_squares;
    std::vector<Fr> right_gradient_squares;
    std::vector<Fr> parent_gradient_squares;
    std::vector<Fr> gain_numerators;
    std::vector<Fr> gain_denominators;
    std::vector<Fr> encoded_gain_numerators;
    std::vector<Fr> quantized_gains;
    std::vector<Fr> encoded_quantized_gains;
    std::vector<Fr> gain_remainders;
    std::vector<Fr> tie_break_scores;
    std::vector<Fr> selector;
    std::vector<Fr> selected_gains;
    std::vector<Fr> selected_tie_break_scores;
    std::vector<Fr> selected_features;
    std::vector<Fr> selected_bins;
    std::vector<Fr> split_masks;
    // Node-level stopping witnesses. `active_node_masks` marks rows with at
    // least one valid split candidate. Depths use the paper's one-based
    // convention, and `depth_limit_masks` only selects active leaves at the
    // public maximum depth.
    std::vector<Fr> active_node_masks;
    std::vector<Fr> leaf_masks;
    std::vector<Fr> node_depths;
    std::vector<Fr> depth_limit_masks;
};

zkboostSharedSplitView build_zkboost_shared_split_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms);

zkboostSharedSplitView build_zkboost_shared_split_view(
    const forestTrainingSemanticTrace& training);

zkboostSharedSplitSummary validate_zkboost_shared_splits(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms);

zkboostSharedSplitSummary validate_zkboost_shared_splits(
    const forestTrainingSemanticTrace& training);
