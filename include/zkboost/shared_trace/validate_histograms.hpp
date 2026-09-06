#pragma once

#include "zkboost/shared_trace/validate_boosting_state.hpp"

#include <cstddef>
#include <vector>

class zkboostSharedHistogramSummary {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t sample_count = 0;
    size_t query_row_count = 0;
    size_t histogram_width = 0;
    size_t histogram_size = 0;
    size_t channel_count = 0;
};

class zkboostSharedHistogramView {
public:
    zkboostSharedHistogramSummary summary;

    std::vector<Fr> query_counts;
    std::vector<Fr> query_pseudo_residuals;
    std::vector<Fr> query_diagonal_hessians;
    std::vector<Fr> lifted_leaf_indices;
    std::vector<Fr> flattened_bucket_indices;

    // Channel order is count, pseudo-residual, diagonal-Hessian.
    std::vector<std::vector<Fr>> leaf_histograms;
    std::vector<std::vector<Fr>> full_histograms;
    std::vector<std::vector<Fr>> left_statistics;
    std::vector<std::vector<Fr>> right_statistics;
};

zkboostSharedHistogramView build_zkboost_shared_histogram_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& boosting);

zkboostSharedHistogramView build_zkboost_shared_histogram_view(
    const forestTrainingSemanticTrace& training);

zkboostSharedHistogramSummary validate_zkboost_shared_histograms(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& boosting);

zkboostSharedHistogramSummary validate_zkboost_shared_histograms(
    const forestTrainingSemanticTrace& training);
