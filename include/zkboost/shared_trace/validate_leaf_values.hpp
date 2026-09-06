#pragma once

#include "zkboost/shared_trace/validate_histograms.hpp"

#include <cstddef>
#include <vector>

class zkboostSharedLeafValueSummary {
public:
    size_t node_count = 0;
    size_t leaf_node_count = 0;
    size_t histogram_width = 0;
    size_t l2_regularization = 0;
    size_t learning_rate_numerator = 1;
    size_t learning_rate_denominator = 1;
};

class zkboostSharedLeafValueView {
public:
    zkboostSharedLeafValueSummary summary;

    std::vector<Fr> leaf_gradients;
    std::vector<Fr> leaf_hessians;
    std::vector<Fr> leaf_denominators;
    std::vector<Fr> encoded_negative_leaf_gradients;
    std::vector<Fr> encoded_leaf_values;
    std::vector<Fr> leaf_remainders;
    std::vector<Fr> decoded_leaf_values;
    std::vector<Fr> leaf_masks;
    std::vector<Fr> committed_leaf_values;
};

zkboostSharedLeafValueView build_zkboost_shared_leaf_value_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms);

zkboostSharedLeafValueView build_zkboost_shared_leaf_value_view(
    const forestTrainingSemanticTrace& training);

zkboostSharedLeafValueSummary validate_zkboost_shared_leaf_values(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms);

zkboostSharedLeafValueSummary validate_zkboost_shared_leaf_values(
    const forestTrainingSemanticTrace& training);
