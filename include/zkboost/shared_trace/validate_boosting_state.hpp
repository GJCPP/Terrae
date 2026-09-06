#pragma once

#include "forest_training_trace_adapter.hpp"

#include <cstddef>
#include <vector>

class zkboostSharedBoostingStateSummary {
public:
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t sample_count = 0;
    size_t tree_count = 0;
    size_t cube_size = 0;
    size_t softmax_scale = 0;
};

class zkboostSharedBoostingStateView {
public:
    zkboostSharedBoostingStateSummary summary;

    std::vector<Fr> tree_scores;
    std::vector<Fr> base_labels;
    std::vector<Fr> cumulative_scores;
    std::vector<Fr> encoded_cumulative_scores;
    std::vector<Fr> softmax_shifts;
    std::vector<Fr> shift_selector;
    std::vector<Fr> lifted_softmax_shifts;
    std::vector<Fr> encoded_shifted_scores;
    std::vector<Fr> softmax_exp_values;
    std::vector<Fr> softmax_denominators;
    std::vector<Fr> lifted_softmax_denominators;
    std::vector<Fr> softmax_probabilities;
    std::vector<Fr> softmax_remainders;
    std::vector<Fr> pseudo_residuals;
    std::vector<Fr> encoded_diagonal_hessians;
    std::vector<Fr> diagonal_hessian_remainders;
    std::vector<Fr> diagonal_hessians;
};

zkboostSharedBoostingStateView build_zkboost_shared_boosting_state_view(
    const forestTrainingSemanticTrace& training);

zkboostSharedBoostingStateSummary validate_zkboost_shared_boosting_state(
    const forestTrainingSemanticTrace& training);
