#pragma once

#include <optional>
#include <string>

// Public metadata for the score-aggregation / class-selection half of quantized
// forest inference.
//
// Expected score layout:
// - query scores are flattened in (iteration, class, sample) order
// - equivalently, they form a T x (K * b) matrix
//
// The helper then proves:
// 1. class_scores is the column-wise sum across the T iterations, and
// 2. predicted_scores is the per-sample maximum across the K classes.
//
// This layer intentionally starts from the fetched score witness. The row fetch
// and box-membership part remains in forestInferencePublicTable / ForestInferenceContext.
class forestPredictionPublicTable {
public:
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t score_bit_bound = 0;

    forestPredictionPublicTable verifier_view() const
    {
        return *this;
    }
};

forestPredictionPublicTable build_forest_prediction_public_table(
    size_t iteration_count,
    size_t class_count,
    size_t score_bit_bound);
