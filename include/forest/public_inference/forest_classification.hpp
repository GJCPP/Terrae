#pragma once

#include <optional>
#include <string>

#include "forest_box_consistency.hpp"
#include "forest_inference.hpp"
#include "forest_legality.hpp"
#include "forest_prediction.hpp"
#include "forest_tree_layout.hpp"

// Public artifact for the current end-to-end committed-forest classification
// workflow:
// 1. fetch `tree | lower_bounds | upper_bounds | score | optional leaf_index`,
// 2. prove the lifted input lies inside the fetched box, and
// 3. aggregate fetched scores into final class scores plus per-sample argmax.
//
// The workflow deliberately reuses the existing `forest_inference` and
// `forest_prediction` layers so the runtime can share one fetched score witness
// across both halves of the proof.
class forestClassificationPublicTable {
public:
    forestTreeLayoutPublicTable layout_table;
    forestLegalityPublicTable legality_table;
    forestInferencePublicTable inference_table;
    forestPredictionPublicTable prediction_table;

    forestClassificationPublicTable verifier_view() const
    {
        forestClassificationPublicTable view = *this;
        view.layout_table = layout_table.verifier_view();
        view.legality_table = legality_table.verifier_view();
        view.inference_table = inference_table.verifier_view();
        view.prediction_table = prediction_table.verifier_view();
        return view;
    }
};

forestClassificationPublicTable build_forest_classification_public_table(
    const kzgPublicParameters& public_params,
    std::string key_prefix,
    const poly& tree_indices,
    const poly& lower_bounds,
    const poly& upper_bounds,
    const poly& scores,
    size_t row_count,
    size_t feature_count,
    size_t input_bit_bound,
    size_t iteration_count,
    size_t class_count,
    size_t score_bit_bound,
    std::optional<poly> leaf_indices = std::nullopt,
    std::optional<size_t> degree_bound = std::nullopt);

forestClassificationPublicTable build_forest_classification_public_table(
    const forestTreeLayoutPublicTable& layout_table,
    const forestLegalityPublicTable& legality_table,
    const forestInferencePublicTable& inference_table,
    const forestPredictionPublicTable& prediction_table);
