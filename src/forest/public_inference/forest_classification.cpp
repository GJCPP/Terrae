#include "forest_classification.hpp"

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
    std::optional<poly> leaf_indices,
    std::optional<size_t> degree_bound)
{
    forestClassificationPublicTable out;
    out.layout_table =
        build_forest_tree_layout_public_table(public_params, key_prefix + ".layout", iteration_count, class_count);
    out.legality_table = build_forest_legality_public_table(
        tree_indices, lower_bounds, upper_bounds, row_count, feature_count, iteration_count * class_count, leaf_indices);
    out.inference_table = build_forest_inference_public_table(public_params,
                                                              key_prefix + ".inference",
                                                              tree_indices,
                                                              lower_bounds,
                                                              upper_bounds,
                                                              scores,
                                                              row_count,
                                                              feature_count,
                                                              input_bit_bound,
                                                              std::move(leaf_indices),
                                                              degree_bound);
    out.prediction_table =
        build_forest_prediction_public_table(iteration_count, class_count, score_bit_bound);
    return out;
}

forestClassificationPublicTable build_forest_classification_public_table(
    const forestTreeLayoutPublicTable& layout_table,
    const forestLegalityPublicTable& legality_table,
    const forestInferencePublicTable& inference_table,
    const forestPredictionPublicTable& prediction_table)
{
    forestClassificationPublicTable out;
    out.layout_table = layout_table;
    out.legality_table = legality_table;
    out.inference_table = inference_table;
    out.prediction_table = prediction_table;
    return out;
}
