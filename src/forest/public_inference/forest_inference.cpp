#include "forest_inference.hpp"

#include <stdexcept>

forestInferencePublicTable build_forest_inference_public_table(
    const kzgPublicParameters& public_params,
    std::string key_prefix,
    const poly& tree_indices,
    const poly& lower_bounds,
    const poly& upper_bounds,
    const poly& scores,
    size_t row_count,
    size_t feature_count,
    size_t input_bit_bound,
    std::optional<poly> leaf_indices,
    std::optional<size_t> degree_bound)
{
    if (input_bit_bound == 0) {
        throw std::runtime_error("build_forest_inference_public_table: input_bit_bound must be non-zero");
    }

    forestInferencePublicTable out;
    out.row_fetch_table = build_forest_row_fetch_public_table(public_params,
                                                              std::move(key_prefix) + ".step",
                                                              tree_indices,
                                                              lower_bounds,
                                                              upper_bounds,
                                                              scores,
                                                              row_count,
                                                              feature_count,
                                                              std::move(leaf_indices),
                                                              degree_bound);
    out.input_bit_bound = input_bit_bound;
    return out;
}
