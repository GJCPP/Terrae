#pragma once

#include <optional>
#include <string>

#include "forest_row_fetch.hpp"

// Public artifact for the first committed-forest inference step:
// - fetch one forest row by tree index and box/score fields
// - prove the lifted input lies inside the fetched half-open box
//
// This layer intentionally stops before score reshaping / aggregation.

class forestInferencePublicTable {
public:
    forestRowFetchPublicTable row_fetch_table;
    size_t input_bit_bound = 0;

    forestInferencePublicTable verifier_view() const
    {
        forestInferencePublicTable view = *this;
        view.row_fetch_table = row_fetch_table.verifier_view();
        return view;
    }
};

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
    std::optional<poly> leaf_indices = std::nullopt,
    std::optional<size_t> degree_bound = std::nullopt);
