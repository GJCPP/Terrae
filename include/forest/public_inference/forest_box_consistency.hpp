#pragma once

#include <optional>

#include "kzg.hpp"

// Public metadata for fixed committed-forest box consistency checks.
//
// The builder validates the public forest rows offline:
// - rows belonging to the same tree have pairwise non-overlapping half-open boxes
// - optional leaf indices are unique within each tree
//
// This is a public-data consistency layer for the current fixed-table inference
// path, so the verifier only needs the metadata that the check was carried out on
// the expected shape.
class forestBoxConsistencyPublicTable {
public:
    size_t row_count = 0;
    size_t feature_count = 0;
    bool includes_leaf_index = false;

    forestBoxConsistencyPublicTable verifier_view() const
    {
        return *this;
    }
};

forestBoxConsistencyPublicTable build_forest_box_consistency_public_table(const poly& tree_indices,
                                                                         const poly& lower_bounds,
                                                                         const poly& upper_bounds,
                                                                         size_t row_count,
                                                                         size_t feature_count,
                                                                         std::optional<poly> leaf_indices = std::nullopt,
                                                                         std::optional<size_t> ignored_tree_index = std::nullopt);
