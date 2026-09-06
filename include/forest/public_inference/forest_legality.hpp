#pragma once

#include <optional>

#include "forest_box_consistency.hpp"
#include "kzg.hpp"

// Public artifact for the currently supported "legal forest" checks over a fixed
// public leaf table.
//
// The builder validates:
// - same-tree leaf boxes are pairwise non-overlapping,
// - optional leaf indices are unique within each tree,
// - every tree index lies in [0, tree_count), and
// - every tree in [0, tree_count) contributes at least one row.
//
// This is intentionally the first reusable legality layer for fixed public
// forests. A fuller witness-side training protocol will still need additional
// routing / parent-child / split-consistency reductions.
class forestLegalityPublicTable {
public:
    forestBoxConsistencyPublicTable box_consistency_table;
    size_t tree_count = 0;

    forestLegalityPublicTable verifier_view() const
    {
        forestLegalityPublicTable view = *this;
        view.box_consistency_table = box_consistency_table.verifier_view();
        return view;
    }
};

forestLegalityPublicTable build_forest_legality_public_table(const poly& tree_indices,
                                                             const poly& lower_bounds,
                                                             const poly& upper_bounds,
                                                             size_t row_count,
                                                             size_t feature_count,
                                                             size_t tree_count,
                                                             std::optional<poly> leaf_indices = std::nullopt);
