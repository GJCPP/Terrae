#include "forest_legality.hpp"

#include "algebra_utils.hpp"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_tree_indices(const poly& tree_indices, size_t row_count, size_t tree_count)
{
    if (tree_count == 0) {
        throw std::runtime_error("build_forest_legality_public_table: tree_count must be non-zero");
    }

    const std::vector<Fr> tree_evals = tree_indices.evaluations_on_domain(row_count);
    if (tree_evals.size() != row_count) {
        throw std::runtime_error("build_forest_legality_public_table: failed to evaluate tree indices");
    }

    std::vector<bool> seen(tree_count, false);
    for (size_t row = 0; row < row_count; ++row) {
        const size_t tree_index =
            canonical_size_t_from_fr(tree_evals[row], "build_forest_legality_public_table tree index");
        if (tree_index == 0) {
            continue;
        }
        if (tree_index > tree_count) {
            throw std::runtime_error("build_forest_legality_public_table: tree index out of range");
        }
        seen[tree_index - 1] = true;
    }

    for (size_t tree = 0; tree < tree_count; ++tree) {
        if (!seen[tree]) {
            throw std::runtime_error("build_forest_legality_public_table: each tree must contribute at least one row");
        }
    }
}

} // namespace

forestLegalityPublicTable build_forest_legality_public_table(const poly& tree_indices,
                                                             const poly& lower_bounds,
                                                             const poly& upper_bounds,
                                                             size_t row_count,
                                                             size_t feature_count,
                                                             size_t tree_count,
                                                             std::optional<poly> leaf_indices)
{
    validate_tree_indices(tree_indices, row_count, tree_count);

    forestLegalityPublicTable out;
    out.box_consistency_table = build_forest_box_consistency_public_table(
        tree_indices, lower_bounds, upper_bounds, row_count, feature_count, std::move(leaf_indices), 0);
    out.tree_count = tree_count;
    return out;
}
