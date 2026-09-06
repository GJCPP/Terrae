#include "forest_box_consistency.hpp"

#include "algebra_utils.hpp"
#include "matrix_utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct forest_box_row {
    size_t tree_index = 0;
    std::vector<size_t> lower_bounds;
    std::vector<size_t> upper_bounds;
    std::optional<size_t> leaf_index;
};

bool half_open_boxes_overlap(const forest_box_row& lhs, const forest_box_row& rhs)
{
    for (size_t feature = 0; feature < lhs.lower_bounds.size(); ++feature) {
        const size_t lower = std::max(lhs.lower_bounds[feature], rhs.lower_bounds[feature]);
        const size_t upper = std::min(lhs.upper_bounds[feature], rhs.upper_bounds[feature]);
        if (lower >= upper) {
            return false;
        }
    }
    return true;
}

void validate_forest_box_shapes(const poly& tree_indices,
                                const poly& lower_bounds,
                                const poly& upper_bounds,
                                size_t row_count,
                                size_t feature_count,
                                const std::optional<poly>& leaf_indices)
{
    if (feature_count == 0 || !isPowerOfTwo(feature_count)) {
        throw std::runtime_error(
            "build_forest_box_consistency_public_table: feature_count must be a non-zero power of two");
    }

    validate_matrix_shape(row_count, 1, "build_forest_box_consistency_public_table tree");
    validate_matrix_shape(row_count, feature_count, "build_forest_box_consistency_public_table lower_bounds");
    validate_matrix_shape(row_count, feature_count, "build_forest_box_consistency_public_table upper_bounds");
    (void)tree_indices;
    (void)lower_bounds;
    (void)upper_bounds;

    if (leaf_indices.has_value()) {
        validate_matrix_shape(row_count, 1, "build_forest_box_consistency_public_table leaf_indices");
    }
}

std::vector<forest_box_row> extract_rows(const poly& tree_indices,
                                         const poly& lower_bounds,
                                         const poly& upper_bounds,
                                         size_t row_count,
                                         size_t feature_count,
                                         const std::optional<poly>& leaf_indices,
                                         std::optional<size_t> ignored_tree_index)
{
    const std::vector<Fr> tree_evals = tree_indices.evaluations_on_domain(row_count);
    const std::vector<Fr> lower_evals = lower_bounds.evaluations_on_domain(row_count * feature_count);
    const std::vector<Fr> upper_evals = upper_bounds.evaluations_on_domain(row_count * feature_count);
    const std::vector<Fr> leaf_evals =
        leaf_indices.has_value() ? leaf_indices->evaluations_on_domain(row_count) : std::vector<Fr>{};

    std::vector<forest_box_row> rows;
    rows.reserve(row_count);
    for (size_t row = 0; row < row_count; ++row) {
        forest_box_row current;
        current.tree_index = canonical_size_t_from_fr(tree_evals[row], "build_forest_box_consistency_public_table tree");
        if (ignored_tree_index.has_value() && current.tree_index == *ignored_tree_index) {
            continue;
        }
        current.lower_bounds.reserve(feature_count);
        current.upper_bounds.reserve(feature_count);
        for (size_t feature = 0; feature < feature_count; ++feature) {
            const size_t index = matrix_index(row, feature, feature_count);
            const size_t lower = canonical_size_t_from_fr(
                lower_evals[index], "build_forest_box_consistency_public_table lower bound");
            const size_t upper = canonical_size_t_from_fr(
                upper_evals[index], "build_forest_box_consistency_public_table upper bound");
            if (lower >= upper) {
                throw std::runtime_error(
                    "build_forest_box_consistency_public_table: each half-open box coordinate must satisfy lower < upper");
            }
            current.lower_bounds.push_back(lower);
            current.upper_bounds.push_back(upper);
        }
        if (leaf_indices.has_value()) {
            current.leaf_index =
                canonical_size_t_from_fr(leaf_evals[row], "build_forest_box_consistency_public_table leaf index");
        }
        rows.push_back(std::move(current));
    }
    return rows;
}

void validate_leaf_indices_unique_within_tree(const std::vector<forest_box_row>& rows)
{
    std::unordered_map<size_t, std::unordered_set<size_t>> seen_by_tree;
    for (const forest_box_row& row : rows) {
        if (!row.leaf_index.has_value()) {
            continue;
        }
        auto& seen = seen_by_tree[row.tree_index];
        if (!seen.insert(*row.leaf_index).second) {
            throw std::runtime_error(
                "build_forest_box_consistency_public_table: duplicate leaf_index within one tree");
        }
    }
}

void validate_pairwise_non_overlap_within_tree(const std::vector<forest_box_row>& rows)
{
    std::unordered_map<size_t, std::vector<const forest_box_row*>> rows_by_tree;
    rows_by_tree.reserve(rows.size());
    for (const forest_box_row& row : rows) {
        rows_by_tree[row.tree_index].push_back(&row);
    }

    for (auto& [tree_index, tree_rows] : rows_by_tree) {
        (void)tree_index;
        std::sort(tree_rows.begin(), tree_rows.end(), [](const forest_box_row* lhs, const forest_box_row* rhs) {
            if (lhs->lower_bounds[0] != rhs->lower_bounds[0]) {
                return lhs->lower_bounds[0] < rhs->lower_bounds[0];
            }
            return lhs->upper_bounds[0] < rhs->upper_bounds[0];
        });

        for (size_t i = 0; i < tree_rows.size(); ++i) {
            const forest_box_row& lhs = *tree_rows[i];
            for (size_t j = i + 1; j < tree_rows.size(); ++j) {
                const forest_box_row& rhs = *tree_rows[j];
                if (rhs.lower_bounds[0] >= lhs.upper_bounds[0]) {
                    break;
                }
                if (half_open_boxes_overlap(lhs, rhs)) {
                    throw std::runtime_error(
                        "build_forest_box_consistency_public_table: overlapping leaf boxes detected within one tree");
                }
            }
        }
    }
}

} // namespace

forestBoxConsistencyPublicTable build_forest_box_consistency_public_table(const poly& tree_indices,
                                                                         const poly& lower_bounds,
                                                                         const poly& upper_bounds,
                                                                         size_t row_count,
                                                                         size_t feature_count,
                                                                         std::optional<poly> leaf_indices,
                                                                         std::optional<size_t> ignored_tree_index)
{
    validate_forest_box_shapes(tree_indices, lower_bounds, upper_bounds, row_count, feature_count, leaf_indices);
    const std::vector<forest_box_row> rows =
        extract_rows(tree_indices, lower_bounds, upper_bounds, row_count, feature_count, leaf_indices, ignored_tree_index);

    if (leaf_indices.has_value()) {
        validate_leaf_indices_unique_within_tree(rows);
    }
    validate_pairwise_non_overlap_within_tree(rows);

    forestBoxConsistencyPublicTable out;
    out.row_count = row_count;
    out.feature_count = feature_count;
    out.includes_leaf_index = leaf_indices.has_value();
    return out;
}
