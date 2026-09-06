#include "forest_row_fetch.hpp"

#include "matrix_utils.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void validate_forest_row_fetch_shape(size_t row_count,
                                     size_t feature_count,
                                     const poly& tree_indices,
                                     const poly& lower_bounds,
                                     const poly& upper_bounds,
                                     const poly& scores,
                                     const std::optional<poly>& leaf_indices)
{
    validate_matrix_shape(row_count, 1, "build_forest_row_fetch_public_table tree");
    validate_matrix_shape(row_count, feature_count, "build_forest_row_fetch_public_table lower_bounds");
    validate_matrix_shape(row_count, feature_count, "build_forest_row_fetch_public_table upper_bounds");
    validate_matrix_shape(row_count, 1, "build_forest_row_fetch_public_table scores");
    (void)tree_indices;
    (void)lower_bounds;
    (void)upper_bounds;
    (void)scores;

    if (leaf_indices.has_value()) {
        validate_matrix_shape(row_count, 1, "build_forest_row_fetch_public_table leaf_indices");
    }
}

std::vector<poly> matrix_column_polynomials(const poly& matrix, size_t row_count, size_t column_count)
{
    const std::vector<Fr> matrix_evals = matrix.evaluations_on_domain(row_count * column_count);
    std::vector<std::vector<Fr>> column_evals(column_count, std::vector<Fr>(row_count, Fr(0)));
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t column = 0; column < column_count; ++column) {
            column_evals[column][row] = matrix_evals[matrix_index(row, column, column_count)];
        }
    }

    std::vector<poly> columns;
    columns.reserve(column_count);
    for (size_t column = 0; column < column_count; ++column) {
        columns.push_back(poly_from_evals(column_evals[column]));
    }
    return columns;
}

std::shared_ptr<const std::vector<Group1>> quotient_commitments_for_row_component(
    const kzgPublicParameters& public_params,
    const poly& component,
    size_t row_count,
    const Fr& lagrange_at_zero)
{
    kzgProtocol temp_kzg;
    temp_kzg.setup(public_params);
    const std::vector<Group1> quotient_openings = temp_kzg.batch_opening_commitments(component, row_count);

    const Fr omega = get_root_of_unity(row_count);
    Fr root = Fr(1);
    auto quotient_commitments = std::make_shared<std::vector<Group1>>();
    quotient_commitments->reserve(row_count);
    for (size_t i = 0; i < row_count; ++i) {
        quotient_commitments->push_back((lagrange_at_zero * root) * quotient_openings[i]);
        Fr::mul(root, root, omega);
    }
    return quotient_commitments;
}

forestCompressedRowComponent build_compressed_row_component(const kzgPublicParameters& public_params,
                                                            std::string key,
                                                            const poly& component,
                                                            size_t row_count,
                                                            const Fr& lagrange_at_zero)
{
    forestCompressedRowComponent out;
    out.key = std::move(key);
    out.commitment = commit_g1_with_public_parameters(public_params, component);
    out.table_g2 = commit_g2_with_public_parameters(public_params, component);
    out.evals = std::make_shared<std::vector<Fr>>(component.evaluations_on_domain(row_count));
    out.quotient_commitments =
        quotient_commitments_for_row_component(public_params, component, row_count, lagrange_at_zero);
    return out;
}

forestCompressedRowLookupPublicTable build_compressed_row_lookup_public_table(
    const kzgPublicParameters& public_params,
    std::string key,
    const poly& tree_indices,
    const poly& lower_bounds,
    const poly& upper_bounds,
    const poly& scores,
    const poly& leaf_indices,
    size_t row_count,
    size_t feature_count,
    std::optional<size_t> degree_bound)
{
    forestCompressedRowLookupPublicTable out;
    out.key = std::move(key);
    out.row_count = row_count;
    out.feature_count = feature_count;
    out.degree_bound = degree_bound;
    fill_cq_lookup_domain_precompute(public_params,
                                     row_count,
                                     out.lagrange_at_zero,
                                     out.vanishing_g2,
                                     out.lagrange_commitments,
                                     out.a0_quotient_commitments);

    out.components.reserve(3 + 2 * feature_count);
    out.components.push_back(build_compressed_row_component(
        public_params, out.key + ".tree", tree_indices, row_count, out.lagrange_at_zero));
    out.components.push_back(build_compressed_row_component(
        public_params, out.key + ".score", scores, row_count, out.lagrange_at_zero));
    out.components.push_back(build_compressed_row_component(
        public_params, out.key + ".leaf", leaf_indices, row_count, out.lagrange_at_zero));

    const std::vector<poly> lower_columns = matrix_column_polynomials(lower_bounds, row_count, feature_count);
    for (size_t column = 0; column < feature_count; ++column) {
        out.components.push_back(build_compressed_row_component(public_params,
                                                                out.key + ".lower." + std::to_string(column),
                                                                lower_columns[column],
                                                                row_count,
                                                                out.lagrange_at_zero));
    }

    const std::vector<poly> upper_columns = matrix_column_polynomials(upper_bounds, row_count, feature_count);
    for (size_t column = 0; column < feature_count; ++column) {
        out.components.push_back(build_compressed_row_component(public_params,
                                                                out.key + ".upper." + std::to_string(column),
                                                                upper_columns[column],
                                                                row_count,
                                                                out.lagrange_at_zero));
    }
    return out;
}

} // namespace

forestRowFetchPublicTable build_forest_row_fetch_public_table(
    const kzgPublicParameters& public_params,
    std::string key_prefix,
    const poly& tree_indices,
    const poly& lower_bounds,
    const poly& upper_bounds,
    const poly& scores,
    size_t row_count,
    size_t feature_count,
    std::optional<poly> leaf_indices,
    std::optional<size_t> degree_bound)
{
    if (feature_count == 0 || !isPowerOfTwo(feature_count)) {
        throw std::runtime_error("build_forest_row_fetch_public_table: feature_count must be a non-zero power of two");
    }

    validate_forest_row_fetch_shape(
        row_count, feature_count, tree_indices, lower_bounds, upper_bounds, scores, leaf_indices);

    forestRowFetchPublicTable out;
    out.row_count = row_count;
    out.feature_count = feature_count;
    out.includes_leaf_index = leaf_indices.has_value();

    std::vector<poly> tables = {tree_indices, lower_bounds, upper_bounds, scores};
    std::vector<size_t> column_counts = {1, feature_count, feature_count, 1};
    if (leaf_indices.has_value()) {
        tables.push_back(*leaf_indices);
        column_counts.push_back(1);
    }

    out.lookup_table = build_cq_concatenated_table_lookup_public_table(
        public_params, key_prefix + ".fetch", tables, row_count, column_counts, degree_bound);
    if (leaf_indices.has_value()) {
        out.lower_lookup_table = build_cq_table_lookup_public_table(
            public_params, key_prefix + ".fetch.lower", lower_bounds, row_count, feature_count, degree_bound);
        out.upper_lookup_table = build_cq_table_lookup_public_table(
            public_params, key_prefix + ".fetch.upper", upper_bounds, row_count, feature_count, degree_bound);
        out.scalar_lookup_table = build_cq_concatenated_table_lookup_public_table(
            public_params,
            key_prefix + ".fetch.scalar",
            std::vector<poly>{tree_indices, scores, *leaf_indices},
            row_count,
            std::vector<size_t>{1, 1, 1},
            degree_bound);
        if (row_count >= 2) {
            out.compressed_row_lookup_table =
                build_compressed_row_lookup_public_table(public_params,
                                                         key_prefix + ".fetch.compressed_row",
                                                         tree_indices,
                                                         lower_bounds,
                                                         upper_bounds,
                                                         scores,
                                                         *leaf_indices,
                                                         row_count,
                                                         feature_count,
                                                         degree_bound);
        }

        const std::vector<Fr> leaf_index_evals = leaf_indices->evaluations_on_domain(row_count);
        auto row_position_by_leaf_index = std::make_shared<std::map<Fr, size_t>>();
        for (size_t row = 0; row < row_count; ++row) {
            if (!row_position_by_leaf_index->emplace(leaf_index_evals[row], row).second) {
                throw std::runtime_error(
                    "build_forest_row_fetch_public_table: leaf indices must be unique when exported");
            }
        }
        out.row_position_by_leaf_index = row_position_by_leaf_index;
    }
    return out;
}
