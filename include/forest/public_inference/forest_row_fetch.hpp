#pragma once

#include <map>
#include <optional>
#include <string>
#include <memory>
#include <vector>

#include "cq_concatenated_table_lookup.hpp"

class forestCompressedRowComponent {
public:
    std::string key;
    Group1 commitment;
    Group2 table_g2;
    std::shared_ptr<const std::vector<Fr>> evals;
    std::shared_ptr<const std::vector<Group1>> quotient_commitments;
};

class forestCompressedRowLookupPublicTable {
public:
    std::string key;
    size_t row_count = 0;
    size_t feature_count = 0;
    std::optional<size_t> degree_bound;
    std::vector<forestCompressedRowComponent> components;
    Fr lagrange_at_zero = Fr(0);
    Group2 vanishing_g2;
    std::shared_ptr<const std::vector<Group1>> lagrange_commitments;
    std::shared_ptr<const std::vector<Group1>> a0_quotient_commitments;

    forestCompressedRowLookupPublicTable verifier_view() const
    {
        forestCompressedRowLookupPublicTable view = *this;
        for (forestCompressedRowComponent& component : view.components) {
            component.evals.reset();
            component.quotient_commitments.reset();
        }
        return view;
    }
};

// Public artifact for the committed-forest row fetch used by inference.
//
// Logical statement without leaf index:
//   tree | lower_bounds | upper_bounds | score ⊆ forest rows
//
// Optional leaf index extends the fetched row to:
//   tree | lower_bounds | upper_bounds | score | leaf_index
//
// The underlying implementation is one fixed-table CQ concatenated lookup.
class forestRowFetchPublicTable {
public:
    cqConcatenatedTableLookupPublicTable lookup_table;
    std::optional<cqTableLookupPublicTable> lower_lookup_table;
    std::optional<cqTableLookupPublicTable> upper_lookup_table;
    std::optional<cqConcatenatedTableLookupPublicTable> scalar_lookup_table;
    std::optional<forestCompressedRowLookupPublicTable> compressed_row_lookup_table;
    size_t row_count = 0;
    size_t feature_count = 0;
    bool includes_leaf_index = false;
    std::shared_ptr<const std::map<Fr, size_t>> row_position_by_leaf_index;

    forestRowFetchPublicTable verifier_view() const
    {
        forestRowFetchPublicTable view = *this;
        view.lookup_table = lookup_table.verifier_view();
        if (lower_lookup_table.has_value()) {
            view.lower_lookup_table = lower_lookup_table->verifier_view();
        }
        if (upper_lookup_table.has_value()) {
            view.upper_lookup_table = upper_lookup_table->verifier_view();
        }
        if (scalar_lookup_table.has_value()) {
            view.scalar_lookup_table = scalar_lookup_table->verifier_view();
        }
        if (compressed_row_lookup_table.has_value()) {
            view.compressed_row_lookup_table = compressed_row_lookup_table->verifier_view();
        }
        view.row_position_by_leaf_index.reset();
        return view;
    }
};

forestRowFetchPublicTable build_forest_row_fetch_public_table(
    const kzgPublicParameters& public_params,
    std::string key_prefix,
    const poly& tree_indices,
    const poly& lower_bounds,
    const poly& upper_bounds,
    const poly& scores,
    size_t row_count,
    size_t feature_count,
    std::optional<poly> leaf_indices = std::nullopt,
    std::optional<size_t> degree_bound = std::nullopt);
