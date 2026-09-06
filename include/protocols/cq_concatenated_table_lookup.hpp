#pragma once

#include <optional>
#include <string>
#include <vector>

#include "table_lookup.hpp"

class cqConcatenatedTableLookupPublicTable {
public:
    // Offline-preprocessed CQ table for one fixed merged layout. Each logical block
    // occupies a disjoint subset of columns inside the merged matrix width.
    cqTableLookupPublicTable merged_table;
    size_t row_count = 0;
    size_t merged_column_count = 0;
    std::vector<size_t> column_counts;
    std::vector<size_t> column_offsets;

    cqConcatenatedTableLookupPublicTable verifier_view() const
    {
        cqConcatenatedTableLookupPublicTable view = *this;
        view.merged_table = merged_table.verifier_view();
        return view;
    }
};

cqConcatenatedTableLookupPublicTable build_cq_concatenated_table_lookup_public_table(
    const kzgPublicParameters& public_params,
    std::string key,
    const std::vector<poly>& tables,
    size_t row_count,
    const std::vector<size_t>& column_counts,
    std::optional<size_t> degree_bound = std::nullopt);

// Choose one merged width for all logical blocks. The current fixed-table CQ path
// uses a row-preserving placement inside a power-of-two width.
size_t cq_concatenated_table_lookup_merged_width(const std::vector<size_t>& column_counts);

std::vector<size_t> cq_concatenated_table_lookup_offsets(size_t merged_column_count,
                                                         const std::vector<size_t>& column_counts);

// Merge query blocks into the same fixed layout that was preprocessed on the table side.
// For example, blocks of widths {1, d, d, 1} become one query matrix of width W,
// where each block only populates its assigned columns.
virCommitment cq_concatenated_table_lookup_query_expression(
    const std::vector<virCommitment>& queries,
    size_t query_row_count,
    const cqConcatenatedTableLookupPublicTable& table,
    const char* context);
