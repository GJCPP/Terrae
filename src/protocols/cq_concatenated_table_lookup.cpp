#include "cq_concatenated_table_lookup.hpp"

#include "lagrange_utils.hpp"
#include "matrix_utils.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

bool cq_concatenated_lookup_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_CQ_CONCATENATED_LOOKUP");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void cq_concatenated_lookup_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = cq_concatenated_lookup_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "cq_concatenated_lookup_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("cq_concatenated_lookup_profile", label, "", elapsed.count(), "ok");
}

void validate_cq_concatenated_shape(size_t row_count,
                                    const std::vector<poly>& tables,
                                    const std::vector<size_t>& column_counts,
                                    const char* context)
{
    if (tables.empty()) {
        throw std::runtime_error(std::string(context) + ": at least one table is required");
    }
    if (tables.size() != column_counts.size()) {
        throw std::runtime_error(std::string(context) + ": table/count size mismatch");
    }

    for (size_t i = 0; i < tables.size(); ++i) {
        validate_matrix_shape(row_count, column_counts[i], context);
        const size_t expected_domain = row_count * column_counts[i];
        if (tables[i].evaluations_on_domain(expected_domain).size() != expected_domain) {
            throw std::runtime_error(std::string(context) + ": failed to evaluate source table");
        }
    }
}

void validate_query_shape(size_t row_count,
                          const std::vector<virCommitment>& queries,
                          const cqConcatenatedTableLookupPublicTable& table,
                          const char* context)
{
    if (queries.empty()) {
        throw std::runtime_error(std::string(context) + ": at least one query block is required");
    }
    if (queries.size() != table.column_counts.size()) {
        throw std::runtime_error(std::string(context) + ": query/count size mismatch");
    }

    for (size_t i = 0; i < queries.size(); ++i) {
        validate_matrix_shape(row_count, table.column_counts[i], context);
    }
}

size_t next_power_of_two(size_t value)
{
    if (value == 0) {
        throw std::runtime_error("cqConcatenatedTableLookup: width must be non-zero");
    }

    size_t out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

virCommitment placement_mask_expression(size_t row_count,
                                        size_t merged_column_count,
                                        size_t column_count,
                                        size_t offset)
{
    if (merged_column_count == 0 || column_count == 0 || merged_column_count % column_count != 0) {
        throw std::runtime_error("cqConcatenatedTableLookup: invalid placement mask shape");
    }
    const size_t stride = merged_column_count / column_count;
    if (offset >= stride) {
        throw std::runtime_error("cqConcatenatedTableLookup: placement offset out of range");
    }
    return x_pow(subgroup_lagrange_expression(stride, offset), row_count * column_count);
}

Fr placement_shift(size_t domain_size, size_t offset)
{
    const Fr inverse_root = inverse_root_of_unity(domain_size);
    return pow_fr(inverse_root, offset);
}

poly merged_table_poly(const std::vector<poly>& tables,
                       size_t row_count,
                       size_t merged_column_count,
                       const std::vector<size_t>& column_counts,
                       const std::vector<size_t>& column_offsets)
{
    // Materialize one fixed merged matrix so CQ preprocessing can be done once.
    // Example: widths {1, 2, 1} may be placed into width 4 as:
    // [tree, box0, score, box1].
    std::vector<Fr> merged_evals(row_count * merged_column_count, Fr(0));

    for (size_t block_index = 0; block_index < tables.size(); ++block_index) {
        const size_t column_count = column_counts[block_index];
        const size_t stride = merged_column_count / column_count;
        const size_t offset = column_offsets[block_index];
        const std::vector<Fr> table_evals = tables[block_index].evaluations_on_domain(row_count * column_count);

        for (size_t row = 0; row < row_count; ++row) {
            for (size_t column = 0; column < column_count; ++column) {
                merged_evals[matrix_index(row, offset + column * stride, merged_column_count)] =
                    table_evals[matrix_index(row, column, column_count)];
            }
        }
    }

    return poly_from_evals(merged_evals);
}

} // namespace

size_t cq_concatenated_table_lookup_merged_width(const std::vector<size_t>& column_counts)
{
    if (column_counts.empty()) {
        throw std::runtime_error("cqConcatenatedTableLookup: at least one column count is required");
    }

    size_t total_width = 0;
    for (size_t column_count : column_counts) {
        if (!isPowerOfTwo(column_count) || column_count == 0) {
            throw std::runtime_error("cqConcatenatedTableLookup: column counts must be non-zero powers of two");
        }
        total_width += column_count;
    }
    return next_power_of_two(total_width);
}

std::vector<size_t> cq_concatenated_table_lookup_offsets(size_t merged_column_count,
                                                         const std::vector<size_t>& column_counts)
{
    std::vector<size_t> order(column_counts.size(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        return column_counts[left] > column_counts[right];
    });

    std::vector<bool> occupied(merged_column_count, false);
    std::vector<size_t> offsets(column_counts.size(), 0);

    for (size_t index : order) {
        const size_t column_count = column_counts[index];
        const size_t stride = merged_column_count / column_count;
        bool placed = false;

        // Greedily reserve one full arithmetic progression for each block so every
        // source column keeps the same row-major ordering after lifting to width W.
        for (size_t offset = 0; offset < stride; ++offset) {
            bool conflict = false;
            for (size_t column = 0; column < column_count; ++column) {
                if (occupied[offset + column * stride]) {
                    conflict = true;
                    break;
                }
            }
            if (conflict) {
                continue;
            }

            offsets[index] = offset;
            for (size_t column = 0; column < column_count; ++column) {
                occupied[offset + column * stride] = true;
            }
            placed = true;
            break;
        }

        if (!placed) {
            throw std::runtime_error("cqConcatenatedTableLookup: failed to place all blocks into merged width");
        }
    }

    return offsets;
}

cqConcatenatedTableLookupPublicTable build_cq_concatenated_table_lookup_public_table(
    const kzgPublicParameters& public_params,
    std::string key,
    const std::vector<poly>& tables,
    size_t row_count,
    const std::vector<size_t>& column_counts,
    std::optional<size_t> degree_bound)
{
    validate_cq_concatenated_shape(row_count, tables, column_counts, "build_cq_concatenated_table_lookup_public_table");

    cqConcatenatedTableLookupPublicTable out;
    out.row_count = row_count;
    out.column_counts = column_counts;
    out.merged_column_count = cq_concatenated_table_lookup_merged_width(column_counts);
    out.column_offsets = cq_concatenated_table_lookup_offsets(out.merged_column_count, column_counts);

    const poly merged = merged_table_poly(tables, row_count, out.merged_column_count, column_counts, out.column_offsets);
    out.merged_table = build_cq_table_lookup_public_table(
        public_params, std::move(key), merged, row_count, out.merged_column_count, degree_bound);
    return out;
}

virCommitment cq_concatenated_table_lookup_query_expression(
    const std::vector<virCommitment>& queries,
    size_t query_row_count,
    const cqConcatenatedTableLookupPublicTable& table,
    const char* context)
{
    cq_concatenated_lookup_profile_scope("query_expression.validate", [&] {
        validate_query_shape(query_row_count, queries, table, context);
    });

    virCommitment merged(Fr(0));
    const size_t merged_domain_size = query_row_count * table.merged_column_count;
    cq_concatenated_lookup_profile_scope("query_expression.block_loop", [&] {
        for (size_t i = 0; i < queries.size(); ++i) {
            const size_t column_count = table.column_counts[i];
            const size_t offset = table.column_offsets[i];
            virCommitment mask;
            cq_concatenated_lookup_profile_scope("query_expression.mask", [&] {
                mask = placement_mask_expression(query_row_count, table.merged_column_count, column_count, offset);
            });
            virCommitment shifted;
            cq_concatenated_lookup_profile_scope("query_expression.shift", [&] {
                shifted = x_scale(queries[i], placement_shift(merged_domain_size, offset));
            });
            cq_concatenated_lookup_profile_scope("query_expression.accumulate", [&] { merged = merged + mask * shifted; });
        }
    });
    return merged;
}
