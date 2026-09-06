#include "matrix_utils.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

void validate_matrix_shape(size_t row_count, size_t column_count, const char* context)
{
    if (!isPowerOfTwo(row_count) || !isPowerOfTwo(column_count) || row_count == 0 || column_count == 0) {
        throw std::runtime_error(std::string(context) + ": row_count and column_count must be non-zero powers of two");
    }
}

size_t matrix_index(size_t row, size_t column, size_t column_count)
{
    return row * column_count + column;
}

Fr field_from_size_t(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

Fr inverse_field_from_size_t(size_t value)
{
    Fr inverse;
    Fr::inv(inverse, field_from_size_t(value));
    return inverse;
}

Fr subgroup_generator(size_t domain_size)
{
    if (domain_size == 0 || !isPowerOfTwo(domain_size)) {
        throw std::runtime_error("subgroup_generator: domain_size must be a non-zero power of two");
    }
    if (domain_size == 1) {
        return Fr(1);
    }
    return get_root_of_unity(domain_size);
}

Fr subgroup_label(size_t domain_size, size_t index)
{
    return pow_fr(subgroup_generator(domain_size), index);
}

virCommitment scale_by_constant(const virCommitment& commitment, const Fr& scalar)
{
    return commitment * virCommitment(scalar);
}

namespace {

std::vector<size_t> matrix_row_positions(size_t column_count, const std::vector<size_t>& rows)
{
    std::vector<size_t> positions;
    positions.reserve(rows.size() * column_count);
    for (size_t row : rows) {
        for (size_t column = 0; column < column_count; ++column) {
            positions.push_back(matrix_index(row, column, column_count));
        }
    }
    return positions;
}

std::string matrix_row_subset_selector_key(size_t row_count,
                                           size_t column_count,
                                           const std::vector<size_t>& rows)
{
    std::string key = std::to_string(row_count) + "x" + std::to_string(column_count) + ":rows";
    for (size_t row : rows) {
        key += ":" + std::to_string(row);
    }
    return key;
}

} // namespace

virCommitment matrix_row_selector_expression(size_t row_count, size_t column_count, size_t row)
{
    return matrix_row_subset_selector_expression(row_count, column_count, std::vector<size_t>{row});
}

virCommitment matrix_row_subset_selector_expression(size_t row_count,
                                                    size_t column_count,
                                                    const std::vector<size_t>& rows)
{
    validate_matrix_shape(row_count, column_count, "matrix_row_subset_selector_expression");
    for (size_t row : rows) {
        if (row >= row_count) {
            throw std::runtime_error("matrix_row_subset_selector_expression: row out of range");
        }
    }

    return subgroup_lagrange_selector_expression(row_count * column_count,
                                                 matrix_row_positions(column_count, rows));
}

virCommitment matrix_column_index_expression(size_t row_count, size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "matrix_column_index_expression");

    std::vector<std::pair<size_t, Fr>> terms;
    terms.reserve(row_count * column_count);
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t column = 0; column < column_count; ++column) {
            terms.emplace_back(matrix_index(row, column, column_count), field_from_size_t(column));
        }
    }
    return subgroup_lagrange_linear_combination_expression(row_count * column_count, terms);
}

const poly& matrix_row_selector_polynomial(size_t row_count, size_t column_count, size_t row)
{
    return matrix_row_subset_selector_polynomial(row_count, column_count, std::vector<size_t>{row});
}

const poly& matrix_row_subset_selector_polynomial(size_t row_count,
                                                  size_t column_count,
                                                  const std::vector<size_t>& rows)
{
    validate_matrix_shape(row_count, column_count, "matrix_row_subset_selector_polynomial");
    for (size_t row : rows) {
        if (row >= row_count) {
            throw std::runtime_error("matrix_row_subset_selector_polynomial: row out of range");
        }
    }

    static std::map<std::string, poly> cache;
    const std::string key = matrix_row_subset_selector_key(row_count, column_count, rows);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    return cache.emplace(key,
                         subgroup_lagrange_selector_polynomial(row_count * column_count,
                                                               matrix_row_positions(column_count, rows)))
        .first->second;
}

namespace {

std::string matrix_row_selector_cache_key(const kzgProtocol& kzg,
                                          size_t row_count,
                                          size_t column_count,
                                          const std::vector<size_t>& rows)
{
    const std::string srs_id = kzg.g1_power(1).value().getStr(16);
    return srs_id + ":matrix_row_subset_selector:" +
           matrix_row_subset_selector_key(row_count, column_count, rows);
}

void cache_matrix_row_selector_group(const kzgProtocol& kzg,
                                     size_t row_count,
                                     size_t column_count,
                                     const std::vector<size_t>& rows,
                                     const Group1& commitment)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = matrix_row_selector_cache_key(kzg, row_count, column_count, rows);
    const auto it = cache.find(key);
    if (it == cache.end()) {
        cache.emplace(key, commitment);
        return;
    }
    if (it->second != commitment) {
        throw std::runtime_error("matrix_row_subset_selector_commitment: inconsistent cached selector commitment");
    }
}

Group1 preprocessed_matrix_row_selector_group(kzgProtocol& kzg,
                                              size_t row_count,
                                              size_t column_count,
                                              const std::vector<size_t>& rows)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = matrix_row_selector_cache_key(kzg, row_count, column_count, rows);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const Group1 commitment =
        kzg.commit_g1_group(matrix_row_subset_selector_polynomial(row_count, column_count, rows));
    cache.emplace(key, commitment);
    return commitment;
}

} // namespace

kzgCommitment matrix_row_selector_commitment(kzgProtocol& kzg,
                                             size_t row_count,
                                             size_t column_count,
                                             size_t row)
{
    return matrix_row_subset_selector_commitment(kzg, row_count, column_count, std::vector<size_t>{row});
}

kzgCommitment matrix_row_subset_selector_commitment(kzgProtocol& kzg,
                                                    size_t row_count,
                                                    size_t column_count,
                                                    const std::vector<size_t>& rows)
{
    const std::string key = "matrix_row_subset_selector:" + matrix_row_subset_selector_key(row_count, column_count, rows);
    const std::shared_ptr<ProofSession> proof_session = kzg.proof_session();
    const bool verifier_mode = proof_session && proof_session->mode() == ProofMode::Verify;
    if (verifier_mode) {
        const kzgCommitment commitment =
            kzg.adopt_public_commitment(key, preprocessed_matrix_row_selector_group(kzg, row_count, column_count, rows));
        if (!kzg.has_degree_bound(virCommitment(commitment), row_count * column_count - 1)) {
            kzg.prove_degree(commitment, row_count * column_count - 1);
        }
        return commitment;
    }

    const poly& selector_poly = matrix_row_subset_selector_polynomial(row_count, column_count, rows);
    const Group1 selector_group = preprocessed_matrix_row_selector_group(kzg, row_count, column_count, rows);
    const kzgCommitment commitment =
        kzg.commit_public_with_group(key, selector_poly, selector_group, row_count * column_count - 1);
    cache_matrix_row_selector_group(kzg, row_count, column_count, rows, commitment.com);
    return commitment;
}

void preprocess_matrix_row_subset_selector(kzgProtocol& kzg,
                                           size_t row_count,
                                           size_t column_count,
                                           const std::vector<size_t>& rows)
{
    (void)preprocessed_matrix_row_selector_group(kzg, row_count, column_count, rows);
}
