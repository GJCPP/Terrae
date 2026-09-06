#pragma once

#include <cstddef>
#include <vector>

#include "algebra_utils.hpp"
#include "kzg.hpp"
#include "lagrange_utils.hpp"

using namespace mcl::bn;

void validate_matrix_shape(size_t row_count, size_t column_count, const char* context);

size_t matrix_index(size_t row, size_t column, size_t column_count);

Fr field_from_size_t(size_t value);
Fr inverse_field_from_size_t(size_t value);

Fr subgroup_generator(size_t domain_size);
Fr subgroup_label(size_t domain_size, size_t index);

virCommitment scale_by_constant(const virCommitment& commitment, const Fr& scalar);

virCommitment matrix_row_selector_expression(size_t row_count, size_t column_count, size_t row);
virCommitment matrix_row_subset_selector_expression(size_t row_count,
                                                    size_t column_count,
                                                    const std::vector<size_t>& rows);
virCommitment matrix_column_index_expression(size_t row_count, size_t column_count);

const poly& matrix_row_selector_polynomial(size_t row_count, size_t column_count, size_t row);
const poly& matrix_row_subset_selector_polynomial(size_t row_count,
                                                  size_t column_count,
                                                  const std::vector<size_t>& rows);

kzgCommitment matrix_row_selector_commitment(kzgProtocol& kzg, size_t row_count, size_t column_count, size_t row);
kzgCommitment matrix_row_subset_selector_commitment(kzgProtocol& kzg,
                                                    size_t row_count,
                                                    size_t column_count,
                                                    const std::vector<size_t>& rows);

void preprocess_matrix_row_subset_selector(kzgProtocol& kzg,
                                           size_t row_count,
                                           size_t column_count,
                                           const std::vector<size_t>& rows);
