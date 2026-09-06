#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "kzg.hpp"
#include "polynomial.hpp"

using namespace mcl::bn;

void validate_lagrange_subgroup_size(size_t subgroup_size, const char* context);

poly subgroup_zero_lagrange_polynomial(size_t subgroup_size);
poly subgroup_lagrange_polynomial(size_t subgroup_size, size_t index);
poly subgroup_lagrange_selector_polynomial(size_t subgroup_size, const std::vector<size_t>& indices);

virCommitment subgroup_zero_lagrange_expression(size_t subgroup_size);
virCommitment subgroup_lagrange_expression(size_t subgroup_size, size_t index);
virCommitment subgroup_lagrange_selector_expression(size_t subgroup_size, const std::vector<size_t>& indices);
virCommitment subgroup_lagrange_linear_combination_expression(
    size_t subgroup_size,
    const std::vector<std::pair<size_t, Fr>>& terms);
