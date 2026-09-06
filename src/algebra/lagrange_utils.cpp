#include "lagrange_utils.hpp"

#include "algebra_utils.hpp"

#include <map>
#include <stdexcept>
#include <string>

namespace {

std::string lagrange_cache_key(size_t subgroup_size, size_t index)
{
    return std::to_string(subgroup_size) + ":" + std::to_string(index);
}

Fr fr_from_size_t(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

Fr subgroup_label_local(size_t subgroup_size, size_t index)
{
    if (subgroup_size == 1) {
        return Fr(1);
    }
    return pow_fr(get_root_of_unity(subgroup_size), index);
}

} // namespace

void validate_lagrange_subgroup_size(size_t subgroup_size, const char* context)
{
    if (subgroup_size == 0 || !isPowerOfTwo(subgroup_size)) {
        throw std::runtime_error(std::string(context) + ": subgroup_size must be a non-zero power of two");
    }
}

poly subgroup_zero_lagrange_polynomial(size_t subgroup_size)
{
    validate_lagrange_subgroup_size(subgroup_size, "subgroup_zero_lagrange_polynomial");
    if (subgroup_size == 1) {
        return poly_from_coeff(std::vector<Fr>{Fr(1)});
    }

    static std::map<size_t, poly> cache;
    const auto it = cache.find(subgroup_size);
    if (it != cache.end()) {
        return it->second;
    }

    const poly zero_basis =
        (vanishing_poly(subgroup_size) / (identity_poly() - Fr(1))) / fr_from_size_t(subgroup_size);
    return cache.emplace(subgroup_size, zero_basis).first->second;
}

poly subgroup_lagrange_polynomial(size_t subgroup_size, size_t index)
{
    validate_lagrange_subgroup_size(subgroup_size, "subgroup_lagrange_polynomial");
    if (index >= subgroup_size) {
        throw std::runtime_error("subgroup_lagrange_polynomial: index out of range");
    }

    static std::map<std::string, poly> cache;
    const std::string key = lagrange_cache_key(subgroup_size, index);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    Fr inverse_label;
    Fr::inv(inverse_label, subgroup_label_local(subgroup_size, index));
    const poly shifted =
        subgroup_zero_lagrange_polynomial(subgroup_size).scale_argument(inverse_label);
    return cache.emplace(key, shifted).first->second;
}

poly subgroup_lagrange_selector_polynomial(size_t subgroup_size, const std::vector<size_t>& indices)
{
    validate_lagrange_subgroup_size(subgroup_size, "subgroup_lagrange_selector_polynomial");

    poly selector;
    for (size_t index : indices) {
        selector = selector + subgroup_lagrange_polynomial(subgroup_size, index);
    }
    return selector;
}

virCommitment subgroup_zero_lagrange_expression(size_t subgroup_size)
{
    validate_lagrange_subgroup_size(subgroup_size, "subgroup_zero_lagrange_expression");
    return subgroup_lagrange_expression(subgroup_size, 0);
}

virCommitment subgroup_lagrange_expression(size_t subgroup_size, size_t index)
{
    validate_lagrange_subgroup_size(subgroup_size, "subgroup_lagrange_expression");
    if (index >= subgroup_size) {
        throw std::runtime_error("subgroup_lagrange_expression: index out of range");
    }

    virCommitment out;
    out.type = virCommitment::Type::LAGRANGE;
    out.lagrange_subgroup_size = subgroup_size;
    out.lagrange_index = index;
    return out;
}

virCommitment subgroup_lagrange_selector_expression(size_t subgroup_size, const std::vector<size_t>& indices)
{
    validate_lagrange_subgroup_size(subgroup_size, "subgroup_lagrange_selector_expression");
    std::vector<std::pair<size_t, Fr>> terms;
    terms.reserve(indices.size());
    for (size_t index : indices) {
        terms.emplace_back(index, Fr(1));
    }
    return subgroup_lagrange_linear_combination_expression(subgroup_size, terms);
}

virCommitment subgroup_lagrange_linear_combination_expression(
    size_t subgroup_size,
    const std::vector<std::pair<size_t, Fr>>& terms)
{
    validate_lagrange_subgroup_size(subgroup_size, "subgroup_lagrange_linear_combination_expression");

    virCommitment out;
    out.type = virCommitment::Type::LAGRANGE_LINEAR_COMBINATION;
    out.lagrange_subgroup_size = subgroup_size;
    for (const auto& [index, coefficient] : terms) {
        if (index >= subgroup_size) {
            throw std::runtime_error("subgroup_lagrange_linear_combination_expression: index out of range");
        }
        if (coefficient.isZero()) {
            continue;
        }
        out.lagrange_terms.emplace_back(index, coefficient);
    }
    return out;
}
