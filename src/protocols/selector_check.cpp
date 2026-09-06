#include "selector_check.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

void validate_boolean_domain(size_t domain_size, const char* context)
{
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error(std::string(context) + ": domain_size must be a non-zero power of two");
    }
}

} // namespace

void selectorCheckProtocol::add_boolean_check(const virCommitment& values, size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("selectorCheckProtocol not setup");
    }
    if (params_.zero_check == nullptr) {
        throw std::runtime_error("selectorCheckProtocol zero-check dependency not setup");
    }

    validate_boolean_domain(domain_size, "selectorCheckProtocol");
    // Step 1: force every entry to be 0/1.
    params_.zero_check->add_boolean_check(values, domain_size, "selector_check.boolean");
    ++params_.check_count;
}

void selectorCheckProtocol::add_one_hot_rows_check(const virCommitment& selector,
                                                   size_t row_count,
                                                   size_t column_count)
{
    if (!is_setup) {
        throw std::runtime_error("selectorCheckProtocol not setup");
    }
    if (params_.zero_check == nullptr || params_.row_sum == nullptr) {
        throw std::runtime_error("selectorCheckProtocol dependencies not setup");
    }

    validate_matrix_shape(row_count, column_count, "selectorCheckProtocol");
    const size_t domain_size = row_count * column_count;

    // Step 1: every cell must be boolean.
    params_.zero_check->add_boolean_check(selector, domain_size, "selector_check.one_hot_cells");

    // Step 2: each row must sum to exactly 1.
    //
    // Example:
    //   selector = [[0, 1, 0, 0],
    //               [0, 0, 1, 0]]
    // Then row_sum(selector) = [1, 1].
    params_.row_sum->add_check(selector, virCommitment(Fr(1)), row_count, column_count);
    ++params_.check_count;
}

void selectorCheckProtocol::add_masked_one_hot_rows_check(const virCommitment& selector,
                                                          const virCommitment& row_mask,
                                                          size_t row_count,
                                                          size_t column_count)
{
    if (!is_setup) {
        throw std::runtime_error("selectorCheckProtocol not setup");
    }
    if (params_.zero_check == nullptr || params_.row_sum == nullptr) {
        throw std::runtime_error("selectorCheckProtocol dependencies not setup");
    }

    validate_matrix_shape(row_count, column_count, "selectorCheckProtocol");
    const size_t domain_size = row_count * column_count;

    // Step 1: selector cells must be 0/1.
    params_.zero_check->add_boolean_check(
        selector, domain_size, "selector_check.masked_one_hot_cells");

    // Step 2: register the shared row-mask Booleanity claim once.
    params_.zero_check->add_boolean_check(row_mask, row_count, "selector_check.row_mask");

    // Step 3: tie the selector row sums to the mask.
    //
    // Example:
    //   selector = [[0, 1, 0, 0],
    //               [0, 0, 0, 0]]
    //   row_mask = [1, 0]
    //
    // The first row is active and must sum to 1.
    // The second row is inactive and must sum to 0.
    params_.row_sum->add_check(selector, row_mask, row_count, column_count);
    ++params_.check_count;
}

bool selectorCheckProtocol::finalize()
{
    begin_finalize("selectorCheckProtocol");
    return true;
}
