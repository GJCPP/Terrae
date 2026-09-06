#include "box_check.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

void boxCheckProtocol::add_check(const virCommitment& values,
                                 const virCommitment& lower_bounds,
                                 const virCommitment& upper_bounds,
                                 size_t row_count,
                                 size_t column_count,
                                 size_t bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("boxCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("boxCheckProtocol dependencies not setup");
    }

    validate_matrix_shape(row_count, column_count, "boxCheckProtocol");
    const size_t domain_size = row_count * column_count;

    // Inclusive lower bound and exclusive upper bound:
    //   lower_bounds <= values < upper_bounds.
    // Both sides reduce to unsigned gap checks.
    const virCommitment lower_gap = values - lower_bounds;
    const virCommitment upper_gap = upper_bounds - virCommitment(Fr(1)) - values;

    params_.less_check->add_check(lower_gap, bit_bound, domain_size);
    params_.less_check->add_check(upper_gap, bit_bound, domain_size);
}

bool boxCheckProtocol::finalize()
{
    begin_finalize("boxCheckProtocol");
    return true;
}
