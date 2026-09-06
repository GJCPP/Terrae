#include "signed_range_check.hpp"

#include <stdexcept>

void add_signed_range_check_srs_requirement(lessCheckSrsEstimator& less_check,
                                            const fixedPointEncoding& encoding,
                                            size_t domain_size)
{
    validate_fixed_point_encoding(encoding, "signedRangeCheckProtocol");
    if (!encoding.signed_values) {
        throw std::runtime_error("signedRangeCheckProtocol: encoding must be signed");
    }
    less_check.add_check(fixed_point_encoded_bit_bound(encoding), domain_size);
}

void signedRangeCheckProtocol::add_check(const virCommitment& encoded_values,
                                         const fixedPointEncoding& encoding,
                                         size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("signedRangeCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("signedRangeCheckProtocol dependencies not setup");
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("signedRangeCheckProtocol: domain size must be a non-zero power of two");
    }

    validate_fixed_point_encoding(encoding, "signedRangeCheckProtocol");
    if (!encoding.signed_values) {
        throw std::runtime_error("signedRangeCheckProtocol: encoding must be signed");
    }

    // Offset-binary signed values are still stored as unsigned field elements.
    // So signed-range validation reduces directly to one unsigned less-check
    // against the encoding's full representable interval.
    params_.less_check->add_check(encoded_values, fixed_point_encoded_bit_bound(encoding), domain_size);
}

bool signedRangeCheckProtocol::finalize()
{
    begin_finalize("signedRangeCheckProtocol");
    return true;
}
