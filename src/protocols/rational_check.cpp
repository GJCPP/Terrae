#include "rational_check.hpp"

#include <stdexcept>

namespace {

void validate_rational_check_domain(size_t domain_size)
{
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("rationalCheckProtocol: domain size must be a non-zero power of two");
    }
}

} // namespace

void rationalCheckProtocol::add_check(const virCommitment& numerator,
                                      const virCommitment& denominator,
                                      const virCommitment& quotient,
                                      size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("rationalCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr) {
        throw std::runtime_error("rationalCheckProtocol dependencies not setup");
    }

    validate_rational_check_domain(domain_size);
    virCommitment relation_zero = numerator - denominator * quotient;
    params_.zero_check->add_check(relation_zero, domain_size);
}

void rationalCheckProtocol::add_masked_check(const virCommitment& numerator,
                                             const virCommitment& denominator,
                                             const virCommitment& quotient,
                                             const virCommitment& mask,
                                             size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("rationalCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr) {
        throw std::runtime_error("rationalCheckProtocol dependencies not setup");
    }

    validate_rational_check_domain(domain_size);
    virCommitment mask_boolean_zero = mask * (mask - virCommitment(Fr(1)));
    virCommitment relation_zero = mask * (numerator - denominator * quotient);
    params_.zero_check->add_check(mask_boolean_zero, domain_size);
    params_.zero_check->add_check(relation_zero, domain_size);
}

bool rationalCheckProtocol::finalize()
{
    begin_finalize("rationalCheckProtocol");
    return true;
}
