#include "fixed_point_division.hpp"

#include <limits>
#include <stdexcept>

namespace {

void validate_fixed_point_division_inputs(const fixedPointEncoding& numerator_encoding,
                                          size_t denominator_fractional_bits,
                                          const fixedPointEncoding& quotient_encoding,
                                          size_t denominator_bit_bound,
                                          size_t domain_size)
{
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("fixedPointDivisionProtocol: domain size must be a non-zero power of two");
    }

    validate_fixed_point_encoding(numerator_encoding, "fixedPointDivisionProtocol");
    validate_fixed_point_encoding(quotient_encoding, "fixedPointDivisionProtocol");
    if (!numerator_encoding.signed_values || !quotient_encoding.signed_values) {
        throw std::runtime_error("fixedPointDivisionProtocol: numerator and quotient encodings must be signed");
    }
    const size_t quotient_denominator_fractional_sum =
        quotient_encoding.fractional_bits + denominator_fractional_bits;
    if (quotient_denominator_fractional_sum < numerator_encoding.fractional_bits) {
        throw std::runtime_error(
            "fixedPointDivisionProtocol: quotient fractional_bits plus denominator fractional_bits must be at least numerator fractional_bits");
    }
    if (denominator_bit_bound == 0) {
        throw std::runtime_error("fixedPointDivisionProtocol: denominator bit bound must be non-zero");
    }
    if (denominator_bit_bound > static_cast<size_t>(Fr::getBitSize() - 1)) {
        throw std::runtime_error(
            "fixedPointDivisionProtocol: denominator bit bound exceeds the safely representable integer range of Fr");
    }
    if (denominator_bit_bound >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error(
            "fixedPointDivisionProtocol: denominator bit bound is too large for size_t-based power-of-two scaling");
    }
}

void validate_signed_denominator_inputs(const fixedPointEncoding& numerator_encoding,
                                        const fixedPointEncoding& denominator_encoding,
                                        const fixedPointEncoding& quotient_encoding,
                                        size_t denominator_bit_bound,
                                        size_t domain_size)
{
    validate_fixed_point_division_inputs(
        numerator_encoding, denominator_encoding.fractional_bits, quotient_encoding, denominator_bit_bound, domain_size);
    validate_fixed_point_encoding(denominator_encoding, "fixedPointDivisionProtocol");
    if (!denominator_encoding.signed_values) {
        throw std::runtime_error("fixedPointDivisionProtocol: signed denominator encoding must be signed");
    }
}

Fr power_of_two_scale(size_t exponent)
{
    return fixed_point_fr_from_size_t(size_t(1) << exponent);
}

void add_fixed_point_division_constraints(const fixedPointDivisionParam& params,
                                          const virCommitment& encoded_numerator,
                                          const virCommitment& denominator,
                                          const virCommitment& denominator_abs,
                                          const virCommitment& denominator_is_negative,
                                          const virCommitment& encoded_quotient,
                                          const virCommitment& remainder,
                                          const virCommitment& mask,
                                          const fixedPointDivisionRecord& record)
{
    const size_t denominator_fractional_bits =
        record.uses_signed_denominator ? record.denominator_encoding.fractional_bits : 0;
    const size_t scale_power = record.quotient_encoding.fractional_bits + denominator_fractional_bits -
                               record.numerator_encoding.fractional_bits;

    const virCommitment decoded_numerator =
        encoded_numerator - virCommitment(fixed_point_offset_fr(record.numerator_encoding));
    const virCommitment decoded_quotient =
        encoded_quotient - virCommitment(fixed_point_offset_fr(record.quotient_encoding));
    const virCommitment scaled_numerator = decoded_numerator * virCommitment(power_of_two_scale(scale_power));
    virCommitment signed_denominator = denominator;
    virCommitment signed_remainder = remainder;

    params.signed_range_check->add_check(encoded_numerator, record.numerator_encoding, record.domain_size);
    params.signed_range_check->add_check(encoded_quotient, record.quotient_encoding, record.domain_size);

    if (record.uses_signed_denominator) {
        params.signed_range_check->add_check(denominator, record.denominator_encoding, record.domain_size);

        virCommitment sign_boolean_zero =
            denominator_is_negative * (denominator_is_negative - virCommitment(Fr(1)));
        virCommitment sign_scale = virCommitment(Fr(1)) - denominator_is_negative * virCommitment(Fr(2));
        const virCommitment decoded_denominator =
            denominator - virCommitment(fixed_point_offset_fr(record.denominator_encoding));
        signed_denominator = denominator_abs * sign_scale;
        signed_remainder = remainder * sign_scale;

        virCommitment denominator_sign_zero = decoded_denominator - signed_denominator;
        if (record.uses_mask) {
            sign_boolean_zero = mask * sign_boolean_zero;
            denominator_sign_zero = mask * denominator_sign_zero;
        }
        params.zero_check->add_check(sign_boolean_zero, record.domain_size);
        params.zero_check->add_check(denominator_sign_zero, record.domain_size);
    }

    virCommitment relation_zero = scaled_numerator - signed_denominator * decoded_quotient - signed_remainder;

    if (record.uses_mask) {
        params.zero_check->add_boolean_check(
            mask, record.domain_size, "fixed_point_division.mask_boolean");
        relation_zero = mask * relation_zero;

        params.less_check->add_masked_check(
            denominator_abs, mask, record.denominator_bit_bound, record.domain_size);
        if (params.nonzero_check != nullptr) {
            params.nonzero_check->add_masked_check(
                denominator_abs, mask, record.domain_size, "fixed_point_division.denominator_nonzero");
        } else {
            params.less_check->add_masked_check(
                denominator_abs - virCommitment(Fr(1)), mask, record.denominator_bit_bound, record.domain_size);
        }
        params.less_check->add_masked_check(remainder, mask, record.denominator_bit_bound, record.domain_size);
        params.less_check->add_masked_check(denominator_abs - virCommitment(Fr(1)) - remainder,
                                            mask,
                                            record.denominator_bit_bound,
                                            record.domain_size);
    } else {
        params.less_check->add_check(denominator_abs, record.denominator_bit_bound, record.domain_size);
        if (params.nonzero_check != nullptr) {
            params.nonzero_check->add_check(
                denominator_abs, record.domain_size, "fixed_point_division.denominator_nonzero");
        } else {
            params.less_check->add_check(
                denominator_abs - virCommitment(Fr(1)), record.denominator_bit_bound, record.domain_size);
        }
        params.less_check->add_check(remainder, record.denominator_bit_bound, record.domain_size);
        params.less_check->add_check(
            denominator_abs - virCommitment(Fr(1)) - remainder, record.denominator_bit_bound, record.domain_size);
    }

    params.zero_check->add_check(relation_zero, record.domain_size);
}

} // namespace

void add_fixed_point_division_srs_requirement(lessCheckSrsEstimator& less_check,
                                              const fixedPointEncoding& numerator_encoding,
                                              const fixedPointEncoding& quotient_encoding,
                                              size_t denominator_bit_bound,
                                              size_t domain_size)
{
    validate_fixed_point_division_inputs(
        numerator_encoding, 0, quotient_encoding, denominator_bit_bound, domain_size);
    add_signed_range_check_srs_requirement(less_check, numerator_encoding, domain_size);
    add_signed_range_check_srs_requirement(less_check, quotient_encoding, domain_size);
    less_check.add_check(denominator_bit_bound, domain_size, 3);
}

void add_fixed_point_division_srs_requirement(lessCheckSrsEstimator& less_check,
                                              const fixedPointEncoding& numerator_encoding,
                                              const fixedPointEncoding& denominator_encoding,
                                              const fixedPointEncoding& quotient_encoding,
                                              size_t denominator_bit_bound,
                                              size_t domain_size)
{
    validate_signed_denominator_inputs(
        numerator_encoding, denominator_encoding, quotient_encoding, denominator_bit_bound, domain_size);
    add_signed_range_check_srs_requirement(less_check, numerator_encoding, domain_size);
    add_signed_range_check_srs_requirement(less_check, quotient_encoding, domain_size);
    add_signed_range_check_srs_requirement(less_check, denominator_encoding, domain_size);
    less_check.add_check(denominator_bit_bound, domain_size, 3);
}

void fixedPointDivisionProtocol::add_check(const virCommitment& encoded_numerator,
                                           const virCommitment& denominator,
                                           const virCommitment& encoded_quotient,
                                           const virCommitment& remainder,
                                           const fixedPointEncoding& numerator_encoding,
                                           const fixedPointEncoding& quotient_encoding,
                                           size_t denominator_bit_bound,
                                           size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("fixedPointDivisionProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.less_check == nullptr ||
        params_.signed_range_check == nullptr) {
        throw std::runtime_error("fixedPointDivisionProtocol dependencies not setup");
    }

    validate_fixed_point_division_inputs(
        numerator_encoding, 0, quotient_encoding, denominator_bit_bound, domain_size);
    add_fixed_point_division_constraints(params_,
                                         encoded_numerator,
                                         denominator,
                                         denominator,
                                         virCommitment(Fr(0)),
                                         encoded_quotient,
                                         remainder,
                                         virCommitment(Fr(1)),
                                         {numerator_encoding, {}, quotient_encoding, denominator_bit_bound, domain_size, false, false});
}

void fixedPointDivisionProtocol::add_masked_check(const virCommitment& encoded_numerator,
                                                  const virCommitment& denominator,
                                                  const virCommitment& encoded_quotient,
                                                  const virCommitment& remainder,
                                                  const virCommitment& mask,
                                                  const fixedPointEncoding& numerator_encoding,
                                                  const fixedPointEncoding& quotient_encoding,
                                                  size_t denominator_bit_bound,
                                                  size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("fixedPointDivisionProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.less_check == nullptr ||
        params_.signed_range_check == nullptr) {
        throw std::runtime_error("fixedPointDivisionProtocol dependencies not setup");
    }

    validate_fixed_point_division_inputs(
        numerator_encoding, 0, quotient_encoding, denominator_bit_bound, domain_size);
    add_fixed_point_division_constraints(params_,
                                         encoded_numerator,
                                         denominator,
                                         denominator,
                                         virCommitment(Fr(0)),
                                         encoded_quotient,
                                         remainder,
                                         mask,
                                         {numerator_encoding, {}, quotient_encoding, denominator_bit_bound, domain_size, true, false});
}

void fixedPointDivisionProtocol::add_check(const virCommitment& encoded_numerator,
                                           const virCommitment& encoded_denominator,
                                           const virCommitment& denominator_abs,
                                           const virCommitment& denominator_is_negative,
                                           const virCommitment& encoded_quotient,
                                           const virCommitment& remainder_abs,
                                           const fixedPointEncoding& numerator_encoding,
                                           const fixedPointEncoding& denominator_encoding,
                                           const fixedPointEncoding& quotient_encoding,
                                           size_t denominator_bit_bound,
                                           size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("fixedPointDivisionProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.less_check == nullptr ||
        params_.signed_range_check == nullptr) {
        throw std::runtime_error("fixedPointDivisionProtocol dependencies not setup");
    }

    validate_signed_denominator_inputs(
        numerator_encoding, denominator_encoding, quotient_encoding, denominator_bit_bound, domain_size);
    add_fixed_point_division_constraints(params_,
                                         encoded_numerator,
                                         encoded_denominator,
                                         denominator_abs,
                                         denominator_is_negative,
                                         encoded_quotient,
                                         remainder_abs,
                                         virCommitment(Fr(1)),
                                         {numerator_encoding,
                                          denominator_encoding,
                                          quotient_encoding,
                                          denominator_bit_bound,
                                          domain_size,
                                          false,
                                          true});
}

void fixedPointDivisionProtocol::add_masked_check(const virCommitment& encoded_numerator,
                                                  const virCommitment& encoded_denominator,
                                                  const virCommitment& denominator_abs,
                                                  const virCommitment& denominator_is_negative,
                                                  const virCommitment& encoded_quotient,
                                                  const virCommitment& remainder_abs,
                                                  const virCommitment& mask,
                                                  const fixedPointEncoding& numerator_encoding,
                                                  const fixedPointEncoding& denominator_encoding,
                                                  const fixedPointEncoding& quotient_encoding,
                                                  size_t denominator_bit_bound,
                                                  size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("fixedPointDivisionProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.less_check == nullptr ||
        params_.signed_range_check == nullptr) {
        throw std::runtime_error("fixedPointDivisionProtocol dependencies not setup");
    }

    validate_signed_denominator_inputs(
        numerator_encoding, denominator_encoding, quotient_encoding, denominator_bit_bound, domain_size);
    add_fixed_point_division_constraints(params_,
                                         encoded_numerator,
                                         encoded_denominator,
                                         denominator_abs,
                                         denominator_is_negative,
                                         encoded_quotient,
                                         remainder_abs,
                                         mask,
                                         {numerator_encoding,
                                          denominator_encoding,
                                          quotient_encoding,
                                          denominator_bit_bound,
                                          domain_size,
                                          true,
                                          true});
}

bool fixedPointDivisionProtocol::finalize()
{
    begin_finalize("fixedPointDivisionProtocol");
    return true;
}
