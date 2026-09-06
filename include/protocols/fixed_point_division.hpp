#pragma once

#include "fixed_point.hpp"
#include "less_check.hpp"
#include "nonzero_check.hpp"
#include "protocol_lifecycle.hpp"
#include "signed_range_check.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class fixedPointDivisionRecord {
public:
    fixedPointEncoding numerator_encoding;
    fixedPointEncoding denominator_encoding;
    fixedPointEncoding quotient_encoding;
    size_t denominator_bit_bound = 0;
    size_t domain_size = 0;
    bool uses_mask = false;
    bool uses_signed_denominator = false;
};

class fixedPointDivisionParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    lessCheckProtocol* less_check = nullptr;
    signedRangeCheckProtocol* signed_range_check = nullptr;
    nonzeroCheckProtocol* nonzero_check = nullptr;
};

void add_fixed_point_division_srs_requirement(lessCheckSrsEstimator& less_check,
                                              const fixedPointEncoding& numerator_encoding,
                                              const fixedPointEncoding& quotient_encoding,
                                              size_t denominator_bit_bound,
                                              size_t domain_size);

void add_fixed_point_division_srs_requirement(lessCheckSrsEstimator& less_check,
                                              const fixedPointEncoding& numerator_encoding,
                                              const fixedPointEncoding& denominator_encoding,
                                              const fixedPointEncoding& quotient_encoding,
                                              size_t denominator_bit_bound,
                                              size_t domain_size);

// Proves a signed fixed-point Euclidean division relation.
//
// Let:
// - `num` be an offset-binary encoded signed scaled integer witness
// - `quo` be an offset-binary encoded signed scaled integer witness
// - `den` be a positive integer witness
// - `rem` be a non-negative integer witness
//
// Then the protocol proves:
//   decode(num) * 2^(quo_frac + den_frac - num_frac) = den * decode(quo) + rem
//   0 <= rem < den
//
// Example with numerator fractional_bits = 0, denominator fractional_bits = 0,
// and quotient fractional_bits = 2:
//   num = -3
//   den = 2
//   quo = -6  (represents -1.5 after dividing by 2^2)
//   rem = 0
//
// because:
//   (-3) * 2^2 = 2 * (-6) + 0
//
// The masked variant only enforces the relation on rows where mask = 1.
//
// Signed-denominator overload:
// - `encoded_denominator` is an offset-binary encoded signed integer witness
// - `denominator_abs = |decode(encoded_denominator)|`
// - `denominator_is_negative` is boolean and indicates the sign of the denominator
// - `remainder_abs = |remainder|`
//
// The protocol proves floor-division semantics by enforcing:
//   decode(encoded_denominator) = denominator_abs * (1 - 2 * denominator_is_negative)
//   decode(num) * 2^(quo_frac + den_frac - num_frac) =
//       decode(encoded_denominator) * decode(quo) +
//       remainder_abs * (1 - 2 * denominator_is_negative)
//   0 <= remainder_abs < denominator_abs
class fixedPointDivisionProtocol : protected ProtocolLifecycle {
public:
    ~fixedPointDivisionProtocol() noexcept
    {
        auto_finalize("fixedPointDivisionProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero,
               lessCheckProtocol& less_check,
               signedRangeCheckProtocol& signed_range)
    {
        setup(kzg, zero, less_check, signed_range, nullptr);
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero,
               lessCheckProtocol& less_check,
               signedRangeCheckProtocol& signed_range,
               nonzeroCheckProtocol& nonzero_check)
    {
        setup(kzg, zero, less_check, signed_range, &nonzero_check);
    }

private:
    void setup(kzgProtocol& kzg,
               zeroProtocol& zero,
               lessCheckProtocol& less_check,
               signedRangeCheckProtocol& signed_range,
               nonzeroCheckProtocol* nonzero_check)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.less_check = &less_check;
        params_.signed_range_check = &signed_range;
        params_.nonzero_check = nonzero_check;
        mark_setup();
    }

public:

    void add_check(const virCommitment& encoded_numerator,
                   const virCommitment& denominator,
                   const virCommitment& encoded_quotient,
                   const virCommitment& remainder,
                   const fixedPointEncoding& numerator_encoding,
                   const fixedPointEncoding& quotient_encoding,
                   size_t denominator_bit_bound,
                   size_t domain_size);

    void add_masked_check(const virCommitment& encoded_numerator,
                          const virCommitment& denominator,
                          const virCommitment& encoded_quotient,
                          const virCommitment& remainder,
                          const virCommitment& mask,
                          const fixedPointEncoding& numerator_encoding,
                          const fixedPointEncoding& quotient_encoding,
                          size_t denominator_bit_bound,
                          size_t domain_size);

    void add_check(const virCommitment& encoded_numerator,
                   const virCommitment& encoded_denominator,
                   const virCommitment& denominator_abs,
                   const virCommitment& denominator_is_negative,
                   const virCommitment& encoded_quotient,
                   const virCommitment& remainder_abs,
                   const fixedPointEncoding& numerator_encoding,
                   const fixedPointEncoding& denominator_encoding,
                   const fixedPointEncoding& quotient_encoding,
                   size_t denominator_bit_bound,
                   size_t domain_size);

    void add_masked_check(const virCommitment& encoded_numerator,
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
                          size_t domain_size);

    bool finalize();

private:
    fixedPointDivisionParam params_;
};
