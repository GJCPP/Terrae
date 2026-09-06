#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"

// First-iteration fixed-point metadata.
//
// Interpretation:
// - values are stored as scaled integers
// - the real value is scaled_value / 2^fractional_bits
// - signed values use offset-binary encoding
//
// Example with total_bits = 4 and fractional_bits = 2:
// - scaled range is [-8, 7]
// - real range is [-2.00, 1.75]
// - encoded value = scaled_value + 8
enum class fixedPointEncodingKind {
    OffsetBinary,
};

class fixedPointEncoding {
public:
    static fixedPointEncoding signed_offset_binary(size_t total_bits, size_t fractional_bits = 0)
    {
        fixedPointEncoding encoding;
        encoding.kind = fixedPointEncodingKind::OffsetBinary;
        encoding.total_bits = total_bits;
        encoding.fractional_bits = fractional_bits;
        encoding.signed_values = true;
        return encoding;
    }

    fixedPointEncodingKind kind = fixedPointEncodingKind::OffsetBinary;
    size_t total_bits = 0;
    size_t fractional_bits = 0;
    bool signed_values = false;
};

inline void validate_fixed_point_encoding(const fixedPointEncoding& encoding, const char* context)
{
    if (encoding.total_bits == 0) {
        throw std::runtime_error(std::string(context) + ": total_bits must be non-zero");
    }
    if (encoding.total_bits >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error(std::string(context) + ": total_bits are too large for size_t-based range tables");
    }
    if (encoding.total_bits > static_cast<size_t>(std::numeric_limits<std::int64_t>::digits)) {
        throw std::runtime_error(std::string(context) + ": total_bits exceed the supported int64 helper range");
    }
    if (encoding.fractional_bits >= encoding.total_bits) {
        throw std::runtime_error(std::string(context) + ": fractional_bits must be smaller than total_bits");
    }
    if (encoding.kind != fixedPointEncodingKind::OffsetBinary) {
        throw std::runtime_error(std::string(context) + ": unsupported fixed-point encoding kind");
    }
    if (encoding.signed_values && encoding.total_bits < 2) {
        throw std::runtime_error(std::string(context) + ": signed encodings need at least 2 total bits");
    }
}

inline size_t fixed_point_signed_offset(const fixedPointEncoding& encoding)
{
    validate_fixed_point_encoding(encoding, "fixed_point_signed_offset");
    if (!encoding.signed_values) {
        throw std::runtime_error("fixed_point_signed_offset: encoding is not signed");
    }
    return size_t(1) << (encoding.total_bits - 1);
}

inline size_t fixed_point_encoded_bit_bound(const fixedPointEncoding& encoding)
{
    validate_fixed_point_encoding(encoding, "fixed_point_encoded_bit_bound");
    return encoding.total_bits;
}

inline std::int64_t fixed_point_min_scaled_value(const fixedPointEncoding& encoding)
{
    return -static_cast<std::int64_t>(fixed_point_signed_offset(encoding));
}

inline std::int64_t fixed_point_max_scaled_value(const fixedPointEncoding& encoding)
{
    return static_cast<std::int64_t>(fixed_point_signed_offset(encoding)) - 1;
}

inline size_t encode_fixed_point_scaled_integer(std::int64_t scaled_value, const fixedPointEncoding& encoding)
{
    validate_fixed_point_encoding(encoding, "encode_fixed_point_scaled_integer");
    if (!encoding.signed_values) {
        throw std::runtime_error("encode_fixed_point_scaled_integer: encoding is not signed");
    }
    if (scaled_value < fixed_point_min_scaled_value(encoding) || scaled_value > fixed_point_max_scaled_value(encoding)) {
        throw std::runtime_error("encode_fixed_point_scaled_integer: scaled value " + std::to_string(scaled_value) +
                                 " out of range [" + std::to_string(fixed_point_min_scaled_value(encoding)) + ", " +
                                 std::to_string(fixed_point_max_scaled_value(encoding)) + "] for encoding(total_bits=" +
                                 std::to_string(encoding.total_bits) + ", fractional_bits=" +
                                 std::to_string(encoding.fractional_bits) + ")");
    }

    return static_cast<size_t>(scaled_value + static_cast<std::int64_t>(fixed_point_signed_offset(encoding)));
}

inline std::int64_t decode_fixed_point_scaled_integer(size_t encoded_value, const fixedPointEncoding& encoding)
{
    validate_fixed_point_encoding(encoding, "decode_fixed_point_scaled_integer");
    if (!encoding.signed_values) {
        throw std::runtime_error("decode_fixed_point_scaled_integer: encoding is not signed");
    }
    if (encoded_value >= (size_t(1) << encoding.total_bits)) {
        throw std::runtime_error("decode_fixed_point_scaled_integer: encoded value out of range for encoding");
    }

    return static_cast<std::int64_t>(encoded_value) - static_cast<std::int64_t>(fixed_point_signed_offset(encoding));
}

inline Fr fixed_point_fr_from_size_t(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

// Returns the offset-binary bias as a field element.
// Example: for total_bits = 4 this returns 8, so encoded_shifted = encoded_score - encoded_shift + 8.
inline Fr fixed_point_offset_fr(const fixedPointEncoding& encoding)
{
    return fixed_point_fr_from_size_t(fixed_point_signed_offset(encoding));
}

// Parses a field element back into a canonical non-negative integer.
// Example: the encoded score 9 for a 4-bit offset-binary value returns size_t(9).
inline size_t fixed_point_size_t_from_fr(const Fr& value, const char* context)
{
    const std::string repr = value.getStr(10);
    if (!repr.empty() && repr.front() == '-') {
        throw std::runtime_error(std::string(context) + ": expected a non-negative field element");
    }

    size_t out = 0;
    try {
        out = static_cast<size_t>(std::stoull(repr));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(context) + ": field element does not fit into size_t");
    }

    if (fixed_point_fr_from_size_t(out) != value) {
        throw std::runtime_error(std::string(context) + ": field element is not a canonical non-negative integer");
    }
    return out;
}

inline poly fixed_point_encoded_poly_from_scaled_integers(const std::vector<std::int64_t>& scaled_values,
                                                          const fixedPointEncoding& encoding)
{
    std::vector<Fr> evals;
    evals.reserve(scaled_values.size());
    for (std::int64_t value : scaled_values) {
        evals.push_back(fixed_point_fr_from_size_t(encode_fixed_point_scaled_integer(value, encoding)));
    }
    return poly_from_evals(evals);
}
