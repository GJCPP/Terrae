#pragma once

#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <limits>
#include <cstdint>

#include <boost/multiprecision/cpp_int.hpp>
#include <mcl/bls12_381.hpp>

using namespace mcl::bn;

bool get_root_of_unity(Fr& omega, size_t n);
Fr get_root_of_unity(size_t n);

inline Fr pow_fr(Fr base, size_t exponent)
{
    Fr result(1);
    while (exponent > 0) {
        if (exponent & 1) {
            Fr::mul(result, result, base);
        }
        exponent >>= 1;
        if (exponent > 0) {
            Fr::mul(base, base, base);
        }
    }
    return result;
}

inline Fr inverse_root_of_unity(size_t n)
{
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("inverse_root_of_unity: n must be a non-zero power of two");
    }
    Fr inv;
    Fr::inv(inv, get_root_of_unity(n));
    return inv;
}

inline Fr fr_from_int(int v)
{
    Fr x;
    x.setStr(std::to_string(v), 10);
    return x;
}

inline Fr fr_from_dec(std::string_view v)
{
    Fr x;
    x.setStr(std::string(v), 10);
    return x;
}

inline size_t canonical_size_t_from_fr(const Fr& value, const char* context)
{
    const std::string repr = value.getStr(10);
    size_t out = 0;
    try {
        out = static_cast<size_t>(std::stoull(repr));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(context) + ": value is not a canonical size_t");
    }
    Fr round_trip;
    round_trip.setStr(std::to_string(out), 10);
    if (round_trip != value) {
        throw std::runtime_error(std::string(context) + ": value is not a canonical size_t");
    }
    return out;
}

inline std::int64_t canonical_signed_int64_from_fr(const Fr& value, const char* context)
{
    using boost::multiprecision::cpp_int;

    auto parse_cpp_int = [context](const std::string& repr) {
        cpp_int out = 0;
        std::istringstream in(repr);
        in >> out;
        if (!in || !in.eof()) {
            throw std::runtime_error(std::string(context) + ": value is not a canonical signed int64");
        }
        return out;
    };

    const cpp_int modulus = parse_cpp_int(Fr::getModulo());
    cpp_int centered = parse_cpp_int(value.getStr(10));
    if (centered > modulus / 2) {
        centered -= modulus;
    }

    if (centered < std::numeric_limits<std::int64_t>::min() ||
        centered > std::numeric_limits<std::int64_t>::max()) {
        throw std::runtime_error(std::string(context) + ": value does not fit into int64");
    }

    const std::int64_t out = centered.convert_to<std::int64_t>();
    Fr round_trip;
    round_trip.setStr(std::to_string(out), 10);
    if (round_trip != value) {
        throw std::runtime_error(std::string(context) + ": value is not a canonical signed int64");
    }
    return out;
}

inline bool has_recorded_opening(const std::vector<Fr>& points, const Fr& x)
{
    for (const Fr& point : points) {
        if (point == x) {
            return true;
        }
    }
    return false;
}

inline std::pair<std::vector<Fr>, std::vector<Fr>> unique_openings(const std::vector<Fr>& xs,
                                                                   const std::vector<Fr>& ys)
{
    std::vector<Fr> unique_xs;
    std::vector<Fr> unique_ys;
    unique_xs.reserve(xs.size());
    unique_ys.reserve(ys.size());

    for (size_t i = 0; i < xs.size(); ++i) {
        if (!has_recorded_opening(unique_xs, xs[i])) {
            unique_xs.push_back(xs[i]);
            unique_ys.push_back(ys[i]);
        }
    }
    return {unique_xs, unique_ys};
}
