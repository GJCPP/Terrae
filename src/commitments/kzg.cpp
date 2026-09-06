#include "algebra_utils.hpp"
#include "kzg.hpp"
#include "lagrange_utils.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mcl/ec.hpp>
#include <mcl/randgen.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace mcl::bn;

namespace {

using KzgClock = std::chrono::steady_clock;

bool kzg_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_KZG");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

bool kzg_debug_fail_enabled()
{
    const char* flag = terrae_getenv("TERRAE_DEBUG_KZG_FAIL");
    return flag != nullptr && flag[0] != '\0' && std::string(flag) != "0" && std::string(flag) != "false";
}

bool kzg_disable_prefer_eval()
{
    const char* flag = terrae_getenv("TERRAE_DISABLE_PREFER_EVAL");
    return flag != nullptr && flag[0] != '\0' && std::string(flag) != "0" && std::string(flag) != "false";
}

Fr inverse_of_two()
{
    static const Fr half = [] {
        Fr inv;
        Fr::inv(inv, fr_from_int(2));
        return inv;
    }();
    return half;
}

size_t kzg_srs_prefix_decode_floor()
{
    const char* value = terrae_getenv("TERRAE_SRS_PREFIX_DECODE_FLOOR");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    try {
        return static_cast<size_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::runtime_error("TERRAE_SRS_PREFIX_DECODE_FLOOR must be a non-negative integer");
    }
}

struct KzgProfileStats {
    std::map<std::string, double> seconds;
    std::map<std::string, size_t> counts;

    void add_seconds(const std::string& label, double value)
    {
        seconds[label] += value;
    }

    void increment(const std::string& label, size_t delta = 1)
    {
        counts[label] += delta;
    }

    void print(const char* scope) const
    {
        const bool stderr_enabled = kzg_profile_enabled();
        const bool structured_enabled = structured_protocol_profile::enabled();
        if (!stderr_enabled && !structured_enabled) {
            return;
        }
        for (const auto& [label, value] : seconds) {
            if (stderr_enabled) {
                std::cerr << "kzg_profile " << scope << " " << label << " " << value << "\n";
            }
            structured_protocol_profile::append_row("kzg_profile", std::string(scope) + "." + label, "", value, "ok");
        }
        for (const auto& [label, value] : counts) {
            if (stderr_enabled) {
                std::cerr << "kzg_profile_count " << scope << " " << label << " " << value << "\n";
            }
            structured_protocol_profile::append_row(
                "kzg_profile_count", std::string(scope) + "." + label, "", static_cast<double>(value), "count");
        }
    }
};

template <class Fn>
auto kzg_profile_scope(KzgProfileStats& stats, const std::string& label, Fn&& fn)
{
    const auto start = KzgClock::now();
    auto result = fn();
    const auto end = KzgClock::now();
    stats.add_seconds(label,
                      std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    return result;
}

template <class Fn>
void kzg_profile_scope_void(KzgProfileStats& stats, const std::string& label, Fn&& fn)
{
    const auto start = KzgClock::now();
    fn();
    const auto end = KzgClock::now();
    stats.add_seconds(label,
                      std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
}

struct CachedSrsPowers {
    std::vector<Group1> g1_powers;
    std::vector<Group2> g2_powers;
    std::vector<G1> g1_raw_powers;
    std::vector<G2> g2_raw_powers;
};

size_t bit_reverse(size_t index, size_t bits)
{
    size_t reversed = 0;
    for (size_t i = 0; i < bits; ++i) {
        reversed = (reversed << 1) | (index & 1);
        index >>= 1;
    }
    return reversed;
}

void bit_reverse_group_array(std::vector<Group1>& values)
{
    const size_t n = values.size();
    const size_t bits = log2N(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t j = bit_reverse(i, bits);
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }
}

std::vector<Group1> group_fft_with_root(std::vector<Group1> values, const Fr& omega, bool inverse)
{
    const size_t n = values.size();
    if (n == 0 || !isPowerOfTwo(n)) {
        throw std::runtime_error("kzg: group FFT requires non-zero power-of-two size");
    }
    bit_reverse_group_array(values);

    for (size_t len = 1; len < n; len <<= 1) {
        const size_t step = n / (len << 1);
        const Fr stage_root = pow_fr(omega, step);
        for (size_t start = 0; start < n; start += (len << 1)) {
            Fr twiddle = Fr(1);
            for (size_t offset = 0; offset < len; ++offset) {
                const Group1 even = values[start + offset];
                const Group1 odd = twiddle * values[start + offset + len];
                values[start + offset] = even + odd;
                values[start + offset + len] = even - odd;
                Fr::mul(twiddle, twiddle, stage_root);
            }
        }
    }

    if (inverse) {
        Fr inv_n;
        Fr n_fr;
        n_fr.setStr(std::to_string(n), 10);
        Fr::inv(inv_n, n_fr);
        for (Group1& value : values) {
            value = inv_n * value;
        }
    }
    return values;
}

std::vector<Group1> group_fft(std::vector<Group1> values)
{
    return group_fft_with_root(std::move(values), get_root_of_unity(values.size()), false);
}

std::vector<Group1> group_ifft(std::vector<Group1> values)
{
    Fr omega = get_root_of_unity(values.size());
    Fr inv_omega;
    Fr::inv(inv_omega, omega);
    return group_fft_with_root(std::move(values), inv_omega, true);
}

std::optional<size_t> add_degree_bounds(size_t left, size_t right)
{
    if (left > std::numeric_limits<size_t>::max() - right) {
        return std::nullopt;
    }
    return left + right;
}

std::optional<size_t> multiply_degree_bound(size_t degree, size_t factor)
{
    if (degree == 0 || factor == 0) {
        return size_t(0);
    }
    if (degree > std::numeric_limits<size_t>::max() / factor) {
        return std::nullopt;
    }
    return degree * factor;
}

std::optional<Fr> recorded_opening_value(const std::vector<Fr>& points,
                                         const std::vector<Fr>& values,
                                         const Fr& x)
{
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i] == x) {
            return values[i];
        }
    }
    return std::nullopt;
}

std::optional<size_t> known_degree_bound(
    const virCommitment& V,
    const std::function<std::optional<size_t>(const kzgCommitment&)>& kzg_degree_bound)
{
    switch (V.type) {
    case virCommitment::Type::KZG:
        return kzg_degree_bound(V.com);
    case virCommitment::Type::SPARSE_POLY:
        return V.leaf.degree();
    case virCommitment::Type::LAGRANGE:
        if (V.lagrange_subgroup_size == 0) {
            throw std::runtime_error("known_degree_bound: LAGRANGE subgroup size must be non-zero");
        }
        return V.lagrange_subgroup_size - 1;
    case virCommitment::Type::LAGRANGE_LINEAR_COMBINATION:
        if (V.lagrange_subgroup_size == 0) {
            throw std::runtime_error(
                "known_degree_bound: LAGRANGE_LINEAR_COMBINATION subgroup size must be non-zero");
        }
        return V.lagrange_terms.empty() ? size_t(0) : V.lagrange_subgroup_size - 1;
    case virCommitment::Type::SUM: {
        if (!V.left || !V.right) {
            throw std::runtime_error("known_degree_bound: SUM commitment has null child");
        }
        const auto left = known_degree_bound(*V.left, kzg_degree_bound);
        const auto right = known_degree_bound(*V.right, kzg_degree_bound);
        if (!left || !right) {
            return std::nullopt;
        }
        return std::max(*left, *right);
    }
    case virCommitment::Type::PRODUCT: {
        if (!V.left || !V.right) {
            throw std::runtime_error("known_degree_bound: PRODUCT commitment has null child");
        }
        const auto left = known_degree_bound(*V.left, kzg_degree_bound);
        const auto right = known_degree_bound(*V.right, kzg_degree_bound);
        if (!left || !right) {
            return std::nullopt;
        }
        return add_degree_bounds(*left, *right);
    }
    case virCommitment::Type::FRACTION: {
        if (!V.left || !V.right) {
            throw std::runtime_error("known_degree_bound: FRACTION commitment has null child");
        }
        const auto numerator = known_degree_bound(*V.left, kzg_degree_bound);
        const auto denominator = known_degree_bound(*V.right, kzg_degree_bound);
        if (!numerator || !denominator || *denominator != 0) {
            return std::nullopt;
        }
        return numerator;
    }
    case virCommitment::Type::X_POWER: {
        if (!V.left) {
            throw std::runtime_error("known_degree_bound: X_POWER commitment has null child");
        }
        if (V.exponent == 0) {
            return size_t(0);
        }
        const auto base = known_degree_bound(*V.left, kzg_degree_bound);
        if (!base) {
            return std::nullopt;
        }
        return multiply_degree_bound(*base, V.exponent);
    }
    case virCommitment::Type::X_SCALE: {
        if (!V.left) {
            throw std::runtime_error("known_degree_bound: X_SCALE commitment has null child");
        }
        return known_degree_bound(*V.left, kzg_degree_bound);
    }
    case virCommitment::Type::INTERLEAVE: {
        if (!V.left || !V.right) {
            throw std::runtime_error("known_degree_bound: INTERLEAVE commitment has null child");
        }
        if (V.exponent == 0) {
            throw std::runtime_error("known_degree_bound: INTERLEAVE domain must be non-zero");
        }
        const std::optional<size_t> left = known_degree_bound(*V.left, kzg_degree_bound);
        const std::optional<size_t> right = known_degree_bound(*V.right, kzg_degree_bound);
        if (!left || !right) {
            return std::nullopt;
        }
        if (*left > std::numeric_limits<size_t>::max() - V.exponent ||
            *right > std::numeric_limits<size_t>::max() - V.exponent) {
            throw std::runtime_error("known_degree_bound: INTERLEAVE degree bound overflow");
        }
        return std::max(*left, *right) + V.exponent;
    }
    }
    throw std::runtime_error("known_degree_bound: unknown virCommitment type");
}

struct CachedBatchOpeningDomain {
    std::vector<Group1> monomial_prefix_fft;
};

struct CommitmentOpeningSet {
    size_t commitment_index = 0;
    std::vector<Fr> points;
    std::vector<Fr> values;
};

std::string batch_opening_commitment_cache_key(const virCommitment& V,
                                               const std::vector<Group1>& g1_powers,
                                               size_t n)
{
    if (V.type != virCommitment::Type::KZG) {
        return "";
    }
    return std::to_string(n) + ":" + g1_powers[1].value().getStr(16) + ":" + V.com.com.value().getStr(16);
}

CachedSrsPowers make_srs_powers(size_t n1, size_t n2)
{
    Fr tau;
    tau.setByCSPRNG();
    const size_t max_count = std::max(n1, n2);
    std::vector<Fr> tau_powers(max_count);
    tau_powers[0] = Fr(1);
    for (size_t i = 1; i < max_count; ++i) {
        Fr::mul(tau_powers[i], tau_powers[i - 1], tau);
    }

    CachedSrsPowers powers;
    powers.g1_powers.resize(n1);
    powers.g2_powers.resize(n2);
    powers.g1_raw_powers.resize(n1);
    powers.g2_raw_powers.resize(n2);

    const Group1 g1_generator = Group1::one();
    const Group2 g2_generator = Group2::one();
    for (size_t i = 0; i < n1; ++i) {
        powers.g1_powers[i] = tau_powers[i] * g1_generator;
        powers.g1_raw_powers[i] = powers.g1_powers[i].value();
    }
    for (size_t i = 0; i < n2; ++i) {
        powers.g2_powers[i] = tau_powers[i] * g2_generator;
        powers.g2_raw_powers[i] = powers.g2_powers[i].value();
    }
    return powers;
}

const CachedSrsPowers& get_cached_srs_powers(size_t n1, size_t n2)
{
    static std::map<std::pair<size_t, size_t>, CachedSrsPowers> cache;
    const auto key = std::make_pair(n1, n2);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(key, make_srs_powers(n1, n2)).first->second;
}

std::string batch_opening_cache_key(const std::vector<Group1>& g1_powers, size_t n)
{
    return std::to_string(n) + ":" + g1_powers[1].value().getStr(16);
}

poly quotient_single_opening(const poly& f, const Fr& x)
{
    const std::vector<Fr> coeffs = f.coefficients();
    if (coeffs.size() <= 1) {
        return poly();
    }

    const size_t degree = coeffs.size() - 1;
    std::vector<Fr> quotient_coeffs(degree, Fr(0));
    quotient_coeffs[degree - 1] = coeffs[degree];
    for (size_t idx = degree - 1; idx > 0; --idx) {
        Fr term;
        Fr::mul(term, x, quotient_coeffs[idx]);
        Fr::add(quotient_coeffs[idx - 1], coeffs[idx], term);
    }
    return poly_from_coeff(quotient_coeffs);
}

std::vector<CommitmentOpeningSet> collect_commitment_opening_sets(const kzgParams& params,
                                                                  size_t commitment_count)
{
    std::vector<CommitmentOpeningSet> sets;
    sets.reserve(commitment_count);
    for (size_t i = 0; i < commitment_count; ++i) {
        const auto [xs, ys] = unique_openings(params.x[i], params.y[i]);
        if (xs.empty()) {
            continue;
        }
        CommitmentOpeningSet set;
        set.commitment_index = i;
        set.points = xs;
        set.values = ys;
        sets.push_back(std::move(set));
    }
    return sets;
}

std::vector<Fr> collect_union_opening_points(const std::vector<CommitmentOpeningSet>& sets)
{
    std::vector<Fr> union_points;
    for (const CommitmentOpeningSet& set : sets) {
        for (const Fr& point : set.points) {
            if (!has_recorded_opening(union_points, point)) {
                union_points.push_back(point);
            }
        }
    }
    return union_points;
}

const CachedBatchOpeningDomain& get_cached_batch_opening_domain(const std::vector<Group1>& g1_powers, size_t n)
{
    static std::map<std::string, CachedBatchOpeningDomain> cache;
    const std::string key = batch_opening_cache_key(g1_powers, n);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const size_t m = n - 1;
    const size_t fft_size = nextPow2(2 * m - 1);
    CachedBatchOpeningDomain domain;
    domain.monomial_prefix_fft.assign(fft_size, Group1::zero());
    for (size_t i = 0; i < m; ++i) {
        domain.monomial_prefix_fft[i] = g1_powers[i];
    }
    domain.monomial_prefix_fft = group_fft(std::move(domain.monomial_prefix_fft));
    return cache.emplace(key, std::move(domain)).first->second;
}

void sync_raw_power_vectors(kzgPublicParameters& params)
{
    params.g1_raw_powers.resize(params.g1_powers.size());
    for (size_t i = 0; i < params.g1_powers.size(); ++i) {
        params.g1_raw_powers[i] = params.g1_powers[i].value();
    }
    params.g2_raw_powers.resize(params.g2_powers.size());
    for (size_t i = 0; i < params.g2_powers.size(); ++i) {
        params.g2_raw_powers[i] = params.g2_powers[i].value();
    }
}

std::shared_ptr<poly> materialize_x_power_poly(const std::shared_ptr<poly>& base, size_t exponent)
{
    if (exponent == 0) {
        Fr constant = base->eval(Fr(1));
        poly result;
        result.add_term(0, constant);
        return std::make_shared<poly>(result);
    }

    if (base->is_sparse()) {
        poly result;
        for (const auto& [exp, coeff] : base->terms()) {
            result.add_term(exp * exponent, coeff);
        }
        return std::make_shared<poly>(result);
    }

    const std::vector<Fr>& coeffs = base->coefficients();
    if (coeffs.empty()) {
        return std::make_shared<poly>();
    }
    std::vector<Fr> composed((coeffs.size() - 1) * exponent + 1, Fr(0));
    for (size_t i = 0; i < coeffs.size(); ++i) {
        composed[i * exponent] = coeffs[i];
    }
    return std::make_shared<poly>(poly_from_coeff(composed));
}

std::shared_ptr<poly> materialize_x_scale_poly(const std::shared_ptr<poly>& base, const Fr& scale)
{
    return std::make_shared<poly>(base->scale_argument(scale));
}

std::shared_ptr<poly> materialize_interleave_poly(const std::shared_ptr<poly>& even,
                                                  const std::shared_ptr<poly>& odd,
                                                  size_t half_domain)
{
    if (!isPowerOfTwo(half_domain) || half_domain == 0) {
        throw std::runtime_error("materialize_interleave_poly: half-domain must be a non-zero power of two");
    }

    const std::vector<Fr>& even_coeffs = even->coefficients();
    const std::vector<Fr>& odd_coeffs = odd->coefficients();
    const size_t max_child_coeffs = std::max(even_coeffs.size(), odd_coeffs.size());
    if (max_child_coeffs == 0) {
        return std::make_shared<poly>();
    }

    std::vector<Fr> out(max_child_coeffs + half_domain, Fr(0));
    const Fr half = inverse_of_two();
    const Fr odd_shift = inverse_root_of_unity(2 * half_domain);
    Fr odd_shift_power(1);
    for (size_t i = 0; i < max_child_coeffs; ++i) {
        if (i < even_coeffs.size()) {
            Fr term;
            Fr::mul(term, even_coeffs[i], half);
            Fr::add(out[i], out[i], term);
            Fr::add(out[i + half_domain], out[i + half_domain], term);
        }
        if (i < odd_coeffs.size()) {
            Fr term;
            Fr::mul(term, odd_coeffs[i], odd_shift_power);
            Fr::mul(term, term, half);
            Fr::add(out[i], out[i], term);
            Fr::sub(out[i + half_domain], out[i + half_domain], term);
        }
        Fr::mul(odd_shift_power, odd_shift_power, odd_shift);
    }

    return std::make_shared<poly>(poly_from_coeff(out));
}

std::shared_ptr<poly> materialize_virtual_poly(
    const virCommitment& V,
    const std::function<std::shared_ptr<poly>(const kzgCommitment&)>& materialize_kzg)
{
    if (V.materialized_cache != nullptr) {
        return V.materialized_cache;
    }

    std::shared_ptr<poly> result;
    switch (V.type) {
    case virCommitment::Type::KZG:
        result = materialize_kzg(V.com);
        break;
    case virCommitment::Type::SPARSE_POLY:
        result = std::make_shared<poly>(V.leaf);
        break;
    case virCommitment::Type::LAGRANGE:
        result = std::make_shared<poly>(subgroup_lagrange_polynomial(V.lagrange_subgroup_size, V.lagrange_index));
        break;
    case virCommitment::Type::LAGRANGE_LINEAR_COMBINATION: {
        poly combined;
        for (const auto& [index, coefficient] : V.lagrange_terms) {
            combined = combined + subgroup_lagrange_polynomial(V.lagrange_subgroup_size, index) * coefficient;
        }
        result = std::make_shared<poly>(combined);
        break;
    }
    case virCommitment::Type::SUM: {
        if (!V.left || !V.right) {
            throw std::runtime_error("materialize: SUM commitment has null child");
        }
        std::shared_ptr<poly> left = materialize_virtual_poly(*V.left, materialize_kzg);
        std::shared_ptr<poly> right = materialize_virtual_poly(*V.right, materialize_kzg);
        result = std::make_shared<poly>(*left + *right);
        break;
    }
    case virCommitment::Type::PRODUCT: {
        if (!V.left || !V.right) {
            throw std::runtime_error("materialize: PRODUCT commitment has null child");
        }
        std::shared_ptr<poly> left = materialize_virtual_poly(*V.left, materialize_kzg);
        std::shared_ptr<poly> right = materialize_virtual_poly(*V.right, materialize_kzg);
        result = std::make_shared<poly>(*left * *right);
        break;
    }
    case virCommitment::Type::FRACTION: {
        if (!V.left || !V.right) {
            throw std::runtime_error("materialize: FRACTION commitment has null child");
        }
        std::shared_ptr<poly> left = materialize_virtual_poly(*V.left, materialize_kzg);
        std::shared_ptr<poly> right = materialize_virtual_poly(*V.right, materialize_kzg);
        result = std::make_shared<poly>(*left / *right);
        break;
    }
    case virCommitment::Type::X_POWER: {
        if (!V.left) {
            throw std::runtime_error("materialize: X_POWER commitment has null child");
        }
        std::shared_ptr<poly> base = materialize_virtual_poly(*V.left, materialize_kzg);
        result = materialize_x_power_poly(base, V.exponent);
        break;
    }
    case virCommitment::Type::X_SCALE: {
        if (!V.left) {
            throw std::runtime_error("materialize: X_SCALE commitment has null child");
        }
        std::shared_ptr<poly> base = materialize_virtual_poly(*V.left, materialize_kzg);
        result = materialize_x_scale_poly(base, V.scale);
        break;
    }
    case virCommitment::Type::INTERLEAVE: {
        if (!V.left || !V.right) {
            throw std::runtime_error("materialize: INTERLEAVE commitment has null child");
        }
        const size_t half_domain = V.exponent;
        if (!isPowerOfTwo(half_domain) || half_domain == 0) {
            throw std::runtime_error("materialize: INTERLEAVE domain must be a non-zero power of two");
        }
        std::shared_ptr<poly> even = materialize_virtual_poly(*V.left, materialize_kzg);
        std::shared_ptr<poly> odd = materialize_virtual_poly(*V.right, materialize_kzg);
        result = materialize_interleave_poly(even, odd, half_domain);
        break;
    }
    }
    if (result == nullptr) {
        throw std::runtime_error("materialize: unknown virCommitment type");
    }
    V.materialized_cache = result;
    return result;
}

bool lookup_root_of_unity_shift(const Fr& scale, size_t n, size_t& shift)
{
    if (n == 0) {
        return false;
    }
    if (n == 1) {
        if (scale == Fr(1)) {
            shift = 0;
            return true;
        }
        return false;
    }

    static std::map<std::string, std::optional<size_t>> cache;
    const std::string key = std::to_string(n) + ":" + scale.getStr(16);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        if (!it->second.has_value()) {
            return false;
        }
        shift = *it->second;
        return true;
    }

    const Fr omega = get_root_of_unity(n);
    Fr current(1);
    for (size_t i = 0; i < n; ++i) {
        if (current == scale) {
            cache[key] = i;
            shift = i;
            return true;
        }
        current *= omega;
    }
    cache[key] = std::nullopt;
    return false;
}

Fr size_to_fr(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

Fr eval_lagrange_basis_at_point(size_t subgroup_size, size_t index, const Fr& x)
{
    const Fr label = pow_fr(get_root_of_unity(subgroup_size), index);
    Fr inverse_label;
    Fr::inv(inverse_label, label);

    Fr denominator = inverse_label * x - Fr(1);
    if (denominator.isZero()) {
        return Fr(1);
    }

    Fr numerator = pow_fr(x, subgroup_size) - Fr(1);
    return numerator / (size_to_fr(subgroup_size) * denominator);
}

Fr eval_lagrange_terms_at_point(size_t subgroup_size,
                                const std::vector<std::pair<size_t, Fr>>& terms,
                                const Fr& x)
{
    if (terms.empty()) {
        return Fr(0);
    }
    const Fr omega = get_root_of_unity(subgroup_size);
    Fr inverse_omega;
    Fr::inv(inverse_omega, omega);
    const Fr numerator = pow_fr(x, subgroup_size) - Fr(1);
    const Fr subgroup_size_fr = size_to_fr(subgroup_size);
    Fr inv_subgroup_size;
    Fr::inv(inv_subgroup_size, subgroup_size_fr);
    const Fr scaled_numerator = numerator * inv_subgroup_size;
    size_t previous_index = terms.front().first;
    Fr inverse_label = pow_fr(inverse_omega, previous_index);
    std::vector<Fr> denominators;
    std::vector<Fr> coefficients;
    denominators.reserve(terms.size());
    coefficients.reserve(terms.size());
    Fr out(0);
    for (const auto& [index, coefficient] : terms) {
        if (index >= previous_index) {
            const size_t delta = index - previous_index;
            if (delta == 1) {
                inverse_label *= inverse_omega;
            } else if (delta != 0) {
                inverse_label *= pow_fr(inverse_omega, delta);
            }
        } else {
            inverse_label = pow_fr(inverse_omega, index);
        }
        previous_index = index;

        const Fr denominator = inverse_label * x - Fr(1);
        if (denominator.isZero()) {
            out += coefficient;
            continue;
        }

        denominators.push_back(denominator);
        coefficients.push_back(coefficient);
    }

    if (denominators.empty()) {
        return out;
    }

    std::vector<Fr> prefix_products(denominators.size());
    Fr running = Fr(1);
    for (size_t i = 0; i < denominators.size(); ++i) {
        prefix_products[i] = running;
        running *= denominators[i];
    }
    Fr inverse_running;
    Fr::inv(inverse_running, running);
    for (size_t i = denominators.size(); i-- > 0;) {
        const Fr inverse_denominator = inverse_running * prefix_products[i];
        inverse_running *= denominators[i];
        out += coefficients[i] * scaled_numerator * inverse_denominator;
    }
    return out;
}

std::vector<std::pair<size_t, Fr>> lagrange_terms_for_virtual(const virCommitment& base, size_t& subgroup_size)
{
    if (base.type == virCommitment::Type::LAGRANGE) {
        subgroup_size = base.lagrange_subgroup_size;
        return {{base.lagrange_index, Fr(1)}};
    }
    if (base.type == virCommitment::Type::LAGRANGE_LINEAR_COMBINATION) {
        subgroup_size = base.lagrange_subgroup_size;
        return base.lagrange_terms;
    }
    subgroup_size = 0;
    return {};
}

std::vector<Fr> eval_lagrange_terms_on_domain(size_t subgroup_size,
                                              const std::vector<std::pair<size_t, Fr>>& terms,
                                              size_t n)
{
    std::vector<Fr> out(n, Fr(0));
    if (subgroup_size == 0) {
        throw std::runtime_error("eval_lagrange_terms_on_domain: subgroup size must be non-zero");
    }

    if (n == subgroup_size) {
        for (const auto& [index, coefficient] : terms) {
            out[index] += coefficient;
        }
        return out;
    }

    const Fr omega = get_root_of_unity(n);
    Fr point(1);
    for (size_t i = 0; i < n; ++i) {
        out[i] = eval_lagrange_terms_at_point(subgroup_size, terms, point);
        point *= omega;
    }
    return out;
}

bool try_eval_x_power_lagrange_on_domain(const virCommitment& base, size_t exponent, size_t n, std::vector<Fr>& out)
{
    size_t subgroup_size = 0;
    const std::vector<std::pair<size_t, Fr>> terms = lagrange_terms_for_virtual(base, subgroup_size);
    if (subgroup_size == 0 || n % subgroup_size != 0 || !isPowerOfTwo(n)) {
        return false;
    }
    const size_t subgroup_step = n / subgroup_size;
    if (subgroup_step == 0 || exponent % subgroup_step != 0) {
        return false;
    }

    std::map<size_t, Fr> value_by_index;
    for (const auto& [index, coefficient] : terms) {
        auto [it, inserted] = value_by_index.emplace(index, Fr(0));
        (void)inserted;
        it->second += coefficient;
    }

    out.assign(n, Fr(0));
    const size_t mask = n - 1;
    for (size_t i = 0; i < n; ++i) {
        const size_t mapped_index = (i * exponent) & mask;
        const size_t subgroup_index = mapped_index / subgroup_step;
        const auto term = value_by_index.find(subgroup_index);
        if (term != value_by_index.end()) {
            out[i] = term->second;
        }
    }
    return true;
}

std::vector<Fr> eval_virtual_on_domain(
    const virCommitment& V,
    size_t n,
    const std::function<std::shared_ptr<poly>(const kzgCommitment&)>& materialize_kzg,
    const std::function<std::vector<Fr>(const kzgCommitment&, size_t)>& evaluate_kzg,
    std::map<std::pair<const virCommitment*, size_t>, std::vector<Fr>>& cache)
{
    const auto cache_key = std::make_pair(&V, n);
    const auto it = cache.find(cache_key);
    if (it != cache.end()) {
        return it->second;
    }
    const auto cached_evaluations = V.evaluation_cache.find(n);
    if (cached_evaluations != V.evaluation_cache.end()) {
        cache.emplace(cache_key, *cached_evaluations->second);
        return *cached_evaluations->second;
    }

    std::vector<Fr> out;
    switch (V.type) {
    case virCommitment::Type::KZG:
        out = evaluate_kzg(V.com, n);
        break;
    case virCommitment::Type::SPARSE_POLY:
        out = V.leaf.evaluations_on_domain(n);
        break;
    case virCommitment::Type::LAGRANGE: {
        out = eval_lagrange_terms_on_domain(V.lagrange_subgroup_size, {{V.lagrange_index, Fr(1)}}, n);
        break;
    }
    case virCommitment::Type::LAGRANGE_LINEAR_COMBINATION: {
        out = eval_lagrange_terms_on_domain(V.lagrange_subgroup_size, V.lagrange_terms, n);
        break;
    }
    case virCommitment::Type::SUM: {
        if (!V.left || !V.right) {
            throw std::runtime_error("eval_virtual_on_domain: SUM commitment has null child");
        }
        const std::vector<Fr> left = eval_virtual_on_domain(*V.left, n, materialize_kzg, evaluate_kzg, cache);
        const std::vector<Fr> right = eval_virtual_on_domain(*V.right, n, materialize_kzg, evaluate_kzg, cache);
        out.resize(n, Fr(0));
        for (size_t i = 0; i < n; ++i) {
            out[i] = left[i] + right[i];
        }
        break;
    }
    case virCommitment::Type::PRODUCT: {
        if (!V.left || !V.right) {
            throw std::runtime_error("eval_virtual_on_domain: PRODUCT commitment has null child");
        }
        const std::vector<Fr> left = eval_virtual_on_domain(*V.left, n, materialize_kzg, evaluate_kzg, cache);
        const std::vector<Fr> right = eval_virtual_on_domain(*V.right, n, materialize_kzg, evaluate_kzg, cache);
        out.resize(n, Fr(0));
        for (size_t i = 0; i < n; ++i) {
            out[i] = left[i] * right[i];
        }
        break;
    }
    case virCommitment::Type::FRACTION: {
        if (!V.left || !V.right) {
            throw std::runtime_error("eval_virtual_on_domain: FRACTION commitment has null child");
        }
        const std::vector<Fr> left = eval_virtual_on_domain(*V.left, n, materialize_kzg, evaluate_kzg, cache);
        const std::vector<Fr> right = eval_virtual_on_domain(*V.right, n, materialize_kzg, evaluate_kzg, cache);
        out.resize(n, Fr(0));
        for (size_t i = 0; i < n; ++i) {
            if (right[i].isZero()) {
                throw std::runtime_error("eval_virtual_on_domain: division by zero in FRACTION");
            }
            out[i] = left[i] / right[i];
        }
        break;
    }
    case virCommitment::Type::X_POWER: {
        if (!V.left) {
            throw std::runtime_error("eval_virtual_on_domain: X_POWER commitment has null child");
        }
        if (n == 0) {
            throw std::runtime_error("eval_virtual_on_domain: domain size must be non-zero");
        }
        const size_t exponent = V.exponent % n;
        if (try_eval_x_power_lagrange_on_domain(*V.left, exponent, n, out)) {
            break;
        }
        const std::vector<Fr> base = eval_virtual_on_domain(*V.left, n, materialize_kzg, evaluate_kzg, cache);
        out.resize(n, Fr(0));
        if (exponent == 0) {
            std::fill(out.begin(), out.end(), base[0]);
            break;
        }
        const size_t mask = n - 1;
        for (size_t i = 0; i < n; ++i) {
            const size_t index = (i * exponent) & mask;
            out[i] = base[index];
        }
        break;
    }
    case virCommitment::Type::X_SCALE: {
        if (!V.left) {
            throw std::runtime_error("eval_virtual_on_domain: X_SCALE commitment has null child");
        }
        size_t shift = 0;
        if (!lookup_root_of_unity_shift(V.scale, n, shift)) {
            out = materialize_virtual_poly(V, materialize_kzg)->evaluations_on_domain(n);
            break;
        }
        const std::vector<Fr> base = eval_virtual_on_domain(*V.left, n, materialize_kzg, evaluate_kzg, cache);
        out.resize(n, Fr(0));
        for (size_t i = 0; i < n; ++i) {
            out[i] = base[(i + shift) % n];
        }
        break;
    }
    case virCommitment::Type::INTERLEAVE: {
        if (!V.left || !V.right) {
            throw std::runtime_error("eval_virtual_on_domain: INTERLEAVE commitment has null child");
        }
        const size_t half_domain = V.exponent;
        if (!isPowerOfTwo(half_domain) || half_domain == 0) {
            throw std::runtime_error("eval_virtual_on_domain: INTERLEAVE domain must be a non-zero power of two");
        }
        if (n != 2 * half_domain) {
            out = materialize_virtual_poly(V, materialize_kzg)->evaluations_on_domain(n);
            break;
        }
        const std::vector<Fr> even = eval_virtual_on_domain(*V.left, half_domain, materialize_kzg, evaluate_kzg, cache);
        const std::vector<Fr> odd = eval_virtual_on_domain(*V.right, half_domain, materialize_kzg, evaluate_kzg, cache);
        out.resize(n, Fr(0));
        for (size_t i = 0; i < half_domain; ++i) {
            out[2 * i] = even[i];
            out[2 * i + 1] = odd[i];
        }
        break;
    }
    }

    auto stored = std::make_shared<const std::vector<Fr>>(std::move(out));
    V.evaluation_cache[n] = stored;
    cache.emplace(cache_key, *stored);
    return *stored;
}

template <class GWrapper, class GRaw>
GWrapper msm_pippenger_fallback(const std::vector<Fr>& scalars,
                                const std::vector<GWrapper>& bases,
                                size_t base_offset = 0)
{
    if (base_offset > bases.size()) {
        throw std::runtime_error("msm_pippenger: base offset exceeds available bases");
    }
    const size_t n = std::min(scalars.size(), bases.size() - base_offset);
    if (n == 0) {
        return GWrapper::zero();
    }

    auto hex_char_to_u8 = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + c - 'a');
        if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + c - 'A');
        throw std::runtime_error("msm_pippenger: non-hex scalar encoding");
    };

    // Store each scalar as little-endian nibbles for cheap bit-window extraction.
    std::vector<std::vector<uint8_t>> scalar_nibbles;
    std::vector<GWrapper> non_zero_bases;
    scalar_nibbles.reserve(n);
    non_zero_bases.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (scalars[i].isZero()) {
            continue;
        }

        // Use canonical field-value hex string (not serialized bytes/internal form).
        std::string hex = scalars[i].getStr(16);
        if (hex.empty()) {
            continue;
        }
        if (hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) {
            hex.erase(0, 2);
        }
        if (!hex.empty() && hex[0] == '-') {
            throw std::runtime_error("msm_pippenger: negative scalar not supported");
        }

        std::vector<uint8_t> nibs;
        nibs.reserve(hex.size());
        for (size_t j = hex.size(); j > 0; --j) {
            nibs.push_back(hex_char_to_u8(hex[j - 1]));
        }
        scalar_nibbles.push_back(std::move(nibs));
        non_zero_bases.push_back(bases[base_offset + i]);
    }

    if (non_zero_bases.empty()) {
        return GWrapper::zero();
    }

    size_t max_bits = 0;
    for (const auto& nibs : scalar_nibbles) {
        if (nibs.empty()) continue;
        size_t top = nibs.size() - 1;
        while (top > 0 && nibs[top] == 0) {
            --top;
        }
        uint8_t ms = nibs[top];
        size_t ms_bits = 0;
        while (ms > 0) {
            ++ms_bits;
            ms >>= 1;
        }
        max_bits = std::max(max_bits, top * 4 + ms_bits);
    }
    if (max_bits == 0) {
        return GWrapper::zero();
    }

    const size_t nz = non_zero_bases.size();
    size_t window = 4;
    if (nz >= 1024) window = 13;
    else if (nz >= 512) window = 12;
    else if (nz >= 256) window = 10;
    else if (nz >= 128) window = 8;
    else if (nz >= 64) window = 7;
    else if (nz >= 32) window = 6;
    else if (nz >= 16) window = 5;

    const size_t bucket_count = size_t(1) << window;
    const size_t num_windows = (max_bits + window - 1) / window;

    auto extract_window = [](const std::vector<uint8_t>& nibs, size_t bit_offset, size_t width) -> size_t {
        size_t v = 0;
        for (size_t b = 0; b < width; ++b) {
            const size_t bit = bit_offset + b;
            const size_t nib_idx = bit >> 2;
            if (nib_idx >= nibs.size()) {
                break;
            }
            const uint8_t nib = nibs[nib_idx];
            const uint8_t bit_val = (nib >> (bit & 3)) & 1;
            v |= (static_cast<size_t>(bit_val) << b);
        }
        return v;
    };

    GWrapper acc = GWrapper::zero();
    for (size_t wi = num_windows; wi > 0; --wi) {
        if (wi != num_windows) {
            for (size_t i = 0; i < window; ++i) {
                acc = acc + acc;
            }
        }

        std::vector<GWrapper> buckets(bucket_count, GWrapper::zero());
        const size_t bit_offset = (wi - 1) * window;
        for (size_t i = 0; i < nz; ++i) {
            const size_t digit = extract_window(scalar_nibbles[i], bit_offset, window);
            if (digit != 0) {
                buckets[digit] += non_zero_bases[i];
            }
        }

        GWrapper running = GWrapper::zero();
        GWrapper window_sum = GWrapper::zero();
        for (size_t b = bucket_count - 1; b > 0; --b) {
            running += buckets[b];
            window_sum += running;
        }
        acc += window_sum;
    }
    return acc;
}

template <class GWrapper, class GRaw>
GWrapper msm_dispatch(const std::vector<Fr>& scalars,
                      const std::vector<GWrapper>& wrapped_bases,
                      const std::vector<GRaw>& raw_bases,
                      size_t base_offset = 0)
{
    if (base_offset > wrapped_bases.size() || scalars.size() > wrapped_bases.size() - base_offset) {
        throw std::runtime_error(
            "msm_dispatch: scalar count exceeds available SRS powers: scalars=" +
            std::to_string(scalars.size()) + " bases=" + std::to_string(wrapped_bases.size()));
    }
    const size_t n = scalars.size();
    if constexpr (std::is_same_v<GWrapper, Group1>) {
    } else if constexpr (std::is_same_v<GWrapper, Group2>) {
    }
    if (n == 0) {
        return GWrapper::zero();
    }
    if (base_offset <= raw_bases.size() && raw_bases.size() - base_offset >= n && n >= 16) {
        GRaw out;
        GRaw::mulVec(out, const_cast<GRaw*>(raw_bases.data() + base_offset), scalars.data(), n);
        return GWrapper(out);
    }
    return msm_pippenger_fallback<GWrapper, GRaw>(scalars, wrapped_bases, base_offset);
}

} // namespace

kzgPublicParameters generate_kzg_public_parameters(size_t deg1_plus_1, size_t deg2_plus_1)
{
    if (deg1_plus_1 == 0 || deg2_plus_1 == 0) {
        throw std::runtime_error("generate_kzg_public_parameters: SRS sizes must be non-zero");
    }

    const CachedSrsPowers powers = make_srs_powers(deg1_plus_1, deg2_plus_1);
    kzgPublicParameters params;
    params.deg1 = deg1_plus_1 - 1;
    params.deg2 = deg2_plus_1 - 1;
    params.g1_powers = powers.g1_powers;
    params.g2_powers = powers.g2_powers;
    params.g1_raw_powers = powers.g1_raw_powers;
    params.g2_raw_powers = powers.g2_raw_powers;
    return params;
}

kzgPublicParameters trim_kzg_public_parameters(const kzgPublicParameters& params,
                                              size_t deg1_plus_1,
                                              size_t deg2_plus_1)
{
    if (deg1_plus_1 == 0 || deg2_plus_1 == 0) {
        throw std::runtime_error("trim_kzg_public_parameters: SRS sizes must be non-zero");
    }
    if (params.g1_powers.size() < deg1_plus_1 || params.g2_powers.size() < deg2_plus_1) {
        throw std::runtime_error("trim_kzg_public_parameters: input SRS is too small");
    }

    kzgPublicParameters trimmed;
    trimmed.deg1 = deg1_plus_1 - 1;
    trimmed.deg2 = deg2_plus_1 - 1;
    trimmed.g1_powers.assign(params.g1_powers.begin(), params.g1_powers.begin() + deg1_plus_1);
    trimmed.g2_powers.assign(params.g2_powers.begin(), params.g2_powers.begin() + deg2_plus_1);
    sync_raw_power_vectors(trimmed);
    return trimmed;
}

void write_kzg_public_parameters(const kzgPublicParameters& params, const std::string& path)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("write_kzg_public_parameters: failed to open output file: " + path);
    }

    out << "deg1 " << params.deg1 << "\n";
    out << "deg2 " << params.deg2 << "\n";
    out << "g1 " << params.g1_powers.size();
    for (const Group1& value : params.g1_powers) {
        out << " " << proof_encoding(value);
    }
    out << "\n";

    out << "g2 " << params.g2_powers.size();
    for (const Group2& value : params.g2_powers) {
        out << " " << proof_encoding(value);
    }
    out << "\n";
}

kzgPublicParameters read_kzg_public_parameters(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("read_kzg_public_parameters: failed to open input file: " + path);
    }

    kzgPublicParameters params;
    std::string label;

    if (!(in >> label >> params.deg1) || label != "deg1") {
        throw std::runtime_error("read_kzg_public_parameters: missing deg1");
    }
    if (!(in >> label >> params.deg2) || label != "deg2") {
        throw std::runtime_error("read_kzg_public_parameters: missing deg2");
    }

    size_t g1_size = 0;
    if (!(in >> label >> g1_size) || label != "g1") {
        throw std::runtime_error("read_kzg_public_parameters: missing g1 powers");
    }
    params.g1_powers.resize(g1_size);
    for (size_t i = 0; i < g1_size; ++i) {
        std::string encoded;
        if (!(in >> encoded)) {
            throw std::runtime_error("read_kzg_public_parameters: truncated g1 powers");
        }
        params.g1_powers[i] = decode_group1(encoded);
    }

    size_t g2_size = 0;
    if (!(in >> label >> g2_size) || label != "g2") {
        throw std::runtime_error("read_kzg_public_parameters: missing g2 powers");
    }
    params.g2_powers.resize(g2_size);
    for (size_t i = 0; i < g2_size; ++i) {
        std::string encoded;
        if (!(in >> encoded)) {
            throw std::runtime_error("read_kzg_public_parameters: truncated g2 powers");
        }
        params.g2_powers[i] = decode_group2(encoded);
    }

    if (params.g1_powers.empty() || params.g2_powers.empty()) {
        throw std::runtime_error("read_kzg_public_parameters: SRS vectors must be non-empty");
    }
    if (params.g1_powers.size() != params.deg1 + 1 || params.g2_powers.size() != params.deg2 + 1) {
        throw std::runtime_error("read_kzg_public_parameters: size metadata does not match SRS vectors");
    }
    sync_raw_power_vectors(params);
    return params;
}

kzgPublicParameters read_kzg_public_parameters(const std::string& path,
                                               size_t deg1_plus_1,
                                               size_t deg2_plus_1)
{
    if (deg1_plus_1 == 0 || deg2_plus_1 == 0) {
        throw std::runtime_error("read_kzg_public_parameters: requested SRS sizes must be non-zero");
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("read_kzg_public_parameters: failed to open input file: " + path);
    }

    size_t file_deg1 = 0;
    size_t file_deg2 = 0;
    std::string label;

    if (!(in >> label >> file_deg1) || label != "deg1") {
        throw std::runtime_error("read_kzg_public_parameters: missing deg1");
    }
    if (!(in >> label >> file_deg2) || label != "deg2") {
        throw std::runtime_error("read_kzg_public_parameters: missing deg2");
    }

    size_t g1_size = 0;
    if (!(in >> label >> g1_size) || label != "g1") {
        throw std::runtime_error("read_kzg_public_parameters: missing g1 powers");
    }
    if (g1_size != file_deg1 + 1) {
        throw std::runtime_error("read_kzg_public_parameters: deg1 metadata does not match g1 vector size");
    }
    if (g1_size < deg1_plus_1) {
        throw std::runtime_error("read_kzg_public_parameters: loaded KZG SRS is too small; need deg1+1 >= " +
                                 std::to_string(deg1_plus_1) + ", deg2+1 >= " +
                                 std::to_string(deg2_plus_1) + ", but got deg1+1 = " +
                                 std::to_string(g1_size) + ", deg2+1 = " +
                                 std::to_string(file_deg2 + 1));
    }

    const size_t decode_floor = kzg_srs_prefix_decode_floor();
    const size_t g1_decode_size = std::min(g1_size, std::max(deg1_plus_1, decode_floor));

    kzgPublicParameters params;
    params.deg1 = g1_decode_size - 1;
    params.g1_powers.resize(g1_decode_size);
    for (size_t i = 0; i < g1_size; ++i) {
        std::string encoded;
        if (!(in >> encoded)) {
            throw std::runtime_error("read_kzg_public_parameters: truncated g1 powers");
        }
        if (i < g1_decode_size) {
            params.g1_powers[i] = decode_group1(encoded);
        }
    }

    size_t g2_size = 0;
    if (!(in >> label >> g2_size) || label != "g2") {
        throw std::runtime_error("read_kzg_public_parameters: missing g2 powers");
    }
    if (g2_size != file_deg2 + 1) {
        throw std::runtime_error("read_kzg_public_parameters: deg2 metadata does not match g2 vector size");
    }
    if (g2_size < deg2_plus_1) {
        throw std::runtime_error("read_kzg_public_parameters: loaded KZG SRS is too small; need deg1+1 >= " +
                                 std::to_string(deg1_plus_1) + ", deg2+1 >= " +
                                 std::to_string(deg2_plus_1) + ", but got deg1+1 = " +
                                 std::to_string(g1_size) + ", deg2+1 = " +
                                 std::to_string(g2_size));
    }
    const size_t g2_decode_size = std::min(g2_size, std::max(deg2_plus_1, decode_floor));
    params.deg2 = g2_decode_size - 1;
    params.g2_powers.resize(g2_decode_size);
    for (size_t i = 0; i < g2_size; ++i) {
        std::string encoded;
        if (!(in >> encoded)) {
            throw std::runtime_error("read_kzg_public_parameters: truncated g2 powers");
        }
        if (i < g2_decode_size) {
            params.g2_powers[i] = decode_group2(encoded);
        }
    }

    sync_raw_power_vectors(params);
    return params;
}

Group1 commit_g1_with_public_parameters(const kzgPublicParameters& params, const poly& p)
{
    proof_cost::add("polynomial_commitment_operation_count");
    proof_cost::add("committed_polynomial_coefficient_count", p.coefficients().size());
    return msm_dispatch<Group1, G1>(p.coefficients(), params.g1_powers, params.g1_raw_powers);
}

Group2 commit_g2_with_public_parameters(const kzgPublicParameters& params, const poly& p)
{
    return msm_dispatch<Group2, G2>(p.coefficients(), params.g2_powers, params.g2_raw_powers);
}

std::string transcript_representation(const kzgCommitment& com)
{
    return transcript_representation(com.com);
}

kzgCommitment kzgProver::commit(const poly& p, size_t index)
{
    kzgCommitment com;
    proof_cost::add("polynomial_commitment_operation_count");
    proof_cost::add("committed_polynomial_coefficient_count", p.coefficients().size());
    com.com = msm_dispatch<Group1, G1>(p.coefficients(), params->g1_powers_view(), params->g1_raw_powers_view());
    com.index = index;
    return com;
}

struct EvalVirtualCache {
    std::map<std::pair<Fr, size_t>, Fr> powers;
    std::map<size_t, Fr> inverse_roots;
};

Fr cached_power(EvalVirtualCache& cache, const Fr& base, size_t exponent)
{
    const auto key = std::make_pair(base, exponent);
    const auto it = cache.powers.find(key);
    if (it != cache.powers.end()) {
        return it->second;
    }
    return cache.powers.emplace(key, pow_fr(base, exponent)).first->second;
}

Fr cached_inverse_root(EvalVirtualCache& cache, size_t n)
{
    const auto it = cache.inverse_roots.find(n);
    if (it != cache.inverse_roots.end()) {
        return it->second;
    }
    return cache.inverse_roots.emplace(n, inverse_root_of_unity(n)).first->second;
}

using VirtualOpeningCallback =
    std::function<void(size_t, const Fr&, const kzgOpeningProvenance&)>;

Fr eval_virtual_cached(const virCommitment& V, Fr x,
                       const std::function<Fr(size_t, const Fr&)>& get_kzg_value,
                       std::map<std::pair<size_t, Fr>, Fr>& kzg_openings,
                       EvalVirtualCache& cache,
                       const kzgOpeningProvenance& provenance,
                       const VirtualOpeningCallback* on_opening)
{
    switch (V.type) {
    case virCommitment::Type::KZG: {
        if (on_opening != nullptr) {
            (*on_opening)(V.com.index, x, provenance);
        }
        const auto existing = kzg_openings.find({V.com.index, x});
        if (existing != kzg_openings.end()) {
            return existing->second;
        }
        Fr y = get_kzg_value(V.com.index, x);
        kzg_openings[{V.com.index, x}] = y;
        return y;
    }
    case virCommitment::Type::SPARSE_POLY:
        return V.leaf.eval(x);
    case virCommitment::Type::LAGRANGE:
        return eval_lagrange_basis_at_point(V.lagrange_subgroup_size, V.lagrange_index, x);
    case virCommitment::Type::LAGRANGE_LINEAR_COMBINATION:
        return eval_lagrange_terms_at_point(V.lagrange_subgroup_size, V.lagrange_terms, x);
    case virCommitment::Type::SUM: {
        Fr a = eval_virtual_cached(*V.left, x, get_kzg_value, kzg_openings, cache, provenance, on_opening);
        Fr b = eval_virtual_cached(*V.right, x, get_kzg_value, kzg_openings, cache, provenance, on_opening);
        Fr sum;
        Fr::add(sum, a, b);
        return sum;
    }
    case virCommitment::Type::PRODUCT: {
        Fr a = eval_virtual_cached(*V.left, x, get_kzg_value, kzg_openings, cache, provenance, on_opening);
        Fr b = eval_virtual_cached(*V.right, x, get_kzg_value, kzg_openings, cache, provenance, on_opening);
        Fr prod;
        Fr::mul(prod, a, b);
        return prod;
    }
    case virCommitment::Type::FRACTION: {
        Fr a = eval_virtual_cached(*V.left, x, get_kzg_value, kzg_openings, cache, provenance, on_opening);
        Fr b = eval_virtual_cached(*V.right, x, get_kzg_value, kzg_openings, cache, provenance, on_opening);
        if (b.isZero()) {
            throw std::runtime_error("eval_virtual: division by zero in FRACTION");
        }
        Fr inv_b;
        Fr::inv(inv_b, b);
        Fr quot;
        Fr::mul(quot, a, inv_b);
        return quot;
    }
    case virCommitment::Type::X_POWER: {
        if (!V.left) {
            throw std::runtime_error("eval_virtual: X_POWER commitment has null child");
        }
        Fr powered_x = cached_power(cache, x, V.exponent);
        kzgOpeningProvenance child_provenance = provenance;
        if (V.exponent != 1) {
            ++child_provenance.x_power_count;
        }
        return eval_virtual_cached(
            *V.left, powered_x, get_kzg_value, kzg_openings, cache, child_provenance, on_opening);
    }
    case virCommitment::Type::X_SCALE: {
        if (!V.left) {
            throw std::runtime_error("eval_virtual: X_SCALE commitment has null child");
        }
        kzgOpeningProvenance child_provenance = provenance;
        if (V.scale != Fr(1)) {
            ++child_provenance.x_scale_count;
        }
        return eval_virtual_cached(
            *V.left, V.scale * x, get_kzg_value, kzg_openings, cache, child_provenance, on_opening);
    }
    case virCommitment::Type::INTERLEAVE: {
        if (!V.left || !V.right) {
            throw std::runtime_error("eval_virtual: INTERLEAVE commitment has null child");
        }
        if (V.exponent == 0) {
            throw std::runtime_error("eval_virtual: INTERLEAVE domain must be non-zero");
        }
        const Fr parity = cached_power(cache, x, V.exponent);
        const Fr half = inverse_of_two();
        const Fr even_selector = (parity + Fr(1)) * half;
        const Fr odd_selector = (Fr(1) - parity) * half;
        const Fr odd_shift = cached_inverse_root(cache, 2 * V.exponent);
        kzgOpeningProvenance even_provenance = provenance;
        ++even_provenance.interleave_layer_count;
        kzgOpeningProvenance odd_provenance = even_provenance;
        ++odd_provenance.interleave_shift_count;
        const Fr even_value = eval_virtual_cached(
            *V.left, x, get_kzg_value, kzg_openings, cache, even_provenance, on_opening);
        const Fr odd_value = eval_virtual_cached(
            *V.right, odd_shift * x, get_kzg_value, kzg_openings, cache, odd_provenance, on_opening);
        return even_selector * even_value + odd_selector * odd_value;
    }
    }
    throw std::runtime_error("eval_virtual: unknown virCommitment type");
}

Fr eval_virtual(const virCommitment& V, Fr x,
                const std::function<Fr(size_t, const Fr&)>& get_kzg_value,
                std::map<std::pair<size_t, Fr>, Fr>& kzg_openings)
{
    EvalVirtualCache cache;
    kzgOpeningProvenance provenance;
    provenance.root_point = x;
    return eval_virtual_cached(V, x, get_kzg_value, kzg_openings, cache, provenance, nullptr);
}

void kzgProtocol::record_opening_provenance(size_t commitment_index,
                                            const Fr& actual_point,
                                            const kzgOpeningProvenance& provenance)
{
    auto& traces = params.opening_provenance[{commitment_index, actual_point}];
    const auto same_trace = [&provenance](const kzgOpeningProvenance& existing) {
        return existing.root_point == provenance.root_point &&
               existing.root_role == provenance.root_role &&
               existing.interleave_layer_count == provenance.interleave_layer_count &&
               existing.interleave_shift_count == provenance.interleave_shift_count &&
               existing.x_power_count == provenance.x_power_count &&
               existing.x_scale_count == provenance.x_scale_count;
    };
    if (std::none_of(traces.begin(), traces.end(), same_trace)) {
        traces.push_back(provenance);
    }
}

Fr kzgProtocol::open(const virCommitment& V, Fr x, kzgOpeningRootRole root_role)
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }

    std::map<std::pair<size_t, Fr>, Fr> kzg_openings;
    std::function<Fr(size_t, const Fr&)> get_kzg = [this](size_t index, const Fr& point) {
        const auto existing = recorded_opening_value(params.x[index], params.y[index], point);
        if (existing) {
            return *existing;
        }
        if (proof_session_ && proof_session_->mode() == ProofMode::Verify) {
            return proof_session_->exchange_fr("kzg.opening.value", {});
        }
        if (proof_session_ && proof_session_->mode() == ProofMode::Prove) {
            return proof_session_->exchange_fr("kzg.opening.value", [&] { return prover.polys[index]->eval(point); });
        }
        return prover.polys[index]->eval(point);
    };
    kzgOpeningProvenance provenance;
    provenance.root_point = x;
    provenance.root_role = root_role;
    const VirtualOpeningCallback on_opening =
        [this](size_t index, const Fr& point, const kzgOpeningProvenance& trace) {
            record_opening_provenance(index, point, trace);
        };
    EvalVirtualCache cache;
    Fr value = eval_virtual_cached(V, x, get_kzg, kzg_openings, cache, provenance, &on_opening);

    // Record openings for each KZG polynomial: add the actual queried point.
    for (const auto& [key, y] : kzg_openings) {
        const size_t index = key.first;
        const Fr& point = key.second;
        const auto existing = recorded_opening_value(params.x[index], params.y[index], point);
        if (existing.has_value()) {
            if (*existing != y) {
                throw std::runtime_error(
                    "kzgProtocol::open detected conflicting opening values for the same commitment and point");
            }
            continue;
        }
        params.x[index].push_back(point);
        params.y[index].push_back(y);
    }

    return value;
}

std::shared_ptr<poly> kzgProtocol::materialize(const virCommitment& V) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }
    auto materialize_kzg = [this](const kzgCommitment& com) -> std::shared_ptr<poly> {
        if (!can_materialize(com)) {
            throw std::runtime_error("materialize: KZG commitment is not locally materialized");
        }
        const size_t index = com.index;
        if (index >= prover.polys.size()) {
            throw std::runtime_error("materialize: KZG commitment index out of range");
        }
        const auto it = materialized_kzg_cache_.find(index);
        if (it != materialized_kzg_cache_.end()) {
            return it->second;
        }
        auto materialized = std::make_shared<poly>(*prover.polys[index]);
        materialized_kzg_cache_.emplace(index, materialized);
        return materialized;
    };
    return materialize_virtual_poly(V, materialize_kzg);
}

std::vector<Fr> kzgProtocol::evaluations(const virCommitment& V,
                                         size_t n,
                                         kzgProtocol::EvaluationMode mode) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }
    if (mode == kzgProtocol::EvaluationMode::Materialize || kzg_disable_prefer_eval()) {
        return materialize(V)->evaluations_on_domain(n);
    }

    auto materialize_kzg = [this](const kzgCommitment& com) -> std::shared_ptr<poly> {
        if (!can_materialize(com)) {
            throw std::runtime_error("materialize: KZG commitment is not locally materialized");
        }
        const size_t index = com.index;
        if (index >= prover.polys.size()) {
            throw std::runtime_error("materialize: KZG commitment index out of range");
        }
        const auto it = materialized_kzg_cache_.find(index);
        if (it != materialized_kzg_cache_.end()) {
            return it->second;
        }
        auto materialized = std::make_shared<poly>(*prover.polys[index]);
        materialized_kzg_cache_.emplace(index, materialized);
        return materialized;
    };
    auto evaluate_kzg = [this](const kzgCommitment& com, size_t domain_size) -> std::vector<Fr> {
        if (!can_materialize(com)) {
            throw std::runtime_error("eval_virtual_on_domain: KZG commitment is not locally materialized");
        }
        const size_t index = com.index;
        if (index >= prover.polys.size()) {
            throw std::runtime_error("eval_virtual_on_domain: KZG commitment index out of range");
        }
        const auto key = std::make_pair(index, domain_size);
        const auto it = direct_evaluation_cache_.find(key);
        if (it != direct_evaluation_cache_.end()) {
            return *it->second;
        }
        auto evaluations =
            std::make_shared<const std::vector<Fr>>(prover.polys[index]->evaluations_on_domain(domain_size));
        direct_evaluation_cache_.emplace(key, evaluations);
        return *evaluations;
    };

    std::map<std::pair<const virCommitment*, size_t>, std::vector<Fr>> cache;
    return eval_virtual_on_domain(V, n, materialize_kzg, evaluate_kzg, cache);
}

std::vector<Group1> kzgProtocol::batch_opening_commitments(const poly& p, size_t n) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }
    if (!isPowerOfTwo(n) || n < 2) {
        throw std::runtime_error("kzgProtocol::batch_opening_commitments: n must be a power of two >= 2");
    }

    const size_t m = n - 1;
    const std::vector<Fr> coeffs = p.coefficients();
    std::vector<Fr> reversed_coeffs(m, Fr(0));
    for (size_t u = 0; u < m; ++u) {
        const size_t coeff_index = n - 1 - u;
        if (coeff_index < coeffs.size()) {
            reversed_coeffs[u] = coeffs[coeff_index];
        }
    }

    const CachedBatchOpeningDomain& cached = get_cached_batch_opening_domain(params.g1_powers_view(), n);
    const size_t fft_size = cached.monomial_prefix_fft.size();
    std::vector<Fr> scalar_fft(fft_size, Fr(0));
    for (size_t i = 0; i < reversed_coeffs.size(); ++i) {
        scalar_fft[i] = reversed_coeffs[i];
    }
    ntt(scalar_fft.data(), fft_size, get_root_of_unity(fft_size));

    std::vector<Group1> group_fft_values = cached.monomial_prefix_fft;
    for (size_t i = 0; i < fft_size; ++i) {
        group_fft_values[i] = scalar_fft[i] * group_fft_values[i];
    }
    group_fft_values = group_ifft(std::move(group_fft_values));
    group_fft_values.resize(2 * m - 1);

    std::vector<Group1> quotient_openings(n, Group1::zero());
    for (size_t i = 0; i < m; ++i) {
        quotient_openings[i] = group_fft_values[m - 1 - i];
    }
    return group_fft(std::move(quotient_openings));
}

std::vector<Group1> kzgProtocol::batch_opening_commitments(const virCommitment& V, size_t n) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }

    static std::map<std::string, std::shared_ptr<const std::vector<Group1>>> cache;
    const std::string key = batch_opening_commitment_cache_key(V, params.g1_powers_view(), n);
    if (!key.empty()) {
        const auto it = cache.find(key);
        if (it != cache.end()) {
            return *it->second;
        }
        auto openings = std::make_shared<std::vector<Group1>>(batch_opening_commitments(*materialize(V), n));
        cache[key] = openings;
        return *openings;
    }

    return batch_opening_commitments(*materialize(V), n);
}

Group2 kzgProtocol::commit_g2(const poly& p) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }
    if (count_only_) {
        return Group2::zero();
    }
    return msm_dispatch<Group2, G2>(p.coefficients(), params.g2_powers_view(), params.g2_raw_powers_view());
}

Group1 kzgProtocol::commit_g1_group(const poly& p) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }
    if (count_only_) {
        return Group1::zero();
    }
    return msm_dispatch<Group1, G1>(p.coefficients(), params.g1_powers_view(), params.g1_raw_powers_view());
}

const Group1& kzgProtocol::g1_power(size_t exponent) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }
    if (count_only_) {
        static const Group1 zero = Group1::zero();
        (void)exponent;
        return zero;
    }
    if (exponent >= params.g1_powers_view().size()) {
        throw std::runtime_error("kzgProtocol::g1_power: exponent out of range");
    }
    return params.g1_powers_view()[exponent];
}

const Group2& kzgProtocol::g2_power(size_t exponent) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }
    if (count_only_) {
        static const Group2 zero = Group2::zero();
        (void)exponent;
        return zero;
    }
    if (exponent >= params.g2_powers_view().size()) {
        if (kzg_debug_fail_enabled()) {
            std::cerr << "kzg g2_power out of range exponent=" << exponent
                      << " available=" << params.g2_powers_view().size()
                      << " deg2=" << params.deg2 << '\n';
        }
        throw std::runtime_error("kzgProtocol::g2_power: exponent out of range");
    }
    return params.g2_powers_view()[exponent];
}

bool kzgProtocol::has_proven_degree(const kzgCommitment& com, size_t degree) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }

    const auto it = params.best_degree_by_index.find(com.index);
    return it != params.best_degree_by_index.end() && it->second <= degree;
}

bool kzgProtocol::has_degree_bound(const virCommitment& V, size_t degree) const
{
    if (!is_setup) {
        throw std::runtime_error("kzgProtocol not setup");
    }

    const auto kzg_degree_bound = [this](const kzgCommitment& com) -> std::optional<size_t> {
        const auto it = params.best_degree_by_index.find(com.index);
        if (it == params.best_degree_by_index.end()) {
            return std::nullopt;
        }
        return it->second;
    };

    const std::optional<size_t> bound = known_degree_bound(V, kzg_degree_bound);
    return bound && *bound <= degree;
}

kzgParams::kzgParams(size_t n1, size_t n2)
{
    deg1 = n1 - 1;
    deg2 = n2 - 1;
    const CachedSrsPowers& cached = get_cached_srs_powers(n1, n2);
    g1_powers = cached.g1_powers;
    g2_powers = cached.g2_powers;
    g1_raw_powers = cached.g1_raw_powers;
    g2_raw_powers = cached.g2_raw_powers;
}

Group1 kzgProver::finalize_degree(Fr c)
{
    KzgProfileStats stats;
    std::map<size_t, std::vector<Fr>> coefficients_by_shift;
    kzg_profile_scope_void(stats, "aggregate_shifted_coefficients", [&] {
        std::vector<Fr> factors(params->degree.size(), Fr(1));
        Fr factor = Fr(1);
        for (size_t i = 0; i < params->degree.size(); ++i) {
            factors[i] = factor;
            factor *= c;
        }

        for (size_t claim = 0; claim < params->degree.size(); ++claim) {
            const auto& entry = params->degree[claim];
            const size_t index = entry[0];
            const size_t claim_degree = entry[1];
            const size_t shift = params->deg1 - claim_degree;
            const auto& coe = polys[index]->coefficients();
            std::vector<Fr>& shifted = coefficients_by_shift[shift];
            if (shifted.size() < coe.size()) {
                shifted.resize(coe.size(), Fr(0));
            }
            for (size_t i = 0; i < coe.size(); ++i) {
                shifted[i] += factors[claim] * coe[i];
            }
        }
    });
    Group1 out = kzg_profile_scope(stats, "aggregate_shifted_coefficients_msm", [&] {
        size_t grouped_scalar_count = 0;
        for (const auto& [shift, coefficients] : coefficients_by_shift) {
            (void)shift;
            grouped_scalar_count += coefficients.size();
        }
        if (grouped_scalar_count > params->g1_powers_view().size() / 2) {
            std::vector<Fr> dense(params->g1_powers_view().size(), Fr(0));
            for (const auto& [shift, coefficients] : coefficients_by_shift) {
                if (shift > dense.size() || coefficients.size() > dense.size() - shift) {
                    throw std::runtime_error("kzgProver::finalize_degree shifted coefficients exceed SRS");
                }
                for (size_t i = 0; i < coefficients.size(); ++i) {
                    dense[shift + i] += coefficients[i];
                }
            }
            return msm_dispatch<Group1, G1>(dense, params->g1_powers_view(), params->g1_raw_powers_view());
        }

        Group1 aggregate = Group1::zero();
        for (const auto& [shift, coefficients] : coefficients_by_shift) {
            aggregate += msm_dispatch<Group1, G1>(coefficients, params->g1_powers_view(), params->g1_raw_powers_view(), shift);
        }
        return aggregate;
    });
    stats.increment("degree_claim_count", params->degree.size());
    stats.print("prover.degree");
    return out;
}

void add_scaled_poly_coefficients(std::vector<Fr>& out, const poly& p, const Fr& scale)
{
    if (scale.isZero() || p.is_zero()) {
        return;
    }

    if (p.is_sparse()) {
        for (const auto& [exp, coeff] : p.terms()) {
            if (coeff.isZero()) {
                continue;
            }
            if (out.size() <= exp) {
                out.resize(exp + 1, Fr(0));
            }
            Fr term;
            Fr::mul(term, coeff, scale);
            Fr::add(out[exp], out[exp], term);
        }
        return;
    }

    const std::vector<Fr>& coeffs = p.coefficients();
    if (out.size() < coeffs.size()) {
        out.resize(coeffs.size(), Fr(0));
    }
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (coeffs[i].isZero()) {
            continue;
        }
        Fr term;
        Fr::mul(term, coeffs[i], scale);
        Fr::add(out[i], out[i], term);
    }
}

std::vector<Group1> kzgProver::finalize()
{
    KzgProfileStats stats;
    assert(polys.size() == params->x.size());
    assert(polys.size() == params->y.size());

    const std::vector<CommitmentOpeningSet> opening_sets = kzg_profile_scope(
        stats, "collect_commitment_opening_sets", [&] {
            return collect_commitment_opening_sets(*params, polys.size());
        });
    if (opening_sets.empty()) {
        stats.print("prover.finalize");
        return {};
    }
    if (params->opening_batch_challenge.isZero()) {
        throw std::runtime_error("kzgProver::finalize missing batched opening challenge");
    }
    const std::vector<Fr> union_points = kzg_profile_scope(stats, "batch.collect_union_points", [&] {
        return collect_union_opening_points(opening_sets);
    });
    if (params->g2_powers_view().size() < union_points.size() + 1) {
        throw std::runtime_error("kzgProver::finalize batched opening requires deg2+1 >= " +
                                 std::to_string(union_points.size() + 1) + ", but SRS only provides deg2+1 = " +
                                 std::to_string(params->g2_powers_view().size()));
    }

    poly combined_q = kzg_profile_scope(stats, "batch.combine_quotients", [&] {
        std::vector<Fr> combined_coefficients;
        Fr factor = Fr(1);
        kzg_profile_scope_void(stats, "batch.build_quotients", [&] {
            for (size_t i = 0; i < opening_sets.size(); ++i) {
                const CommitmentOpeningSet& set = opening_sets[i];
                const size_t index = set.commitment_index;
                poly quotient;
                if (set.points.size() == 1) {
                    stats.increment("batch.single_point_commitments");
                    quotient = quotient_single_opening(*polys[index], set.points[0]);
                } else {
                    stats.increment("batch.multi_point_commitments");
                    const poly h = poly_from_interpolation(set.points, set.values);
                    const poly z = poly_as_vanishing(set.points);
                    quotient = (*polys[index] - h) / z;
                }
                add_scaled_poly_coefficients(combined_coefficients, quotient, factor);
                Fr::mul(factor, factor, params->opening_batch_challenge);
            }
        });
        return poly_from_coeff(combined_coefficients);
    });
    std::vector<Group1> proof;
    proof.push_back(kzg_profile_scope(stats, "batch.msm", [&] {
        return msm_dispatch<Group1, G1>(combined_q.coefficients(), params->g1_powers_view(), params->g1_raw_powers_view());
    }));
    stats.increment("batch.commitment_count", opening_sets.size());
    stats.increment("batch.union_point_count", union_points.size());
    stats.print("prover.finalize");
    return proof;
}

Fr kzgVerifier::finalize_degree_1()
{
    if (params == nullptr || params->transcript == nullptr) {
        throw std::runtime_error("kzgVerifier transcript not setup");
    }
    degree_challenge = params->transcript->squeeze_fr("kzg.degree.challenge");
    const Fr c = degree_challenge;

    std::map<size_t, Group1> aggregate_by_shift;
    Fr factor = Fr(1);
    for (const auto& [index, degree] : params->degree) {
        const size_t shift = params->deg1 - degree;
        aggregate_by_shift[shift] += factor * coms[index].com;
        factor *= c;
    }
    degree_pairing_g1.clear();
    degree_pairing_g2.clear();
    degree_pairing_g1.reserve(aggregate_by_shift.size());
    degree_pairing_g2.reserve(aggregate_by_shift.size());
    for (const auto& [shift, aggregate] : aggregate_by_shift) {
        degree_pairing_g1.push_back(aggregate);
        degree_pairing_g2.push_back(params->g2_powers_view()[shift]);
    }
    // combined_com.value().setOne();
    // for (const auto& [index, claim_degree] : params->degree) {
    //     size_t k = params->deg1 - claim_degree;
    //     GroupT e = pairing(coms[index].com, params->g2_powers[k]);
    //     Fp12 t;
    //     Fp12::pow(t, e.value(), factor);
    //     Fp12::mul(combined_com.value(), combined_com.value(), t);
    //     Fr::mul(factor, factor, c);
    // }
    return c;
}

bool kzgVerifier::finalize_degree_2(const Group1 &f) const
{
    std::vector<Group1> g1;
    std::vector<Group2> g2;
    g1.reserve(degree_pairing_g1.size() + 1);
    g2.reserve(degree_pairing_g2.size() + 1);
    g1.push_back(f);
    g2.push_back(Group2::one());
    for (size_t i = 0; i < degree_pairing_g1.size(); ++i) {
        g1.push_back(-degree_pairing_g1[i]);
        g2.push_back(degree_pairing_g2[i]);
    }
    proof_cost::add("pairing_count", g1.size());
    return pairing_product(g1, g2) == GroupT::zero();
}

bool kzgVerifier::prepare_finalization(const std::vector<Group1> &q)
{
    opening_pairing_g1.clear();
    opening_pairing_g2.clear();
    opening_finalization_prepared = false;
    KzgProfileStats stats;
    size_t n = coms.size();

    assert(n == params->x.size());
    assert(n == params->y.size());
    const std::vector<CommitmentOpeningSet> opening_sets = kzg_profile_scope(
        stats, "collect_commitment_opening_sets", [&] {
            return collect_commitment_opening_sets(*params, n);
        });
    if (opening_sets.empty()) {
        opening_finalization_prepared = true;
        stats.print("verifier.finalize");
        return q.empty();
    }
    if (q.size() != 1) {
        throw std::runtime_error("kzgVerifier::finalize expected one batched opening proof");
    }
    if (params->opening_batch_challenge.isZero()) {
        throw std::runtime_error("kzgVerifier::finalize missing batched opening challenge");
    }

    const std::vector<Fr> union_points = kzg_profile_scope(stats, "batch.collect_union_points", [&] {
        return collect_union_opening_points(opening_sets);
    });
    if (union_points.empty()) {
        throw std::runtime_error("kzgVerifier::finalize collected an empty opening union");
    }
    if (params->g2_powers_view().size() < union_points.size() + 1) {
        throw std::runtime_error("kzgVerifier::finalize batched opening requires deg2+1 >= " +
                                 std::to_string(union_points.size() + 1) + ", but SRS only provides deg2+1 = " +
                                 std::to_string(params->g2_powers_view().size()));
    }

    const poly common_vanishing = kzg_profile_scope(stats, "batch.build_common_vanishing", [&] {
        return poly_as_vanishing(union_points);
    });
    const Group2 common_vanishing_commitment = kzg_profile_scope(stats, "batch.commit_common_vanishing_g2", [&] {
        return msm_dispatch<Group2, G2>(
            common_vanishing.coefficients(), params->g2_powers_view(), params->g2_raw_powers_view());
    });

    std::vector<Fr> factors(opening_sets.size(), Fr(1));
    Fr factor = Fr(1);
    for (size_t i = 0; i < opening_sets.size(); ++i) {
        factors[i] = factor;
        Fr::mul(factor, factor, params->opening_batch_challenge);
    }
    for (const CommitmentOpeningSet& set : opening_sets) {
        if (set.points.size() == 1) {
            stats.increment("batch.single_point_commitments");
        } else {
            stats.increment("batch.multi_point_commitments");
        }
    }

    std::map<std::string, Group1> grouped_opening_terms;
    kzg_profile_scope_void(stats, "batch.opening_set_terms", [&] {
        for (size_t i = 0; i < opening_sets.size(); ++i) {
            const CommitmentOpeningSet& set = opening_sets[i];
            Group1 interpolation_commitment;
            if (set.points.size() == 1) {
                interpolation_commitment = set.values[0] * Group1::one();
            } else {
                const poly interpolation = poly_from_interpolation(set.points, set.values);
                interpolation_commitment =
                    msm_dispatch<Group1, G1>(interpolation.coefficients(), params->g1_powers_view(), params->g1_raw_powers_view());
            }

            std::string key(union_points.size(), '0');
            for (const Fr& point : set.points) {
                bool found = false;
                for (size_t j = 0; j < union_points.size(); ++j) {
                    if (union_points[j] == point) {
                        key[j] = '1';
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw std::runtime_error("kzgVerifier::finalize opening point missing from union");
                }
            }
            grouped_opening_terms[key] += factors[i] * (coms[set.commitment_index].com - interpolation_commitment);
        }
    });

    std::vector<Group1> pairing_g1;
    std::vector<Group2> pairing_g2;
    pairing_g1.reserve(grouped_opening_terms.size() + 1);
    pairing_g2.reserve(grouped_opening_terms.size() + 1);
    kzg_profile_scope_void(stats, "batch.auxiliary_pairings", [&] {
        for (const auto& [key, aggregate_term] : grouped_opening_terms) {
            std::vector<Fr> points;
            points.reserve(union_points.size());
            for (size_t i = 0; i < key.size(); ++i) {
                if (key[i] == '1') {
                    points.push_back(union_points[i]);
                }
            }
            const poly set_vanishing = poly_as_vanishing(points);
            const auto [auxiliary, remainder] = div(common_vanishing, set_vanishing);
            if (!remainder.is_zero()) {
                throw std::runtime_error("kzgVerifier::finalize common vanishing is not divisible by an opening set");
            }
            const Group2 auxiliary_commitment =
                msm_dispatch<Group2, G2>(auxiliary.coefficients(), params->g2_powers_view(), params->g2_raw_powers_view());
            pairing_g1.push_back(-aggregate_term);
            pairing_g2.push_back(auxiliary_commitment);
            stats.increment("batch.auxiliary_pairing_count");
        }
    });
    if (grouped_opening_terms.empty()) {
        throw std::runtime_error("kzgVerifier::finalize produced no grouped opening terms");
    }

    pairing_g1.push_back(q[0]);
    pairing_g2.push_back(common_vanishing_commitment);
    opening_pairing_g1 = std::move(pairing_g1);
    opening_pairing_g2 = std::move(pairing_g2);
    opening_finalization_prepared = true;
    stats.increment("batch.commitment_count", opening_sets.size());
    stats.increment("batch.union_point_count", union_points.size());
    stats.print("verifier.finalize");
    return true;
}

bool kzgVerifier::verify_prepared_finalization() const
{
    if (!opening_finalization_prepared) {
        throw std::runtime_error("kzgVerifier finalization was not prepared");
    }
    if (opening_pairing_g1.empty()) {
        return true;
    }
    proof_cost::add("pairing_count", opening_pairing_g1.size());
    return pairing_product(opening_pairing_g1, opening_pairing_g2) == GroupT::zero();
}

bool kzgVerifier::finalize(const std::vector<Group1> &q)
{
    return prepare_finalization(q) && verify_prepared_finalization();
}


bool kzgProtocol::finalize()
{
    if (count_only_) {
        begin_finalize("kzgProtocol");
        return true;
    }
    const bool verify_mode = proof_session_ && proof_session_->mode() == ProofMode::Verify;
    if (verify_mode && verifier_finalization_prepared_) {
        return verify_prepared_finalization();
    }

    KzgProfileStats stats;
    begin_finalize("kzgProtocol");
    if (!verify_mode && prover.polys.empty()) {
        return true;
    }
    if (verify_mode && verifier.coms.empty() && params.degree.empty()) {
        return true;
    }

    if (!kzg_profile_scope(stats, "finalize_degree", [&] {
            return finalize_degree();
        })) {
        stats.print("protocol.finalize");
        return false;
    }

    std::vector<Group1> q;
    const std::vector<CommitmentOpeningSet> opening_sets = kzg_profile_scope(
        stats, "collect_commitment_opening_sets", [&] {
            return collect_commitment_opening_sets(params, verifier.coms.size());
        });
    if (proof_cost::enabled()) {
    std::size_t polynomial_point_claim_count = 0;
    std::size_t max_points_per_opened_commitment = 0;
    std::size_t single_point_opened_commitment_count = 0;
    std::size_t multi_point_opened_commitment_count = 0;
    for (const CommitmentOpeningSet& set : opening_sets) {
        polynomial_point_claim_count += set.points.size();
        max_points_per_opened_commitment =
            std::max(max_points_per_opened_commitment, set.points.size());
        if (set.points.size() == 1) {
            ++single_point_opened_commitment_count;
        } else if (!set.points.empty()) {
            ++multi_point_opened_commitment_count;
        }
    }
    const std::vector<Fr> cost_union_points = collect_union_opening_points(opening_sets);

    struct OpeningPointBucket {
        std::size_t claim_count = 0;
        std::size_t interleaved_claim_count = 0;
        std::size_t interleave_shifted_claim_count = 0;
        std::size_t x_power_claim_count = 0;
        std::size_t x_scale_claim_count = 0;
        std::size_t unattributed_claim_count = 0;
        std::size_t root_equal_claim_count = 0;
        std::array<std::size_t, 6> role_claim_counts{};
    };
    const std::array<std::pair<kzgOpeningRootRole, const char*>, 6> role_descriptors = {{
        {kzgOpeningRootRole::SumDirectZero, "sum_direct_zero"},
        {kzgOpeningRootRole::SumBatched, "sum_batched"},
        {kzgOpeningRootRole::ZeroCheck, "zero_check"},
        {kzgOpeningRootRole::CqLookup, "cq_lookup"},
        {kzgOpeningRootRole::FixedEvaluation, "fixed_evaluation"},
        {kzgOpeningRootRole::ExplicitCheck, "explicit_check"},
    }};
    std::map<Fr, std::size_t> point_bucket_indices;
    for (std::size_t i = 0; i < cost_union_points.size(); ++i) {
        point_bucket_indices.emplace(cost_union_points[i], i);
    }
    std::vector<OpeningPointBucket> point_buckets(cost_union_points.size());
    std::array<std::size_t, 6> role_claim_counts{};
    std::array<std::set<Fr>, 6> role_actual_points;
    std::set<Fr> opening_root_points;
    std::set<Fr> interleave_shifted_actual_points;
    std::size_t provenance_covered_claim_count = 0;
    std::size_t provenance_missing_claim_count = 0;
    std::size_t provenance_trace_count = 0;
    std::size_t interleaved_claim_count = 0;
    std::size_t interleave_shifted_claim_count = 0;
    std::size_t x_power_claim_count = 0;
    std::size_t x_scale_claim_count = 0;
    std::size_t unattributed_claim_count = 0;
    std::size_t max_interleave_layer_count = 0;
    std::size_t max_interleave_shift_count = 0;

    for (const CommitmentOpeningSet& set : opening_sets) {
        for (const Fr& point : set.points) {
            const auto bucket_it = point_bucket_indices.find(point);
            if (bucket_it == point_bucket_indices.end()) {
                throw std::runtime_error("KZG cost accounting lost an opening point bucket");
            }
            OpeningPointBucket& bucket = point_buckets[bucket_it->second];
            ++bucket.claim_count;
            const auto provenance_it = params.opening_provenance.find({set.commitment_index, point});
            if (provenance_it == params.opening_provenance.end() || provenance_it->second.empty()) {
                ++provenance_missing_claim_count;
                ++unattributed_claim_count;
                ++bucket.unattributed_claim_count;
                continue;
            }

            ++provenance_covered_claim_count;
            provenance_trace_count += provenance_it->second.size();
            bool interleaved = false;
            bool interleave_shifted = false;
            bool x_power = false;
            bool x_scale = false;
            bool root_equal = false;
            std::array<bool, 6> roles_seen{};
            for (const kzgOpeningProvenance& trace : provenance_it->second) {
                opening_root_points.insert(trace.root_point);
                root_equal = root_equal || trace.root_point == point;
                interleaved = interleaved || trace.interleave_layer_count != 0;
                interleave_shifted = interleave_shifted || trace.interleave_shift_count != 0;
                x_power = x_power || trace.x_power_count != 0;
                x_scale = x_scale || trace.x_scale_count != 0;
                max_interleave_layer_count =
                    std::max(max_interleave_layer_count, trace.interleave_layer_count);
                max_interleave_shift_count =
                    std::max(max_interleave_shift_count, trace.interleave_shift_count);
                for (std::size_t role_index = 0; role_index < role_descriptors.size(); ++role_index) {
                    roles_seen[role_index] =
                        roles_seen[role_index] || trace.root_role == role_descriptors[role_index].first;
                }
            }
            if (interleaved) {
                ++interleaved_claim_count;
                ++bucket.interleaved_claim_count;
            }
            if (interleave_shifted) {
                ++interleave_shifted_claim_count;
                ++bucket.interleave_shifted_claim_count;
                interleave_shifted_actual_points.insert(point);
            }
            if (x_power) {
                ++x_power_claim_count;
                ++bucket.x_power_claim_count;
            }
            if (x_scale) {
                ++x_scale_claim_count;
                ++bucket.x_scale_claim_count;
            }
            if (root_equal) {
                ++bucket.root_equal_claim_count;
            }
            bool attributed = false;
            for (std::size_t role_index = 0; role_index < role_descriptors.size(); ++role_index) {
                if (!roles_seen[role_index]) {
                    continue;
                }
                attributed = true;
                ++role_claim_counts[role_index];
                ++bucket.role_claim_counts[role_index];
                role_actual_points[role_index].insert(point);
            }
            if (!attributed) {
                ++unattributed_claim_count;
                ++bucket.unattributed_claim_count;
            }
        }
    }

    std::size_t bucket_claim_sum = 0;
    std::size_t bucket_interleaved_claim_sum = 0;
    std::size_t bucket_interleave_shifted_claim_sum = 0;
    std::size_t bucket_x_power_claim_sum = 0;
    std::size_t bucket_x_scale_claim_sum = 0;
    std::size_t bucket_unattributed_claim_sum = 0;
    std::array<std::size_t, 6> bucket_role_claim_sums{};
    for (std::size_t i = 0; i < point_buckets.size(); ++i) {
        const OpeningPointBucket& bucket = point_buckets[i];
        bucket_claim_sum += bucket.claim_count;
        bucket_interleaved_claim_sum += bucket.interleaved_claim_count;
        bucket_interleave_shifted_claim_sum += bucket.interleave_shifted_claim_count;
        bucket_x_power_claim_sum += bucket.x_power_claim_count;
        bucket_x_scale_claim_sum += bucket.x_scale_claim_count;
        bucket_unattributed_claim_sum += bucket.unattributed_claim_count;
        if (bucket.interleaved_claim_count > bucket.claim_count ||
            bucket.interleave_shifted_claim_count > bucket.interleaved_claim_count ||
            bucket.x_power_claim_count > bucket.claim_count ||
            bucket.x_scale_claim_count > bucket.claim_count ||
            bucket.unattributed_claim_count > bucket.claim_count ||
            bucket.root_equal_claim_count > bucket.claim_count) {
            throw std::runtime_error("KZG opening point bucket counters exceed the bucket claim count");
        }
        const std::string prefix = "opening_point_" + std::to_string(i) + "_";
        for (std::size_t role_index = 0; role_index < role_descriptors.size(); ++role_index) {
            if (bucket.role_claim_counts[role_index] > bucket.claim_count) {
                throw std::runtime_error("KZG opening role count exceeds the bucket claim count");
            }
            bucket_role_claim_sums[role_index] += bucket.role_claim_counts[role_index];
        }
    }
    if (bucket_claim_sum != polynomial_point_claim_count ||
        bucket_interleaved_claim_sum != interleaved_claim_count ||
        bucket_interleave_shifted_claim_sum != interleave_shifted_claim_count ||
        bucket_x_power_claim_sum != x_power_claim_count ||
        bucket_x_scale_claim_sum != x_scale_claim_count ||
        bucket_unattributed_claim_sum != unattributed_claim_count ||
        bucket_role_claim_sums != role_claim_counts ||
        provenance_covered_claim_count + provenance_missing_claim_count != polynomial_point_claim_count ||
        point_buckets.size() != cost_union_points.size()) {
        throw std::runtime_error("KZG opening provenance counters are internally inconsistent");
    }
    if (provenance_missing_claim_count != 0) {
        throw std::runtime_error("KZG opening provenance is missing for one or more polynomial-point claims");
    }

    proof_cost::add("polynomial_point_claim_count", polynomial_point_claim_count);
    for (std::size_t role_index = 0; role_index < role_descriptors.size(); ++role_index) {
        const std::string prefix = std::string("opening_role_") + role_descriptors[role_index].second;
    }
    }
    params.opening_batch_challenge = Fr(0);
    if (!opening_sets.empty()) {
        params.opening_batch_challenge = kzg_profile_scope(stats, "sample_opening_batch_challenge", [&] {
            return params.transcript->squeeze_nonzero_fr("kzg.opening.batch.challenge");
        });
    }
    const size_t proof_count = opening_sets.empty() ? 0 : 1;
    if (proof_session_ && proof_session_->mode() == ProofMode::Prove) {
        q = kzg_profile_scope(stats, "exchange_opening_proofs", [&] {
            return proof_session_->exchange_group1_vector("kzg.opening.proof", proof_count, [this] {
                return prover.finalize();
            });
        });
    } else if (proof_session_ && proof_session_->mode() == ProofMode::Verify) {
        q = kzg_profile_scope(stats, "exchange_opening_proofs", [&] {
            return proof_session_->exchange_group1_vector("kzg.opening.proof", proof_count, {});
        });
    } else {
        q = kzg_profile_scope(stats, "prover_finalize", [&] {
            return prover.finalize();
        });
    }
    const bool prepared = kzg_profile_scope(stats, "verifier_finalize.prepare", [&] {
        return verifier.prepare_finalization(q);
    });
    const bool ok = defer_verifier_pairing_checks_
                        ? prepared
                        : (prepared && kzg_profile_scope(stats, "verifier_finalize.pairing", [&] {
                               return verifier.verify_prepared_finalization();
                           }));
    stats.increment("batched_opening_commitment_count", opening_sets.size());
    stats.increment("opening_proof_count", proof_count);
    stats.print("protocol.finalize");
    return ok;
}

bool kzgProtocol::prepare_verifier_finalization()
{
    if (!(proof_session_ && proof_session_->mode() == ProofMode::Verify)) {
        throw std::runtime_error("kzgProtocol::prepare_verifier_finalization is verifier-only");
    }
    if (verifier_finalization_prepared_) {
        return verifier_finalization_prepare_ok_;
    }
    defer_verifier_pairing_checks_ = true;
    try {
        verifier_finalization_prepare_ok_ = finalize();
    } catch (...) {
        defer_verifier_pairing_checks_ = false;
        throw;
    }
    defer_verifier_pairing_checks_ = false;
    verifier_finalization_prepared_ = true;
    return verifier_finalization_prepare_ok_;
}

bool kzgProtocol::verify_prepared_finalization()
{
    if (!verifier_finalization_prepared_) {
        throw std::runtime_error("kzgProtocol verifier finalization was not prepared");
    }
    if (!verifier_finalization_prepare_ok_) {
        return false;
    }
    if (!verifier.opening_finalization_prepared) {
        return true;
    }
    return verifier.verify_prepared_finalization();
}

bool kzgProtocol::finalize_degree()
{
    if (params.degree.empty()) {
        return true;
    }
    std::set<size_t> degree_commitments;
    for (const auto& [index, degree] : params.degree) {
        (void)degree;
        degree_commitments.insert(index);
    }
    size_t required_deg2_plus_1 = 1;
    for (const auto& [index, degree] : params.degree) {
        (void)index;
        if (degree > params.deg1) {
            continue;
        }
        required_deg2_plus_1 = std::max(required_deg2_plus_1, params.deg1 - degree + 1);
    }
    if (params.g2_powers_view().size() < required_deg2_plus_1) {
        if (kzg_debug_fail_enabled()) {
            std::cerr << "kzg finalize_degree requires deg2+1 >= " << required_deg2_plus_1
                      << " but available=" << params.g2_powers_view().size()
                      << " deg1=" << (params.deg1 + 1) << '\n';
            for (const auto& [index, degree] : params.degree) {
                if (degree > params.deg1) {
                    continue;
                }
                const size_t shift = params.deg1 - degree;
                std::cerr << "  degree claim index=" << index
                          << " degree=" << degree
                          << " requires_g2_exponent=" << shift
                          << " requires_deg2_plus_1>=" << (shift + 1) << '\n';
            }
        }
        throw std::runtime_error("kzg finalize_degree requires deg2+1 >= " +
                                 std::to_string(required_deg2_plus_1) +
                                 ", but SRS only provides deg2+1 = " +
                                 std::to_string(params.g2_powers_view().size()));
    }
    if (!(proof_session_ && proof_session_->mode() == ProofMode::Verify)) {
        // Validate claimed degree constraints before constructing aggregated proofs.
        // Without this, a wrong degree claim can cause out-of-bounds writes in
        // prover.finalize_degree() due to shifted coefficient indexing.
        for (const auto& [index, degree] : params.degree) {
            if (index >= prover.polys.size()) {
                if (kzg_debug_fail_enabled()) {
                    std::cerr << "kzg finalize_degree invalid commitment index " << index
                              << " polys_size " << prover.polys.size() << '\n';
                }
                return false;
            }
            if (degree > params.deg1) {
                if (kzg_debug_fail_enabled()) {
                    std::cerr << "kzg finalize_degree claimed degree exceeds SRS at commitment index " << index
                              << " degree " << degree << " deg1 " << params.deg1 << '\n';
                }
                return false;
            }
            const size_t coeff_count = prover.polys[index]->coefficients().size();
            if (coeff_count > degree + 1) {
                if (kzg_debug_fail_enabled()) {
                    std::cerr << "kzg finalize_degree coefficient bound mismatch at commitment index " << index
                              << " coeff_count " << coeff_count << " claimed_degree " << degree << '\n';
                }
                return false;
            }
        }
    }
    Fr c = verifier.finalize_degree_1();
    Group1 f;
    if (proof_session_ && proof_session_->mode() == ProofMode::Prove) {
        f = proof_session_->exchange_group1("kzg.degree.proof", [this, c] { return prover.finalize_degree(c); });
    } else if (proof_session_ && proof_session_->mode() == ProofMode::Verify) {
        f = proof_session_->exchange_group1("kzg.degree.proof", {});
    } else {
        f = prover.finalize_degree(c);
    }
    return verifier.finalize_degree_2(f);
}

bool kzgProtocol::can_materialize(const kzgCommitment& com) const
{
    return opaque_commitment_indices_.find(com.index) == opaque_commitment_indices_.end();
}

bool same_virtual_commitment_alias(const virCommitment& left, const virCommitment& right)
{
    if (left.type != right.type) {
        return false;
    }

    switch (left.type) {
    case virCommitment::Type::KZG:
        return left.com.index == right.com.index && left.com.com == right.com.com;
    case virCommitment::Type::SUM:
    case virCommitment::Type::PRODUCT:
    case virCommitment::Type::FRACTION:
    case virCommitment::Type::INTERLEAVE:
        return left.left && right.left && left.right && right.right &&
               same_virtual_commitment_alias(*left.left, *right.left) &&
               same_virtual_commitment_alias(*left.right, *right.right);
    case virCommitment::Type::X_POWER:
        return left.exponent == right.exponent && left.left && right.left &&
               same_virtual_commitment_alias(*left.left, *right.left);
    case virCommitment::Type::X_SCALE:
        return left.scale == right.scale && left.left && right.left &&
               same_virtual_commitment_alias(*left.left, *right.left);
    case virCommitment::Type::LAGRANGE:
        return left.lagrange_subgroup_size == right.lagrange_subgroup_size &&
               left.lagrange_index == right.lagrange_index;
    case virCommitment::Type::SPARSE_POLY:
    case virCommitment::Type::LAGRANGE_LINEAR_COMBINATION:
        return false;
    }
    return false;
}

bool is_zero_virtual_commitment(const virCommitment& commitment)
{
    return commitment.type == virCommitment::Type::SPARSE_POLY && commitment.leaf.is_zero();
}

virCommitment operator+(const virCommitment &left, const virCommitment &right) {
    virCommitment ret;
    ret.type = virCommitment::Type::SUM;
    ret.left = std::make_shared<const virCommitment>(left);
    ret.right = std::make_shared<const virCommitment>(right);
    return ret;
}

virCommitment operator-(const virCommitment &left, const virCommitment &right) {
    if (same_virtual_commitment_alias(left, right)) {
        return virCommitment(Fr(0));
    }

    virCommitment ret;
    ret.type = virCommitment::Type::SUM;
    ret.left = std::make_shared<const virCommitment>(left);
    ret.right = std::make_shared<const virCommitment>(right * -Fr(1));
    return ret;
}

virCommitment operator*(const virCommitment &left, const virCommitment &right) {
    virCommitment ret;
    ret.type = virCommitment::Type::PRODUCT;
    ret.left = std::make_shared<const virCommitment>(left);
    ret.right = std::make_shared<const virCommitment>(right);
    return ret;
}

virCommitment operator/(const virCommitment &left, const virCommitment &right) {
    virCommitment ret;
    ret.type = virCommitment::Type::FRACTION;
    ret.left = std::make_shared<const virCommitment>(left);
    ret.right = std::make_shared<const virCommitment>(right);
    return ret;
}

virCommitment x_pow(const virCommitment& base, size_t exponent) {
    virCommitment ret;
    ret.type = virCommitment::Type::X_POWER;
    ret.left = std::make_shared<const virCommitment>(base);
    ret.exponent = exponent;
    return ret;
}

virCommitment x_scale(const virCommitment& base, const Fr& scale) {
    virCommitment ret;
    ret.type = virCommitment::Type::X_SCALE;
    ret.left = std::make_shared<const virCommitment>(base);
    ret.scale = scale;
    return ret;
}
