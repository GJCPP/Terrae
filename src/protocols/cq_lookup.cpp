#include "cq_lookup.hpp"

#include "algebra_utils.hpp"
#include "polynomial.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <optional>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

bool cq_lookup_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_CQ_LOOKUP");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void cq_lookup_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = cq_lookup_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "cq_lookup_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("cq_lookup_profile", label, "", elapsed.count(), "ok");
}

struct CachedCqDomainData {
    Fr lagrange_at_zero = Fr(0);
    Group2 vanishing_g2;
    std::shared_ptr<const std::vector<Group1>> lagrange_commitments;
    std::shared_ptr<const std::vector<Group1>> a0_quotient_commitments;
};

struct CachedCqTableData {
    Group2 table_g2;
    std::shared_ptr<const std::vector<Fr>> table_evals;
    std::shared_ptr<const std::map<Fr, size_t>> index_by_value;
    std::shared_ptr<const std::vector<Group1>> quotient_commitments;
};

struct CqDomainCacheKey {
    std::string srs_identity;
    size_t domain_size = 0;

    bool operator==(const CqDomainCacheKey& other) const
    {
        return srs_identity == other.srs_identity && domain_size == other.domain_size;
    }
};

struct CqDomainCacheKeyHash {
    size_t operator()(const CqDomainCacheKey& key) const
    {
        size_t seed = std::hash<std::string>{}(key.srs_identity);
        seed ^= std::hash<size_t>{}(key.domain_size) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct CqTableCacheKey {
    std::string srs_identity;
    size_t domain_size = 0;
    std::string commitment_repr;

    bool operator==(const CqTableCacheKey& other) const
    {
        return srs_identity == other.srs_identity && domain_size == other.domain_size &&
               commitment_repr == other.commitment_repr;
    }
};

struct CqTableCacheKeyHash {
    size_t operator()(const CqTableCacheKey& key) const
    {
        size_t seed = std::hash<std::string>{}(key.srs_identity);
        seed ^= std::hash<size_t>{}(key.domain_size) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<std::string>{}(key.commitment_repr) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct CqLookupBetaWitness {
    bool support_valid = true;
    bool denominators_valid = true;
    bool quotient_valid = true;
    Group1 multiplicity_commitment;
    Group1 a_commitment;
    Group1 qa_commitment;
    Group1 a0_quotient_commitment;
    Fr a0 = Fr(0);
    poly b0_poly;
    poly qb_poly;
    Group1 b0_group;
    Group1 qb_group;
};

struct CqLookupPrechallengeWitness {
    std::shared_ptr<const std::vector<Fr>> query_evals;
    std::shared_ptr<const poly> query_poly;
    std::shared_ptr<const std::vector<std::pair<size_t, Fr>>> multiplicities;
    bool support_valid = true;
    Group1 multiplicity_commitment = Group1::zero();
};

Fr fr_from_size_t(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

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
        throw std::runtime_error("cqLookup: group FFT requires non-zero power-of-two size");
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
        Fr::inv(inv_n, fr_from_size_t(n));
        for (Group1& value : values) {
            value = inv_n * value;
        }
    }
    return values;
}

std::vector<Group1> group_ifft(std::vector<Group1> values)
{
    Fr omega = get_root_of_unity(values.size());
    Fr inv_omega;
    Fr::inv(inv_omega, omega);
    return group_fft_with_root(std::move(values), inv_omega, true);
}

Group1 linear_combination(const std::vector<Group1>& bases,
                          const std::vector<std::pair<size_t, Fr>>& terms)
{
    std::vector<G1> selected_bases;
    std::vector<Fr> selected_scalars;
    selected_bases.reserve(terms.size());
    selected_scalars.reserve(terms.size());

    for (const auto& [index, coeff] : terms) {
        if (index >= bases.size()) {
            throw std::runtime_error("cqLookup: basis index out of range");
        }
        if (!coeff.isZero()) {
            selected_bases.push_back(bases[index].v);
            selected_scalars.push_back(coeff);
        }
    }

    if (selected_bases.empty()) {
        return Group1::zero();
    }
    if (selected_bases.size() >= 16) {
        G1 out;
        G1::mulVec(out, selected_bases.data(), selected_scalars.data(), selected_bases.size());
        return Group1(out);
    }

    Group1 sum = Group1::zero();
    for (size_t i = 0; i < selected_bases.size(); ++i) {
        sum += selected_scalars[i] * Group1(selected_bases[i]);
    }
    return sum;
}

Group1 aligned_linear_combination(const std::vector<Group1>& bases,
                                  const std::vector<std::pair<size_t, Fr>>& terms)
{
    if (bases.size() != terms.size()) {
        throw std::runtime_error("cqLookup: sparse basis size mismatch");
    }

    std::vector<G1> selected_bases;
    std::vector<Fr> selected_scalars;
    selected_bases.reserve(terms.size());
    selected_scalars.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) {
        const Fr& coeff = terms[i].second;
        if (!coeff.isZero()) {
            selected_bases.push_back(bases[i].v);
            selected_scalars.push_back(coeff);
        }
    }

    if (selected_bases.empty()) {
        return Group1::zero();
    }
    if (selected_bases.size() >= 16) {
        G1 out;
        G1::mulVec(out, selected_bases.data(), selected_scalars.data(), selected_bases.size());
        return Group1(out);
    }

    Group1 sum = Group1::zero();
    for (size_t i = 0; i < selected_bases.size(); ++i) {
        sum += selected_scalars[i] * Group1(selected_bases[i]);
    }
    return sum;
}

std::vector<std::pair<size_t, Fr>> multiplicity_terms(const std::vector<Fr>& query_evals,
                                                      const cqLookupTable& table,
                                                      bool& support_valid)
{
    std::map<size_t, Fr> frequency_by_index;
    support_valid = true;

    for (const Fr& value : query_evals) {
        const auto it = table.index_by_value->find(value);
        if (it == table.index_by_value->end()) {
            support_valid = false;
            continue;
        }

        auto [frequency_it, _] = frequency_by_index.emplace(it->second, Fr(0));
        Fr::add(frequency_it->second, frequency_it->second, Fr(1));
    }

    std::vector<std::pair<size_t, Fr>> terms;
    terms.reserve(frequency_by_index.size());
    for (const auto& [index, coeff] : frequency_by_index) {
        terms.emplace_back(index, coeff);
    }
    return terms;
}

std::vector<std::pair<size_t, Fr>> scale_terms_by_inverse_denominators(
    const std::vector<std::pair<size_t, Fr>>& terms,
    const std::vector<Fr>& table_evals,
    const Fr& beta,
    bool& denominators_valid)
{
    denominators_valid = true;
    std::vector<std::pair<size_t, Fr>> scaled = terms;
    for (auto& [index, coeff] : scaled) {
        Fr denom = table_evals[index] + beta;
        if (denom.isZero()) {
            denominators_valid = false;
            coeff = Fr(0);
            continue;
        }
        Fr inv;
        Fr::inv(inv, denom);
        Fr::mul(coeff, coeff, inv);
    }
    return scaled;
}

std::vector<std::pair<size_t, Fr>> scale_terms_by_sparse_inverse_denominators(
    const std::vector<std::pair<size_t, Fr>>& terms,
    const std::vector<Fr>& table_evals_at_terms,
    const Fr& beta,
    bool& denominators_valid)
{
    if (terms.size() != table_evals_at_terms.size()) {
        throw std::runtime_error("cqLookup: sparse table eval size mismatch");
    }

    denominators_valid = true;
    std::vector<std::pair<size_t, Fr>> scaled = terms;
    for (size_t i = 0; i < scaled.size(); ++i) {
        Fr denom = table_evals_at_terms[i] + beta;
        if (denom.isZero()) {
            denominators_valid = false;
            scaled[i].second = Fr(0);
            continue;
        }
        Fr inv;
        Fr::inv(inv, denom);
        Fr::mul(scaled[i].second, scaled[i].second, inv);
    }
    return scaled;
}

std::vector<Fr> inverse_evals(const std::vector<Fr>& values, const Fr& beta, bool& denominators_valid)
{
    denominators_valid = true;
    std::vector<Fr> out(values.size(), Fr(0));
    for (size_t i = 0; i < values.size(); ++i) {
        Fr denom = values[i] + beta;
        if (denom.isZero()) {
            denominators_valid = false;
            continue;
        }
        Fr::inv(out[i], denom);
    }
    return out;
}

Fr sum_terms(const std::vector<std::pair<size_t, Fr>>& terms)
{
    Fr sum = Fr(0);
    for (const auto& [_, value] : terms) {
        Fr::add(sum, sum, value);
    }
    return sum;
}

std::pair<poly, bool> divide_by_vanishing_coeff_form(const poly& product_minus_one, size_t n)
{
    const std::vector<Fr>& coeffs = product_minus_one.coefficients();
    if (coeffs.size() <= n) {
        bool remainder_is_zero = true;
        for (const Fr& coeff : coeffs) {
            if (!coeff.isZero()) {
                remainder_is_zero = false;
                break;
            }
        }

        poly quotient;
        quotient.init_zero();
        return {quotient, remainder_is_zero};
    }

    // Divide by X^n - 1 without assuming degree < 2n.  For
    // f(X) = q(X)(X^n - 1) + r(X), coefficients satisfy
    // f_k = q_{k-n} - q_k + r_k.  Recover q from high degree to low.
    std::vector<Fr> quotient_coeffs(coeffs.size() - n, Fr(0));
    for (size_t k = coeffs.size(); k-- > n;) {
        quotient_coeffs[k - n] = coeffs[k];
        if (k < quotient_coeffs.size()) {
            Fr::add(quotient_coeffs[k - n], quotient_coeffs[k - n], quotient_coeffs[k]);
        }
    }

    std::vector<Fr> remainder_coeffs(n, Fr(0));
    for (size_t k = 0; k < n; ++k) {
        if (k < coeffs.size()) {
            remainder_coeffs[k] = coeffs[k];
        }
        if (k < quotient_coeffs.size()) {
            Fr::add(remainder_coeffs[k], remainder_coeffs[k], quotient_coeffs[k]);
        }
    }

    bool remainder_is_zero = true;
    for (const Fr& coeff : remainder_coeffs) {
        if (!coeff.isZero()) {
            remainder_is_zero = false;
            break;
        }
    }

    poly quotient;
    quotient.init_with_coeff(quotient_coeffs);
    return {quotient, remainder_is_zero};
}

std::pair<Group1, Group1> commit_g1_group_pair(kzgProtocol& kzg, const poly& left, const poly& right)
{
    return {kzg.commit_g1_group(left), kzg.commit_g1_group(right)};
}

CqLookupBetaWitness build_cq_lookup_beta_witness(const cqLookupParam& params,
                                                 const virCommitment& query,
                                                 size_t n,
                                                 const Fr& beta,
                                                 const std::vector<Fr>* precomputed_query_evals = nullptr,
                                                 const poly* precomputed_query_poly = nullptr,
                                                 const std::vector<std::pair<size_t, Fr>>* precomputed_multiplicities =
                                                     nullptr,
                                                 const std::vector<Fr>* precomputed_table_evals_at_terms = nullptr,
                                                 const std::vector<Group1>* precomputed_quotient_commitments_at_terms =
                                                     nullptr,
                                                 std::optional<bool> precomputed_support_valid = std::nullopt)
{
    if (params.kzg == nullptr || params.table == nullptr) {
        throw std::runtime_error("cqLookupProtocol not setup");
    }
    if (!isPowerOfTwo(n) || n < 2) {
        throw std::runtime_error("cqLookupProver: query domain size must be a power of two >= 2");
    }
    const bool has_sparse_table_data =
        precomputed_multiplicities != nullptr && precomputed_table_evals_at_terms != nullptr &&
        precomputed_quotient_commitments_at_terms != nullptr;
    if (!has_sparse_table_data && (params.table->table_evals == nullptr || params.table->index_by_value == nullptr ||
                                   params.table->quotient_commitments == nullptr)) {
        throw std::runtime_error("cqLookupProver requires table preprocessing; verifier artifacts are not enough");
    }
    if (params.table->lagrange_commitments == nullptr || params.table->a0_quotient_commitments == nullptr) {
        throw std::runtime_error("cqLookupProver requires domain preprocessing; verifier artifacts are not enough");
    }

    CqLookupBetaWitness witness;
    const cqLookupTable& table = *params.table;
    kzgProtocol& kzg = *params.kzg;
    std::vector<Fr> local_query_evals;
    const std::vector<Fr>* query_evals = precomputed_query_evals;
    std::shared_ptr<poly> local_query_poly;
    const poly* query_poly = precomputed_query_poly;
    cq_lookup_profile_scope("beta_witness.evaluate_query", [&] {
        if (query_evals == nullptr) {
            local_query_evals = kzg.evaluations(query, n, kzgProtocol::EvaluationMode::PreferEval);
            query_evals = &local_query_evals;
        }
    });
    cq_lookup_profile_scope("beta_witness.materialize_query", [&] {
        if (query_poly == nullptr) {
            local_query_poly = kzg.materialize(query);
            query_poly = local_query_poly.get();
        }
    });

    std::vector<std::pair<size_t, Fr>> local_multiplicities;
    const std::vector<std::pair<size_t, Fr>>* multiplicities = precomputed_multiplicities;
    cq_lookup_profile_scope("beta_witness.multiplicity_terms", [&] {
        if (multiplicities != nullptr) {
            witness.support_valid = precomputed_support_valid.value_or(true);
        } else {
            local_multiplicities = multiplicity_terms(*query_evals, table, witness.support_valid);
            multiplicities = &local_multiplicities;
        }
    });
    std::vector<std::pair<size_t, Fr>> a_terms;
    cq_lookup_profile_scope("beta_witness.scale_terms", [&] {
        if (precomputed_table_evals_at_terms != nullptr) {
            a_terms = scale_terms_by_sparse_inverse_denominators(
                *multiplicities, *precomputed_table_evals_at_terms, beta, witness.denominators_valid);
        } else {
            a_terms =
                scale_terms_by_inverse_denominators(*multiplicities, *table.table_evals, beta, witness.denominators_valid);
        }
    });
    bool b_denominators_valid = true;
    std::vector<Fr> b_evals;
    cq_lookup_profile_scope("beta_witness.inverse_evals", [&] {
        b_evals = inverse_evals(*query_evals, beta, b_denominators_valid);
    });
    witness.denominators_valid = witness.denominators_valid && b_denominators_valid;

    cq_lookup_profile_scope("beta_witness.commit_groups", [&] {
        witness.multiplicity_commitment = linear_combination(*table.lagrange_commitments, *multiplicities);
        witness.a_commitment = linear_combination(*table.lagrange_commitments, a_terms);
        if (precomputed_quotient_commitments_at_terms != nullptr) {
            witness.qa_commitment = aligned_linear_combination(*precomputed_quotient_commitments_at_terms, a_terms);
        } else {
            witness.qa_commitment = linear_combination(*table.quotient_commitments, a_terms);
        }
        witness.a0_quotient_commitment = linear_combination(*table.a0_quotient_commitments, a_terms);
    });
    witness.a0 = table.lagrange_at_zero * sum_terms(a_terms);

    poly b_poly;
    cq_lookup_profile_scope("beta_witness.build_b_poly", [&] { b_poly = poly_from_evals(b_evals); });
    const Fr b_at_zero = b_poly.eval(Fr(0));
    witness.b0_poly = (b_poly - b_at_zero) / identity_poly();

    poly qb_poly;
    cq_lookup_profile_scope("beta_witness.build_numerator", [&] {
        const poly product = b_poly * (*query_poly + beta);
        poly product_minus_one = product - Fr(1);
        bool quotient_valid = true;
        std::tie(qb_poly, quotient_valid) = divide_by_vanishing_coeff_form(product_minus_one, n);
        witness.quotient_valid = quotient_valid;
    });
    witness.qb_poly = qb_poly;

    cq_lookup_profile_scope("beta_witness.commit_witness_groups", [&] {
        cq_lookup_profile_scope("beta_witness.commit_witness_groups.b0_qb", [&] {
            std::tie(witness.b0_group, witness.qb_group) =
                commit_g1_group_pair(kzg, witness.b0_poly, witness.qb_poly);
        });
    });
    return witness;
}

CqLookupPrechallengeWitness build_cq_lookup_prechallenge_witness(const cqLookupParam& params,
                                                                 const virCommitment& query,
                                                                 size_t n)
{
    if (params.kzg == nullptr || params.table == nullptr) {
        throw std::runtime_error("cqLookupProtocol not setup");
    }
    if (!isPowerOfTwo(n) || n == 0) {
        throw std::runtime_error("cqLookupProver: query domain size must be a non-zero power of two");
    }
    if (params.table->table_evals == nullptr || params.table->index_by_value == nullptr ||
        params.table->lagrange_commitments == nullptr) {
        throw std::runtime_error("cqLookupProver requires table preprocessing; verifier artifacts are not enough");
    }

    CqLookupPrechallengeWitness witness;
    auto query_evals = std::make_shared<std::vector<Fr>>();
    cq_lookup_profile_scope("multiplicity.evaluate_query", [&] { *query_evals = params.kzg->evaluations(query, n); });
    std::shared_ptr<const poly> query_poly;
    cq_lookup_profile_scope("multiplicity.materialize_query", [&] { query_poly = params.kzg->materialize(query); });

    auto multiplicities = std::make_shared<std::vector<std::pair<size_t, Fr>>>();
    cq_lookup_profile_scope("multiplicity.multiplicity_terms", [&] {
        *multiplicities = multiplicity_terms(*query_evals, *params.table, witness.support_valid);
    });

    cq_lookup_profile_scope("multiplicity.commit_group", [&] {
        witness.multiplicity_commitment = linear_combination(*params.table->lagrange_commitments, *multiplicities);
    });

    witness.query_evals = std::move(query_evals);
    witness.query_poly = std::move(query_poly);
    witness.multiplicities = std::move(multiplicities);
    return witness;
}

CachedCqDomainData build_cached_cq_domain_data(kzgProtocol& kzg, size_t n)
{
    CachedCqDomainData cache;

    std::vector<Group1> monomial_commitments(n, Group1::zero());
    cq_lookup_profile_scope("cache_domain.collect_monomials", [&] {
        for (size_t i = 0; i < n; ++i) {
            monomial_commitments[i] = kzg.g1_power(i);
        }
    });

    std::shared_ptr<std::vector<Group1>> lagrange_commitments;
    cq_lookup_profile_scope("cache_domain.group_ifft", [&] {
        lagrange_commitments = std::make_shared<std::vector<Group1>>(group_ifft(monomial_commitments));
    });

    // Lagrange-basis commitments and related quotient bases depend only on the
    // subgroup size, so they can be reused across all tables of that size.
    cache.lagrange_at_zero = fr_from_size_t(n);
    Fr::inv(cache.lagrange_at_zero, cache.lagrange_at_zero);
    cache.vanishing_g2 = kzg.g2_power(n) - Group2::one();
    cache.lagrange_commitments = lagrange_commitments;

    const Group1 x_n_minus_1 = kzg.g1_power(n - 1);
    const Fr omega = get_root_of_unity(n);
    Fr inv_omega;
    Fr::inv(inv_omega, omega);
    Fr inv_root = Fr(1);
    auto a0_quotients = std::make_shared<std::vector<Group1>>();
    cq_lookup_profile_scope("cache_domain.build_a0_quotients", [&] {
        a0_quotients->reserve(n);
        for (size_t i = 0; i < n; ++i) {
            a0_quotients->push_back(inv_root * (*lagrange_commitments)[i] - cache.lagrange_at_zero * x_n_minus_1);
            Fr::mul(inv_root, inv_root, inv_omega);
        }
    });
    cache.a0_quotient_commitments = a0_quotients;

    return cache;
}

std::string cq_srs_identity(const kzgProtocol& kzg)
{
    const size_t g1_fingerprint_index = kzg.deg1() == 0 ? 0 : 1;
    const size_t g2_fingerprint_index = kzg.deg2() == 0 ? 0 : 1;
    return std::to_string(kzg.deg1()) + "|" + std::to_string(kzg.deg2()) + "|" +
           transcript_representation(kzg.g1_power(0)) + "|" +
           transcript_representation(kzg.g1_power(g1_fingerprint_index)) + "|" +
           transcript_representation(kzg.g2_power(g2_fingerprint_index));
}

CqDomainCacheKey cq_domain_cache_key(const kzgProtocol& kzg, size_t n)
{
    return {cq_srs_identity(kzg), n};
}

std::optional<CqTableCacheKey> cq_table_cache_key(const kzgProtocol& kzg, const virCommitment& table, size_t n)
{
    if (table.type != virCommitment::Type::KZG) {
        return std::nullopt;
    }
    return CqTableCacheKey{cq_srs_identity(kzg), n, transcript_representation(table.com)};
}

const CachedCqDomainData& get_cached_cq_domain_data(kzgProtocol& kzg, size_t n)
{
    static std::unordered_map<CqDomainCacheKey, CachedCqDomainData, CqDomainCacheKeyHash> cache;
    const CqDomainCacheKey key = cq_domain_cache_key(kzg, n);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        cq_lookup_profile_scope("cache_domain.hit", []() {});
        return it->second;
    }
    cq_lookup_profile_scope("cache_domain.miss_build", []() {});
    return cache.emplace(key, build_cached_cq_domain_data(kzg, n)).first->second;
}

CachedCqTableData build_cached_cq_table_data(kzgProtocol& kzg,
                                             const virCommitment& table,
                                             size_t n,
                                             const CachedCqDomainData& cached_domain)
{
    CachedCqTableData cache;

    std::shared_ptr<std::vector<Fr>> table_evals;
    cq_lookup_profile_scope("cache_table.evaluate_table", [&] {
        table_evals = std::make_shared<std::vector<Fr>>(
            kzg.evaluations(table, n, kzgProtocol::EvaluationMode::PreferEval));
    });
    auto index_by_value = std::make_shared<std::map<Fr, size_t>>();
    cq_lookup_profile_scope("cache_table.index_values", [&] {
        for (size_t i = 0; i < table_evals->size(); ++i) {
            if (!index_by_value->emplace((*table_evals)[i], i).second) {
                throw std::runtime_error("cqLookupTable: table values must be distinct");
            }
        }
    });

    poly table_poly;
    cq_lookup_profile_scope("cache_table.materialize_table", [&] { table_poly = *kzg.materialize(table); });
    cq_lookup_profile_scope("cache_table.commit_table_g2", [&] { cache.table_g2 = kzg.commit_g2(table_poly); });
    cache.table_evals = table_evals;
    cache.index_by_value = index_by_value;
    std::vector<Group1> quotient_openings;
    cq_lookup_profile_scope("cache_table.batch_openings", [&] {
        quotient_openings = kzg.batch_opening_commitments(table, n);
    });

    const Fr omega = get_root_of_unity(n);
    Fr root = Fr(1);
    auto quotient_commitments = std::make_shared<std::vector<Group1>>();
    quotient_commitments->reserve(n);
    for (size_t i = 0; i < n; ++i) {
        quotient_commitments->push_back((cached_domain.lagrange_at_zero * root) * quotient_openings[i]);
        Fr::mul(root, root, omega);
    }
    cache.quotient_commitments = quotient_commitments;

    return cache;
}

const CachedCqTableData& get_cached_cq_table_data(kzgProtocol& kzg,
                                                  const virCommitment& table,
                                                  size_t n,
                                                  const CachedCqDomainData& cached_domain)
{
    static std::unordered_map<CqTableCacheKey, CachedCqTableData, CqTableCacheKeyHash> cache;
    const std::optional<CqTableCacheKey> key = cq_table_cache_key(kzg, table, n);
    if (key.has_value()) {
        const auto it = cache.find(*key);
        if (it != cache.end()) {
            cq_lookup_profile_scope("cache_table.hit", []() {});
            return it->second;
        }
        cq_lookup_profile_scope("cache_table.miss_build", []() {});
        return cache.emplace(*key, build_cached_cq_table_data(kzg, table, n, cached_domain)).first->second;
    }

    static std::unique_ptr<CachedCqTableData> fallback;
    cq_lookup_profile_scope("cache_table.fallback_build", []() {});
    fallback = std::make_unique<CachedCqTableData>(build_cached_cq_table_data(kzg, table, n, cached_domain));
    return *fallback;
}

CachedCqTableData build_cq_table_data_from_evaluations(kzgProtocol& kzg,
                                                       const std::vector<Fr>& table_evaluations,
                                                       size_t n,
                                                       const CachedCqDomainData& cached_domain)
{
    CachedCqTableData cache;

    auto table_evals = std::make_shared<std::vector<Fr>>(table_evaluations);
    auto index_by_value = std::make_shared<std::map<Fr, size_t>>();
    for (size_t i = 0; i < table_evals->size(); ++i) {
        if (!index_by_value->emplace((*table_evals)[i], i).second) {
            throw std::runtime_error("cqLookupTable: table values must be distinct");
        }
    }

    const poly table_poly = poly_from_evals(*table_evals);
    cache.table_g2 = kzg.commit_g2(table_poly);
    cache.table_evals = table_evals;
    cache.index_by_value = index_by_value;
    const std::vector<Group1> quotient_openings = kzg.batch_opening_commitments(table_poly, n);

    const Fr omega = get_root_of_unity(n);
    Fr root = Fr(1);
    auto quotient_commitments = std::make_shared<std::vector<Group1>>();
    quotient_commitments->reserve(n);
    for (size_t i = 0; i < n; ++i) {
        quotient_commitments->push_back((cached_domain.lagrange_at_zero * root) * quotient_openings[i]);
        Fr::mul(root, root, omega);
    }
    cache.quotient_commitments = quotient_commitments;

    return cache;
}

} // namespace

cqLookupPublicTable build_cq_lookup_public_table(const kzgPublicParameters& public_params,
                                                 std::string key,
                                                 const poly& table,
                                                 size_t domain_size,
                                                 std::optional<size_t> degree_bound)
{
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("build_cq_lookup_public_table: domain size must be a non-zero power of two");
    }

    cqLookupPublicTable out;
    out.key = std::move(key);
    out.domain_size = domain_size;
    out.degree_bound = degree_bound;
    out.commitment = commit_g1_with_public_parameters(public_params, table);
    out.table_g2 = commit_g2_with_public_parameters(public_params, table);
    out.table_evals = std::make_shared<std::vector<Fr>>(table.evaluations_on_domain(domain_size));
    auto index_by_value = std::make_shared<std::map<Fr, size_t>>();
    for (size_t i = 0; i < out.table_evals->size(); ++i) {
        if (!index_by_value->emplace((*out.table_evals)[i], i).second) {
            throw std::runtime_error("build_cq_lookup_public_table: table values must be distinct");
        }
    }
    out.index_by_value = index_by_value;

    kzgProtocol temp_kzg;
    temp_kzg.setup(public_params);
    cqLookupTable temp_table;
    temp_table.setup_from_evaluations(temp_kzg, virCommitment(table), *out.table_evals);
    out.lagrange_at_zero = temp_table.lagrange_at_zero;
    out.vanishing_g2 = temp_table.vanishing_g2;
    out.lagrange_commitments = temp_table.lagrange_commitments;
    out.quotient_commitments = temp_table.quotient_commitments;
    out.a0_quotient_commitments = temp_table.a0_quotient_commitments;
    return out;
}

void fill_cq_lookup_domain_precompute(const kzgPublicParameters& public_params,
                                      size_t domain_size,
                                      Fr& lagrange_at_zero,
                                      Group2& vanishing_g2,
                                      std::shared_ptr<const std::vector<Group1>>& lagrange_commitments,
                                      std::shared_ptr<const std::vector<Group1>>& a0_quotient_commitments)
{
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("fill_cq_lookup_domain_precompute: domain size must be a non-zero power of two");
    }

    kzgProtocol temp_kzg;
    temp_kzg.setup(public_params);
    const CachedCqDomainData& cached = get_cached_cq_domain_data(temp_kzg, domain_size);
    lagrange_at_zero = cached.lagrange_at_zero;
    vanishing_g2 = cached.vanishing_g2;
    lagrange_commitments = cached.lagrange_commitments;
    a0_quotient_commitments = cached.a0_quotient_commitments;
}

void cqLookupTable::setup(kzgProtocol& kzg, const virCommitment& table, size_t n)
{
    if (!isPowerOfTwo(n) || n == 0) {
        throw std::runtime_error("cqLookupTable: domain size must be a non-zero power of two");
    }

    domain_size = n;
    table_commitment = table;
    table_evals.reset();
    index_by_value.reset();
    lagrange_commitments.reset();
    quotient_commitments.reset();
    a0_quotient_commitments.reset();

    const CachedCqDomainData* cached = nullptr;
    cq_lookup_profile_scope("table_setup.get_domain_cache", [&] { cached = &get_cached_cq_domain_data(kzg, n); });
    // Table-specific preprocessing adds distinct-value indexing and quotient
    // commitments on top of the shared domain cache.
    const CachedCqTableData* cached_table = nullptr;
    cq_lookup_profile_scope("table_setup.get_table_cache", [&] {
        cached_table = &get_cached_cq_table_data(kzg, table, n, *cached);
    });
    table_evals = cached_table->table_evals;
    index_by_value = cached_table->index_by_value;
    table_g2 = cached_table->table_g2;
    vanishing_g2 = cached->vanishing_g2;
    lagrange_commitments = cached->lagrange_commitments;
    quotient_commitments = cached_table->quotient_commitments;
    a0_quotient_commitments = cached->a0_quotient_commitments;
    lagrange_at_zero = cached->lagrange_at_zero;
}

void cqLookupTable::setup(kzgProtocol& kzg, const cqLookupPublicTable& table, const virCommitment& table_commitment)
{
    if (table.table_evals != nullptr && table.quotient_commitments != nullptr) {
        setup_from_evaluations_with_precomputed(
            kzg,
            table_commitment,
            *table.table_evals,
            table.table_g2,
            table.quotient_commitments,
            table.index_by_value,
            table.lagrange_commitments,
            table.a0_quotient_commitments,
            table.lagrange_at_zero,
            table.vanishing_g2);
        return;
    }
    if (table.table_evals != nullptr) {
        setup_from_evaluations(kzg, table_commitment, *table.table_evals);
        return;
    }
    setup_verifier(
        kzg,
        table_commitment,
        table.domain_size,
        table.table_g2,
        table.lagrange_commitments,
        table.a0_quotient_commitments,
        table.lagrange_at_zero,
        table.vanishing_g2);
}

void cqLookupTable::setup_from_evaluations(kzgProtocol& kzg,
                                           const virCommitment& table_commitment,
                                           const std::vector<Fr>& table_evaluations)
{
    const size_t n = table_evaluations.size();
    if (!isPowerOfTwo(n) || n == 0) {
        throw std::runtime_error("cqLookupTable: evaluation count must be a non-zero power of two");
    }

    domain_size = n;
    this->table_commitment = table_commitment;
    table_evals.reset();
    index_by_value.reset();
    lagrange_commitments.reset();
    quotient_commitments.reset();
    a0_quotient_commitments.reset();

    const CachedCqDomainData* cached = nullptr;
    cq_lookup_profile_scope("table_setup_from_evals.get_domain_cache", [&] {
        cached = &get_cached_cq_domain_data(kzg, n);
    });
    CachedCqTableData rebuilt;
    cq_lookup_profile_scope("table_setup_from_evals.rebuild_table", [&] {
        rebuilt = build_cq_table_data_from_evaluations(kzg, table_evaluations, n, *cached);
    });
    table_evals = rebuilt.table_evals;
    index_by_value = rebuilt.index_by_value;
    table_g2 = rebuilt.table_g2;
    vanishing_g2 = cached->vanishing_g2;
    lagrange_commitments = cached->lagrange_commitments;
    quotient_commitments = rebuilt.quotient_commitments;
    a0_quotient_commitments = cached->a0_quotient_commitments;
    lagrange_at_zero = cached->lagrange_at_zero;
}

void cqLookupTable::setup_from_evaluations_with_precomputed(
    kzgProtocol& kzg,
    const virCommitment& table_commitment,
    const std::vector<Fr>& table_evaluations,
    const Group2& precomputed_table_g2,
    std::shared_ptr<const std::vector<Group1>> precomputed_quotient_commitments,
    std::shared_ptr<const std::map<Fr, size_t>> precomputed_index_by_value,
    std::shared_ptr<const std::vector<Group1>> precomputed_lagrange_commitments,
    std::shared_ptr<const std::vector<Group1>> precomputed_a0_quotient_commitments,
    std::optional<Fr> precomputed_lagrange_at_zero,
    std::optional<Group2> precomputed_vanishing_g2)
{
    const size_t n = table_evaluations.size();
    if (!isPowerOfTwo(n) || n == 0) {
        throw std::runtime_error("cqLookupTable: evaluation count must be a non-zero power of two");
    }

    domain_size = n;
    this->table_commitment = table_commitment;
    table_evals.reset();
    index_by_value.reset();
    lagrange_commitments.reset();
    quotient_commitments.reset();
    a0_quotient_commitments.reset();

    const CachedCqDomainData* cached = nullptr;
    if (precomputed_lagrange_commitments != nullptr && precomputed_a0_quotient_commitments != nullptr &&
        precomputed_lagrange_at_zero.has_value() && precomputed_vanishing_g2.has_value()) {
        lagrange_commitments = std::move(precomputed_lagrange_commitments);
        a0_quotient_commitments = std::move(precomputed_a0_quotient_commitments);
        lagrange_at_zero = *precomputed_lagrange_at_zero;
        vanishing_g2 = *precomputed_vanishing_g2;
    } else {
        cq_lookup_profile_scope("table_setup_from_precomputed.get_domain_cache", [&] {
            cached = &get_cached_cq_domain_data(kzg, n);
        });
        vanishing_g2 = cached->vanishing_g2;
        lagrange_commitments = cached->lagrange_commitments;
        a0_quotient_commitments = cached->a0_quotient_commitments;
        lagrange_at_zero = cached->lagrange_at_zero;
    }

    table_evals = std::make_shared<std::vector<Fr>>(table_evaluations);
    if (precomputed_index_by_value != nullptr) {
        index_by_value = std::move(precomputed_index_by_value);
    } else {
        cq_lookup_profile_scope("table_setup_from_precomputed.build_index", [&] {
            auto built_index_by_value = std::make_shared<std::map<Fr, size_t>>();
            for (size_t i = 0; i < table_evals->size(); ++i) {
                if (!built_index_by_value->emplace((*table_evals)[i], i).second) {
                    throw std::runtime_error("cqLookupTable: table values must be distinct");
                }
            }
            index_by_value = built_index_by_value;
        });
    }

    table_g2 = precomputed_table_g2;
    quotient_commitments = std::move(precomputed_quotient_commitments);
}

void cqLookupTable::setup_verifier(kzgProtocol& kzg,
                                   const virCommitment& table_commitment,
                                   size_t n,
                                   const Group2& verifier_table_g2,
                                   std::shared_ptr<const std::vector<Group1>> precomputed_lagrange_commitments,
                                   std::shared_ptr<const std::vector<Group1>> precomputed_a0_quotient_commitments,
                                   std::optional<Fr> precomputed_lagrange_at_zero,
                                   std::optional<Group2> precomputed_vanishing_g2)
{
    if (!isPowerOfTwo(n) || n == 0) {
        throw std::runtime_error("cqLookupTable: domain size must be a non-zero power of two");
    }

    domain_size = n;
    this->table_commitment = table_commitment;
    table_evals.reset();
    index_by_value.reset();
    lagrange_commitments.reset();
    quotient_commitments.reset();
    a0_quotient_commitments.reset();

    table_g2 = verifier_table_g2;
    if (kzg.is_count_only()) {
        lagrange_at_zero = Fr(1);
        vanishing_g2 = Group2::zero();
        return;
    }
    if (precomputed_lagrange_commitments != nullptr && precomputed_a0_quotient_commitments != nullptr &&
        precomputed_lagrange_at_zero.has_value() && precomputed_vanishing_g2.has_value()) {
        vanishing_g2 = *precomputed_vanishing_g2;
        lagrange_commitments = std::move(precomputed_lagrange_commitments);
        a0_quotient_commitments = std::move(precomputed_a0_quotient_commitments);
        lagrange_at_zero = *precomputed_lagrange_at_zero;
    } else {
        const CachedCqDomainData* cached = nullptr;
        cq_lookup_profile_scope("table_setup_verifier.get_domain_cache", [&] {
            cached = &get_cached_cq_domain_data(kzg, n);
        });
        vanishing_g2 = cached->vanishing_g2;
        lagrange_commitments = cached->lagrange_commitments;
        a0_quotient_commitments = cached->a0_quotient_commitments;
        lagrange_at_zero = cached->lagrange_at_zero;
    }
}

Fr cqLookupVerifier::beta_challenge()
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("cqLookupVerifier not setup");
    }
    return params->kzg->transcript().squeeze_fr("cq_lookup.beta");
}

Fr cqLookupVerifier::gamma_challenge()
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("cqLookupVerifier not setup");
    }
    return params->kzg->transcript().squeeze_fr("cq_lookup.gamma");
}

void cqLookupVerifier::add_check(const cqLookupCheckRecord& record)
{
    checks.push_back(record);
}

bool cqLookupVerifier::finalize() const
{
    const cqLookupTable& table = *params->table;
    kzgProtocol& kzg = *params->kzg;

    for (const cqLookupCheckRecord& check : checks) {
        if (!check.support_valid || !check.denominators_valid || !check.quotient_valid) {
            return false;
        }

        // Pairing check for the table-side rational witness:
        //   a = multiplicity / (table + beta)
        // modulo the subgroup vanishing polynomial.
        const GroupT lhs = pairing(check.a_commitment, table.table_g2);
        const GroupT rhs =
            pairing(check.qa_commitment, table.vanishing_g2) +
            pairing(check.multiplicity_commitment - check.beta * check.a_commitment, Group2::one());
        if (lhs != rhs) {
            return false;
        }

        const Group1 a_minus_a0 = check.a_commitment - check.a0 * Group1::one();
        if (pairing(a_minus_a0, Group2::one()) !=
            pairing(check.a0_quotient_commitment, kzg.g2_power(1))) {
            return false;
        }

        const Fr b0_at_gamma = kzg.open(check.b0_commitment, check.gamma, kzgOpeningRootRole::CqLookup);
        const Fr query_at_gamma = kzg.open(check.query_commitment, check.gamma, kzgOpeningRootRole::CqLookup);
        const Fr qb_at_gamma = kzg.open(check.qb_commitment, check.gamma, kzgOpeningRootRole::CqLookup);

        const Fr b_at_zero =
            fr_from_size_t(table.domain_size) * check.a0 / fr_from_size_t(check.query_domain_size);
        const Fr b_at_gamma = b0_at_gamma * check.gamma + b_at_zero;
        const Fr zh_at_gamma = vanishing_poly(check.query_domain_size).eval(check.gamma);
        if (zh_at_gamma.isZero()) {
            return false;
        }

        // Query-side rational identity:
        //   b * (query + beta) - 1 = qb * Z_H.
        const Fr expected_qb =
            (b_at_gamma * (query_at_gamma + check.beta) - Fr(1)) / zh_at_gamma;
        if (qb_at_gamma != expected_qb) {
            return false;
        }
    }

    return true;
}

cqLookupCheckRecord cqLookupProver::add_check(const virCommitment& query,
                                              size_t n,
                                              const Fr& beta)
{
    const CqLookupBetaWitness witness = build_cq_lookup_beta_witness(*params, query, n, beta);
    cqLookupCheckRecord record;
    record.query_commitment = query;
    record.query_domain_size = n;
    record.beta = beta;

    record.support_valid = witness.support_valid;
    record.denominators_valid = witness.denominators_valid;
    record.quotient_valid = witness.quotient_valid;
    record.multiplicity_commitment = witness.multiplicity_commitment;
    record.a_commitment = witness.a_commitment;
    record.qa_commitment = witness.qa_commitment;
    record.a0_quotient_commitment = witness.a0_quotient_commitment;
    record.a0 = witness.a0;
    record.b0_commitment = params->kzg->commit(witness.b0_poly);
    if (record.b0_commitment.com != witness.b0_group) {
        throw std::runtime_error("cqLookupProver: b0 commitment registration mismatch");
    }
    params->kzg->prove_degree(record.b0_commitment, n - 2);
    record.qb_commitment = params->kzg->commit(witness.qb_poly);
    if (record.qb_commitment.com != witness.qb_group) {
        throw std::runtime_error("cqLookupProver: qb commitment registration mismatch");
    }

    checks.push_back(record);
    return record;
}

Group1 cqLookupProver::multiplicity_commitment(const virCommitment& query, size_t n, bool* support_valid)
{
    const CqLookupPrechallengeWitness witness = build_cq_lookup_prechallenge_witness(*params, query, n);
    if (support_valid != nullptr) {
        *support_valid = witness.support_valid;
    }
    return witness.multiplicity_commitment;
}

void cqLookupProtocol::add_check(const virCommitment& query, size_t n)
{
    structured_protocol_profile::timed_void(
        "protocol_call",
        "cq_lookup.add_check",
        "query_domain_size=" + std::to_string(n) + ";with_multiplicity_commitment=0",
        [&] {
            if (!is_setup) {
                throw std::runtime_error("cqLookupProtocol not setup");
            }
            if (params.kzg == nullptr || params.table == nullptr) {
                throw std::runtime_error("cqLookupProtocol dependencies not setup");
            }

            PendingCheck pending;
            pending.query_commitment = query;
            pending.query_domain_size = n;

            const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                // Multiplicity is exchanged before sampling `beta`, so it remains
                // Fiat-Shamir-bound to the lookup challenge.
                pending.multiplicity_commitment = proof_session->exchange_group1("cq_lookup.multiplicity", {});
            } else {
                if (proof_session && proof_session->mode() == ProofMode::Prove) {
                    const CqLookupPrechallengeWitness witness = build_cq_lookup_prechallenge_witness(params, query, n);
                    pending.cached_query_evals = witness.query_evals;
                    pending.cached_query_poly = witness.query_poly;
                    pending.cached_multiplicity_terms = witness.multiplicities;
                    pending.cached_support_valid = witness.support_valid;
                    pending.multiplicity_commitment = proof_session->exchange_group1("cq_lookup.multiplicity", [&] {
                        return witness.multiplicity_commitment;
                    });
                } else {
                    pending.multiplicity_commitment = prover.multiplicity_commitment(query, n);
                }
            }
            pending_checks_.push_back(std::move(pending));
        });
}

void cqLookupProtocol::add_check(const virCommitment& query, size_t n, const Group1& multiplicity_commitment)
{
    structured_protocol_profile::timed_void(
        "protocol_call",
        "cq_lookup.add_check",
        "query_domain_size=" + std::to_string(n) + ";with_multiplicity_commitment=1",
        [&] {
            if (!is_setup) {
                throw std::runtime_error("cqLookupProtocol not setup");
            }
            if (params.kzg == nullptr || params.table == nullptr) {
                throw std::runtime_error("cqLookupProtocol dependencies not setup");
            }

            PendingCheck pending;
            pending.query_commitment = query;
            pending.query_domain_size = n;

            const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                pending.multiplicity_commitment = proof_session->exchange_group1("cq_lookup.multiplicity", {});
            } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
                pending.multiplicity_commitment =
                    proof_session->exchange_group1("cq_lookup.multiplicity", [&] { return multiplicity_commitment; });
            } else {
                pending.multiplicity_commitment = multiplicity_commitment;
            }
            pending_checks_.push_back(std::move(pending));
        });
}

void cqLookupProtocol::add_check(const virCommitment& query, size_t n, const cqLookupSparseWitness& witness)
{
    structured_protocol_profile::timed_void(
        "protocol_call",
        "cq_lookup.add_check",
        "query_domain_size=" + std::to_string(n) + ";with_sparse_witness=1",
        [&] {
            if (!is_setup) {
                throw std::runtime_error("cqLookupProtocol not setup");
            }
            if (params.kzg == nullptr || params.table == nullptr) {
                throw std::runtime_error("cqLookupProtocol dependencies not setup");
            }
            if (witness.multiplicity_terms.size() != witness.table_evals_at_terms.size() ||
                witness.multiplicity_terms.size() != witness.quotient_commitments_at_terms.size()) {
                throw std::runtime_error("cqLookupProtocol: sparse witness size mismatch");
            }

            PendingCheck pending;
            pending.query_commitment = query;
            pending.query_domain_size = n;
            pending.cached_multiplicity_terms =
                std::make_shared<const std::vector<std::pair<size_t, Fr>>>(witness.multiplicity_terms);
            pending.cached_table_evals_at_terms =
                std::make_shared<const std::vector<Fr>>(witness.table_evals_at_terms);
            pending.cached_quotient_commitments_at_terms =
                std::make_shared<const std::vector<Group1>>(witness.quotient_commitments_at_terms);
            pending.cached_support_valid = witness.support_valid;

            const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                pending.multiplicity_commitment = proof_session->exchange_group1("cq_lookup.multiplicity", {});
            } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
                pending.multiplicity_commitment =
                    proof_session->exchange_group1("cq_lookup.multiplicity", [&] {
                        return witness.multiplicity_commitment;
                    });
            } else {
                pending.multiplicity_commitment = witness.multiplicity_commitment;
            }
            pending_checks_.push_back(std::move(pending));
        });
}

bool cqLookupProtocol::finalize()
{
    if (is_finalized || opening_prepared_ || opening_completed_) {
        throw std::runtime_error("cqLookupProtocol already finalized");
    }
    if (!prepare_opening()) {
        return false;
    }

    const Fr gamma = verifier.gamma_challenge();
    return complete_opening(gamma);
}

bool cqLookupProtocol::prepare_opening()
{
    if (opening_prepared_) {
        return true;
    }
    begin_finalize("cqLookupProtocol");
    if (pending_checks_.empty()) {
        opening_prepared_ = true;
        return true;
    }

    const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
    Fr beta = Fr(0);
    cq_lookup_profile_scope("prepare_opening.beta_challenge", [&] { beta = verifier.beta_challenge(); });

    for (const PendingCheck& pending : pending_checks_) {
        cqLookupCheckRecord record;
        record.query_commitment = pending.query_commitment;
        record.query_domain_size = pending.query_domain_size;
        record.beta = beta;
        record.multiplicity_commitment = pending.multiplicity_commitment;

        if (proof_session && proof_session->mode() == ProofMode::Verify) {
            cq_lookup_profile_scope("prepare_opening.verify.consume_record", [&] {
                const Group1 b0_group = proof_session->exchange_group1("cq_lookup.b0", {});
                record.b0_commitment = params.kzg->adopt_commitment(b0_group);
                params.kzg->prove_degree(record.b0_commitment, pending.query_domain_size - 2);

                const Group1 qb_group = proof_session->exchange_group1("cq_lookup.qb", {});
                record.qb_commitment = params.kzg->adopt_commitment(qb_group);

                record.support_valid = proof_session->exchange_bool("cq_lookup.support_valid", {});
                record.denominators_valid = proof_session->exchange_bool("cq_lookup.denominators_valid", {});
                record.quotient_valid = proof_session->exchange_bool("cq_lookup.quotient_valid", {});
                record.a_commitment = proof_session->exchange_group1("cq_lookup.a", {});
                record.qa_commitment = proof_session->exchange_group1("cq_lookup.qa", {});
                record.a0_quotient_commitment = proof_session->exchange_group1("cq_lookup.a0_quotient", {});
                record.a0 = proof_session->exchange_fr("cq_lookup.a0", {});
            });
        } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
            CqLookupBetaWitness witness;
            cq_lookup_profile_scope("prepare_opening.prove.beta_witness", [&] {
                witness = build_cq_lookup_beta_witness(params,
                                                       pending.query_commitment,
                                                       pending.query_domain_size,
                                                       beta,
                                                       pending.cached_query_evals.get(),
                                                       pending.cached_query_poly.get(),
                                                       pending.cached_multiplicity_terms.get(),
                                                       pending.cached_table_evals_at_terms.get(),
                                                       pending.cached_quotient_commitments_at_terms.get(),
                                                       pending.cached_multiplicity_terms != nullptr
                                                           ? std::optional<bool>(pending.cached_support_valid)
                                                           : std::nullopt);
            });
            if (pending.multiplicity_commitment != Group1::zero() &&
                witness.multiplicity_commitment != pending.multiplicity_commitment) {
                throw std::runtime_error("cqLookupProtocol: multiplicity commitment changed after beta");
            }

            record.support_valid = witness.support_valid;
            record.denominators_valid = witness.denominators_valid;
            record.quotient_valid = witness.quotient_valid;
            record.multiplicity_commitment = witness.multiplicity_commitment;

            cq_lookup_profile_scope("prepare_opening.prove.bind_record", [&] {
                const Group1 b0_group =
                    proof_session->exchange_group1("cq_lookup.b0", [&] { return witness.b0_group; });
                kzgCommitment precomputed_b0;
                precomputed_b0.com = witness.b0_group;
                record.b0_commitment = params.kzg->commit(witness.b0_poly, precomputed_b0);
                params.kzg->prove_degree(record.b0_commitment, pending.query_domain_size - 2);

                const Group1 qb_group =
                    proof_session->exchange_group1("cq_lookup.qb", [&] { return witness.qb_group; });
                kzgCommitment precomputed_qb;
                precomputed_qb.com = witness.qb_group;
                record.qb_commitment = params.kzg->commit(witness.qb_poly, precomputed_qb);
                if (b0_group != witness.b0_group || qb_group != witness.qb_group) {
                    throw std::runtime_error("cqLookupProtocol: transported KZG commitments mismatch");
                }

                record.support_valid =
                    proof_session->exchange_bool("cq_lookup.support_valid", [&] { return witness.support_valid; });
                record.denominators_valid = proof_session->exchange_bool(
                    "cq_lookup.denominators_valid", [&] { return witness.denominators_valid; });
                record.quotient_valid =
                    proof_session->exchange_bool("cq_lookup.quotient_valid", [&] { return witness.quotient_valid; });
                record.a_commitment =
                    proof_session->exchange_group1("cq_lookup.a", [&] { return witness.a_commitment; });
                record.qa_commitment =
                    proof_session->exchange_group1("cq_lookup.qa", [&] { return witness.qa_commitment; });
                record.a0_quotient_commitment = proof_session->exchange_group1(
                    "cq_lookup.a0_quotient", [&] { return witness.a0_quotient_commitment; });
                record.a0 = proof_session->exchange_fr("cq_lookup.a0", [&] { return witness.a0; });
            });
            prover.checks.push_back(record);
        } else {
            cq_lookup_profile_scope("prepare_opening.local_record", [&] {
                const CqLookupBetaWitness witness =
                    build_cq_lookup_beta_witness(params,
                                                 pending.query_commitment,
                                                 pending.query_domain_size,
                                                 beta,
                                                 pending.cached_query_evals.get(),
                                                 pending.cached_query_poly.get(),
                                                 pending.cached_multiplicity_terms.get(),
                                                 pending.cached_table_evals_at_terms.get(),
                                                 pending.cached_quotient_commitments_at_terms.get(),
                                                 pending.cached_multiplicity_terms != nullptr
                                                     ? std::optional<bool>(pending.cached_support_valid)
                                                     : std::nullopt);
                record.support_valid = witness.support_valid;
                record.denominators_valid = witness.denominators_valid;
                record.quotient_valid = witness.quotient_valid;
                record.multiplicity_commitment = witness.multiplicity_commitment;
                record.a_commitment = witness.a_commitment;
                record.qa_commitment = witness.qa_commitment;
                record.a0_quotient_commitment = witness.a0_quotient_commitment;
                record.a0 = witness.a0;
                record.b0_commitment = params.kzg->commit(witness.b0_poly);
                if (record.b0_commitment.com != witness.b0_group) {
                    throw std::runtime_error("cqLookupProtocol: b0 commitment registration mismatch");
                }
                params.kzg->prove_degree(record.b0_commitment, pending.query_domain_size - 2);
                record.qb_commitment = params.kzg->commit(witness.qb_poly);
                if (record.qb_commitment.com != witness.qb_group) {
                    throw std::runtime_error("cqLookupProtocol: qb commitment registration mismatch");
                }
                prover.checks.push_back(record);
            });
            if (pending.multiplicity_commitment != Group1::zero() &&
                record.multiplicity_commitment != pending.multiplicity_commitment) {
                throw std::runtime_error("cqLookupProtocol: local multiplicity commitment mismatch");
            }
        }

        cq_lookup_profile_scope("prepare_opening.verifier_add_check", [&] { verifier.add_check(record); });
    }
    pending_checks_.clear();
    opening_prepared_ = true;
    return true;
}

bool cqLookupProtocol::complete_opening(const Fr& gamma)
{
    if (opening_completed_) {
        return true;
    }
    if (!opening_prepared_) {
        throw std::runtime_error("cqLookupProtocol opening completed before prepare_opening");
    }
    cq_lookup_profile_scope("complete_opening.attach_gamma", [&] {
        for (cqLookupCheckRecord& record : verifier.checks) {
            record.gamma = gamma;
        }
        for (cqLookupCheckRecord& record : prover.checks) {
            record.gamma = gamma;
        }
    });
    opening_completed_ = true;
    bool ok = false;
    cq_lookup_profile_scope("complete_opening.verifier_finalize", [&] { ok = verifier.finalize(); });
    return ok;
}
