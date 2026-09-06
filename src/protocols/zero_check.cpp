#include "zero_check.hpp"
#include "algebra_utils.hpp"
#include "terrae_env.hpp"

#include <map>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <numeric>

namespace {

Fr vanishing_eval_at(size_t domain_size, const Fr& x)
{
    return pow_fr(x, domain_size) - Fr(1);
}

std::vector<Fr> quotient_coefficients_by_subgroup_vanishing(const std::vector<Fr>& coeffs, size_t domain_size)
{
    if (coeffs.size() <= domain_size) {
        return {};
    }

    // For f(X) = q(X) * (X^n - 1), coefficients satisfy
    // f_k = q_{k-n} - q_k.  Recover q from high degree to low degree.
    std::vector<Fr> quotient(coeffs.size() - domain_size, Fr(0));
    for (size_t k = coeffs.size(); k-- > domain_size;) {
        quotient[k - domain_size] = coeffs[k];
        if (k < quotient.size()) {
            Fr::add(quotient[k - domain_size], quotient[k - domain_size], quotient[k]);
        }
    }
    return quotient;
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

void add_coefficients(std::vector<Fr>& out, const std::vector<Fr>& delta)
{
    if (out.size() < delta.size()) {
        out.resize(delta.size(), Fr(0));
    }
    for (size_t i = 0; i < delta.size(); ++i) {
        Fr::add(out[i], out[i], delta[i]);
    }
}

bool zero_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_ZERO");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
auto zero_profile_scope(const char* label, Fn&& fn)
{
    if (!zero_profile_enabled()) {
        return fn();
    }
    const auto start = std::chrono::steady_clock::now();
    auto result = fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cerr << "zero_profile " << label << " " << elapsed.count() << "\n";
    return result;
}

template <class Fn>
void zero_profile_scope_void(const char* label, Fn&& fn)
{
    if (!zero_profile_enabled()) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cerr << "zero_profile " << label << " " << elapsed.count() << "\n";
}

struct VirtualStats {
    size_t nodes = 0;
    size_t kzg = 0;
    size_t sparse = 0;
    size_t sum = 0;
    size_t product = 0;
    size_t fraction = 0;
    size_t x_power = 0;
    size_t x_scale = 0;
    size_t interleave = 0;
    size_t lagrange = 0;
    size_t lagrange_linear_combination = 0;
    size_t lagrange_terms = 0;
};

void collect_virtual_stats(const virCommitment& commitment, VirtualStats& stats)
{
    ++stats.nodes;
    switch (commitment.type) {
    case virCommitment::Type::KZG:
        ++stats.kzg;
        return;
    case virCommitment::Type::SPARSE_POLY:
        ++stats.sparse;
        return;
    case virCommitment::Type::SUM:
        ++stats.sum;
        break;
    case virCommitment::Type::PRODUCT:
        ++stats.product;
        break;
    case virCommitment::Type::FRACTION:
        ++stats.fraction;
        break;
    case virCommitment::Type::X_POWER:
        ++stats.x_power;
        break;
    case virCommitment::Type::X_SCALE:
        ++stats.x_scale;
        break;
    case virCommitment::Type::INTERLEAVE:
        ++stats.interleave;
        break;
    case virCommitment::Type::LAGRANGE:
        ++stats.lagrange;
        return;
    case virCommitment::Type::LAGRANGE_LINEAR_COMBINATION:
        ++stats.lagrange_linear_combination;
        stats.lagrange_terms += commitment.lagrange_terms.size();
        return;
    }
    if (commitment.left) {
        collect_virtual_stats(*commitment.left, stats);
    }
    if (commitment.right) {
        collect_virtual_stats(*commitment.right, stats);
    }
}

} // namespace

Fr zeroVerifier::challenge()
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("zeroVerifier not setup");
    }
    eta = params->kzg->transcript().squeeze_fr("zero.eta");
    return eta;
}

bool zeroVerifier::finalize(kzgCommitment q, const Fr& opening_point)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("zeroVerifier not setup");
    }
    if (opening_point.isZero()) {
        throw std::runtime_error("zeroVerifier opening point must be non-zero");
    }

    // Check the batched identity at one fresh point:
    //   sum_i eta^i f_i(X) v_N(X) / v_{n_i}(X) = q(X) v_N(X).
    x = opening_point;
    Fr RHS = zero_profile_scope("open_quotient", [&] {
        return params->kzg->open(q, x, kzgOpeningRootRole::ZeroCheck);
    });
    std::map<size_t, Fr> vanishing_evals;
    const auto vanishing_at = [&](size_t domain_size) -> const Fr& {
        const auto it = vanishing_evals.find(domain_size);
        if (it != vanishing_evals.end()) {
            return it->second;
        }
        return vanishing_evals.emplace(domain_size, vanishing_eval_at(domain_size, x)).first->second;
    };

    RHS = RHS * vanishing_at(params->max_domain_size);
    std::map<size_t, virCommitment> grouped_commits;
    std::map<size_t, std::vector<std::string>> labels_by_domain;
    zero_profile_scope_void("group_commits", [&] {
        Fr factor = Fr(1);
        for (size_t i = 0; i < commits.size(); i++) {
            const size_t domain_size = params->domain_sizes[i];
            if (i < params->labels.size() && !params->labels[i].empty()) {
                labels_by_domain[domain_size].push_back(params->labels[i]);
            }
            const auto it = grouped_commits.find(domain_size);
            if (it == grouped_commits.end()) {
                grouped_commits.emplace(domain_size, virCommitment(Fr(0)) + commits[i] * factor);
            } else {
                it->second = it->second + commits[i] * factor;
            }
            factor *= eta;
        }
    });
    Fr LHS = 0;
    zero_profile_scope_void("open_grouped_commits", [&] {
        for (const auto& [domain_size, grouped_commit] : grouped_commits) {
            Fr f;
            if (zero_profile_enabled()) {
                VirtualStats stats;
                collect_virtual_stats(grouped_commit, stats);
                std::cerr << "zero_profile_stats domain=" << domain_size
                          << " nodes=" << stats.nodes
                          << " kzg=" << stats.kzg
                          << " sparse=" << stats.sparse
                          << " sum=" << stats.sum
                          << " product=" << stats.product
                          << " fraction=" << stats.fraction
                          << " x_power=" << stats.x_power
                          << " x_scale=" << stats.x_scale
                          << " interleave=" << stats.interleave
                          << " lagrange=" << stats.lagrange
                          << " lagrange_lc=" << stats.lagrange_linear_combination
                          << " lagrange_terms=" << stats.lagrange_terms << "\n";
                const auto labels_it = labels_by_domain.find(domain_size);
                if (labels_it != labels_by_domain.end()) {
                    for (const std::string& label : labels_it->second) {
                        std::cerr << "zero_profile_label domain=" << domain_size
                                  << " label=" << label << "\n";
                    }
                }
                const auto start = std::chrono::steady_clock::now();
                f = params->kzg->open(grouped_commit, x, kzgOpeningRootRole::ZeroCheck);
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                std::cerr << "zero_profile open_domain."
                          << domain_size << " " << elapsed.count() << "\n";
            } else {
                f = params->kzg->open(grouped_commit, x, kzgOpeningRootRole::ZeroCheck);
            }
            LHS = LHS + f * vanishing_at(params->max_domain_size) / vanishing_at(domain_size);
        }
    });
    if (LHS == RHS) {
        return true;
    }

    if (const char* debug = terrae_getenv("TERRAE_DEBUG_ZERO_FAIL"); debug != nullptr && debug[0] != '\0') {
        std::cerr << "zero_check mismatch at x=" << x.getStr(10) << " eta=" << eta.getStr(10) << '\n';
        std::cerr << "zero_check lhs=" << LHS.getStr(10) << " rhs=" << RHS.getStr(10) << '\n';
        Fr term_factor = Fr(1);
        for (size_t i = 0; i < commits.size(); ++i) {
            Fr f = params->kzg->open(commits[i], x, kzgOpeningRootRole::ZeroCheck);
            Fr term = f * term_factor * vanishing_at(params->max_domain_size) / vanishing_at(params->domain_sizes[i]);
            std::cerr << "zero_check term[" << i << "] n=" << params->domain_sizes[i];
            if (i < params->labels.size() && !params->labels[i].empty()) {
                std::cerr << " label=" << params->labels[i];
            }
            std::cerr << " value=" << term.getStr(10) << '\n';
            term_factor *= eta;
        }
    }
    return false;
}

kzgCommitment zeroProver::finalize(Fr eta)
{
    std::map<size_t, std::vector<Fr>> grouped_coefficients;
    zero_profile_scope_void("prove.materialize_and_accumulate_groups", [&] {
        Fr factor = Fr(1);
        for (size_t i = 0; i < commits.size(); ++i) {
            const size_t domain_size = params->domain_sizes[i];
            const std::shared_ptr<poly> materialized = params->kzg->materialize(commits[i]);
            add_scaled_poly_coefficients(grouped_coefficients[domain_size], *materialized, factor);
            factor *= eta;
        }
    });

    std::vector<Fr> sum_coefficients;
    for (const auto& [domain_size, grouped_coeffs] : grouped_coefficients) {
        // Checks sharing the same domain size can be aggregated before the exact
        // division by that domain's vanishing polynomial.
        zero_profile_scope_void("prove.divide_group", [&] {
            add_coefficients(sum_coefficients,
                             quotient_coefficients_by_subgroup_vanishing(grouped_coeffs, domain_size));
        });
    }
    return zero_profile_scope("prove.commit_quotient", [&] {
        return params->kzg->commit(poly_from_coeff(sum_coefficients));
    });
}

bool zeroProtocol::finalize() {
    if (is_finalized || opening_prepared_ || opening_completed_) {
        throw std::runtime_error("zeroProtocol already finalized");
    }
    if (!prepare_opening()) {
        return false;
    }

    const Fr local_opening = params.kzg->transcript().squeeze_nonzero_fr("zero.x");
    return complete_opening(local_opening);
}

bool zeroProtocol::prepare_opening()
{
    if (opening_prepared_) {
        return true;
    }
    begin_finalize("zeroProtocol");
    if (params.domain_sizes.empty()) {
        opening_prepared_ = true;
        return true;
    }

    const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
    params.max_domain_size = *std::max_element(params.domain_sizes.begin(), params.domain_sizes.end());

    auto eta = verifier.challenge();
    if (proof_session && proof_session->mode() == ProofMode::Prove) {
        // The quotient commitment is Fiat-Shamir-visible only through the later
        // opening, so it is transported without an explicit transcript label.
        const kzgCommitment prover_q = prover.finalize(eta);
        const Group1 q_group = proof_session->transport_group1([&] { return prover_q.com; });
        prepared_q_ = prover_q;
        if (prepared_q_.com != q_group) {
            throw std::runtime_error("zeroProtocol: transported quotient commitment mismatch");
        }
    } else if (proof_session && proof_session->mode() == ProofMode::Verify) {
        const Group1 q_group = proof_session->transport_group1({});
        prepared_q_ = params.kzg->adopt_commitment(q_group);
    } else {
        prepared_q_ = prover.finalize(eta);
    }
    opening_prepared_ = true;
    return true;
}

bool zeroProtocol::complete_opening(const Fr& x)
{
    if (opening_completed_) {
        return true;
    }
    if (params.domain_sizes.empty()) {
        return true;
    }
    if (!opening_prepared_) {
        throw std::runtime_error("zeroProtocol opening completed before prepare_opening");
    }
    opening_completed_ = true;
    return verifier.finalize(prepared_q_, x);
}
