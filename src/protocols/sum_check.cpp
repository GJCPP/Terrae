#include "sum_check.hpp"
#include "algebra_utils.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <map>
#include <string>

namespace {

bool sum_profile_enabled()
{
    const char* flag = terrae_getenv("TERRAE_PROFILE_SUM");
    return flag != nullptr && flag[0] != '\0' && std::string(flag) != "0" && std::string(flag) != "false";
}

template <class Fn>
auto sum_profile_scope(const char* label, Fn&& fn)
{
    if (!sum_profile_enabled()) {
        return fn();
    }
    const auto start = std::chrono::steady_clock::now();
    auto out = fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cerr << "sum_profile " << label << " " << elapsed.count() << "\n";
    return out;
}

template <class Fn>
void sum_profile_scope_void(const char* label, Fn&& fn)
{
    if (!sum_profile_enabled()) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cerr << "sum_profile " << label << " " << elapsed.count() << "\n";
}

Fr vanishing_eval_at(size_t domain_size, const Fr& x)
{
    return pow_fr(x, domain_size) - Fr(1);
}

} // namespace

void sumVerifier::add_check(virCommitment& c, Fr claim) {
    commits.push_back(c);
    claims.push_back(claim);
}

void sumVerifier::add_direct_opening_check(Fr opened_value, Fr expected_value)
{
    direct_opening_checks.emplace_back(opened_value, expected_value);
}

bool sumVerifier::finalize() const
{
    for (const auto& [opened_value, expected_value] : direct_opening_checks) {
        if (opened_value != expected_value) {
            return false;
        }
    }
    return true;
}

// For each check, f(X) - claim/n = r_i(X) X + q_i(X) v_n(X).
// Batch into one quotient and one remainder:
//   Q(X) = sum_i eta^i q_i(X)
//   R(X) = sum_i eta^i r_i(X) v_N(X) / v_n(X), deg R <= N - 2.
std::pair<poly, poly> sumProver::batched_witness(const std::vector<sumPendingCheck>& checks,
                                                 Fr eta,
                                                 size_t max_domain_size) const
{
    poly batched_quotient;
    poly batched_remainder;
    std::map<size_t, poly> grouped_centered;
    Fr factor = Fr(1);
    sum_profile_scope_void("batched_witness.group_materialize", [&] {
        for (const sumPendingCheck& check : checks) {
            poly centered = *params->kzg->materialize(check.commitment);
            centered = centered - check.claim / check.domain_size;
            grouped_centered[check.domain_size] = grouped_centered[check.domain_size] + centered * factor;
            factor *= eta;
        }
    });

    const poly max_vanishing = vanishing_poly(max_domain_size);
    sum_profile_scope_void("batched_witness.divide_domains", [&] {
        for (const auto& [domain_size, centered] : grouped_centered) {
            auto [q, remainder] = div(centered, vanishing_poly(domain_size));
            assert(remainder.is_zero() || remainder.coefficients().front() == Fr(0));
            const poly lifted_remainder =
                (remainder / identity_poly()) * (max_vanishing / vanishing_poly(domain_size));
            batched_quotient = batched_quotient + q;
            batched_remainder = batched_remainder + lifted_remainder;
        }
    });
    return {batched_quotient, batched_remainder};
}

void sumProtocol::add_check(virCommitment &c, Fr claim, size_t n) {
    if (!isPowerOfTwo(n)) {
        throw std::runtime_error("n must be a power of two");
    }
    proof_cost::record_relation("sum_check", n);
    const bool direct_opening = params.kzg->has_degree_bound(c, n - 1);
    if (params.soundness_counter != nullptr) {
        params.soundness_counter->record_sum_claim(direct_opening, n);
    }
    params.domain_sizes.push_back(n);
    verifier.add_check(c, claim);

    if (direct_opening) {
        const Fr opened_value = params.kzg->open(c, Fr(0), kzgOpeningRootRole::SumDirectZero);
        verifier.add_direct_opening_check(opened_value, claim / n);
        return;
    }

    pending_checks_.push_back({c, claim, n});
}

bool sumProtocol::finalize_pending()
{
    if (pending_finalized_) {
        return true;
    }
    pending_finalized_ = true;
    if (pending_checks_.empty()) {
        return true;
    }
    if (params.kzg == nullptr) {
        throw std::runtime_error("sumProtocol not setup");
    }

    params.max_domain_size = 0;
    for (const sumPendingCheck& check : pending_checks_) {
        params.max_domain_size = std::max(params.max_domain_size, check.domain_size);
    }
    size_t source_domain_size_sum = 0;
    for (const sumPendingCheck& check : pending_checks_) {
        source_domain_size_sum += check.domain_size;
    }

    const Fr eta = params.kzg->transcript().squeeze_fr("sum.eta");
    const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
    kzgCommitment quotient_commitment;
    kzgCommitment remainder_commitment;
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        sum_profile_scope_void("adopt_batched_witness", [&] {
            const Group1 quotient_group = proof_session->transport_group1({});
            quotient_commitment = params.kzg->adopt_commitment(quotient_group);
            const Group1 remainder_group = proof_session->transport_group1({});
            remainder_commitment = params.kzg->adopt_commitment(remainder_group);
            params.kzg->prove_degree(remainder_commitment, params.max_domain_size - 2);
        });
    } else {
        const auto [quotient, remainder] = sum_profile_scope("compute_batched_witness", [&] {
            return prover.batched_witness(pending_checks_, eta, params.max_domain_size);
        });
        const kzgCommitment prover_quotient = sum_profile_scope("commit_batched_quotient", [&] {
            return params.kzg->commit(quotient);
        });
        const kzgCommitment prover_remainder = sum_profile_scope("commit_batched_remainder", [&] {
            return params.kzg->commit(remainder, params.max_domain_size - 2);
        });
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 quotient_group = proof_session->transport_group1([&] { return prover_quotient.com; });
            if (quotient_group != prover_quotient.com) {
                throw std::runtime_error("sumProtocol: transported batched quotient commitment mismatch");
            }
            const Group1 remainder_group = proof_session->transport_group1([&] { return prover_remainder.com; });
            if (remainder_group != prover_remainder.com) {
                throw std::runtime_error("sumProtocol: transported batched remainder commitment mismatch");
            }
        }
        quotient_commitment = prover_quotient;
        remainder_commitment = prover_remainder;
    }

    const Fr x = params.kzg->transcript().squeeze_nonzero_fr("sum.x");
    std::map<size_t, Fr> vanishing_evals;
    const auto vanishing_at = [&](size_t domain_size) -> const Fr& {
        const auto it = vanishing_evals.find(domain_size);
        if (it != vanishing_evals.end()) {
            return it->second;
        }
        return vanishing_evals.emplace(domain_size, vanishing_eval_at(domain_size, x)).first->second;
    };
    const Fr max_vanishing_at_x = vanishing_at(params.max_domain_size);
    Fr lhs = Fr(0);
    Fr factor = Fr(1);
    sum_profile_scope_void("open_batched_inputs", [&] {
        for (const sumPendingCheck& check : pending_checks_) {
            const Fr value = params.kzg->open(check.commitment, x, kzgOpeningRootRole::SumBatched) - check.claim / check.domain_size;
            lhs += value * factor * max_vanishing_at_x / vanishing_at(check.domain_size);
            factor *= eta;
        }
    });
    const Fr quotient_value = sum_profile_scope("open_batched_quotient", [&] {
        return params.kzg->open(quotient_commitment, x, kzgOpeningRootRole::SumBatched);
    });
    const Fr remainder_value = sum_profile_scope("open_batched_remainder", [&] {
        return params.kzg->open(remainder_commitment, x, kzgOpeningRootRole::SumBatched);
    });
    const Fr rhs = quotient_value * max_vanishing_at_x + remainder_value * x;
    return lhs == rhs;
}
