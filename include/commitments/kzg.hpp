#pragma once
#include <iostream>
#include <fstream>
#include <cstdint>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"
#include "profiling/proof_cost.hpp"
#include "proof_session.hpp"
#include "protocol_lifecycle.hpp"
#include "transcript.hpp"

using namespace mcl::bn;

class kzgCommitment {
public:
    Group1 com;
    size_t index;
};

enum class kzgOpeningRootRole : std::uint8_t {
    Unspecified = 0,
    SumDirectZero,
    SumBatched,
    ZeroCheck,
    CqLookup,
    FixedEvaluation,
    ExplicitCheck,
};

struct kzgOpeningProvenance {
    Fr root_point = Fr(0);
    kzgOpeningRootRole root_role = kzgOpeningRootRole::Unspecified;
    size_t interleave_layer_count = 0;
    size_t interleave_shift_count = 0;
    size_t x_power_count = 0;
    size_t x_scale_count = 0;
};

/*
* virtual polynomial commitment
* type:
* 0: kzg polynomial commitment
* 1: sum of two virtual polynomials
* 2: product of two virtual polynomials
* 3: fraction of two virtual polynomials
* 4: direct polynomial leaf (including constants)
* 5: virtual polynomial evaluated at X^n
* 6: virtual polynomial evaluated at cX
* 7: one subgroup Lagrange basis polynomial
* 8: sparse linear combination of subgroup Lagrange basis polynomials
*/
class virCommitment { 
public:
    
    enum class Type {
        KZG,
        SUM,
        PRODUCT,
        FRACTION,
        SPARSE_POLY,
        X_POWER,
        X_SCALE,
        INTERLEAVE,
        LAGRANGE,
        LAGRANGE_LINEAR_COMBINATION,
    };
    
    virCommitment() : type(Type::SPARSE_POLY), leaf() {}

    virCommitment(kzgCommitment com) : type(Type::KZG), com(com) {}

    virCommitment(const poly& poly) : type(Type::SPARSE_POLY), leaf(poly) {}

    // Compatibility helper: treat a field element as a constant polynomial.
    virCommitment(const Fr& c) : type(Type::SPARSE_POLY) {
        leaf.add_term(0, c);
    }

    Type type;
    kzgCommitment com;
    std::shared_ptr<const virCommitment> left = nullptr;
    std::shared_ptr<const virCommitment> right = nullptr;
    poly leaf;
    size_t exponent = 0;
    Fr scale = Fr(0);
    size_t lagrange_subgroup_size = 0;
    size_t lagrange_index = 0;
    std::vector<std::pair<size_t, Fr>> lagrange_terms;
    mutable std::shared_ptr<poly> materialized_cache = nullptr;
    mutable std::map<size_t, std::shared_ptr<const std::vector<Fr>>> evaluation_cache;
};

std::string transcript_representation(const kzgCommitment& com);

virCommitment operator+(const virCommitment& left, const virCommitment& right);
virCommitment operator-(const virCommitment& left, const virCommitment& right);
virCommitment operator*(const virCommitment& left, const virCommitment& right);
virCommitment operator/(const virCommitment& left, const virCommitment& right);
virCommitment x_pow(const virCommitment& base, size_t exponent);
virCommitment x_scale(const virCommitment& base, const Fr& scale);
bool same_virtual_commitment_alias(const virCommitment& left, const virCommitment& right);
bool is_zero_virtual_commitment(const virCommitment& commitment);

/**
 * Evaluate virtual commitment at x (prover uses polys; verifier uses opened values).
 * Collects (index, value) for each KZG leaf into kzg_openings.
 */
Fr eval_virtual(const virCommitment& V, Fr x,
                const std::function<Fr(size_t, const Fr&)>& get_kzg_value,
                std::map<std::pair<size_t, Fr>, Fr>& kzg_openings);

// public parameter accessed only by kzgProtocol
class kzgPublicParameters {
public:
    size_t deg1 = 0;
    size_t deg2 = 0;
    std::vector<Group1> g1_powers;
    std::vector<Group2> g2_powers;
    std::vector<G1> g1_raw_powers;
    std::vector<G2> g2_raw_powers;
};

kzgPublicParameters generate_kzg_public_parameters(size_t deg1_plus_1, size_t deg2_plus_1);
kzgPublicParameters trim_kzg_public_parameters(const kzgPublicParameters& params,
                                              size_t deg1_plus_1,
                                              size_t deg2_plus_1);
void write_kzg_public_parameters(const kzgPublicParameters& params, const std::string& path);
kzgPublicParameters read_kzg_public_parameters(const std::string& path);
kzgPublicParameters read_kzg_public_parameters(const std::string& path,
                                               size_t deg1_plus_1,
                                               size_t deg2_plus_1);
Group1 commit_g1_with_public_parameters(const kzgPublicParameters& params, const poly& p);
Group2 commit_g2_with_public_parameters(const kzgPublicParameters& params, const poly& p);
inline kzgPublicParameters load_kzg_public_parameters(const std::string& path)
{
    return read_kzg_public_parameters(path);
}

inline kzgPublicParameters load_kzg_public_parameters(const std::string& path,
                                                      size_t deg1_plus_1,
                                                      size_t deg2_plus_1)
{
    return read_kzg_public_parameters(path, deg1_plus_1, deg2_plus_1);
}

class kzgParams {
public:
    kzgParams(size_t deg1_plus_1 = 1, size_t deg2_plus_1 = 1);

    size_t deg1;
    size_t deg2;
    std::vector<Group1> g1_powers;
    std::vector<Group2> g2_powers;
    std::vector<G1> g1_raw_powers;
    std::vector<G2> g2_raw_powers;
    const kzgPublicParameters* borrowed_public_parameters = nullptr;

    void borrow_public_parameters(const kzgPublicParameters& public_parameters)
    {
        deg1 = public_parameters.deg1;
        deg2 = public_parameters.deg2;
        borrowed_public_parameters = &public_parameters;
        g1_powers.clear();
        g2_powers.clear();
        g1_raw_powers.clear();
        g2_raw_powers.clear();
    }

    const std::vector<Group1>& g1_powers_view() const
    {
        return borrowed_public_parameters ? borrowed_public_parameters->g1_powers : g1_powers;
    }

    const std::vector<Group2>& g2_powers_view() const
    {
        return borrowed_public_parameters ? borrowed_public_parameters->g2_powers : g2_powers;
    }

    const std::vector<G1>& g1_raw_powers_view() const
    {
        return borrowed_public_parameters ? borrowed_public_parameters->g1_raw_powers : g1_raw_powers;
    }

    const std::vector<G2>& g2_raw_powers_view() const
    {
        return borrowed_public_parameters ? borrowed_public_parameters->g2_raw_powers : g2_raw_powers;
    }

    std::vector<std::vector<Fr>> x, y; // x[i][j] is the j-th opening point for the i-th polynomial
    std::map<std::pair<size_t, Fr>, std::vector<kzgOpeningProvenance>> opening_provenance;
    std::vector<std::array<size_t, 2>> degree; // degree[i] = [index, deg]
    std::unordered_map<size_t, size_t> best_degree_by_index;
    Fr opening_batch_challenge = Fr(0);
    ProofTranscript* transcript = nullptr;
};

class kzgProver {
public:
    std::vector<std::shared_ptr<poly>> polys;
    const kzgParams *params;

    kzgCommitment commit(const poly& p, size_t index);

    inline void add_commit(poly p) {
        polys.push_back(std::make_shared<poly>(std::move(p)));
    }

    inline Fr open(const kzgCommitment& com, Fr x) {
        return polys[com.index]->eval(x);
    }

    Group1 finalize_degree(Fr c);

    std::vector<Group1> finalize();

};

class kzgVerifier {
public:
    inline void commit(const kzgCommitment& com) {
        coms.push_back(com);
    }

    inline void add_commit(const kzgCommitment& com) {
        coms.push_back(com);
    }

    // draw challenge
    Fr finalize_degree_1();

    bool finalize_degree_2(const Group1& f) const;

    // Build verifier-side KZG equations separately from their final pairing
    // checks for callers with explicit per-proof preprocessing.
    bool prepare_finalization(const std::vector<Group1>& q);
    bool verify_prepared_finalization() const;

    bool finalize(const std::vector<Group1>& q);

    std::vector<kzgCommitment> coms;
    GroupT combined_com;
    std::vector<Group1> degree_pairing_g1;
    std::vector<Group2> degree_pairing_g2;
    std::vector<Group1> opening_pairing_g1;
    std::vector<Group2> opening_pairing_g2;
    bool opening_finalization_prepared = false;
    Fr degree_challenge = Fr(0);
    const kzgParams *params;

};

class kzgProtocol : protected ProtocolLifecycle {
protected:
    size_t next_index = 0;

    kzgParams params;
    kzgProver prover;
    kzgVerifier verifier;
    std::map<std::string, kzgCommitment> public_commitment_cache;
    std::set<size_t> opaque_commitment_indices_;
    std::shared_ptr<ProofTranscript> transcript_ = std::make_shared<ProofTranscript>();
    std::shared_ptr<ProofSession> proof_session_ = nullptr;
    std::unordered_map<std::string, Fr> shared_opening_challenge_cache_;
    mutable std::unordered_map<size_t, std::shared_ptr<poly>> materialized_kzg_cache_;
    mutable std::map<std::pair<size_t, size_t>, std::shared_ptr<const std::vector<Fr>>> direct_evaluation_cache_;
    bool defer_verifier_pairing_checks_ = false;
    bool verifier_finalization_prepared_ = false;
    bool verifier_finalization_prepare_ok_ = false;
    bool count_only_ = false;

    void record_opening_provenance(size_t commitment_index,
                                   const Fr& actual_point,
                                   const kzgOpeningProvenance& provenance);

public:
    enum class EvaluationMode {
        Materialize,
        PreferEval,
    };

    std::size_t commitment_count() const noexcept
    {
        return next_index;
    }

    bool is_count_only() const noexcept
    {
        return count_only_;
    }

    inline void setup_count_only(size_t deg1_plus_1, size_t deg2_plus_1) {
        if (deg1_plus_1 == 0 || deg2_plus_1 == 0) {
            throw std::runtime_error("kzgProtocol::setup_count_only requires non-zero dimensions");
        }
        next_index = 0;
        params = kzgParams();
        params.deg1 = deg1_plus_1 - 1;
        params.deg2 = deg2_plus_1 - 1;
        transcript_ = std::make_shared<ProofTranscript>();
        proof_session_ = nullptr;
        params.transcript = transcript_.get();
        prover = kzgProver();
        verifier = kzgVerifier();
        prover.params = &params;
        verifier.params = &params;
        public_commitment_cache.clear();
        opaque_commitment_indices_.clear();
        shared_opening_challenge_cache_.clear();
        materialized_kzg_cache_.clear();
        direct_evaluation_cache_.clear();
        defer_verifier_pairing_checks_ = false;
        verifier_finalization_prepared_ = false;
        verifier_finalization_prepare_ok_ = false;
        count_only_ = true;
        mark_setup();
    }
    inline void setup(size_t deg1_plus_1, size_t deg2_plus_1) {
        (void)deg1_plus_1;
        (void)deg2_plus_1;
        throw std::runtime_error(
            "kzgProtocol::setup(deg1, deg2) is disabled; load or generate explicit kzgPublicParameters and call "
            "setup(public_parameters) instead");
    }

    inline void setup(const kzgPublicParameters& public_parameters) {
        next_index = 0;
        params = kzgParams();
        params.deg1 = public_parameters.deg1;
        params.deg2 = public_parameters.deg2;
        params.g1_powers = public_parameters.g1_powers;
        params.g2_powers = public_parameters.g2_powers;
        params.g1_raw_powers = public_parameters.g1_raw_powers;
        params.g2_raw_powers = public_parameters.g2_raw_powers;
        transcript_ = std::make_shared<ProofTranscript>();
        proof_session_ = nullptr;
        params.transcript = transcript_.get();
        prover = kzgProver();
        verifier = kzgVerifier();
        prover.params = &params;
        verifier.params = &params;
        public_commitment_cache.clear();
        opaque_commitment_indices_.clear();
        shared_opening_challenge_cache_.clear();
        materialized_kzg_cache_.clear();
        direct_evaluation_cache_.clear();
        defer_verifier_pairing_checks_ = false;
        verifier_finalization_prepared_ = false;
        verifier_finalization_prepare_ok_ = false;
        count_only_ = false;
        mark_setup();
    }
    // The caller must keep public_parameters alive for the full protocol lifetime.
    inline void setup_borrowed(const kzgPublicParameters& public_parameters) {
        next_index = 0;
        params = kzgParams();
        params.borrow_public_parameters(public_parameters);
        transcript_ = std::make_shared<ProofTranscript>();
        proof_session_ = nullptr;
        params.transcript = transcript_.get();
        prover = kzgProver();
        verifier = kzgVerifier();
        prover.params = &params;
        verifier.params = &params;
        public_commitment_cache.clear();
        opaque_commitment_indices_.clear();
        shared_opening_challenge_cache_.clear();
        materialized_kzg_cache_.clear();
        direct_evaluation_cache_.clear();
        defer_verifier_pairing_checks_ = false;
        verifier_finalization_prepared_ = false;
        verifier_finalization_prepare_ok_ = false;
        count_only_ = false;
        mark_setup();
    }

    inline void setup(kzgPublicParameters&& public_parameters) {
        next_index = 0;
        params = kzgParams();
        params.deg1 = public_parameters.deg1;
        params.deg2 = public_parameters.deg2;
        params.g1_powers = std::move(public_parameters.g1_powers);
        params.g2_powers = std::move(public_parameters.g2_powers);
        params.g1_raw_powers = std::move(public_parameters.g1_raw_powers);
        params.g2_raw_powers = std::move(public_parameters.g2_raw_powers);
        transcript_ = std::make_shared<ProofTranscript>();
        proof_session_ = nullptr;
        params.transcript = transcript_.get();
        prover = kzgProver();
        verifier = kzgVerifier();
        prover.params = &params;
        verifier.params = &params;
        public_commitment_cache.clear();
        opaque_commitment_indices_.clear();
        shared_opening_challenge_cache_.clear();
        materialized_kzg_cache_.clear();
        direct_evaluation_cache_.clear();
        defer_verifier_pairing_checks_ = false;
        verifier_finalization_prepared_ = false;
        verifier_finalization_prepare_ok_ = false;
        count_only_ = false;
        mark_setup();
    }

    void set_transcript(std::shared_ptr<ProofTranscript> transcript)
    {
        if (!transcript) {
            throw std::runtime_error("kzgProtocol transcript must not be null");
        }
        transcript_ = std::move(transcript);
        params.transcript = transcript_.get();
    }

    ProofTranscript& transcript()
    {
        if (!transcript_) {
            throw std::runtime_error("kzgProtocol transcript not initialized");
        }
        return *transcript_;
    }

    void set_proof_session(std::shared_ptr<ProofSession> proof_session)
    {
        proof_session_ = std::move(proof_session);
        if (proof_session_) {
            set_transcript(proof_session_->shared_transcript());
        }
    }

    std::shared_ptr<ProofSession> proof_session() const
    {
        return proof_session_;
    }

    size_t deg1() const
    {
        return params.deg1;
    }

    size_t deg2() const
    {
        return params.deg2;
    }

    kzgPublicParameters public_parameters() const
    {
        kzgPublicParameters public_params;
        public_params.deg1 = params.deg1;
        public_params.deg2 = params.deg2;
        public_params.g1_powers = params.g1_powers_view();
        public_params.g2_powers = params.g2_powers_view();
        public_params.g1_raw_powers = params.g1_raw_powers_view();
        public_params.g2_raw_powers = params.g2_raw_powers_view();
        return public_params;
    }

    const ProofTranscript& transcript() const
    {
        if (!transcript_) {
            throw std::runtime_error("kzgProtocol transcript not initialized");
        }
        return *transcript_;
    }

    Fr shared_opening_challenge(const std::string& stage_label, bool nonzero = true)
    {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }

        const auto it = shared_opening_challenge_cache_.find(stage_label);
        if (it != shared_opening_challenge_cache_.end()) {
            return it->second;
        }

        const std::string transcript_label = "kzg.shared_opening." + stage_label;
        const Fr sampled =
            nonzero ? transcript().squeeze_nonzero_fr(transcript_label) : transcript().squeeze_fr(transcript_label);
        shared_opening_challenge_cache_.emplace(stage_label, sampled);
        return sampled;
    }
    
    ~kzgProtocol() noexcept {
        auto_finalize("kzgProtocol", [this] { return finalize(); });
    }

    inline kzgCommitment commit(poly p) {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }
        assert(params.x.size() == next_index);
        assert(params.y.size() == next_index);

        kzgCommitment com;
        if (count_only_) {
            com.index = next_index++;
            com.com = Fr(static_cast<std::int64_t>(com.index + 1)) * Group1::one();
        } else {
            com = prover.commit(p, next_index++);
            prover.add_commit(std::move(p));
        }
        verifier.commit(com);
        transcript().absorb_string("kzg.commitment", transcript_representation(com));
        params.x.resize(next_index);
        params.y.resize(next_index);

        return com;
    }

    inline kzgCommitment commit(poly p, size_t max_degree) {
        auto ret = commit(std::move(p));
        prove_degree(ret, max_degree);
        return ret;
    }

    inline kzgCommitment commit_with_group(poly p, const Group1& group_commitment) {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }
        assert(params.x.size() == next_index);
        assert(params.y.size() == next_index);

        kzgCommitment com;
        com.com = group_commitment;
        com.index = next_index++;
        if (count_only_) {
            com.com = Fr(static_cast<std::int64_t>(com.index + 1)) * Group1::one();
        }
        proof_cost::add("adopted_polynomial_commitment_count");
        proof_cost::add("adopted_polynomial_coefficient_count", p.coefficients().size());
        proof_cost::set_max("max_adopted_polynomial_degree",
                            p.coefficients().empty() ? 0 : p.coefficients().size() - 1);
        if (!count_only_) {
            prover.add_commit(std::move(p));
        }
        verifier.add_commit(com);
        transcript().absorb_string("kzg.commitment", transcript_representation(com));
        params.x.resize(next_index);
        params.y.resize(next_index);
        return com;
    }

    inline kzgCommitment commit_with_group(poly p, const Group1& group_commitment, size_t max_degree) {
        auto ret = commit_with_group(std::move(p), group_commitment);
        prove_degree(ret, max_degree);
        return ret;
    }

    inline kzgCommitment adopt_commitment(const Group1& group_commitment) {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }
        kzgCommitment com;
        com.com = group_commitment;
        com.index = next_index++;
        if (count_only_) {
            com.com = Fr(static_cast<std::int64_t>(com.index + 1)) * Group1::one();
        }
        if (prover.polys.size() < next_index) {
            prover.polys.resize(next_index);
        }
        opaque_commitment_indices_.insert(com.index);
        verifier.add_commit(com);
        transcript().absorb_string("kzg.commitment", transcript_representation(com));
        params.x.resize(next_index);
        params.y.resize(next_index);
        proof_cost::add("opaque_adopted_commitment_count");
        return com;
    }

    inline kzgCommitment commit_public(const std::string& key, const poly& p) {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }

        const auto it = public_commitment_cache.find(key);
        if (it != public_commitment_cache.end()) {
            return it->second;
        }

        const kzgCommitment com = commit(p);
        proof_cost::add("public_commitment_count");
        public_commitment_cache.emplace(key, com);
        return com;
    }

    inline kzgCommitment commit_public(const std::string& key, const poly& p, size_t max_degree) {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }

        const auto it = public_commitment_cache.find(key);
        if (it != public_commitment_cache.end()) {
            if (!has_proven_degree(it->second, max_degree)) {
                prove_degree(it->second, max_degree);
            }
            return it->second;
        }

        const kzgCommitment com = commit(p, max_degree);
        proof_cost::add("public_commitment_count");
        public_commitment_cache.emplace(key, com);
        return com;
    }

    inline kzgCommitment commit_public_with_group(const std::string& key,
                                                  const poly& p,
                                                  const Group1& group_commitment) {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }

        const auto it = public_commitment_cache.find(key);
        if (it != public_commitment_cache.end()) {
            if (!count_only_ && it->second.com != group_commitment) {
                throw std::runtime_error("kzgProtocol public commitment key reused with different group element");
            }
            return it->second;
        }

        const kzgCommitment com = commit_with_group(p, group_commitment);
        proof_cost::add("public_commitment_count");
        public_commitment_cache.emplace(key, com);
        return com;
    }

    inline kzgCommitment commit_public_with_group(const std::string& key,
                                                  const poly& p,
                                                  const Group1& group_commitment,
                                                  size_t max_degree) {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }

        const auto it = public_commitment_cache.find(key);
        if (it != public_commitment_cache.end()) {
            if (!count_only_ && it->second.com != group_commitment) {
                throw std::runtime_error("kzgProtocol public commitment key reused with different group element");
            }
            if (!has_proven_degree(it->second, max_degree)) {
                prove_degree(it->second, max_degree);
            }
            return it->second;
        }

        const kzgCommitment com = commit_with_group(p, group_commitment, max_degree);
        proof_cost::add("public_commitment_count");
        public_commitment_cache.emplace(key, com);
        return com;
    }

    inline kzgCommitment adopt_public_commitment(const std::string& key, const Group1& group_commitment) {
        if (!is_setup) {
            throw std::runtime_error("kzgProtocol not setup");
        }

        const auto it = public_commitment_cache.find(key);
        if (it != public_commitment_cache.end()) {
            if (!count_only_ && it->second.com != group_commitment) {
                throw std::runtime_error("kzgProtocol public commitment key reused with different group element");
            }
            return it->second;
        }

        kzgCommitment com = adopt_commitment(group_commitment);
        proof_cost::add("public_commitment_count");
        public_commitment_cache.emplace(key, com);
        return com;
    }

    // Assuming (p, com) is valid pair, e.g. obtained by linear combination of previous commitments.
    inline kzgCommitment commit(poly p, kzgCommitment com) {
        com.index = next_index++;
        if (count_only_) {
            com.com = Fr(static_cast<std::int64_t>(com.index + 1)) * Group1::one();
        }
        proof_cost::add("derived_polynomial_commitment_count");
        proof_cost::add("derived_polynomial_coefficient_count", p.coefficients().size());
        proof_cost::set_max("max_derived_polynomial_degree",
                            p.coefficients().empty() ? 0 : p.coefficients().size() - 1);
        if (!count_only_) {
            prover.add_commit(std::move(p));
        }
        verifier.add_commit(com);
        transcript().absorb_string("kzg.commitment", transcript_representation(com));
        params.x.resize(next_index);
        params.y.resize(next_index);
        return com;
    }

    inline Fr open(const kzgCommitment& com,
                   Fr x,
                   kzgOpeningRootRole role = kzgOpeningRootRole::Unspecified) {
        kzgOpeningProvenance provenance;
        provenance.root_point = x;
        provenance.root_role = role;
        record_opening_provenance(com.index, x, provenance);
        for (size_t i = 0; i < params.x[com.index].size(); ++i) {
            if (params.x[com.index][i] == x) {
                return params.y[com.index][i];
            }
        }

        Fr y;
        if (proof_session_ && proof_session_->mode() == ProofMode::Verify) {
            y = proof_session_->exchange_fr("kzg.opening.value", {});
        } else if (proof_session_ && proof_session_->mode() == ProofMode::Prove) {
            y = proof_session_->exchange_fr("kzg.opening.value", [&] { return prover.open(com, x); });
        } else {
            y = prover.open(com, x);
        }
        params.x[com.index].push_back(x);
        params.y[com.index].push_back(y);
        return y;
    }

    inline void add_opening_check(const kzgCommitment& com,
                                  const Fr& x,
                                  const Fr& expected_y,
                                  kzgOpeningRootRole role = kzgOpeningRootRole::ExplicitCheck) {
        kzgOpeningProvenance provenance;
        provenance.root_point = x;
        provenance.root_role = role;
        record_opening_provenance(com.index, x, provenance);
        for (size_t i = 0; i < params.x[com.index].size(); ++i) {
            if (params.x[com.index][i] != x) {
                continue;
            }
            if (params.y[com.index][i] != expected_y) {
                throw std::runtime_error("kzgProtocol::add_opening_check conflicting opening value");
            }
            return;
        }

        if (proof_session_ && proof_session_->mode() == ProofMode::Prove &&
            com.index < prover.polys.size() && can_materialize(com)) {
            const Fr actual_y = prover.open(com, x);
            if (actual_y != expected_y) {
                throw std::runtime_error("kzgProtocol::add_opening_check expected opening mismatch");
            }
        }

        params.x[com.index].push_back(x);
        params.y[com.index].push_back(expected_y);
    }

    /**
     * Open a virtual polynomial commitment at point x.
     * Reduces to opening each underlying KZG commitment at x.
     * Returns the value V(x), and records (x, y_i) for each KZG polynomial i that V depends on.
     */
    Fr open(const virCommitment& V,
            Fr x,
            kzgOpeningRootRole role = kzgOpeningRootRole::Unspecified);

    /**
     * Materialize the polynomial represented by a virtual commitment.
     * Resolves each KZG leaf to its committed prover-side polynomial and then
     * applies the virtual expression operators.
     */
    std::shared_ptr<poly> materialize(const virCommitment& V) const;

    /**
     * Evaluate a virtual commitment on the multiplicative subgroup of size n.
     * This materializes the virtual polynomial once, then derives the domain
     * evaluations via the polynomial utilities.
     */
    std::vector<Fr> evaluations(const virCommitment& V,
                                size_t n,
                                EvaluationMode mode = EvaluationMode::Materialize) const;

    /**
     * Batch-compute KZG opening commitments for every point in the size-n
     * roots-of-unity domain. The i-th output is a commitment to
     * (p(X) - p(w^i)) / (X - w^i).
     */
    std::vector<Group1> batch_opening_commitments(const poly& p, size_t n) const;

    /**
     * Cached variant for virtual commitments. KZG leaves may reuse previously
     * computed full-domain opening commitments keyed by SRS/domain/commitment.
     */
    std::vector<Group1> batch_opening_commitments(const virCommitment& V, size_t n) const;

    Group2 commit_g2(const poly& p) const;
    Group1 commit_g1_group(const poly& p) const;

    const Group1& g1_power(size_t exponent) const;

    const Group2& g2_power(size_t exponent) const;

    inline void prove_degree(const kzgCommitment& com, size_t degree) {
        params.degree.push_back({com.index, degree});
        const auto it = params.best_degree_by_index.find(com.index);
        if (it == params.best_degree_by_index.end() || degree < it->second) {
            params.best_degree_by_index[com.index] = degree;
        }
        transcript().absorb_string("kzg.degree.commitment", transcript_representation(com));
        transcript().absorb_size("kzg.degree.bound", degree);
    }

    bool has_proven_degree(const kzgCommitment& com, size_t degree) const;
    bool has_degree_bound(const virCommitment& V, size_t degree) const;
    bool can_materialize(const kzgCommitment& com) const;
    
    bool finalize();

    // Verifier-only split of finalize(). It consumes the same proof elements
    // and transcript challenges, but postpones the final pairing checks until
    // verify_prepared_finalization().
    bool prepare_verifier_finalization();
    bool verify_prepared_finalization();

protected:
    bool finalize_degree();
};
