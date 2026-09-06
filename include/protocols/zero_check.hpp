#pragma once
#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "profiling/soundness_counter.hpp"

using namespace mcl::bn;

/*
* zero check: prove that a polynomial f is zero over a given domain <oemga_n>
* where omega_n is the n-th root of unity
* for batched check, we show
* \sum_i \eta^i f_i(X) v_N(X) / v_ni(X) = q(X) * v_N(X),
* where \eta is a challenge, N is the max n_i.
*/
class zeroParam {
public:
    std::vector<size_t> domain_sizes;
    std::vector<std::string> labels;
    size_t max_domain_size;
    kzgProtocol *kzg;
    terraeSoundnessCounter* soundness_counter = nullptr;
};

class zeroVerifier {
public:

    void add_check(virCommitment& c) {
        commits.push_back(c);
    }

    Fr challenge();

    bool finalize(kzgCommitment q, const Fr& x);

    std::vector<virCommitment> commits;
    Fr eta, x;
    const zeroParam *params;
};

class zeroProver {
public:
    
    void add_check(virCommitment& c) {
        commits.push_back(c);
    }

    kzgCommitment finalize(Fr eta);

    std::vector<virCommitment> commits;
    const zeroParam *params;
};

class zeroProtocol : protected ProtocolLifecycle {
public:
    ~zeroProtocol() noexcept {
        auto_finalize("zeroProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg) {
        params.kzg = &kzg;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    // Register one polynomial that should vanish on the size-`n` subgroup.
    void set_soundness_counter(terraeSoundnessCounter* counter) { params.soundness_counter = counter; }

    void add_check(virCommitment& c, size_t n, const std::string& label = "") {
        if (!isPowerOfTwo(n)) {
            throw std::runtime_error("n must be a power of two");
        }
        proof_cost::record_relation("zero_check", n);
        if (params.soundness_counter != nullptr) {
            params.soundness_counter->record_zero_claim(n);
        }
        params.domain_sizes.push_back(n);
        params.labels.push_back(label);
        verifier.add_check(c);
        const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
        if (!(proof_session && proof_session->mode() == ProofMode::Verify)) {
            prover.add_check(c);
        }
    }

    // Register a Booleanity claim once for a given committed polynomial and
    // domain. Composite protocols frequently reuse the same mask in several
    // masked subprotocols; proving b(b-1)=0 for every reuse is redundant.
    // Alias-based de-duplication is conservative: unrelated commitments are
    // never merged even when they happen to have equal evaluations.
    void add_boolean_check(const virCommitment& values, size_t n, const std::string& label = "") {
        for (const auto& claim : boolean_claims_) {
            if (claim.domain_size == n && same_virtual_commitment_alias(claim.values, values)) {
                return;
            }
        }
        boolean_claims_.push_back({values, n});
        virCommitment boolean_zero = values * (values - virCommitment(Fr(1)));
        add_check(boolean_zero, n, label);
    }

    bool needs_opening() const
    {
        return !params.domain_sizes.empty();
    }

    bool prepare_opening();
    bool complete_opening(const Fr& x);
    bool finalize();

protected:
    struct BooleanClaim {
        virCommitment values;
        size_t domain_size = 0;
    };

    std::vector<BooleanClaim> boolean_claims_;
    zeroVerifier verifier;
    zeroProver prover;
    zeroParam params;
    bool opening_prepared_ = false;
    bool opening_completed_ = false;
    kzgCommitment prepared_q_;
};
