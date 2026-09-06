#pragma once
#include <iostream>
#include <utility>
#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"
#include "polynomial.hpp"
#include "protocol_lifecycle.hpp"
#include "profiling/soundness_counter.hpp"
#include "zero_check.hpp"

/*
* sum-check: prove that the sum of a polynomial (over a domain) is equal to a given value.
* For each polynomial f, use Aurora's sumcheck:
* \sum_i f_i = a <===> f(X) - sum/N = r(X) X + q(X) v_N(X), with deg r <= N-2.
*
* After committing r(X) and proving its degree, the rest is
* f(X) - r(X) X = q(X) v_N(X), which is a zero-check.
*/

using namespace mcl::bn;

class sumParam {
public:
    std::vector<size_t> domain_sizes;
    size_t max_domain_size;
    kzgProtocol *kzg;
    zeroProtocol *zero_check;
    terraeSoundnessCounter* soundness_counter = nullptr;
};

class sumVerifier {
public:
    void add_check(virCommitment& c, Fr claim);
    // Some reductions already open the relevant value directly at `X = 0`.
    // These checks are cheaper to store here than to route through zero-check.
    void add_direct_opening_check(Fr opened_value, Fr expected_value);
    bool finalize() const;
    
    std::vector<virCommitment> commits;
    std::vector<Fr> claims;
    std::vector<std::pair<Fr, Fr>> direct_opening_checks;
    const sumParam *params;
};

class sumPendingCheck {
public:
    virCommitment commitment;
    Fr claim;
    size_t domain_size = 0;
};

class sumProver {
public:
    std::pair<poly, poly> batched_witness(const std::vector<sumPendingCheck>& checks,
                                          Fr eta,
                                          size_t max_domain_size) const;

    std::vector<poly> quotients;
    const sumParam *params;
};

class sumProtocol : protected ProtocolLifecycle {
public:
    void setup(kzgProtocol& kzg, zeroProtocol& zero) {
        params.kzg = &kzg;
        params.zero_check = &zero;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    ~sumProtocol() noexcept {
        auto_finalize("sumProtocol", [this] { return finalize(); });
    }

    void set_soundness_counter(terraeSoundnessCounter* counter) { params.soundness_counter = counter; }

    // Register one statement that the sum of `c` over the size-`n` subgroup is
    // exactly `claim`.
    void add_check(virCommitment& c, Fr claim, size_t n);

    bool finalize() {
        begin_finalize("sumProtocol");
        return finalize_pending() && verifier.finalize();
    }
protected:
    bool finalize_pending();

    sumParam params;
    sumProver prover;
    sumVerifier verifier;
    std::vector<sumPendingCheck> pending_checks_;
    bool pending_finalized_ = false;
};
