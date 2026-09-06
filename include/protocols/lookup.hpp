#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "profiling/soundness_counter.hpp"
#include "sum_check.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class lookupCheckRecord {
public:
    size_t query_domain_size = 0;
    size_t table_domain_size = 0;
    Fr challenge = Fr(0);
    Fr query_claim = Fr(0);
    Fr table_claim = Fr(0);
    kzgCommitment multiplicity_commitment;
    kzgCommitment query_inverse_commitment;
    kzgCommitment table_weighted_inverse_commitment;
};

class lookupParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    sumProtocol* sum_check = nullptr;
    terraeSoundnessCounter* soundness_counter = nullptr;
};

class lookupVerifier {
public:
    Fr challenge();

    void add_check(const virCommitment& multiplicity, const lookupCheckRecord& record);

    bool finalize() const;

    std::vector<virCommitment> multiplicities;
    std::vector<lookupCheckRecord> checks;
    const lookupParam* params = nullptr;
};

class lookupProver {
public:
    std::pair<virCommitment, lookupCheckRecord> add_check(const virCommitment& f,
                                                          const virCommitment& t,
                                                          const Fr& challenge,
                                                          size_t n,
                                                          size_t N);

    std::vector<lookupCheckRecord> checks;
    const lookupParam* params = nullptr;
};

// Classic multiset inclusion check over two evaluation domains.
//
// Statement:
// - every query value in `f` appears in table `t`
// - multiplicities are tracked by a witness polynomial over the table domain
// - inverse witnesses tie the multiset identity to one transcript challenge
class lookupProtocol : protected ProtocolLifecycle {
public:
    ~lookupProtocol() noexcept
    {
        auto_finalize("lookupProtocol", [this] { return finalize(); });
    }

    void set_soundness_counter(terraeSoundnessCounter* counter) { params.soundness_counter = counter; }

    void setup(kzgProtocol& kzg, zeroProtocol& zero, sumProtocol& sum)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        params.sum_check = &sum;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    // Register one lookup statement with query-domain size `n` and table-domain
    // size `N`.
    void add_check(const virCommitment& f, const virCommitment& t, size_t n, size_t N, std::string debug_label = "");

    bool finalize()
    {
        begin_finalize("lookupProtocol");
        materialize_pending_checks();
        return verifier.finalize();
    }

protected:
    struct pendingLookupCheck {
        virCommitment query;
        virCommitment table;
        size_t query_domain_size = 0;
        size_t table_domain_size = 0;
        std::string debug_label;
    };

    void materialize_pending_checks();
    void add_materialized_check(const virCommitment& f,
                                const virCommitment& t,
                                size_t n,
                                size_t N,
                                bool table_is_contiguous_range = false);

    lookupParam params;
    lookupProver prover;
    lookupVerifier verifier;
    std::vector<pendingLookupCheck> pending_checks;
};
