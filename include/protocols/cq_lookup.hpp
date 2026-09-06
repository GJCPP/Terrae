#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"

using namespace mcl::bn;

// Verifier-facing artifact for a preprocessed CQ lookup table.
//
// The prover view may additionally carry `table_evals` so witnesses can be
// derived without rescanning the original polynomial commitment.
class cqLookupPublicTable {
public:
    std::string key;
    size_t domain_size = 0;
    std::optional<size_t> degree_bound;
    Group1 commitment;
    Group2 table_g2;
    Fr lagrange_at_zero = Fr(0);
    Group2 vanishing_g2;
    std::shared_ptr<const std::vector<Fr>> table_evals;
    std::shared_ptr<const std::map<Fr, size_t>> index_by_value;
    std::shared_ptr<const std::vector<Group1>> lagrange_commitments;
    std::shared_ptr<const std::vector<Group1>> quotient_commitments;
    std::shared_ptr<const std::vector<Group1>> a0_quotient_commitments;

    cqLookupPublicTable verifier_view() const
    {
        cqLookupPublicTable view = *this;
        view.table_evals.reset();
        view.index_by_value.reset();
        view.quotient_commitments.reset();
        return view;
    }
};

cqLookupPublicTable build_cq_lookup_public_table(const kzgPublicParameters& public_params,
                                                 std::string key,
                                                 const poly& table,
                                                 size_t domain_size,
                                                 std::optional<size_t> degree_bound = std::nullopt);

void fill_cq_lookup_domain_precompute(const kzgPublicParameters& public_params,
                                      size_t domain_size,
                                      Fr& lagrange_at_zero,
                                      Group2& vanishing_g2,
                                      std::shared_ptr<const std::vector<Group1>>& lagrange_commitments,
                                      std::shared_ptr<const std::vector<Group1>>& a0_quotient_commitments);

// Cached table-side preprocessing shared by all CQ lookup checks against the
// same committed table.
class cqLookupTable {
public:
    void setup(kzgProtocol& kzg, const virCommitment& table, size_t domain_size);
    void setup(kzgProtocol& kzg, const cqLookupPublicTable& table, const virCommitment& table_commitment);
    void setup_from_evaluations(kzgProtocol& kzg,
                                const virCommitment& table_commitment,
                                const std::vector<Fr>& table_evaluations);
    void setup_from_evaluations_with_precomputed(
        kzgProtocol& kzg,
        const virCommitment& table_commitment,
        const std::vector<Fr>& table_evaluations,
        const Group2& table_g2,
        std::shared_ptr<const std::vector<Group1>> quotient_commitments,
        std::shared_ptr<const std::map<Fr, size_t>> index_by_value = nullptr,
        std::shared_ptr<const std::vector<Group1>> lagrange_commitments = nullptr,
        std::shared_ptr<const std::vector<Group1>> a0_quotient_commitments = nullptr,
        std::optional<Fr> lagrange_at_zero = std::nullopt,
        std::optional<Group2> vanishing_g2 = std::nullopt);
    void setup_verifier(kzgProtocol& kzg,
                        const virCommitment& table_commitment,
                        size_t domain_size,
                        const Group2& table_g2,
                        std::shared_ptr<const std::vector<Group1>> lagrange_commitments = nullptr,
                        std::shared_ptr<const std::vector<Group1>> a0_quotient_commitments = nullptr,
                        std::optional<Fr> lagrange_at_zero = std::nullopt,
                        std::optional<Group2> vanishing_g2 = std::nullopt);

    size_t domain_size = 0;
    Fr lagrange_at_zero = Fr(0);
    virCommitment table_commitment;
    std::shared_ptr<const std::vector<Fr>> table_evals;
    std::shared_ptr<const std::map<Fr, size_t>> index_by_value;
    std::shared_ptr<const std::vector<Group1>> lagrange_commitments;
    std::shared_ptr<const std::vector<Group1>> quotient_commitments;
    std::shared_ptr<const std::vector<Group1>> a0_quotient_commitments;
    Group2 table_g2;
    Group2 vanishing_g2;
};

class cqLookupCheckRecord {
public:
    virCommitment query_commitment;
    size_t query_domain_size = 0;
    Fr beta = Fr(0);
    Fr gamma = Fr(0);
    bool support_valid = true;
    bool denominators_valid = true;
    bool quotient_valid = true;
    Group1 multiplicity_commitment;
    Group1 a_commitment;
    Group1 qa_commitment;
    Group1 a0_quotient_commitment;
    Fr a0 = Fr(0);
    kzgCommitment b0_commitment;
    kzgCommitment qb_commitment;
};

class cqLookupSparseWitness {
public:
    Group1 multiplicity_commitment;
    std::vector<std::pair<size_t, Fr>> multiplicity_terms;
    std::vector<Fr> table_evals_at_terms;
    std::vector<Group1> quotient_commitments_at_terms;
    bool support_valid = true;
};

class cqLookupParam {
public:
    kzgProtocol* kzg = nullptr;
    cqLookupTable* table = nullptr;
};

class cqLookupVerifier {
public:
    Fr beta_challenge();
    Fr gamma_challenge();
    void add_check(const cqLookupCheckRecord& record);
    bool finalize() const;

    std::vector<cqLookupCheckRecord> checks;
    const cqLookupParam* params = nullptr;
};

class cqLookupProver {
public:
    Group1 multiplicity_commitment(const virCommitment& query, size_t n, bool* support_valid = nullptr);
    cqLookupCheckRecord add_check(const virCommitment& query, size_t n, const Fr& beta);

    std::vector<cqLookupCheckRecord> checks;
    const cqLookupParam* params = nullptr;
};

// CQ lookup reduces one query vector against one distinct-value public table.
//
// Compared with `lookupProtocol`, this path moves most table-side work into a
// reusable preprocessing object and keeps only the beta/gamma-specific witness
// commitments in the proof.
class cqLookupProtocol : protected ProtocolLifecycle {
public:
    ~cqLookupProtocol() noexcept
    {
        auto_finalize("cqLookupProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, cqLookupTable& table)
    {
        params.kzg = &kzg;
        params.table = &table;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    // Register one lookup statement over a query domain of size `n`.
    void add_check(const virCommitment& query, size_t n);
    void add_check(const virCommitment& query, size_t n, const Group1& multiplicity_commitment);
    void add_check(const virCommitment& query, size_t n, const cqLookupSparseWitness& witness);

    bool needs_opening() const
    {
        return !pending_checks_.empty() || opening_prepared_;
    }

    bool prepare_opening();
    bool complete_opening(const Fr& gamma);
    bool finalize();

protected:
    class PendingCheck {
    public:
        virCommitment query_commitment;
        size_t query_domain_size = 0;
        Group1 multiplicity_commitment = Group1::zero();
        std::shared_ptr<const std::vector<Fr>> cached_query_evals;
        std::shared_ptr<const poly> cached_query_poly;
        std::shared_ptr<const std::vector<std::pair<size_t, Fr>>> cached_multiplicity_terms;
        std::shared_ptr<const std::vector<Fr>> cached_table_evals_at_terms;
        std::shared_ptr<const std::vector<Group1>> cached_quotient_commitments_at_terms;
        bool cached_support_valid = true;
    };

    cqLookupParam params;
    cqLookupProver prover;
    cqLookupVerifier verifier;
    std::vector<PendingCheck> pending_checks_;
    bool opening_prepared_ = false;
    bool opening_completed_ = false;
};
