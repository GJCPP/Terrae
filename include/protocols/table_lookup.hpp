#pragma once

#include <optional>
#include <string>
#include <memory>
#include <vector>

#include "cq_lookup.hpp"
#include "lookup.hpp"
#include "protocol_lifecycle.hpp"
#include "profiling/soundness_counter.hpp"
#include "sum_check.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

// Optional prover witness: for each query row, point to the matching table row.
class tableLookupWitness {
public:
    static tableLookupWitness from_positions(size_t table_row_count, const std::vector<size_t>& row_positions);

    std::vector<Fr> row_labels;
};

class tableLookupCheckRecord {
public:
    size_t query_row_count = 0;
    size_t table_row_count = 0;
    size_t column_count = 0;
    Fr rho = Fr(0);
    kzgCommitment row_index_commitment;
};

class tableLookupParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    sumProtocol* sum_check = nullptr;
    terraeSoundnessCounter* soundness_counter = nullptr;
};

class tableLookupVerifier {
public:
    Fr challenge();
    void add_check(const tableLookupCheckRecord& record);
    bool finalize() const;

    std::vector<tableLookupCheckRecord> checks;
    const tableLookupParam* params = nullptr;
};

class tableLookupProver {
public:
    tableLookupCheckRecord add_check(const virCommitment& query,
                                     const virCommitment& table,
                                     const Fr& rho,
                                     size_t query_row_count,
                                     size_t table_row_count,
                                     size_t column_count);

    tableLookupCheckRecord add_check(const tableLookupWitness& witness,
                                     const Fr& rho,
                                     size_t query_row_count,
                                     size_t table_row_count,
                                     size_t column_count);

    std::vector<tableLookupCheckRecord> checks;
    const tableLookupParam* params = nullptr;
};

// Matrix lookup reduces row membership to one scalar lookup by attaching two
// labels to every cell:
// - the column label,
// - and the matched table-row label.
//
// The random linear combination with challenge `rho` turns row equality into a
// scalar lookup while a zero-check enforces that the row label is constant
// across each query row.
class tableLookupProtocol : protected ProtocolLifecycle {
public:
    explicit tableLookupProtocol(lookupProtocol& reduced_lookup)
        : reduced_lookup(&reduced_lookup)
    {
    }

    ~tableLookupProtocol() noexcept
    {
        auto_finalize("tableLookupProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero, sumProtocol& sum)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        params.sum_check = &sum;

        prover = tableLookupProver();
        verifier = tableLookupVerifier();

        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    void set_soundness_counter(terraeSoundnessCounter* counter) { params.soundness_counter = counter; }

    void add_check(const virCommitment& query,
                   const virCommitment& table,
                   size_t query_rows,
                   size_t table_rows,
                   size_t columns);

    void add_check(const virCommitment& query,
                   const virCommitment& table,
                   size_t query_rows,
                   size_t table_rows,
                   size_t columns,
                   const tableLookupWitness& witness);

    void add_check(const virCommitment& query,
                   const virCommitment& table,
                   size_t query_rows,
                   size_t table_rows,
                   size_t columns,
                   const tableLookupWitness& witness,
                   tableLookupCheckRecord& out_record);

    bool finalize();

protected:
    void add_check_with_optional_witness(const virCommitment& query,
                                         const virCommitment& table,
                                         size_t query_rows,
                                         size_t table_rows,
                                         size_t columns,
                                         const tableLookupWitness* witness,
                                         tableLookupCheckRecord* out_record);

    tableLookupParam params;
    tableLookupProver prover;
    tableLookupVerifier verifier;
    lookupProtocol* reduced_lookup = nullptr;
};

class cqTableLookupTable {
public:
    void setup(kzgProtocol& kzg, const virCommitment& table, size_t row_count, size_t column_count);
    void setup(const class cqTableLookupPublicTable& table, const virCommitment& table_commitment);

    std::string public_key;
    size_t row_count = 0;
    size_t column_count = 0;
    size_t domain_size = 0;
    virCommitment table_commitment;
    Fr lagrange_at_zero = Fr(0);
    Group2 vanishing_g2;
    Group2 table_g2;
    std::shared_ptr<const std::vector<Fr>> table_evals;
    std::shared_ptr<const std::vector<Group1>> lagrange_commitments;
    std::shared_ptr<const std::vector<Group1>> table_quotient_commitments;
    std::shared_ptr<const std::vector<Group1>> a0_quotient_commitments;
    std::shared_ptr<const std::vector<Fr>> row_label_evals;
    std::shared_ptr<const std::vector<Fr>> column_label_evals;
    poly row_label_poly;
    poly column_label_poly;
    Group2 row_label_g2;
    Group2 column_label_g2;
    Group1 row_label_g1;
    std::shared_ptr<const std::vector<Group1>> row_label_quotient_commitments;
    std::shared_ptr<const std::vector<Group1>> column_label_quotient_commitments;
};

class cqTableLookupPublicTable {
public:
    std::string key;
    size_t row_count = 0;
    size_t column_count = 0;
    size_t domain_size = 0;
    std::optional<size_t> degree_bound;
    Group1 commitment;
    Group2 table_g2;
    Fr lagrange_at_zero = Fr(0);
    Group2 vanishing_g2;
    std::shared_ptr<const std::vector<Fr>> table_evals;
    std::shared_ptr<const std::vector<Group1>> lagrange_commitments;
    std::shared_ptr<const std::vector<Group1>> table_quotient_commitments;
    std::shared_ptr<const std::vector<Group1>> a0_quotient_commitments;
    Group2 row_label_g2;
    Group2 column_label_g2;
    Group1 row_label_g1;
    std::shared_ptr<const std::vector<Group1>> row_label_quotient_commitments;
    std::shared_ptr<const std::vector<Group1>> column_label_quotient_commitments;

    cqTableLookupPublicTable verifier_view() const
    {
        cqTableLookupPublicTable view = *this;
        view.table_evals.reset();
        view.table_quotient_commitments.reset();
        view.row_label_quotient_commitments.reset();
        view.column_label_quotient_commitments.reset();
        return view;
    }
};

cqTableLookupPublicTable build_cq_table_lookup_public_table(const kzgPublicParameters& public_params,
                                                            std::string key,
                                                            const poly& table,
                                                            size_t row_count,
                                                            size_t column_count,
                                                            std::optional<size_t> degree_bound = std::nullopt);

void preprocess_table_lookup_row_labels(kzgProtocol& kzg, size_t row_count, size_t column_count);

class cqTableLookupCheckRecord {
public:
    size_t query_row_count = 0;
    size_t column_count = 0;
    Fr rho = Fr(0);
    kzgCommitment row_index_commitment;
};

class cqTableLookupParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    cqTableLookupTable* table = nullptr;
};

class cqTableLookupVerifier {
public:
    Fr challenge();
    void add_check(const cqTableLookupCheckRecord& record);
    bool finalize() const;

    std::vector<cqTableLookupCheckRecord> checks;
    const cqTableLookupParam* params = nullptr;
    Fr sampled_rho = Fr(0);
    bool has_sampled_rho = false;
};

class cqTableLookupProver {
public:
    cqTableLookupCheckRecord add_check(const virCommitment& query,
                                       size_t query_row_count,
                                       const Fr& rho);

    cqTableLookupCheckRecord add_check(const tableLookupWitness& witness,
                                       size_t query_row_count,
                                       const Fr& rho);

    std::vector<cqTableLookupCheckRecord> checks;
    const cqTableLookupParam* params = nullptr;
};

// CQ-preprocessed version of matrix lookup.
//
// The row/column-label reduction is the same as `tableLookupProtocol`, but the
// final scalar lookup is discharged by `cqLookupProtocol` against a reusable
// preprocessed labeled table.
class cqTableLookupProtocol : protected ProtocolLifecycle {
public:
    explicit cqTableLookupProtocol(cqLookupProtocol& reduced_lookup)
        : reduced_lookup(&reduced_lookup)
    {
    }

    ~cqTableLookupProtocol() noexcept
    {
        auto_finalize("cqTableLookupProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero, cqTableLookupTable& table)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        params.table = &table;

        prover = cqTableLookupProver();
        verifier = cqTableLookupVerifier();
        reduced_table = cqLookupTable();

        prover.params = &params;
        verifier.params = &params;
        reduced_lookup_ready = false;
        mark_setup();
    }

    void add_check(const virCommitment& query, size_t query_rows);
    void add_check(const virCommitment& query, size_t query_rows, cqTableLookupCheckRecord& out_record);
    void add_check(const virCommitment& query, size_t query_rows, const tableLookupWitness& witness);
    void add_check(const virCommitment& query,
                   size_t query_rows,
                   const tableLookupWitness& witness,
                   cqTableLookupCheckRecord& out_record);

    bool finalize();

protected:
    void add_check_with_optional_witness(const virCommitment& query,
                                         size_t query_rows,
                                         const tableLookupWitness* witness,
                                         cqTableLookupCheckRecord* out_record);
    void ensure_reduced_lookup_ready(const Fr& rho, bool needs_dense_table);

    cqTableLookupParam params;
    cqTableLookupProver prover;
    cqTableLookupVerifier verifier;
    bool reduced_lookup_ready = false;
    cqLookupTable reduced_table;
    cqLookupProtocol* reduced_lookup = nullptr;
};
