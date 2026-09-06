#pragma once

#include <vector>

#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class indexRoutingCheckRecord {
public:
    size_t target_row_count = 0;
    size_t source_row_count = 0;
    size_t column_count = 0;
    size_t channel_count = 0;
    bool uses_mask = false;
    bool enforce_mask_boolean = true;
    std::vector<kzgCommitment> routed_rows_commitments;
    kzgCommitment lifted_mask_commitment;
};

class indexRoutingParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
    concatenatedTableLookupProtocol* concatenated_table_lookup = nullptr;
};

class indexRoutingVerifier {
public:
    void add_check(const indexRoutingCheckRecord& record);
    bool finalize() const;

    std::vector<indexRoutingCheckRecord> checks;
    const indexRoutingParam* params = nullptr;
};

class indexRoutingProver {
public:
    indexRoutingCheckRecord add_check(const virCommitment& row_indices,
                                      const virCommitment* mask,
                                      const std::vector<virCommitment>& tables,
                                      size_t target_row_count,
                                      size_t source_row_count,
                                      size_t column_count);

    std::vector<indexRoutingCheckRecord> checks;
    const indexRoutingParam* params = nullptr;
};

// Proves that an index vector routes rows from a source table into an output matrix.
//
// Unmasked statement:
// - `row_indices[target_row]` is the selected source-row index
// - `output[target_row] = table[row_indices[target_row]]`
//
// Masked statement:
// - active rows route `table[row_indices[target_row]]`
// - inactive rows are forced to all zero in the final output
// - inactive rows are internally routed through dummy row 0 only for the lookup reduction
//
// The reduction is a shared-row lookup:
//   row_indices || routed_rows ⊆ [0, 1, ..., source_row_count - 1] || table
//
// This is the index-vector form of elementary-matrix left multiplication.
class indexRoutingProtocol : protected ProtocolLifecycle {
public:
    ~indexRoutingProtocol() noexcept
    {
        auto_finalize("indexRoutingProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero,
               columnReplicationProtocol& column_replication,
               concatenatedTableLookupProtocol& concatenated_table_lookup)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.column_replication = &column_replication;
        params_.concatenated_table_lookup = &concatenated_table_lookup;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& row_indices,
                   const virCommitment& table,
                   const virCommitment& output,
                   size_t target_row_count,
                   size_t source_row_count,
                   size_t column_count);

    void add_check(const virCommitment& row_indices,
                   const std::vector<virCommitment>& tables,
                   const std::vector<virCommitment>& outputs,
                   size_t target_row_count,
                   size_t source_row_count,
                   size_t column_count);

    void add_masked_check(const virCommitment& row_indices,
                          const virCommitment& mask,
                          const virCommitment& table,
                          const virCommitment& output,
                          size_t target_row_count,
                          size_t source_row_count,
                          size_t column_count);

    void add_masked_check(const virCommitment& row_indices,
                          const virCommitment& mask,
                          const std::vector<virCommitment>& tables,
                          const std::vector<virCommitment>& outputs,
                          size_t target_row_count,
                          size_t source_row_count,
                          size_t column_count);

    // Use only when the exact mask commitment is already constrained to be
    // Boolean by another proof that the caller verifies as part of the same
    // certificate. All standalone entry points above remain self-contained.
    void add_masked_check_with_certified_mask(const virCommitment& row_indices,
                                              const virCommitment& mask,
                                              const std::vector<virCommitment>& tables,
                                              const std::vector<virCommitment>& outputs,
                                              size_t target_row_count,
                                              size_t source_row_count,
                                              size_t column_count);

    bool finalize();

private:
    void add_check_internal(const virCommitment& row_indices,
                            const virCommitment* mask,
                            const std::vector<virCommitment>& tables,
                            const std::vector<virCommitment>& outputs,
                            size_t target_row_count,
                            size_t source_row_count,
                            size_t column_count,
                            bool enforce_mask_boolean = true);

    indexRoutingParam params_;
    indexRoutingProver prover_;
    indexRoutingVerifier verifier_;
    std::vector<virCommitment> row_indices_;
    std::vector<virCommitment> masks_;
    std::vector<std::vector<virCommitment>> table_batches_;
    std::vector<std::vector<virCommitment>> output_batches_;
};
