#pragma once

#include "column_replication.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "matrix_sum.hpp"
#include "protocol_lifecycle.hpp"
#include "selector_check.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class maskedArgmaxCheckRecord {
public:
    size_t row_count = 0;
    size_t column_count = 0;
    size_t bit_bound = 0;
    kzgCommitment lifted_max_commitment;
    kzgCommitment lifted_row_mask_commitment;
};

class maskedArgmaxParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    selectorCheckProtocol* selector_check = nullptr;
    matrixRowSumProtocol* row_sum = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

class maskedArgmaxVerifier {
public:
    void add_check(const virCommitment& matrix,
                   const virCommitment& max_values,
                   const virCommitment& selector,
                   const virCommitment& candidate_mask,
                   const virCommitment& row_mask,
                   const maskedArgmaxCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> matrices;
    std::vector<virCommitment> max_vectors;
    std::vector<virCommitment> selectors;
    std::vector<virCommitment> candidate_masks;
    std::vector<virCommitment> row_masks;
    std::vector<maskedArgmaxCheckRecord> checks;
    const maskedArgmaxParam* params = nullptr;
};

class maskedArgmaxProver {
public:
    maskedArgmaxCheckRecord add_check(const virCommitment& max_values,
                                      const virCommitment& row_mask,
                                      size_t row_count,
                                      size_t column_count,
                                      size_t bit_bound);

    std::vector<maskedArgmaxCheckRecord> checks;
    const maskedArgmaxParam* params = nullptr;
};

// Proves a masked row-wise argmax.
//
// Compared with `argmaxProtocol`, this version adds:
// - `row_mask`: rows with 1 must pick exactly one candidate; rows with 0 pick none
// - `candidate_mask`: per-entry eligibility mask; only eligible candidates may be selected
//
// Example:
// - row values:      `[7, 2, 9, 5]`
// - candidate_mask:  `[0, 1, 0, 1]`
// - row_mask:        `1`
//
// Then only the 2nd and 4th entries are eligible candidates for that row, so
// the selector must choose one of those two positions and `max_values` must be
// the maximum over the eligible set, i.e. `max(2, 5) = 5`.
//
// On active rows the claimed `max_values` must be the selected maximum.
// On inactive rows the selector is all-zero and `max_values` is forced to 0.
class maskedArgmaxProtocol : protected ProtocolLifecycle {
public:
    ~maskedArgmaxProtocol() noexcept
    {
        auto_finalize("maskedArgmaxProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero_check,
               selectorCheckProtocol& selector_check,
               matrixRowSumProtocol& row_sum,
               columnReplicationProtocol& column_replication,
               lessCheckProtocol& less_check)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero_check;
        params_.selector_check = &selector_check;
        params_.row_sum = &row_sum;
        params_.column_replication = &column_replication;
        params_.less_check = &less_check;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    maskedArgmaxCheckRecord add_check(const virCommitment& matrix,
                                     const virCommitment& max_values,
                                     const virCommitment& selector,
                                     const virCommitment& candidate_mask,
                                     const virCommitment& row_mask,
                                     size_t row_count,
                                     size_t column_count,
                                     size_t bit_bound);

    bool finalize();

private:
    maskedArgmaxParam params_;
    maskedArgmaxProver prover_;
    maskedArgmaxVerifier verifier_;
};
