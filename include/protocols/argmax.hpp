#pragma once

#include "column_replication.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "matrix_sum.hpp"
#include "protocol_lifecycle.hpp"
#include "selector_check.hpp"

using namespace mcl::bn;

class argmaxCheckRecord {
public:
    size_t row_count = 0;
    size_t column_count = 0;
    size_t bit_bound = 0;
    kzgCommitment lifted_max_commitment;
};

class argmaxParam {
public:
    kzgProtocol* kzg = nullptr;
    selectorCheckProtocol* selector_check = nullptr;
    matrixRowSumProtocol* row_sum = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

class argmaxVerifier {
public:
    void add_check(const virCommitment& matrix,
                   const virCommitment& max_values,
                   const virCommitment& selector,
                   const argmaxCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> matrices;
    std::vector<virCommitment> max_vectors;
    std::vector<virCommitment> selectors;
    std::vector<argmaxCheckRecord> checks;
    const argmaxParam* params = nullptr;
};

class argmaxProver {
public:
    argmaxCheckRecord add_check(const virCommitment& max_values,
                                size_t row_count,
                                size_t column_count,
                                size_t bit_bound);

    std::vector<argmaxCheckRecord> checks;
    const argmaxParam* params = nullptr;
};

// Proves that max_values[i] is the row-wise maximum selected by a one-hot selector.
//
// Statement:
// - selector is one-hot in every row
// - max_values = row_sum(selector o matrix)
// - max_values replicated across columns dominates matrix entry-wise
//
// Example:
//   matrix =
//     [[1, 5, 2, 4],
//      [3, 2, 7, 6]]
//
//   selector =
//     [[0, 1, 0, 0],
//      [0, 0, 1, 0]]
//
//   max_values = [5, 7]
//
// The protocol proves:
// - the selector chooses exactly one entry per row
// - the chosen entries are 5 and 7
// - 5 dominates every entry in row 0, and 7 dominates every entry in row 1
class argmaxProtocol : protected ProtocolLifecycle {
public:
    ~argmaxProtocol() noexcept
    {
        auto_finalize("argmaxProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               selectorCheckProtocol& selector_check,
               matrixRowSumProtocol& row_sum,
               columnReplicationProtocol& column_replication,
               lessCheckProtocol& less_check)
    {
        params_.kzg = &kzg;
        params_.selector_check = &selector_check;
        params_.row_sum = &row_sum;
        params_.column_replication = &column_replication;
        params_.less_check = &less_check;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    // bit_bound is the unsigned bound used for the domination gaps:
    //   replicated_max - matrix in [0, 2^bit_bound)
    //
    // In other words, every entry-wise gap must be a non-negative bounded integer.
    void add_check(const virCommitment& matrix,
                   const virCommitment& max_values,
                   const virCommitment& selector,
                   size_t row_count,
                   size_t column_count,
                   size_t bit_bound);

    bool finalize();

private:
    argmaxParam params_;
    argmaxProver prover_;
    argmaxVerifier verifier_;
};
