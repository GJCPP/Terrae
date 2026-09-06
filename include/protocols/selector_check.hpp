#pragma once

#include "kzg.hpp"
#include "matrix_sum.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class selectorCheckParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    matrixRowSumProtocol* row_sum = nullptr;
    size_t check_count = 0;
};

// Proves boolean / one-hot selector properties over row-major matrices.
//
// The intended input shape is:
// - boolean check: any vector over a power-of-two domain
// - one-hot checks: a row-major matrix flattened onto row_count * column_count points
//
// Example one-hot matrix:
//   [[0, 1, 0],
//    [1, 0, 0]]
//
// Example masked one-hot matrix:
//   selector = [[0, 1, 0],
//               [0, 0, 0]]
//   mask     = [1, 0]
//
// The second row is disabled by the mask, so its row sum is allowed to be 0.
class selectorCheckProtocol : protected ProtocolLifecycle {
public:
    ~selectorCheckProtocol() noexcept
    {
        auto_finalize("selectorCheckProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero, matrixRowSumProtocol& row_sum)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.row_sum = &row_sum;
        mark_setup();
    }

    // Prove every entry is in {0, 1}.
    //
    // Algebraically this is the usual identity
    //   b * (b - 1) = 0
    // enforced over the whole domain.
    void add_boolean_check(const virCommitment& values, size_t domain_size);

    // Prove a row-major matrix is one-hot in every row:
    // - each entry is boolean
    // - each row sums to exactly 1
    void add_one_hot_rows_check(const virCommitment& selector, size_t row_count, size_t column_count);

    // Prove a row-major matrix is masked one-hot:
    // - selector entries are boolean
    // - mask entries are boolean
    // - row_sum(selector)[i] = row_mask[i]
    //
    // So mask[i] = 1 means "this row must be one-hot",
    // while mask[i] = 0 means "this row must be all zero".
    void add_masked_one_hot_rows_check(const virCommitment& selector,
                                       const virCommitment& row_mask,
                                       size_t row_count,
                                       size_t column_count);

    bool finalize();

private:
    selectorCheckParam params_;
};
