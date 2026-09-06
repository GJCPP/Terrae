#pragma once

#include "kzg.hpp"
#include "polynomial.hpp"
#include "protocol_lifecycle.hpp"
#include "column_replication.hpp"
#include "zero_check.hpp"

#include <set>

using namespace mcl::bn;

// Column sums of a row-major flattened matrix.
// Example for [[a, b], [c, d]]: returns [a + c, b + d].
poly matrix_column_sum_polynomial(const poly& matrix, size_t row_count, size_t column_count);

// Column-wise prefix sums of a row-major flattened matrix.
// Example for [[a, b], [c, d]]: returns [[a, b], [a + c, b + d]].
poly matrix_column_prefix_sum_polynomial(const poly& matrix, size_t row_count, size_t column_count);

// Row sums of a row-major flattened matrix.
// Example for [[a, b, c], [d, e, f]]: returns [a + b + c, d + e + f].
poly matrix_row_sum_polynomial(const poly& matrix, size_t row_count, size_t column_count);

// Row-wise prefix sums of a row-major flattened matrix.
// Example for [[a, b, c], [d, e, f]]:
// returns [[a, a + b, a + b + c], [d, d + e, d + e + f]].
poly matrix_row_prefix_sum_polynomial(const poly& matrix, size_t row_count, size_t column_count);

// Precompute public selector commitments used by column-prefix checks.
void preprocess_matrix_column_prefix_selectors(kzgProtocol& kzg, size_t row_count, size_t column_count);

// Minimal metadata needed to replay one column-prefix check on the verifier side.
class matrixColumnPrefixSumCheckRecord {
public:
    size_t row_count = 0;
    size_t column_count = 0;
};

// Shared dependencies for the column-prefix protocol.
class matrixColumnPrefixSumParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
    size_t check_count = 0;
};

// Verifier state for batched column-prefix checks.
class matrixColumnPrefixSumVerifier {
public:
    void add_check(const virCommitment& matrix,
                   const virCommitment& prefix_sum,
                   const matrixColumnPrefixSumCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> matrices;
    std::vector<virCommitment> prefix_sums;
    std::vector<matrixColumnPrefixSumCheckRecord> checks;
    const matrixColumnPrefixSumParam* params = nullptr;
};

// Prover state for batched column-prefix checks.
class matrixColumnPrefixSumProver {
public:
    matrixColumnPrefixSumCheckRecord add_check(const virCommitment& matrix,
                                              const virCommitment& prefix_sum,
                                              size_t row_count,
                                              size_t column_count);

    std::vector<matrixColumnPrefixSumCheckRecord> checks;
    const matrixColumnPrefixSumParam* params = nullptr;
};

// Proves that prefix_sum is the column-wise prefix sum of matrix:
// prefix_sum[i][j] = sum_{r <= i} matrix[r][j].
class matrixColumnPrefixSumProtocol : protected ProtocolLifecycle {
public:
    ~matrixColumnPrefixSumProtocol() noexcept
    {
        auto_finalize("matrixColumnPrefixSumProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero, columnReplicationProtocol& column_replication)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        params.column_replication = &column_replication;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    void add_check(const virCommitment& matrix,
                   const virCommitment& prefix_sum,
                   size_t row_count,
                   size_t column_count);

    bool finalize();

protected:
    matrixColumnPrefixSumParam params;
    matrixColumnPrefixSumProver prover;
    matrixColumnPrefixSumVerifier verifier;
};

// Minimal metadata needed to replay one row-prefix check on the verifier side.
class matrixRowPrefixSumCheckRecord {
public:
    size_t row_count = 0;
    size_t column_count = 0;
};

// Shared dependencies for the row-prefix protocol.
class matrixRowPrefixSumParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    size_t check_count = 0;
};

// Verifier state for batched row-prefix checks.
class matrixRowPrefixSumVerifier {
public:
    void add_check(const virCommitment& matrix,
                   const virCommitment& prefix_sum,
                   const matrixRowPrefixSumCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> matrices;
    std::vector<virCommitment> prefix_sums;
    std::vector<matrixRowPrefixSumCheckRecord> checks;
    const matrixRowPrefixSumParam* params = nullptr;
};

// Prover state for batched row-prefix checks.
class matrixRowPrefixSumProver {
public:
    matrixRowPrefixSumCheckRecord add_check(const virCommitment& matrix,
                                           const virCommitment& prefix_sum,
                                           size_t row_count,
                                           size_t column_count);

    std::vector<matrixRowPrefixSumCheckRecord> checks;
    const matrixRowPrefixSumParam* params = nullptr;
};

// Proves that prefix_sum is the row-wise prefix sum of matrix:
// prefix_sum[i][j] = sum_{c <= j} matrix[i][c].
class matrixRowPrefixSumProtocol : protected ProtocolLifecycle {
public:
    ~matrixRowPrefixSumProtocol() noexcept
    {
        auto_finalize("matrixRowPrefixSumProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    void add_check(const virCommitment& matrix,
                   const virCommitment& prefix_sum,
                   size_t row_count,
                   size_t column_count);

    bool finalize();

protected:
    matrixRowPrefixSumParam params;
    matrixRowPrefixSumProver prover;
    matrixRowPrefixSumVerifier verifier;
};

// Metadata for one column-sum check, including the prover-generated prefix witness.
class matrixColumnSumCheckRecord {
public:
    size_t row_count = 0;
    size_t column_count = 0;
    kzgCommitment prefix_commitment;
};

// Shared dependencies for the column-sum protocol.
class matrixColumnSumParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    size_t check_count = 0;
};

// Verifier state for batched column-sum checks.
class matrixColumnSumVerifier {
public:
    void add_check(const virCommitment& matrix,
                   const virCommitment& column_sum,
                   const matrixColumnSumCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> matrices;
    std::vector<virCommitment> column_sums;
    std::vector<matrixColumnSumCheckRecord> checks;
    const matrixColumnSumParam* params = nullptr;
};

// Prover state for batched column-sum checks.
class matrixColumnSumProver {
public:
    matrixColumnSumCheckRecord add_check(const virCommitment& matrix,
                                         const virCommitment& column_sum,
                                         size_t row_count,
                                         size_t column_count);

    std::vector<matrixColumnSumCheckRecord> checks;
    const matrixColumnSumParam* params = nullptr;
};

// Proves that column_sum[j] = sum_i matrix[i][j].
// Internally this reduces the claim to a column-prefix witness and checks
// that the final row of that witness equals the claimed sums.
class matrixColumnSumProtocol : protected ProtocolLifecycle {
public:
    ~matrixColumnSumProtocol() noexcept
    {
        auto_finalize("matrixColumnSumProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    void add_check(const virCommitment& matrix,
                   const virCommitment& column_sum,
                   size_t row_count,
                   size_t column_count);

    bool finalize();

protected:
    matrixColumnSumParam params;
    matrixColumnSumProver prover;
    matrixColumnSumVerifier verifier;
};

// Metadata for one row-sum check, including the prover-generated prefix witness.
class matrixRowSumCheckRecord {
public:
    size_t row_count = 0;
    size_t column_count = 0;
    kzgCommitment prefix_commitment;
};

// Shared dependencies for the row-sum protocol.
class matrixRowSumParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    size_t check_count = 0;
};

// Verifier state for batched row-sum checks.
class matrixRowSumVerifier {
public:
    void add_check(const virCommitment& matrix,
                   const virCommitment& row_sum,
                   const matrixRowSumCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> matrices;
    std::vector<virCommitment> row_sums;
    std::vector<matrixRowSumCheckRecord> checks;
    const matrixRowSumParam* params = nullptr;
};

// Prover state for batched row-sum checks.
class matrixRowSumProver {
public:
    matrixRowSumCheckRecord add_check(const virCommitment& matrix,
                                      size_t row_count,
                                      size_t column_count);

    std::vector<matrixRowSumCheckRecord> checks;
    const matrixRowSumParam* params = nullptr;
};

// Proves that row_sum[i] = sum_j matrix[i][j].
// Internally this reduces the claim to a row-prefix witness and checks
// that the last column of that witness equals the claimed sums.
class matrixRowSumProtocol : protected ProtocolLifecycle {
public:
    ~matrixRowSumProtocol() noexcept
    {
        auto_finalize("matrixRowSumProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        prover.params = &params;
        verifier.params = &params;
        mark_setup();
    }

    void add_check(const virCommitment& matrix,
                   const virCommitment& row_sum,
                   size_t row_count,
                   size_t column_count);

    bool finalize();

protected:
    matrixRowSumParam params;
    matrixRowSumProver prover;
    matrixRowSumVerifier verifier;
};
