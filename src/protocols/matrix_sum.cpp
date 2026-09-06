#include "matrix_sum.hpp"

#include "algebra_utils.hpp"
#include "matrix_utils.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <map>
#include <vector>

namespace {

bool matrix_sum_profile_enabled()
{
    const char* flag = terrae_getenv("TERRAE_PROFILE_MATRIX_SUM");
    return flag != nullptr && flag[0] != '\0' && std::string(flag) != "0" && std::string(flag) != "false";
}

template <class Fn>
void matrix_sum_profile_scope(const char* label, Fn&& fn)
{
    if (!matrix_sum_profile_enabled()) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cerr << "matrix_sum_profile " << label << " " << elapsed.count() << "\n";
}

// Materialize the matrix on its natural row-major evaluation domain.
std::vector<Fr> matrix_evaluations(const poly& matrix, size_t row_count, size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "matrix_evaluations");
    return matrix.evaluations_on_domain(row_count * column_count);
}

// Shift that maps a row-prefix witness to its last column.
Fr last_column_shift(size_t row_count, size_t column_count)
{
    return pow_fr(get_root_of_unity(row_count * column_count), column_count - 1);
}

// Enforce column-wise prefix sums.
//
// Example for a 3x2 matrix:
//   matrix     = [[a, b], [c, d], [e, f]]
//   prefix_sum = [[a, b], [a + c, b + d], [a + c + e, b + d + f]]
//
// The first-row selector enforces prefix_sum[0][j] = matrix[0][j], while the
// non-first rows enforce prefix_sum[i][j] - prefix_sum[i - 1][j] = matrix[i][j].
void add_column_prefix_sum_constraints(zeroProtocol& zero,
                                       const virCommitment& matrix,
                                       const virCommitment& prefix_sum,
                                       const virCommitment& first_row_selector,
                                       size_t row_count,
                                       size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "matrixColumnPrefixSumProtocol");

    const size_t domain_size = row_count * column_count;
    const Fr previous_row_shift = inverse_root_of_unity(row_count);
    const virCommitment non_first_row_selector = virCommitment(Fr(1)) - first_row_selector;

    virCommitment boundary_zero = (prefix_sum - matrix) * first_row_selector;
    virCommitment step_zero = (prefix_sum - x_scale(prefix_sum, previous_row_shift) - matrix) * non_first_row_selector;

    zero.add_check(boundary_zero, domain_size);
    zero.add_check(step_zero, domain_size);
}

// Enforce the column-sum reduction from the design notes:
//   S(x) = (1 / row_count) * s(x^row_count)
//   B(w_row X) - B(X) = A(w_row X) - S(w_row X)
//
// Here B is a cyclic column-prefix witness for A - S. The telescoping sum of the
// cyclic difference over each column forces the total residual sum to vanish,
// which is equivalent to column_sum being the true sum along that column.
void add_column_sum_constraints(zeroProtocol& zero,
                                const virCommitment& matrix,
                                const virCommitment& column_sum,
                                const virCommitment& prefix_sum,
                                size_t row_count,
                                size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "matrixColumnSumProtocol");

    const size_t domain_size = row_count * column_count;
    const Fr previous_row_shift = inverse_root_of_unity(row_count);
    const virCommitment repeated_average =
        scale_by_constant(x_pow(column_sum, row_count), inverse_field_from_size_t(row_count));
    virCommitment cyclic_step_zero =
        prefix_sum - x_scale(prefix_sum, previous_row_shift) - (matrix - repeated_average);
    zero.add_check(cyclic_step_zero, domain_size);
}

// Enforce row-wise prefix sums.
//
// Example for a 2x3 matrix:
//   matrix     = [[a, b, c], [d, e, f]]
//   prefix_sum = [[a, a + b, a + b + c], [d, d + e, d + e + f]]
//
// The domain of size row_count isolates the first column of every row, and the
// vanishing polynomial on that domain masks out those boundary cells in the
// recursive step constraint.
void add_row_prefix_sum_constraints(zeroProtocol& zero,
                                    const virCommitment& matrix,
                                    const virCommitment& prefix_sum,
                                    size_t row_count,
                                    size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "matrixRowPrefixSumProtocol");

    const size_t domain_size = row_count * column_count;
    const Fr previous_column_shift = inverse_root_of_unity(domain_size);

    virCommitment boundary_zero = prefix_sum - matrix;
    virCommitment step_zero =
        (prefix_sum - x_scale(prefix_sum, previous_column_shift) - matrix) * virCommitment(vanishing_poly(row_count));

    zero.add_check(boundary_zero, row_count);
    zero.add_check(step_zero, domain_size);
}

} // namespace

void preprocess_matrix_column_prefix_selectors(kzgProtocol& kzg, size_t row_count, size_t column_count)
{
    (void)kzg;
    validate_matrix_shape(row_count, column_count, "preprocess_matrix_column_prefix_selectors");
}

void matrixColumnPrefixSumVerifier::add_check(const virCommitment& matrix,
                                              const virCommitment& prefix_sum,
                                              const matrixColumnPrefixSumCheckRecord& record)
{
    matrices.push_back(matrix);
    prefix_sums.push_back(prefix_sum);
    checks.push_back(record);
}

bool matrixColumnPrefixSumVerifier::finalize() const
{
    return true;
}

matrixColumnPrefixSumCheckRecord matrixColumnPrefixSumProver::add_check(const virCommitment&,
                                                                       const virCommitment&,
                                                                       size_t row_count,
                                                                       size_t column_count)
{
    matrixColumnPrefixSumCheckRecord record;
    record.row_count = row_count;
    record.column_count = column_count;
    checks.push_back(record);
    return record;
}

void matrixRowPrefixSumVerifier::add_check(const virCommitment& matrix,
                                           const virCommitment& prefix_sum,
                                           const matrixRowPrefixSumCheckRecord& record)
{
    matrices.push_back(matrix);
    prefix_sums.push_back(prefix_sum);
    checks.push_back(record);
}

bool matrixRowPrefixSumVerifier::finalize() const
{
    return true;
}

matrixRowPrefixSumCheckRecord matrixRowPrefixSumProver::add_check(const virCommitment&,
                                                                  const virCommitment&,
                                                                  size_t row_count,
                                                                  size_t column_count)
{
    matrixRowPrefixSumCheckRecord record;
    record.row_count = row_count;
    record.column_count = column_count;
    checks.push_back(record);
    return record;
}

void matrixColumnSumVerifier::add_check(const virCommitment& matrix,
                                        const virCommitment& column_sum,
                                        const matrixColumnSumCheckRecord& record)
{
    matrices.push_back(matrix);
    column_sums.push_back(column_sum);
    checks.push_back(record);
}

bool matrixColumnSumVerifier::finalize() const
{
    return true;
}

// Generate the column-prefix witness used to reduce column sums to a final-row check.
matrixColumnSumCheckRecord matrixColumnSumProver::add_check(const virCommitment& matrix,
                                                            const virCommitment& column_sum,
                                                            size_t row_count,
                                                            size_t column_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("matrixColumnSumProver not setup");
    }

    matrixColumnSumCheckRecord record;
    record.row_count = row_count;
    record.column_count = column_count;
    const std::vector<Fr> matrix_evals =
        params->kzg->materialize(matrix)->evaluations_on_domain(row_count * column_count);
    const std::vector<Fr> column_sum_evals = params->kzg->materialize(column_sum)->evaluations_on_domain(column_count);
    const Fr inverse_row_count = inverse_field_from_size_t(row_count);

    std::vector<Fr> residual_evals(row_count * column_count, Fr(0));
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t column = 0; column < column_count; ++column) {
            Fr scaled_column_sum;
            Fr::mul(scaled_column_sum, column_sum_evals[column], inverse_row_count);
            Fr::sub(residual_evals[matrix_index(row, column, column_count)],
                    matrix_evals[matrix_index(row, column, column_count)],
                    scaled_column_sum);
        }
    }

    const poly prefix_poly =
        matrix_column_prefix_sum_polynomial(poly_from_evals(residual_evals), row_count, column_count);
    record.prefix_commitment = params->kzg->commit(prefix_poly, row_count * column_count - 1);
    checks.push_back(record);
    return record;
}

void matrixRowSumVerifier::add_check(const virCommitment& matrix,
                                     const virCommitment& row_sum,
                                     const matrixRowSumCheckRecord& record)
{
    matrices.push_back(matrix);
    row_sums.push_back(row_sum);
    checks.push_back(record);
}

bool matrixRowSumVerifier::finalize() const
{
    return true;
}

// Generate the row-prefix witness used to reduce row sums to a last-column check.
matrixRowSumCheckRecord matrixRowSumProver::add_check(const virCommitment&,
                                                      size_t row_count,
                                                      size_t column_count)
{
    matrixRowSumCheckRecord record;
    record.row_count = row_count;
    record.column_count = column_count;
    checks.push_back(record);
    return record;
}

// Compute column sums directly from the row-major evaluation table.
poly matrix_column_sum_polynomial(const poly& matrix, size_t row_count, size_t column_count)
{
    const std::vector<Fr> matrix_evals = matrix_evaluations(matrix, row_count, column_count);
    std::vector<Fr> column_sums(column_count, Fr(0));

    for (size_t row = 0; row < row_count; ++row) {
        for (size_t column = 0; column < column_count; ++column) {
            Fr::add(column_sums[column], column_sums[column], matrix_evals[matrix_index(row, column, column_count)]);
        }
    }

    return poly_from_evals(column_sums);
}

// Compute the explicit column-prefix witness used by matrixColumnPrefixSumProtocol
// and matrixColumnSumProtocol.
poly matrix_column_prefix_sum_polynomial(const poly& matrix, size_t row_count, size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "matrix_column_prefix_sum_polynomial");

    const std::vector<Fr> matrix_evals = matrix.evaluations_on_domain(row_count * column_count);
    std::vector<Fr> prefix(row_count * column_count, Fr(0));

    for (size_t column = 0; column < column_count; ++column) {
        Fr running = Fr(0);
        for (size_t row = 0; row < row_count; ++row) {
            const size_t index = matrix_index(row, column, column_count);
            Fr::add(running, running, matrix_evals[index]);
            prefix[index] = running;
        }
    }

    return poly_from_evals(prefix);
}

// Compute row sums directly from the row-major evaluation table.
poly matrix_row_sum_polynomial(const poly& matrix, size_t row_count, size_t column_count)
{
    const std::vector<Fr> matrix_evals = matrix_evaluations(matrix, row_count, column_count);
    std::vector<Fr> row_sums(row_count, Fr(0));

    for (size_t row = 0; row < row_count; ++row) {
        for (size_t column = 0; column < column_count; ++column) {
            Fr::add(row_sums[row], row_sums[row], matrix_evals[matrix_index(row, column, column_count)]);
        }
    }

    return poly_from_evals(row_sums);
}

// Compute the explicit row-prefix witness used by matrixRowPrefixSumProtocol
// and matrixRowSumProtocol.
poly matrix_row_prefix_sum_polynomial(const poly& matrix, size_t row_count, size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "matrix_row_prefix_sum_polynomial");

    const std::vector<Fr> matrix_evals = matrix.evaluations_on_domain(row_count * column_count);
    std::vector<Fr> prefix(row_count * column_count, Fr(0));

    for (size_t row = 0; row < row_count; ++row) {
        Fr running = Fr(0);
        for (size_t column = 0; column < column_count; ++column) {
            const size_t index = matrix_index(row, column, column_count);
            Fr::add(running, running, matrix_evals[index]);
            prefix[index] = running;
        }
    }

    return poly_from_evals(prefix);
}

// Register one proof that prefix_sum is the column-wise prefix sum of matrix.
void matrixColumnPrefixSumProtocol::add_check(const virCommitment& matrix,
                                              const virCommitment& prefix_sum,
                                              size_t row_count,
                                              size_t column_count)
{
    if (!is_setup) {
        throw std::runtime_error("matrixColumnPrefixSumProtocol not setup");
    }
    if (params.kzg == nullptr || params.zero_check == nullptr || params.column_replication == nullptr) {
        throw std::runtime_error("matrixColumnPrefixSumProtocol dependencies not setup");
    }

    matrixColumnPrefixSumCheckRecord record;
    matrix_sum_profile_scope("column_prefix.prover_record", [&] {
        record = prover.add_check(matrix, prefix_sum, row_count, column_count);
    });
    matrix_sum_profile_scope("column_prefix.verifier_add_check", [&] {
        verifier.add_check(matrix, prefix_sum, record);
    });
    virCommitment first_row_selector;
    matrix_sum_profile_scope("column_prefix.first_row_selector_expression", [&] {
        first_row_selector = matrix_row_selector_expression(row_count, column_count, 0);
    });
    matrix_sum_profile_scope("column_prefix.add_constraints", [&] {
        add_column_prefix_sum_constraints(
            *params.zero_check, matrix, prefix_sum, first_row_selector, row_count, column_count);
    });
    ++params.check_count;
}

bool matrixColumnPrefixSumProtocol::finalize()
{
    begin_finalize("matrixColumnPrefixSumProtocol");
    return verifier.finalize();
}

// Register one proof that prefix_sum is the row-wise prefix sum of matrix.
void matrixRowPrefixSumProtocol::add_check(const virCommitment& matrix,
                                           const virCommitment& prefix_sum,
                                           size_t row_count,
                                           size_t column_count)
{
    if (!is_setup) {
        throw std::runtime_error("matrixRowPrefixSumProtocol not setup");
    }
    if (params.kzg == nullptr || params.zero_check == nullptr) {
        throw std::runtime_error("matrixRowPrefixSumProtocol dependencies not setup");
    }

    const matrixRowPrefixSumCheckRecord record = prover.add_check(matrix, prefix_sum, row_count, column_count);
    verifier.add_check(matrix, prefix_sum, record);
    add_row_prefix_sum_constraints(*params.zero_check, matrix, prefix_sum, row_count, column_count);
    ++params.check_count;
}

bool matrixRowPrefixSumProtocol::finalize()
{
    begin_finalize("matrixRowPrefixSumProtocol");
    return verifier.finalize();
}

// Register one proof that column_sum equals the column-wise total of matrix.
// This reduces to:
// 1. prove a column-prefix witness is correct; and
// 2. prove the final row of that witness equals column_sum.
void matrixColumnSumProtocol::add_check(const virCommitment& matrix,
                                        const virCommitment& column_sum,
                                        size_t row_count,
                                        size_t column_count)
{
    if (!is_setup) {
        throw std::runtime_error("matrixColumnSumProtocol not setup");
    }
    if (params.kzg == nullptr || params.zero_check == nullptr) {
        throw std::runtime_error("matrixColumnSumProtocol dependencies not setup");
    }

    validate_matrix_shape(row_count, column_count, "matrixColumnSumProtocol");

    matrixColumnSumCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.row_count = row_count;
        record.column_count = column_count;
        const Group1 prefix_group = proof_session->transport_group1({});
        record.prefix_commitment = params.kzg->adopt_commitment(prefix_group);
        params.kzg->prove_degree(record.prefix_commitment, row_count * column_count - 1);
    } else {
        record = prover.add_check(matrix, column_sum, row_count, column_count);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 prefix_group = proof_session->transport_group1([&] { return record.prefix_commitment.com; });
            if (prefix_group != record.prefix_commitment.com) {
                throw std::runtime_error("matrixColumnSumProtocol: transported prefix commitment mismatch");
            }
        }
    }
    verifier.add_check(matrix, column_sum, record);
    const virCommitment prefix_sum(record.prefix_commitment);
    add_column_sum_constraints(*params.zero_check, matrix, column_sum, prefix_sum, row_count, column_count);
    ++params.check_count;
}

bool matrixColumnSumProtocol::finalize()
{
    begin_finalize("matrixColumnSumProtocol");
    return verifier.finalize();
}

// Register one proof that row_sum equals the row-wise total of matrix.
// This reduces to:
// 1. prove a row-prefix witness is correct; and
// 2. prove the last column of that witness equals row_sum.
void matrixRowSumProtocol::add_check(const virCommitment& matrix,
                                     const virCommitment& row_sum,
                                     size_t row_count,
                                     size_t column_count)
{
    if (!is_setup) {
        throw std::runtime_error("matrixRowSumProtocol not setup");
    }
    if (params.kzg == nullptr || params.zero_check == nullptr) {
        throw std::runtime_error("matrixRowSumProtocol dependencies not setup");
    }

    validate_matrix_shape(row_count, column_count, "matrixRowSumProtocol");

    const matrixRowSumCheckRecord record = prover.add_check(matrix, row_count, column_count);
    verifier.add_check(matrix, row_sum, record);
}

bool matrixRowSumProtocol::finalize()
{
    begin_finalize("matrixRowSumProtocol");

    std::vector<std::pair<size_t, size_t>> group_order;
    std::map<std::pair<size_t, size_t>, std::vector<size_t>> grouped_check_indices;
    for (size_t i = 0; i < verifier.checks.size(); ++i) {
        const auto key =
            std::make_pair(verifier.checks[i].row_count, verifier.checks[i].column_count);
        if (grouped_check_indices.find(key) == grouped_check_indices.end()) {
            group_order.push_back(key);
        }
        grouped_check_indices[key].push_back(i);
    }

    const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
    for (const auto& key : group_order) {
        const size_t row_count = key.first;
        const size_t column_count = key.second;
        const std::vector<size_t>& group_indices = grouped_check_indices.at(key);

        virCommitment combined_matrix = verifier.matrices[group_indices[0]];
        virCommitment combined_row_sum = verifier.row_sums[group_indices[0]];
        if (group_indices.size() > 1) {
            const std::string label = "matrix_row_sum.batch." + std::to_string(row_count) + "." +
                                      std::to_string(column_count);
            const Fr lambda = params.kzg->transcript().squeeze_nonzero_fr(label);
            Fr lambda_power = lambda;
            for (size_t pos = 1; pos < group_indices.size(); ++pos) {
                combined_matrix = combined_matrix + verifier.matrices[group_indices[pos]] * virCommitment(lambda_power);
                combined_row_sum = combined_row_sum + verifier.row_sums[group_indices[pos]] * virCommitment(lambda_power);
                Fr::mul(lambda_power, lambda_power, lambda);
            }
        }

        kzgCommitment prefix_commitment;
        if (proof_session && proof_session->mode() == ProofMode::Verify) {
            const Group1 prefix_group = proof_session->transport_group1({});
            prefix_commitment = params.kzg->adopt_commitment(prefix_group);
            params.kzg->prove_degree(prefix_commitment, row_count * column_count - 1);
        } else {
            const poly prefix_poly =
                matrix_row_prefix_sum_polynomial(*params.kzg->materialize(combined_matrix), row_count, column_count);
            prefix_commitment = params.kzg->commit(prefix_poly, row_count * column_count - 1);
            if (proof_session && proof_session->mode() == ProofMode::Prove) {
                const Group1 prefix_group = proof_session->transport_group1([&] { return prefix_commitment.com; });
                if (prefix_group != prefix_commitment.com) {
                    throw std::runtime_error("matrixRowSumProtocol: transported prefix commitment mismatch");
                }
            }
        }

        const virCommitment prefix_sum(prefix_commitment);
        add_row_prefix_sum_constraints(*params.zero_check, combined_matrix, prefix_sum, row_count, column_count);
        virCommitment final_column_zero =
            x_scale(prefix_sum, last_column_shift(row_count, column_count)) - combined_row_sum;
        params.zero_check->add_check(final_column_zero, row_count);
        ++params.check_count;
    }

    return verifier.finalize();
}
