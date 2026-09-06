#include "argmax.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

void validate_argmax_parameters(size_t row_count, size_t column_count, size_t bit_bound)
{
    validate_matrix_shape(row_count, column_count, "argmaxProtocol");
    if (bit_bound > static_cast<size_t>(Fr::getBitSize() - 1)) {
        throw std::runtime_error("argmaxProtocol: bit bound exceeds the safely representable integer range of Fr");
    }
}

} // namespace

void argmaxVerifier::add_check(const virCommitment& matrix,
                               const virCommitment& max_values,
                               const virCommitment& selector,
                               const argmaxCheckRecord& record)
{
    matrices.push_back(matrix);
    max_vectors.push_back(max_values);
    selectors.push_back(selector);
    checks.push_back(record);
}

bool argmaxVerifier::finalize() const
{
    return true;
}

argmaxCheckRecord argmaxProver::add_check(const virCommitment& max_values,
                                          size_t row_count,
                                          size_t column_count,
                                          size_t bit_bound)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("argmaxProver not setup");
    }

    argmaxCheckRecord record;
    record.row_count = row_count;
    record.column_count = column_count;
    record.bit_bound = bit_bound;

    // Build the row-wise max vector lifted back to matrix shape.
    //
    // Example:
    //   max_values = [5, 7]
    //   row_count = 2, column_count = 4
    //
    // produces
    //   lifted_max =
    //     [[5, 5, 5, 5],
    //      [7, 7, 7, 7]]
    //
    // This lets us compare the claimed row maximum against every entry
    // in the corresponding row using one element-wise range check.
    const poly lifted_max_poly =
        params->kzg->materialize(max_values)->column_replicated(row_count, column_count);
    record.lifted_max_commitment = params->kzg->commit(lifted_max_poly, row_count * column_count - 1);
    checks.push_back(record);
    return record;
}

void argmaxProtocol::add_check(const virCommitment& matrix,
                               const virCommitment& max_values,
                               const virCommitment& selector,
                               size_t row_count,
                               size_t column_count,
                               size_t bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("argmaxProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.selector_check == nullptr || params_.row_sum == nullptr ||
        params_.column_replication == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("argmaxProtocol dependencies not setup");
    }

    validate_argmax_parameters(row_count, column_count, bit_bound);
    proof_cost::record_relation("argmax", row_count * column_count);

    argmaxCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        // On the verifier side we only receive the lifted-max commitment from the proof.
        record.row_count = row_count;
        record.column_count = column_count;
        record.bit_bound = bit_bound;
        const Group1 lifted_max_group = proof_session->transport_group1({});
        record.lifted_max_commitment = params_.kzg->adopt_commitment(lifted_max_group);
        params_.kzg->prove_degree(record.lifted_max_commitment, row_count * column_count - 1);
    } else {
        // On the prover side we materialize and commit the lifted-max witness.
        record = prover_.add_check(max_values, row_count, column_count, bit_bound);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 lifted_max_group =
                proof_session->transport_group1([&] { return record.lifted_max_commitment.com; });
            if (lifted_max_group != record.lifted_max_commitment.com) {
                throw std::runtime_error("argmaxProtocol: transported lifted max commitment mismatch");
            }
        }
    }
    verifier_.add_check(matrix, max_values, selector, record);

    // Step 1: selector must be one-hot in every row.
    params_.selector_check->add_one_hot_rows_check(selector, row_count, column_count);

    // Step 2: the selected entries must add up to the claimed row maxima.
    //
    // Since each row of selector is one-hot, selector * matrix keeps exactly
    // one entry from each row and zeroes the others out. Row-summing then
    // extracts the selected value.
    //
    // Example:
    //   row      = [1, 5, 2, 4]
    //   selector = [0, 1, 0, 0]
    //   product  = [0, 5, 0, 0]
    //   row sum  = 5
    params_.row_sum->add_check(selector * matrix, max_values, row_count, column_count);

    const virCommitment lifted_max(record.lifted_max_commitment);

    // Step 3: prove lifted_max really is max_values replicated across columns.
    params_.column_replication->add_check(max_values, lifted_max, row_count, column_count);

    // Step 4: prove the replicated maximum dominates every candidate entry.
    //
    // The less-check sees the entry-wise gaps:
    //   lifted_max - matrix
    //
    // If every gap is in [0, 2^bit_bound), then every row maximum is at least
    // as large as every entry in that row, and all gaps are properly bounded.
    params_.less_check->add_check(lifted_max - matrix, bit_bound, row_count * column_count);
}

bool argmaxProtocol::finalize()
{
    begin_finalize("argmaxProtocol");
    return verifier_.finalize();
}
