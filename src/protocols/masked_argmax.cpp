#include "masked_argmax.hpp"

#include "matrix_utils.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

bool masked_argmax_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_MASKED_ARGMAX");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void masked_argmax_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = masked_argmax_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "masked_argmax_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("masked_argmax_profile", label, "", elapsed.count(), "ok");
}

void validate_masked_argmax_parameters(size_t row_count, size_t column_count, size_t bit_bound)
{
    validate_matrix_shape(row_count, column_count, "maskedArgmaxProtocol");
    if (bit_bound > static_cast<size_t>(Fr::getBitSize() - 1)) {
        throw std::runtime_error("maskedArgmaxProtocol: bit bound exceeds the safely representable integer range of Fr");
    }
}

} // namespace

void maskedArgmaxVerifier::add_check(const virCommitment& matrix,
                                     const virCommitment& max_values,
                                     const virCommitment& selector,
                                     const virCommitment& candidate_mask,
                                     const virCommitment& row_mask,
                                     const maskedArgmaxCheckRecord& record)
{
    matrices.push_back(matrix);
    max_vectors.push_back(max_values);
    selectors.push_back(selector);
    candidate_masks.push_back(candidate_mask);
    row_masks.push_back(row_mask);
    checks.push_back(record);
}

bool maskedArgmaxVerifier::finalize() const
{
    return true;
}

maskedArgmaxCheckRecord maskedArgmaxProver::add_check(const virCommitment& max_values,
                                                      const virCommitment& row_mask,
                                                      size_t row_count,
                                                      size_t column_count,
                                                      size_t bit_bound)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("maskedArgmaxProver not setup");
    }

    maskedArgmaxCheckRecord record;
    record.row_count = row_count;
    record.column_count = column_count;
    record.bit_bound = bit_bound;

    // Lift both row-wise vectors back to matrix shape so later constraints can
    // stay purely entry-wise.
    const poly lifted_max_poly =
        params->kzg->materialize(max_values)->column_replicated(row_count, column_count);
    record.lifted_max_commitment = params->kzg->commit(lifted_max_poly, row_count * column_count - 1);

    const poly lifted_row_mask_poly =
        params->kzg->materialize(row_mask)->column_replicated(row_count, column_count);
    record.lifted_row_mask_commitment = params->kzg->commit(lifted_row_mask_poly, row_count * column_count - 1);

    checks.push_back(record);
    return record;
}

maskedArgmaxCheckRecord maskedArgmaxProtocol::add_check(const virCommitment& matrix,
                                     const virCommitment& max_values,
                                     const virCommitment& selector,
                                     const virCommitment& candidate_mask,
                                     const virCommitment& row_mask,
                                     size_t row_count,
                                                        size_t column_count,
                                                        size_t bit_bound)
{
    proof_cost::record_relation("argmax", row_count * column_count);
    if (!is_setup) {
        throw std::runtime_error("maskedArgmaxProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.selector_check == nullptr ||
        params_.row_sum == nullptr || params_.column_replication == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("maskedArgmaxProtocol dependencies not setup");
    }

    validate_masked_argmax_parameters(row_count, column_count, bit_bound);

    maskedArgmaxCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.row_count = row_count;
        record.column_count = column_count;
        record.bit_bound = bit_bound;
        const Group1 lifted_max_group = proof_session->transport_group1({});
        record.lifted_max_commitment = params_.kzg->adopt_commitment(lifted_max_group);
        params_.kzg->prove_degree(record.lifted_max_commitment, row_count * column_count - 1);

        const Group1 lifted_row_mask_group = proof_session->transport_group1({});
        record.lifted_row_mask_commitment = params_.kzg->adopt_commitment(lifted_row_mask_group);
        params_.kzg->prove_degree(record.lifted_row_mask_commitment, row_count * column_count - 1);
    } else {
        record = prover_.add_check(max_values, row_mask, row_count, column_count, bit_bound);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 lifted_max_group =
                proof_session->transport_group1([&] { return record.lifted_max_commitment.com; });
            if (lifted_max_group != record.lifted_max_commitment.com) {
                throw std::runtime_error("maskedArgmaxProtocol: transported lifted max commitment mismatch");
            }

            const Group1 lifted_row_mask_group =
                proof_session->transport_group1([&] { return record.lifted_row_mask_commitment.com; });
            if (lifted_row_mask_group != record.lifted_row_mask_commitment.com) {
                throw std::runtime_error("maskedArgmaxProtocol: transported lifted row-mask commitment mismatch");
            }
        }
    }
    verifier_.add_check(matrix, max_values, selector, candidate_mask, row_mask, record);

    const size_t domain_size = row_count * column_count;

    // Step 1: active rows are one-hot, inactive rows are all zero.
    masked_argmax_profile_scope("selector_check.masked_one_hot_rows", [&] {
        params_.selector_check->add_masked_one_hot_rows_check(selector, row_mask, row_count, column_count);
    });
    // Step 2: candidate eligibility is itself boolean.
    masked_argmax_profile_scope("selector_check.boolean_candidate_mask", [&] {
        params_.selector_check->add_boolean_check(candidate_mask, domain_size);
    });

    const virCommitment lifted_row_mask(record.lifted_row_mask_commitment);
    masked_argmax_profile_scope("column_replication.row_mask", [&] {
        params_.column_replication->add_check(row_mask, lifted_row_mask, row_count, column_count);
    });

    // Step 3: candidate masks are either the lifted row mask or zero entry-wise.
    virCommitment candidate_row_zero = candidate_mask * (candidate_mask - lifted_row_mask);
    masked_argmax_profile_scope("zero.candidate_row", [&] {
        params_.zero_check->add_check(candidate_row_zero, domain_size);
    });

    // Step 4: the selector may only choose eligible candidates.
    virCommitment selector_mask_zero = selector * (virCommitment(Fr(1)) - candidate_mask);
    masked_argmax_profile_scope("zero.selector_mask", [&] {
        params_.zero_check->add_check(selector_mask_zero, domain_size);
    });

    // Step 5: row-summing the selected entries yields the claimed maxima.
    masked_argmax_profile_scope("row_sum.selected_values", [&] {
        params_.row_sum->add_check(selector * matrix, max_values, row_count, column_count);
    });

    const virCommitment lifted_max(record.lifted_max_commitment);
    masked_argmax_profile_scope("column_replication.max_values", [&] {
        params_.column_replication->add_check(max_values, lifted_max, row_count, column_count);
    });

    // Step 6: on eligible entries, the selected maximum dominates every
    // candidate in the row.
    masked_argmax_profile_scope("less.masked_dominance", [&] {
        params_.less_check->add_masked_check(lifted_max - matrix, candidate_mask, bit_bound, domain_size);
    });
    return record;
}

bool maskedArgmaxProtocol::finalize()
{
    begin_finalize("maskedArgmaxProtocol");
    return verifier_.finalize();
}
