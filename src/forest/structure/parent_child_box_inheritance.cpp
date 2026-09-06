#include "parent_child_box_inheritance.hpp"

#include "matrix_utils.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace {

void validate_parent_child_box_parameters(size_t node_count, size_t feature_count, size_t bit_bound)
{
    validate_matrix_shape(node_count, feature_count, "parentChildBoxInheritanceProtocol");
    if (bit_bound == 0 || bit_bound > static_cast<size_t>(Fr::getBitSize() - 1)) {
        throw std::runtime_error(
            "parentChildBoxInheritanceProtocol: bit_bound must be non-zero and fit in the safely represented Fr range");
    }
}

kzgCommitment adopt_transported_commitment(kzgProtocol& kzg,
                                           const std::shared_ptr<ProofSession>& proof_session,
                                           size_t degree_bound)
{
    const Group1 commitment_group = proof_session->transport_group1({});
    const kzgCommitment commitment = kzg.adopt_commitment(commitment_group);
    kzg.prove_degree(commitment, degree_bound);
    return commitment;
}

bool try_size_t_from_fr(const Fr& value, size_t& out)
{
    const std::string repr = value.getStr(10);
    if (!repr.empty() && repr.front() == '-') {
        return false;
    }

    size_t parsed = 0;
    try {
        parsed = static_cast<size_t>(std::stoull(repr));
    } catch (const std::exception&) {
        return false;
    }

    if (field_from_size_t(parsed) != value) {
        return false;
    }

    out = parsed;
    return true;
}

virCommitment maybe_gate(const virCommitment& expr, const virCommitment* lifted_mask)
{
    if (lifted_mask == nullptr) {
        return expr;
    }
    return (*lifted_mask) * expr;
}

} // namespace

void parentChildBoxInheritanceVerifier::add_check(const parentChildBoxInheritanceCheckRecord& record)
{
    checks.push_back(record);
}

bool parentChildBoxInheritanceVerifier::finalize() const
{
    return true;
}

parentChildBoxInheritanceCheckRecord parentChildBoxInheritanceProver::add_check(const virCommitment& left_child_indices,
                                                                                const virCommitment& right_child_indices,
                                                                                const virCommitment& parent_lower,
                                                                                const virCommitment& parent_upper,
                                                                                const virCommitment& thresholds,
                                                                                const virCommitment* mask,
                                                                                size_t node_count,
                                                                                size_t feature_count,
                                                                                size_t bit_bound)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("parentChildBoxInheritanceProver not setup");
    }

    parentChildBoxInheritanceCheckRecord record;
    record.node_count = node_count;
    record.feature_count = feature_count;
    record.bit_bound = bit_bound;
    record.uses_mask = mask != nullptr;

    const poly lifted_threshold_poly =
        params->kzg->materialize(thresholds)->column_replicated(node_count, feature_count);
    record.lifted_threshold_commitment = params->kzg->commit(lifted_threshold_poly, node_count * feature_count - 1);

    if (mask != nullptr) {
        const poly lifted_mask_poly = params->kzg->materialize(*mask)->column_replicated(node_count, feature_count);
        record.lifted_mask_commitment = params->kzg->commit(lifted_mask_poly, node_count * feature_count - 1);
    }

    const std::vector<Fr> left_index_evals = params->kzg->evaluations(left_child_indices, node_count);
    const std::vector<Fr> right_index_evals = params->kzg->evaluations(right_child_indices, node_count);
    std::vector<Fr> mask_evals;
    const std::vector<Fr>* maybe_mask_evals = nullptr;
    if (mask != nullptr) {
        mask_evals = params->kzg->evaluations(*mask, node_count);
        maybe_mask_evals = &mask_evals;
    }

    auto routed_from_positions = [&](const std::vector<Fr>& index_evals, const virCommitment& table) {
        const std::vector<Fr> table_evals = params->kzg->evaluations(table, node_count * feature_count);
        std::vector<Fr> routed(node_count * feature_count, Fr(0));
        for (size_t row = 0; row < node_count; ++row) {
            const bool active = maybe_mask_evals == nullptr || (*maybe_mask_evals)[row] != Fr(0);
            if (!active) {
                continue;
            }

            size_t child = 0;
            if (!try_size_t_from_fr(index_evals[row], child) || child >= node_count) {
                child = 0;
            }
            for (size_t feature = 0; feature < feature_count; ++feature) {
                routed[matrix_index(row, feature, feature_count)] =
                    table_evals[matrix_index(child, feature, feature_count)];
            }
        }
        return poly_from_evals(routed);
    };

    record.left_child_lower_commitment =
        params->kzg->commit(routed_from_positions(left_index_evals, parent_lower), node_count * feature_count - 1);
    record.left_child_upper_commitment =
        params->kzg->commit(routed_from_positions(left_index_evals, parent_upper), node_count * feature_count - 1);
    record.right_child_lower_commitment =
        params->kzg->commit(routed_from_positions(right_index_evals, parent_lower), node_count * feature_count - 1);
    record.right_child_upper_commitment =
        params->kzg->commit(routed_from_positions(right_index_evals, parent_upper), node_count * feature_count - 1);

    checks.push_back(record);
    return record;
}

void parentChildBoxInheritanceProtocol::add_check(const virCommitment& feature_selector,
                                                  const virCommitment& left_child_indices,
                                                  const virCommitment& right_child_indices,
                                                  const virCommitment& parent_lower,
                                                  const virCommitment& parent_upper,
                                                  const virCommitment& left_lower,
                                                  const virCommitment& left_upper,
                                                  const virCommitment& right_lower,
                                                  const virCommitment& right_upper,
                                                  const virCommitment& thresholds,
                                                  size_t node_count,
                                                  size_t feature_count,
                                                  size_t bit_bound)
{
    add_check_internal(feature_selector,
                       nullptr,
                       left_child_indices,
                       right_child_indices,
                       parent_lower,
                       parent_upper,
                       left_lower,
                       left_upper,
                       right_lower,
                       right_upper,
                       thresholds,
                       node_count,
                       feature_count,
                       bit_bound);
}

void parentChildBoxInheritanceProtocol::add_masked_check(const virCommitment& feature_selector,
                                                         const virCommitment& mask,
                                                         const virCommitment& left_child_indices,
                                                         const virCommitment& right_child_indices,
                                                         const virCommitment& parent_lower,
                                                         const virCommitment& parent_upper,
                                                         const virCommitment& left_lower,
                                                         const virCommitment& left_upper,
                                                         const virCommitment& right_lower,
                                                         const virCommitment& right_upper,
                                                         const virCommitment& thresholds,
                                                         size_t node_count,
                                                         size_t feature_count,
                                                         size_t bit_bound)
{
    add_check_internal(feature_selector,
                       &mask,
                       left_child_indices,
                       right_child_indices,
                       parent_lower,
                       parent_upper,
                       left_lower,
                       left_upper,
                       right_lower,
                       right_upper,
                       thresholds,
                       node_count,
                       feature_count,
                       bit_bound);
}

void parentChildBoxInheritanceProtocol::add_check_internal(const virCommitment& feature_selector,
                                                           const virCommitment* mask,
                                                           const virCommitment& left_child_indices,
                                                           const virCommitment& right_child_indices,
                                                           const virCommitment& parent_lower,
                                                           const virCommitment& parent_upper,
                                                           const virCommitment& left_lower,
                                                           const virCommitment& left_upper,
                                                           const virCommitment& right_lower,
                                                           const virCommitment& right_upper,
                                                           const virCommitment& thresholds,
                                                           size_t node_count,
                                                           size_t feature_count,
                                                           size_t bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("parentChildBoxInheritanceProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.selector_check == nullptr ||
        params_.column_replication == nullptr || params_.index_routing == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("parentChildBoxInheritanceProtocol dependencies not setup");
    }

    validate_parent_child_box_parameters(node_count, feature_count, bit_bound);
    const size_t matrix_domain_size = node_count * feature_count;

    parentChildBoxInheritanceCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.node_count = node_count;
        record.feature_count = feature_count;
        record.bit_bound = bit_bound;
        record.uses_mask = mask != nullptr;
        record.lifted_threshold_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, matrix_domain_size - 1);
        if (record.uses_mask) {
            record.lifted_mask_commitment = adopt_transported_commitment(*params_.kzg, proof_session, matrix_domain_size - 1);
        }
        record.left_child_lower_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, matrix_domain_size - 1);
        record.left_child_upper_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, matrix_domain_size - 1);
        record.right_child_lower_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, matrix_domain_size - 1);
        record.right_child_upper_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, matrix_domain_size - 1);
    } else {
        record = prover_.add_check(
            left_child_indices, right_child_indices, parent_lower, parent_upper, thresholds, mask, node_count, feature_count, bit_bound);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 lifted_threshold_group =
                proof_session->transport_group1([&] { return record.lifted_threshold_commitment.com; });
            if (lifted_threshold_group != record.lifted_threshold_commitment.com) {
                throw std::runtime_error(
                    "parentChildBoxInheritanceProtocol: transported lifted-threshold commitment mismatch");
            }
            if (record.uses_mask) {
                const Group1 lifted_mask_group =
                    proof_session->transport_group1([&] { return record.lifted_mask_commitment.com; });
                if (lifted_mask_group != record.lifted_mask_commitment.com) {
                    throw std::runtime_error(
                        "parentChildBoxInheritanceProtocol: transported lifted-mask commitment mismatch");
                }
            }

            const Group1 left_child_lower_group =
                proof_session->transport_group1([&] { return record.left_child_lower_commitment.com; });
            if (left_child_lower_group != record.left_child_lower_commitment.com) {
                throw std::runtime_error(
                    "parentChildBoxInheritanceProtocol: transported left-child-lower commitment mismatch");
            }

            const Group1 left_child_upper_group =
                proof_session->transport_group1([&] { return record.left_child_upper_commitment.com; });
            if (left_child_upper_group != record.left_child_upper_commitment.com) {
                throw std::runtime_error(
                    "parentChildBoxInheritanceProtocol: transported left-child-upper commitment mismatch");
            }

            const Group1 right_child_lower_group =
                proof_session->transport_group1([&] { return record.right_child_lower_commitment.com; });
            if (right_child_lower_group != record.right_child_lower_commitment.com) {
                throw std::runtime_error(
                    "parentChildBoxInheritanceProtocol: transported right-child-lower commitment mismatch");
            }

            const Group1 right_child_upper_group =
                proof_session->transport_group1([&] { return record.right_child_upper_commitment.com; });
            if (right_child_upper_group != record.right_child_upper_commitment.com) {
                throw std::runtime_error(
                    "parentChildBoxInheritanceProtocol: transported right-child-upper commitment mismatch");
            }
        }
    }

    verifier_.add_check(record);
    feature_selectors_.push_back(feature_selector);
    masks_.push_back(mask == nullptr ? virCommitment(Fr(1)) : *mask);
    left_child_indices_.push_back(left_child_indices);
    right_child_indices_.push_back(right_child_indices);
    parent_lowers_.push_back(parent_lower);
    parent_uppers_.push_back(parent_upper);
    left_lowers_.push_back(left_lower);
    left_uppers_.push_back(left_upper);
    right_lowers_.push_back(right_lower);
    right_uppers_.push_back(right_upper);
    thresholds_.push_back(thresholds);

    if (mask == nullptr) {
        params_.selector_check->add_one_hot_rows_check(feature_selector, node_count, feature_count);
    } else {
        params_.selector_check->add_masked_one_hot_rows_check(feature_selector, *mask, node_count, feature_count);
    }
}

bool parentChildBoxInheritanceProtocol::finalize()
{
    begin_finalize("parentChildBoxInheritanceProtocol");

    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        const parentChildBoxInheritanceCheckRecord& record = verifier_.checks[i];
        const size_t matrix_domain_size = record.node_count * record.feature_count;
        const virCommitment lifted_threshold(record.lifted_threshold_commitment);
        const virCommitment selector = feature_selectors_[i];
        const virCommitment left_child_lower(record.left_child_lower_commitment);
        const virCommitment left_child_upper(record.left_child_upper_commitment);
        const virCommitment right_child_lower(record.right_child_lower_commitment);
        const virCommitment right_child_upper(record.right_child_upper_commitment);

        params_.column_replication->add_check(thresholds_[i], lifted_threshold, record.node_count, record.feature_count);

        std::optional<virCommitment> lifted_mask;
        if (record.uses_mask) {
            lifted_mask = virCommitment(record.lifted_mask_commitment);
            params_.column_replication->add_check(masks_[i], *lifted_mask, record.node_count, record.feature_count);
        }

        if (lifted_mask.has_value()) {
            params_.index_routing->add_masked_check(left_child_indices_[i],
                                                    masks_[i],
                                                    std::vector<virCommitment>{parent_lowers_[i], parent_uppers_[i]},
                                                    std::vector<virCommitment>{left_child_lower, left_child_upper},
                                                    record.node_count,
                                                    record.node_count,
                                                    record.feature_count);
            params_.index_routing->add_masked_check(right_child_indices_[i],
                                                    masks_[i],
                                                    std::vector<virCommitment>{parent_lowers_[i], parent_uppers_[i]},
                                                    std::vector<virCommitment>{right_child_lower, right_child_upper},
                                                    record.node_count,
                                                    record.node_count,
                                                    record.feature_count);
        } else {
            params_.index_routing->add_check(left_child_indices_[i],
                                             std::vector<virCommitment>{parent_lowers_[i], parent_uppers_[i]},
                                             std::vector<virCommitment>{left_child_lower, left_child_upper},
                                             record.node_count,
                                             record.node_count,
                                             record.feature_count);
            params_.index_routing->add_check(right_child_indices_[i],
                                             std::vector<virCommitment>{parent_lowers_[i], parent_uppers_[i]},
                                             std::vector<virCommitment>{right_child_lower, right_child_upper},
                                             record.node_count,
                                             record.node_count,
                                             record.feature_count);
        }

        virCommitment left_child_lower_zero =
            left_lowers_[i] - left_child_lower;
        virCommitment left_child_upper_zero =
            left_uppers_[i] - left_child_upper;
        virCommitment right_child_lower_zero =
            right_lowers_[i] - right_child_lower;
        virCommitment right_child_upper_zero =
            right_uppers_[i] - right_child_upper;
        params_.zero_check->add_check(left_child_lower_zero, matrix_domain_size);
        params_.zero_check->add_check(left_child_upper_zero, matrix_domain_size);
        params_.zero_check->add_check(right_child_lower_zero, matrix_domain_size);
        params_.zero_check->add_check(right_child_upper_zero, matrix_domain_size);

        virCommitment left_lower_zero = maybe_gate(left_lowers_[i] - parent_lowers_[i], lifted_mask ? &*lifted_mask : nullptr);
        params_.zero_check->add_check(left_lower_zero, matrix_domain_size);

        virCommitment right_upper_zero =
            maybe_gate(right_uppers_[i] - parent_uppers_[i], lifted_mask ? &*lifted_mask : nullptr);
        params_.zero_check->add_check(right_upper_zero, matrix_domain_size);

        virCommitment non_selected = virCommitment(Fr(1)) - selector;

        virCommitment left_upper_non_selected_zero =
            maybe_gate(non_selected * (left_uppers_[i] - parent_uppers_[i]), lifted_mask ? &*lifted_mask : nullptr);
        params_.zero_check->add_check(left_upper_non_selected_zero, matrix_domain_size);

        virCommitment right_lower_non_selected_zero =
            maybe_gate(non_selected * (right_lowers_[i] - parent_lowers_[i]), lifted_mask ? &*lifted_mask : nullptr);
        params_.zero_check->add_check(right_lower_non_selected_zero, matrix_domain_size);

        virCommitment left_threshold_zero =
            maybe_gate(selector * (left_uppers_[i] - lifted_threshold), lifted_mask ? &*lifted_mask : nullptr);
        params_.zero_check->add_check(left_threshold_zero, matrix_domain_size);

        virCommitment right_threshold_zero =
            maybe_gate(selector * (right_lowers_[i] - lifted_threshold), lifted_mask ? &*lifted_mask : nullptr);
        params_.zero_check->add_check(right_threshold_zero, matrix_domain_size);

        // Every parent/child half-open interval should remain non-empty.
        if (lifted_mask.has_value()) {
            params_.less_check->add_masked_check(
                parent_uppers_[i] - virCommitment(Fr(1)) - parent_lowers_[i], *lifted_mask, record.bit_bound, matrix_domain_size);
            params_.less_check->add_masked_check(
                left_uppers_[i] - virCommitment(Fr(1)) - left_lowers_[i], *lifted_mask, record.bit_bound, matrix_domain_size);
            params_.less_check->add_masked_check(
                right_uppers_[i] - virCommitment(Fr(1)) - right_lowers_[i], *lifted_mask, record.bit_bound, matrix_domain_size);
        } else {
            params_.less_check->add_check(
                parent_uppers_[i] - virCommitment(Fr(1)) - parent_lowers_[i], record.bit_bound, matrix_domain_size);
            params_.less_check->add_check(
                left_uppers_[i] - virCommitment(Fr(1)) - left_lowers_[i], record.bit_bound, matrix_domain_size);
            params_.less_check->add_check(
                right_uppers_[i] - virCommitment(Fr(1)) - right_lowers_[i], record.bit_bound, matrix_domain_size);
        }

        // Strict threshold placement is already implied on the selected feature:
        // the left/right child intervals are non-empty, and their touching
        // endpoints are constrained to equal the threshold.
    }

    return verifier_.finalize();
}
