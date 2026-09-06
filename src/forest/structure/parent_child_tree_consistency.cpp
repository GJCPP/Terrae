#include "parent_child_tree_consistency.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_parent_child_tree_parameters(size_t node_count)
{
    validate_matrix_shape(node_count, 1, "parentChildTreeConsistencyProtocol");
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

kzgCommitment adopt_transported_commitment(kzgProtocol& kzg,
                                           const std::shared_ptr<ProofSession>& proof_session,
                                           size_t degree_bound)
{
    const Group1 commitment_group = proof_session->transport_group1({});
    const kzgCommitment commitment = kzg.adopt_commitment(commitment_group);
    kzg.prove_degree(commitment, degree_bound);
    return commitment;
}

std::vector<Fr> routed_tree_ids_from_witness(const std::vector<Fr>& child_index_evals,
                                             const std::vector<Fr>& node_tree_evals,
                                             const std::vector<Fr>* maybe_mask_evals,
                                             size_t node_count)
{
    std::vector<Fr> out(child_index_evals.size(), Fr(0));
    for (size_t parent = 0; parent < child_index_evals.size(); ++parent) {
        const bool active = maybe_mask_evals == nullptr || (*maybe_mask_evals)[parent] != Fr(0);
        if (!active) {
            continue;
        }

        size_t child = 0;
        if (!try_size_t_from_fr(child_index_evals[parent], child) || child >= node_count) {
            child = 0;
        }
        // Route each active parent through its selected child index.
        out[parent] = node_tree_evals[child];
    }
    return out;
}

} // namespace

void parentChildTreeConsistencyVerifier::add_check(const parentChildTreeConsistencyCheckRecord& record)
{
    checks.push_back(record);
}

bool parentChildTreeConsistencyVerifier::finalize() const
{
    return true;
}

parentChildTreeConsistencyCheckRecord parentChildTreeConsistencyProver::add_check(
    const virCommitment& left_child_indices,
    const virCommitment& right_child_indices,
    const virCommitment& node_tree_ids,
    const virCommitment* mask,
    size_t node_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("parentChildTreeConsistencyProver not setup");
    }

    const std::vector<Fr> left_index_evals = params->kzg->evaluations(left_child_indices, node_count);
    const std::vector<Fr> right_index_evals = params->kzg->evaluations(right_child_indices, node_count);
    const std::vector<Fr> node_tree_evals = params->kzg->evaluations(node_tree_ids, node_count);

    std::vector<Fr> mask_evals;
    const std::vector<Fr>* maybe_mask_evals = nullptr;
    if (mask != nullptr) {
        mask_evals = params->kzg->evaluations(*mask, node_count);
        maybe_mask_evals = &mask_evals;
    }

    parentChildTreeConsistencyCheckRecord record;
    record.node_count = node_count;
    record.uses_mask = mask != nullptr;
    record.left_child_tree_ids_commitment =
        params->kzg->commit(poly_from_evals(routed_tree_ids_from_witness(
            left_index_evals, node_tree_evals, maybe_mask_evals, node_count)),
                            node_count - 1);
    record.right_child_tree_ids_commitment =
        params->kzg->commit(poly_from_evals(routed_tree_ids_from_witness(
            right_index_evals, node_tree_evals, maybe_mask_evals, node_count)),
                            node_count - 1);

    checks.push_back(record);
    return record;
}

void parentChildTreeConsistencyProtocol::add_check(const virCommitment& left_child_indices,
                                                   const virCommitment& right_child_indices,
                                                   const virCommitment& node_tree_ids,
                                                   size_t node_count)
{
    add_check_internal(left_child_indices, right_child_indices, nullptr, node_tree_ids, node_count);
}

void parentChildTreeConsistencyProtocol::add_masked_check(const virCommitment& left_child_indices,
                                                          const virCommitment& right_child_indices,
                                                          const virCommitment& mask,
                                                          const virCommitment& node_tree_ids,
                                                          size_t node_count)
{
    add_check_internal(left_child_indices, right_child_indices, &mask, node_tree_ids, node_count);
}

void parentChildTreeConsistencyProtocol::add_check_internal(const virCommitment& left_child_indices,
                                                            const virCommitment& right_child_indices,
                                                            const virCommitment* mask,
                                                            const virCommitment& node_tree_ids,
                                                            size_t node_count)
{
    if (!is_setup) {
        throw std::runtime_error("parentChildTreeConsistencyProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.index_routing == nullptr) {
        throw std::runtime_error("parentChildTreeConsistencyProtocol dependencies not setup");
    }

    validate_parent_child_tree_parameters(node_count);

    parentChildTreeConsistencyCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.node_count = node_count;
        record.uses_mask = mask != nullptr;
        record.left_child_tree_ids_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, node_count - 1);
        record.right_child_tree_ids_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, node_count - 1);
    } else {
        record = prover_.add_check(left_child_indices, right_child_indices, node_tree_ids, mask, node_count);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 left_group =
                proof_session->transport_group1([&] { return record.left_child_tree_ids_commitment.com; });
            if (left_group != record.left_child_tree_ids_commitment.com) {
                throw std::runtime_error(
                    "parentChildTreeConsistencyProtocol: transported left-child-tree-id commitment mismatch");
            }

            const Group1 right_group =
                proof_session->transport_group1([&] { return record.right_child_tree_ids_commitment.com; });
            if (right_group != record.right_child_tree_ids_commitment.com) {
                throw std::runtime_error(
                    "parentChildTreeConsistencyProtocol: transported right-child-tree-id commitment mismatch");
            }
        }
    }

    verifier_.add_check(record);
    left_child_indices_.push_back(left_child_indices);
    right_child_indices_.push_back(right_child_indices);
    masks_.push_back(mask == nullptr ? virCommitment(Fr(1)) : *mask);
    node_tree_ids_.push_back(node_tree_ids);
}

bool parentChildTreeConsistencyProtocol::finalize()
{
    begin_finalize("parentChildTreeConsistencyProtocol");

    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        const parentChildTreeConsistencyCheckRecord& record = verifier_.checks[i];
        const virCommitment left_child_tree_ids(record.left_child_tree_ids_commitment);
        const virCommitment right_child_tree_ids(record.right_child_tree_ids_commitment);

        if (record.uses_mask) {
            // Inactive parents are routed through row 0 only for witness
            // totality; equality is checked against `mask * node_tree_ids`.
            params_.index_routing->add_masked_check(left_child_indices_[i],
                                                    masks_[i],
                                                    node_tree_ids_[i],
                                                    left_child_tree_ids,
                                                    record.node_count,
                                                    record.node_count,
                                                    1);
            params_.index_routing->add_masked_check(right_child_indices_[i],
                                                    masks_[i],
                                                    node_tree_ids_[i],
                                                    right_child_tree_ids,
                                                    record.node_count,
                                                    record.node_count,
                                                    1);

            const virCommitment expected_parent_tree_ids = masks_[i] * node_tree_ids_[i];
            virCommitment left_tree_zero = left_child_tree_ids - expected_parent_tree_ids;
            virCommitment right_tree_zero = right_child_tree_ids - expected_parent_tree_ids;
            params_.zero_check->add_check(left_tree_zero, record.node_count);
            params_.zero_check->add_check(right_tree_zero, record.node_count);
            continue;
        }

        // Unmasked path: both routed child-tree vectors must equal the parent
        // tree-id vector pointwise.
        params_.index_routing->add_check(
            left_child_indices_[i], node_tree_ids_[i], left_child_tree_ids, record.node_count, record.node_count, 1);
        params_.index_routing->add_check(
            right_child_indices_[i], node_tree_ids_[i], right_child_tree_ids, record.node_count, record.node_count, 1);
        virCommitment left_tree_zero = left_child_tree_ids - node_tree_ids_[i];
        virCommitment right_tree_zero = right_child_tree_ids - node_tree_ids_[i];
        params_.zero_check->add_check(left_tree_zero, record.node_count);
        params_.zero_check->add_check(right_tree_zero, record.node_count);
    }

    return verifier_.finalize();
}
