#include "node_role_consistency.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_node_role_parameters(size_t tree_count, size_t node_count)
{
    validate_matrix_shape(tree_count, 1, "nodeRoleConsistencyProtocol root indices");
    validate_matrix_shape(node_count, 1, "nodeRoleConsistencyProtocol child indices");
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

std::vector<Fr> occupancy_from_indices(const std::vector<Fr>& index_evals,
                                       const std::vector<Fr>* maybe_weights,
                                       size_t node_count)
{
    std::vector<Fr> occupancy(node_count, Fr(0));
    for (size_t i = 0; i < index_evals.size(); ++i) {
        size_t parsed = 0;
        if (!try_size_t_from_fr(index_evals[i], parsed) || parsed >= node_count) {
            parsed = 0;
        }

        const Fr weight = maybe_weights == nullptr ? Fr(1) : (*maybe_weights)[i];
        // Occupancy is a histogram over selected destination nodes.
        Fr::add(occupancy[parsed], occupancy[parsed], weight);
    }
    return occupancy;
}

} // namespace

void nodeRoleConsistencyVerifier::add_check(const nodeRoleConsistencyCheckRecord& record)
{
    checks.push_back(record);
}

bool nodeRoleConsistencyVerifier::finalize() const
{
    return true;
}

nodeRoleConsistencyCheckRecord nodeRoleConsistencyProver::add_check(const virCommitment& root_indices,
                                                                    const virCommitment& node_leaf_mask,
                                                                    const virCommitment& left_child_indices,
                                                                    const virCommitment& right_child_indices,
                                                                    const virCommitment* parent_mask,
                                                                    size_t tree_count,
                                                                    size_t node_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("nodeRoleConsistencyProver not setup");
    }
    (void)node_leaf_mask;

    const std::vector<Fr> root_index_evals = params->kzg->evaluations(root_indices, tree_count);
    const std::vector<Fr> left_index_evals = params->kzg->evaluations(left_child_indices, node_count);
    const std::vector<Fr> right_index_evals = params->kzg->evaluations(right_child_indices, node_count);

    std::vector<Fr> mask_evals;
    const std::vector<Fr>* maybe_mask_evals = nullptr;
    if (parent_mask != nullptr) {
        mask_evals = params->kzg->evaluations(*parent_mask, node_count);
        maybe_mask_evals = &mask_evals;
    }

    const std::vector<Fr> root_occupancy_evals = occupancy_from_indices(root_index_evals, nullptr, node_count);
    const std::vector<Fr> left_occupancy_evals = occupancy_from_indices(left_index_evals, maybe_mask_evals, node_count);
    const std::vector<Fr> right_occupancy_evals = occupancy_from_indices(right_index_evals, maybe_mask_evals, node_count);

    std::vector<Fr> total_occupancy_evals(node_count, Fr(0));
    for (size_t i = 0; i < node_count; ++i) {
        total_occupancy_evals[i] = root_occupancy_evals[i] + left_occupancy_evals[i] + right_occupancy_evals[i];
    }

    nodeRoleConsistencyCheckRecord record;
    record.tree_count = tree_count;
    record.node_count = node_count;
    record.uses_mask = parent_mask != nullptr;
    record.root_occupancy_commitment = params->kzg->commit(poly_from_evals(root_occupancy_evals), node_count - 1);
    record.left_child_occupancy_commitment = params->kzg->commit(poly_from_evals(left_occupancy_evals), node_count - 1);
    record.right_child_occupancy_commitment = params->kzg->commit(poly_from_evals(right_occupancy_evals), node_count - 1);
    record.total_occupancy_commitment = params->kzg->commit(poly_from_evals(total_occupancy_evals), node_count - 1);

    checks.push_back(record);
    return record;
}

void nodeRoleConsistencyProtocol::add_check(const virCommitment& root_indices,
                                            const virCommitment& node_leaf_mask,
                                            const virCommitment& left_child_indices,
                                            const virCommitment& right_child_indices,
                                            size_t tree_count,
                                            size_t node_count)
{
    add_check_internal(
        root_indices, node_leaf_mask, left_child_indices, right_child_indices, nullptr, tree_count, node_count);
}

void nodeRoleConsistencyProtocol::add_masked_check(const virCommitment& root_indices,
                                                   const virCommitment& node_leaf_mask,
                                                   const virCommitment& left_child_indices,
                                                   const virCommitment& right_child_indices,
                                                   const virCommitment& parent_mask,
                                                   size_t tree_count,
                                                   size_t node_count)
{
    add_check_internal(
        root_indices, node_leaf_mask, left_child_indices, right_child_indices, &parent_mask, tree_count, node_count);
}

void nodeRoleConsistencyProtocol::add_check_internal(const virCommitment& root_indices,
                                                     const virCommitment& node_leaf_mask,
                                                     const virCommitment& left_child_indices,
                                                     const virCommitment& right_child_indices,
                                                     const virCommitment* parent_mask,
                                                     size_t tree_count,
                                                     size_t node_count)
{
    if (!is_setup) {
        throw std::runtime_error("nodeRoleConsistencyProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.hist_agg == nullptr) {
        throw std::runtime_error("nodeRoleConsistencyProtocol dependencies not setup");
    }

    validate_node_role_parameters(tree_count, node_count);

    nodeRoleConsistencyCheckRecord record;
    record.tree_count = tree_count;
    record.node_count = node_count;
    record.uses_mask = parent_mask != nullptr;

    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.root_occupancy_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, node_count - 1);
        record.left_child_occupancy_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, node_count - 1);
        record.right_child_occupancy_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, node_count - 1);
        record.total_occupancy_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, node_count - 1);
    } else {
        record = prover_.add_check(
            root_indices, node_leaf_mask, left_child_indices, right_child_indices, parent_mask, tree_count, node_count);

        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 root_group =
                proof_session->transport_group1([&] { return record.root_occupancy_commitment.com; });
            if (root_group != record.root_occupancy_commitment.com) {
                throw std::runtime_error(
                    "nodeRoleConsistencyProtocol: transported root-occupancy commitment mismatch");
            }

            const Group1 left_group =
                proof_session->transport_group1([&] { return record.left_child_occupancy_commitment.com; });
            if (left_group != record.left_child_occupancy_commitment.com) {
                throw std::runtime_error(
                    "nodeRoleConsistencyProtocol: transported left-child-occupancy commitment mismatch");
            }

            const Group1 right_group =
                proof_session->transport_group1([&] { return record.right_child_occupancy_commitment.com; });
            if (right_group != record.right_child_occupancy_commitment.com) {
                throw std::runtime_error(
                    "nodeRoleConsistencyProtocol: transported right-child-occupancy commitment mismatch");
            }

            const Group1 total_group =
                proof_session->transport_group1([&] { return record.total_occupancy_commitment.com; });
            if (total_group != record.total_occupancy_commitment.com) {
                throw std::runtime_error(
                    "nodeRoleConsistencyProtocol: transported total-occupancy commitment mismatch");
            }
        }
    }

    verifier_.add_check(record);
    root_indices_.push_back(root_indices);
    node_leaf_masks_.push_back(node_leaf_mask);
    left_child_indices_.push_back(left_child_indices);
    right_child_indices_.push_back(right_child_indices);
    parent_masks_.push_back(parent_mask == nullptr ? virCommitment(Fr(1)) : *parent_mask);
}

bool nodeRoleConsistencyProtocol::finalize()
{
    begin_finalize("nodeRoleConsistencyProtocol");

    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        const nodeRoleConsistencyCheckRecord& record = verifier_.checks[i];
        const virCommitment root_occupancy(record.root_occupancy_commitment);
        const virCommitment left_occupancy(record.left_child_occupancy_commitment);
        const virCommitment right_occupancy(record.right_child_occupancy_commitment);
        const virCommitment total_occupancy(record.total_occupancy_commitment);
        const virCommitment child_weights = record.uses_mask ? parent_masks_[i] : virCommitment(Fr(1));
        const virCommitment effective_left_indices =
            record.uses_mask ? left_child_indices_[i] * parent_masks_[i] : left_child_indices_[i];
        const virCommitment effective_right_indices =
            record.uses_mask ? right_child_indices_[i] * parent_masks_[i] : right_child_indices_[i];

        params_.hist_agg->add_check(root_indices_[i], virCommitment(Fr(1)), root_occupancy, record.tree_count, record.node_count);
        params_.hist_agg->add_check(
            effective_left_indices, child_weights, left_occupancy, record.node_count, record.node_count);
        params_.hist_agg->add_check(
            effective_right_indices, child_weights, right_occupancy, record.node_count, record.node_count);

        // Recombine the three occupancies and prove the total is boolean, so no
        // node can be claimed by more than one incoming role.
        virCommitment occupancy_zero = total_occupancy - root_occupancy - left_occupancy - right_occupancy;
        params_.zero_check->add_check(occupancy_zero, record.node_count);
        virCommitment total_occupancy_boolean_zero = total_occupancy * (total_occupancy - virCommitment(Fr(1)));
        params_.zero_check->add_check(total_occupancy_boolean_zero, record.node_count);
        params_.zero_check->add_boolean_check(
            node_leaf_masks_[i], record.node_count, "node_role_consistency.leaf_mask_boolean");
        virCommitment exact_coverage_zero = total_occupancy - parent_masks_[i] - node_leaf_masks_[i];
        params_.zero_check->add_check(exact_coverage_zero, record.node_count);
    }

    return verifier_.finalize();
}
