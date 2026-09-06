#include "forest_depth_consistency.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_forest_depth_parameters(size_t tree_count, size_t node_count)
{
    validate_matrix_shape(tree_count, 1, "forestDepthConsistencyProtocol root indices");
    validate_matrix_shape(node_count, 1, "forestDepthConsistencyProtocol node vectors");
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

std::vector<Fr> route_depths_from_indices(const std::vector<Fr>& index_evals,
                                          const std::vector<Fr>& depth_evals,
                                          const std::vector<Fr>* maybe_mask_evals,
                                          size_t node_count)
{
    std::vector<Fr> out(index_evals.size(), Fr(0));
    for (size_t row = 0; row < index_evals.size(); ++row) {
        const bool active = maybe_mask_evals == nullptr || (*maybe_mask_evals)[row] != Fr(0);
        if (!active) {
            continue;
        }

        size_t index = 0;
        if (!try_size_t_from_fr(index_evals[row], index) || index >= node_count) {
            index = 0;
        }
        out[row] = depth_evals[index];
    }
    return out;
}

} // namespace

void forestDepthConsistencyVerifier::add_check(const forestDepthConsistencyCheckRecord& record)
{
    checks.push_back(record);
}

bool forestDepthConsistencyVerifier::finalize() const
{
    return true;
}

forestDepthConsistencyCheckRecord forestDepthConsistencyProver::add_check(const virCommitment& root_indices,
                                                                          const virCommitment& parent_mask,
                                                                          const virCommitment& left_child_indices,
                                                                          const virCommitment& right_child_indices,
                                                                          const virCommitment& depths,
                                                                          size_t tree_count,
                                                                          size_t node_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("forestDepthConsistencyProver not setup");
    }

    const std::vector<Fr> root_index_evals = params->kzg->evaluations(root_indices, tree_count);
    const std::vector<Fr> mask_evals = params->kzg->evaluations(parent_mask, node_count);
    const std::vector<Fr> left_index_evals = params->kzg->evaluations(left_child_indices, node_count);
    const std::vector<Fr> right_index_evals = params->kzg->evaluations(right_child_indices, node_count);
    const std::vector<Fr> depth_evals = params->kzg->evaluations(depths, node_count);

    std::vector<Fr> root_depths(tree_count, Fr(0));
    for (size_t tree = 0; tree < tree_count; ++tree) {
        size_t root = 0;
        if (!try_size_t_from_fr(root_index_evals[tree], root) || root >= node_count) {
            root = 0;
        }
        root_depths[tree] = depth_evals[root];
    }

    forestDepthConsistencyCheckRecord record;
    record.tree_count = tree_count;
    record.node_count = node_count;
    record.root_depths_commitment = params->kzg->commit(poly_from_evals(root_depths), tree_count - 1);
    record.left_child_depths_commitment = params->kzg->commit(
        poly_from_evals(route_depths_from_indices(left_index_evals, depth_evals, &mask_evals, node_count)),
        node_count - 1);
    record.right_child_depths_commitment = params->kzg->commit(
        poly_from_evals(route_depths_from_indices(right_index_evals, depth_evals, &mask_evals, node_count)),
        node_count - 1);

    checks.push_back(record);
    return record;
}

void forestDepthConsistencyProtocol::add_check(const virCommitment& root_indices,
                                               const virCommitment& parent_mask,
                                               const virCommitment& left_child_indices,
                                               const virCommitment& right_child_indices,
                                               const virCommitment& depths,
                                               size_t tree_count,
                                               size_t node_count)
{
    if (!is_setup) {
        throw std::runtime_error("forestDepthConsistencyProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.index_routing == nullptr) {
        throw std::runtime_error("forestDepthConsistencyProtocol dependencies not setup");
    }

    validate_forest_depth_parameters(tree_count, node_count);

    forestDepthConsistencyCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.tree_count = tree_count;
        record.node_count = node_count;
        record.root_depths_commitment = adopt_transported_commitment(*params_.kzg, proof_session, tree_count - 1);
        record.left_child_depths_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, node_count - 1);
        record.right_child_depths_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, node_count - 1);
    } else {
        record =
            prover_.add_check(root_indices, parent_mask, left_child_indices, right_child_indices, depths, tree_count, node_count);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 root_depths_group =
                proof_session->transport_group1([&] { return record.root_depths_commitment.com; });
            if (root_depths_group != record.root_depths_commitment.com) {
                throw std::runtime_error(
                    "forestDepthConsistencyProtocol: transported root-depths commitment mismatch");
            }

            const Group1 left_depths_group =
                proof_session->transport_group1([&] { return record.left_child_depths_commitment.com; });
            if (left_depths_group != record.left_child_depths_commitment.com) {
                throw std::runtime_error(
                    "forestDepthConsistencyProtocol: transported left-child-depths commitment mismatch");
            }

            const Group1 right_depths_group =
                proof_session->transport_group1([&] { return record.right_child_depths_commitment.com; });
            if (right_depths_group != record.right_child_depths_commitment.com) {
                throw std::runtime_error(
                    "forestDepthConsistencyProtocol: transported right-child-depths commitment mismatch");
            }
        }
    }

    verifier_.add_check(record);
    root_indices_.push_back(root_indices);
    parent_masks_.push_back(parent_mask);
    left_child_indices_.push_back(left_child_indices);
    right_child_indices_.push_back(right_child_indices);
    depths_.push_back(depths);
}

bool forestDepthConsistencyProtocol::finalize()
{
    begin_finalize("forestDepthConsistencyProtocol");

    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        const forestDepthConsistencyCheckRecord& record = verifier_.checks[i];
        const virCommitment root_depths(record.root_depths_commitment);
        const virCommitment left_child_depths(record.left_child_depths_commitment);
        const virCommitment right_child_depths(record.right_child_depths_commitment);

        params_.index_routing->add_check(
            root_indices_[i], depths_[i], root_depths, record.tree_count, record.node_count, 1);
        params_.index_routing->add_masked_check(left_child_indices_[i],
                                                parent_masks_[i],
                                                depths_[i],
                                                left_child_depths,
                                                record.node_count,
                                                record.node_count,
                                                1);
        params_.index_routing->add_masked_check(right_child_indices_[i],
                                                parent_masks_[i],
                                                depths_[i],
                                                right_child_depths,
                                                record.node_count,
                                                record.node_count,
                                                1);

        virCommitment root_depth_zero = root_depths - virCommitment(Fr(1));
        params_.zero_check->add_check(root_depth_zero, record.tree_count);

        const virCommitment expected_child_depths =
            parent_masks_[i] * (depths_[i] + virCommitment(Fr(1)));
        virCommitment left_depth_zero = left_child_depths - expected_child_depths;
        virCommitment right_depth_zero = right_child_depths - expected_child_depths;
        params_.zero_check->add_check(left_depth_zero, record.node_count);
        params_.zero_check->add_check(right_depth_zero, record.node_count);
    }

    return verifier_.finalize();
}
