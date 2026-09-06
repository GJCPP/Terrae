#include "root_selector_consistency.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_root_selector_parameters(size_t tree_count, size_t node_count)
{
    validate_matrix_shape(tree_count, 1, "rootSelectorConsistencyProtocol root indices");
    validate_matrix_shape(node_count, 1, "rootSelectorConsistencyProtocol node ids");
}

poly canonical_tree_id_polynomial(size_t tree_count)
{
    std::vector<Fr> evals;
    evals.reserve(tree_count);
    for (size_t tree = 0; tree < tree_count; ++tree) {
        evals.push_back(field_from_size_t(tree + 1));
    }
    return poly_from_evals(evals);
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

std::vector<Fr> selected_tree_ids_from_witness(const std::vector<Fr>& root_index_evals,
                                               const std::vector<Fr>& node_tree_evals,
                                               size_t node_count)
{
    std::vector<Fr> out(root_index_evals.size(), Fr(0));
    for (size_t tree = 0; tree < root_index_evals.size(); ++tree) {
        size_t parsed = 0;
        if (!try_size_t_from_fr(root_index_evals[tree], parsed) || parsed >= node_count) {
            parsed = 0;
        }
        out[tree] = node_tree_evals[parsed];
    }
    return out;
}

} // namespace

void rootSelectorConsistencyVerifier::add_check(const rootSelectorConsistencyCheckRecord& record)
{
    checks.push_back(record);
}

bool rootSelectorConsistencyVerifier::finalize() const
{
    return true;
}

rootSelectorConsistencyCheckRecord rootSelectorConsistencyProver::add_check(const virCommitment& root_indices,
                                                                           const virCommitment& node_tree_ids,
                                                                           size_t tree_count,
                                                                           size_t node_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("rootSelectorConsistencyProver not setup");
    }

    const std::vector<Fr> root_index_evals = params->kzg->evaluations(root_indices, tree_count);
    const std::vector<Fr> node_tree_evals = params->kzg->evaluations(node_tree_ids, node_count);

    rootSelectorConsistencyCheckRecord record;
    record.tree_count = tree_count;
    record.node_count = node_count;
    record.selected_tree_ids_commitment = params->kzg->commit(
        poly_from_evals(selected_tree_ids_from_witness(root_index_evals, node_tree_evals, node_count)),
        tree_count - 1);

    checks.push_back(record);
    return record;
}

void rootSelectorConsistencyProtocol::add_check(const virCommitment& root_indices,
                                                const virCommitment& node_tree_ids,
                                                size_t tree_count,
                                                size_t node_count)
{
    if (!is_setup) {
        throw std::runtime_error("rootSelectorConsistencyProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.hist_agg == nullptr ||
        params_.index_routing == nullptr) {
        throw std::runtime_error("rootSelectorConsistencyProtocol dependencies not setup");
    }

    validate_root_selector_parameters(tree_count, node_count);
    if (params_.soundness_counter != nullptr) {
        params_.soundness_counter->record_deterministic_root_uniqueness(tree_count);
    }

    rootSelectorConsistencyCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.tree_count = tree_count;
        record.node_count = node_count;
        record.selected_tree_ids_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, tree_count - 1);
    } else {
        record = prover_.add_check(root_indices, node_tree_ids, tree_count, node_count);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 selected_group =
                proof_session->transport_group1([&] { return record.selected_tree_ids_commitment.com; });
            if (selected_group != record.selected_tree_ids_commitment.com) {
                throw std::runtime_error(
                    "rootSelectorConsistencyProtocol: transported selected-tree-ids commitment mismatch");
            }
        }
    }

    verifier_.add_check(record);
    root_indices_.push_back(root_indices);
    node_tree_ids_.push_back(node_tree_ids);
}

void rootSelectorConsistencyProtocol::register_expected_tree_ids_commitment(size_t tree_count, const Group1& commitment)
{
    expected_tree_ids_commitments_[tree_count] = commitment;
}

bool rootSelectorConsistencyProtocol::finalize()
{
    begin_finalize("rootSelectorConsistencyProtocol");

    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        const rootSelectorConsistencyCheckRecord& record = verifier_.checks[i];
        const virCommitment selected_tree_ids(record.selected_tree_ids_commitment);
        const std::string expected_key =
            "root_selector_consistency.expected_tree_ids:" + std::to_string(record.tree_count);
        const poly expected_tree_id_poly = canonical_tree_id_polynomial(record.tree_count);
        virCommitment expected_tree_ids;
        const auto expected_it = expected_tree_ids_commitments_.find(record.tree_count);
        const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
        const bool verifier_mode = proof_session && proof_session->mode() == ProofMode::Verify;
        if (expected_it != expected_tree_ids_commitments_.end() && verifier_mode) {
            expected_tree_ids = virCommitment(
                params_.kzg->adopt_public_commitment(expected_key, expected_it->second));
            if (!params_.kzg->has_degree_bound(expected_tree_ids, record.tree_count - 1)) {
                params_.kzg->prove_degree(expected_tree_ids.com, record.tree_count - 1);
            }
        } else {
            expected_tree_ids = virCommitment(
                params_.kzg->commit_public(expected_key, expected_tree_id_poly, record.tree_count - 1));
        }

        // The canonical expected tree ids are distinct, so the routing relation
        // also prevents one physical node from serving as two tree roots.
        // Routing proof: the selected root of tree `t` must belong to tree `t`.
        params_.index_routing->add_check(
            root_indices_[i], node_tree_ids_[i], selected_tree_ids, record.tree_count, record.node_count, 1);
        virCommitment selected_tree_zero = selected_tree_ids - expected_tree_ids;
        params_.zero_check->add_check(selected_tree_zero, record.tree_count);
    }

    return verifier_.finalize();
}
