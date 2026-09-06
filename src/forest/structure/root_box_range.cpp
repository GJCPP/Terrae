#include "root_box_range.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_root_box_range_parameters(size_t tree_count, size_t node_count, size_t feature_count, size_t full_upper_bound)
{
    validate_matrix_shape(tree_count, 1, "rootBoxRangeProtocol root indices");
    validate_matrix_shape(node_count, feature_count, "rootBoxRangeProtocol node boxes");
    validate_matrix_shape(tree_count, feature_count, "rootBoxRangeProtocol routed root boxes");
    if (full_upper_bound == 0) {
        throw std::runtime_error("rootBoxRangeProtocol: full_upper_bound must be non-zero");
    }
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

poly routed_box_polynomial(const std::vector<Fr>& root_index_evals,
                           const std::vector<Fr>& box_evals,
                           size_t tree_count,
                           size_t node_count,
                           size_t feature_count)
{
    std::vector<Fr> out(tree_count * feature_count, Fr(0));
    for (size_t tree = 0; tree < tree_count; ++tree) {
        size_t selected_node = 0;
        if (!try_size_t_from_fr(root_index_evals[tree], selected_node) || selected_node >= node_count) {
            selected_node = 0;
        }
        // Copy the selected root row into the tree-local routed box matrix.
        for (size_t feature = 0; feature < feature_count; ++feature) {
            out[matrix_index(tree, feature, feature_count)] = box_evals[matrix_index(selected_node, feature, feature_count)];
        }
    }
    return poly_from_evals(out);
}

} // namespace

void rootBoxRangeVerifier::add_check(const rootBoxRangeCheckRecord& record)
{
    checks.push_back(record);
}

bool rootBoxRangeVerifier::finalize() const
{
    return true;
}

rootBoxRangeCheckRecord rootBoxRangeProver::add_check(const virCommitment& root_indices,
                                                      const virCommitment& lower_bounds,
                                                      const virCommitment& upper_bounds,
                                                      size_t tree_count,
                                                      size_t node_count,
                                                      size_t feature_count,
                                                      size_t full_upper_bound)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("rootBoxRangeProver not setup");
    }

    const std::vector<Fr> root_index_evals = params->kzg->evaluations(root_indices, tree_count);
    const std::vector<Fr> lower_evals = params->kzg->evaluations(lower_bounds, node_count * feature_count);
    const std::vector<Fr> upper_evals = params->kzg->evaluations(upper_bounds, node_count * feature_count);

    rootBoxRangeCheckRecord record;
    record.tree_count = tree_count;
    record.node_count = node_count;
    record.feature_count = feature_count;
    record.full_upper_bound = full_upper_bound;
    record.root_lower_commitment =
        params->kzg->commit(routed_box_polynomial(root_index_evals, lower_evals, tree_count, node_count, feature_count),
                            tree_count * feature_count - 1);
    record.root_upper_commitment =
        params->kzg->commit(routed_box_polynomial(root_index_evals, upper_evals, tree_count, node_count, feature_count),
                            tree_count * feature_count - 1);

    checks.push_back(record);
    return record;
}

void rootBoxRangeProtocol::add_check(const virCommitment& root_indices,
                                     const virCommitment& lower_bounds,
                                     const virCommitment& upper_bounds,
                                     size_t tree_count,
                                     size_t node_count,
                                     size_t feature_count,
                                     size_t full_upper_bound)
{
    if (!is_setup) {
        throw std::runtime_error("rootBoxRangeProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.index_routing == nullptr) {
        throw std::runtime_error("rootBoxRangeProtocol dependencies not setup");
    }

    validate_root_box_range_parameters(tree_count, node_count, feature_count, full_upper_bound);

    rootBoxRangeCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.tree_count = tree_count;
        record.node_count = node_count;
        record.feature_count = feature_count;
        record.full_upper_bound = full_upper_bound;
        record.root_lower_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, tree_count * feature_count - 1);
        record.root_upper_commitment =
            adopt_transported_commitment(*params_.kzg, proof_session, tree_count * feature_count - 1);
    } else {
        record = prover_.add_check(
            root_indices, lower_bounds, upper_bounds, tree_count, node_count, feature_count, full_upper_bound);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 lower_group = proof_session->transport_group1([&] { return record.root_lower_commitment.com; });
            if (lower_group != record.root_lower_commitment.com) {
                throw std::runtime_error("rootBoxRangeProtocol: transported root-lower commitment mismatch");
            }

            const Group1 upper_group = proof_session->transport_group1([&] { return record.root_upper_commitment.com; });
            if (upper_group != record.root_upper_commitment.com) {
                throw std::runtime_error("rootBoxRangeProtocol: transported root-upper commitment mismatch");
            }
        }
    }

    verifier_.add_check(record);
    root_indices_.push_back(root_indices);
    lower_bounds_.push_back(lower_bounds);
    upper_bounds_.push_back(upper_bounds);
}

bool rootBoxRangeProtocol::finalize()
{
    begin_finalize("rootBoxRangeProtocol");

    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        const rootBoxRangeCheckRecord& record = verifier_.checks[i];
        const virCommitment root_lower(record.root_lower_commitment);
        const virCommitment root_upper(record.root_upper_commitment);
        const size_t routed_domain_size = record.tree_count * record.feature_count;

        // Route the committed node boxes through the root index vector.
        params_.index_routing->add_check(
            root_indices_[i], lower_bounds_[i], root_lower, record.tree_count, record.node_count, record.feature_count);
        params_.index_routing->add_check(
            root_indices_[i], upper_bounds_[i], root_upper, record.tree_count, record.node_count, record.feature_count);

        // The selected root box must be exactly `[0, full_upper_bound]` on every
        // feature coordinate.
        virCommitment root_lower_zero = root_lower;
        virCommitment root_upper_zero = root_upper - virCommitment(field_from_size_t(record.full_upper_bound));
        params_.zero_check->add_check(root_lower_zero, routed_domain_size);
        params_.zero_check->add_check(root_upper_zero, routed_domain_size);
    }

    return verifier_.finalize();
}
