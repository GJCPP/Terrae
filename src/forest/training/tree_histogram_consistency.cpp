#include "tree_histogram_consistency.hpp"

#include "matrix_utils.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool tree_histogram_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_TREE_HISTOGRAM");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
auto tree_histogram_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = tree_histogram_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        return fn();
    }
    const auto start = std::chrono::steady_clock::now();
    auto result = fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "tree_histogram_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("tree_histogram_profile", label, "", elapsed.count(), "ok");
    return result;
}

void validate_tree_histogram_parameters(size_t node_count, size_t histogram_width)
{
    validate_matrix_shape(node_count, 1, "treeHistogramConsistencyProtocol child indices");
    validate_matrix_shape(node_count, histogram_width, "treeHistogramConsistencyProtocol histograms");
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

class routedChildPositions {
public:
    std::vector<size_t> positions;
    std::vector<bool> active_rows;
};

routedChildPositions routed_positions_from_child_indices(const std::vector<Fr>& child_index_evals,
                                                         const std::vector<Fr>* maybe_parent_mask_evals,
                                                         size_t node_count)
{
    routedChildPositions routed;
    routed.positions.assign(node_count, 0);
    routed.active_rows.assign(node_count, true);
    for (size_t parent = 0; parent < node_count; ++parent) {
        const bool active = maybe_parent_mask_evals == nullptr || (*maybe_parent_mask_evals)[parent] != Fr(0);
        if (!active) {
            routed.positions[parent] = 0;
            routed.active_rows[parent] = false;
            continue;
        }

        size_t child = 0;
        if (!try_size_t_from_fr(child_index_evals[parent], child) || child >= node_count) {
            child = 0;
        }
        routed.positions[parent] = child;
    }
    return routed;
}

poly routed_histogram_polynomial_from_positions(const routedChildPositions& child_positions,
                                                const std::vector<Fr>& full_histogram_evals,
                                                size_t node_count,
                                                size_t histogram_width)
{
    std::vector<Fr> routed(node_count * histogram_width, Fr(0));
    for (size_t parent = 0; parent < node_count; ++parent) {
        if (!child_positions.active_rows[parent]) {
            continue;
        }
        const size_t child = child_positions.positions[parent];

        for (size_t column = 0; column < histogram_width; ++column) {
            routed[matrix_index(parent, column, histogram_width)] =
                full_histogram_evals[matrix_index(child, column, histogram_width)];
        }
    }
    return poly_from_evals(routed);
}

} // namespace

void treeHistogramConsistencyVerifier::add_check(const treeHistogramConsistencyCheckRecord& record)
{
    checks.push_back(record);
}

bool treeHistogramConsistencyVerifier::finalize() const
{
    return true;
}

treeHistogramConsistencyCheckRecord treeHistogramConsistencyProver::add_check(
    const std::vector<virCommitment>& full_histograms,
    const virCommitment& left_child_indices,
    const virCommitment& right_child_indices,
    const virCommitment* parent_mask,
    size_t node_count,
    size_t histogram_width)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("treeHistogramConsistencyProver not setup");
    }

    if (full_histograms.empty()) {
        throw std::runtime_error("treeHistogramConsistencyProver: full histogram batch must be non-empty");
    }

    const std::vector<Fr> left_index_evals =
        tree_histogram_profile_scope("evaluate_left_child_indices",
                                     [&] { return params->kzg->evaluations(left_child_indices, node_count); });
    const std::vector<Fr> right_index_evals =
        tree_histogram_profile_scope("evaluate_right_child_indices",
                                     [&] { return params->kzg->evaluations(right_child_indices, node_count); });

    std::vector<Fr> parent_mask_evals;
    const std::vector<Fr>* maybe_parent_mask_evals = nullptr;
    if (parent_mask != nullptr) {
        parent_mask_evals = tree_histogram_profile_scope("evaluate_parent_mask",
                                                         [&] { return params->kzg->evaluations(*parent_mask, node_count); });
        maybe_parent_mask_evals = &parent_mask_evals;
    }

    const routedChildPositions left_positions =
        tree_histogram_profile_scope("derive_left_positions",
                                     [&] { return routed_positions_from_child_indices(left_index_evals, maybe_parent_mask_evals, node_count); });
    const routedChildPositions right_positions =
        tree_histogram_profile_scope("derive_right_positions",
                                     [&] { return routed_positions_from_child_indices(right_index_evals, maybe_parent_mask_evals, node_count); });

    treeHistogramConsistencyCheckRecord record;
    record.node_count = node_count;
    record.histogram_width = histogram_width;
    record.channel_count = full_histograms.size();
    record.uses_parent_mask = parent_mask != nullptr;
    record.left_histogram_commitments.reserve(full_histograms.size());
    record.right_histogram_commitments.reserve(full_histograms.size());

    struct ChannelWitness {
        poly left_routed;
        poly right_routed;
        Group1 left_group;
        Group1 right_group;
    };
    std::vector<ChannelWitness> channel_witnesses(full_histograms.size());
    tree_histogram_profile_scope("build_channel_witnesses", [&] {
        for (size_t channel = 0; channel < full_histograms.size(); ++channel) {
            const std::vector<Fr> full_histogram_evals =
                params->kzg->evaluations(full_histograms[channel], node_count * histogram_width);
            ChannelWitness witness;
            witness.left_routed =
                routed_histogram_polynomial_from_positions(left_positions, full_histogram_evals, node_count, histogram_width);
            witness.right_routed =
                routed_histogram_polynomial_from_positions(right_positions, full_histogram_evals, node_count, histogram_width);
            witness.left_group = params->kzg->commit_g1_group(witness.left_routed);
            witness.right_group = params->kzg->commit_g1_group(witness.right_routed);
            channel_witnesses[channel] = std::move(witness);
        }
        return 0;
    });

    for (ChannelWitness& witness : channel_witnesses) {
        kzgCommitment left_commitment;
        left_commitment.com = witness.left_group;
        left_commitment = params->kzg->commit(witness.left_routed, left_commitment);
        params->kzg->prove_degree(left_commitment, node_count * histogram_width - 1);
        record.left_histogram_commitments.push_back(left_commitment);

        kzgCommitment right_commitment;
        right_commitment.com = witness.right_group;
        right_commitment = params->kzg->commit(witness.right_routed, right_commitment);
        params->kzg->prove_degree(right_commitment, node_count * histogram_width - 1);
        record.right_histogram_commitments.push_back(right_commitment);
    }

    checks.push_back(record);
    return record;
}

void treeHistogramConsistencyProtocol::add_check(const virCommitment& full_histogram,
                                                 const virCommitment& leaf_histogram,
                                                 const virCommitment& left_child_indices,
                                                 const virCommitment& right_child_indices,
                                                 size_t node_count,
                                                 size_t histogram_width)
{
    add_check(std::vector<virCommitment>{full_histogram},
              std::vector<virCommitment>{leaf_histogram},
              left_child_indices,
              right_child_indices,
              node_count,
              histogram_width);
}

void treeHistogramConsistencyProtocol::add_check(const std::vector<virCommitment>& full_histograms,
                                                 const std::vector<virCommitment>& leaf_histograms,
                                                 const virCommitment& left_child_indices,
                                                 const virCommitment& right_child_indices,
                                                 size_t node_count,
                                                 size_t histogram_width)
{
    add_check_internal(
        full_histograms, leaf_histograms, nullptr, left_child_indices, right_child_indices, node_count, histogram_width);
}

void treeHistogramConsistencyProtocol::add_masked_check(const virCommitment& full_histogram,
                                                        const virCommitment& leaf_histogram,
                                                        const virCommitment& parent_mask,
                                                        const virCommitment& left_child_indices,
                                                        const virCommitment& right_child_indices,
                                                        size_t node_count,
                                                        size_t histogram_width)
{
    add_masked_check(std::vector<virCommitment>{full_histogram},
                     std::vector<virCommitment>{leaf_histogram},
                     parent_mask,
                     left_child_indices,
                     right_child_indices,
                     node_count,
                     histogram_width);
}

void treeHistogramConsistencyProtocol::add_masked_check(const std::vector<virCommitment>& full_histograms,
                                                        const std::vector<virCommitment>& leaf_histograms,
                                                        const virCommitment& parent_mask,
                                                        const virCommitment& left_child_indices,
                                                        const virCommitment& right_child_indices,
                                                        size_t node_count,
                                                        size_t histogram_width)
{
    add_check_internal(
        full_histograms, leaf_histograms, &parent_mask, left_child_indices, right_child_indices, node_count, histogram_width);
}

void treeHistogramConsistencyProtocol::add_masked_check_with_certified_parent_mask(
    const std::vector<virCommitment>& full_histograms,
    const std::vector<virCommitment>& leaf_histograms,
    const virCommitment& parent_mask,
    const virCommitment& left_child_indices,
    const virCommitment& right_child_indices,
    size_t node_count,
    size_t histogram_width)
{
    add_check_internal(full_histograms,
                       leaf_histograms,
                       &parent_mask,
                       left_child_indices,
                       right_child_indices,
                       node_count,
                       histogram_width,
                       false);
}

void treeHistogramConsistencyProtocol::add_check_internal(const std::vector<virCommitment>& full_histograms,
                                                          const std::vector<virCommitment>& leaf_histograms,
                                                          const virCommitment* parent_mask,
                                                          const virCommitment& left_child_indices,
                                                          const virCommitment& right_child_indices,
                                                          size_t node_count,
                                                          size_t histogram_width,
                                                          bool enforce_parent_mask_boolean)
{
    if (!is_setup) {
        throw std::runtime_error("treeHistogramConsistencyProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.index_routing == nullptr) {
        throw std::runtime_error("treeHistogramConsistencyProtocol dependencies not setup");
    }

    validate_tree_histogram_parameters(node_count, histogram_width);
    if (full_histograms.empty() || full_histograms.size() != leaf_histograms.size()) {
        throw std::runtime_error("treeHistogramConsistencyProtocol: histogram batch metadata size mismatch");
    }

    treeHistogramConsistencyCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.node_count = node_count;
        record.histogram_width = histogram_width;
        record.channel_count = full_histograms.size();
        record.uses_parent_mask = parent_mask != nullptr;
        record.enforce_parent_mask_boolean = enforce_parent_mask_boolean;
        record.left_histogram_commitments.reserve(record.channel_count);
        record.right_histogram_commitments.reserve(record.channel_count);
        for (size_t channel = 0; channel < record.channel_count; ++channel) {
            record.left_histogram_commitments.push_back(
                adopt_transported_commitment(*params_.kzg, proof_session, node_count * histogram_width - 1));
            record.right_histogram_commitments.push_back(
                adopt_transported_commitment(*params_.kzg, proof_session, node_count * histogram_width - 1));
        }
    } else {
        record = prover_.add_check(
            full_histograms, left_child_indices, right_child_indices, parent_mask, node_count, histogram_width);
        record.enforce_parent_mask_boolean = enforce_parent_mask_boolean;
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            for (size_t channel = 0; channel < record.channel_count; ++channel) {
                const Group1 left_group =
                    proof_session->transport_group1([&] { return record.left_histogram_commitments[channel].com; });
                if (left_group != record.left_histogram_commitments[channel].com) {
                    throw std::runtime_error(
                        "treeHistogramConsistencyProtocol: transported left-histogram commitment mismatch");
                }

                const Group1 right_group =
                    proof_session->transport_group1([&] { return record.right_histogram_commitments[channel].com; });
                if (right_group != record.right_histogram_commitments[channel].com) {
                    throw std::runtime_error(
                        "treeHistogramConsistencyProtocol: transported right-histogram commitment mismatch");
                }
            }
        }
    }

    verifier_.add_check(record);
    full_histogram_batches_.push_back(full_histograms);
    leaf_histogram_batches_.push_back(leaf_histograms);
    parent_masks_.push_back(parent_mask == nullptr ? virCommitment(Fr(1)) : *parent_mask);
    left_child_indices_.push_back(left_child_indices);
    right_child_indices_.push_back(right_child_indices);
}

bool treeHistogramConsistencyProtocol::finalize()
{
    begin_finalize("treeHistogramConsistencyProtocol");
    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        const treeHistogramConsistencyCheckRecord& record = verifier_.checks[i];
        std::vector<virCommitment> left_histograms;
        std::vector<virCommitment> right_histograms;
        left_histograms.reserve(record.channel_count);
        right_histograms.reserve(record.channel_count);
        for (size_t channel = 0; channel < record.channel_count; ++channel) {
            left_histograms.emplace_back(record.left_histogram_commitments[channel]);
            right_histograms.emplace_back(record.right_histogram_commitments[channel]);
        }

        if (record.uses_parent_mask) {
            auto add_masked_route = [&](const virCommitment& child_indices,
                                        const std::vector<virCommitment>& child_histograms) {
                if (record.enforce_parent_mask_boolean) {
                    params_.index_routing->add_masked_check(child_indices,
                                                            parent_masks_[i],
                                                            full_histogram_batches_[i],
                                                            child_histograms,
                                                            record.node_count,
                                                            record.node_count,
                                                            record.histogram_width);
                } else {
                    params_.index_routing->add_masked_check_with_certified_mask(child_indices,
                                                                                parent_masks_[i],
                                                                                full_histogram_batches_[i],
                                                                                child_histograms,
                                                                                record.node_count,
                                                                                record.node_count,
                                                                                record.histogram_width);
                }
            };
            add_masked_route(left_child_indices_[i], left_histograms);
            add_masked_route(right_child_indices_[i], right_histograms);
        } else {
            params_.index_routing->add_check(left_child_indices_[i],
                                             full_histogram_batches_[i],
                                             left_histograms,
                                             record.node_count,
                                             record.node_count,
                                             record.histogram_width);
            params_.index_routing->add_check(right_child_indices_[i],
                                             full_histogram_batches_[i],
                                             right_histograms,
                                             record.node_count,
                                             record.node_count,
                                             record.histogram_width);
        }

        for (size_t channel = 0; channel < record.channel_count; ++channel) {
            const virCommitment left_histogram = left_histograms[channel];
            const virCommitment right_histogram = right_histograms[channel];

            virCommitment decomposition_zero =
                full_histogram_batches_[i][channel] - leaf_histogram_batches_[i][channel] - left_histogram - right_histogram;
            params_.zero_check->add_check(decomposition_zero,
                                          record.node_count * record.histogram_width,
                                          "tree_histogram_consistency.decomposition[" + std::to_string(i) + "," +
                                              std::to_string(channel) + "]");
        }
    }

    return verifier_.finalize();
}
