#pragma once

#include "commitments/kzg.hpp"
#include "forest_training_trace_adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

class zkboostSharedTreeEnsemble {
public:
    size_t tree_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t max_depth = 0;
    size_t internal_node_count_per_tree = 0;
    size_t leaf_count_per_tree = 0;

    // Paper-aligned logical model object:
    // T = { (f_k, t_k, w_k) }_{k in [m]}
    // encoded as canonical full binary trees with dummy splits on pruned
    // subtrees. Vectors are flattened tree-major.
    std::vector<std::int64_t> split_features;
    std::vector<std::int64_t> split_thresholds;
    std::vector<std::int64_t> leaf_values;
};

class zkboostSharedForestColumns {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;

    // Explicit node-table view derived from one shared training trace.
    // This is no longer the logical `com_T`; it is a transitional auxiliary
    // artifact used by the current routing/fetch proof backend.
    std::vector<std::int64_t> root_indices;
    std::vector<std::int64_t> tree_ids;
    std::vector<std::int64_t> split_masks;
    std::vector<std::int64_t> leaf_masks;
    std::vector<std::int64_t> split_features;
    std::vector<std::int64_t> thresholds;
    std::vector<std::int64_t> left_children;
    std::vector<std::int64_t> right_children;
    std::vector<std::int64_t> leaf_values;
};

class zkboostSharedForestCommitments {
public:
    // Logical commitments for the paper-style forest object
    // com_T <=> Enc_T(f, t, w).
    Group1 com_f;
    Group1 com_t;
    Group1 com_w;
};

class zkboostSharedKzgSrsRequirement {
public:
    size_t deg1_plus_1 = 1;
    size_t deg2_plus_1 = 1;
    size_t min_degree_bound_plus_1 = 0;
    size_t opening_deg2_plus_1_floor = 1;
};

class zkboostSharedCommittedForestSummary {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t max_depth = 0;
    size_t internal_node_count_per_tree = 0;
    size_t leaf_count_per_tree = 0;
    size_t split_domain_size = 0;
    size_t leaf_domain_size = 0;
    size_t commitment_count = 0;
    size_t commitment_bytes = 0;
};

class zkboostSharedCommittedForest {
public:
    zkboostSharedTreeEnsemble ensemble;
    zkboostSharedForestCommitments commitments;
    zkboostSharedCommittedForestSummary summary;
};

void validate_zkboost_shared_tree_ensemble(const zkboostSharedTreeEnsemble& ensemble);
void validate_zkboost_shared_forest_columns(const zkboostSharedForestColumns& forest);

zkboostSharedTreeEnsemble shared_tree_ensemble_from_training_trace(
    const forestTrainingSemanticTrace& training);

zkboostSharedForestColumns shared_forest_columns_from_training_trace(
    const forestTrainingSemanticTrace& training);

size_t canonical_padded_zkboost_shared_forest_length(size_t raw_length);

zkboostSharedKzgSrsRequirement max_zkboost_shared_kzg_srs_requirement(
    const zkboostSharedKzgSrsRequirement& left,
    const zkboostSharedKzgSrsRequirement& right);

zkboostSharedKzgSrsRequirement required_zkboost_shared_forest_commitment_srs_requirement(
    const zkboostSharedTreeEnsemble& ensemble);

zkboostSharedKzgSrsRequirement required_zkboost_shared_forest_commitment_srs_requirement(
    const zkboostSharedForestColumns& forest);

zkboostSharedKzgSrsRequirement required_zkboost_shared_forest_commitment_srs_requirement_from_training_trace(
    const forestTrainingSemanticTrace& training);

size_t required_zkboost_shared_forest_commitment_srs_size(
    const zkboostSharedTreeEnsemble& ensemble);

size_t required_zkboost_shared_forest_commitment_srs_size(
    const zkboostSharedForestColumns& forest);

size_t required_zkboost_shared_forest_commitment_srs_size_from_training_trace(
    const forestTrainingSemanticTrace& training);

std::vector<Fr> encode_zkboost_shared_forest_column(
    const std::vector<std::int64_t>& values);

std::vector<Fr> encode_zkboost_shared_forest_column_padded(
    const std::vector<std::int64_t>& values);

zkboostSharedForestCommitments commit_zkboost_shared_forest(
    const kzgPublicParameters& public_params,
    const zkboostSharedTreeEnsemble& ensemble);

zkboostSharedForestCommitments commit_zkboost_shared_forest_from_training_trace(
    const kzgPublicParameters& public_params,
    const forestTrainingSemanticTrace& training);

zkboostSharedCommittedForest build_zkboost_shared_committed_forest(
    const kzgPublicParameters& public_params,
    const forestTrainingSemanticTrace& training);
