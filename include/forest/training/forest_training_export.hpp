#pragma once

#include <optional>
#include <string>

#include "forest_committed_forest_public.hpp"
#include "forest_classification.hpp"
#include "forest_inference.hpp"
#include "forest_training_workflow.hpp"

class forestCommittedNodeTable {
public:
    poly root_indices;
    poly split_node_mask;
    poly node_depths;
    poly left_child_indices;
    poly right_child_indices;
    poly node_tree_ids;
    poly feature_selector;
    poly lower_bounds;
    poly upper_bounds;
    poly thresholds;
    poly node_leaf_mask;
    poly node_leaf_values;
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
};

class forestExportedLeafTable {
public:
    poly tree_indices;
    poly lower_bounds;
    poly upper_bounds;
    poly scores;
    poly leaf_indices;
    size_t row_count = 0;
    size_t feature_count = 0;
};

forestCommittedNodeTable forest_committed_node_table_from_training_inputs(const forestTrainingProofInputs& inputs);

kzgSrsRequirement required_committed_forest_public_commitments_srs(
    const forestCommittedNodeTable& committed_forest);

kzgSrsRequirement required_committed_forest_public_commitments_srs(
    const forestTrainingProofInputs& inputs);

forestCommittedNodePublicCommitments commit_committed_forest_public_commitments(
    const kzgPublicParameters& public_params,
    const forestCommittedNodeTable& committed_forest,
    const std::string& key_prefix = "committed_forest");

forestExportedLeafTable export_leaf_table_from_committed_forest(const forestCommittedNodeTable& committed_forest);

forestInferencePublicTable build_forest_inference_public_table_from_exported_forest(
    const kzgPublicParameters& public_params,
    std::string key_prefix,
    const forestExportedLeafTable& table,
    size_t input_bit_bound,
    std::optional<size_t> degree_bound = std::nullopt);

// Build only verifier-visible table metadata. No SRS or PCS operation is
// performed; this is used by the exact soundness counter.
forestClassificationPublicTable build_forest_classification_public_table_metadata_from_exported_forest(
    std::string key_prefix,
    const forestExportedLeafTable& table,
    size_t input_bit_bound,
    size_t iteration_count,
    size_t class_count,
    size_t score_bit_bound,
    std::optional<size_t> degree_bound = std::nullopt);

forestClassificationPublicTable build_forest_classification_public_table_from_exported_forest(
    const kzgPublicParameters& public_params,
    std::string key_prefix,
    const forestExportedLeafTable& table,
    size_t input_bit_bound,
    size_t iteration_count,
    size_t class_count,
    size_t score_bit_bound,
    std::optional<size_t> degree_bound = std::nullopt);
