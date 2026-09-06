#pragma once

#include "forest_training_trace_adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

class zkboostSharedValidateForestSummary {
public:
    size_t tree_count = 0;
    size_t sample_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t max_depth = 0;
    size_t checked_paths = 0;
    size_t split_node_count = 0;
    size_t leaf_node_count = 0;
};

class zkboostSharedForestPathClaim {
public:
    size_t tree = 0;
    size_t sample = 0;
    std::vector<std::int64_t> path_features;
    std::vector<std::int64_t> path_thresholds;
    std::vector<std::int64_t> permuted_inputs;
    std::vector<std::int64_t> path_branch_bits;
    std::int64_t trace_leaf_index = 0;
    std::int64_t logical_leaf_id = 0;
    std::int64_t tree_score = 0;
};

class zkboostSharedForestSplitFieldReadClaim {
public:
    size_t node = 0;
    size_t tree = 0;
    size_t internal_node_index = 0;
    std::int64_t feature = 0;
    std::int64_t threshold = 0;
};

class zkboostSharedForestPathReadClaim {
public:
    size_t query = 0;
    size_t tree = 0;
    size_t sample = 0;
    size_t step = 0;
    std::int64_t logical_leaf_id = 0;
    size_t internal_node_index = 0;
    std::int64_t feature = 0;
    std::int64_t threshold = 0;
};

class zkboostSharedForestLeafValueReadClaim {
public:
    size_t query = 0;
    size_t tree = 0;
    size_t sample = 0;
    std::int64_t logical_leaf_id = 0;
    std::int64_t tree_score = 0;
};

class zkboostSharedValidateForestRoutingView {
public:
    zkboostSharedValidateForestSummary summary;
    std::vector<zkboostSharedForestPathClaim> path_claims;
    std::vector<std::int64_t> public_query_inputs;
    std::vector<std::int64_t> trace_leaf_indices;
    std::vector<std::int64_t> logical_leaf_ids;
    std::vector<std::int64_t> tree_scores;
};

class zkboostSharedValidateForestFetchView {
public:
    zkboostSharedValidateForestSummary summary;
    std::vector<std::int64_t> public_split_masks;
    std::vector<std::int64_t> public_split_features;
    std::vector<std::int64_t> public_split_thresholds;
    std::vector<zkboostSharedForestSplitFieldReadClaim> split_field_reads;
    std::vector<zkboostSharedForestPathReadClaim> path_reads;
    std::vector<zkboostSharedForestLeafValueReadClaim> leaf_value_reads;
};

class zkboostSharedValidateForestView {
public:
    zkboostSharedValidateForestSummary summary;
    zkboostSharedValidateForestRoutingView routing;
    zkboostSharedValidateForestFetchView fetch;
};

zkboostSharedValidateForestRoutingView build_zkboost_shared_validate_forest_routing_view(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);

zkboostSharedValidateForestFetchView build_zkboost_shared_validate_forest_fetch_view(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);

zkboostSharedValidateForestView build_zkboost_shared_validate_forest_view(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);

zkboostSharedValidateForestSummary validate_zkboost_shared_forest(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);
