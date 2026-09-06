#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "forest_committed_inference_workflow.hpp"
#include "forest_training_loss.hpp"
#include "forest_training_then_inference_io.hpp"

class forestSemanticInferenceQueryBatch {
public:
    std::vector<std::int64_t> query_tree_ids;
    std::vector<std::int64_t> lifted_inputs;
    // Optional logical-sample to training-sample map used by standalone
    // inference workflows. When empty, consumers interpret inference sample
    // `i` as training sample `i`.
    std::vector<std::int64_t> sample_indices;
    size_t query_count = 0;
    size_t feature_count = 0;
};

// Higher-level semantic trace for the current committed-forest training route.
//
// This format is intentionally closer to application data than
// `forestTrainingProofInputs`:
// - the dataset is stored once as a `query_count x feature_count` matrix,
// - the committed forest carries one `node_leaf_values` vector rather than
//   pre-expanded per-query `tree_scores`,
// - external inputs provide tree structure plus split rules rather than
//   explicit hyper-box encodings,
// - the trace stores one inferred `leaf_indices` vector over `(tree, sample)`,
// - and the adapter derives:
//   node/child/leaf boxes,
//   training query expansion,
//   tree scores,
//   bucket addresses,
//   count / pseudo-residual / Hessian histogram channels,
//   split statistics,
//   exact + quantized gains,
//   selection witnesses,
//   leaf-value witnesses,
//   and the global boosting-state witnesses.
class forestTrainingSemanticTrace {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t query_count = 0;
    size_t input_bit_bound = 0;
    size_t full_upper_bound = 0;
    size_t structure_bit_bound = 0;
    fixedPointEncoding gradient_encoding;
    fixedPointEncoding gain_numerator_encoding;
    fixedPointEncoding quantized_gain_encoding;
    fixedPointEncoding leaf_value_encoding;
    fixedPointEncoding score_encoding;
    fixedPointEncoding softmax_encoding;
    size_t hessian_bit_bound = 0;
    size_t gain_denominator_bit_bound = 0;
    size_t leaf_denominator_bit_bound = 0;
    size_t gain_gap_bit_bound = 0;
    forestTrainingLoss loss_mode = forestTrainingLoss::FirstOrderRss;
    size_t l2_regularization = 0;
    size_t learning_rate_numerator = 1;
    size_t learning_rate_denominator = 1;
    Fr split_penalty = Fr(0);
    Fr split_gain_threshold = Fr(0);
    // One-based maximum node depth: roots have depth 1 and every active leaf
    // must have depth at most max_depth.
    size_t max_depth = 0;

    std::vector<std::int64_t> root_indices;
    std::vector<std::int64_t> split_node_mask;
    std::vector<std::int64_t> left_child_indices;
    std::vector<std::int64_t> right_child_indices;
    std::vector<std::int64_t> node_tree_ids;
    std::vector<std::int64_t> feature_selector;
    std::vector<std::int64_t> thresholds;
    std::vector<std::int64_t> node_leaf_mask;
    std::vector<std::int64_t> node_leaf_values;

    std::vector<std::int64_t> training_inputs;
    std::vector<std::int64_t> base_labels;
    std::vector<std::int64_t> leaf_indices;
    std::vector<std::int64_t> candidate_mask;
    std::vector<std::int64_t> selected_candidate_indices;
};

forestInferenceQueryBatch assemble_forest_inference_query_batch_from_trace(
    const forestSemanticInferenceQueryBatch& trace);

forestTrainingProofInputs assemble_forest_training_inputs_from_trace(
    const forestTrainingSemanticTrace& trace);

forestCommittedInferenceInputs assemble_forest_committed_inference_inputs_from_trace(
    const forestTrainingSemanticTrace& trace,
    const forestSemanticInferenceQueryBatch& inference_queries);

forestTrainingThenInferenceInputs assemble_forest_training_then_inference_inputs_from_trace(
    const forestTrainingSemanticTrace& trace,
    const forestSemanticInferenceQueryBatch& inference_queries);

void write_forest_training_then_inference_case_from_trace(
    const std::string& path,
    const forestTrainingSemanticTrace& trace,
    const forestSemanticInferenceQueryBatch& inference_queries);
