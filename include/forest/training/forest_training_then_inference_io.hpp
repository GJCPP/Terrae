#pragma once

#include <string>

#include "forest_training_then_inference_workflow.hpp"

class forestTrainingThenInferenceCaseSummary {
public:
    size_t training_tree_count = 0;
    size_t training_node_count = 0;
    size_t training_feature_count = 0;
    size_t training_query_count = 0;
    size_t inference_query_count = 0;
    size_t exported_leaf_row_count = 0;
    bool explicit_inference_witness_provided = false;
};

// Validates the outer application-facing inputs for the combined
// training-then-inference workflow and summarizes the single exported-inference
// proof implied by the committed forest.
forestTrainingThenInferenceCaseSummary summarize_forest_training_then_inference_inputs(
    const forestTrainingThenInferenceInputs& inputs);

void validate_forest_training_then_inference_inputs(const forestTrainingThenInferenceInputs& inputs);

// Returns a human-readable description of the plain-text case format used by
// write/read_forest_training_then_inference_case().
std::string forest_training_then_inference_case_format_description();

// Returns a human-readable description of the plain-text bundle format used by
// write/read_forest_training_then_inference_bundle().
std::string forest_training_then_inference_bundle_format_description();

void write_forest_training_then_inference_case(const std::string& path,
                                               const forestTrainingThenInferenceInputs& inputs);

forestTrainingThenInferenceInputs read_forest_training_then_inference_case(const std::string& path);

void write_forest_training_then_inference_bundle(const std::string& path,
                                                 const forestTrainingThenInferenceArtifact& artifact);

forestTrainingThenInferenceArtifact read_forest_training_then_inference_bundle(const std::string& path);
