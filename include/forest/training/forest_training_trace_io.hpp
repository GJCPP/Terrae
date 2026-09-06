#pragma once

#include <string>

#include "forest_training_trace_adapter.hpp"

class forestTrainingSemanticTraceCase {
public:
    forestTrainingSemanticTrace training_trace;
    forestSemanticInferenceQueryBatch inference_queries;
};

void write_forest_training_semantic_trace(const std::string& path,
                                          const forestTrainingSemanticTrace& trace);

forestTrainingSemanticTrace read_forest_training_semantic_trace(const std::string& path);

void write_forest_semantic_inference_query_trace(const std::string& path,
                                                 const forestSemanticInferenceQueryBatch& queries);

forestSemanticInferenceQueryBatch read_forest_semantic_inference_query_trace(const std::string& path);

class forestTrainingSemanticTraceCaseSummary {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t class_count = 0;
    size_t training_query_count = 0;
    size_t inference_query_count = 0;
    size_t trace_leaf_assignment_count = 0;
    size_t trace_candidate_count = 0;
    size_t active_split_node_count = 0;
    size_t active_candidate_count = 0;
    size_t leaf_node_count = 0;
    size_t exported_leaf_row_count = 0;
    size_t derived_training_query_row_count = 0;
    size_t derived_bucket_address_count = 0;
    size_t derived_boosting_entry_count = 0;
};

forestTrainingSemanticTraceCaseSummary summarize_forest_training_semantic_trace_case(
    const forestTrainingSemanticTraceCase& trace_case);

void validate_forest_training_semantic_trace_case(const forestTrainingSemanticTraceCase& trace_case);

std::string forest_training_semantic_trace_case_format_description();

std::string forest_training_semantic_trace_contract_description();

std::string forest_training_semantic_trace_case_summary_description(
    const forestTrainingSemanticTraceCaseSummary& summary);

void write_forest_training_semantic_trace_case(const std::string& path,
                                               const forestTrainingSemanticTraceCase& trace_case);

forestTrainingSemanticTraceCase read_forest_training_semantic_trace_case(const std::string& path);

forestTrainingSemanticTraceCase read_forest_training_semantic_trace_case(
    const std::string& training_trace_path,
    const std::string& inference_trace_path);

forestTrainingThenInferenceInputs read_forest_training_then_inference_inputs_from_trace_case(
    const std::string& path);

forestTrainingThenInferenceInputs read_forest_training_then_inference_inputs_from_split_traces(
    const std::string& training_trace_path,
    const std::string& inference_trace_path);
