#pragma once

#include "zkboost/inference/trace.hpp"
#include "zkboost/training/trace.hpp"

#include <string>

class zkboostTraceSummary {
public:
    size_t round_count = 0;
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t leaf_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t training_query_count = 0;
    size_t inference_query_count = 0;
};

void validate_zkboost_training_trace(const zkboostTrainingTrace& trace);
void validate_zkboost_inference_trace(const zkboostInferenceTrace& trace);

zkboostTraceSummary summarize_zkboost_traces(const zkboostTrainingTrace& training,
                                             const zkboostInferenceTrace& inference);

std::string zkboost_trace_summary_description(const zkboostTraceSummary& summary);
std::string zkboost_trace_format_description();

void write_zkboost_training_trace(const std::string& path, const zkboostTrainingTrace& trace);
zkboostTrainingTrace read_zkboost_training_trace(const std::string& path);

void write_zkboost_inference_trace(const std::string& path, const zkboostInferenceTrace& trace);
zkboostInferenceTrace read_zkboost_inference_trace(const std::string& path);
