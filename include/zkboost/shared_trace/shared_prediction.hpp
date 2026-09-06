#pragma once

#include "forest_training_trace_adapter.hpp"

#include <cstdint>
#include <string>
#include <vector>

class zkboostSharedTracePrediction {
public:
    std::vector<std::int64_t> predicted_labels;
    std::vector<std::int64_t> actual_labels;
    std::vector<std::int64_t> tree_scores;
    std::vector<std::int64_t> class_scores;
    std::vector<std::int64_t> predicted_scores;
    std::vector<std::int64_t> class_selector;
    std::string prediction_digest;
    size_t accuracy_numerator = 0;
    size_t accuracy_denominator = 0;
};

// Plain, shared-trace prediction and witness derivation. This function does not
// build a proof by itself; the CertXGB prediction PLONK layer constrains the
// returned tree scores, class sums, selector, winning scores, and labels. When
// `inference.sample_indices` is present, predictions and labels are evaluated on
// that logical subset/order of the training dataset.
zkboostSharedTracePrediction derive_zkboost_shared_trace_prediction(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);
