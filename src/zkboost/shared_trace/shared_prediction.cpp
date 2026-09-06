#include "zkboost/shared_trace/shared_prediction.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_trace_prediction: " + message);
    }
}

size_t checked_size(std::int64_t value, const std::string& context)
{
    require_condition(value >= 0, context + " must be non-negative");
    return static_cast<size_t>(value);
}

std::int64_t checked_add(std::int64_t left,
                         std::int64_t right,
                         const std::string& context)
{
    if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
        throw std::runtime_error("zkboost_shared_trace_prediction: " + context +
                                 " exceeds int64 range");
    }
    return left + right;
}

size_t selected_feature_for_node(const forestTrainingSemanticTrace& trace, size_t node)
{
    size_t selected = trace.feature_count;
    for (size_t feature = 0; feature < trace.feature_count; ++feature) {
        const size_t offset = node * trace.feature_count + feature;
        const std::int64_t value = trace.feature_selector.at(offset);
        require_condition(value == 0 || value == 1, "feature_selector must be binary");
        if (value == 0) {
            continue;
        }
        require_condition(selected == trace.feature_count, "split node has multiple selected features");
        selected = feature;
    }
    require_condition(selected < trace.feature_count, "split node has no selected feature");
    return selected;
}

std::vector<std::int64_t> derive_actual_labels(const forestTrainingSemanticTrace& trace)
{
    require_condition(trace.base_labels.size() == trace.class_count * trace.query_count,
                      "base_labels length mismatch");
    std::vector<std::int64_t> labels(trace.query_count, 0);
    for (size_t sample = 0; sample < trace.query_count; ++sample) {
        size_t selected = trace.class_count;
        size_t column_sum = 0;
        for (size_t cls = 0; cls < trace.class_count; ++cls) {
            const std::int64_t value = trace.base_labels[cls * trace.query_count + sample];
            require_condition(value == 0 || value == 1, "base_labels must be binary");
            if (value == 1) {
                selected = cls;
                ++column_sum;
            }
        }
        require_condition(column_sum == 1, "base_labels column must be one-hot");
        labels[sample] = static_cast<std::int64_t>(selected);
    }
    return labels;
}

size_t training_sample_index_for_inference_sample(const forestTrainingSemanticTrace& training,
                                                  const forestSemanticInferenceQueryBatch& inference,
                                                  size_t sample_count,
                                                  size_t inference_sample)
{
    if (inference.sample_indices.empty()) {
        require_condition(inference_sample < training.query_count,
                          "inference sample exceeds training query_count");
        return inference_sample;
    }
    require_condition(inference.sample_indices.size() == sample_count,
                      "inference sample_indices length mismatch");
    const size_t training_sample =
        checked_size(inference.sample_indices.at(inference_sample), "sample_indices");
    require_condition(training_sample < training.query_count,
                      "sample_indices entry exceeds training query_count");
    return training_sample;
}

void validate_shapes(const forestTrainingSemanticTrace& training,
                     const forestSemanticInferenceQueryBatch& inference,
                     size_t sample_count)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(training.iteration_count > 0, "iteration_count must be non-zero");
    require_condition(training.class_count > 0, "class_count must be non-zero");
    require_condition(training.tree_count == training.iteration_count * training.class_count,
                      "tree_count must equal iteration_count * class_count");
    require_condition(training.feature_count > 0, "feature_count must be non-zero");
    require_condition(inference.feature_count == training.feature_count, "inference feature_count mismatch");
    require_condition(inference.query_count == training.tree_count * sample_count,
                      "inference query_count must equal tree_count * sample_count");
    require_condition(sample_count <= training.query_count,
                      "inference sample_count exceeds training query_count");
    require_condition(training.root_indices.size() == training.tree_count, "root_indices length mismatch");
    require_condition(training.split_node_mask.size() == training.node_count, "split_node_mask length mismatch");
    require_condition(training.left_child_indices.size() == training.node_count, "left_child_indices length mismatch");
    require_condition(training.right_child_indices.size() == training.node_count, "right_child_indices length mismatch");
    require_condition(training.node_tree_ids.size() == training.node_count, "node_tree_ids length mismatch");
    require_condition(training.feature_selector.size() == training.node_count * training.feature_count,
                      "feature_selector length mismatch");
    require_condition(training.thresholds.size() == training.node_count, "thresholds length mismatch");
    require_condition(training.node_leaf_mask.size() == training.node_count, "node_leaf_mask length mismatch");
    require_condition(training.node_leaf_values.size() == training.node_count, "node_leaf_values length mismatch");
    require_condition(training.training_inputs.size() == training.query_count * training.feature_count,
                      "training_inputs length mismatch");
    require_condition(inference.query_tree_ids.size() == inference.query_count, "query_tree_ids length mismatch");
    require_condition(inference.lifted_inputs.size() == inference.query_count * inference.feature_count,
                      "lifted_inputs length mismatch");
}

void validate_inference_matches_training_rows(const forestTrainingSemanticTrace& training,
                                              const forestSemanticInferenceQueryBatch& inference,
                                              size_t sample_count)
{
    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        for (size_t sample = 0; sample < sample_count; ++sample) {
            const size_t query = tree * sample_count + sample;
            const size_t training_sample =
                training_sample_index_for_inference_sample(training, inference, sample_count, sample);
            require_condition(checked_size(inference.query_tree_ids[query], "query_tree_ids") == tree + 1,
                              "inference queries must be grouped by tree");
            for (size_t feature = 0; feature < training.feature_count; ++feature) {
                const size_t lifted_offset = query * training.feature_count + feature;
                const size_t training_offset = training_sample * training.feature_count + feature;
                require_condition(inference.lifted_inputs[lifted_offset] == training.training_inputs[training_offset],
                                  "inference lifted_inputs must match training_inputs for shared accuracy");
            }
        }
    }
}

size_t route_sample(const forestTrainingSemanticTrace& trace,
                    const forestSemanticInferenceQueryBatch& inference,
                    size_t query,
                    size_t tree)
{
    size_t node = checked_size(trace.root_indices.at(tree), "root_indices");
    require_condition(node < trace.node_count, "root index out of range");
    for (size_t step = 0; step <= trace.node_count; ++step) {
        require_condition(checked_size(trace.node_tree_ids.at(node), "node_tree_ids") == tree + 1,
                          "routing left the tree");
        const std::int64_t split = trace.split_node_mask.at(node);
        const std::int64_t leaf = trace.node_leaf_mask.at(node);
        require_condition(split == 0 || split == 1, "split_node_mask must be binary");
        require_condition(leaf == 0 || leaf == 1, "node_leaf_mask must be binary");
        require_condition(!(split == 1 && leaf == 1), "node cannot be both split and leaf");
        if (split == 0) {
            require_condition(leaf == 1, "routing reached an inactive non-leaf node");
            return node;
        }

        const size_t feature = selected_feature_for_node(trace, node);
        const std::int64_t value = inference.lifted_inputs.at(query * trace.feature_count + feature);
        const std::int64_t threshold = trace.thresholds.at(node);
        const size_t next = checked_size(
            value < threshold ? trace.left_child_indices.at(node) : trace.right_child_indices.at(node),
            "child index");
        require_condition(next < trace.node_count, "child index out of range");
        node = next;
    }
    throw std::runtime_error("zkboost_shared_trace_prediction: routing cycle detected");
}

std::string digest_predictions(const std::vector<std::int64_t>& labels)
{
    constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    std::uint64_t hash = kFnvOffset;
    auto absorb = [&](std::uint64_t value) {
        for (size_t byte = 0; byte < 8; ++byte) {
            hash ^= (value >> (byte * 8)) & 0xffu;
            hash *= kFnvPrime;
        }
    };
    absorb(labels.size());
    for (std::int64_t label : labels) {
        absorb(static_cast<std::uint64_t>(label));
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

} // namespace

zkboostSharedTracePrediction derive_zkboost_shared_trace_prediction(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(inference.query_count % training.tree_count == 0,
                      "inference query_count must be divisible by tree_count");
    const size_t sample_count = inference.query_count / training.tree_count;
    validate_shapes(training, inference, sample_count);
    validate_inference_matches_training_rows(training, inference, sample_count);

    zkboostSharedTracePrediction out;
    const std::vector<std::int64_t> all_actual_labels = derive_actual_labels(training);
    out.actual_labels.reserve(sample_count);
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const size_t training_sample =
            training_sample_index_for_inference_sample(training, inference, sample_count, sample);
        out.actual_labels.push_back(all_actual_labels.at(training_sample));
    }
    out.tree_scores.assign(training.tree_count * sample_count, 0);
    out.class_scores.assign(training.class_count * sample_count, 0);

    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        const size_t cls = tree % training.class_count;
        for (size_t sample = 0; sample < sample_count; ++sample) {
            const size_t query = tree * sample_count + sample;
            const size_t leaf = route_sample(training, inference, query, tree);
            const std::int64_t score = training.node_leaf_values.at(leaf);
            out.tree_scores[query] = score;
            out.class_scores[cls * sample_count + sample] =
                checked_add(out.class_scores[cls * sample_count + sample],
                            score,
                            "class score aggregation");
        }
    }

    out.predicted_labels.assign(sample_count, 0);
    out.predicted_scores.assign(sample_count, 0);
    out.class_selector.assign(training.class_count * sample_count, 0);
    for (size_t sample = 0; sample < sample_count; ++sample) {
        size_t best_class = 0;
        std::int64_t best_score = out.class_scores[sample];
        for (size_t cls = 1; cls < training.class_count; ++cls) {
            const std::int64_t score = out.class_scores[cls * sample_count + sample];
            if (score > best_score) {
                best_score = score;
                best_class = cls;
            }
        }
        out.predicted_labels[sample] = static_cast<std::int64_t>(best_class);
        out.predicted_scores[sample] = best_score;
        out.class_selector[best_class * sample_count + sample] = 1;
    }

    out.accuracy_denominator = sample_count;
    for (size_t sample = 0; sample < sample_count; ++sample) {
        if (out.predicted_labels[sample] == out.actual_labels[sample]) {
            ++out.accuracy_numerator;
        }
    }
    out.prediction_digest = digest_predictions(out.predicted_labels);
    return out;
}
