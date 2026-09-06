#include "zkboost/shared_trace/validate_forest.hpp"

#include "zkboost/shared_trace/forest_commitments.hpp"
#include "zkboost/shared_trace/shared_prediction.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_validate_forest: " + message);
    }
}

size_t checked_size(std::int64_t value, const std::string& context)
{
    require_condition(value >= 0, context + " must be non-negative");
    return static_cast<size_t>(value);
}

size_t node_feature_offset(const forestTrainingSemanticTrace& trace, size_t node, size_t feature)
{
    return node * trace.feature_count + feature;
}

size_t training_input_offset(const forestTrainingSemanticTrace& trace, size_t sample, size_t feature)
{
    return sample * trace.feature_count + feature;
}

size_t lifted_input_offset(const forestTrainingSemanticTrace& trace, size_t query, size_t feature)
{
    return query * trace.feature_count + feature;
}

size_t tree_sample_offset(size_t sample_count, size_t tree, size_t sample)
{
    return tree * sample_count + sample;
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

size_t logical_leaf_bit(std::int64_t logical_leaf_id, size_t max_depth, size_t step)
{
    require_condition(logical_leaf_id > 0, "logical leaf id must be positive");
    const size_t shift = max_depth - 1 - step;
    return ((logical_leaf_id - 1) & (std::int64_t(1) << shift)) != 0 ? 1 : 0;
}

size_t rightmost_descendant_leaf_heap_index(size_t heap_index, size_t depth, size_t max_depth)
{
    size_t out = heap_index;
    while (depth < max_depth) {
        out = (out * 2) + 2;
        ++depth;
    }
    return out;
}

size_t selected_feature_for_split_node(const forestTrainingSemanticTrace& trace, size_t node)
{
    size_t selected = trace.feature_count;
    for (size_t feature = 0; feature < trace.feature_count; ++feature) {
        const std::int64_t value = trace.feature_selector.at(node_feature_offset(trace, node, feature));
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

void validate_shapes(const forestTrainingSemanticTrace& training,
                     const forestSemanticInferenceQueryBatch& inference,
                     size_t sample_count)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(training.node_count > 0, "node_count must be non-zero");
    require_condition(training.feature_count > 0, "feature_count must be non-zero");
    require_condition(training.bin_count > 0, "bin_count must be non-zero");
    require_condition(training.class_count > 0, "class_count must be non-zero");
    require_condition(sample_count <= training.query_count,
                      "inference sample_count exceeds training query_count");
    require_condition(inference.feature_count == training.feature_count, "inference feature_count mismatch");
    require_condition(inference.query_count == training.tree_count * sample_count,
                      "inference query_count must equal tree_count * sample_count");

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
    require_condition(training.leaf_indices.size() == training.tree_count * training.query_count,
                      "leaf_indices length mismatch");
    require_condition(inference.query_tree_ids.size() == inference.query_count, "query_tree_ids length mismatch");
    require_condition(inference.lifted_inputs.size() == inference.query_count * inference.feature_count,
                      "lifted_inputs length mismatch");
}

void validate_node_roles_and_structure(const forestTrainingSemanticTrace& trace,
                                       zkboostSharedValidateForestSummary& summary)
{
    for (size_t tree = 0; tree < trace.tree_count; ++tree) {
        const size_t root = checked_size(trace.root_indices.at(tree), "root_indices");
        require_condition(root < trace.node_count, "root index out of range");
        require_condition(checked_size(trace.node_tree_ids.at(root), "node_tree_ids") == tree + 1,
                          "root node belongs to the wrong tree");
    }

    for (size_t node = 0; node < trace.node_count; ++node) {
        const size_t tree = checked_size(trace.node_tree_ids.at(node), "node_tree_ids");

        const std::int64_t split = trace.split_node_mask.at(node);
        const std::int64_t leaf = trace.node_leaf_mask.at(node);
        require_condition(split == 0 || split == 1, "split_node_mask must be binary");
        require_condition(leaf == 0 || leaf == 1, "node_leaf_mask must be binary");
        require_condition(!(split == 1 && leaf == 1), "node cannot be both split and leaf");
        if (split == 0 && leaf == 0) {
            require_condition(tree == 0, "dummy node must use tree ID 0");
            continue;
        }
        require_condition(tree >= 1 && tree <= trace.tree_count, "real node tree ID out of range");

        if (split == 1) {
            ++summary.split_node_count;
            (void)selected_feature_for_split_node(trace, node);
            const size_t left = checked_size(trace.left_child_indices.at(node), "left_child_indices");
            const size_t right = checked_size(trace.right_child_indices.at(node), "right_child_indices");
            require_condition(left < trace.node_count, "left child index out of range");
            require_condition(right < trace.node_count, "right child index out of range");
            require_condition(checked_size(trace.node_tree_ids.at(left), "node_tree_ids") == tree,
                              "left child belongs to the wrong tree");
            require_condition(checked_size(trace.node_tree_ids.at(right), "node_tree_ids") == tree,
                              "right child belongs to the wrong tree");
            require_condition(trace.thresholds.at(node) >= 0, "split threshold must be non-negative");
            require_condition(static_cast<size_t>(trace.thresholds.at(node)) < trace.bin_count,
                              "split threshold exceeds bin_count");
        } else if (leaf == 1) {
            ++summary.leaf_node_count;
        }
    }
}

void validate_inference_rows(const forestTrainingSemanticTrace& training,
                             const forestSemanticInferenceQueryBatch& inference,
                             size_t sample_count)
{
    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        for (size_t sample = 0; sample < sample_count; ++sample) {
            const size_t query = tree_sample_offset(sample_count, tree, sample);
            const size_t training_sample =
                training_sample_index_for_inference_sample(training, inference, sample_count, sample);
            require_condition(checked_size(inference.query_tree_ids.at(query), "query_tree_ids") == tree + 1,
                              "inference queries must be grouped by tree");
            for (size_t feature = 0; feature < training.feature_count; ++feature) {
                const std::int64_t value = inference.lifted_inputs.at(lifted_input_offset(training, query, feature));
                require_condition(value >= 0, "inference lifted input must be non-negative");
                require_condition(static_cast<size_t>(value) < training.bin_count,
                                  "inference lifted input exceeds bin_count");
                require_condition(value ==
                                      training.training_inputs.at(
                                          training_input_offset(training, training_sample, feature)),
                                  "inference lifted_inputs must match training_inputs");
            }
        }
    }
}

class sharedTraceForestEmbedding {
public:
    explicit sharedTraceForestEmbedding(size_t node_count)
        : split_internal_indices(node_count, std::numeric_limits<size_t>::max())
        , leaf_logical_ids(node_count, 0)
    {
    }

    std::vector<size_t> split_internal_indices;
    std::vector<std::int64_t> leaf_logical_ids;
};

void embed_trace_node(const forestTrainingSemanticTrace& training,
                      const zkboostSharedTreeEnsemble& ensemble,
                      sharedTraceForestEmbedding& embedding,
                      size_t tree,
                      size_t node,
                      size_t heap_index,
                      size_t depth)
{
    require_condition(node < training.node_count, "embedded node index out of range");
    require_condition(checked_size(training.node_tree_ids.at(node), "node_tree_ids") == tree + 1,
                      "embedded node belongs to the wrong tree");

    const std::int64_t split = training.split_node_mask.at(node);
    const std::int64_t leaf = training.node_leaf_mask.at(node);
    require_condition(split == 0 || split == 1, "split_node_mask must be binary");
    require_condition(leaf == 0 || leaf == 1, "node_leaf_mask must be binary");
    require_condition(!(split == 1 && leaf == 1), "node cannot be both split and leaf");

    if (split == 1) {
        require_condition(depth < ensemble.max_depth, "trace split exceeds canonical max_depth");
        require_condition(heap_index < ensemble.internal_node_count_per_tree,
                          "trace split mapped outside canonical internal range");
        require_condition(embedding.split_internal_indices.at(node) == std::numeric_limits<size_t>::max(),
                          "trace split node embedded twice");
        embedding.split_internal_indices.at(node) = heap_index;

        const size_t left = checked_size(training.left_child_indices.at(node), "left_child_indices");
        const size_t right = checked_size(training.right_child_indices.at(node), "right_child_indices");
        embed_trace_node(training, ensemble, embedding, tree, left, (heap_index * 2) + 1, depth + 1);
        embed_trace_node(training, ensemble, embedding, tree, right, (heap_index * 2) + 2, depth + 1);
        return;
    }

    require_condition(leaf == 1, "reachable inactive node must be a leaf");
    const size_t leaf_heap_index =
        rightmost_descendant_leaf_heap_index(heap_index, depth, ensemble.max_depth);
    require_condition(leaf_heap_index >= ensemble.internal_node_count_per_tree,
                      "leaf heap index must lie in canonical leaf range");
    const size_t logical_leaf_zero_based = leaf_heap_index - ensemble.internal_node_count_per_tree;
    require_condition(logical_leaf_zero_based < ensemble.leaf_count_per_tree,
                      "logical leaf index exceeds canonical leaf range");
    embedding.leaf_logical_ids.at(node) = static_cast<std::int64_t>(logical_leaf_zero_based + 1);
}

sharedTraceForestEmbedding derive_trace_embedding(const forestTrainingSemanticTrace& training,
                                                 const zkboostSharedTreeEnsemble& ensemble)
{
    sharedTraceForestEmbedding out(training.node_count);
    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        const size_t root = checked_size(training.root_indices.at(tree), "root_indices");
        embed_trace_node(training, ensemble, out, tree, root, 0, 0);
    }
    return out;
}

size_t route_explicit_trace_leaf(const forestTrainingSemanticTrace& training,
                                 const forestSemanticInferenceQueryBatch& inference,
                                 size_t sample_count,
                                 size_t tree,
                                 size_t sample)
{
    const size_t query = tree_sample_offset(sample_count, tree, sample);
    size_t node = checked_size(training.root_indices.at(tree), "root_indices");
    for (size_t step = 0; step <= training.node_count; ++step) {
        require_condition(node < training.node_count, "routing node out of range");
        require_condition(checked_size(training.node_tree_ids.at(node), "node_tree_ids") == tree + 1,
                          "routing left the tree");
        const std::int64_t split = training.split_node_mask.at(node);
        const std::int64_t leaf = training.node_leaf_mask.at(node);

        if (split == 0) {
            require_condition(leaf == 1, "routing reached an inactive non-leaf node");
            return node;
        }

        const size_t feature = selected_feature_for_split_node(training, node);
        const std::int64_t threshold = training.thresholds.at(node);
        const std::int64_t value = inference.lifted_inputs.at(lifted_input_offset(training, query, feature));
        const std::int64_t branch = value < threshold ? 0 : 1;
        node = checked_size(branch == 0 ? training.left_child_indices.at(node)
                                        : training.right_child_indices.at(node),
                            "child index");
    }

    throw std::runtime_error("zkboost_shared_validate_forest: routing cycle detected");
}

zkboostSharedForestPathClaim build_logical_path_claim(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    size_t sample_count,
    const zkboostSharedTreeEnsemble& ensemble,
    const sharedTraceForestEmbedding& embedding,
    size_t tree,
    size_t sample)
{
    const size_t query = tree_sample_offset(sample_count, tree, sample);
    const size_t trace_leaf = route_explicit_trace_leaf(training, inference, sample_count, tree, sample);
    const std::int64_t logical_leaf_id = embedding.leaf_logical_ids.at(trace_leaf);
    require_condition(logical_leaf_id > 0, "trace leaf does not map to a logical leaf id");

    zkboostSharedForestPathClaim claim;
    claim.tree = tree;
    claim.sample = sample;
    claim.trace_leaf_index = static_cast<std::int64_t>(trace_leaf);
    claim.logical_leaf_id = logical_leaf_id;

    size_t current = 0;
    const size_t split_base = tree * ensemble.internal_node_count_per_tree;
    for (size_t step = 0; step < ensemble.max_depth; ++step) {
        require_condition(current < ensemble.internal_node_count_per_tree,
                          "logical path stepped outside internal-node range");
        const size_t feature =
            checked_size(ensemble.split_features.at(split_base + current), "logical split feature");
        const std::int64_t threshold = ensemble.split_thresholds.at(split_base + current);
        const std::int64_t value = inference.lifted_inputs.at(lifted_input_offset(training, query, feature));
        const std::int64_t branch = value < threshold ? 0 : 1;
        require_condition(branch == static_cast<std::int64_t>(logical_leaf_bit(logical_leaf_id,
                                                                               ensemble.max_depth,
                                                                               step)),
                          "logical leaf id does not match path comparisons");

        claim.path_features.push_back(static_cast<std::int64_t>(feature));
        claim.path_thresholds.push_back(threshold);
        claim.permuted_inputs.push_back(value);
        claim.path_branch_bits.push_back(branch);
        current = (current * 2) + 1 + static_cast<size_t>(branch);
    }

    const size_t leaf_base = tree * ensemble.leaf_count_per_tree;
    const size_t logical_leaf_zero_based = static_cast<size_t>(logical_leaf_id - 1);
    claim.tree_score = ensemble.leaf_values.at(leaf_base + logical_leaf_zero_based);
    return claim;
}

size_t path_internal_node_index_from_logical_leaf_id(std::int64_t logical_leaf_id,
                                                     size_t max_depth,
                                                     size_t step)
{
    require_condition(step <= max_depth, "path step exceeds max_depth");
    size_t current = 0;
    for (size_t i = 0; i < step; ++i) {
        current = (current * 2) + 1 +
                  logical_leaf_bit(logical_leaf_id, max_depth, i);
    }
    return current;
}

zkboostSharedValidateForestFetchView build_fetch_view_from_routing(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedTreeEnsemble& ensemble,
    const sharedTraceForestEmbedding& embedding,
    size_t sample_count,
    const zkboostSharedValidateForestSummary& summary,
    const zkboostSharedValidateForestRoutingView& routing)
{
    zkboostSharedValidateForestFetchView fetch;
    fetch.summary = summary;
    fetch.public_split_masks = training.split_node_mask;
    fetch.public_split_features.assign(training.node_count, 0);
    fetch.public_split_thresholds.assign(training.node_count, 0);

    for (size_t node = 0; node < training.node_count; ++node) {
        if (training.split_node_mask.at(node) == 0) {
            continue;
        }
        fetch.public_split_features.at(node) =
            static_cast<std::int64_t>(selected_feature_for_split_node(training, node));
        fetch.public_split_thresholds.at(node) = training.thresholds.at(node);

        zkboostSharedForestSplitFieldReadClaim split_read;
        split_read.node = node;
        split_read.tree = checked_size(training.node_tree_ids.at(node), "node_tree_ids") - 1;
        split_read.internal_node_index = embedding.split_internal_indices.at(node);
        split_read.feature = fetch.public_split_features.at(node);
        split_read.threshold = fetch.public_split_thresholds.at(node);
        fetch.split_field_reads.push_back(split_read);
    }

    fetch.path_reads.reserve(routing.summary.checked_paths * ensemble.max_depth);
    fetch.leaf_value_reads.reserve(routing.summary.checked_paths);
    for (const zkboostSharedForestPathClaim& claim : routing.path_claims) {
        const size_t query = tree_sample_offset(sample_count, claim.tree, claim.sample);
        for (size_t step = 0; step < claim.path_features.size(); ++step) {
            zkboostSharedForestPathReadClaim path_read;
            path_read.query = query;
            path_read.tree = claim.tree;
            path_read.sample = claim.sample;
            path_read.step = step;
            path_read.logical_leaf_id = claim.logical_leaf_id;
            path_read.internal_node_index = path_internal_node_index_from_logical_leaf_id(
                claim.logical_leaf_id,
                ensemble.max_depth,
                step);
            path_read.feature = claim.path_features.at(step);
            path_read.threshold = claim.path_thresholds.at(step);
            fetch.path_reads.push_back(path_read);
        }

        zkboostSharedForestLeafValueReadClaim leaf_read;
        leaf_read.query = query;
        leaf_read.tree = claim.tree;
        leaf_read.sample = claim.sample;
        leaf_read.logical_leaf_id = claim.logical_leaf_id;
        leaf_read.tree_score = claim.tree_score;
        fetch.leaf_value_reads.push_back(leaf_read);
    }

    return fetch;
}

} // namespace

zkboostSharedValidateForestRoutingView build_zkboost_shared_validate_forest_routing_view(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(inference.query_count % training.tree_count == 0,
                      "inference query_count must be divisible by tree_count");
    const size_t sample_count = inference.query_count / training.tree_count;
    validate_shapes(training, inference, sample_count);

    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(training);
    const sharedTraceForestEmbedding embedding =
        derive_trace_embedding(training, ensemble);

    zkboostSharedValidateForestRoutingView routing;
    routing.summary.tree_count = training.tree_count;
    routing.summary.sample_count = sample_count;
    routing.summary.node_count = training.node_count;
    routing.summary.feature_count = training.feature_count;
    routing.summary.max_depth = ensemble.max_depth;
    routing.summary.checked_paths = training.tree_count * sample_count;

    validate_node_roles_and_structure(training, routing.summary);
    validate_inference_rows(training, inference, sample_count);

    const zkboostSharedTracePrediction prediction =
        derive_zkboost_shared_trace_prediction(training, inference);
    routing.tree_scores = prediction.tree_scores;
    routing.public_query_inputs = inference.lifted_inputs;
    routing.trace_leaf_indices.reserve(training.tree_count * sample_count);
    routing.logical_leaf_ids.reserve(training.tree_count * sample_count);
    routing.path_claims.reserve(training.tree_count * sample_count);

    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        for (size_t sample = 0; sample < sample_count; ++sample) {
            zkboostSharedForestPathClaim claim =
                build_logical_path_claim(training,
                                         inference,
                                         sample_count,
                                         ensemble,
                                         embedding,
                                         tree,
                                         sample);
            const size_t query = tree_sample_offset(sample_count, tree, sample);
            const size_t training_sample =
                training_sample_index_for_inference_sample(training, inference, sample_count, sample);
            const size_t training_query = tree_sample_offset(training.query_count, tree, training_sample);
            require_condition(training.leaf_indices.at(training_query) == claim.trace_leaf_index,
                              "trace leaf index does not match routed trace leaf");
            require_condition(routing.tree_scores.at(query) == claim.tree_score,
                              "tree_scores entry does not match logical leaf value");
            routing.trace_leaf_indices.push_back(claim.trace_leaf_index);
            routing.logical_leaf_ids.push_back(claim.logical_leaf_id);
            routing.path_claims.push_back(std::move(claim));
        }
    }

    return routing;
}

zkboostSharedValidateForestFetchView build_zkboost_shared_validate_forest_fetch_view(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    const zkboostSharedValidateForestRoutingView routing =
        build_zkboost_shared_validate_forest_routing_view(training, inference);
    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(training);
    const sharedTraceForestEmbedding embedding =
        derive_trace_embedding(training, ensemble);
    return build_fetch_view_from_routing(
        training,
        ensemble,
        embedding,
        routing.summary.sample_count,
        routing.summary,
        routing);
}

zkboostSharedValidateForestView build_zkboost_shared_validate_forest_view(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(training);
    const sharedTraceForestEmbedding embedding =
        derive_trace_embedding(training, ensemble);

    zkboostSharedValidateForestView view;
    view.routing = build_zkboost_shared_validate_forest_routing_view(training, inference);
    view.fetch = build_fetch_view_from_routing(
        training,
        ensemble,
        embedding,
        view.routing.summary.sample_count,
        view.routing.summary,
        view.routing);
    view.summary = view.routing.summary;
    return view;
}

zkboostSharedValidateForestSummary validate_zkboost_shared_forest(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    return build_zkboost_shared_validate_forest_view(training, inference).summary;
}
