#include "zkboost/shared_trace/forest_commitments.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_forest_commitments: " + message);
    }
}

size_t checked_size(std::int64_t value, const std::string& context)
{
    require_condition(value >= 0, context + " must be non-negative");
    return static_cast<size_t>(value);
}

size_t next_power_of_two(size_t value)
{
    size_t out = 1;
    while (out < value) {
        out <<= 1U;
    }
    return out;
}

size_t node_feature_offset(const forestTrainingSemanticTrace& trace, size_t node, size_t feature)
{
    return node * trace.feature_count + feature;
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

Fr fr_from_i64(std::int64_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

Group1 commit_public_column(const kzgPublicParameters& public_params,
                            const std::string& key,
                            const std::vector<std::int64_t>& values)
{
    kzgProtocol prover;
    prover.setup(public_params);
    const std::vector<Fr> encoded = encode_zkboost_shared_forest_column_padded(values);
    const kzgCommitment commitment =
        prover.commit_public(key,
                             poly_from_evals(encoded),
                             canonical_padded_zkboost_shared_forest_length(values.size()) - 1);
    return commitment.com;
}

size_t group1_byte_size(const Group1& value)
{
    return proof_encoding(value).size() / 2;
}

size_t safe_pow2(size_t exponent, const char* context)
{
    require_condition(exponent < (std::numeric_limits<size_t>::digits - 1), std::string(context) + " too large");
    return size_t(1) << exponent;
}

void recompute_shared_kzg_srs_requirement(zkboostSharedKzgSrsRequirement& requirement)
{
    const size_t degree_shift_requirement =
        requirement.min_degree_bound_plus_1 == 0
            ? size_t(1)
            : requirement.deg1_plus_1 - requirement.min_degree_bound_plus_1 + 1;
    requirement.deg2_plus_1 = std::max(requirement.opening_deg2_plus_1_floor, degree_shift_requirement);
}

struct treeReachabilityState {
    std::vector<std::uint8_t> entered;
    std::vector<std::uint8_t> exited;
};

void validate_training_trace_shapes(const forestTrainingSemanticTrace& training)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(training.node_count > 0, "node_count must be non-zero");
    require_condition(training.feature_count > 0, "feature_count must be non-zero");
    require_condition(training.bin_count > 0, "bin_count must be non-zero");
    require_condition(training.max_depth > 0,
                      "semantic max_depth must be positive because the root has depth one");
    require_condition(training.root_indices.size() == training.tree_count, "root_indices length mismatch");
    require_condition(training.node_tree_ids.size() == training.node_count, "node_tree_ids length mismatch");
    require_condition(training.split_node_mask.size() == training.node_count, "split_node_mask length mismatch");
    require_condition(training.node_leaf_mask.size() == training.node_count, "node_leaf_mask length mismatch");
    require_condition(training.feature_selector.size() == training.node_count * training.feature_count,
                      "feature_selector length mismatch");
    require_condition(training.thresholds.size() == training.node_count, "thresholds length mismatch");
    require_condition(training.left_child_indices.size() == training.node_count, "left_child_indices length mismatch");
    require_condition(training.right_child_indices.size() == training.node_count, "right_child_indices length mismatch");
    require_condition(training.node_leaf_values.size() == training.node_count, "node_leaf_values length mismatch");
}

size_t compute_tree_leaf_depth(const forestTrainingSemanticTrace& trace,
                               size_t tree,
                               size_t node,
                               size_t depth,
                               treeReachabilityState& state)
{
    require_condition(node < trace.node_count, "reachable node index out of range");
    require_condition(trace.node_tree_ids.at(node) >= 0, "node_tree_ids must be non-negative");
    require_condition(checked_size(trace.node_tree_ids.at(node), "node_tree_ids") == tree + 1,
                      "reachable node belongs to the wrong tree");
    require_condition(state.entered.at(node) == 0, "reachable subtree has multiple parents or a cycle");
    state.entered.at(node) = 1;

    const std::int64_t split = trace.split_node_mask.at(node);
    const std::int64_t leaf = trace.node_leaf_mask.at(node);
    require_condition(split == 0 || split == 1, "split_node_mask must be binary");
    require_condition(leaf == 0 || leaf == 1, "node_leaf_mask must be binary");
    require_condition(!(split == 1 && leaf == 1), "node cannot be both split and leaf");

    size_t out = depth;
    if (split == 1) {
        (void)selected_feature_for_split_node(trace, node);
        require_condition(trace.thresholds.at(node) >= 0, "split threshold must be non-negative");
        require_condition(static_cast<size_t>(trace.thresholds.at(node)) < trace.bin_count,
                          "split threshold exceeds bin_count");
        const size_t left = checked_size(trace.left_child_indices.at(node), "left_child_indices");
        const size_t right = checked_size(trace.right_child_indices.at(node), "right_child_indices");
        out = std::max(compute_tree_leaf_depth(trace, tree, left, depth + 1, state),
                       compute_tree_leaf_depth(trace, tree, right, depth + 1, state));
    } else {
        require_condition(leaf == 1, "reachable inactive node must be a leaf");
    }

    state.exited.at(node) = 1;
    return out;
}

struct sharedForestTreeShape {
    size_t max_depth = 0;
    size_t internal_node_count_per_tree = 0;
    size_t leaf_count_per_tree = 0;
};

sharedForestTreeShape derive_shared_tree_shape(const forestTrainingSemanticTrace& training)
{
    validate_training_trace_shapes(training);

    size_t max_leaf_node_depth = 1;
    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        const size_t root = checked_size(training.root_indices.at(tree), "root_indices");
        require_condition(root < training.node_count, "root index out of range");

        treeReachabilityState state;
        state.entered.assign(training.node_count, 0);
        state.exited.assign(training.node_count, 0);
        max_leaf_node_depth = std::max(
            max_leaf_node_depth,
            compute_tree_leaf_depth(training, tree, root, 1, state));
    }
    require_condition(max_leaf_node_depth <= training.max_depth,
                      "forest leaf exceeds the one-based semantic max_depth");

    sharedForestTreeShape out;
    // Keep the encoded split vector non-empty even if the trace contains a
    // degenerate root leaf; the resulting dummy split still represents the same
    // logical constant subtree under the shared-trace compatibility regime.
    // The canonical circuit stores one branch decision per edge, so its
    // path depth is the one-based maximum node depth minus one.
    out.max_depth = std::max<size_t>(1, max_leaf_node_depth - 1);
    out.leaf_count_per_tree = safe_pow2(out.max_depth, "shared tree max_depth");
    out.internal_node_count_per_tree = out.leaf_count_per_tree - 1;
    return out;
}

size_t tree_split_offset(const zkboostSharedTreeEnsemble& ensemble, size_t tree)
{
    return tree * ensemble.internal_node_count_per_tree;
}

size_t tree_leaf_offset(const zkboostSharedTreeEnsemble& ensemble, size_t tree)
{
    return tree * ensemble.leaf_count_per_tree;
}

void fill_dummy_subtree(zkboostSharedTreeEnsemble& ensemble,
                        size_t tree,
                        size_t heap_index,
                        size_t depth,
                        std::int64_t leaf_value)
{
    if (depth == ensemble.max_depth) {
        const size_t leaf_local = heap_index - ensemble.internal_node_count_per_tree;
        ensemble.leaf_values.at(tree_leaf_offset(ensemble, tree) + leaf_local) = leaf_value;
        return;
    }

    const size_t split_index = tree_split_offset(ensemble, tree) + heap_index;
    ensemble.split_features.at(split_index) = 0;
    // Inputs are validated to lie in [0, bin_count), so threshold zero makes
    // every dummy comparison take the right branch. This gives an early trace
    // leaf one canonical descendant independent of the query input.
    ensemble.split_thresholds.at(split_index) = 0;

    fill_dummy_subtree(ensemble, tree, (heap_index * 2) + 1, depth + 1, leaf_value);
    fill_dummy_subtree(ensemble, tree, (heap_index * 2) + 2, depth + 1, leaf_value);
}

void fill_canonical_subtree(const forestTrainingSemanticTrace& trace,
                            zkboostSharedTreeEnsemble& ensemble,
                            size_t tree,
                            size_t node,
                            size_t heap_index,
                            size_t depth)
{
    require_condition(node < trace.node_count, "canonical subtree node out of range");
    require_condition(checked_size(trace.node_tree_ids.at(node), "node_tree_ids") == tree + 1,
                      "canonical subtree crossed tree boundary");

    const std::int64_t split = trace.split_node_mask.at(node);
    const std::int64_t leaf = trace.node_leaf_mask.at(node);
    require_condition(split == 0 || split == 1, "split_node_mask must be binary");
    require_condition(leaf == 0 || leaf == 1, "node_leaf_mask must be binary");
    require_condition(!(split == 1 && leaf == 1), "node cannot be both split and leaf");

    if (split == 1) {
        require_condition(depth < ensemble.max_depth, "split node exceeds canonical max_depth");
        const size_t split_index = tree_split_offset(ensemble, tree) + heap_index;
        ensemble.split_features.at(split_index) =
            static_cast<std::int64_t>(selected_feature_for_split_node(trace, node));
        ensemble.split_thresholds.at(split_index) = trace.thresholds.at(node);

        const size_t left = checked_size(trace.left_child_indices.at(node), "left_child_indices");
        const size_t right = checked_size(trace.right_child_indices.at(node), "right_child_indices");
        fill_canonical_subtree(trace, ensemble, tree, left, (heap_index * 2) + 1, depth + 1);
        fill_canonical_subtree(trace, ensemble, tree, right, (heap_index * 2) + 2, depth + 1);
        return;
    }

    require_condition(leaf == 1, "reachable inactive node must be a leaf");
    fill_dummy_subtree(ensemble, tree, heap_index, depth, trace.node_leaf_values.at(node));
}

zkboostSharedCommittedForestSummary summarize_committed_forest(
    const zkboostSharedTreeEnsemble& ensemble,
    size_t node_count,
    const zkboostSharedForestCommitments& commitments)
{
    zkboostSharedCommittedForestSummary out;
    out.tree_count = ensemble.tree_count;
    out.node_count = node_count;
    out.feature_count = ensemble.feature_count;
    out.bin_count = ensemble.bin_count;
    out.max_depth = ensemble.max_depth;
    out.internal_node_count_per_tree = ensemble.internal_node_count_per_tree;
    out.leaf_count_per_tree = ensemble.leaf_count_per_tree;
    out.split_domain_size =
        canonical_padded_zkboost_shared_forest_length(ensemble.split_features.size());
    out.leaf_domain_size =
        canonical_padded_zkboost_shared_forest_length(ensemble.leaf_values.size());
    out.commitment_count = 3;
    out.commitment_bytes =
        group1_byte_size(commitments.com_f) +
        group1_byte_size(commitments.com_t) +
        group1_byte_size(commitments.com_w);
    return out;
}

} // namespace

void validate_zkboost_shared_tree_ensemble(const zkboostSharedTreeEnsemble& ensemble)
{
    require_condition(ensemble.tree_count > 0, "tree_count must be non-zero");
    require_condition(ensemble.feature_count > 0, "feature_count must be non-zero");
    require_condition(ensemble.bin_count > 0, "bin_count must be non-zero");
    require_condition(ensemble.max_depth > 0, "max_depth must be non-zero");
    require_condition(ensemble.internal_node_count_per_tree > 0,
                      "internal_node_count_per_tree must be non-zero");
    require_condition(ensemble.leaf_count_per_tree > 0,
                      "leaf_count_per_tree must be non-zero");
    require_condition(ensemble.internal_node_count_per_tree + 1 == ensemble.leaf_count_per_tree,
                      "leaf_count_per_tree must equal internal_node_count_per_tree + 1");
    require_condition(ensemble.leaf_count_per_tree == safe_pow2(ensemble.max_depth, "ensemble max_depth"),
                      "leaf_count_per_tree does not match max_depth");
    require_condition(ensemble.split_features.size() ==
                          ensemble.tree_count * ensemble.internal_node_count_per_tree,
                      "split_features length mismatch");
    require_condition(ensemble.split_thresholds.size() ==
                          ensemble.tree_count * ensemble.internal_node_count_per_tree,
                      "split_thresholds length mismatch");
    require_condition(ensemble.leaf_values.size() ==
                          ensemble.tree_count * ensemble.leaf_count_per_tree,
                      "leaf_values length mismatch");

    for (std::int64_t feature : ensemble.split_features) {
        require_condition(feature >= 0, "split feature must be non-negative");
        require_condition(static_cast<size_t>(feature) < ensemble.feature_count,
                          "split feature out of range");
    }
    for (std::int64_t threshold : ensemble.split_thresholds) {
        require_condition(threshold >= 0, "split threshold must be non-negative");
        require_condition(static_cast<size_t>(threshold) < ensemble.bin_count,
                          "split threshold out of range");
    }
}

void validate_zkboost_shared_forest_columns(const zkboostSharedForestColumns& forest)
{
    require_condition(forest.tree_count > 0, "tree_count must be non-zero");
    require_condition(forest.node_count > 0, "node_count must be non-zero");
    require_condition(forest.feature_count > 0, "feature_count must be non-zero");
    require_condition(forest.bin_count > 0, "bin_count must be non-zero");

    require_condition(forest.root_indices.size() == forest.tree_count, "root_indices length mismatch");
    require_condition(forest.tree_ids.size() == forest.node_count, "tree_ids length mismatch");
    require_condition(forest.split_masks.size() == forest.node_count, "split_masks length mismatch");
    require_condition(forest.leaf_masks.size() == forest.node_count, "leaf_masks length mismatch");
    require_condition(forest.split_features.size() == forest.node_count, "split_features length mismatch");
    require_condition(forest.thresholds.size() == forest.node_count, "thresholds length mismatch");
    require_condition(forest.left_children.size() == forest.node_count, "left_children length mismatch");
    require_condition(forest.right_children.size() == forest.node_count, "right_children length mismatch");
    require_condition(forest.leaf_values.size() == forest.node_count, "leaf_values length mismatch");

    for (size_t tree = 0; tree < forest.tree_count; ++tree) {
        const size_t root = checked_size(forest.root_indices.at(tree), "root_indices");
        require_condition(root < forest.node_count, "root index out of range");
        require_condition(checked_size(forest.tree_ids.at(root), "tree_ids") == tree + 1,
                          "root must belong to its tree");
    }

    for (size_t node = 0; node < forest.node_count; ++node) {
        const size_t tree = checked_size(forest.tree_ids.at(node), "tree_ids");

        const std::int64_t split = forest.split_masks.at(node);
        const std::int64_t leaf = forest.leaf_masks.at(node);
        require_condition(split == 0 || split == 1, "split_masks must be binary");
        require_condition(leaf == 0 || leaf == 1, "leaf_masks must be binary");
        require_condition(!(split == 1 && leaf == 1), "node cannot be both split and leaf");
        if (split == 0 && leaf == 0) {
            require_condition(tree == 0, "dummy node must use tree ID 0");
            continue;
        }
        require_condition(tree >= 1 && tree <= forest.tree_count, "real tree ID out of range");

        if (split == 1) {
            const size_t feature = checked_size(forest.split_features.at(node), "split_features");
            require_condition(feature < forest.feature_count, "split feature out of range");
            require_condition(forest.thresholds.at(node) >= 0, "threshold must be non-negative");
            require_condition(static_cast<size_t>(forest.thresholds.at(node)) < forest.bin_count,
                              "threshold exceeds bin_count");

            const size_t left = checked_size(forest.left_children.at(node), "left_children");
            const size_t right = checked_size(forest.right_children.at(node), "right_children");
            require_condition(left < forest.node_count, "left child out of range");
            require_condition(right < forest.node_count, "right child out of range");
            require_condition(checked_size(forest.tree_ids.at(left), "tree_ids") == tree,
                              "left child belongs to the wrong tree");
            require_condition(checked_size(forest.tree_ids.at(right), "tree_ids") == tree,
                              "right child belongs to the wrong tree");
        } else {
            require_condition(forest.split_features.at(node) == 0,
                              "inactive node must have zero split feature");
        }
    }
}

zkboostSharedTreeEnsemble shared_tree_ensemble_from_training_trace(
    const forestTrainingSemanticTrace& training)
{
    const sharedForestTreeShape shape = derive_shared_tree_shape(training);

    zkboostSharedTreeEnsemble out;
    out.tree_count = training.tree_count;
    out.feature_count = training.feature_count;
    out.bin_count = training.bin_count;
    out.max_depth = shape.max_depth;
    out.internal_node_count_per_tree = shape.internal_node_count_per_tree;
    out.leaf_count_per_tree = shape.leaf_count_per_tree;
    out.split_features.assign(out.tree_count * out.internal_node_count_per_tree, 0);
    out.split_thresholds.assign(out.tree_count * out.internal_node_count_per_tree, 0);
    out.leaf_values.assign(out.tree_count * out.leaf_count_per_tree, 0);

    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        const size_t root = checked_size(training.root_indices.at(tree), "root_indices");
        fill_canonical_subtree(training, out, tree, root, 0, 0);
    }

    validate_zkboost_shared_tree_ensemble(out);
    return out;
}

zkboostSharedForestColumns shared_forest_columns_from_training_trace(
    const forestTrainingSemanticTrace& training)
{
    validate_training_trace_shapes(training);

    zkboostSharedForestColumns out;
    out.tree_count = training.tree_count;
    out.node_count = training.node_count;
    out.feature_count = training.feature_count;
    out.bin_count = training.bin_count;
    out.root_indices = training.root_indices;
    out.tree_ids = training.node_tree_ids;
    out.split_masks = training.split_node_mask;
    out.leaf_masks = training.node_leaf_mask;
    out.thresholds = training.thresholds;
    out.left_children = training.left_child_indices;
    out.right_children = training.right_child_indices;
    out.leaf_values = training.node_leaf_values;
    out.split_features.assign(training.node_count, 0);

    for (size_t node = 0; node < training.node_count; ++node) {
        if (training.split_node_mask.at(node) == 0) {
            continue;
        }
        out.split_features.at(node) =
            static_cast<std::int64_t>(selected_feature_for_split_node(training, node));
    }

    validate_zkboost_shared_forest_columns(out);
    return out;
}

size_t canonical_padded_zkboost_shared_forest_length(size_t raw_length)
{
    require_condition(raw_length > 0, "raw forest encoding length must be non-zero");
    return next_power_of_two(raw_length);
}

zkboostSharedKzgSrsRequirement max_zkboost_shared_kzg_srs_requirement(
    const zkboostSharedKzgSrsRequirement& left,
    const zkboostSharedKzgSrsRequirement& right)
{
    zkboostSharedKzgSrsRequirement out;
    out.deg1_plus_1 = std::max(left.deg1_plus_1, right.deg1_plus_1);
    out.opening_deg2_plus_1_floor =
        std::max(left.opening_deg2_plus_1_floor, right.opening_deg2_plus_1_floor);
    if (left.min_degree_bound_plus_1 == 0) {
        out.min_degree_bound_plus_1 = right.min_degree_bound_plus_1;
    } else if (right.min_degree_bound_plus_1 == 0) {
        out.min_degree_bound_plus_1 = left.min_degree_bound_plus_1;
    } else {
        out.min_degree_bound_plus_1 =
            std::min(left.min_degree_bound_plus_1, right.min_degree_bound_plus_1);
    }
    recompute_shared_kzg_srs_requirement(out);
    return out;
}

zkboostSharedKzgSrsRequirement required_zkboost_shared_forest_commitment_srs_requirement(
    const zkboostSharedTreeEnsemble& ensemble)
{
    validate_zkboost_shared_tree_ensemble(ensemble);
    const size_t split_domain =
        canonical_padded_zkboost_shared_forest_length(ensemble.split_features.size());
    const size_t leaf_domain =
        canonical_padded_zkboost_shared_forest_length(ensemble.leaf_values.size());
    zkboostSharedKzgSrsRequirement out;
    out.deg1_plus_1 = std::max(split_domain, leaf_domain);
    out.min_degree_bound_plus_1 = std::min(split_domain, leaf_domain);
    recompute_shared_kzg_srs_requirement(out);
    return out;
}

zkboostSharedKzgSrsRequirement required_zkboost_shared_forest_commitment_srs_requirement(
    const zkboostSharedForestColumns& forest)
{
    validate_zkboost_shared_forest_columns(forest);
    const size_t root_domain = canonical_padded_zkboost_shared_forest_length(forest.root_indices.size());
    const size_t node_domain = canonical_padded_zkboost_shared_forest_length(forest.node_count);
    zkboostSharedKzgSrsRequirement out;
    out.deg1_plus_1 = std::max(root_domain, node_domain);
    out.min_degree_bound_plus_1 = std::min(root_domain, node_domain);
    recompute_shared_kzg_srs_requirement(out);
    return out;
}

zkboostSharedKzgSrsRequirement required_zkboost_shared_forest_commitment_srs_requirement_from_training_trace(
    const forestTrainingSemanticTrace& training)
{
    return required_zkboost_shared_forest_commitment_srs_requirement(
        shared_tree_ensemble_from_training_trace(training));
}

size_t required_zkboost_shared_forest_commitment_srs_size(
    const zkboostSharedTreeEnsemble& ensemble)
{
    return required_zkboost_shared_forest_commitment_srs_requirement(ensemble).deg1_plus_1;
}

size_t required_zkboost_shared_forest_commitment_srs_size(
    const zkboostSharedForestColumns& forest)
{
    return required_zkboost_shared_forest_commitment_srs_requirement(forest).deg1_plus_1;
}

size_t required_zkboost_shared_forest_commitment_srs_size_from_training_trace(
    const forestTrainingSemanticTrace& training)
{
    return required_zkboost_shared_forest_commitment_srs_requirement_from_training_trace(training).deg1_plus_1;
}

std::vector<Fr> encode_zkboost_shared_forest_column(
    const std::vector<std::int64_t>& values)
{
    require_condition(!values.empty(), "forest column encoding requires non-empty values");
    std::vector<Fr> out;
    out.reserve(values.size());
    for (std::int64_t value : values) {
        out.push_back(fr_from_i64(value));
    }
    return out;
}

std::vector<Fr> encode_zkboost_shared_forest_column_padded(
    const std::vector<std::int64_t>& values)
{
    std::vector<Fr> out = encode_zkboost_shared_forest_column(values);
    out.resize(canonical_padded_zkboost_shared_forest_length(values.size()), Fr(0));
    return out;
}

zkboostSharedForestCommitments commit_zkboost_shared_forest(
    const kzgPublicParameters& public_params,
    const zkboostSharedTreeEnsemble& ensemble)
{
    validate_zkboost_shared_tree_ensemble(ensemble);

    zkboostSharedForestCommitments out;
    out.com_f =
        commit_public_column(public_params, "zkboost.shared.forest.logical.split_features", ensemble.split_features);
    out.com_t =
        commit_public_column(public_params, "zkboost.shared.forest.logical.split_thresholds", ensemble.split_thresholds);
    out.com_w =
        commit_public_column(public_params, "zkboost.shared.forest.logical.leaf_values", ensemble.leaf_values);
    return out;
}

zkboostSharedForestCommitments commit_zkboost_shared_forest_from_training_trace(
    const kzgPublicParameters& public_params,
    const forestTrainingSemanticTrace& training)
{
    return commit_zkboost_shared_forest(
        public_params,
        shared_tree_ensemble_from_training_trace(training));
}

zkboostSharedCommittedForest build_zkboost_shared_committed_forest(
    const kzgPublicParameters& public_params,
    const forestTrainingSemanticTrace& training)
{
    zkboostSharedCommittedForest out;
    out.ensemble = shared_tree_ensemble_from_training_trace(training);
    out.commitments = commit_zkboost_shared_forest(public_params, out.ensemble);
    out.summary = summarize_committed_forest(out.ensemble, training.node_count, out.commitments);
    return out;
}
