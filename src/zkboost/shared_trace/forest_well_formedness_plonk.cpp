#include "zkboost/shared_trace/forest_well_formedness_plonk.hpp"

#include "algebra_utils.hpp"
#include "zkboost/compiler/generic_circuit.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr std::int64_t kNodeRangeTag = 4100000001LL;
constexpr std::int64_t kFeatureRangeTag = 4100000002LL;
constexpr std::int64_t kThresholdRangeTag = 4100000003LL;

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_forest_well_formedness: " + message);
    }
}

std::int64_t checked_i64(size_t value, const std::string& name)
{
    require_condition(value <= static_cast<size_t>(std::numeric_limits<std::int64_t>::max()),
                      name + " exceeds int64 range");
    return static_cast<std::int64_t>(value);
}

size_t checked_size(std::int64_t value, const std::string& name)
{
    require_condition(value >= 0, name + " must be non-negative");
    return static_cast<size_t>(value);
}

size_t node_feature_offset(const forestTrainingSemanticTrace& training,
                           size_t node,
                           size_t feature)
{
    return node * training.feature_count + feature;
}

std::int64_t selected_feature(const forestTrainingSemanticTrace& training, size_t node)
{
    size_t selected = training.feature_count;
    for (size_t feature = 0; feature < training.feature_count; ++feature) {
        const std::int64_t bit = training.feature_selector.at(
            node_feature_offset(training, node, feature));
        require_condition(bit == 0 || bit == 1, "feature selector must be binary");
        if (bit == 0) {
            continue;
        }
        require_condition(selected == training.feature_count,
                          "split node has multiple selected features");
        selected = feature;
    }
    require_condition(selected < training.feature_count,
                      "split node has no selected feature");
    return checked_i64(selected, "selected feature");
}

size_t tree_internal_offset(const zkboostSharedForestWellFormednessView& view, size_t tree)
{
    return tree * view.ensemble.internal_node_count_per_tree;
}

size_t tree_leaf_offset(const zkboostSharedForestWellFormednessView& view, size_t tree)
{
    return tree * view.ensemble.leaf_count_per_tree;
}

void fill_well_formedness_subtree(const forestTrainingSemanticTrace& training,
                                  zkboostSharedForestWellFormednessView& view,
                                  size_t tree,
                                  size_t node,
                                  size_t heap_index,
                                  size_t depth)
{
    require_condition(node < training.node_count, "source node out of range");
    require_condition(checked_size(training.node_tree_ids.at(node), "source tree") == tree + 1,
                      "source node belongs to another tree");
    const std::int64_t split = training.split_node_mask.at(node);
    const std::int64_t leaf = training.node_leaf_mask.at(node);
    require_condition(split == 0 || split == 1, "split mask must be binary");
    require_condition(leaf == 0 || leaf == 1, "leaf mask must be binary");
    require_condition(split + leaf == 1, "reachable node must be exactly split or leaf");

    if (depth == view.ensemble.max_depth) {
        require_condition(split == 0, "split node exceeds canonical depth");
        const size_t leaf_local = heap_index - view.ensemble.internal_node_count_per_tree;
        const size_t slot = tree_leaf_offset(view, tree) + leaf_local;
        view.leaf_source_nodes.at(slot) = checked_i64(node, "leaf source node");
        view.leaf_source_split_masks.at(slot) = split;
        view.source_leaf_values.at(slot) = training.node_leaf_values.at(node);
        require_condition(view.ensemble.leaf_values.at(slot) == training.node_leaf_values.at(node),
                          "canonical leaf value does not match source leaf");
        return;
    }

    const size_t slot = tree_internal_offset(view, tree) + heap_index;
    view.internal_source_nodes.at(slot) = checked_i64(node, "internal source node");
    view.source_split_masks.at(slot) = split;
    if (split != 0) {
        const size_t left = checked_size(training.left_child_indices.at(node), "left child");
        const size_t right = checked_size(training.right_child_indices.at(node), "right child");
        view.dummy_masks.at(slot) = 0;
        view.source_features.at(slot) = selected_feature(training, node);
        view.source_thresholds.at(slot) = training.thresholds.at(node);
        view.source_left_children.at(slot) = checked_i64(left, "left child");
        view.source_right_children.at(slot) = checked_i64(right, "right child");
        require_condition(view.ensemble.split_features.at(slot) == view.source_features.at(slot),
                          "canonical feature does not match source split");
        require_condition(view.ensemble.split_thresholds.at(slot) == view.source_thresholds.at(slot),
                          "canonical threshold does not match source split");
        fill_well_formedness_subtree(training, view, tree, left,
                                     heap_index * 2 + 1, depth + 1);
        fill_well_formedness_subtree(training, view, tree, right,
                                     heap_index * 2 + 2, depth + 1);
        return;
    }

    view.dummy_masks.at(slot) = 1;
    view.source_features.at(slot) = 0;
    view.source_thresholds.at(slot) = 0;
    view.source_left_children.at(slot) = checked_i64(node, "dummy left child");
    view.source_right_children.at(slot) = checked_i64(node, "dummy right child");
    require_condition(view.ensemble.split_features.at(slot) == 0,
                      "dummy feature must be zero");
    require_condition(view.ensemble.split_thresholds.at(slot) == 0,
                      "dummy threshold must be zero");
    fill_well_formedness_subtree(training, view, tree, node,
                                 heap_index * 2 + 1, depth + 1);
    fill_well_formedness_subtree(training, view, tree, node,
                                 heap_index * 2 + 2, depth + 1);
}

class publicInputLayout {
public:
    explicit publicInputLayout(const zkboostSharedForestWellFormednessSummary& summary)
    {
        const size_t s = summary.internal_slot_count;
        const size_t l = summary.leaf_slot_count;
        canonical_feature_offset = 0;
        canonical_threshold_offset = canonical_feature_offset + s;
        canonical_leaf_offset = canonical_threshold_offset + s;
        dummy_offset = canonical_leaf_offset + l;
        internal_source_offset = dummy_offset + s;
        source_split_offset = internal_source_offset + s;
        source_feature_offset = source_split_offset + s;
        source_threshold_offset = source_feature_offset + s;
        source_left_offset = source_threshold_offset + s;
        source_right_offset = source_left_offset + s;
        leaf_source_offset = source_right_offset + s;
        leaf_source_split_offset = leaf_source_offset + l;
        source_leaf_value_offset = leaf_source_split_offset + l;
        root_source_offset = source_leaf_value_offset + l;
        total_size = root_source_offset + summary.tree_count;
    }

    size_t canonical_feature_offset = 0;
    size_t canonical_threshold_offset = 0;
    size_t canonical_leaf_offset = 0;
    size_t dummy_offset = 0;
    size_t internal_source_offset = 0;
    size_t source_split_offset = 0;
    size_t source_feature_offset = 0;
    size_t source_threshold_offset = 0;
    size_t source_left_offset = 0;
    size_t source_right_offset = 0;
    size_t leaf_source_offset = 0;
    size_t leaf_source_split_offset = 0;
    size_t source_leaf_value_offset = 0;
    size_t root_source_offset = 0;
    size_t total_size = 0;
};

void validate_view_shapes(const zkboostSharedForestWellFormednessView& view)
{
    const size_t s = view.summary.internal_slot_count;
    const size_t l = view.summary.leaf_slot_count;
    require_condition(view.ensemble.split_features.size() == s, "canonical feature size mismatch");
    require_condition(view.ensemble.split_thresholds.size() == s, "canonical threshold size mismatch");
    require_condition(view.ensemble.leaf_values.size() == l, "canonical leaf size mismatch");
    require_condition(view.dummy_masks.size() == s, "dummy mask size mismatch");
    require_condition(view.internal_source_nodes.size() == s, "internal source size mismatch");
    require_condition(view.source_split_masks.size() == s, "source split size mismatch");
    require_condition(view.source_features.size() == s, "source feature size mismatch");
    require_condition(view.source_thresholds.size() == s, "source threshold size mismatch");
    require_condition(view.source_left_children.size() == s, "source left size mismatch");
    require_condition(view.source_right_children.size() == s, "source right size mismatch");
    require_condition(view.leaf_source_nodes.size() == l, "leaf source size mismatch");
    require_condition(view.leaf_source_split_masks.size() == l, "leaf source split size mismatch");
    require_condition(view.source_leaf_values.size() == l, "source leaf value size mismatch");
    require_condition(view.root_source_nodes.size() == view.summary.tree_count,
                      "root source size mismatch");
}

class sharedForestWellFormednessCircuitCompiler : public zkboostGenericCircuitCompilerBase {
public:
    explicit sharedForestWellFormednessCircuitCompiler(
        const zkboostSharedForestWellFormednessView& view)
        : view_(view)
    {
        validate_view_shapes(view_);
        add_range_tables();
        initialize_public_inputs();
    }

    void compile()
    {
        const size_t per_tree_internal = view_.ensemble.internal_node_count_per_tree;
        const size_t per_tree_leaves = view_.ensemble.leaf_count_per_tree;
        const zkboostAssignedCircuitValue zero = constant(0);
        for (size_t tree = 0; tree < view_.summary.tree_count; ++tree) {
            const size_t internal_base = tree * per_tree_internal;
            const size_t leaf_base = tree * per_tree_leaves;
            constrain_equal(internal_source_nodes_.at(internal_base), root_source_nodes_.at(tree));

            for (size_t local = 0; local < per_tree_internal; ++local) {
                const size_t slot = internal_base + local;
                const zkboostAssignedCircuitValue dummy = dummy_masks_.at(slot);
                const zkboostAssignedCircuitValue real = one_minus(dummy);
                constrain_boolean(dummy);
                constrain_equal(real, source_split_masks_.at(slot));

                // Canonical values are either equal to these ranged source
                // values or forced to zero, so separate range lookups are redundant.
                add_lookup(source_features_.at(slot), constant(kFeatureRangeTag), zero);
                add_lookup(source_thresholds_.at(slot), constant(kThresholdRangeTag), zero);
                add_lookup(internal_source_nodes_.at(slot), constant(kNodeRangeTag), zero);
                add_lookup(source_left_children_.at(slot), constant(kNodeRangeTag), zero);
                add_lookup(source_right_children_.at(slot), constant(kNodeRangeTag), zero);

                constrain_when(real, canonical_features_.at(slot), source_features_.at(slot));
                constrain_when(real, canonical_thresholds_.at(slot), source_thresholds_.at(slot));
                constrain_when(dummy, canonical_features_.at(slot), zero);
                constrain_when(dummy, canonical_thresholds_.at(slot), zero);

                const size_t left_local = local * 2 + 1;
                const size_t right_local = local * 2 + 2;
                constrain_equal(canonical_source(tree, left_local), source_left_children_.at(slot));
                constrain_equal(canonical_source(tree, right_local), source_right_children_.at(slot));

                if (local != 0) {
                    const size_t parent = internal_base + (local - 1) / 2;
                    const zkboostAssignedCircuitValue forbidden =
                        multiply(dummy_masks_.at(parent), real);
                    constrain_equal(forbidden, zero);
                }
            }

            for (size_t local = 0; local < per_tree_leaves; ++local) {
                const size_t slot = leaf_base + local;
                add_lookup(leaf_source_nodes_.at(slot), constant(kNodeRangeTag), zero);
                constrain_equal(leaf_source_split_masks_.at(slot), zero);
                constrain_equal(canonical_leaf_values_.at(slot), source_leaf_values_.at(slot));
            }
        }
    }

private:
    void add_range_tables()
    {
        for (size_t node = 0; node < view_.summary.node_count; ++node) {
            add_lookup_table_row(checked_i64(node, "node range"), kNodeRangeTag, 0);
        }
        for (size_t feature = 0; feature < view_.summary.feature_count; ++feature) {
            add_lookup_table_row(checked_i64(feature, "feature range"), kFeatureRangeTag, 0);
        }
        for (size_t threshold = 0; threshold < view_.summary.bin_count; ++threshold) {
            add_lookup_table_row(checked_i64(threshold, "threshold range"), kThresholdRangeTag, 0);
        }
    }

    void append_public(const std::vector<std::int64_t>& values,
                       std::vector<zkboostAssignedCircuitValue>& assigned)
    {
        assigned.reserve(values.size());
        for (const std::int64_t value : values) {
            assigned.push_back(public_value(value));
        }
    }

    void initialize_public_inputs()
    {
        append_public(view_.ensemble.split_features, canonical_features_);
        append_public(view_.ensemble.split_thresholds, canonical_thresholds_);
        append_public(view_.ensemble.leaf_values, canonical_leaf_values_);
        append_public(view_.dummy_masks, dummy_masks_);
        append_public(view_.internal_source_nodes, internal_source_nodes_);
        append_public(view_.source_split_masks, source_split_masks_);
        append_public(view_.source_features, source_features_);
        append_public(view_.source_thresholds, source_thresholds_);
        append_public(view_.source_left_children, source_left_children_);
        append_public(view_.source_right_children, source_right_children_);
        append_public(view_.leaf_source_nodes, leaf_source_nodes_);
        append_public(view_.leaf_source_split_masks, leaf_source_split_masks_);
        append_public(view_.source_leaf_values, source_leaf_values_);
        append_public(view_.root_source_nodes, root_source_nodes_);
    }

    zkboostAssignedCircuitValue canonical_source(size_t tree, size_t heap_index) const
    {
        if (heap_index < view_.ensemble.internal_node_count_per_tree) {
            return internal_source_nodes_.at(
                tree * view_.ensemble.internal_node_count_per_tree + heap_index);
        }
        return leaf_source_nodes_.at(
            tree * view_.ensemble.leaf_count_per_tree +
            heap_index - view_.ensemble.internal_node_count_per_tree);
    }

    void constrain_when(const zkboostAssignedCircuitValue& enabled,
                        const zkboostAssignedCircuitValue& left,
                        const zkboostAssignedCircuitValue& right)
    {
        const zkboostAssignedCircuitValue difference =
            add_scaled(left, Fr(1), right, Fr(-1), Fr(0));
        const zkboostAssignedCircuitValue gated = multiply(enabled, difference);
        constrain_equal(gated, constant(0));
    }

    const zkboostSharedForestWellFormednessView& view_;
    std::vector<zkboostAssignedCircuitValue> canonical_features_;
    std::vector<zkboostAssignedCircuitValue> canonical_thresholds_;
    std::vector<zkboostAssignedCircuitValue> canonical_leaf_values_;
    std::vector<zkboostAssignedCircuitValue> dummy_masks_;
    std::vector<zkboostAssignedCircuitValue> internal_source_nodes_;
    std::vector<zkboostAssignedCircuitValue> source_split_masks_;
    std::vector<zkboostAssignedCircuitValue> source_features_;
    std::vector<zkboostAssignedCircuitValue> source_thresholds_;
    std::vector<zkboostAssignedCircuitValue> source_left_children_;
    std::vector<zkboostAssignedCircuitValue> source_right_children_;
    std::vector<zkboostAssignedCircuitValue> leaf_source_nodes_;
    std::vector<zkboostAssignedCircuitValue> leaf_source_split_masks_;
    std::vector<zkboostAssignedCircuitValue> source_leaf_values_;
    std::vector<zkboostAssignedCircuitValue> root_source_nodes_;
};

std::vector<std::int64_t> decode_public_slice(const std::vector<Fr>& inputs,
                                              size_t offset,
                                              size_t count,
                                              const char* context)
{
    require_condition(offset <= inputs.size() && count <= inputs.size() - offset,
                      std::string(context) + " public-input slice out of range");
    std::vector<std::int64_t> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out.push_back(canonical_signed_int64_from_fr(inputs.at(offset + i), context));
    }
    return out;
}

bool same_commitments(const zkboostSharedForestCommitments& left,
                      const zkboostSharedForestCommitments& right)
{
    return left.com_f == right.com_f && left.com_t == right.com_t && left.com_w == right.com_w;
}

} // namespace

zkboostSharedForestWellFormednessView build_zkboost_shared_forest_well_formedness_view(
    const forestTrainingSemanticTrace& training)
{
    zkboostSharedForestWellFormednessView out;
    out.ensemble = shared_tree_ensemble_from_training_trace(training);
    out.summary.tree_count = training.tree_count;
    out.summary.node_count = training.node_count;
    out.summary.feature_count = training.feature_count;
    out.summary.bin_count = training.bin_count;
    out.summary.max_depth = out.ensemble.max_depth;
    out.summary.internal_slot_count = out.ensemble.split_features.size();
    out.summary.leaf_slot_count = out.ensemble.leaf_values.size();

    const size_t s = out.summary.internal_slot_count;
    const size_t l = out.summary.leaf_slot_count;
    out.dummy_masks.assign(s, 0);
    out.internal_source_nodes.assign(s, 0);
    out.source_split_masks.assign(s, 0);
    out.source_features.assign(s, 0);
    out.source_thresholds.assign(s, 0);
    out.source_left_children.assign(s, 0);
    out.source_right_children.assign(s, 0);
    out.leaf_source_nodes.assign(l, 0);
    out.leaf_source_split_masks.assign(l, 0);
    out.source_leaf_values.assign(l, 0);
    out.root_source_nodes = training.root_indices;

    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        const size_t root = checked_size(training.root_indices.at(tree), "root index");
        fill_well_formedness_subtree(training, out, tree, root, 0, 0);
    }
    for (const std::int64_t dummy : out.dummy_masks) {
        out.summary.dummy_internal_slot_count += dummy == 1 ? 1 : 0;
    }
    validate_view_shapes(out);
    return out;
}

zkboostSharedForestWellFormednessPlonkInstance
build_zkboost_shared_forest_well_formedness_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedForestCommitments& commitments,
    const kzgPublicParameters& public_parameters)
{
    const zkboostSharedForestWellFormednessView view =
        build_zkboost_shared_forest_well_formedness_view(training);
    sharedForestWellFormednessCircuitCompiler compiler(view);
    compiler.compile();
    const zkboostCompiledCircuitStatement statement =
        compile_zkboost_generic_circuit_statement("shared_forest_well_formedness",
                                                  compiler.build_circuit(),
                                                  public_parameters);

    zkboostSharedForestWellFormednessPlonkInstance out;
    out.setup = statement.setup;
    out.witness = statement.witness;
    out.public_inputs = statement.public_inputs;
    out.summary = view.summary;
    out.view = view;
    out.commitments = commitments;
    return out;
}

size_t required_zkboost_shared_forest_well_formedness_plonk_srs_size(
    const forestTrainingSemanticTrace& training)
{
    const zkboostSharedForestWellFormednessView view =
        build_zkboost_shared_forest_well_formedness_view(training);
    sharedForestWellFormednessCircuitCompiler compiler(view);
    compiler.compile();
    return required_zkboost_generic_circuit_plonk_srs_size(compiler.build_circuit());
}

plonkProof prove_zkboost_shared_forest_well_formedness_plonk(
    const zkboostSharedForestWellFormednessPlonkInstance& instance)
{
    return prove_plonk(instance.setup.proving_key, instance.witness, instance.public_inputs);
}

bool verify_zkboost_shared_forest_well_formedness_plonk(
    const zkboostSharedForestWellFormednessPlonkInstance& instance,
    const plonkProof& proof)
{
    return verify_plonk(instance.setup.verification_key, instance.public_inputs, proof) &&
           verify_zkboost_shared_forest_well_formedness_commitment_binding(instance);
}

bool verify_zkboost_shared_forest_well_formedness_commitment_binding(
    const zkboostSharedForestWellFormednessPlonkInstance& instance)
{
    try {
        const publicInputLayout layout(instance.summary);
        if (instance.public_inputs.size() != layout.total_size) {
            return false;
        }
        zkboostSharedTreeEnsemble ensemble = instance.view.ensemble;
        ensemble.split_features = decode_public_slice(instance.public_inputs,
                                                      layout.canonical_feature_offset,
                                                      instance.summary.internal_slot_count,
                                                      "WF committed feature");
        ensemble.split_thresholds = decode_public_slice(instance.public_inputs,
                                                        layout.canonical_threshold_offset,
                                                        instance.summary.internal_slot_count,
                                                        "WF committed threshold");
        ensemble.leaf_values = decode_public_slice(instance.public_inputs,
                                                   layout.canonical_leaf_offset,
                                                   instance.summary.leaf_slot_count,
                                                   "WF committed leaf");
        validate_zkboost_shared_tree_ensemble(ensemble);
        const zkboostSharedForestCommitments expected =
            commit_zkboost_shared_forest(instance.setup.verification_key.public_parameters, ensemble);
        return same_commitments(expected, instance.commitments);
    } catch (const std::exception&) {
        return false;
    }
}

bool link_zkboost_shared_forest_well_formedness_to_fetch(
    const zkboostSharedForestWellFormednessPlonkInstance& well_formedness,
    const zkboostSharedValidateForestFetchInstance& fetch)
{
    if (well_formedness.summary.tree_count != fetch.summary.tree_count ||
        well_formedness.summary.feature_count != fetch.summary.feature_count ||
        well_formedness.summary.max_depth != fetch.summary.max_depth ||
        !same_commitments(well_formedness.commitments, fetch.commitments)) {
        return false;
    }
    const publicInputLayout layout(well_formedness.summary);
    try {
        return decode_public_slice(well_formedness.public_inputs,
                                   layout.canonical_feature_offset,
                                   well_formedness.summary.internal_slot_count,
                                   "WF/fetch feature") == fetch.ensemble.split_features &&
               decode_public_slice(well_formedness.public_inputs,
                                   layout.canonical_threshold_offset,
                                   well_formedness.summary.internal_slot_count,
                                   "WF/fetch threshold") == fetch.ensemble.split_thresholds &&
               decode_public_slice(well_formedness.public_inputs,
                                   layout.canonical_leaf_offset,
                                   well_formedness.summary.leaf_slot_count,
                                   "WF/fetch leaf") == fetch.ensemble.leaf_values;
    } catch (const std::exception&) {
        return false;
    }
}

bool link_zkboost_shared_forest_well_formedness_to_splits(
    const zkboostSharedForestWellFormednessPlonkInstance& well_formedness,
    const zkboostSharedSplitPlonkInstance& splits)
{
    const size_t n = splits.summary.node_count;
    const size_t c = splits.summary.candidate_count;
    if (well_formedness.summary.node_count != n ||
        well_formedness.summary.feature_count != splits.summary.feature_count) {
        return false;
    }
    const size_t selected_feature_offset = 16 * c + 2 * n;
    const size_t selected_bin_offset = 16 * c + 3 * n;
    const size_t split_mask_offset = 16 * c + 4 * n;
    const size_t active_mask_offset = 16 * c + 5 * n;
    const size_t leaf_mask_offset = 16 * c + 6 * n;
    const size_t node_depth_offset = 16 * c + 7 * n;
    const size_t depth_limit_offset = 16 * c + 8 * n;
    if (splits.public_inputs.size() < depth_limit_offset + n) {
        return false;
    }

    const publicInputLayout layout(well_formedness.summary);
    const Fr zero(0);
    const Fr one(1);
    std::vector<size_t> linked_depths(n, 0);
    std::vector<Fr> linked_leaf_masks(n, zero);
    std::vector<bool> linked_nodes(n, false);
    const auto record_node = [&](size_t node, size_t depth, const Fr& leaf_mask) {
        if (node >= n) {
            return false;
        }
        if (!linked_nodes[node] || depth < linked_depths[node]) {
            linked_nodes[node] = true;
            linked_depths[node] = depth;
            linked_leaf_masks[node] = leaf_mask;
        } else if (linked_leaf_masks[node] != leaf_mask) {
            return false;
        }
        return true;
    };

    const size_t per_tree_internal =
        well_formedness.view.ensemble.internal_node_count_per_tree;
    for (size_t slot = 0; slot < well_formedness.summary.internal_slot_count; ++slot) {
        const size_t node = canonical_size_t_from_fr(
            well_formedness.public_inputs.at(layout.internal_source_offset + slot),
            "WF internal source node");
        if (node >= n) {
            return false;
        }
        const Fr& source_split = well_formedness.public_inputs.at(layout.source_split_offset + slot);
        if (source_split != splits.public_inputs.at(split_mask_offset + node)) {
            return false;
        }
        if (source_split == one) {
            if (well_formedness.public_inputs.at(layout.source_feature_offset + slot) !=
                    splits.public_inputs.at(selected_feature_offset + node) ||
                well_formedness.public_inputs.at(layout.source_threshold_offset + slot) !=
                    splits.public_inputs.at(selected_bin_offset + node) + one) {
                return false;
            }
        } else if (source_split != zero) {
            return false;
        }

        size_t local = slot % per_tree_internal;
        size_t depth = 1;
        while (local != 0) {
            local = (local - 1) / 2;
            ++depth;
        }
        if (!record_node(node, depth, one - source_split)) {
            return false;
        }
    }
    const size_t canonical_leaf_depth = well_formedness.summary.max_depth + 1;
    for (size_t slot = 0; slot < well_formedness.summary.leaf_slot_count; ++slot) {
        const size_t node = canonical_size_t_from_fr(
            well_formedness.public_inputs.at(layout.leaf_source_offset + slot),
            "WF leaf source node");
        if (node >= n ||
            well_formedness.public_inputs.at(layout.leaf_source_split_offset + slot) !=
                splits.public_inputs.at(split_mask_offset + node) ||
            splits.public_inputs.at(split_mask_offset + node) != zero ||
            !record_node(node, canonical_leaf_depth, one)) {
            return false;
        }
    }

    for (size_t node = 0; node < n; ++node) {
        if (!linked_nodes[node]) {
            continue;
        }
        if (splits.public_inputs.at(leaf_mask_offset + node) != linked_leaf_masks[node] ||
            splits.public_inputs.at(node_depth_offset + node) != field_from_size_t(linked_depths[node])) {
            return false;
        }
        const Fr& depth_limit = splits.public_inputs.at(depth_limit_offset + node);
        // The split circuit already binds depth_limit to its semantic max_depth;
        // node_depth was linked above to the WF source depth. The padded canonical
        // leaf depth is not the semantic depth for an early leaf (notably d=1).
        const Fr& active = splits.public_inputs.at(active_mask_offset + node);
        if (depth_limit != zero &&
            (depth_limit != one || active != one || linked_leaf_masks[node] != one)) {
            return false;
        }
    }
    return true;
}

bool link_zkboost_shared_forest_well_formedness_to_leaf_values(
    const zkboostSharedForestWellFormednessPlonkInstance& well_formedness,
    const zkboostSharedLeafValuePlonkInstance& leaf_values)
{
    const size_t n = leaf_values.summary.node_count;
    if (well_formedness.summary.node_count != n) {
        return false;
    }
    const size_t decoded_leaf_offset = 3 * n;
    if (leaf_values.public_inputs.size() < decoded_leaf_offset + n) {
        return false;
    }
    const publicInputLayout layout(well_formedness.summary);
    for (size_t slot = 0; slot < well_formedness.summary.leaf_slot_count; ++slot) {
        const size_t node = canonical_size_t_from_fr(
            well_formedness.public_inputs.at(layout.leaf_source_offset + slot),
            "WF leaf source node");
        if (node >= n ||
            well_formedness.public_inputs.at(layout.source_leaf_value_offset + slot) !=
                leaf_values.public_inputs.at(decoded_leaf_offset + node)) {
            return false;
        }
    }
    return true;
}
