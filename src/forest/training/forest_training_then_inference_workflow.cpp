#include "forest_training_then_inference_workflow.hpp"

#include "less_check.hpp"
#include "matrix_utils.hpp"
#include "training_runtime_utils.hpp"

#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void validate_query_batch_shape(const forestInferenceQueryBatch& queries)
{
    if (queries.query_count == 0 || queries.feature_count == 0) {
        throw std::runtime_error("forest_training_then_inference_workflow: query_count and feature_count must be non-zero");
    }
    if (!isPowerOfTwo(queries.query_count) || !isPowerOfTwo(queries.feature_count)) {
        throw std::runtime_error("forest_training_then_inference_workflow: query_count and feature_count must be powers of two");
    }

    validate_matrix_shape(queries.query_count, 1, "forest_training_then_inference_workflow query_tree_ids");
    validate_matrix_shape(queries.query_count, queries.feature_count, "forest_training_then_inference_workflow lifted_inputs");
}

void validate_committed_forest_shape(const forestCommittedNodeTable& committed_forest)
{
    if (committed_forest.tree_count == 0 || committed_forest.node_count == 0 || committed_forest.feature_count == 0) {
        throw std::runtime_error(
            "forest_training_then_inference_workflow: committed forest shape must be non-zero");
    }
    validate_matrix_shape(committed_forest.tree_count, 1, "forest_training_then_inference_workflow root indices");
    validate_matrix_shape(committed_forest.node_count, 1, "forest_training_then_inference_workflow committed node vectors");
    validate_matrix_shape(committed_forest.node_count,
                          committed_forest.feature_count,
                          "forest_training_then_inference_workflow committed feature selector");
    validate_matrix_shape(
        committed_forest.node_count, committed_forest.feature_count, "forest_training_then_inference_workflow committed node boxes");
}

void validate_witness_batch_shape(const forestExportedInferenceWitnessBatch& batch)
{
    if (batch.query_count == 0 || batch.feature_count == 0) {
        throw std::runtime_error("forest_training_then_inference_workflow: witness batch shape must be non-zero");
    }
    if (!isPowerOfTwo(batch.query_count) || !isPowerOfTwo(batch.feature_count)) {
        throw std::runtime_error("forest_training_then_inference_workflow: witness batch dimensions must be powers of two");
    }

    validate_matrix_shape(batch.query_count, 1, "forest_training_then_inference_workflow query_tree_ids witness");
    validate_matrix_shape(batch.query_count, batch.feature_count, "forest_training_then_inference_workflow lifted_inputs witness");
    validate_matrix_shape(batch.query_count, batch.feature_count, "forest_training_then_inference_workflow lower_bounds witness");
    validate_matrix_shape(batch.query_count, batch.feature_count, "forest_training_then_inference_workflow upper_bounds witness");
    validate_matrix_shape(batch.query_count, 1, "forest_training_then_inference_workflow scores witness");
    validate_matrix_shape(batch.query_count, 1, "forest_training_then_inference_workflow leaf_indices witness");
}

void validate_witness_batch_shape(const forestExportedInferenceBatchShape& batch)
{
    if (batch.query_count == 0 || batch.feature_count == 0) {
        throw std::runtime_error("forest_training_then_inference_workflow: witness batch shape must be non-zero");
    }
    if (!isPowerOfTwo(batch.query_count) || !isPowerOfTwo(batch.feature_count)) {
        throw std::runtime_error("forest_training_then_inference_workflow: witness batch dimensions must be powers of two");
    }
}

void validate_witness_matches_queries(const forestExportedInferenceWitnessBatch& batch,
                                      const forestInferenceQueryBatch& queries)
{
    validate_query_batch_shape(queries);
    validate_witness_batch_shape(batch);

    if (batch.query_count != queries.query_count || batch.feature_count != queries.feature_count) {
        throw std::runtime_error(
            "forest_training_then_inference_workflow: explicit inference witness shape must match inference queries");
    }
    if (batch.query_tree_ids.evaluations_on_domain(batch.query_count) !=
        queries.query_tree_ids.evaluations_on_domain(queries.query_count)) {
        throw std::runtime_error(
            "forest_training_then_inference_workflow: explicit inference witness query_tree_ids must match inference queries");
    }
    if (batch.lifted_inputs.evaluations_on_domain(batch.query_count * batch.feature_count) !=
        queries.lifted_inputs.evaluations_on_domain(queries.query_count * queries.feature_count)) {
        throw std::runtime_error(
            "forest_training_then_inference_workflow: explicit inference witness lifted_inputs must match inference queries");
    }
}

size_t exported_inference_merged_width(size_t feature_count)
{
    return cq_concatenated_table_lookup_merged_width({1, feature_count, feature_count, 1, 1});
}

void update_exported_inference_requirement(kzgSrsRequirement& requirement,
                                           size_t query_count,
                                           size_t feature_count,
                                           size_t exported_leaf_row_count)
{
    update_kzg_srs_requirement(requirement, query_count);
    update_kzg_srs_requirement(requirement, query_count * feature_count);
    const size_t merged_width = exported_inference_merged_width(feature_count);
    update_kzg_srs_requirement(requirement, exported_leaf_row_count * merged_width);
    update_kzg_srs_requirement(requirement, query_count * merged_width);
}

void update_exported_inference_range_requirement(kzgSrsRequirement& requirement,
                                                 size_t query_count,
                                                 size_t feature_count,
                                                 size_t input_bit_bound)
{
    lessCheckSrsEstimator estimator;
    estimator.add_check(input_bit_bound, query_count * feature_count, 2);
    update_kzg_srs_requirement(requirement, estimator.required_domain_size());
}

void update_exported_inference_plan_requirement(kzgSrsRequirement& requirement,
                                                size_t query_count,
                                                size_t feature_count,
                                                size_t committed_node_count,
                                                size_t exported_leaf_row_count,
                                                size_t input_bit_bound)
{
    update_kzg_srs_requirement(requirement, committed_node_count);
    update_kzg_srs_requirement(requirement, committed_node_count * feature_count);
    update_exported_inference_requirement(requirement, query_count, feature_count, exported_leaf_row_count);
    update_exported_inference_range_requirement(requirement, query_count, feature_count, input_bit_bound);
    require_kzg_constant_degree_claim(requirement);
}

size_t next_power_of_two_nonzero(size_t value)
{
    if (value == 0) {
        throw std::runtime_error("forest_training_then_inference_workflow: value must be non-zero");
    }
    size_t out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

size_t exported_leaf_row_count_from_training_inputs(const forestTrainingProofInputs& inputs)
{
    if (inputs.node_count == 0) {
        throw std::runtime_error("forest_training_then_inference_workflow: node_count must be non-zero");
    }

    const std::vector<Fr> leaf_mask_evals = inputs.node_leaf_mask.evaluations_on_domain(inputs.node_count);
    size_t leaf_count = 0;
    for (const Fr& value : leaf_mask_evals) {
        const size_t leaf_mask = canonical_size_t_from_fr(
            value, "forest_training_then_inference_workflow node leaf mask");
        if (leaf_mask > 1) {
            throw std::runtime_error("forest_training_then_inference_workflow: node leaf mask must be boolean");
        }
        leaf_count += leaf_mask;
    }
    if (leaf_count == 0) {
        throw std::runtime_error("forest_training_then_inference_workflow: forest has no leaf rows");
    }
    return next_power_of_two_nonzero(leaf_count);
}

template <class Fn>
double seconds_for(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    std::forward<Fn>(fn)();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

class committedForestTraversal {
public:
    std::unordered_map<size_t, size_t> root_by_tree_id;
    std::unordered_map<size_t, size_t> exported_row_by_leaf_index;
    std::vector<size_t> split_features;
    std::vector<size_t> thresholds;
    std::vector<size_t> left_children;
    std::vector<size_t> right_children;
    std::vector<uint8_t> split_mask;
    std::vector<uint8_t> leaf_mask;
};

committedForestTraversal build_committed_forest_traversal(const forestCommittedNodeTable& committed_forest,
                                                          const forestExportedLeafTable& exported_forest)
{
    validate_committed_forest_shape(committed_forest);

    if (exported_forest.row_count == 0 || exported_forest.feature_count == 0) {
        throw std::runtime_error("derive_exported_inference_witness_batch: exported forest shape must be non-zero");
    }
    if (committed_forest.feature_count != exported_forest.feature_count) {
        throw std::runtime_error(
            "derive_exported_inference_witness_batch: committed forest and exported forest feature counts must match");
    }

    committedForestTraversal out;
    out.split_features.assign(committed_forest.node_count, 0);
    out.thresholds.assign(committed_forest.node_count, 0);
    out.left_children.assign(committed_forest.node_count, 0);
    out.right_children.assign(committed_forest.node_count, 0);
    out.split_mask.assign(committed_forest.node_count, 0);
    out.leaf_mask.assign(committed_forest.node_count, 0);

    const std::vector<Fr> root_evals = committed_forest.root_indices.evaluations_on_domain(committed_forest.tree_count);
    const std::vector<Fr> split_mask_evals = committed_forest.split_node_mask.evaluations_on_domain(committed_forest.node_count);
    const std::vector<Fr> left_child_evals =
        committed_forest.left_child_indices.evaluations_on_domain(committed_forest.node_count);
    const std::vector<Fr> right_child_evals =
        committed_forest.right_child_indices.evaluations_on_domain(committed_forest.node_count);
    const std::vector<Fr> node_tree_evals = committed_forest.node_tree_ids.evaluations_on_domain(committed_forest.node_count);
    const std::vector<Fr> feature_selector_evals =
        committed_forest.feature_selector.evaluations_on_domain(committed_forest.node_count * committed_forest.feature_count);
    const std::vector<Fr> threshold_evals = committed_forest.thresholds.evaluations_on_domain(committed_forest.node_count);
    const std::vector<Fr> leaf_mask_evals = committed_forest.node_leaf_mask.evaluations_on_domain(committed_forest.node_count);

    for (size_t tree_slot = 0; tree_slot < committed_forest.tree_count; ++tree_slot) {
        const size_t root_index =
            canonical_size_t_from_fr(root_evals[tree_slot], "derive_exported_inference_witness_batch root index");
        if (root_index >= committed_forest.node_count) {
            throw std::runtime_error("derive_exported_inference_witness_batch: root index out of range");
        }
        const size_t tree_id =
            canonical_size_t_from_fr(node_tree_evals[root_index], "derive_exported_inference_witness_batch root tree id");
        if (!out.root_by_tree_id.emplace(tree_id, root_index).second) {
            throw std::runtime_error("derive_exported_inference_witness_batch: duplicate committed forest root tree id");
        }
    }

    for (size_t node = 0; node < committed_forest.node_count; ++node) {
        const size_t split_mask =
            canonical_size_t_from_fr(split_mask_evals[node], "derive_exported_inference_witness_batch split mask");
        const size_t leaf_mask =
            canonical_size_t_from_fr(leaf_mask_evals[node], "derive_exported_inference_witness_batch leaf mask");
        if (split_mask > 1 || leaf_mask > 1) {
            throw std::runtime_error("derive_exported_inference_witness_batch: split and leaf masks must be boolean");
        }
        if (split_mask != 0 && leaf_mask != 0) {
            throw std::runtime_error("derive_exported_inference_witness_batch: node cannot be both split and leaf");
        }

        out.split_mask[node] = static_cast<uint8_t>(split_mask);
        out.leaf_mask[node] = static_cast<uint8_t>(leaf_mask);
        out.thresholds[node] =
            canonical_size_t_from_fr(threshold_evals[node], "derive_exported_inference_witness_batch threshold");
        out.left_children[node] =
            canonical_size_t_from_fr(left_child_evals[node], "derive_exported_inference_witness_batch left child");
        out.right_children[node] =
            canonical_size_t_from_fr(right_child_evals[node], "derive_exported_inference_witness_batch right child");

        if (split_mask == 0) {
            continue;
        }

        if (out.left_children[node] >= committed_forest.node_count || out.right_children[node] >= committed_forest.node_count) {
            throw std::runtime_error("derive_exported_inference_witness_batch: child index out of range");
        }

        bool seen_selected_feature = false;
        for (size_t feature = 0; feature < committed_forest.feature_count; ++feature) {
            const size_t selector_index = node * committed_forest.feature_count + feature;
            const size_t selector_value = canonical_size_t_from_fr(
                feature_selector_evals[selector_index], "derive_exported_inference_witness_batch feature selector");
            if (selector_value > 1) {
                throw std::runtime_error("derive_exported_inference_witness_batch: feature selector must be boolean");
            }
            if (selector_value == 0) {
                continue;
            }
            if (seen_selected_feature) {
                throw std::runtime_error("derive_exported_inference_witness_batch: split node feature selector must be one-hot");
            }
            seen_selected_feature = true;
            out.split_features[node] = feature;
        }

        if (!seen_selected_feature) {
            throw std::runtime_error("derive_exported_inference_witness_batch: split node is missing a selected feature");
        }
    }

    const std::vector<Fr> exported_leaf_index_evals = exported_forest.leaf_indices.evaluations_on_domain(exported_forest.row_count);
    for (size_t row = 0; row < exported_forest.row_count; ++row) {
        const size_t leaf_index = canonical_size_t_from_fr(
            exported_leaf_index_evals[row], "derive_exported_inference_witness_batch exported leaf index");
        if (!out.exported_row_by_leaf_index.emplace(leaf_index, row).second) {
            throw std::runtime_error("derive_exported_inference_witness_batch: duplicate exported leaf index");
        }
    }

    return out;
}

void derive_prediction_witness(forestExportedInferenceWitnessBatch& batch,
                               size_t iteration_count,
                               size_t class_count)
{
    if (iteration_count == 0 || class_count == 0 ||
        batch.query_count % (iteration_count * class_count) != 0) {
        throw std::runtime_error(
            "forest_training_then_inference_workflow: invalid classification score layout");
    }
    const size_t sample_count = batch.query_count / (iteration_count * class_count);
    if (!isPowerOfTwo(sample_count)) {
        throw std::runtime_error(
            "forest_training_then_inference_workflow: classification sample count must be a power of two");
    }

    const std::vector<Fr> scores = batch.scores.evaluations_on_domain(batch.query_count);
    std::vector<Fr> class_scores(class_count * sample_count, Fr(0));
    for (size_t iteration = 0; iteration < iteration_count; ++iteration) {
        for (size_t cls = 0; cls < class_count; ++cls) {
            for (size_t sample = 0; sample < sample_count; ++sample) {
                const size_t score_index = iteration * class_count * sample_count + cls * sample_count + sample;
                class_scores[cls * sample_count + sample] += scores[score_index];
            }
        }
    }

    std::vector<Fr> predicted_scores(sample_count, Fr(0));
    std::vector<Fr> class_selector(class_count * sample_count, Fr(0));
    for (size_t sample = 0; sample < sample_count; ++sample) {
        size_t selected_class = 0;
        std::int64_t selected_score = canonical_signed_int64_from_fr(
            class_scores[sample], "derive_prediction_witness class score");
        for (size_t cls = 1; cls < class_count; ++cls) {
            const std::int64_t candidate = canonical_signed_int64_from_fr(
                class_scores[cls * sample_count + sample], "derive_prediction_witness class score");
            if (candidate > selected_score) {
                selected_class = cls;
                selected_score = candidate;
            }
        }
        predicted_scores[sample] = class_scores[selected_class * sample_count + sample];
        class_selector[selected_class * sample_count + sample] = Fr(1);
    }

    batch.class_scores = poly_from_evals(class_scores);
    batch.predicted_scores = poly_from_evals(predicted_scores);
    batch.class_selector = poly_from_evals(class_selector);
    batch.iteration_count = iteration_count;
    batch.class_count = class_count;
    batch.sample_count = sample_count;
}

void update_exported_classification_requirement(kzgSrsRequirement& requirement,
                                                size_t query_count,
                                                size_t feature_count,
                                                size_t leaf_count,
                                                size_t iteration_count,
                                                size_t class_count,
                                                size_t sample_count,
                                                size_t input_bit_bound,
                                                size_t score_bit_bound)
{
    update_exported_inference_requirement(requirement, query_count, feature_count, leaf_count);
    update_kzg_srs_requirement(requirement, iteration_count * class_count);
    update_kzg_srs_requirement(requirement, class_count * sample_count);
    update_kzg_srs_requirement(requirement, sample_count);

    lessCheckSrsEstimator estimator;
    estimator.add_check(input_bit_bound, query_count * feature_count, 2);
    estimator.add_check(score_bit_bound, class_count * sample_count);
    update_kzg_srs_requirement(requirement, estimator.required_domain_size());
    require_kzg_constant_degree_claim(requirement);
}

} // namespace

forestExportedInferenceWitnessBatch derive_exported_inference_witness_batch(
    const forestCommittedNodeTable& committed_forest,
    const forestExportedLeafTable& exported_forest,
    const forestInferenceQueryBatch& queries)
{
    validate_query_batch_shape(queries);

    validate_committed_forest_shape(committed_forest);
    if (queries.feature_count != committed_forest.feature_count || queries.feature_count != exported_forest.feature_count) {
        throw std::runtime_error(
            "derive_exported_inference_witness_batch: query feature count must match the committed and exported forest");
    }

    const committedForestTraversal traversal = build_committed_forest_traversal(committed_forest, exported_forest);

    const std::vector<Fr> query_tree_evals = queries.query_tree_ids.evaluations_on_domain(queries.query_count);
    const std::vector<Fr> query_input_evals =
        queries.lifted_inputs.evaluations_on_domain(queries.query_count * queries.feature_count);

    const std::vector<Fr> tree_evals = exported_forest.tree_indices.evaluations_on_domain(exported_forest.row_count);
    const std::vector<Fr> lower_evals =
        exported_forest.lower_bounds.evaluations_on_domain(exported_forest.row_count * exported_forest.feature_count);
    const std::vector<Fr> upper_evals =
        exported_forest.upper_bounds.evaluations_on_domain(exported_forest.row_count * exported_forest.feature_count);
    const std::vector<Fr> score_evals = exported_forest.scores.evaluations_on_domain(exported_forest.row_count);
    const std::vector<Fr> leaf_index_evals = exported_forest.leaf_indices.evaluations_on_domain(exported_forest.row_count);

    std::vector<Fr> matched_lower;
    std::vector<Fr> matched_upper;
    std::vector<Fr> matched_scores;
    std::vector<Fr> matched_leaf_indices;
    matched_lower.reserve(queries.query_count * queries.feature_count);
    matched_upper.reserve(queries.query_count * queries.feature_count);
    matched_scores.reserve(queries.query_count);
    matched_leaf_indices.reserve(queries.query_count);

    for (size_t query_row = 0; query_row < queries.query_count; ++query_row) {
        const size_t query_tree_id =
            canonical_size_t_from_fr(query_tree_evals[query_row], "derive_exported_inference_witness_batch query tree id");
        const auto root_it = traversal.root_by_tree_id.find(query_tree_id);
        if (root_it == traversal.root_by_tree_id.end()) {
            throw std::runtime_error("derive_exported_inference_witness_batch: query tree id has no committed root");
        }

        size_t node = root_it->second;
        while (traversal.leaf_mask[node] == 0) {
            if (traversal.split_mask[node] == 0) {
                throw std::runtime_error(
                    "derive_exported_inference_witness_batch: query reached a committed node that is neither split nor leaf");
            }
            const size_t selected_feature = traversal.split_features[node];
            const size_t query_index = query_row * queries.feature_count + selected_feature;
            const size_t query_value = canonical_size_t_from_fr(
                query_input_evals[query_index], "derive_exported_inference_witness_batch query input");
            node = (query_value < traversal.thresholds[node]) ? traversal.left_children[node] : traversal.right_children[node];
        }

        const auto exported_row_it = traversal.exported_row_by_leaf_index.find(node);
        if (exported_row_it == traversal.exported_row_by_leaf_index.end()) {
            throw std::runtime_error("derive_exported_inference_witness_batch: committed leaf is missing from the exported forest");
        }
        const size_t matched_row = exported_row_it->second;
        const size_t exported_tree_id = canonical_size_t_from_fr(
            tree_evals[matched_row], "derive_exported_inference_witness_batch exported tree id");
        if (exported_tree_id != query_tree_id) {
            throw std::runtime_error("derive_exported_inference_witness_batch: exported leaf tree id does not match the query tree id");
        }

        matched_scores.push_back(score_evals[matched_row]);
        matched_leaf_indices.push_back(leaf_index_evals[matched_row]);
        for (size_t feature = 0; feature < queries.feature_count; ++feature) {
            const size_t exported_index = matched_row * exported_forest.feature_count + feature;
            matched_lower.push_back(lower_evals[exported_index]);
            matched_upper.push_back(upper_evals[exported_index]);
        }
    }

    forestExportedInferenceWitnessBatch out;
    out.query_tree_ids = queries.query_tree_ids;
    out.lifted_inputs = queries.lifted_inputs;
    out.lower_bounds = poly_from_evals(matched_lower);
    out.upper_bounds = poly_from_evals(matched_upper);
    out.scores = poly_from_evals(matched_scores);
    out.leaf_indices = poly_from_evals(matched_leaf_indices);
    out.query_count = queries.query_count;
    out.feature_count = queries.feature_count;
    return out;
}

forestExportedInferencePlan assemble_exported_inference_plan(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const std::string& key_prefix)
{
    forestExportedInferencePlanProfile profile;
    return assemble_exported_inference_plan_with_profile(public_params, inputs, profile, key_prefix);
}

forestExportedInferencePlan assemble_exported_inference_plan_with_profile(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    forestExportedInferencePlanProfile& profile,
    const std::string& key_prefix)
{
    profile = {};
    forestExportedInferencePlan out;
    profile.total_seconds = seconds_for([&] {
        profile.ensure_srs_seconds = seconds_for([&] {
            ensure_kzg_srs_covers(
                public_params, required_exported_forest_inference_srs(inputs), "forest_training_then_inference_workflow");
        });
        profile.build_committed_forest_table_seconds = seconds_for([&] {
            out.committed_forest = forest_committed_node_table_from_training_inputs(inputs.training_inputs);
        });
        profile.commit_committed_forest_public_inputs_seconds = seconds_for([&] {
            out.committed_forest_public_inputs = commit_committed_forest_public_commitments(
                public_params, out.committed_forest, "committed_forest");
        });
        profile.export_leaf_table_seconds = seconds_for([&] {
            out.exported_leaf_table = export_leaf_table_from_committed_forest(out.committed_forest);
        });
        profile.build_inference_public_table_seconds = seconds_for([&] {
            out.inference_table = build_forest_inference_public_table_from_exported_forest(
                public_params,
                key_prefix + ".inference_table",
                out.exported_leaf_table,
                inputs.training_inputs.input_bit_bound,
                degree_bound_for_domain(out.exported_leaf_table.row_count * out.exported_leaf_table.feature_count));
        });
        profile.prepare_inference_witness_seconds = seconds_for([&] {
            if (inputs.explicit_inference_witness.has_value()) {
                validate_witness_matches_queries(*inputs.explicit_inference_witness, inputs.inference_queries);
                out.witness_batch = *inputs.explicit_inference_witness;
            } else {
                out.witness_batch =
                    derive_exported_inference_witness_batch(out.committed_forest, out.exported_leaf_table, inputs.inference_queries);
            }
            derive_prediction_witness(
                out.witness_batch, inputs.training_inputs.iteration_count, inputs.training_inputs.class_count);
            out.classification_table = build_forest_classification_public_table(
                build_forest_tree_layout_public_table(public_params,
                                                      key_prefix + ".classification_table.layout",
                                                      inputs.training_inputs.iteration_count,
                                                      inputs.training_inputs.class_count),
                build_forest_legality_public_table(out.exported_leaf_table.tree_indices,
                                                   out.exported_leaf_table.lower_bounds,
                                                   out.exported_leaf_table.upper_bounds,
                                                   out.exported_leaf_table.row_count,
                                                   out.exported_leaf_table.feature_count,
                                                   inputs.training_inputs.iteration_count * inputs.training_inputs.class_count,
                                                   out.exported_leaf_table.leaf_indices),
                out.inference_table,
                build_forest_prediction_public_table(inputs.training_inputs.iteration_count,
                                                     inputs.training_inputs.class_count,
                                                     fixed_point_encoded_bit_bound(inputs.training_inputs.score_encoding)));
        });
    });
    return out;
}

kzgSrsRequirement required_training_then_inference_srs(const forestTrainingThenInferenceInputs& inputs)
{
    kzgSrsRequirement requirement = required_forest_training_proof_srs(inputs.training_inputs);
    const forestWellFormednessInputs well_formedness_inputs =
        forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs);
    requirement = merge_kzg_srs_requirements(
        requirement, required_forest_well_formedness_srs(well_formedness_inputs));
    require_kzg_distinct_opening_points(requirement, 2);
    validate_query_batch_shape(inputs.inference_queries);
    update_exported_inference_requirement(requirement,
                                          inputs.inference_queries.query_count,
                                          inputs.inference_queries.feature_count,
                                          exported_leaf_row_count_from_training_inputs(inputs.training_inputs));
    update_exported_inference_range_requirement(requirement,
                                                inputs.inference_queries.query_count,
                                                inputs.inference_queries.feature_count,
                                                inputs.training_inputs.input_bit_bound);
    return requirement;
}

kzgSrsRequirement required_training_then_inference_srs(const forestTrainingThenInferenceVerifierBundle& bundle)
{
    kzgSrsRequirement requirement = required_forest_training_proof_srs(bundle.training_public_inputs);
    const forestWellFormednessPublicInputs well_formedness_public_inputs =
        forest_well_formedness_public_inputs_from_training_public_inputs(bundle.training_public_inputs);
    requirement = merge_kzg_srs_requirements(
        requirement, required_forest_well_formedness_srs(well_formedness_public_inputs));
    require_kzg_distinct_opening_points(requirement, 2);
    validate_witness_batch_shape(bundle.inference_batch_shape);
    update_exported_inference_requirement(requirement,
                                          bundle.inference_batch_shape.query_count,
                                          bundle.inference_batch_shape.feature_count,
                                          bundle.inference_table.row_fetch_table.row_count);
    update_exported_inference_range_requirement(requirement,
                                                bundle.inference_batch_shape.query_count,
                                                bundle.inference_batch_shape.feature_count,
                                                bundle.inference_table.input_bit_bound);
    return requirement;
}

kzgSrsRequirement required_exported_forest_inference_srs(const forestTrainingThenInferenceInputs& inputs)
{
    validate_query_batch_shape(inputs.inference_queries);
    kzgSrsRequirement requirement;
    update_exported_inference_plan_requirement(
        requirement,
        inputs.inference_queries.query_count,
        inputs.inference_queries.feature_count,
        inputs.training_inputs.node_count,
        exported_leaf_row_count_from_training_inputs(inputs.training_inputs),
        inputs.training_inputs.input_bit_bound);
    return requirement;
}

kzgSrsRequirement required_exported_forest_inference_srs(const forestInferencePublicTable& table,
                                                         const forestExportedInferenceWitnessBatch& batch)
{
    validate_witness_batch_shape(batch);

    kzgSrsRequirement requirement;
    update_exported_inference_requirement(requirement, batch.query_count, batch.feature_count, table.row_fetch_table.row_count);
    update_exported_inference_range_requirement(
        requirement, batch.query_count, batch.feature_count, table.input_bit_bound);
    require_kzg_constant_degree_claim(requirement);
    return requirement;
}

kzgSrsRequirement required_exported_forest_inference_srs(const forestInferencePublicTable& table,
                                                         const forestExportedInferenceBatchShape& batch_shape)
{
    validate_witness_batch_shape(batch_shape);

    kzgSrsRequirement requirement;
    update_exported_inference_requirement(
        requirement, batch_shape.query_count, batch_shape.feature_count, table.row_fetch_table.row_count);
    update_exported_inference_range_requirement(
        requirement, batch_shape.query_count, batch_shape.feature_count, table.input_bit_bound);
    require_kzg_constant_degree_claim(requirement);
    return requirement;
}

kzgSrsRequirement required_exported_forest_classification_srs(
    const forestTrainingThenInferenceInputs& inputs)
{
    validate_query_batch_shape(inputs.inference_queries);
    const size_t tree_rows = inputs.training_inputs.iteration_count * inputs.training_inputs.class_count;
    if (inputs.inference_queries.query_count % tree_rows != 0) {
        throw std::runtime_error("forest_training_then_inference_workflow: invalid classification query layout");
    }
    const size_t exported_leaf_row_count = exported_leaf_row_count_from_training_inputs(inputs.training_inputs);
    kzgSrsRequirement requirement;
    update_exported_inference_plan_requirement(requirement,
                                               inputs.inference_queries.query_count,
                                               inputs.inference_queries.feature_count,
                                               inputs.training_inputs.node_count,
                                               exported_leaf_row_count,
                                               inputs.training_inputs.input_bit_bound);
    update_exported_classification_requirement(requirement,
                                               inputs.inference_queries.query_count,
                                               inputs.inference_queries.feature_count,
                                               exported_leaf_row_count,
                                               inputs.training_inputs.iteration_count,
                                               inputs.training_inputs.class_count,
                                               inputs.inference_queries.query_count / tree_rows,
                                               inputs.training_inputs.input_bit_bound,
                                               fixed_point_encoded_bit_bound(inputs.training_inputs.score_encoding));
    return requirement;
}

kzgSrsRequirement required_exported_forest_classification_srs(
    const forestClassificationPublicTable& table, const forestExportedInferenceWitnessBatch& batch)
{
    validate_witness_batch_shape(batch);
    kzgSrsRequirement requirement;
    update_exported_classification_requirement(requirement, batch.query_count, batch.feature_count,
                                               table.inference_table.row_fetch_table.row_count,
                                               batch.iteration_count, batch.class_count, batch.sample_count,
                                               table.inference_table.input_bit_bound,
                                               table.prediction_table.score_bit_bound);
    return requirement;
}

kzgSrsRequirement required_exported_forest_classification_srs(
    const forestClassificationPublicTable& table, const forestExportedInferenceBatchShape& batch)
{
    kzgSrsRequirement requirement;
    update_exported_classification_requirement(requirement, batch.query_count, batch.feature_count,
                                               table.inference_table.row_fetch_table.row_count,
                                               batch.iteration_count, batch.class_count, batch.sample_count,
                                               table.inference_table.input_bit_bound,
                                               table.prediction_table.score_bit_bound);
    return requirement;
}

bool run_exported_forest_inference_workflow(
    ProofRuntime& runtime,
    const forestInferencePublicTable& table,
    const forestExportedInferenceWitnessBatch& batch,
    const std::string& label_prefix)
{
    validate_witness_batch_shape(batch);

    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.soundness_counter().set_stage(terraeSoundnessStage::Inference);
        auto& context = program.create_forest_inference_context(
            runtime.mode() == ProofMode::Prove ? table : table.verifier_view());
        context.add_check(
            witness_input(runtime, label_prefix + ".query_tree_ids", batch.query_tree_ids, vector_degree_bound(batch.query_count)),
            witness_input(
                runtime, label_prefix + ".lifted_inputs", batch.lifted_inputs, matrix_degree_bound(batch.query_count, batch.feature_count)),
            witness_input(
                runtime, label_prefix + ".lower_bounds", batch.lower_bounds, matrix_degree_bound(batch.query_count, batch.feature_count)),
            witness_input(
                runtime, label_prefix + ".upper_bounds", batch.upper_bounds, matrix_degree_bound(batch.query_count, batch.feature_count)),
            witness_input(runtime, label_prefix + ".scores", batch.scores, vector_degree_bound(batch.query_count)),
            witness_input(
                runtime, label_prefix + ".leaf_indices", batch.leaf_indices, vector_degree_bound(batch.query_count)),
            batch.query_count);
    });
}

bool add_exported_forest_inference_verifier_workflow_to_program(
    ProofProgram& program,
    const forestInferencePublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix)
{
    program.soundness_counter().set_stage(terraeSoundnessStage::Inference);
    validate_witness_batch_shape(batch_shape);

    auto& context = program.create_forest_inference_context(table.verifier_view());
    context.add_check(WitnessCommitmentInput::verifier(
                          label_prefix + ".query_tree_ids", vector_degree_bound(batch_shape.query_count)),
                      WitnessCommitmentInput::verifier(label_prefix + ".lifted_inputs",
                                                       matrix_degree_bound(batch_shape.query_count,
                                                                           batch_shape.feature_count)),
                      WitnessCommitmentInput::verifier(label_prefix + ".lower_bounds",
                                                       matrix_degree_bound(batch_shape.query_count,
                                                                           batch_shape.feature_count)),
                      WitnessCommitmentInput::verifier(label_prefix + ".upper_bounds",
                                                       matrix_degree_bound(batch_shape.query_count,
                                                                           batch_shape.feature_count)),
                      WitnessCommitmentInput::verifier(label_prefix + ".scores",
                                                       vector_degree_bound(batch_shape.query_count)),
                          WitnessCommitmentInput::verifier(label_prefix + ".leaf_indices",
                                                           vector_degree_bound(batch_shape.query_count)),
                          batch_shape.query_count);
    return true;
}

bool run_exported_forest_inference_verifier_workflow(
    ProofRuntime& runtime,
    const forestInferencePublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        add_exported_forest_inference_verifier_workflow_to_program(
            program, table, batch_shape, label_prefix);
    });
}

ProofArtifact prove_exported_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestInferencePublicTable& table,
    const forestExportedInferenceWitnessBatch& batch,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_exported_forest_inference_srs(table, batch), "forest_training_then_inference_workflow");
    return prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_exported_forest_inference_workflow(runtime, table, batch, label_prefix);
    });
}

VerificationResult verify_exported_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestInferencePublicTable& table,
    const forestExportedInferenceWitnessBatch& batch,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_exported_forest_inference_srs(table, batch), "forest_training_then_inference_workflow");
    return verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_exported_forest_inference_workflow(runtime, table, batch, label_prefix);
    });
}

VerificationResult verify_exported_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestInferencePublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params,
                          required_exported_forest_inference_srs(table, batch_shape),
                          "forest_training_then_inference_workflow");
    return verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_exported_forest_inference_verifier_workflow(runtime, table, batch_shape, label_prefix);
    });
}

forestExportedInferenceVerifierPreprocessing prepare_exported_forest_inference_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestInferencePublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params,
                          required_exported_forest_inference_srs(table, batch_shape),
                          "forest_training_then_inference_workflow");

    forestExportedInferenceVerifierPreprocessing out;
    out.runtime = ProofRuntime::verifier_ptr(public_params, artifact.proof);
    out.program = std::make_unique<ProofProgram>(*out.runtime);
    add_exported_forest_inference_verifier_workflow_to_program(
        *out.program, table.verifier_view(), batch_shape, label_prefix);
    if (!out.program->prepare_verifier_preprocessing()) {
        throw std::runtime_error("forest_training_then_inference_workflow: inference verifier preprocessing failed");
    }
    if (!out.program->complete_verifier_opening_preprocessing()) {
        throw std::runtime_error(
            "forest_training_then_inference_workflow: inference verifier opening preprocessing failed");
    }
    return out;
}

VerificationResult verify_exported_forest_inference_preprocessed(
    forestExportedInferenceVerifierPreprocessing& preprocessing)
{
    if (!preprocessing.runtime || !preprocessing.program) {
        throw std::runtime_error("forest_training_then_inference_workflow: missing inference verifier preprocessing");
    }

    VerificationResult result;
    result.accepted = preprocessing.program->verify_preprocessed();
    result.proof_fully_consumed = preprocessing.runtime->proof_fully_consumed();
    result.transcript_state = preprocessing.runtime->transcript_state();
    return result;
}

bool run_exported_forest_classification_workflow(
    ProofRuntime& runtime, const forestClassificationPublicTable& table,
    const forestExportedInferenceWitnessBatch& batch, const std::string& label_prefix)
{
    validate_witness_batch_shape(batch);
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.soundness_counter().set_stage(terraeSoundnessStage::Inference);
        auto& context = program.create_forest_classification_context(
            runtime.mode() == ProofMode::Prove ? table : table.verifier_view());
        context.add_check(
            witness_input(runtime, label_prefix + ".query_tree_ids", batch.query_tree_ids, vector_degree_bound(batch.query_count)),
            witness_input(runtime, label_prefix + ".lifted_inputs", batch.lifted_inputs, matrix_degree_bound(batch.query_count, batch.feature_count)),
            witness_input(runtime, label_prefix + ".lower_bounds", batch.lower_bounds, matrix_degree_bound(batch.query_count, batch.feature_count)),
            witness_input(runtime, label_prefix + ".upper_bounds", batch.upper_bounds, matrix_degree_bound(batch.query_count, batch.feature_count)),
            witness_input(runtime, label_prefix + ".scores", batch.scores, vector_degree_bound(batch.query_count)),
            witness_input(runtime, label_prefix + ".leaf_indices", batch.leaf_indices, vector_degree_bound(batch.query_count)),
            witness_input(runtime, label_prefix + ".class_scores", batch.class_scores, matrix_degree_bound(batch.class_count, batch.sample_count)),
            witness_input(runtime, label_prefix + ".predicted_scores", batch.predicted_scores, vector_degree_bound(batch.sample_count)),
            witness_input(runtime, label_prefix + ".class_selector", batch.class_selector, matrix_degree_bound(batch.class_count, batch.sample_count)),
            batch.query_count);
    });
}

bool add_exported_forest_classification_verifier_workflow_to_program(
    ProofProgram& program, const forestClassificationPublicTable& table,
    const forestExportedInferenceBatchShape& batch, const std::string& label_prefix)
{
    program.soundness_counter().set_stage(terraeSoundnessStage::Inference);
    validate_witness_batch_shape(batch);

    auto& context = program.create_forest_classification_context(table.verifier_view());
    context.add_check(
        WitnessCommitmentInput::verifier(label_prefix + ".query_tree_ids", vector_degree_bound(batch.query_count)),
        WitnessCommitmentInput::verifier(label_prefix + ".lifted_inputs", matrix_degree_bound(batch.query_count, batch.feature_count)),
        WitnessCommitmentInput::verifier(label_prefix + ".lower_bounds", matrix_degree_bound(batch.query_count, batch.feature_count)),
        WitnessCommitmentInput::verifier(label_prefix + ".upper_bounds", matrix_degree_bound(batch.query_count, batch.feature_count)),
        WitnessCommitmentInput::verifier(label_prefix + ".scores", vector_degree_bound(batch.query_count)),
        WitnessCommitmentInput::verifier(label_prefix + ".leaf_indices", vector_degree_bound(batch.query_count)),
        WitnessCommitmentInput::verifier(label_prefix + ".class_scores", matrix_degree_bound(batch.class_count, batch.sample_count)),
        WitnessCommitmentInput::verifier(label_prefix + ".predicted_scores", vector_degree_bound(batch.sample_count)),
        WitnessCommitmentInput::verifier(label_prefix + ".class_selector", matrix_degree_bound(batch.class_count, batch.sample_count)),
        batch.query_count);
    return true;
}

ProofArtifact prove_exported_forest_classification_with_runtime(
    const kzgPublicParameters& public_params, const forestClassificationPublicTable& table,
    const forestExportedInferenceWitnessBatch& batch, const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params, required_exported_forest_classification_srs(table, batch),
                          "forest_training_then_inference_workflow");
    return prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_exported_forest_classification_workflow(runtime, table, batch, label_prefix);
    });
}

VerificationResult verify_exported_forest_classification_with_runtime(
    const kzgPublicParameters& public_params, const ProofArtifact& artifact,
    const forestClassificationPublicTable& table, const forestExportedInferenceBatchShape& batch,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params, required_exported_forest_classification_srs(table, batch),
                          "forest_training_then_inference_workflow");
    return verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            return add_exported_forest_classification_verifier_workflow_to_program(
                program, table, batch, label_prefix);
        });
    });
}

forestExportedClassificationVerifierPreprocessing prepare_exported_forest_classification_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestClassificationPublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params,
                          required_exported_forest_classification_srs(table, batch_shape),
                          "forest_training_then_inference_workflow");

    forestExportedClassificationVerifierPreprocessing out;
    out.runtime = ProofRuntime::verifier_ptr(public_params, artifact.proof);
    out.program = std::make_unique<ProofProgram>(*out.runtime);
    add_exported_forest_classification_verifier_workflow_to_program(
        *out.program, table.verifier_view(), batch_shape, label_prefix);
    if (!out.program->prepare_verifier_preprocessing()) {
        throw std::runtime_error("forest_training_then_inference_workflow: classification verifier preprocessing failed");
    }
    if (!out.program->complete_verifier_opening_preprocessing()) {
        throw std::runtime_error(
            "forest_training_then_inference_workflow: classification verifier opening preprocessing failed");
    }
    return out;
}

VerificationResult verify_exported_forest_classification_preprocessed(
    forestExportedClassificationVerifierPreprocessing& preprocessing)
{
    if (!preprocessing.runtime || !preprocessing.program) {
        throw std::runtime_error("forest_training_then_inference_workflow: missing classification verifier preprocessing");
    }

    VerificationResult result;
    result.accepted = preprocessing.program->verify_preprocessed();
    result.proof_fully_consumed = preprocessing.runtime->proof_fully_consumed();
    result.transcript_state = preprocessing.runtime->transcript_state();
    return result;
}

forestTrainingThenInferenceArtifact prove_training_then_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_training_then_inference_srs(inputs), "forest_training_then_inference_workflow");
    const forestExportedInferencePlan plan =
        assemble_exported_inference_plan(public_params, inputs, label_prefix + ".export");
    return prove_training_then_inference_with_runtime(public_params, inputs, plan, label_prefix);
}

forestTrainingThenInferenceArtifact prove_training_then_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const forestExportedInferencePlan& plan,
    const std::string& label_prefix)
{
    forestTrainingThenInferenceArtifact out;
    const forestWellFormednessInputs well_formedness_inputs =
        forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs);
    out.well_formedness_artifact = prove_forest_well_formedness_with_runtime(
        public_params,
        well_formedness_inputs,
        plan.committed_forest_public_inputs,
        label_prefix + ".well_formedness");
    out.training_artifact = prove_forest_training_proof_with_runtime(
        public_params,
        inputs.training_inputs,
        plan.committed_forest_public_inputs,
        label_prefix + ".training");
    out.exported_leaf_table = plan.exported_leaf_table;
    out.inference_artifact = prove_exported_forest_inference_with_runtime(
        public_params, plan.inference_table, plan.witness_batch, label_prefix + ".inference");
    return out;
}

ProofArtifact prove_forest_training_with_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingProofInputs& training_inputs,
    const forestWellFormednessInputs& well_formedness_inputs,
    const forestCommittedNodePublicCommitments& committed_forest_public_inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params,
                          required_forest_training_proof_srs(training_inputs),
                          "forest_training_with_well_formedness");
    ensure_kzg_srs_covers(public_params,
                          required_forest_well_formedness_srs(well_formedness_inputs),
                          "forest_training_with_well_formedness");
    return prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            add_forest_well_formedness_workflow_to_program(
                runtime,
                program,
                well_formedness_inputs,
                label_prefix + ".well_formedness",
                &committed_forest_public_inputs);
            add_forest_training_proof_workflow_to_program(
                runtime,
                program,
                training_inputs,
                label_prefix + ".training",
                &committed_forest_public_inputs);
        });
    });
}

VerificationResult verify_forest_training_with_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestTrainingProofPublicInputs& training_inputs,
    const forestWellFormednessPublicInputs& well_formedness_inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params,
                          required_forest_training_proof_srs(training_inputs),
                          "forest_training_with_well_formedness");
    ensure_kzg_srs_covers(public_params,
                          required_forest_well_formedness_srs(well_formedness_inputs),
                          "forest_training_with_well_formedness");
    const forestTrainingProofInputs verifier_training_inputs =
        forest_training_proof_verifier_inputs_from_public_inputs(training_inputs);
    return verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            add_forest_well_formedness_verifier_workflow_to_program(
                runtime,
                program,
                well_formedness_inputs,
                label_prefix + ".well_formedness");
            add_forest_training_proof_workflow_to_program(
                runtime,
                program,
                verifier_training_inputs,
                label_prefix + ".training",
                &training_inputs.committed_forest);
        });
    });
}

forestTrainingWithWellFormednessVerifierPreprocessing
prepare_forest_training_with_well_formedness_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestTrainingProofPublicInputs& training_inputs,
    const forestWellFormednessPublicInputs& well_formedness_inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params,
                          required_forest_training_proof_srs(training_inputs),
                          "forest_training_with_well_formedness");
    ensure_kzg_srs_covers(public_params,
                          required_forest_well_formedness_srs(well_formedness_inputs),
                          "forest_training_with_well_formedness");

    forestTrainingWithWellFormednessVerifierPreprocessing out;
    out.runtime = ProofRuntime::verifier_ptr(public_params, artifact.proof);
    out.program = std::make_unique<ProofProgram>(*out.runtime);
    add_forest_well_formedness_verifier_workflow_to_program(
        *out.runtime,
        *out.program,
        well_formedness_inputs,
        label_prefix + ".well_formedness");
    const forestTrainingProofInputs verifier_training_inputs =
        forest_training_proof_verifier_inputs_from_public_inputs(training_inputs);
    add_forest_training_proof_workflow_to_program(
        *out.runtime,
        *out.program,
        verifier_training_inputs,
        label_prefix + ".training",
        &training_inputs.committed_forest);
    if (!out.program->prepare_verifier_preprocessing()) {
        throw std::runtime_error(
            "forest_training_with_well_formedness: verifier preprocessing failed");
    }
    if (!out.program->complete_verifier_opening_preprocessing()) {
        throw std::runtime_error(
            "forest_training_with_well_formedness: opening preprocessing failed");
    }
    return out;
}

VerificationResult verify_forest_training_with_well_formedness_preprocessed(
    forestTrainingWithWellFormednessVerifierPreprocessing& preprocessing)
{
    if (!preprocessing.runtime || !preprocessing.program) {
        throw std::runtime_error(
            "forest_training_with_well_formedness: missing verifier preprocessing");
    }
    VerificationResult result;
    result.accepted = preprocessing.program->verify_preprocessed();
    result.proof_fully_consumed = preprocessing.runtime->proof_fully_consumed();
    result.transcript_state = preprocessing.runtime->transcript_state();
    return result;
}

forestTrainingThenInferenceVerifierBundle prepare_training_then_inference_verifier_bundle(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const forestTrainingThenInferenceArtifact& artifact,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_training_then_inference_srs(inputs), "forest_training_then_inference_workflow");
    const forestExportedInferencePlan plan =
        assemble_exported_inference_plan(public_params, inputs, label_prefix + ".export");
    return prepare_training_then_inference_verifier_bundle(inputs, artifact, plan);
}

forestTrainingThenInferenceVerifierBundle prepare_training_then_inference_verifier_bundle(
    const forestTrainingThenInferenceInputs& inputs,
    const forestTrainingThenInferenceArtifact& artifact,
    const forestExportedInferencePlan& plan)
{
    forestTrainingThenInferenceVerifierBundle out;
    out.well_formedness_artifact = artifact.well_formedness_artifact;
    out.training_artifact = artifact.training_artifact;
    out.inference_artifact = artifact.inference_artifact;
    out.training_public_inputs =
        forest_training_proof_public_inputs_from_inputs(plan.committed_forest_public_inputs, inputs.training_inputs);
    out.inference_table = plan.inference_table.verifier_view();
    out.inference_batch_shape = {plan.witness_batch.query_count, plan.witness_batch.feature_count};
    return out;
}

forestTrainingThenInferencePreparedRun prove_and_prepare_training_then_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_training_then_inference_srs(inputs), "forest_training_then_inference_workflow");
    forestTrainingThenInferencePreparedRun out;
    out.exported_inference_plan =
        assemble_exported_inference_plan(public_params, inputs, label_prefix + ".export");
    out.artifact = prove_training_then_inference_with_runtime(public_params, inputs, out.exported_inference_plan, label_prefix);
    out.verifier_bundle =
        prepare_training_then_inference_verifier_bundle(inputs, out.artifact, out.exported_inference_plan);
    return out;
}

forestTrainingThenInferenceVerificationResult verify_training_then_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceVerifierBundle& bundle,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_training_then_inference_srs(bundle), "forest_training_then_inference_workflow");
    forestTrainingThenInferenceVerificationResult out;
    const forestWellFormednessPublicInputs well_formedness_public_inputs =
        forest_well_formedness_public_inputs_from_training_public_inputs(bundle.training_public_inputs);
    out.well_formedness_result = verify_forest_well_formedness_with_runtime(
        public_params,
        bundle.well_formedness_artifact,
        well_formedness_public_inputs,
        label_prefix + ".well_formedness");
    out.training_result = verify_forest_training_proof_with_runtime(
        public_params, bundle.training_artifact, bundle.training_public_inputs, label_prefix + ".training");

    out.accepted = out.well_formedness_result.accepted && out.training_result.accepted;
    out.proof_fully_consumed =
        out.well_formedness_result.proof_fully_consumed && out.training_result.proof_fully_consumed;

    out.inference_result = verify_exported_forest_inference_with_runtime(public_params,
                                                                         bundle.inference_artifact,
                                                                         bundle.inference_table,
                                                                         bundle.inference_batch_shape,
                                                                         label_prefix + ".inference");
    out.accepted = out.accepted && out.inference_result.accepted;
    out.proof_fully_consumed = out.proof_fully_consumed && out.inference_result.proof_fully_consumed;

    return out;
}
