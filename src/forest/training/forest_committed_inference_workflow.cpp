#include "forest_committed_inference_workflow.hpp"

#include "matrix_utils.hpp"
#include "training_runtime_utils.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void validate_query_batch_shape(const forestInferenceQueryBatch& queries)
{
    if (queries.query_count == 0 || queries.feature_count == 0) {
        throw std::runtime_error("forest_committed_inference_workflow: query_count and feature_count must be non-zero");
    }
    if (!isPowerOfTwo(queries.query_count) || !isPowerOfTwo(queries.feature_count)) {
        throw std::runtime_error("forest_committed_inference_workflow: query_count and feature_count must be powers of two");
    }

    validate_matrix_shape(queries.query_count, 1, "forest_committed_inference_workflow query_tree_ids");
    validate_matrix_shape(queries.query_count, queries.feature_count, "forest_committed_inference_workflow lifted_inputs");
}

void validate_committed_forest_shape(const forestCommittedNodeTable& committed_forest)
{
    if (committed_forest.tree_count == 0 || committed_forest.node_count == 0 || committed_forest.feature_count == 0) {
        throw std::runtime_error("forest_committed_inference_workflow: committed forest shape must be non-zero");
    }

    validate_matrix_shape(committed_forest.tree_count, 1, "forest_committed_inference_workflow root indices");
    validate_matrix_shape(committed_forest.node_count, 1, "forest_committed_inference_workflow committed node vectors");
    validate_matrix_shape(committed_forest.node_count,
                          committed_forest.feature_count,
                          "forest_committed_inference_workflow committed node boxes");
    validate_matrix_shape(committed_forest.node_count,
                          committed_forest.feature_count,
                          "forest_committed_inference_workflow feature selector");
}

void validate_witness_batch_shape(const forestCommittedInferenceWitnessBatch& batch)
{
    if (batch.query_count == 0 || batch.feature_count == 0) {
        throw std::runtime_error("forest_committed_inference_workflow: witness batch shape must be non-zero");
    }
    if (!isPowerOfTwo(batch.query_count) || !isPowerOfTwo(batch.feature_count)) {
        throw std::runtime_error("forest_committed_inference_workflow: witness batch dimensions must be powers of two");
    }

    validate_matrix_shape(batch.query_count, 1, "forest_committed_inference_workflow leaf_indices");
    validate_matrix_shape(batch.query_count, batch.feature_count, "forest_committed_inference_workflow leaf_lower_bounds");
    validate_matrix_shape(batch.query_count, batch.feature_count, "forest_committed_inference_workflow leaf_upper_bounds");
    validate_matrix_shape(batch.query_count, 1, "forest_committed_inference_workflow tree_scores");
}

void validate_witness_matches_queries(const forestCommittedInferenceWitnessBatch& batch,
                                      const forestInferenceQueryBatch& queries)
{
    validate_witness_batch_shape(batch);
    validate_query_batch_shape(queries);
    if (batch.query_count != queries.query_count || batch.feature_count != queries.feature_count) {
        throw std::runtime_error("forest_committed_inference_workflow: explicit witness shape must match query shape");
    }
}

size_t canonical_size_t_or_throw(const Fr& value, const char* context)
{
    return canonical_size_t_from_fr(value, context);
}

ProtocolPolynomialInput forest_public_input(ProofRuntime& runtime,
                                            const std::string& key,
                                            const poly& polynomial,
                                            const Group1* commitment,
                                            size_t degree_bound)
{
    if (runtime.mode() == ProofMode::Prove) {
        return PublicCommitmentInput::prover(key, polynomial, degree_bound);
    }
    if (commitment == nullptr) {
        throw std::runtime_error(
            "forest_committed_inference_workflow: missing committed-forest public commitment in verifier mode");
    }
    return PublicCommitmentInput::verifier(key, *commitment, degree_bound);
}

ProtocolPolynomialInput query_public_input(ProofRuntime& runtime,
                                           const std::string& key,
                                           const poly& polynomial,
                                           const Group1* commitment,
                                           size_t degree_bound)
{
    if (runtime.mode() == ProofMode::Prove) {
        if (commitment != nullptr) {
            return PublicCommitmentInput::prover(key, polynomial, degree_bound);
        }
        return PublicPolynomialInput(key, polynomial, degree_bound);
    }
    if (commitment != nullptr) {
        return PublicCommitmentInput::verifier(key, *commitment, degree_bound);
    }
    return PublicPolynomialInput(key, polynomial, degree_bound);
}

class committedForestTraversal {
public:
    std::unordered_map<size_t, size_t> root_by_tree_id;
    std::vector<size_t> split_features;
    std::vector<size_t> thresholds;
    std::vector<size_t> left_children;
    std::vector<size_t> right_children;
    std::vector<uint8_t> split_mask;
    std::vector<uint8_t> leaf_mask;
    std::vector<Fr> lower_bounds;
    std::vector<Fr> upper_bounds;
    std::vector<Fr> leaf_values;
};

committedForestTraversal build_committed_forest_traversal(const forestCommittedNodeTable& committed_forest)
{
    validate_committed_forest_shape(committed_forest);

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

    out.lower_bounds =
        committed_forest.lower_bounds.evaluations_on_domain(committed_forest.node_count * committed_forest.feature_count);
    out.upper_bounds =
        committed_forest.upper_bounds.evaluations_on_domain(committed_forest.node_count * committed_forest.feature_count);
    out.leaf_values = committed_forest.node_leaf_values.evaluations_on_domain(committed_forest.node_count);

    for (size_t tree_slot = 0; tree_slot < committed_forest.tree_count; ++tree_slot) {
        const size_t root_index =
            canonical_size_t_or_throw(root_evals[tree_slot], "forest_committed_inference_workflow root index");
        if (root_index >= committed_forest.node_count) {
            throw std::runtime_error("forest_committed_inference_workflow: root index out of range");
        }
        const size_t tree_id =
            canonical_size_t_or_throw(node_tree_evals[root_index], "forest_committed_inference_workflow root tree id");
        if (!out.root_by_tree_id.emplace(tree_id, root_index).second) {
            throw std::runtime_error("forest_committed_inference_workflow: duplicate root tree id");
        }
    }

    for (size_t node = 0; node < committed_forest.node_count; ++node) {
        const size_t split_mask =
            canonical_size_t_or_throw(split_mask_evals[node], "forest_committed_inference_workflow split mask");
        const size_t leaf_mask =
            canonical_size_t_or_throw(leaf_mask_evals[node], "forest_committed_inference_workflow leaf mask");
        if (split_mask > 1 || leaf_mask > 1) {
            throw std::runtime_error("forest_committed_inference_workflow: split and leaf masks must be boolean");
        }
        if (split_mask != 0 && leaf_mask != 0) {
            throw std::runtime_error("forest_committed_inference_workflow: node cannot be both split and leaf");
        }

        out.split_mask[node] = static_cast<uint8_t>(split_mask);
        out.leaf_mask[node] = static_cast<uint8_t>(leaf_mask);
        out.thresholds[node] =
            canonical_size_t_or_throw(threshold_evals[node], "forest_committed_inference_workflow threshold");
        out.left_children[node] =
            canonical_size_t_or_throw(left_child_evals[node], "forest_committed_inference_workflow left child");
        out.right_children[node] =
            canonical_size_t_or_throw(right_child_evals[node], "forest_committed_inference_workflow right child");

        if (split_mask == 0) {
            continue;
        }
        if (out.left_children[node] >= committed_forest.node_count || out.right_children[node] >= committed_forest.node_count) {
            throw std::runtime_error("forest_committed_inference_workflow: child index out of range");
        }

        bool seen_selected_feature = false;
        for (size_t feature = 0; feature < committed_forest.feature_count; ++feature) {
            const size_t selector_index = node * committed_forest.feature_count + feature;
            const size_t selector_value = canonical_size_t_or_throw(
                feature_selector_evals[selector_index], "forest_committed_inference_workflow feature selector");
            if (selector_value > 1) {
                throw std::runtime_error("forest_committed_inference_workflow: feature selector must be boolean");
            }
            if (selector_value == 0) {
                continue;
            }
            if (seen_selected_feature) {
                throw std::runtime_error("forest_committed_inference_workflow: split feature selector must be one-hot");
            }
            seen_selected_feature = true;
            out.split_features[node] = feature;
        }
        if (!seen_selected_feature) {
            throw std::runtime_error("forest_committed_inference_workflow: split node is missing a selected feature");
        }
    }

    return out;
}

} // namespace

forestInferenceQueryPublicCommitments commit_forest_inference_query_public_commitments(
    const kzgPublicParameters& public_params,
    const forestInferenceQueryBatch& queries,
    const std::string& key_prefix)
{
    validate_query_batch_shape(queries);

    kzgProtocol kzg;
    kzg.setup(public_params);

    forestInferenceQueryPublicCommitments out;
    out.query_tree_ids =
        kzg.commit_public(key_prefix + ".query_tree_ids", queries.query_tree_ids, queries.query_count - 1).com;
    out.lifted_inputs =
        kzg.commit_public(key_prefix + ".lifted_inputs",
                          queries.lifted_inputs,
                          queries.query_count * queries.feature_count - 1)
            .com;
    return out;
}

forestCommittedInferencePublicInputs forest_committed_inference_public_inputs_from_inputs(
    const kzgPublicParameters& public_params,
    const forestCommittedInferenceInputs& inputs,
    const std::string& forest_key_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_committed_forest_inference_srs(inputs), "forest_committed_inference_workflow");
    forestCommittedInferencePublicInputs out;
    out.committed_forest =
        commit_committed_forest_public_commitments(public_params, inputs.committed_forest, forest_key_prefix);
    out.queries = inputs.queries;
    out.query_commitments = commit_forest_inference_query_public_commitments(
        public_params, inputs.queries, forest_key_prefix + ".queries");
    out.input_bit_bound = inputs.input_bit_bound;
    return out;
}

kzgSrsRequirement required_committed_forest_inference_srs(const forestCommittedInferenceInputs& inputs)
{
    validate_committed_forest_shape(inputs.committed_forest);
    validate_query_batch_shape(inputs.queries);

    kzgSrsRequirement requirement;
    update_kzg_srs_requirement(requirement, inputs.committed_forest.tree_count);
    update_kzg_srs_requirement(requirement, inputs.committed_forest.node_count);
    update_kzg_srs_requirement(requirement, inputs.committed_forest.node_count * inputs.committed_forest.feature_count);
    update_kzg_srs_requirement(requirement, inputs.queries.query_count);
    update_kzg_srs_requirement(requirement, inputs.queries.query_count * inputs.queries.feature_count);
    require_kzg_distinct_opening_points(requirement, 1);
    return requirement;
}

kzgSrsRequirement required_committed_forest_inference_srs(const forestCommittedInferencePublicInputs& inputs)
{
    validate_query_batch_shape(inputs.queries);

    kzgSrsRequirement requirement;
    update_kzg_srs_requirement(requirement, inputs.committed_forest.tree_count);
    update_kzg_srs_requirement(requirement, inputs.committed_forest.node_count);
    update_kzg_srs_requirement(requirement, inputs.committed_forest.node_count * inputs.committed_forest.feature_count);
    update_kzg_srs_requirement(requirement, inputs.queries.query_count);
    update_kzg_srs_requirement(requirement, inputs.queries.query_count * inputs.queries.feature_count);
    require_kzg_distinct_opening_points(requirement, 1);
    return requirement;
}

forestCommittedInferenceWitnessBatch derive_committed_inference_witness_batch(
    const forestCommittedNodeTable& committed_forest,
    const forestInferenceQueryBatch& queries)
{
    validate_committed_forest_shape(committed_forest);
    validate_query_batch_shape(queries);
    if (queries.feature_count != committed_forest.feature_count) {
        throw std::runtime_error(
            "forest_committed_inference_workflow: query feature count must match committed forest feature count");
    }

    const committedForestTraversal traversal = build_committed_forest_traversal(committed_forest);
    const std::vector<Fr> query_tree_evals = queries.query_tree_ids.evaluations_on_domain(queries.query_count);
    const std::vector<Fr> query_input_evals =
        queries.lifted_inputs.evaluations_on_domain(queries.query_count * queries.feature_count);

    std::vector<Fr> leaf_index_evals;
    std::vector<Fr> lower_evals;
    std::vector<Fr> upper_evals;
    std::vector<Fr> score_evals;
    leaf_index_evals.reserve(queries.query_count);
    lower_evals.reserve(queries.query_count * queries.feature_count);
    upper_evals.reserve(queries.query_count * queries.feature_count);
    score_evals.reserve(queries.query_count);

    for (size_t query_row = 0; query_row < queries.query_count; ++query_row) {
        const size_t query_tree_id = canonical_size_t_or_throw(
            query_tree_evals[query_row], "forest_committed_inference_workflow query tree id");
        const auto root_it = traversal.root_by_tree_id.find(query_tree_id);
        if (root_it == traversal.root_by_tree_id.end()) {
            throw std::runtime_error("forest_committed_inference_workflow: query tree id has no committed root");
        }

        size_t node = root_it->second;
        while (traversal.leaf_mask[node] == 0) {
            if (traversal.split_mask[node] == 0) {
                throw std::runtime_error("forest_committed_inference_workflow: query row " +
                                         std::to_string(query_row) + " reached node " + std::to_string(node) +
                                         " that is neither split nor leaf");
            }
            const size_t selected_feature = traversal.split_features[node];
            const size_t query_index = query_row * queries.feature_count + selected_feature;
            const size_t query_value = canonical_size_t_or_throw(
                query_input_evals[query_index], "forest_committed_inference_workflow query input");
            node = (query_value < traversal.thresholds[node]) ? traversal.left_children[node] : traversal.right_children[node];
        }

        leaf_index_evals.push_back(field_from_size_t(node));
        score_evals.push_back(traversal.leaf_values[node]);
        for (size_t feature = 0; feature < queries.feature_count; ++feature) {
            const size_t offset = node * queries.feature_count + feature;
            lower_evals.push_back(traversal.lower_bounds[offset]);
            upper_evals.push_back(traversal.upper_bounds[offset]);
        }
    }

    forestCommittedInferenceWitnessBatch out;
    out.leaf_indices = poly_from_evals(leaf_index_evals);
    out.leaf_lower_bounds = poly_from_evals(lower_evals);
    out.leaf_upper_bounds = poly_from_evals(upper_evals);
    out.tree_scores = poly_from_evals(score_evals);
    out.query_count = queries.query_count;
    out.feature_count = queries.feature_count;
    return out;
}

bool add_committed_forest_inference_workflow_to_program(ProofRuntime& runtime,
                                                        ProofProgram& program,
                                                        const forestCommittedInferenceInputs& inputs,
                                                        const std::string& label_prefix,
                                                        const forestCommittedNodePublicCommitments* committed_forest_public_inputs)
{
    program.soundness_counter().set_stage(terraeSoundnessStage::Inference);
    validate_committed_forest_shape(inputs.committed_forest);
    validate_query_batch_shape(inputs.queries);
    if (inputs.input_bit_bound == 0) {
        throw std::runtime_error("forest_committed_inference_workflow: input_bit_bound must be non-zero");
    }

    const forestCommittedInferenceWitnessBatch witness_batch = inputs.explicit_witness.has_value()
                                                                   ? *inputs.explicit_witness
                                                                   : derive_committed_inference_witness_batch(
                                                                         inputs.committed_forest, inputs.queries);
    validate_witness_matches_queries(witness_batch, inputs.queries);

    forestTrainingInferenceWitness witness;
    workflow_profile_scope_void("forest_committed_inference_workflow.bind_queries_and_witness", [&] {
        witness.query_tree_ids = program.bind_virtual(query_public_input(runtime,
                                                                        label_prefix + ".query_tree_ids",
                                                                        inputs.queries.query_tree_ids,
                                                                        nullptr,
                                                                        vector_degree_bound(inputs.queries.query_count)));
        witness.lifted_inputs = program.bind_virtual(query_public_input(runtime,
                                                                       label_prefix + ".lifted_inputs",
                                                                       inputs.queries.lifted_inputs,
                                                                       nullptr,
                                                                       matrix_degree_bound(inputs.queries.query_count,
                                                                                           inputs.queries.feature_count)));
        witness.leaf_indices = program.bind_virtual(witness_input(
            runtime, label_prefix + ".leaf_indices", witness_batch.leaf_indices, vector_degree_bound(inputs.queries.query_count)));
        witness.leaf_lower_bounds = program.bind_virtual(witness_input(
            runtime,
            label_prefix + ".leaf_lower_bounds",
            witness_batch.leaf_lower_bounds,
            matrix_degree_bound(inputs.queries.query_count, inputs.queries.feature_count)));
        witness.leaf_upper_bounds = program.bind_virtual(witness_input(
            runtime,
            label_prefix + ".leaf_upper_bounds",
            witness_batch.leaf_upper_bounds,
            matrix_degree_bound(inputs.queries.query_count, inputs.queries.feature_count)));
        witness.tree_scores = program.bind_virtual(witness_input(
            runtime, label_prefix + ".tree_scores", witness_batch.tree_scores, vector_degree_bound(inputs.queries.query_count)));
    });

    workflow_profile_scope_void("forest_committed_inference_workflow.bind_committed_forest", [&] {
        witness.node_tree_ids = program.bind_virtual(forest_public_input(runtime,
                                                                         "committed_forest.node_tree_ids",
                                                                         inputs.committed_forest.node_tree_ids,
                                                                         committed_forest_public_inputs
                                                                             ? &committed_forest_public_inputs->node_tree_ids
                                                                             : nullptr,
                                                                         vector_degree_bound(inputs.committed_forest.node_count)));
        witness.node_lower_bounds = program.bind_virtual(forest_public_input(runtime,
                                                                             "committed_forest.lower_bounds",
                                                                             inputs.committed_forest.lower_bounds,
                                                                             committed_forest_public_inputs
                                                                                 ? &committed_forest_public_inputs->lower_bounds
                                                                                 : nullptr,
                                                                             matrix_degree_bound(inputs.committed_forest.node_count,
                                                                                                 inputs.committed_forest.feature_count)));
        witness.node_upper_bounds = program.bind_virtual(forest_public_input(runtime,
                                                                             "committed_forest.upper_bounds",
                                                                             inputs.committed_forest.upper_bounds,
                                                                             committed_forest_public_inputs
                                                                                 ? &committed_forest_public_inputs->upper_bounds
                                                                                 : nullptr,
                                                                             matrix_degree_bound(inputs.committed_forest.node_count,
                                                                                                 inputs.committed_forest.feature_count)));
        witness.node_leaf_mask = program.bind_virtual(forest_public_input(runtime,
                                                                          "committed_forest.node_leaf_mask",
                                                                          inputs.committed_forest.node_leaf_mask,
                                                                          committed_forest_public_inputs
                                                                              ? &committed_forest_public_inputs->node_leaf_mask
                                                                              : nullptr,
                                                                          vector_degree_bound(inputs.committed_forest.node_count)));
        witness.node_leaf_values = program.bind_virtual(forest_public_input(runtime,
                                                                            "committed_forest.node_leaf_values",
                                                                            inputs.committed_forest.node_leaf_values,
                                                                            committed_forest_public_inputs
                                                                                ? &committed_forest_public_inputs->node_leaf_values
                                                                                : nullptr,
                                                                            vector_degree_bound(inputs.committed_forest.node_count)));
    });

    workflow_profile_scope_void("forest_committed_inference_workflow.add_training_inference_check", [&] {
        program.forest_training_inference().add_check(witness,
                                                      inputs.queries.query_count,
                                                      inputs.committed_forest.node_count,
                                                      inputs.committed_forest.feature_count,
                                                      inputs.input_bit_bound);
    });
    return true;
}

bool run_committed_forest_inference_workflow(ProofRuntime& runtime,
                                             const forestCommittedInferenceInputs& inputs,
                                             const std::string& label_prefix,
                                             const forestCommittedNodePublicCommitments* committed_forest_public_inputs)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        return add_committed_forest_inference_workflow_to_program(
            runtime, program, inputs, label_prefix, committed_forest_public_inputs);
    });
}

bool add_committed_forest_inference_verifier_workflow_to_program(ProofRuntime& runtime,
                                                                 ProofProgram& program,
                                                                 const forestCommittedInferencePublicInputs& inputs,
                                                                 const std::string& label_prefix)
{
    program.soundness_counter().set_stage(terraeSoundnessStage::Inference);
    validate_query_batch_shape(inputs.queries);
    if (inputs.input_bit_bound == 0) {
        throw std::runtime_error("forest_committed_inference_workflow: input_bit_bound must be non-zero");
    }

    const size_t query_count = inputs.queries.query_count;
    const size_t feature_count = inputs.queries.feature_count;
    const size_t node_count = inputs.committed_forest.node_count;

    forestTrainingInferenceWitness witness;
    workflow_profile_scope_void("forest_committed_inference_verifier.bind_queries_and_witness", [&] {
        const Group1* query_tree_commitment =
            inputs.query_commitments.has_value() ? &inputs.query_commitments->query_tree_ids : nullptr;
        const Group1* lifted_inputs_commitment =
            inputs.query_commitments.has_value() ? &inputs.query_commitments->lifted_inputs : nullptr;
        witness.query_tree_ids = program.bind_virtual(query_public_input(runtime,
                                                                        label_prefix + ".query_tree_ids",
                                                                        inputs.queries.query_tree_ids,
                                                                        query_tree_commitment,
                                                                        vector_degree_bound(query_count)));
        witness.lifted_inputs = program.bind_virtual(query_public_input(runtime,
                                                                       label_prefix + ".lifted_inputs",
                                                                       inputs.queries.lifted_inputs,
                                                                       lifted_inputs_commitment,
                                                                       matrix_degree_bound(query_count, feature_count)));
        witness.leaf_indices = program.bind_virtual(
            WitnessCommitmentInput::verifier(label_prefix + ".leaf_indices", vector_degree_bound(query_count)));
        witness.leaf_lower_bounds = program.bind_virtual(
            WitnessCommitmentInput::verifier(label_prefix + ".leaf_lower_bounds", matrix_degree_bound(query_count, feature_count)));
        witness.leaf_upper_bounds = program.bind_virtual(
            WitnessCommitmentInput::verifier(label_prefix + ".leaf_upper_bounds", matrix_degree_bound(query_count, feature_count)));
        witness.tree_scores = program.bind_virtual(
            WitnessCommitmentInput::verifier(label_prefix + ".tree_scores", vector_degree_bound(query_count)));
    });

    workflow_profile_scope_void("forest_committed_inference_verifier.bind_committed_forest", [&] {
        witness.node_tree_ids = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.node_tree_ids", inputs.committed_forest.node_tree_ids, vector_degree_bound(node_count)));
        witness.node_lower_bounds = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.lower_bounds",
            inputs.committed_forest.lower_bounds,
            matrix_degree_bound(node_count, feature_count)));
        witness.node_upper_bounds = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.upper_bounds",
            inputs.committed_forest.upper_bounds,
            matrix_degree_bound(node_count, feature_count)));
        witness.node_leaf_mask = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.node_leaf_mask", inputs.committed_forest.node_leaf_mask, vector_degree_bound(node_count)));
        witness.node_leaf_values = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.node_leaf_values", inputs.committed_forest.node_leaf_values, vector_degree_bound(node_count)));
    });

    workflow_profile_scope_void("forest_committed_inference_verifier.add_training_inference_check", [&] {
        program.forest_training_inference().add_check(
            witness, query_count, node_count, feature_count, inputs.input_bit_bound);
    });
    return true;
}

bool run_committed_forest_inference_verifier_workflow(
    ProofRuntime& runtime,
    const forestCommittedInferencePublicInputs& inputs,
    const std::string& label_prefix)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        return add_committed_forest_inference_verifier_workflow_to_program(runtime, program, inputs, label_prefix);
    });
}

ProofArtifact prove_committed_forest_inference_with_runtime(const kzgPublicParameters& public_params,
                                                            const forestCommittedInferenceInputs& inputs,
                                                            const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_committed_forest_inference_srs(inputs), "forest_committed_inference_workflow");
    return prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_committed_forest_inference_workflow(runtime, inputs, label_prefix, nullptr);
    });
}

VerificationResult verify_committed_forest_inference_with_runtime(const kzgPublicParameters& public_params,
                                                                  const ProofArtifact& artifact,
                                                                  const forestCommittedInferenceInputs& inputs,
                                                                  const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_committed_forest_inference_srs(inputs), "forest_committed_inference_workflow");
    const forestCommittedInferencePublicInputs public_inputs =
        forest_committed_inference_public_inputs_from_inputs(public_params, inputs);
    return verify_committed_forest_inference_with_runtime(public_params, artifact, public_inputs, label_prefix);
}

VerificationResult verify_committed_forest_inference_with_runtime(const kzgPublicParameters& public_params,
                                                                  const ProofArtifact& artifact,
                                                                  const forestCommittedInferencePublicInputs& inputs,
                                                                  const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_committed_forest_inference_srs(inputs), "forest_committed_inference_workflow");
    return verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_committed_forest_inference_verifier_workflow(runtime, inputs, label_prefix);
    });
}
