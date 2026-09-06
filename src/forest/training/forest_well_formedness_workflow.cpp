#include "forest_well_formedness_workflow.hpp"

#include "matrix_utils.hpp"
#include "protocols/less_check.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace {

size_t conservative_well_formedness_range_lookup_domain(size_t node_count,
                                                        size_t feature_count,
                                                        size_t structure_bit_bound)
{
    lessCheckSrsEstimator estimator;
    // Parent/child box inheritance needs only the parent/left/right
    // non-empty checks; threshold placement follows from the two children.
    // The workflow additionally owns the node-sized maximum-depth check.
    estimator.add_check(structure_bit_bound, node_count * feature_count, 3);
    estimator.add_check(structure_bit_bound, node_count);
    return estimator.required_domain_size();
}

void validate_well_formedness_inputs(const forestWellFormednessInputs& inputs)
{
    const forestCommittedNodeTable& forest = inputs.committed_forest;
    if (forest.tree_count == 0 || forest.node_count == 0 || forest.feature_count == 0) {
        throw std::runtime_error(
            "forest_well_formedness_workflow: tree_count, node_count, and feature_count must be non-zero");
    }
    if (inputs.full_upper_bound == 0 || inputs.structure_bit_bound == 0) {
        throw std::runtime_error(
            "forest_well_formedness_workflow: full_upper_bound and structure_bit_bound must be non-zero");
    }
    if (inputs.max_depth == std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(
            "forest_well_formedness_workflow: max_depth is too large");
    }
    if (inputs.structure_bit_bound < std::numeric_limits<size_t>::digits &&
        inputs.max_depth >= (size_t(1) << inputs.structure_bit_bound)) {
        throw std::runtime_error(
            "forest_well_formedness_workflow: max_depth does not fit structure_bit_bound");
    }

    validate_matrix_shape(forest.tree_count, 1, "forest_well_formedness_workflow root indices");
    validate_matrix_shape(forest.node_count, 1, "forest_well_formedness_workflow node vectors");
    validate_matrix_shape(forest.node_count, 1, "forest_well_formedness_workflow node depths");
    validate_matrix_shape(forest.node_count,
                          forest.feature_count,
                          "forest_well_formedness_workflow feature selector");
    validate_matrix_shape(
        forest.node_count, forest.feature_count, "forest_well_formedness_workflow node boxes");
    validate_matrix_shape(
        forest.node_count, forest.feature_count, "forest_well_formedness_workflow left lower");
    validate_matrix_shape(
        forest.node_count, forest.feature_count, "forest_well_formedness_workflow left upper");
    validate_matrix_shape(
        forest.node_count, forest.feature_count, "forest_well_formedness_workflow right lower");
    validate_matrix_shape(
        forest.node_count, forest.feature_count, "forest_well_formedness_workflow right upper");
}

void validate_well_formedness_public_inputs(const forestWellFormednessPublicInputs& inputs)
{
    if (inputs.committed_forest.tree_count == 0 || inputs.committed_forest.node_count == 0 ||
        inputs.committed_forest.feature_count == 0) {
        throw std::runtime_error(
            "forest_well_formedness_workflow: public committed forest shape must be non-zero");
    }
    if (inputs.full_upper_bound == 0 || inputs.structure_bit_bound == 0) {
        throw std::runtime_error(
            "forest_well_formedness_workflow: public bounds must be non-zero");
    }
    if (inputs.max_depth == std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(
            "forest_well_formedness_workflow: public max_depth is too large");
    }
    if (inputs.structure_bit_bound < std::numeric_limits<size_t>::digits &&
        inputs.max_depth >= (size_t(1) << inputs.structure_bit_bound)) {
        throw std::runtime_error(
            "forest_well_formedness_workflow: public max_depth does not fit structure_bit_bound");
    }
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
            "forest_well_formedness_workflow: missing committed-forest public commitment in verifier mode");
    }
    return PublicCommitmentInput::verifier(key, *commitment, degree_bound);
}

} // namespace

forestWellFormednessInputs forest_well_formedness_inputs_from_training_inputs(
    const forestTrainingProofInputs& inputs)
{
    forestWellFormednessInputs out;
    out.committed_forest = forest_committed_node_table_from_training_inputs(inputs);
    out.left_lower = inputs.left_lower;
    out.left_upper = inputs.left_upper;
    out.right_lower = inputs.right_lower;
    out.right_upper = inputs.right_upper;
    out.full_upper_bound = inputs.full_upper_bound;
    out.structure_bit_bound = inputs.structure_bit_bound;
    out.max_depth = inputs.max_depth;
    validate_well_formedness_inputs(out);
    return out;
}

forestWellFormednessPublicInputs forest_well_formedness_public_inputs_from_training_public_inputs(
    const forestTrainingProofPublicInputs& inputs)
{
    forestWellFormednessPublicInputs out;
    out.committed_forest = inputs.committed_forest;
    out.full_upper_bound = inputs.full_upper_bound;
    out.structure_bit_bound = inputs.structure_bit_bound;
    out.max_depth = inputs.max_depth;
    validate_well_formedness_public_inputs(out);
    return out;
}

forestWellFormednessPublicInputs forest_well_formedness_public_inputs_from_inputs(
    const kzgPublicParameters& public_params,
    const forestWellFormednessInputs& inputs,
    const std::string& forest_key_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_forest_well_formedness_srs(inputs), "forest_well_formedness_workflow");
    return forest_well_formedness_public_inputs_from_inputs(
        commit_committed_forest_public_commitments(public_params, inputs.committed_forest, forest_key_prefix),
        inputs);
}

forestWellFormednessPublicInputs forest_well_formedness_public_inputs_from_inputs(
    const forestCommittedNodePublicCommitments& committed_forest,
    const forestWellFormednessInputs& inputs)
{
    validate_well_formedness_inputs(inputs);
    forestWellFormednessPublicInputs out;
    out.committed_forest = committed_forest;
    out.full_upper_bound = inputs.full_upper_bound;
    out.structure_bit_bound = inputs.structure_bit_bound;
    out.max_depth = inputs.max_depth;
    return out;
}

kzgSrsRequirement required_forest_well_formedness_srs(const forestWellFormednessInputs& inputs)
{
    validate_well_formedness_inputs(inputs);
    kzgSrsRequirement requirement;
    update_kzg_srs_requirement(requirement, inputs.committed_forest.tree_count);
    update_kzg_srs_requirement(requirement, inputs.committed_forest.node_count);
    update_kzg_srs_requirement(
        requirement, inputs.committed_forest.node_count * inputs.committed_forest.feature_count);
    update_kzg_srs_requirement(
        requirement,
        conservative_well_formedness_range_lookup_domain(inputs.committed_forest.node_count,
                                                         inputs.committed_forest.feature_count,
                                                         inputs.structure_bit_bound));
    require_kzg_distinct_opening_points(requirement, 1);
    return requirement;
}

kzgSrsRequirement required_forest_well_formedness_srs(const forestWellFormednessPublicInputs& inputs)
{
    validate_well_formedness_public_inputs(inputs);
    kzgSrsRequirement requirement;
    update_kzg_srs_requirement(requirement, inputs.committed_forest.tree_count);
    update_kzg_srs_requirement(requirement, inputs.committed_forest.node_count);
    update_kzg_srs_requirement(
        requirement, inputs.committed_forest.node_count * inputs.committed_forest.feature_count);
    update_kzg_srs_requirement(
        requirement,
        conservative_well_formedness_range_lookup_domain(inputs.committed_forest.node_count,
                                                         inputs.committed_forest.feature_count,
                                                         inputs.structure_bit_bound));
    require_kzg_distinct_opening_points(requirement, 1);
    return requirement;
}

bool add_forest_well_formedness_workflow_to_program(
    ProofRuntime& runtime,
    ProofProgram& program,
    const forestWellFormednessInputs& inputs,
    const std::string& label_prefix,
    const forestCommittedNodePublicCommitments* committed_forest_public_inputs)
{
    program.soundness_counter().set_stage(terraeSoundnessStage::WellFormedness);
    validate_well_formedness_inputs(inputs);
    const forestCommittedNodeTable& forest = inputs.committed_forest;

    workflow_profile_scope_void("forest_well_formedness_workflow.bind_committed_forest", [&] {
        const virCommitment root_indices = program.bind_virtual(forest_public_input(runtime,
                                                                                   "committed_forest.root_indices",
                                                                                   forest.root_indices,
                                                                                   committed_forest_public_inputs
                                                                                       ? &committed_forest_public_inputs->root_indices
                                                                                       : nullptr,
                                                                                   vector_degree_bound(forest.tree_count)));
        const virCommitment split_node_mask = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.split_node_mask",
            forest.split_node_mask,
            committed_forest_public_inputs ? &committed_forest_public_inputs->split_node_mask : nullptr,
            vector_degree_bound(forest.node_count)));
        const virCommitment node_leaf_mask = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.node_leaf_mask",
            forest.node_leaf_mask,
            committed_forest_public_inputs ? &committed_forest_public_inputs->node_leaf_mask : nullptr,
            vector_degree_bound(forest.node_count)));
        const virCommitment node_leaf_values = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.node_leaf_values",
            forest.node_leaf_values,
            committed_forest_public_inputs ? &committed_forest_public_inputs->node_leaf_values : nullptr,
            vector_degree_bound(forest.node_count)));
        const virCommitment node_depths = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.node_depths",
            forest.node_depths,
            committed_forest_public_inputs ? &committed_forest_public_inputs->node_depths : nullptr,
            vector_degree_bound(forest.node_count)));
        const virCommitment left_child_indices = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.left_child_indices",
            forest.left_child_indices,
            committed_forest_public_inputs ? &committed_forest_public_inputs->left_child_indices : nullptr,
            vector_degree_bound(forest.node_count)));
        const virCommitment right_child_indices = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.right_child_indices",
            forest.right_child_indices,
            committed_forest_public_inputs ? &committed_forest_public_inputs->right_child_indices : nullptr,
            vector_degree_bound(forest.node_count)));
        const virCommitment node_tree_ids = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.node_tree_ids",
            forest.node_tree_ids,
            committed_forest_public_inputs ? &committed_forest_public_inputs->node_tree_ids : nullptr,
            vector_degree_bound(forest.node_count)));
        const virCommitment feature_selector = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.feature_selector",
            forest.feature_selector,
            committed_forest_public_inputs ? &committed_forest_public_inputs->feature_selector : nullptr,
            matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment lower_bounds = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.lower_bounds",
            forest.lower_bounds,
            committed_forest_public_inputs ? &committed_forest_public_inputs->lower_bounds : nullptr,
            matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment upper_bounds = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.upper_bounds",
            forest.upper_bounds,
            committed_forest_public_inputs ? &committed_forest_public_inputs->upper_bounds : nullptr,
            matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment left_lower = program.bind_virtual(witness_input(
            runtime, label_prefix + ".left_lower", inputs.left_lower, matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment left_upper = program.bind_virtual(witness_input(
            runtime, label_prefix + ".left_upper", inputs.left_upper, matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment right_lower = program.bind_virtual(witness_input(
            runtime, label_prefix + ".right_lower", inputs.right_lower, matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment right_upper = program.bind_virtual(witness_input(
            runtime, label_prefix + ".right_upper", inputs.right_upper, matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment thresholds = program.bind_virtual(forest_public_input(
            runtime,
            "committed_forest.thresholds",
            forest.thresholds,
            committed_forest_public_inputs ? &committed_forest_public_inputs->thresholds : nullptr,
            vector_degree_bound(forest.node_count)));
        if (committed_forest_public_inputs != nullptr) {
            program.root_selector_consistency().register_expected_tree_ids_commitment(
                forest.tree_count, committed_forest_public_inputs->expected_tree_ids);
        }

        program.forest_structure().add_masked_check(root_indices,
                                                    split_node_mask,
                                                    node_leaf_mask,
                                                    node_depths,
                                                    left_child_indices,
                                                    right_child_indices,
                                                    node_tree_ids,
                                                    feature_selector,
                                                    lower_bounds,
                                                    upper_bounds,
                                                    left_lower,
                                                    left_upper,
                                                    right_lower,
                                                    right_upper,
                                                    thresholds,
                                                    forest.tree_count,
                                                    forest.node_count,
                                                    forest.feature_count,
                                                    inputs.full_upper_bound,
                                                    inputs.structure_bit_bound);

        const virCommitment active_node_mask = split_node_mask + node_leaf_mask;
        program.zero().add_boolean_check(
            split_node_mask, forest.node_count, "forest_well_formedness.split_mask_boolean");
        program.less_check().add_masked_check(
            virCommitment(field_from_size_t(inputs.max_depth)) - node_depths,
            active_node_mask,
            inputs.structure_bit_bound,
            forest.node_count);

        virCommitment non_leaf_score =
            (virCommitment(Fr(1)) - node_leaf_mask) * node_leaf_values;
        program.zero().add_check(
            non_leaf_score, forest.node_count, "forest_well_formedness.non_leaf_scores");
    });
    return true;
}

bool add_forest_well_formedness_verifier_workflow_to_program(
    ProofRuntime& runtime,
    ProofProgram& program,
    const forestWellFormednessPublicInputs& inputs,
    const std::string& label_prefix)
{
    program.soundness_counter().set_stage(terraeSoundnessStage::WellFormedness);
    validate_well_formedness_public_inputs(inputs);
    const forestCommittedNodePublicCommitments& forest = inputs.committed_forest;

    workflow_profile_scope_void("forest_well_formedness_verifier.bind_committed_forest", [&] {
        const virCommitment root_indices = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.root_indices", forest.root_indices, vector_degree_bound(forest.tree_count)));
        const virCommitment split_node_mask = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.split_node_mask", forest.split_node_mask, vector_degree_bound(forest.node_count)));
        const virCommitment node_leaf_mask = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.node_leaf_mask", forest.node_leaf_mask, vector_degree_bound(forest.node_count)));
        const virCommitment node_leaf_values = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.node_leaf_values", forest.node_leaf_values, vector_degree_bound(forest.node_count)));
        const virCommitment node_depths = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.node_depths", forest.node_depths, vector_degree_bound(forest.node_count)));
        const virCommitment left_child_indices = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.left_child_indices", forest.left_child_indices, vector_degree_bound(forest.node_count)));
        const virCommitment right_child_indices = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.right_child_indices", forest.right_child_indices, vector_degree_bound(forest.node_count)));
        const virCommitment node_tree_ids = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.node_tree_ids", forest.node_tree_ids, vector_degree_bound(forest.node_count)));
        const virCommitment feature_selector = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.feature_selector",
            forest.feature_selector,
            matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment lower_bounds = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.lower_bounds",
            forest.lower_bounds,
            matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment upper_bounds = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.upper_bounds",
            forest.upper_bounds,
            matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment left_lower = program.bind_virtual(WitnessCommitmentInput::verifier(
            label_prefix + ".left_lower", matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment left_upper = program.bind_virtual(WitnessCommitmentInput::verifier(
            label_prefix + ".left_upper", matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment right_lower = program.bind_virtual(WitnessCommitmentInput::verifier(
            label_prefix + ".right_lower", matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment right_upper = program.bind_virtual(WitnessCommitmentInput::verifier(
            label_prefix + ".right_upper", matrix_degree_bound(forest.node_count, forest.feature_count)));
        const virCommitment thresholds = program.bind_virtual(PublicCommitmentInput::verifier(
            "committed_forest.thresholds", forest.thresholds, vector_degree_bound(forest.node_count)));
        program.root_selector_consistency().register_expected_tree_ids_commitment(
            forest.tree_count, forest.expected_tree_ids);

        program.forest_structure().add_masked_check(root_indices,
                                                    split_node_mask,
                                                    node_leaf_mask,
                                                    node_depths,
                                                    left_child_indices,
                                                    right_child_indices,
                                                    node_tree_ids,
                                                    feature_selector,
                                                    lower_bounds,
                                                    upper_bounds,
                                                    left_lower,
                                                    left_upper,
                                                    right_lower,
                                                    right_upper,
                                                    thresholds,
                                                    forest.tree_count,
                                                    forest.node_count,
                                                    forest.feature_count,
                                                    inputs.full_upper_bound,
                                                    inputs.structure_bit_bound);

        const virCommitment active_node_mask = split_node_mask + node_leaf_mask;
        program.zero().add_boolean_check(
            split_node_mask, forest.node_count, "forest_well_formedness.split_mask_boolean");
        program.less_check().add_masked_check(
            virCommitment(field_from_size_t(inputs.max_depth)) - node_depths,
            active_node_mask,
            inputs.structure_bit_bound,
            forest.node_count);

        virCommitment non_leaf_score =
            (virCommitment(Fr(1)) - node_leaf_mask) * node_leaf_values;
        program.zero().add_check(
            non_leaf_score, forest.node_count, "forest_well_formedness.non_leaf_scores");
    });
    (void)runtime;
    return true;
}

bool run_forest_well_formedness_workflow(
    ProofRuntime& runtime,
    const forestWellFormednessInputs& inputs,
    const std::string& label_prefix,
    const forestCommittedNodePublicCommitments* committed_forest_public_inputs)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        return add_forest_well_formedness_workflow_to_program(
            runtime, program, inputs, label_prefix, committed_forest_public_inputs);
    });
}

bool run_forest_well_formedness_verifier_workflow(
    ProofRuntime& runtime,
    const forestWellFormednessPublicInputs& inputs,
    const std::string& label_prefix)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        return add_forest_well_formedness_verifier_workflow_to_program(
            runtime, program, inputs, label_prefix);
    });
}

ProofArtifact prove_forest_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const forestWellFormednessInputs& inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_forest_well_formedness_srs(inputs), "forest_well_formedness_workflow");
    return prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_forest_well_formedness_workflow(runtime, inputs, label_prefix, nullptr);
    });
}

ProofArtifact prove_forest_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const forestWellFormednessInputs& inputs,
    const forestCommittedNodePublicCommitments& committed_forest_public_inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_forest_well_formedness_srs(inputs), "forest_well_formedness_workflow");
    return prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_forest_well_formedness_workflow(
            runtime, inputs, label_prefix, &committed_forest_public_inputs);
    });
}

VerificationResult verify_forest_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestWellFormednessInputs& inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_forest_well_formedness_srs(inputs), "forest_well_formedness_workflow");
    const forestWellFormednessPublicInputs public_inputs =
        forest_well_formedness_public_inputs_from_inputs(public_params, inputs);
    return verify_forest_well_formedness_with_runtime(public_params, artifact, public_inputs, label_prefix);
}

VerificationResult verify_forest_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestWellFormednessPublicInputs& inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_forest_well_formedness_srs(inputs), "forest_well_formedness_workflow");
    return verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_forest_well_formedness_verifier_workflow(runtime, inputs, label_prefix);
    });
}

forestWellFormednessVerifierPreprocessing prepare_forest_well_formedness_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestWellFormednessPublicInputs& inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(
        public_params, required_forest_well_formedness_srs(inputs), "forest_well_formedness_workflow");

    forestWellFormednessVerifierPreprocessing out;
    out.runtime = ProofRuntime::verifier_ptr(public_params, artifact.proof);
    out.program = std::make_unique<ProofProgram>(*out.runtime);
    add_forest_well_formedness_verifier_workflow_to_program(
        *out.runtime, *out.program, inputs, label_prefix);
    if (!out.program->prepare_verifier_preprocessing()) {
        throw std::runtime_error("forest_well_formedness_workflow: verifier preprocessing failed");
    }
    return out;
}

VerificationResult verify_forest_well_formedness_preprocessed(
    forestWellFormednessVerifierPreprocessing& preprocessing)
{
    if (!preprocessing.runtime || !preprocessing.program) {
        throw std::runtime_error("forest_well_formedness_workflow: missing verifier preprocessing");
    }

    VerificationResult result;
    result.accepted = preprocessing.program->verify_preprocessed();
    result.proof_fully_consumed = preprocessing.runtime->proof_fully_consumed();
    result.transcript_state = preprocessing.runtime->transcript_state();
    return result;
}
