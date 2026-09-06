#include "forest_training_workflow.hpp"

#include "fixed_point_division.hpp"
#include "forest_training_export.hpp"
#include "training_runtime_utils.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

void validate_training_workflow_inputs(const forestTrainingProofInputs& inputs)
{
    if (inputs.tree_count == 0 || inputs.node_count == 0 || inputs.feature_count == 0 || inputs.bin_count == 0 ||
        inputs.iteration_count == 0 || inputs.class_count == 0 || inputs.query_count == 0) {
        throw std::runtime_error("forest_training_workflow: shape parameters must be non-zero");
    }
    if (inputs.tree_count != inputs.iteration_count * inputs.class_count) {
        throw std::runtime_error("forest_training_workflow: tree_count must equal iteration_count * class_count");
    }
    if (inputs.learning_rate_numerator == 0 || inputs.learning_rate_denominator == 0) {
        throw std::runtime_error("forest_training_workflow: learning rate terms must be non-zero");
    }
    validate_matrix_shape(inputs.node_count, 1, "forest_training_workflow node vectors");
}

void ensure_polynomials_match_on_prover(ProofRuntime& runtime,
                                        const poly& expected,
                                        const poly& actual,
                                        const char* context)
{
    if (runtime.mode() != ProofMode::Prove) {
        return;
    }
    const std::vector<Fr> expected_coeffs = expected.coefficients();
    const std::vector<Fr> actual_coeffs = actual.coefficients();
    if (expected_coeffs.size() != actual_coeffs.size()) {
        throw std::runtime_error(std::string("forest_training_workflow: redundant witness mismatch for ") + context);
    }
    for (size_t i = 0; i < expected_coeffs.size(); ++i) {
        if (expected_coeffs[i] != actual_coeffs[i]) {
            throw std::runtime_error(std::string("forest_training_workflow: redundant witness mismatch for ") + context);
        }
    }
}

size_t checked_mul_size(size_t left, size_t right, const char* context)
{
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        throw std::runtime_error(std::string(context) + ": size overflow");
    }
    return left * right;
}

size_t ceil_log2_at_least_one(size_t value)
{
    if (value <= 1) {
        return 1;
    }
    size_t out = 0;
    size_t power = 1;
    while (power < value) {
        power *= 2;
        ++out;
    }
    return out;
}

size_t unsigned_channel_as_signed_total_bits_requirement(size_t bit_bound, size_t fractional_bits)
{
    if (bit_bound == 0) {
        throw std::runtime_error("forest_training_workflow: bit bound must be non-zero");
    }
    return std::max(bit_bound + 1, fractional_bits + 1);
}

size_t conservative_training_range_lookup_domain(const forestTrainingProofInputs& inputs)
{
    const size_t query_row_count = checked_mul_size(
        inputs.tree_count, inputs.query_count, "forest_training_workflow query rows");
    const size_t candidate_width = checked_mul_size(
        inputs.feature_count, inputs.bin_count, "forest_training_workflow candidate width");
    const size_t query_feature_domain = checked_mul_size(
        query_row_count, inputs.feature_count, "forest_training_workflow query feature domain");
    const size_t candidate_domain = checked_mul_size(
        inputs.node_count, candidate_width, "forest_training_workflow candidate domain");
    const size_t boosting_domain = checked_mul_size(
        checked_mul_size(inputs.iteration_count,
                         inputs.class_count,
                         "forest_training_workflow boosting class domain"),
        inputs.query_count,
        "forest_training_workflow boosting domain");

    lessCheckSrsEstimator estimator;
    const size_t feature_bit_bound = ceil_log2_at_least_one(inputs.feature_count);
    const size_t bin_bit_bound = ceil_log2_at_least_one(inputs.bin_count);
    const size_t candidate_bit_bound = feature_bit_bound + bin_bit_bound;
    const size_t gain_gap_stopping_bit_bound = inputs.gain_gap_bit_bound + 1;

    // Training-time inference and histogram bucket addressing. Leaf-index
    // membership is owned by inference lookup, and flattened-address
    // membership by histogram aggregation.
    estimator.add_check(inputs.input_bit_bound, query_feature_domain, 2);
    estimator.add_check(bin_bit_bound, query_feature_domain);

    // Split statistics, exact split gain, and quantized split gain.
    estimator.add_check(fixed_point_encoded_bit_bound(inputs.gradient_encoding), candidate_domain, 3);
    estimator.add_check(inputs.hessian_bit_bound, candidate_domain, 2);
    estimator.add_check(inputs.gain_denominator_bit_bound, candidate_domain, 4);
    add_fixed_point_division_srs_requirement(estimator,
                                             inputs.gain_numerator_encoding,
                                             inputs.quantized_gain_encoding,
                                             inputs.gain_denominator_bit_bound,
                                             candidate_domain);

    // Split selection and stopping-condition checks.
    estimator.add_check(inputs.gain_gap_bit_bound + candidate_bit_bound + 1, candidate_domain);
    // Candidate child-count nonzero now uses inverse witnesses instead of range checks.
    estimator.add_check(feature_bit_bound, inputs.node_count);
    estimator.add_check(bin_bit_bound, inputs.node_count);
    estimator.add_check(gain_gap_stopping_bit_bound, inputs.node_count, 2);
    estimator.add_check(inputs.structure_bit_bound, inputs.node_count);

    // Leaf values.
    estimator.add_check(inputs.hessian_bit_bound, inputs.node_count);
    estimator.add_check(inputs.leaf_denominator_bit_bound, inputs.node_count);
    add_fixed_point_division_srs_requirement(estimator,
                                             inputs.gradient_encoding,
                                             inputs.leaf_value_encoding,
                                             inputs.leaf_denominator_bit_bound,
                                             inputs.node_count);

    // Boosting state, softmax, and diagonal Hessian divisions.
    estimator.add_check(fixed_point_encoded_bit_bound(inputs.score_encoding), boosting_domain, 2);
    const size_t softmax_scale_bit_bound = inputs.softmax_encoding.fractional_bits + 1;
    const size_t exp_output_total_bits =
        unsigned_channel_as_signed_total_bits_requirement(softmax_scale_bit_bound, 0);
    const fixedPointEncoding exp_output_encoding =
        fixedPointEncoding::signed_offset_binary(exp_output_total_bits, 0);
    const size_t softmax_denominator_bit_bound =
        softmax_scale_bit_bound + ceil_log2_at_least_one(inputs.class_count) + 1;
    add_fixed_point_division_srs_requirement(estimator,
                                             exp_output_encoding,
                                             inputs.softmax_encoding,
                                             softmax_denominator_bit_bound,
                                             boosting_domain);
    const fixedPointEncoding hessian_encoding =
        fixedPointEncoding::signed_offset_binary(
            unsigned_channel_as_signed_total_bits_requirement(
                inputs.hessian_bit_bound, inputs.softmax_encoding.fractional_bits),
            inputs.softmax_encoding.fractional_bits);
    const fixedPointEncoding hessian_product_encoding =
        fixedPointEncoding::signed_offset_binary(
            unsigned_channel_as_signed_total_bits_requirement(
                inputs.hessian_bit_bound + inputs.softmax_encoding.fractional_bits + 1,
                inputs.softmax_encoding.fractional_bits),
            inputs.softmax_encoding.fractional_bits);
    add_fixed_point_division_srs_requirement(estimator,
                                             hessian_product_encoding,
                                             hessian_encoding,
                                             softmax_scale_bit_bound,
                                             boosting_domain);

    return estimator.required_domain_size();
}

} // namespace

forestTrainingProofInputs forest_training_proof_verifier_inputs_from_public_inputs(
    const forestTrainingProofPublicInputs& inputs)
{
    forestTrainingProofInputs out;
    out.tree_count = inputs.tree_count;
    out.node_count = inputs.node_count;
    out.feature_count = inputs.feature_count;
    out.bin_count = inputs.bin_count;
    out.iteration_count = inputs.iteration_count;
    out.class_count = inputs.class_count;
    out.query_count = inputs.query_count;
    out.input_bit_bound = inputs.input_bit_bound;
    out.full_upper_bound = inputs.full_upper_bound;
    out.structure_bit_bound = inputs.structure_bit_bound;
    out.gradient_encoding = inputs.gradient_encoding;
    out.gain_numerator_encoding = inputs.gain_numerator_encoding;
    out.quantized_gain_encoding = inputs.quantized_gain_encoding;
    out.leaf_value_encoding = inputs.leaf_value_encoding;
    out.score_encoding = inputs.score_encoding;
    out.softmax_encoding = inputs.softmax_encoding;
    out.hessian_bit_bound = inputs.hessian_bit_bound;
    out.gain_denominator_bit_bound = inputs.gain_denominator_bit_bound;
    out.leaf_denominator_bit_bound = inputs.leaf_denominator_bit_bound;
    out.gain_gap_bit_bound = inputs.gain_gap_bit_bound;
    out.loss_mode = inputs.loss_mode;
    out.l2_regularization = inputs.l2_regularization;
    out.learning_rate_numerator = inputs.learning_rate_numerator;
    out.learning_rate_denominator = inputs.learning_rate_denominator;
    out.split_penalty = inputs.split_penalty;
    out.split_gain_threshold = inputs.split_gain_threshold;
    out.max_depth = inputs.max_depth;
    return out;
}

namespace {

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
        throw std::runtime_error("forest_training_workflow: missing committed-forest public commitment in verifier mode");
    }
    return PublicCommitmentInput::verifier(key, *commitment, degree_bound);
}

} // namespace

forestTrainingProofPublicInputs forest_training_proof_public_inputs_from_inputs(
    const kzgPublicParameters& public_params,
    const forestTrainingProofInputs& inputs,
    const std::string& forest_key_prefix)
{
    return forest_training_proof_public_inputs_from_inputs(
        commit_committed_forest_public_commitments(
            public_params, forest_committed_node_table_from_training_inputs(inputs), forest_key_prefix),
        inputs);
}

forestTrainingProofPublicInputs forest_training_proof_public_inputs_from_inputs(
    const forestCommittedNodePublicCommitments& committed_forest,
    const forestTrainingProofInputs& inputs)
{
    forestTrainingProofPublicInputs out;
    out.committed_forest = committed_forest;
    out.tree_count = inputs.tree_count;
    out.node_count = inputs.node_count;
    out.feature_count = inputs.feature_count;
    out.bin_count = inputs.bin_count;
    out.iteration_count = inputs.iteration_count;
    out.class_count = inputs.class_count;
    out.query_count = inputs.query_count;
    out.input_bit_bound = inputs.input_bit_bound;
    out.full_upper_bound = inputs.full_upper_bound;
    out.structure_bit_bound = inputs.structure_bit_bound;
    out.gradient_encoding = inputs.gradient_encoding;
    out.gain_numerator_encoding = inputs.gain_numerator_encoding;
    out.quantized_gain_encoding = inputs.quantized_gain_encoding;
    out.leaf_value_encoding = inputs.leaf_value_encoding;
    out.score_encoding = inputs.score_encoding;
    out.softmax_encoding = inputs.softmax_encoding;
    out.hessian_bit_bound = inputs.hessian_bit_bound;
    out.gain_denominator_bit_bound = inputs.gain_denominator_bit_bound;
    out.leaf_denominator_bit_bound = inputs.leaf_denominator_bit_bound;
    out.gain_gap_bit_bound = inputs.gain_gap_bit_bound;
    out.loss_mode = inputs.loss_mode;
    out.l2_regularization = inputs.l2_regularization;
    out.learning_rate_numerator = inputs.learning_rate_numerator;
    out.learning_rate_denominator = inputs.learning_rate_denominator;
    out.split_penalty = inputs.split_penalty;
    out.split_gain_threshold = inputs.split_gain_threshold;
    out.max_depth = inputs.max_depth;
    return out;
}

kzgSrsRequirement required_forest_training_proof_srs(const forestTrainingProofInputs& inputs)
{
    validate_training_workflow_inputs(inputs);

    kzgSrsRequirement requirement;
    const size_t query_row_count = inputs.tree_count * inputs.query_count;
    const size_t candidate_width = inputs.feature_count * inputs.bin_count;
    update_kzg_srs_requirement(requirement, inputs.tree_count);
    update_kzg_srs_requirement(requirement, inputs.node_count);
    update_kzg_srs_requirement(requirement, inputs.node_count * inputs.feature_count);
    update_kzg_srs_requirement(requirement, query_row_count);
    update_kzg_srs_requirement(requirement, query_row_count * inputs.feature_count);
    update_kzg_srs_requirement(requirement, inputs.node_count * candidate_width);
    update_kzg_srs_requirement(requirement, inputs.iteration_count * inputs.query_count);
    update_kzg_srs_requirement(requirement, inputs.iteration_count * inputs.class_count * inputs.query_count);
    update_kzg_srs_requirement(requirement, inputs.class_count * inputs.query_count);
    update_kzg_srs_requirement(requirement, conservative_training_range_lookup_domain(inputs));
    require_kzg_constant_degree_claim(requirement);
    require_kzg_distinct_opening_points(requirement, 1);
    return requirement;
}

kzgSrsRequirement required_forest_training_proof_srs(const forestTrainingProofPublicInputs& inputs)
{
    const forestTrainingProofInputs verifier_inputs = forest_training_proof_verifier_inputs_from_public_inputs(inputs);
    return required_forest_training_proof_srs(verifier_inputs);
}

bool add_forest_training_proof_workflow_to_program(
    ProofRuntime& runtime,
    ProofProgram& program,
    const forestTrainingProofInputs& inputs,
    const std::string& label_prefix,
    const forestCommittedNodePublicCommitments* committed_forest_public_inputs)
{
    program.soundness_counter().set_stage(terraeSoundnessStage::Training);
    validate_training_workflow_inputs(inputs);

    const size_t query_row_count = inputs.tree_count * inputs.query_count;
    const size_t candidate_width = inputs.feature_count * inputs.bin_count;

    forestTrainingProofWitness witness;
        workflow_profile_scope_void("forest_training_workflow.bind_committed_forest", [&] {
            witness.root_indices = program.bind_virtual(forest_public_input(runtime,
                                                                            "committed_forest.root_indices",
                                                                            inputs.root_indices,
                                                                            committed_forest_public_inputs
                                                                                ? &committed_forest_public_inputs->root_indices
                                                                                : nullptr,
                                                                            vector_degree_bound(inputs.tree_count)));
            witness.split_node_mask = program.bind_virtual(forest_public_input(runtime,
                                                                               "committed_forest.split_node_mask",
                                                                               inputs.split_node_mask,
                                                                               committed_forest_public_inputs
                                                                                   ? &committed_forest_public_inputs->split_node_mask
                                                                                   : nullptr,
                                                                               vector_degree_bound(inputs.node_count)));
            witness.node_leaf_mask = program.bind_virtual(forest_public_input(runtime,
                                                                              "committed_forest.node_leaf_mask",
                                                                              inputs.node_leaf_mask,
                                                                              committed_forest_public_inputs
                                                                                  ? &committed_forest_public_inputs->node_leaf_mask
                                                                                  : nullptr,
                                                                              vector_degree_bound(inputs.node_count)));
            witness.depths = program.bind_virtual(forest_public_input(
                runtime,
                "committed_forest.node_depths",
                inputs.node_depths,
                committed_forest_public_inputs
                    ? &committed_forest_public_inputs->node_depths
                    : nullptr,
                vector_degree_bound(inputs.node_count)));
            witness.left_child_indices = program.bind_virtual(forest_public_input(runtime,
                                                                                  "committed_forest.left_child_indices",
                                                                                  inputs.left_child_indices,
                                                                                  committed_forest_public_inputs
                                                                                      ? &committed_forest_public_inputs
                                                                                             ->left_child_indices
                                                                                      : nullptr,
                                                                                  vector_degree_bound(inputs.node_count)));
            witness.right_child_indices = program.bind_virtual(forest_public_input(runtime,
                                                                                   "committed_forest.right_child_indices",
                                                                                   inputs.right_child_indices,
                                                                                   committed_forest_public_inputs
                                                                                       ? &committed_forest_public_inputs
                                                                                              ->right_child_indices
                                                                                       : nullptr,
                                                                                   vector_degree_bound(inputs.node_count)));
            witness.node_tree_ids = program.bind_virtual(forest_public_input(runtime,
                                                                             "committed_forest.node_tree_ids",
                                                                             inputs.node_tree_ids,
                                                                             committed_forest_public_inputs
                                                                                 ? &committed_forest_public_inputs->node_tree_ids
                                                                                 : nullptr,
                                                                             vector_degree_bound(inputs.node_count)));
            witness.feature_selector = program.bind_virtual(forest_public_input(runtime,
                                                                               "committed_forest.feature_selector",
                                                                               inputs.feature_selector,
                                                                               committed_forest_public_inputs
                                                                                   ? &committed_forest_public_inputs
                                                                                          ->feature_selector
                                                                                   : nullptr,
                                                                               matrix_degree_bound(inputs.node_count,
                                                                                                   inputs.feature_count)));
            witness.lower_bounds = program.bind_virtual(forest_public_input(runtime,
                                                                            "committed_forest.lower_bounds",
                                                                            inputs.lower_bounds,
                                                                            committed_forest_public_inputs
                                                                                ? &committed_forest_public_inputs->lower_bounds
                                                                                : nullptr,
                                                                            matrix_degree_bound(inputs.node_count,
                                                                                                inputs.feature_count)));
            witness.upper_bounds = program.bind_virtual(forest_public_input(runtime,
                                                                            "committed_forest.upper_bounds",
                                                                            inputs.upper_bounds,
                                                                            committed_forest_public_inputs
                                                                                ? &committed_forest_public_inputs->upper_bounds
                                                                                : nullptr,
                                                                            matrix_degree_bound(inputs.node_count,
                                                                                                inputs.feature_count)));
            witness.left_lower = program.bind_virtual(witness_input(
                runtime, label_prefix + ".left_lower", inputs.left_lower, matrix_degree_bound(inputs.node_count, inputs.feature_count)));
            witness.left_upper = program.bind_virtual(witness_input(
                runtime, label_prefix + ".left_upper", inputs.left_upper, matrix_degree_bound(inputs.node_count, inputs.feature_count)));
            witness.right_lower = program.bind_virtual(witness_input(
                runtime, label_prefix + ".right_lower", inputs.right_lower, matrix_degree_bound(inputs.node_count, inputs.feature_count)));
            witness.right_upper = program.bind_virtual(witness_input(
                runtime, label_prefix + ".right_upper", inputs.right_upper, matrix_degree_bound(inputs.node_count, inputs.feature_count)));
            witness.thresholds = program.bind_virtual(forest_public_input(runtime,
                                                                          "committed_forest.thresholds",
                                                                          inputs.thresholds,
                                                                          committed_forest_public_inputs
                                                                              ? &committed_forest_public_inputs->thresholds
                                                                              : nullptr,
                                                                          vector_degree_bound(inputs.node_count)));
        });
        if (committed_forest_public_inputs != nullptr) {
            program.root_selector_consistency().register_expected_tree_ids_commitment(
                inputs.tree_count, committed_forest_public_inputs->expected_tree_ids);
        }

        workflow_profile_scope_void("forest_training_workflow.bind_histogram_refresh_core", [&] {
            witness.histogram_refresh.inference.query_tree_ids = program.bind_virtual(
                witness_input(runtime, label_prefix + ".query_tree_ids", inputs.query_tree_ids, vector_degree_bound(query_row_count)));
            witness.histogram_refresh.inference.lifted_inputs = program.bind_virtual(witness_input(
                runtime, label_prefix + ".lifted_inputs", inputs.lifted_inputs, matrix_degree_bound(query_row_count, inputs.feature_count)));
            witness.histogram_refresh.inference.leaf_indices = program.bind_virtual(
                witness_input(runtime, label_prefix + ".leaf_indices", inputs.leaf_indices, vector_degree_bound(query_row_count)));
            witness.histogram_refresh.inference.leaf_lower_bounds = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".leaf_lower_bounds",
                inputs.leaf_lower_bounds,
                matrix_degree_bound(query_row_count, inputs.feature_count)));
            witness.histogram_refresh.inference.leaf_upper_bounds = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".leaf_upper_bounds",
                inputs.leaf_upper_bounds,
                matrix_degree_bound(query_row_count, inputs.feature_count)));
            witness.histogram_refresh.inference.tree_scores = program.bind_virtual(
                witness_input(runtime, label_prefix + ".tree_scores", inputs.tree_scores, vector_degree_bound(query_row_count)));
            witness.histogram_refresh.inference.node_tree_ids = witness.node_tree_ids;
            witness.histogram_refresh.inference.node_lower_bounds = witness.lower_bounds;
            witness.histogram_refresh.inference.node_upper_bounds = witness.upper_bounds;
            witness.histogram_refresh.inference.node_leaf_mask = witness.node_leaf_mask;
            witness.histogram_refresh.inference.node_leaf_values = program.bind_virtual(forest_public_input(
                runtime,
                "committed_forest.node_leaf_values",
                inputs.node_leaf_values,
                committed_forest_public_inputs ? &committed_forest_public_inputs->node_leaf_values : nullptr,
                vector_degree_bound(inputs.node_count)));
            witness.histogram_refresh.lifted_leaf_indices = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".lifted_leaf_indices",
                inputs.lifted_leaf_indices,
                matrix_degree_bound(query_row_count, inputs.feature_count)));
            witness.histogram_refresh.flattened_bucket_indices = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".flattened_bucket_indices",
                inputs.flattened_bucket_indices,
                matrix_degree_bound(query_row_count, inputs.feature_count)));
            witness.histogram_refresh.parent_mask = witness.split_node_mask;
            witness.histogram_refresh.left_child_indices = witness.left_child_indices;
            witness.histogram_refresh.right_child_indices = witness.right_child_indices;
        });

        workflow_profile_scope_void("forest_training_workflow.bind_histogram_channels", [&] {
            for (size_t channel = 0; channel < inputs.channels.size(); ++channel) {
                const forestTrainingProofChannelInputs& channel_inputs = inputs.channels[channel];
                const std::string base = label_prefix + ".channel" + std::to_string(channel);
                witness.histogram_refresh.channels.query_values.push_back(program.bind_virtual(
                    witness_input(runtime, base + ".query_values", channel_inputs.query_values, vector_degree_bound(query_row_count))));
                witness.histogram_refresh.channels.lifted_values.push_back(program.bind_virtual(witness_input(
                    runtime, base + ".lifted_values", channel_inputs.lifted_values, matrix_degree_bound(query_row_count, inputs.feature_count))));
                witness.histogram_refresh.channels.leaf_histograms.push_back(program.bind_virtual(witness_input(
                    runtime, base + ".leaf_histograms", channel_inputs.leaf_histograms, matrix_degree_bound(inputs.node_count, candidate_width))));
                witness.histogram_refresh.channels.full_histograms.push_back(program.bind_virtual(witness_input(
                    runtime, base + ".full_histograms", channel_inputs.full_histograms, matrix_degree_bound(inputs.node_count, candidate_width))));
                ensure_polynomials_match_on_prover(
                    runtime, channel_inputs.full_histograms, channel_inputs.split_histograms, "split_histograms");
                witness.split_statistics.histograms.push_back(witness.histogram_refresh.channels.full_histograms.back());
                witness.split_statistics.left_statistics.push_back(program.bind_virtual(witness_input(
                    runtime, base + ".left_statistics", channel_inputs.left_statistics, matrix_degree_bound(inputs.node_count, candidate_width))));
                witness.split_statistics.right_statistics.push_back(program.bind_virtual(witness_input(
                    runtime, base + ".right_statistics", channel_inputs.right_statistics, matrix_degree_bound(inputs.node_count, candidate_width))));
            }
        });

        workflow_profile_scope_void("forest_training_workflow.bind_gain_and_selection", [&] {
            ensure_polynomials_match_on_prover(
                runtime, inputs.left_gradients, inputs.channels[1].left_statistics, "left_gradients");
            ensure_polynomials_match_on_prover(
                runtime, inputs.right_gradients, inputs.channels[1].right_statistics, "right_gradients");
            ensure_polynomials_match_on_prover(
                runtime, inputs.left_hessians, inputs.channels[2].left_statistics, "left_hessians");
            ensure_polynomials_match_on_prover(
                runtime, inputs.right_hessians, inputs.channels[2].right_statistics, "right_hessians");
            witness.split_gain.left_counts = witness.split_statistics.left_statistics[0];
            witness.split_gain.right_counts = witness.split_statistics.right_statistics[0];
            witness.split_gain.left_gradients = witness.split_statistics.left_statistics[1];
            witness.split_gain.right_gradients = witness.split_statistics.right_statistics[1];
            witness.split_gain.left_hessians = witness.split_statistics.left_statistics[2];
            witness.split_gain.right_hessians = witness.split_statistics.right_statistics[2];
            witness.split_gain.left_gradient_squares = program.bind_virtual(witness_input(
                runtime, label_prefix + ".left_gradient_squares", inputs.left_gradient_squares, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.split_gain.right_gradient_squares = program.bind_virtual(witness_input(
                runtime, label_prefix + ".right_gradient_squares", inputs.right_gradient_squares, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.split_gain.parent_gradient_squares = program.bind_virtual(witness_input(
                runtime, label_prefix + ".parent_gradient_squares", inputs.parent_gradient_squares, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.split_gain.gain_numerators = program.bind_virtual(witness_input(
                runtime, label_prefix + ".gain_numerators", inputs.gain_numerators, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.split_gain.gain_denominators = program.bind_virtual(witness_input(
                runtime, label_prefix + ".gain_denominators", inputs.gain_denominators, matrix_degree_bound(inputs.node_count, candidate_width)));

            witness.quantized_split_gain.exact_gain_numerators = witness.split_gain.gain_numerators;
            witness.quantized_split_gain.gain_denominators = witness.split_gain.gain_denominators;
            witness.quantized_split_gain.encoded_gain_numerators = program.bind_virtual(witness_input(
                runtime, label_prefix + ".encoded_gain_numerators", inputs.encoded_gain_numerators, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.quantized_split_gain.encoded_quantized_gains = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".encoded_quantized_gains",
                inputs.encoded_quantized_gains,
                matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.quantized_split_gain.gain_remainders = program.bind_virtual(witness_input(
                runtime, label_prefix + ".gain_remainders", inputs.gain_remainders, matrix_degree_bound(inputs.node_count, candidate_width)));

            witness.split_selection.quantized_gains = program.bind_virtual(witness_input(
                runtime, label_prefix + ".quantized_gains", inputs.quantized_gains, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.split_selection.tie_break_scores = program.bind_virtual(witness_input(
                runtime, label_prefix + ".tie_break_scores", inputs.tie_break_scores, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.split_selection.candidate_mask = program.bind_virtual(witness_input(
                runtime, label_prefix + ".candidate_mask", inputs.candidate_mask, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.split_selection.active_node_mask = program.bind_virtual(
                witness_input(runtime, label_prefix + ".active_node_mask", inputs.active_node_mask, vector_degree_bound(inputs.node_count)));
            witness.depth_limit_mask = program.bind_virtual(
                witness_input(runtime, label_prefix + ".depth_limit_mask", inputs.depth_limit_mask, vector_degree_bound(inputs.node_count)));
            witness.split_selection.selector = program.bind_virtual(
                witness_input(runtime, label_prefix + ".selector", inputs.selector, matrix_degree_bound(inputs.node_count, candidate_width)));
            witness.split_selection.selected_gains = program.bind_virtual(
                witness_input(runtime, label_prefix + ".selected_gains", inputs.selected_gains, vector_degree_bound(inputs.node_count)));
            witness.split_selection.selected_tie_break_scores = program.bind_virtual(witness_input(
                runtime, label_prefix + ".selected_tie_break_scores", inputs.selected_tie_break_scores, vector_degree_bound(inputs.node_count)));
            witness.split_selection.selected_candidate_indices = program.bind_virtual(witness_input(
                runtime, label_prefix + ".selected_candidate_indices", inputs.selected_candidate_indices, vector_degree_bound(inputs.node_count)));
            witness.split_selection.selected_features = program.bind_virtual(witness_input(
                runtime, label_prefix + ".selected_features", inputs.selected_features, vector_degree_bound(inputs.node_count)));
            witness.split_selection.selected_bins = program.bind_virtual(
                witness_input(runtime, label_prefix + ".selected_bins", inputs.selected_bins, vector_degree_bound(inputs.node_count)));
        });

        workflow_profile_scope_void("forest_training_workflow.bind_leaf_and_boosting", [&] {
            witness.leaf_value.leaf_gradients = program.bind_virtual(
                witness_input(runtime, label_prefix + ".leaf_gradients", inputs.leaf_gradients, vector_degree_bound(inputs.node_count)));
            witness.leaf_value.leaf_hessians = program.bind_virtual(
                witness_input(runtime, label_prefix + ".leaf_hessians", inputs.leaf_hessians, vector_degree_bound(inputs.node_count)));
            witness.leaf_value.leaf_denominators = program.bind_virtual(witness_input(
                runtime, label_prefix + ".leaf_denominators", inputs.leaf_denominators, vector_degree_bound(inputs.node_count)));
            witness.leaf_value.encoded_negative_leaf_gradients = program.bind_virtual(witness_input(
                runtime, label_prefix + ".encoded_negative_leaf_gradients", inputs.encoded_negative_leaf_gradients, vector_degree_bound(inputs.node_count)));
            witness.leaf_value.encoded_leaf_values = program.bind_virtual(witness_input(
                runtime, label_prefix + ".encoded_leaf_values", inputs.encoded_leaf_values, vector_degree_bound(inputs.node_count)));
            witness.leaf_value.leaf_remainders = program.bind_virtual(
                witness_input(runtime, label_prefix + ".leaf_remainders", inputs.leaf_remainders, vector_degree_bound(inputs.node_count)));

            witness.boosting_state.tree_scores = witness.histogram_refresh.inference.tree_scores;
            witness.boosting_state.cumulative_scores = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".cumulative_scores",
                inputs.cumulative_scores,
                matrix_degree_bound(inputs.iteration_count, inputs.class_count * inputs.query_count)));
            witness.boosting_state.encoded_cumulative_scores = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".encoded_cumulative_scores",
                inputs.encoded_cumulative_scores,
                matrix_degree_bound(inputs.iteration_count, inputs.class_count * inputs.query_count)));
            witness.boosting_state.softmax_shifts = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".softmax_shifts",
                inputs.softmax_shifts,
                matrix_degree_bound(inputs.iteration_count, inputs.query_count)));
            witness.boosting_state.shift_selector = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".shift_selector",
                inputs.shift_selector,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.lifted_softmax_shifts = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".lifted_softmax_shifts",
                inputs.lifted_softmax_shifts,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.encoded_shifted_scores = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".encoded_shifted_scores",
                inputs.encoded_shifted_scores,
                matrix_degree_bound(inputs.iteration_count, inputs.class_count * inputs.query_count)));
            witness.boosting_state.softmax_exp_values = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".softmax_exp_values",
                inputs.softmax_exp_values,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.softmax_denominators = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".softmax_denominators",
                inputs.softmax_denominators,
                matrix_degree_bound(inputs.iteration_count, inputs.query_count)));
            witness.boosting_state.lifted_softmax_denominators = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".lifted_softmax_denominators",
                inputs.lifted_softmax_denominators,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.softmax_probabilities = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".softmax_probabilities",
                inputs.softmax_probabilities,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.softmax_remainders = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".softmax_remainders",
                inputs.softmax_remainders,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.base_labels = program.bind_virtual(witness_input(
                runtime, label_prefix + ".base_labels", inputs.base_labels, matrix_degree_bound(inputs.class_count, inputs.query_count)));
            witness.boosting_state.pseudo_residuals = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".pseudo_residuals",
                inputs.pseudo_residuals,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.encoded_diagonal_hessians = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".encoded_diagonal_hessians",
                inputs.encoded_diagonal_hessians,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.diagonal_hessian_remainders = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".diagonal_hessian_remainders",
                inputs.diagonal_hessian_remainders,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
            witness.boosting_state.diagonal_hessians = program.bind_virtual(witness_input(
                runtime,
                label_prefix + ".diagonal_hessians",
                inputs.diagonal_hessians,
                degree_bound_for_domain(inputs.iteration_count * inputs.class_count * inputs.query_count)));
        });

        workflow_profile_scope_void("forest_training_workflow.add_training_proof_check", [&] {
            program.forest_training_proof().add_check(witness,
                                                      inputs.tree_count,
                                                      inputs.node_count,
                                                      inputs.feature_count,
                                                      inputs.bin_count,
                                                      inputs.iteration_count,
                                                      inputs.class_count,
                                                      inputs.query_count,
                                                      inputs.input_bit_bound,
                                                      inputs.full_upper_bound,
                                                      inputs.structure_bit_bound,
                                                      inputs.gradient_encoding,
                                                      inputs.gain_numerator_encoding,
                                                      inputs.quantized_gain_encoding,
                                                      inputs.leaf_value_encoding,
                                                      inputs.score_encoding,
                                                      inputs.softmax_encoding,
                                                      inputs.hessian_bit_bound,
                                                      inputs.gain_denominator_bit_bound,
                                                      inputs.leaf_denominator_bit_bound,
                                                      inputs.gain_gap_bit_bound,
                                                      inputs.l2_regularization,
                                                      inputs.split_penalty,
                                                      inputs.split_gain_threshold,
                                                      inputs.max_depth,
                                                      inputs.learning_rate_numerator,
                                                      inputs.learning_rate_denominator,
                                                      inputs.loss_mode);
        });
    return true;
}

bool run_forest_training_proof_workflow(ProofRuntime& runtime,
                                        const forestTrainingProofInputs& inputs,
                                        const std::string& label_prefix,
                                        const forestCommittedNodePublicCommitments* committed_forest_public_inputs)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        return add_forest_training_proof_workflow_to_program(
            runtime, program, inputs, label_prefix, committed_forest_public_inputs);
    });
}

ProofArtifact prove_forest_training_proof_with_runtime(const kzgPublicParameters& public_params,
                                                       const forestTrainingProofInputs& inputs,
                                                       const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params, required_forest_training_proof_srs(inputs), "forest_training_workflow");
    return prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_forest_training_proof_workflow(runtime, inputs, label_prefix, nullptr);
    });
}

ProofArtifact prove_forest_training_proof_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingProofInputs& inputs,
    const forestCommittedNodePublicCommitments& committed_forest_public_inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params, required_forest_training_proof_srs(inputs), "forest_training_workflow");
    return prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_forest_training_proof_workflow(runtime, inputs, label_prefix, &committed_forest_public_inputs);
    });
}

VerificationResult verify_forest_training_proof_with_runtime(const kzgPublicParameters& public_params,
                                                             const ProofArtifact& artifact,
                                                             const forestTrainingProofInputs& inputs,
                                                             const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params, required_forest_training_proof_srs(inputs), "forest_training_workflow");
    const forestTrainingProofPublicInputs public_inputs =
        forest_training_proof_public_inputs_from_inputs(public_params, inputs);
    return verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_forest_training_proof_workflow(
            runtime, inputs, label_prefix, &public_inputs.committed_forest);
    });
}

VerificationResult verify_forest_training_proof_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestTrainingProofPublicInputs& inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params, required_forest_training_proof_srs(inputs), "forest_training_workflow");
    const forestTrainingProofInputs verifier_inputs = forest_training_proof_verifier_inputs_from_public_inputs(inputs);
    return verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_forest_training_proof_workflow(
            runtime, verifier_inputs, label_prefix, &inputs.committed_forest);
    });
}

forestTrainingProofVerifierPreprocessing prepare_forest_training_proof_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestTrainingProofPublicInputs& inputs,
    const std::string& label_prefix)
{
    ensure_kzg_srs_covers(public_params, required_forest_training_proof_srs(inputs), "forest_training_workflow");

    forestTrainingProofVerifierPreprocessing out;
    out.runtime = ProofRuntime::verifier_ptr(public_params, artifact.proof);
    out.program = std::make_unique<ProofProgram>(*out.runtime);
    const forestTrainingProofInputs verifier_inputs =
        forest_training_proof_verifier_inputs_from_public_inputs(inputs);
    add_forest_training_proof_workflow_to_program(
        *out.runtime, *out.program, verifier_inputs, label_prefix, &inputs.committed_forest);
    if (!out.program->prepare_verifier_preprocessing()) {
        throw std::runtime_error("forest_training_workflow: verifier preprocessing failed");
    }
    if (!out.program->complete_verifier_opening_preprocessing()) {
        throw std::runtime_error("forest_training_workflow: verifier opening preprocessing failed");
    }
    return out;
}

VerificationResult verify_forest_training_proof_preprocessed(
    forestTrainingProofVerifierPreprocessing& preprocessing)
{
    if (!preprocessing.runtime || !preprocessing.program) {
        throw std::runtime_error("forest_training_workflow: missing verifier preprocessing");
    }

    VerificationResult result;
    result.accepted = preprocessing.program->verify_preprocessed();
    result.proof_fully_consumed = preprocessing.runtime->proof_fully_consumed();
    result.transcript_state = preprocessing.runtime->transcript_state();
    return result;
}
