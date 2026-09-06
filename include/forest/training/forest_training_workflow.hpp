#pragma once

#include <array>
#include <memory>
#include <string>

#include "forest_committed_forest_public.hpp"
#include "forest_training_proof.hpp"
#include "proof_runtime.hpp"
#include "training_runtime_utils.hpp"

class forestTrainingProofChannelInputs {
public:
    poly query_values;
    poly lifted_values;
    poly leaf_histograms;
    poly full_histograms;
    poly split_histograms;
    poly left_statistics;
    poly right_statistics;
};

class forestTrainingProofInputs {
public:
    poly root_indices;
    poly split_node_mask;
    poly node_depths;
    poly left_child_indices;
    poly right_child_indices;
    poly node_tree_ids;
    poly feature_selector;
    poly lower_bounds;
    poly upper_bounds;
    poly left_lower;
    poly left_upper;
    poly right_lower;
    poly right_upper;
    poly thresholds;

    poly query_tree_ids;
    poly lifted_inputs;
    poly leaf_indices;
    poly leaf_lower_bounds;
    poly leaf_upper_bounds;
    poly tree_scores;
    poly training_node_tree_ids;
    poly training_node_lower_bounds;
    poly training_node_upper_bounds;
    poly node_leaf_mask;
    poly node_leaf_values;
    poly lifted_leaf_indices;
    poly flattened_bucket_indices;

    std::array<forestTrainingProofChannelInputs, 3> channels;

    poly left_gradients;
    poly right_gradients;
    poly left_hessians;
    poly right_hessians;
    poly left_gradient_squares;
    poly right_gradient_squares;
    poly parent_gradient_squares;
    poly gain_numerators;
    poly gain_denominators;
    poly encoded_gain_numerators;
    poly encoded_quantized_gains;
    poly gain_remainders;

    poly quantized_gains;
    poly tie_break_scores;
    poly candidate_mask;
    poly active_node_mask;
    poly depth_limit_mask;
    poly selector;
    poly selected_gains;
    poly selected_tie_break_scores;
    poly selected_candidate_indices;
    poly selected_features;
    poly selected_bins;

    poly leaf_gradients;
    poly leaf_hessians;
    poly leaf_denominators;
    poly encoded_negative_leaf_gradients;
    poly encoded_leaf_values;
    poly leaf_remainders;

    poly cumulative_scores;
    poly encoded_cumulative_scores;
    poly softmax_shifts;
    poly shift_selector;
    poly lifted_softmax_shifts;
    poly encoded_shifted_scores;
    poly softmax_exp_values;
    poly softmax_denominators;
    poly lifted_softmax_denominators;
    poly softmax_probabilities;
    poly softmax_remainders;
    poly base_labels;
    poly pseudo_residuals;
    poly encoded_diagonal_hessians;
    poly diagonal_hessian_remainders;
    poly diagonal_hessians;

    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t query_count = 0;
    size_t input_bit_bound = 0;
    size_t full_upper_bound = 0;
    size_t structure_bit_bound = 0;
    fixedPointEncoding gradient_encoding;
    fixedPointEncoding gain_numerator_encoding;
    fixedPointEncoding quantized_gain_encoding;
    fixedPointEncoding leaf_value_encoding;
    fixedPointEncoding score_encoding;
    fixedPointEncoding softmax_encoding;
    size_t hessian_bit_bound = 0;
    size_t gain_denominator_bit_bound = 0;
    size_t leaf_denominator_bit_bound = 0;
    size_t gain_gap_bit_bound = 0;
    forestTrainingLoss loss_mode = forestTrainingLoss::FirstOrderRss;
    size_t l2_regularization = 0;
    size_t learning_rate_numerator = 1;
    size_t learning_rate_denominator = 1;
    Fr split_penalty = Fr(0);
    Fr split_gain_threshold = Fr(0);
    size_t max_depth = 0;
};

class forestTrainingProofPublicInputs {
public:
    forestCommittedNodePublicCommitments committed_forest;
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t query_count = 0;
    size_t input_bit_bound = 0;
    size_t full_upper_bound = 0;
    size_t structure_bit_bound = 0;
    fixedPointEncoding gradient_encoding;
    fixedPointEncoding gain_numerator_encoding;
    fixedPointEncoding quantized_gain_encoding;
    fixedPointEncoding leaf_value_encoding;
    fixedPointEncoding score_encoding;
    fixedPointEncoding softmax_encoding;
    size_t hessian_bit_bound = 0;
    size_t gain_denominator_bit_bound = 0;
    size_t leaf_denominator_bit_bound = 0;
    size_t gain_gap_bit_bound = 0;
    forestTrainingLoss loss_mode = forestTrainingLoss::FirstOrderRss;
    size_t l2_regularization = 0;
    size_t learning_rate_numerator = 1;
    size_t learning_rate_denominator = 1;
    Fr split_penalty = Fr(0);
    Fr split_gain_threshold = Fr(0);
    size_t max_depth = 0;
};

class forestTrainingProofVerifierPreprocessing {
public:
    std::unique_ptr<ProofRuntime> runtime;
    std::unique_ptr<ProofProgram> program;
};

forestTrainingProofPublicInputs forest_training_proof_public_inputs_from_inputs(
    const kzgPublicParameters& public_params,
    const forestTrainingProofInputs& inputs,
    const std::string& forest_key_prefix = "committed_forest");

forestTrainingProofPublicInputs forest_training_proof_public_inputs_from_inputs(
    const forestCommittedNodePublicCommitments& committed_forest,
    const forestTrainingProofInputs& inputs);

forestTrainingProofInputs forest_training_proof_verifier_inputs_from_public_inputs(
    const forestTrainingProofPublicInputs& inputs);

kzgSrsRequirement required_forest_training_proof_srs(const forestTrainingProofInputs& inputs);

kzgSrsRequirement required_forest_training_proof_srs(const forestTrainingProofPublicInputs& inputs);

bool add_forest_training_proof_workflow_to_program(
    ProofRuntime& runtime,
    ProofProgram& program,
    const forestTrainingProofInputs& inputs,
    const std::string& label_prefix = "forest_training",
    const forestCommittedNodePublicCommitments* committed_forest_public_inputs = nullptr);

bool run_forest_training_proof_workflow(ProofRuntime& runtime,
                                        const forestTrainingProofInputs& inputs,
                                        const std::string& label_prefix = "forest_training",
                                        const forestCommittedNodePublicCommitments* committed_forest_public_inputs =
                                            nullptr);

ProofArtifact prove_forest_training_proof_with_runtime(const kzgPublicParameters& public_params,
                                                       const forestTrainingProofInputs& inputs,
                                                       const std::string& label_prefix = "forest_training");

ProofArtifact prove_forest_training_proof_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingProofInputs& inputs,
    const forestCommittedNodePublicCommitments& committed_forest_public_inputs,
    const std::string& label_prefix = "forest_training");

VerificationResult verify_forest_training_proof_with_runtime(const kzgPublicParameters& public_params,
                                                             const ProofArtifact& artifact,
                                                             const forestTrainingProofInputs& inputs,
                                                             const std::string& label_prefix = "forest_training");

VerificationResult verify_forest_training_proof_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestTrainingProofPublicInputs& inputs,
    const std::string& label_prefix = "forest_training");

forestTrainingProofVerifierPreprocessing prepare_forest_training_proof_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestTrainingProofPublicInputs& inputs,
    const std::string& label_prefix = "forest_training");

VerificationResult verify_forest_training_proof_preprocessed(
    forestTrainingProofVerifierPreprocessing& preprocessing);
