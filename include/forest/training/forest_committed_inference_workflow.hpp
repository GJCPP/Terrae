#pragma once

#include <optional>
#include <string>

#include "forest_training_export.hpp"
#include "forest_training_then_inference_workflow.hpp"
#include "training_runtime_utils.hpp"

class forestCommittedInferenceWitnessBatch {
public:
    poly leaf_indices;
    poly leaf_lower_bounds;
    poly leaf_upper_bounds;
    poly tree_scores;
    size_t query_count = 0;
    size_t feature_count = 0;
};

class forestCommittedInferenceInputs {
public:
    forestCommittedNodeTable committed_forest;
    forestInferenceQueryBatch queries;
    size_t input_bit_bound = 0;
    std::optional<forestCommittedInferenceWitnessBatch> explicit_witness;
};

class forestInferenceQueryPublicCommitments {
public:
    Group1 query_tree_ids;
    Group1 lifted_inputs;
};

class forestCommittedInferencePublicInputs {
public:
    forestCommittedNodePublicCommitments committed_forest;
    forestInferenceQueryBatch queries;
    std::optional<forestInferenceQueryPublicCommitments> query_commitments;
    size_t input_bit_bound = 0;
};

forestCommittedInferencePublicInputs forest_committed_inference_public_inputs_from_inputs(
    const kzgPublicParameters& public_params,
    const forestCommittedInferenceInputs& inputs,
    const std::string& forest_key_prefix = "committed_forest");

forestInferenceQueryPublicCommitments commit_forest_inference_query_public_commitments(
    const kzgPublicParameters& public_params,
    const forestInferenceQueryBatch& queries,
    const std::string& key_prefix = "forest_inference_queries");

kzgSrsRequirement required_committed_forest_inference_srs(const forestCommittedInferenceInputs& inputs);

kzgSrsRequirement required_committed_forest_inference_srs(const forestCommittedInferencePublicInputs& inputs);

forestCommittedInferenceWitnessBatch derive_committed_inference_witness_batch(
    const forestCommittedNodeTable& committed_forest,
    const forestInferenceQueryBatch& queries);

bool run_committed_forest_inference_workflow(
    ProofRuntime& runtime,
    const forestCommittedInferenceInputs& inputs,
    const std::string& label_prefix = "forest_committed_inference",
    const forestCommittedNodePublicCommitments* committed_forest_public_inputs = nullptr);

bool run_committed_forest_inference_verifier_workflow(
    ProofRuntime& runtime,
    const forestCommittedInferencePublicInputs& inputs,
    const std::string& label_prefix = "forest_committed_inference");

bool add_committed_forest_inference_workflow_to_program(
    ProofRuntime& runtime,
    ProofProgram& program,
    const forestCommittedInferenceInputs& inputs,
    const std::string& label_prefix = "forest_committed_inference",
    const forestCommittedNodePublicCommitments* committed_forest_public_inputs = nullptr);

bool add_committed_forest_inference_verifier_workflow_to_program(
    ProofRuntime& runtime,
    ProofProgram& program,
    const forestCommittedInferencePublicInputs& inputs,
    const std::string& label_prefix = "forest_committed_inference");

ProofArtifact prove_committed_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestCommittedInferenceInputs& inputs,
    const std::string& label_prefix = "forest_committed_inference");

VerificationResult verify_committed_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestCommittedInferenceInputs& inputs,
    const std::string& label_prefix = "forest_committed_inference");

VerificationResult verify_committed_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestCommittedInferencePublicInputs& inputs,
    const std::string& label_prefix = "forest_committed_inference");
