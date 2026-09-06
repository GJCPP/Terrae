#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "forest_training_export.hpp"
#include "forest_well_formedness_workflow.hpp"
#include "training_runtime_utils.hpp"

class forestInferenceQueryBatch {
public:
    poly query_tree_ids;
    poly lifted_inputs;
    size_t query_count = 0;
    size_t feature_count = 0;
};

class forestExportedInferenceWitnessBatch {
public:
    poly query_tree_ids;
    poly lifted_inputs;
    poly lower_bounds;
    poly upper_bounds;
    poly scores;
    poly leaf_indices;
    poly class_scores;
    poly predicted_scores;
    poly class_selector;
    size_t query_count = 0;
    size_t feature_count = 0;
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t sample_count = 0;
};

class forestTrainingThenInferenceInputs {
public:
    forestTrainingProofInputs training_inputs;
    forestInferenceQueryBatch inference_queries;
    std::optional<forestExportedInferenceWitnessBatch> explicit_inference_witness;
};

class forestExportedInferencePlan {
public:
    forestCommittedNodeTable committed_forest;
    forestCommittedNodePublicCommitments committed_forest_public_inputs;
    forestExportedLeafTable exported_leaf_table;
    forestInferencePublicTable inference_table;
    forestClassificationPublicTable classification_table;
    forestExportedInferenceWitnessBatch witness_batch;
};

class forestExportedInferencePlanProfile {
public:
    double ensure_srs_seconds = 0.0;
    double build_committed_forest_table_seconds = 0.0;
    double commit_committed_forest_public_inputs_seconds = 0.0;
    double export_leaf_table_seconds = 0.0;
    double build_inference_public_table_seconds = 0.0;
    double prepare_inference_witness_seconds = 0.0;
    double total_seconds = 0.0;

    double commit_forest_seconds() const
    {
        return build_committed_forest_table_seconds + commit_committed_forest_public_inputs_seconds;
    }
};

class forestTrainingThenInferenceArtifact {
public:
    ProofArtifact well_formedness_artifact;
    ProofArtifact training_artifact;
    ProofArtifact inference_artifact;
    forestExportedLeafTable exported_leaf_table;
};

class forestExportedInferenceBatchShape {
public:
    size_t query_count = 0;
    size_t feature_count = 0;
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t sample_count = 0;
};

class forestExportedInferenceVerifierPreprocessing {
public:
    std::unique_ptr<ProofRuntime> runtime;
    std::unique_ptr<ProofProgram> program;
};

class forestExportedClassificationVerifierPreprocessing {
public:
    std::unique_ptr<ProofRuntime> runtime;
    std::unique_ptr<ProofProgram> program;
};

class forestTrainingWithWellFormednessVerifierPreprocessing {
public:
    std::unique_ptr<ProofRuntime> runtime;
    std::unique_ptr<ProofProgram> program;
};

class forestTrainingThenInferenceVerifierBundle {
public:
    ProofArtifact well_formedness_artifact;
    ProofArtifact training_artifact;
    ProofArtifact inference_artifact;
    forestTrainingProofPublicInputs training_public_inputs;
    forestInferencePublicTable inference_table;
    forestExportedInferenceBatchShape inference_batch_shape;
};

class forestTrainingThenInferenceVerificationResult {
public:
    VerificationResult well_formedness_result;
    VerificationResult training_result;
    VerificationResult inference_result;
    bool accepted = false;
    bool proof_fully_consumed = false;
};

class forestTrainingThenInferencePreparedRun {
public:
    forestExportedInferencePlan exported_inference_plan;
    forestTrainingThenInferenceArtifact artifact;
    forestTrainingThenInferenceVerifierBundle verifier_bundle;
};

// A single proof session containing both forest well-formedness and training.
// All zero/lookup claims contributed by the two workflows are finalized by one
// ProofProgram and therefore reduce to one KZG batched opening proof.
ProofArtifact prove_forest_training_with_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingProofInputs& training_inputs,
    const forestWellFormednessInputs& well_formedness_inputs,
    const forestCommittedNodePublicCommitments& committed_forest_public_inputs,
    const std::string& label_prefix = "forest_training_with_well_formedness");

VerificationResult verify_forest_training_with_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestTrainingProofPublicInputs& training_inputs,
    const forestWellFormednessPublicInputs& well_formedness_inputs,
    const std::string& label_prefix = "forest_training_with_well_formedness");

forestTrainingWithWellFormednessVerifierPreprocessing
prepare_forest_training_with_well_formedness_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestTrainingProofPublicInputs& training_inputs,
    const forestWellFormednessPublicInputs& well_formedness_inputs,
    const std::string& label_prefix = "forest_training_with_well_formedness");

VerificationResult verify_forest_training_with_well_formedness_preprocessed(
    forestTrainingWithWellFormednessVerifierPreprocessing& preprocessing);

kzgSrsRequirement required_training_then_inference_srs(const forestTrainingThenInferenceInputs& inputs);

kzgSrsRequirement required_training_then_inference_srs(const forestTrainingThenInferenceVerifierBundle& bundle);

kzgSrsRequirement required_exported_forest_inference_srs(const forestTrainingThenInferenceInputs& inputs);

kzgSrsRequirement required_exported_forest_inference_srs(const forestInferencePublicTable& table,
                                                         const forestExportedInferenceWitnessBatch& batch);

kzgSrsRequirement required_exported_forest_inference_srs(const forestInferencePublicTable& table,
                                                         const forestExportedInferenceBatchShape& batch_shape);

kzgSrsRequirement required_exported_forest_classification_srs(const forestTrainingThenInferenceInputs& inputs);
kzgSrsRequirement required_exported_forest_classification_srs(
    const forestClassificationPublicTable& table, const forestExportedInferenceWitnessBatch& batch);
kzgSrsRequirement required_exported_forest_classification_srs(
    const forestClassificationPublicTable& table, const forestExportedInferenceBatchShape& batch_shape);

forestExportedInferenceWitnessBatch derive_exported_inference_witness_batch(
    const forestCommittedNodeTable& committed_forest,
    const forestExportedLeafTable& exported_forest,
    const forestInferenceQueryBatch& queries);

forestExportedInferencePlan assemble_exported_inference_plan(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const std::string& key_prefix = "forest_training_then_inference");

forestExportedInferencePlan assemble_exported_inference_plan_with_profile(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    forestExportedInferencePlanProfile& profile,
    const std::string& key_prefix = "forest_training_then_inference");

bool run_exported_forest_inference_workflow(
    ProofRuntime& runtime,
    const forestInferencePublicTable& table,
    const forestExportedInferenceWitnessBatch& batch,
    const std::string& label_prefix = "forest_exported_inference");

bool add_exported_forest_inference_verifier_workflow_to_program(
    ProofProgram& program,
    const forestInferencePublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix = "forest_exported_inference");

bool run_exported_forest_inference_verifier_workflow(
    ProofRuntime& runtime,
    const forestInferencePublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix = "forest_exported_inference");

ProofArtifact prove_exported_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestInferencePublicTable& table,
    const forestExportedInferenceWitnessBatch& batch,
    const std::string& label_prefix = "forest_exported_inference");

VerificationResult verify_exported_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestInferencePublicTable& table,
    const forestExportedInferenceWitnessBatch& batch,
    const std::string& label_prefix = "forest_exported_inference");

VerificationResult verify_exported_forest_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestInferencePublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix = "forest_exported_inference");

forestExportedInferenceVerifierPreprocessing prepare_exported_forest_inference_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestInferencePublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix = "forest_exported_inference");

VerificationResult verify_exported_forest_inference_preprocessed(
    forestExportedInferenceVerifierPreprocessing& preprocessing);


bool run_exported_forest_classification_workflow(
    ProofRuntime& runtime,
    const forestClassificationPublicTable& table,
    const forestExportedInferenceWitnessBatch& batch,
    const std::string& label_prefix = "forest_exported_classification");

bool add_exported_forest_classification_verifier_workflow_to_program(
    ProofProgram& program,
    const forestClassificationPublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix = "forest_exported_classification");

ProofArtifact prove_exported_forest_classification_with_runtime(
    const kzgPublicParameters& public_params,
    const forestClassificationPublicTable& table,
    const forestExportedInferenceWitnessBatch& batch,
    const std::string& label_prefix = "forest_exported_classification");

VerificationResult verify_exported_forest_classification_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestClassificationPublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix = "forest_exported_classification");

forestExportedClassificationVerifierPreprocessing prepare_exported_forest_classification_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestClassificationPublicTable& table,
    const forestExportedInferenceBatchShape& batch_shape,
    const std::string& label_prefix = "forest_exported_classification");

VerificationResult verify_exported_forest_classification_preprocessed(
    forestExportedClassificationVerifierPreprocessing& preprocessing);

forestTrainingThenInferenceArtifact prove_training_then_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const std::string& label_prefix = "forest_training_then_inference");

forestTrainingThenInferenceArtifact prove_training_then_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const forestExportedInferencePlan& plan,
    const std::string& label_prefix = "forest_training_then_inference");

forestTrainingThenInferenceVerifierBundle prepare_training_then_inference_verifier_bundle(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const forestTrainingThenInferenceArtifact& artifact,
    const std::string& label_prefix = "forest_training_then_inference");

forestTrainingThenInferenceVerifierBundle prepare_training_then_inference_verifier_bundle(
    const forestTrainingThenInferenceInputs& inputs,
    const forestTrainingThenInferenceArtifact& artifact,
    const forestExportedInferencePlan& plan);

forestTrainingThenInferencePreparedRun prove_and_prepare_training_then_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceInputs& inputs,
    const std::string& label_prefix = "forest_training_then_inference");

forestTrainingThenInferenceVerificationResult verify_training_then_inference_with_runtime(
    const kzgPublicParameters& public_params,
    const forestTrainingThenInferenceVerifierBundle& bundle,
    const std::string& label_prefix = "forest_training_then_inference");
