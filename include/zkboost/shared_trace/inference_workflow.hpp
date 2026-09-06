#pragma once

#include "proof_runtime.hpp"
#include "zkboost/shared_trace/forest_commitments.hpp"
#include "zkboost/shared_trace/prediction_plonk.hpp"
#include "zkboost/shared_trace/shared_prediction.hpp"
#include "zkboost/shared_trace/validate_forest_fetch.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"

class zkboostSharedInferenceSummary {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t sample_count = 0;
    size_t checked_paths = 0;
};

class zkboostSharedInferenceInstance {
public:
    zkboostSharedCommittedForest committed_forest;
    zkboostSharedValidateForestPlonkInstance inference_plonk;
    zkboostSharedValidateForestFetchInstance fetch;
    zkboostSharedPredictionView prediction_view;
    zkboostSharedTracePrediction prediction;
    zkboostSharedInferenceSummary summary;
};

class zkboostSharedInferenceProof {
public:
    plonkProof inference_plonk;
    SerializedProof forest_fetch;
    SerializedProof cross_session_opening_batch;
};

class zkboostSharedInferenceVerifierBundle {
public:
    plonkPreprocessedVerifierInput inference_plonk;
    zkboostSharedValidateForestFetchVerifierBundle forest_fetch;
    const zkboostSharedValidateForestFetchInstance* fetch_instance = nullptr;
    const kzgPublicParameters* public_parameters = nullptr;
    bool committed_table_ok = false;
    bool links_ok = false;
};

class zkboostSharedInferenceVerifierPreprocessing {
public:
    std::unique_ptr<ProofRuntime> runtime;
    std::unique_ptr<ProofProgram> program;
    const zkboostSharedValidateForestFetchVerifierBundle* forest_fetch = nullptr;
    const SerializedProof* forest_fetch_proof = nullptr;
};

zkboostSharedInferenceInstance build_zkboost_shared_inference_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostSharedInferenceInstance build_zkboost_shared_inference_instance_from_committed_forest(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const zkboostSharedCommittedForest& committed_forest,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostSharedKzgSrsRequirement required_zkboost_shared_inference_srs_requirement(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

size_t required_zkboost_shared_inference_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostSharedInferenceProof prove_zkboost_shared_inference(
    const zkboostSharedInferenceInstance& instance);

bool verify_zkboost_shared_inference_links(
    const zkboostSharedInferenceInstance& instance);

bool verify_zkboost_shared_inference(
    const zkboostSharedInferenceInstance& instance,
    const zkboostSharedInferenceProof& proof);

zkboostSharedInferenceVerifierBundle prepare_zkboost_shared_inference_verifier_bundle(
    const zkboostSharedInferenceInstance& instance);

zkboostSharedInferenceVerifierPreprocessing
prepare_zkboost_shared_inference_verifier_preprocessing(
    const zkboostSharedInferenceVerifierBundle& verifier_bundle,
    const zkboostSharedInferenceProof& proof);

bool verify_zkboost_shared_inference_preprocessed(
    zkboostSharedInferenceVerifierPreprocessing& preprocessing);

bool verify_zkboost_shared_inference_preprocessed(
    const zkboostSharedInferenceVerifierBundle& verifier_bundle,
    const zkboostSharedInferenceProof& proof);

size_t serialized_zkboost_shared_inference_proof_bytes(
    const zkboostSharedInferenceProof& proof);
