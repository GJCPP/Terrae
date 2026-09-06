#pragma once

#include "plonk/plonk.hpp"
#include "zkboost/compiler/generic_circuit.hpp"
#include "zkboost/shared_trace/shared_prediction.hpp"
#include "zkboost/shared_trace/validate_forest_fetch.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

class zkboostSharedPredictionSummary {
public:
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t sample_count = 0;
    size_t tree_score_count = 0;
    size_t score_bit_bound = 0;
};

class zkboostSharedPredictionView {
public:
    zkboostSharedPredictionSummary summary;
    std::vector<std::int64_t> tree_scores;
    std::vector<std::int64_t> class_scores;
    std::vector<std::int64_t> predicted_scores;
    std::vector<std::int64_t> class_selector;
    std::vector<std::int64_t> predicted_labels;
};

class zkboostSharedPredictionPlonkInstance {
public:
    plonkSetup setup;
    plonkWitness witness;
    std::vector<Fr> public_inputs;
    zkboostSharedPredictionView view;
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs;
};

zkboostSharedPredictionView build_zkboost_shared_prediction_view(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);

zkboostSharedPredictionView build_zkboost_shared_prediction_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedTracePrediction& prediction);

std::vector<Fr> encode_zkboost_shared_prediction_public_inputs(
    const zkboostSharedPredictionView& view);

zkboostSharedPredictionPlonkInstance build_zkboost_shared_prediction_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostSharedPredictionPlonkInstance build_zkboost_shared_prediction_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedTracePrediction& prediction,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostSharedPredictionPlonkInstance build_zkboost_shared_prediction_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedPredictionView& view,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostGenericCircuit build_zkboost_shared_prediction_generic_circuit_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedPredictionView& view,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

size_t required_zkboost_shared_prediction_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

size_t required_zkboost_shared_prediction_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedPredictionView& view,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

plonkProof prove_zkboost_shared_prediction_plonk(
    const zkboostSharedPredictionPlonkInstance& instance);

bool verify_zkboost_shared_prediction_plonk(
    const zkboostSharedPredictionPlonkInstance& instance,
    const plonkProof& proof);

bool verify_zkboost_shared_prediction_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof);

bool link_zkboost_shared_validate_forest_to_prediction(
    const zkboostSharedValidateForestFetchInstance& forest_fetch_instance,
    const zkboostSharedPredictionPlonkInstance& prediction_instance);
