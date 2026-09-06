#include "zkboost/shared_trace/inference_workflow.hpp"

#include "plonk/prover.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

void require_condition(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(std::string("zkboost_shared_inference_workflow: ") + message);
    }
}

zkboostSharedInferenceSummary summarize_inference(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    zkboostSharedInferenceSummary out;
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(inference.query_count % training.tree_count == 0,
                      "inference query_count must be divisible by tree_count");
    out.tree_count = training.tree_count;
    out.node_count = training.node_count;
    out.feature_count = training.feature_count;
    out.bin_count = training.bin_count;
    out.sample_count = inference.query_count / training.tree_count;
    out.checked_paths = inference.query_count;
    return out;
}

zkboostGenericCircuit build_unified_inference_circuit(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedValidateForestView& routing_view,
    const zkboostSharedPredictionView& prediction_view,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    return concatenate_zkboost_generic_circuits({
        build_zkboost_shared_validate_forest_generic_circuit_from_view(
            routing_view, training.bin_count, public_input_mode),
        build_zkboost_shared_prediction_generic_circuit_from_view(
            training, prediction_view, public_input_mode),
    });
}

bool add_deferred_plonk_prover_openings(
    kzgProtocol& kzg,
    const plonkSetup& setup,
    const std::vector<Fr>& public_inputs,
    const plonkDeferredOpeningProof& deferred)
{
    plonkDeferredOpeningClaims claims;
    const plonkPreprocessedVerifierInput preprocessed =
        preprocess_plonk_verifier_input(setup.verification_key, public_inputs);
    if (!build_plonk_deferred_opening_claims(preprocessed, deferred.proof, claims)) {
        return false;
    }
    if (deferred.openings.zeta != claims.zeta ||
        deferred.openings.zeta_omega != claims.zeta_omega ||
        deferred.openings.batched_zeta_eval != claims.zeta_value ||
        deferred.openings.batched_zeta_omega_eval != claims.zeta_omega_value) {
        return false;
    }

    kzgCommitment expected_zeta;
    expected_zeta.com = claims.zeta_commitment;
    const kzgCommitment zeta_commitment =
        kzg.commit(deferred.openings.batched_zeta, expected_zeta);
    kzg.add_opening_check(zeta_commitment, claims.zeta, claims.zeta_value);

    kzgCommitment expected_zeta_omega;
    expected_zeta_omega.com = claims.zeta_omega_commitment;
    const kzgCommitment zeta_omega_commitment =
        kzg.commit(deferred.openings.batched_zeta_omega, expected_zeta_omega);
    kzg.add_opening_check(
        zeta_omega_commitment, claims.zeta_omega, claims.zeta_omega_value);
    return true;
}

bool add_deferred_plonk_verifier_openings(
    kzgProtocol& kzg,
    const plonkPreprocessedVerifierInput& preprocessed,
    const plonkProof& proof)
{
    plonkDeferredOpeningClaims claims;
    if (!build_plonk_deferred_opening_claims(preprocessed, proof, claims)) {
        return false;
    }
    const kzgCommitment zeta_commitment =
        kzg.adopt_commitment(claims.zeta_commitment);
    kzg.add_opening_check(zeta_commitment, claims.zeta, claims.zeta_value);
    const kzgCommitment zeta_omega_commitment =
        kzg.adopt_commitment(claims.zeta_omega_commitment);
    kzg.add_opening_check(
        zeta_omega_commitment, claims.zeta_omega, claims.zeta_omega_value);
    return true;
}

} // namespace

zkboostSharedInferenceInstance build_zkboost_shared_inference_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    const zkboostSharedCommittedForest committed_forest =
        build_zkboost_shared_committed_forest(public_parameters, training);
    return build_zkboost_shared_inference_instance_from_committed_forest(
        training,
        inference,
        committed_forest,
        public_parameters,
        public_input_mode);
}

zkboostSharedInferenceInstance build_zkboost_shared_inference_instance_from_committed_forest(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const zkboostSharedCommittedForest& committed_forest,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(inference.query_count % training.tree_count == 0,
                      "inference query_count must be divisible by tree_count");
    validate_zkboost_shared_tree_ensemble(committed_forest.ensemble);
    require_condition(committed_forest.summary.tree_count == training.tree_count,
                      "committed forest tree_count mismatch");
    require_condition(committed_forest.summary.node_count == training.node_count,
                      "committed forest node_count mismatch");
    require_condition(committed_forest.summary.feature_count == training.feature_count,
                      "committed forest feature_count mismatch");
    require_condition(committed_forest.summary.bin_count == training.bin_count,
                      "committed forest bin_count mismatch");

    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);

    zkboostSharedInferenceInstance out;
    out.committed_forest = committed_forest;
    out.prediction = derive_zkboost_shared_trace_prediction(training, inference);
    out.prediction_view =
        build_zkboost_shared_prediction_view(training, out.prediction);
    const zkboostGenericCircuit inference_circuit =
        build_unified_inference_circuit(
            training, view, out.prediction_view, public_input_mode);
    const zkboostCompiledCircuitStatement statement =
        compile_zkboost_generic_circuit_statement(
            "shared_inference", inference_circuit, public_parameters);
    out.inference_plonk.setup = statement.setup;
    out.inference_plonk.witness = statement.witness;
    out.inference_plonk.public_inputs = statement.public_inputs;
    out.inference_plonk.linked_input_digest =
        digest_zkboost_shared_validate_forest_public_inputs(view);
    out.inference_plonk.public_input_mode = public_input_mode;
    out.inference_plonk.summary = view.summary;
    out.inference_plonk.committed_table =
        build_zkboost_shared_validate_forest_committed_table_metadata(
            training.bin_count,
            inference_circuit.lookup_table_rows.size(),
            statement.setup.verification_key);
    out.fetch = build_zkboost_shared_validate_forest_fetch_instance_from_view(
        view,
        out.committed_forest.ensemble,
        out.committed_forest.commitments,
        public_parameters);
    out.summary = summarize_inference(training, inference);
    return out;
}

zkboostSharedKzgSrsRequirement required_zkboost_shared_inference_srs_requirement(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(training);
    const zkboostSharedPredictionView prediction =
        build_zkboost_shared_prediction_view(training, inference);
    const zkboostGenericCircuit inference_circuit =
        build_unified_inference_circuit(
            training, view, prediction, public_input_mode);
    const zkboostSharedKzgSrsRequirement fetch_requirement =
        required_zkboost_shared_validate_forest_fetch_srs_requirement_from_view(
            view, ensemble);
    zkboostSharedKzgSrsRequirement out =
        max_zkboost_shared_kzg_srs_requirement(
            max_zkboost_shared_kzg_srs_requirement(
                required_zkboost_shared_forest_commitment_srs_requirement(ensemble),
                zkboostSharedKzgSrsRequirement{
                    required_zkboost_generic_circuit_plonk_srs_size(
                        inference_circuit),
                    2,
                    0,
                    2,
                }),
            fetch_requirement);
    // Forest fetches use an independent prepared batch.  The circuit batch
    // contains only the unified PLONK circuit's zeta and zeta*omega points.
    out.opening_deg2_plus_1_floor = std::max(
        out.opening_deg2_plus_1_floor,
        size_t(3));
    const size_t degree_shift_requirement =
        out.min_degree_bound_plus_1 == 0
            ? size_t(1)
            : out.deg1_plus_1 - out.min_degree_bound_plus_1 + 1;
    out.deg2_plus_1 = std::max(
        out.deg2_plus_1,
        std::max(out.opening_deg2_plus_1_floor, degree_shift_requirement));
    return out;
}

size_t required_zkboost_shared_inference_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    return required_zkboost_shared_inference_srs_requirement(training, inference, public_input_mode).deg1_plus_1;
}

zkboostSharedInferenceProof prove_zkboost_shared_inference(
    const zkboostSharedInferenceInstance& instance)
{
    const plonkDeferredOpeningProof inference_plonk =
        prove_plonk_deferred_opening(
            instance.inference_plonk.setup.proving_key,
            instance.inference_plonk.witness,
            instance.inference_plonk.public_inputs);

    ProofRuntime runtime =
        ProofRuntime::prover_borrowed(instance.fetch.public_parameters);
    ProofProgram program(runtime);
    const bool ok = add_deferred_plonk_prover_openings(
             runtime.kzg(),
             instance.inference_plonk.setup,
             instance.inference_plonk.public_inputs,
             inference_plonk);
    if (!ok || !program.finalize()) {
        throw std::runtime_error(
            "prove_zkboost_shared_inference: cross-session opening batch failed");
    }

    zkboostSharedInferenceProof out;
    out.inference_plonk = inference_plonk.proof;
    out.forest_fetch = prove_zkboost_shared_validate_forest_fetch(instance.fetch);
    out.cross_session_opening_batch = runtime.take_proof();
    return out;
}

bool verify_zkboost_shared_inference_links(
    const zkboostSharedInferenceInstance& instance)
{
    const zkboostSharedPredictionSummary& prediction =
        instance.prediction_view.summary;
    const bool prediction_shape_ok =
        instance.fetch.summary.tree_count ==
            prediction.iteration_count * prediction.class_count &&
        instance.fetch.summary.sample_count == prediction.sample_count &&
        instance.fetch.summary.checked_paths == prediction.tree_score_count;
    return prediction_shape_ok &&
           link_zkboost_shared_validate_forest_to_fetch(
               instance.inference_plonk,
               instance.fetch) &&
           instance.fetch.view.routing.tree_scores ==
               instance.prediction_view.tree_scores;
}

bool verify_zkboost_shared_inference(
    const zkboostSharedInferenceInstance& instance,
    const zkboostSharedInferenceProof& proof)
{
    return verify_zkboost_shared_inference_preprocessed(
        prepare_zkboost_shared_inference_verifier_bundle(instance),
        proof);
}

zkboostSharedInferenceVerifierBundle prepare_zkboost_shared_inference_verifier_bundle(
    const zkboostSharedInferenceInstance& instance)
{
    zkboostSharedInferenceVerifierBundle out;
    out.inference_plonk = preprocess_plonk_verifier_input(
        instance.inference_plonk.setup.verification_key,
        instance.inference_plonk.public_inputs);
    out.forest_fetch = prepare_zkboost_shared_validate_forest_fetch_verifier_bundle(
        instance.fetch);
    out.fetch_instance = &instance.fetch;
    out.public_parameters = &instance.fetch.public_parameters;
    out.committed_table_ok = verify_zkboost_shared_validate_forest_committed_table(
        instance.inference_plonk.setup.verification_key,
        instance.inference_plonk.committed_table);
    out.links_ok = verify_zkboost_shared_inference_links(instance);
    return out;
}

zkboostSharedInferenceVerifierPreprocessing
prepare_zkboost_shared_inference_verifier_preprocessing(
    const zkboostSharedInferenceVerifierBundle& verifier_bundle,
    const zkboostSharedInferenceProof& proof)
{
    if (!verifier_bundle.committed_table_ok ||
        !verifier_bundle.links_ok ||
        verifier_bundle.fetch_instance == nullptr ||
        verifier_bundle.public_parameters == nullptr) {
        throw std::runtime_error(
            "prepare_zkboost_shared_inference_verifier_preprocessing: invalid bundle");
    }

    zkboostSharedInferenceVerifierPreprocessing out;
    out.runtime = ProofRuntime::verifier_ptr(
        *verifier_bundle.public_parameters,
        proof.cross_session_opening_batch);
    out.program = std::make_unique<ProofProgram>(*out.runtime);
    out.forest_fetch = &verifier_bundle.forest_fetch;
    out.forest_fetch_proof = &proof.forest_fetch;
    const bool ok = add_deferred_plonk_verifier_openings(
             out.runtime->kzg(),
             verifier_bundle.inference_plonk,
             proof.inference_plonk);
    if (!ok || !out.program->prepare_verifier_preprocessing() ||
        !out.program->complete_verifier_opening_preprocessing()) {
        throw std::runtime_error(
            "prepare_zkboost_shared_inference_verifier_preprocessing: preprocessing failed");
    }
    return out;
}

bool verify_zkboost_shared_inference_preprocessed(
    zkboostSharedInferenceVerifierPreprocessing& preprocessing)
{
    if (!preprocessing.runtime ||
        !preprocessing.program ||
        !preprocessing.program->verify_preprocessed() ||
        !preprocessing.runtime->proof_fully_consumed() ||
        preprocessing.forest_fetch == nullptr ||
        preprocessing.forest_fetch_proof == nullptr) {
        return false;
    }
    return verify_zkboost_shared_validate_forest_fetch_preprocessed(
        *preprocessing.forest_fetch, *preprocessing.forest_fetch_proof);
}

bool verify_zkboost_shared_inference_preprocessed(
    const zkboostSharedInferenceVerifierBundle& verifier_bundle,
    const zkboostSharedInferenceProof& proof)
{
    try {
        zkboostSharedInferenceVerifierPreprocessing preprocessing =
            prepare_zkboost_shared_inference_verifier_preprocessing(
                verifier_bundle, proof);
        return verify_zkboost_shared_inference_preprocessed(preprocessing);
    } catch (const std::exception&) {
        return false;
    }
}

size_t serialized_zkboost_shared_inference_proof_bytes(
    const zkboostSharedInferenceProof& proof)
{
    const size_t omitted_opening_bytes = 2 * G1::getSerializedByteSize();
    const size_t plonk_bytes = serialize_plonk_proof(proof.inference_plonk).bytes.size();
    if (plonk_bytes < omitted_opening_bytes) {
        throw std::runtime_error(
            "serialized_zkboost_shared_inference_proof_bytes: invalid PLONK proof size");
    }
    return plonk_bytes - omitted_opening_bytes +
           proof.forest_fetch.bytes.size() +
           proof.cross_session_opening_batch.bytes.size();
}
