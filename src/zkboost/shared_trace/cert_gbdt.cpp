#include "zkboost/shared_trace/cert_gbdt.hpp"

#include "plonk/prover.hpp"
#include "proof_runtime.hpp"
#include "terrae_env.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require_condition(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(std::string("zkboost_shared_cert_gbdt: ") + message);
    }
}

bool zkboost_verify_profile_enabled()
{
    const char* flag = terrae_getenv("TERRAE_PROFILE_ZKBOOST_VERIFY");
    return flag != nullptr && flag[0] != '\0' && std::string(flag) != "0" && std::string(flag) != "false";
}

template <class Fn>
bool timed_verify_step(const char* label, Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    const bool ok = fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cerr << "zkboost_verify_profile " << label << " " << elapsed.count()
              << " ok=" << (ok ? 1 : 0) << "\n";
    return ok;
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

bool add_cross_session_verifier_workflow(
    const zkboostSharedCertGBDTVerifierBundle& verifier_bundle,
    const zkboostSharedCertGBDTProof& proof,
    ProofRuntime& runtime,
    ProofProgram& program)
{
    bool ok = add_zkboost_shared_weighted_logup_verifier_to_program(
        program, verifier_bundle.histogram_logup);
    ok = add_deferred_plonk_verifier_openings(
             runtime.kzg(), verifier_bundle.forest, proof.forest) && ok;
    ok = add_deferred_plonk_verifier_openings(
             runtime.kzg(),
             verifier_bundle.forest_well_formedness,
             proof.forest_well_formedness) && ok;
    ok = add_deferred_plonk_verifier_openings(
             runtime.kzg(), verifier_bundle.boosting_state, proof.boosting_state) && ok;
    ok = add_deferred_plonk_verifier_openings(
             runtime.kzg(), verifier_bundle.histograms, proof.histograms) && ok;
    ok = add_deferred_plonk_verifier_openings(
             runtime.kzg(), verifier_bundle.leaf_values, proof.leaf_values) && ok;
    ok = add_deferred_plonk_verifier_openings(
             runtime.kzg(), verifier_bundle.splits, proof.splits) && ok;
    return ok;
}

zkboostSharedCertGBDTSummary summarize_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const zkboostSharedSplitView& splits)
{
    zkboostSharedCertGBDTSummary out;
    out.tree_count = training.tree_count;
    out.node_count = training.node_count;
    out.sample_count = training.query_count;
    out.feature_count = training.feature_count;
    out.bin_count = training.bin_count;
    out.checked_paths = inference.query_count;
    out.candidate_count = splits.summary.candidate_count;
    return out;
}

} // namespace

zkboostSharedCertGBDTInstance build_zkboost_shared_cert_gbdt_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(inference.query_count == training.tree_count * training.query_count,
                      "inference query_count must equal tree_count * training query_count");

    const zkboostSharedBoostingStateView boosting =
        build_zkboost_shared_boosting_state_view(training);
    const zkboostSharedHistogramView histograms =
        build_zkboost_shared_histogram_view(training, boosting);
    const zkboostSharedLeafValueView leaf_values =
        build_zkboost_shared_leaf_value_view(training, histograms);
    const zkboostSharedSplitView splits =
        build_zkboost_shared_split_view(training, histograms);

    zkboostSharedCertGBDTInstance out;
    const zkboostSharedValidateForestView forest_view =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedCommittedForest committed_forest =
        build_zkboost_shared_committed_forest(public_parameters, training);

    out.forest = build_zkboost_shared_validate_forest_plonk_instance_from_view(
        forest_view,
        training.bin_count,
        public_parameters);
    out.forest_fetch = build_zkboost_shared_validate_forest_fetch_instance_from_view(
        forest_view,
        committed_forest.ensemble,
        committed_forest.commitments,
        public_parameters);
    out.forest_well_formedness =
        build_zkboost_shared_forest_well_formedness_plonk_instance(
            training,
            committed_forest.commitments,
            public_parameters);
    out.boosting_state = build_zkboost_shared_boosting_state_plonk_instance_from_view(
        training,
        boosting,
        public_parameters);
    out.histograms = build_zkboost_shared_histogram_plonk_instance_from_view(
        training,
        histograms,
        public_parameters);
    out.histogram_logup = build_zkboost_shared_weighted_logup_instance(
        training,
        histograms,
        public_parameters);
    out.leaf_values = build_zkboost_shared_leaf_value_plonk_instance_from_view(
        training,
        leaf_values,
        public_parameters);
    out.splits = build_zkboost_shared_split_plonk_instance_from_view(
        training,
        splits,
        public_parameters);
    out.summary = summarize_instance(training, inference, splits);
    return out;
}

zkboostSharedKzgSrsRequirement required_zkboost_shared_cert_gbdt_srs_requirement(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(inference.query_count == training.tree_count * training.query_count,
                      "inference query_count must equal tree_count * training query_count");

    const zkboostSharedBoostingStateView boosting =
        build_zkboost_shared_boosting_state_view(training);
    const zkboostSharedHistogramView histograms =
        build_zkboost_shared_histogram_view(training, boosting);
    const zkboostSharedLeafValueView leaf_values =
        build_zkboost_shared_leaf_value_view(training, histograms);
    const zkboostSharedSplitView splits =
        build_zkboost_shared_split_view(training, histograms);
    const zkboostSharedValidateForestView forest_view =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(training);

    zkboostSharedKzgSrsRequirement out =
        required_zkboost_shared_forest_commitment_srs_requirement(ensemble);
    out = max_zkboost_shared_kzg_srs_requirement(
        out,
        zkboostSharedKzgSrsRequirement{
            required_zkboost_shared_validate_forest_plonk_srs_size_from_view(
                forest_view,
                training.bin_count),
            2,
            0,
            2,
        });
    const zkboostSharedKzgSrsRequirement fetch_requirement =
        required_zkboost_shared_validate_forest_fetch_srs_requirement_from_view(
            forest_view, ensemble);
    out = max_zkboost_shared_kzg_srs_requirement(out, fetch_requirement);
    out = max_zkboost_shared_kzg_srs_requirement(
        out,
        zkboostSharedKzgSrsRequirement{
            required_zkboost_shared_forest_well_formedness_plonk_srs_size(training),
            2,
            0,
            2,
        });
    out = max_zkboost_shared_kzg_srs_requirement(
        out,
        zkboostSharedKzgSrsRequirement{
            required_zkboost_shared_boosting_state_plonk_srs_size_from_view(
                training,
                boosting),
            2,
            0,
            2,
        });
    out = max_zkboost_shared_kzg_srs_requirement(
        out,
        zkboostSharedKzgSrsRequirement{
            required_zkboost_shared_histogram_plonk_srs_size_from_view(
                training,
                histograms),
            2,
            0,
            2,
        });
    out = max_zkboost_shared_kzg_srs_requirement(
        out,
        zkboostSharedKzgSrsRequirement{
            required_zkboost_shared_weighted_logup_srs_size_from_view(
                training,
                histograms),
            2,
            0,
            2,
        });
    out = max_zkboost_shared_kzg_srs_requirement(
        out,
        zkboostSharedKzgSrsRequirement{
            required_zkboost_shared_leaf_value_plonk_srs_size_from_view(
                training,
                leaf_values),
            2,
            0,
            2,
        });
    out = max_zkboost_shared_kzg_srs_requirement(
        out,
        zkboostSharedKzgSrsRequirement{
            required_zkboost_shared_split_plonk_srs_size_from_view(
                training,
                splits),
            2,
            0,
            2,
        });
    // Forest fetches use an independent prepared batch.  The circuit batch
    // contains twelve PLONK points and four weighted-LogUp points.
    out.opening_deg2_plus_1_floor = std::max(
        out.opening_deg2_plus_1_floor,
        size_t(17));
    const size_t degree_shift_requirement =
        out.min_degree_bound_plus_1 == 0
            ? size_t(1)
            : out.deg1_plus_1 - out.min_degree_bound_plus_1 + 1;
    out.deg2_plus_1 = std::max(
        out.deg2_plus_1,
        std::max(out.opening_deg2_plus_1_floor, degree_shift_requirement));
    return out;
}

size_t required_zkboost_shared_cert_gbdt_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    return required_zkboost_shared_cert_gbdt_srs_requirement(training, inference).deg1_plus_1;
}

zkboostSharedCertGBDTProof prove_zkboost_shared_cert_gbdt(
    const zkboostSharedCertGBDTInstance& instance)
{
    const plonkDeferredOpeningProof forest =
        prove_plonk_deferred_opening(instance.forest.setup.proving_key,
                                     instance.forest.witness,
                                     instance.forest.public_inputs);
    const plonkDeferredOpeningProof forest_well_formedness =
        prove_plonk_deferred_opening(
            instance.forest_well_formedness.setup.proving_key,
            instance.forest_well_formedness.witness,
            instance.forest_well_formedness.public_inputs);
    const plonkDeferredOpeningProof boosting_state =
        prove_plonk_deferred_opening(instance.boosting_state.setup.proving_key,
                                     instance.boosting_state.witness,
                                     instance.boosting_state.public_inputs);
    const plonkDeferredOpeningProof histograms =
        prove_plonk_deferred_opening(instance.histograms.setup.proving_key,
                                     instance.histograms.witness,
                                     instance.histograms.public_inputs);
    const plonkDeferredOpeningProof leaf_values =
        prove_plonk_deferred_opening(instance.leaf_values.setup.proving_key,
                                     instance.leaf_values.witness,
                                     instance.leaf_values.public_inputs);
    const plonkDeferredOpeningProof splits =
        prove_plonk_deferred_opening(instance.splits.setup.proving_key,
                                     instance.splits.witness,
                                     instance.splits.public_inputs);

    ProofRuntime runtime =
        ProofRuntime::prover_borrowed(instance.histogram_logup.public_parameters);
    ProofProgram program(runtime);
    bool ok = add_zkboost_shared_weighted_logup_to_program(
        program, instance.histogram_logup);
    ok = add_deferred_plonk_prover_openings(
             runtime.kzg(), instance.forest.setup, instance.forest.public_inputs, forest) && ok;
    ok = add_deferred_plonk_prover_openings(runtime.kzg(),
                                            instance.forest_well_formedness.setup,
                                            instance.forest_well_formedness.public_inputs,
                                            forest_well_formedness) && ok;
    ok = add_deferred_plonk_prover_openings(runtime.kzg(),
                                            instance.boosting_state.setup,
                                            instance.boosting_state.public_inputs,
                                            boosting_state) && ok;
    ok = add_deferred_plonk_prover_openings(runtime.kzg(),
                                            instance.histograms.setup,
                                            instance.histograms.public_inputs,
                                            histograms) && ok;
    ok = add_deferred_plonk_prover_openings(runtime.kzg(),
                                            instance.leaf_values.setup,
                                            instance.leaf_values.public_inputs,
                                            leaf_values) && ok;
    ok = add_deferred_plonk_prover_openings(
             runtime.kzg(), instance.splits.setup, instance.splits.public_inputs, splits) && ok;
    if (!ok || !program.finalize()) {
        throw std::runtime_error(
            "prove_zkboost_shared_cert_gbdt: cross-session opening batch failed");
    }

    zkboostSharedCertGBDTProof out;
    out.forest = forest.proof;
    out.forest_fetch = prove_zkboost_shared_validate_forest_fetch(instance.forest_fetch);
    out.forest_well_formedness = forest_well_formedness.proof;
    out.boosting_state = boosting_state.proof;
    out.histograms = histograms.proof;
    out.leaf_values = leaf_values.proof;
    out.splits = splits.proof;
    out.cross_session_opening_batch = runtime.take_proof();
    return out;
}

bool zkboostSharedCertGBDTLinkChecks::all_ok() const
{
    return forest_to_fetch &&
           forest_well_formedness_commitment_binding &&
           forest_well_formedness_to_fetch &&
           forest_well_formedness_to_splits &&
           forest_well_formedness_to_leaf_values &&
           forest_to_boosting_state &&
           forest_to_histograms &&
           boosting_state_to_histograms &&
           histograms_to_weighted_logup &&
           histograms_to_leaf_values &&
           forest_to_leaf_values &&
           histograms_to_splits &&
           forest_to_splits;
}

zkboostSharedCertGBDTLinkChecks check_zkboost_shared_cert_gbdt_links(
    const zkboostSharedCertGBDTInstance& instance)
{
    zkboostSharedCertGBDTLinkChecks out;
    out.forest_to_fetch = link_zkboost_shared_validate_forest_to_fetch(
        instance.forest, instance.forest_fetch);
    out.forest_well_formedness_commitment_binding =
        verify_zkboost_shared_forest_well_formedness_commitment_binding(
            instance.forest_well_formedness);
    out.forest_well_formedness_to_fetch =
        link_zkboost_shared_forest_well_formedness_to_fetch(
            instance.forest_well_formedness, instance.forest_fetch);
    out.forest_well_formedness_to_splits =
        link_zkboost_shared_forest_well_formedness_to_splits(
            instance.forest_well_formedness, instance.splits);
    out.forest_well_formedness_to_leaf_values =
        link_zkboost_shared_forest_well_formedness_to_leaf_values(
            instance.forest_well_formedness, instance.leaf_values);
    out.forest_to_boosting_state = link_zkboost_shared_validate_forest_to_boosting_state(
        instance.forest, instance.boosting_state);
    out.forest_to_histograms = link_zkboost_shared_validate_forest_to_histograms(
        instance.forest, instance.histograms);
    out.boosting_state_to_histograms = link_zkboost_shared_boosting_state_to_histograms(
        instance.boosting_state, instance.histograms);
    out.histograms_to_weighted_logup = link_zkboost_shared_histograms_to_weighted_logup(
        instance.histograms, instance.histogram_logup);
    out.histograms_to_leaf_values = link_zkboost_shared_histograms_to_leaf_values(
        instance.histograms, instance.leaf_values);
    out.forest_to_leaf_values = link_zkboost_shared_validate_forest_to_leaf_values(
        instance.forest, instance.leaf_values);
    out.histograms_to_splits = link_zkboost_shared_histograms_to_splits(
        instance.histograms, instance.splits);
    out.forest_to_splits = link_zkboost_shared_validate_forest_to_splits(
        instance.forest, instance.splits);
    return out;
}

bool verify_zkboost_shared_cert_gbdt_links(
    const zkboostSharedCertGBDTInstance& instance)
{
    return check_zkboost_shared_cert_gbdt_links(instance).all_ok();
}

bool verify_zkboost_shared_cert_gbdt(
    const zkboostSharedCertGBDTInstance& instance,
    const zkboostSharedCertGBDTProof& proof)
{
    return verify_zkboost_shared_cert_gbdt_preprocessed(
        prepare_zkboost_shared_cert_gbdt_verifier_bundle(instance),
        proof);
}

zkboostSharedCertGBDTVerifierBundle prepare_zkboost_shared_cert_gbdt_verifier_bundle(
    const zkboostSharedCertGBDTInstance& instance)
{
    zkboostSharedCertGBDTVerifierBundle out;
    out.forest = preprocess_plonk_verifier_input(
        instance.forest.setup.verification_key,
        instance.forest.public_inputs);
    out.forest_well_formedness = preprocess_plonk_verifier_input(
        instance.forest_well_formedness.setup.verification_key,
        instance.forest_well_formedness.public_inputs);
    out.boosting_state = preprocess_plonk_verifier_input(
        instance.boosting_state.setup.verification_key,
        instance.boosting_state.public_inputs);
    out.histograms = preprocess_plonk_verifier_input(
        instance.histograms.setup.verification_key,
        instance.histograms.public_inputs);
    out.histogram_logup = prepare_zkboost_shared_weighted_logup_verifier_bundle(
        instance.histogram_logup);
    out.leaf_values = preprocess_plonk_verifier_input(
        instance.leaf_values.setup.verification_key,
        instance.leaf_values.public_inputs);
    out.splits = preprocess_plonk_verifier_input(
        instance.splits.setup.verification_key,
        instance.splits.public_inputs);
    out.forest_fetch = prepare_zkboost_shared_validate_forest_fetch_verifier_bundle(
        instance.forest_fetch);
    out.forest_fetch_instance = &instance.forest_fetch;
    out.public_parameters = &instance.histogram_logup.public_parameters;
    out.forest_committed_table_ok = verify_zkboost_shared_validate_forest_committed_table(
        instance.forest.setup.verification_key,
        instance.forest.committed_table);
    out.links_ok = verify_zkboost_shared_cert_gbdt_links(instance);
    return out;
}

zkboostSharedCertGBDTVerifierPreprocessing
prepare_zkboost_shared_cert_gbdt_verifier_preprocessing(
    const zkboostSharedCertGBDTVerifierBundle& verifier_bundle,
    const zkboostSharedCertGBDTProof& proof)
{
    if (!verifier_bundle.forest_committed_table_ok ||
        !verifier_bundle.links_ok ||
        verifier_bundle.public_parameters == nullptr ||
        verifier_bundle.forest_fetch_instance == nullptr) {
        throw std::runtime_error(
            "prepare_zkboost_shared_cert_gbdt_verifier_preprocessing: invalid bundle");
    }
    zkboostSharedCertGBDTVerifierPreprocessing out;
    out.runtime = ProofRuntime::verifier_ptr(
        *verifier_bundle.public_parameters, proof.cross_session_opening_batch);
    out.program = std::make_unique<ProofProgram>(*out.runtime);
    out.forest_fetch = &verifier_bundle.forest_fetch;
    out.forest_fetch_proof = &proof.forest_fetch;
    if (!add_cross_session_verifier_workflow(
            verifier_bundle, proof, *out.runtime, *out.program)) {
        throw std::runtime_error(
            "prepare_zkboost_shared_cert_gbdt_verifier_preprocessing: invalid claims");
    }
    if (!out.program->prepare_verifier_preprocessing()) {
        throw std::runtime_error(
            "prepare_zkboost_shared_cert_gbdt_verifier_preprocessing: preprocessing failed");
    }
    if (!out.program->complete_verifier_opening_preprocessing()) {
        throw std::runtime_error(
            "prepare_zkboost_shared_cert_gbdt_verifier_preprocessing: opening preprocessing failed");
    }
    return out;
}

bool verify_zkboost_shared_cert_gbdt_preprocessed(
    zkboostSharedCertGBDTVerifierPreprocessing& preprocessing)
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

bool verify_zkboost_shared_cert_gbdt_preprocessed(
    const zkboostSharedCertGBDTVerifierBundle& verifier_bundle,
    const zkboostSharedCertGBDTProof& proof)
{
    const auto verify_cross_session_batch = [&]() {
        try {
            zkboostSharedCertGBDTVerifierPreprocessing preprocessing =
                prepare_zkboost_shared_cert_gbdt_verifier_preprocessing(
                    verifier_bundle, proof);
            return verify_zkboost_shared_cert_gbdt_preprocessed(preprocessing);
        } catch (const std::exception&) {
            return false;
        }
    };
    if (zkboost_verify_profile_enabled()) {
        return timed_verify_step(
            "cross_session_opening_batch", verify_cross_session_batch);
    }
    return verify_cross_session_batch();
}

size_t serialized_zkboost_shared_cert_gbdt_proof_bytes(
    const zkboostSharedCertGBDTProof& proof)
{
    const size_t omitted_opening_bytes = 2 * G1::getSerializedByteSize();
    const auto deferred_plonk_bytes = [&](const plonkProof& plonk_proof) {
        return serialize_plonk_proof(plonk_proof).bytes.size() - omitted_opening_bytes;
    };
    return deferred_plonk_bytes(proof.forest) +
           deferred_plonk_bytes(proof.forest_well_formedness) +
           deferred_plonk_bytes(proof.boosting_state) +
           deferred_plonk_bytes(proof.histograms) +
           deferred_plonk_bytes(proof.leaf_values) +
           deferred_plonk_bytes(proof.splits) +
           proof.forest_fetch.bytes.size() +
           proof.cross_session_opening_batch.bytes.size();
}
