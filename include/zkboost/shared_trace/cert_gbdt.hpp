#pragma once

#include "proof_runtime.hpp"

#include "zkboost/shared_trace/forest_well_formedness_plonk.hpp"
#include "zkboost/shared_trace/validate_forest_fetch.hpp"
#include "zkboost/shared_trace/validate_boosting_state_plonk.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"
#include "zkboost/shared_trace/validate_histograms_plonk.hpp"
#include "zkboost/shared_trace/validate_leaf_values_plonk.hpp"
#include "zkboost/shared_trace/validate_splits_plonk.hpp"
#include "zkboost/shared_trace/weighted_logup.hpp"

class zkboostSharedCertGBDTSummary {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t sample_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t checked_paths = 0;
    size_t candidate_count = 0;
};

class zkboostSharedCertGBDTInstance {
public:
    zkboostSharedValidateForestPlonkInstance forest;
    zkboostSharedValidateForestFetchInstance forest_fetch;
    zkboostSharedForestWellFormednessPlonkInstance forest_well_formedness;
    zkboostSharedBoostingStatePlonkInstance boosting_state;
    zkboostSharedHistogramPlonkInstance histograms;
    zkboostSharedWeightedLogupInstance histogram_logup;
    zkboostSharedLeafValuePlonkInstance leaf_values;
    zkboostSharedSplitPlonkInstance splits;
    zkboostSharedCertGBDTSummary summary;
};

class zkboostSharedCertGBDTProof {
public:
    plonkProof forest;
    SerializedProof forest_fetch;
    plonkProof forest_well_formedness;
    plonkProof boosting_state;
    plonkProof histograms;
    SerializedProof histogram_logup;
    plonkProof leaf_values;
    plonkProof splits;
    SerializedProof cross_session_opening_batch;
};

class zkboostSharedCertGBDTLinkChecks {
public:
    bool forest_to_fetch = false;
    bool forest_well_formedness_commitment_binding = false;
    bool forest_well_formedness_to_fetch = false;
    bool forest_well_formedness_to_splits = false;
    bool forest_well_formedness_to_leaf_values = false;
    bool forest_to_boosting_state = false;
    bool forest_to_histograms = false;
    bool boosting_state_to_histograms = false;
    bool histograms_to_weighted_logup = false;
    bool histograms_to_leaf_values = false;
    bool forest_to_leaf_values = false;
    bool histograms_to_splits = false;
    bool forest_to_splits = false;

    bool all_ok() const;
};

class zkboostSharedCertGBDTVerifierBundle {
public:
    plonkPreprocessedVerifierInput forest;
    plonkPreprocessedVerifierInput forest_well_formedness;
    plonkPreprocessedVerifierInput boosting_state;
    plonkPreprocessedVerifierInput histograms;
    zkboostSharedWeightedLogupVerifierBundle histogram_logup;
    plonkPreprocessedVerifierInput leaf_values;
    plonkPreprocessedVerifierInput splits;
    zkboostSharedValidateForestFetchVerifierBundle forest_fetch;
    const zkboostSharedValidateForestFetchInstance* forest_fetch_instance = nullptr;
    const kzgPublicParameters* public_parameters = nullptr;
    bool forest_committed_table_ok = false;
    bool links_ok = false;
};

class zkboostSharedCertGBDTVerifierPreprocessing {
public:
    std::unique_ptr<ProofRuntime> runtime;
    std::unique_ptr<ProofProgram> program;
    const zkboostSharedValidateForestFetchVerifierBundle* forest_fetch = nullptr;
    const SerializedProof* forest_fetch_proof = nullptr;
};

zkboostSharedCertGBDTInstance build_zkboost_shared_cert_gbdt_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters);

zkboostSharedKzgSrsRequirement required_zkboost_shared_cert_gbdt_srs_requirement(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);

size_t required_zkboost_shared_cert_gbdt_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);

zkboostSharedCertGBDTProof prove_zkboost_shared_cert_gbdt(
    const zkboostSharedCertGBDTInstance& instance);

zkboostSharedCertGBDTLinkChecks check_zkboost_shared_cert_gbdt_links(
    const zkboostSharedCertGBDTInstance& instance);

bool verify_zkboost_shared_cert_gbdt_links(
    const zkboostSharedCertGBDTInstance& instance);

bool verify_zkboost_shared_cert_gbdt(
    const zkboostSharedCertGBDTInstance& instance,
    const zkboostSharedCertGBDTProof& proof);

zkboostSharedCertGBDTVerifierBundle prepare_zkboost_shared_cert_gbdt_verifier_bundle(
    const zkboostSharedCertGBDTInstance& instance);

zkboostSharedCertGBDTVerifierPreprocessing
prepare_zkboost_shared_cert_gbdt_verifier_preprocessing(
    const zkboostSharedCertGBDTVerifierBundle& verifier_bundle,
    const zkboostSharedCertGBDTProof& proof);

bool verify_zkboost_shared_cert_gbdt_preprocessed(
    zkboostSharedCertGBDTVerifierPreprocessing& preprocessing);

bool verify_zkboost_shared_cert_gbdt_preprocessed(
    const zkboostSharedCertGBDTVerifierBundle& verifier_bundle,
    const zkboostSharedCertGBDTProof& proof);

size_t serialized_zkboost_shared_cert_gbdt_proof_bytes(
    const zkboostSharedCertGBDTProof& proof);
