#pragma once

#include "commitments/proof_session.hpp"
#include "zkboost/shared_trace/forest_commitments.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"

#include <string>
#include <vector>

class zkboostSharedValidateForestFetchSummary {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t sample_count = 0;
    size_t max_depth = 0;
    size_t checked_paths = 0;
    size_t split_domain_size = 0;
    size_t leaf_domain_size = 0;
    size_t split_field_read_count = 0;
    size_t path_read_count = 0;
    size_t leaf_value_read_count = 0;
};

class zkboostSharedValidateForestFetchInstance {
public:
    zkboostSharedValidateForestFetchSummary summary;
    zkboostSharedValidateForestView view;
    zkboostSharedTreeEnsemble ensemble;
    zkboostSharedForestCommitments commitments;
    std::string linked_input_digest;
    kzgPublicParameters public_parameters;
};

class zkboostSharedValidateForestFetchVerifierOpeningGroup {
public:
    std::vector<size_t> set_indices;
    Group2 auxiliary_commitment;
};

class zkboostSharedValidateForestFetchVerifierBundle {
public:
    ProofTranscript transcript_prefix;
    GroupT degree_combined_commitment;
    Group2 common_vanishing_commitment;
    std::vector<Group1> opening_set_terms;
    std::vector<zkboostSharedValidateForestFetchVerifierOpeningGroup> opening_groups;
};

zkboostSharedValidateForestFetchInstance build_zkboost_shared_validate_forest_fetch_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters);

zkboostSharedValidateForestFetchInstance build_zkboost_shared_validate_forest_fetch_instance_from_view(
    const zkboostSharedValidateForestView& view,
    const zkboostSharedTreeEnsemble& ensemble,
    const kzgPublicParameters& public_parameters);

zkboostSharedValidateForestFetchInstance build_zkboost_shared_validate_forest_fetch_instance_from_view(
    const zkboostSharedValidateForestView& view,
    const zkboostSharedTreeEnsemble& ensemble,
    const zkboostSharedForestCommitments& commitments,
    const kzgPublicParameters& public_parameters);

size_t required_zkboost_shared_validate_forest_fetch_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);

size_t required_zkboost_shared_validate_forest_fetch_srs_size(
    const forestTrainingSemanticTrace& training);

size_t required_zkboost_shared_validate_forest_fetch_srs_size_from_ensemble(
    const zkboostSharedTreeEnsemble& ensemble);

zkboostSharedKzgSrsRequirement required_zkboost_shared_validate_forest_fetch_srs_requirement(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference);

zkboostSharedKzgSrsRequirement required_zkboost_shared_validate_forest_fetch_srs_requirement_from_view(
    const zkboostSharedValidateForestView& view,
    const zkboostSharedTreeEnsemble& ensemble);

zkboostSharedKzgSrsRequirement required_zkboost_shared_validate_forest_fetch_srs_requirement_from_ensemble(
    const zkboostSharedTreeEnsemble& ensemble);

bool add_zkboost_shared_validate_forest_fetch_to_kzg(
    kzgProtocol& kzg,
    const zkboostSharedValidateForestFetchInstance& instance);

SerializedProof prove_zkboost_shared_validate_forest_fetch(
    const zkboostSharedValidateForestFetchInstance& instance);

bool verify_zkboost_shared_validate_forest_fetch(
    const zkboostSharedValidateForestFetchInstance& instance,
    const SerializedProof& proof);

zkboostSharedValidateForestFetchVerifierBundle prepare_zkboost_shared_validate_forest_fetch_verifier_bundle(
    const zkboostSharedValidateForestFetchInstance& instance);

bool verify_zkboost_shared_validate_forest_fetch_preprocessed(
    const zkboostSharedValidateForestFetchVerifierBundle& verifier_bundle,
    const SerializedProof& proof);

bool link_zkboost_shared_validate_forest_to_fetch(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedValidateForestFetchInstance& fetch_instance);
