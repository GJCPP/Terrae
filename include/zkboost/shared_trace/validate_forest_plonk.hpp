#pragma once

#include "plonk/plonk.hpp"
#include "zkboost/compiler/generic_circuit.hpp"
#include "zkboost/shared_trace/validate_forest.hpp"

#include <cstddef>
#include <string>
#include <vector>

enum class zkboostSharedValidateForestPublicInputMode {
    OnlinePublicInputs,
    PreprocessedConstants,
};

class zkboostSharedValidateForestCommittedTable {
public:
    size_t comparison_row_count = 0;
    size_t total_lookup_row_count = 0;
    size_t table_domain_size = 0;

    plonkPolynomialCommitment left_commitment;
    plonkPolynomialCommitment right_commitment;
    plonkPolynomialCommitment output_commitment;
};

class zkboostSharedValidateForestPlonkInstance {
public:
    plonkSetup setup;
    plonkWitness witness;
    std::vector<Fr> public_inputs;
    std::string linked_input_digest;
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs;

    zkboostSharedValidateForestSummary summary;
    zkboostSharedValidateForestCommittedTable committed_table;
};

std::vector<Fr> encode_zkboost_shared_validate_forest_public_inputs(
    const zkboostSharedValidateForestView& view);

std::string digest_zkboost_shared_validate_forest_public_inputs(
    const zkboostSharedValidateForestView& view);

zkboostSharedValidateForestPlonkInstance build_zkboost_shared_validate_forest_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostSharedValidateForestPlonkInstance build_zkboost_shared_validate_forest_plonk_instance_from_view(
    const zkboostSharedValidateForestView& view,
    size_t bin_count,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostGenericCircuit build_zkboost_shared_validate_forest_generic_circuit_from_view(
    const zkboostSharedValidateForestView& view,
    size_t bin_count,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

zkboostSharedValidateForestCommittedTable
build_zkboost_shared_validate_forest_committed_table_metadata(
    size_t bin_count,
    size_t total_lookup_row_count,
    const plonkVerificationKey& verification_key);

size_t required_zkboost_shared_validate_forest_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

size_t required_zkboost_shared_validate_forest_plonk_srs_size_from_view(
    const zkboostSharedValidateForestView& view,
    size_t bin_count,
    zkboostSharedValidateForestPublicInputMode public_input_mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);

bool verify_zkboost_shared_validate_forest_committed_table(
    const plonkVerificationKey& verification_key,
    const zkboostSharedValidateForestCommittedTable& committed_table);

plonkProof prove_zkboost_shared_validate_forest_plonk(
    const zkboostSharedValidateForestPlonkInstance& instance);

bool verify_zkboost_shared_validate_forest_plonk(
    const zkboostSharedValidateForestPlonkInstance& instance,
    const plonkProof& proof);

bool verify_zkboost_shared_validate_forest_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof);
