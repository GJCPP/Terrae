#pragma once

#include "plonk/plonk.hpp"
#include "zkboost/shared_trace/validate_boosting_state.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"

#include <vector>

class zkboostSharedBoostingStatePlonkInstance {
public:
    plonkSetup setup;
    plonkWitness witness;
    std::vector<Fr> public_inputs;

    zkboostSharedBoostingStateSummary summary;
};

std::vector<Fr> encode_zkboost_shared_boosting_state_public_inputs(
    const zkboostSharedBoostingStateView& view);

zkboostSharedBoostingStatePlonkInstance build_zkboost_shared_boosting_state_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const kzgPublicParameters& public_parameters);

zkboostSharedBoostingStatePlonkInstance build_zkboost_shared_boosting_state_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& view,
    const kzgPublicParameters& public_parameters);

size_t required_zkboost_shared_boosting_state_plonk_srs_size(
    const forestTrainingSemanticTrace& training);

size_t required_zkboost_shared_boosting_state_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& view);

plonkProof prove_zkboost_shared_boosting_state_plonk(
    const zkboostSharedBoostingStatePlonkInstance& instance);

bool verify_zkboost_shared_boosting_state_plonk(
    const zkboostSharedBoostingStatePlonkInstance& instance,
    const plonkProof& proof);

bool verify_zkboost_shared_boosting_state_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof);

bool link_zkboost_shared_validate_forest_to_boosting_state(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedBoostingStatePlonkInstance& boosting_instance);
