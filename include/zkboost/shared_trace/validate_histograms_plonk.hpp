#pragma once

#include "plonk/plonk.hpp"
#include "zkboost/shared_trace/validate_boosting_state_plonk.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"
#include "zkboost/shared_trace/validate_histograms.hpp"

#include <vector>

class zkboostSharedHistogramPlonkInstance {
public:
    plonkSetup setup;
    plonkWitness witness;
    std::vector<Fr> public_inputs;

    zkboostSharedHistogramSummary summary;
};

std::vector<Fr> encode_zkboost_shared_histogram_public_inputs(
    const zkboostSharedHistogramView& view,
    const forestTrainingSemanticTrace& training);

zkboostSharedHistogramPlonkInstance build_zkboost_shared_histogram_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const kzgPublicParameters& public_parameters);

zkboostSharedHistogramPlonkInstance build_zkboost_shared_histogram_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& boosting,
    const kzgPublicParameters& public_parameters);

zkboostSharedHistogramPlonkInstance build_zkboost_shared_histogram_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& view,
    const kzgPublicParameters& public_parameters);

size_t required_zkboost_shared_histogram_plonk_srs_size(
    const forestTrainingSemanticTrace& training);

size_t required_zkboost_shared_histogram_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& boosting);

size_t required_zkboost_shared_histogram_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& view);

plonkProof prove_zkboost_shared_histogram_plonk(
    const zkboostSharedHistogramPlonkInstance& instance);

bool verify_zkboost_shared_histogram_plonk(
    const zkboostSharedHistogramPlonkInstance& instance,
    const plonkProof& proof);

bool verify_zkboost_shared_histogram_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof);

bool link_zkboost_shared_validate_forest_to_histograms(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedHistogramPlonkInstance& histogram_instance);

bool link_zkboost_shared_boosting_state_to_histograms(
    const zkboostSharedBoostingStatePlonkInstance& boosting_instance,
    const zkboostSharedHistogramPlonkInstance& histogram_instance);
