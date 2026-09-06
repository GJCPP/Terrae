#pragma once

#include "plonk/plonk.hpp"
#include "zkboost/shared_trace/validate_histograms_plonk.hpp"
#include "zkboost/shared_trace/validate_splits.hpp"

#include <vector>

class zkboostSharedSplitPlonkInstance {
public:
    plonkSetup setup;
    plonkWitness witness;
    std::vector<Fr> public_inputs;

    zkboostSharedSplitSummary summary;
};

std::vector<Fr> encode_zkboost_shared_split_public_inputs(
    const zkboostSharedSplitView& view);

zkboostSharedSplitPlonkInstance build_zkboost_shared_split_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const kzgPublicParameters& public_parameters);

zkboostSharedSplitPlonkInstance build_zkboost_shared_split_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms,
    const kzgPublicParameters& public_parameters);

zkboostSharedSplitPlonkInstance build_zkboost_shared_split_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedSplitView& view,
    const kzgPublicParameters& public_parameters);

size_t required_zkboost_shared_split_plonk_srs_size(
    const forestTrainingSemanticTrace& training);

size_t required_zkboost_shared_split_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms);

size_t required_zkboost_shared_split_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedSplitView& view);

plonkProof prove_zkboost_shared_split_plonk(
    const zkboostSharedSplitPlonkInstance& instance);

bool verify_zkboost_shared_split_plonk(
    const zkboostSharedSplitPlonkInstance& instance,
    const plonkProof& proof);

bool verify_zkboost_shared_split_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof);

bool link_zkboost_shared_histograms_to_splits(
    const zkboostSharedHistogramPlonkInstance& histogram_instance,
    const zkboostSharedSplitPlonkInstance& split_instance);

bool link_zkboost_shared_validate_forest_to_splits(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedSplitPlonkInstance& split_instance);
