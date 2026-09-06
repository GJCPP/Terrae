#pragma once

#include "plonk/plonk.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"
#include "zkboost/shared_trace/validate_histograms_plonk.hpp"
#include "zkboost/shared_trace/validate_leaf_values.hpp"

#include <vector>

class zkboostSharedLeafValuePlonkInstance {
public:
    plonkSetup setup;
    plonkWitness witness;
    std::vector<Fr> public_inputs;

    zkboostSharedLeafValueSummary summary;
};

std::vector<Fr> encode_zkboost_shared_leaf_value_public_inputs(
    const zkboostSharedLeafValueView& view);

zkboostSharedLeafValuePlonkInstance build_zkboost_shared_leaf_value_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const kzgPublicParameters& public_parameters);

zkboostSharedLeafValuePlonkInstance build_zkboost_shared_leaf_value_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms,
    const kzgPublicParameters& public_parameters);

zkboostSharedLeafValuePlonkInstance build_zkboost_shared_leaf_value_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedLeafValueView& view,
    const kzgPublicParameters& public_parameters);

size_t required_zkboost_shared_leaf_value_plonk_srs_size(
    const forestTrainingSemanticTrace& training);

size_t required_zkboost_shared_leaf_value_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms);

size_t required_zkboost_shared_leaf_value_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedLeafValueView& view);

plonkProof prove_zkboost_shared_leaf_value_plonk(
    const zkboostSharedLeafValuePlonkInstance& instance);

bool verify_zkboost_shared_leaf_value_plonk(
    const zkboostSharedLeafValuePlonkInstance& instance,
    const plonkProof& proof);

bool verify_zkboost_shared_leaf_value_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof);

bool link_zkboost_shared_histograms_to_leaf_values(
    const zkboostSharedHistogramPlonkInstance& histogram_instance,
    const zkboostSharedLeafValuePlonkInstance& leaf_value_instance);

bool link_zkboost_shared_validate_forest_to_leaf_values(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedLeafValuePlonkInstance& leaf_value_instance);
