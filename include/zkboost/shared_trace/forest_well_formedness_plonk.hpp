#pragma once

#include "plonk/plonk.hpp"
#include "zkboost/shared_trace/forest_commitments.hpp"
#include "zkboost/shared_trace/validate_forest_fetch.hpp"
#include "zkboost/shared_trace/validate_leaf_values_plonk.hpp"
#include "zkboost/shared_trace/validate_splits_plonk.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

class zkboostSharedForestWellFormednessSummary {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t max_depth = 0;
    size_t internal_slot_count = 0;
    size_t leaf_slot_count = 0;
    size_t dummy_internal_slot_count = 0;
};

class zkboostSharedForestWellFormednessView {
public:
    zkboostSharedForestWellFormednessSummary summary;
    zkboostSharedTreeEnsemble ensemble;

    std::vector<std::int64_t> dummy_masks;
    std::vector<std::int64_t> internal_source_nodes;
    std::vector<std::int64_t> source_split_masks;
    std::vector<std::int64_t> source_features;
    std::vector<std::int64_t> source_thresholds;
    std::vector<std::int64_t> source_left_children;
    std::vector<std::int64_t> source_right_children;
    std::vector<std::int64_t> leaf_source_nodes;
    std::vector<std::int64_t> leaf_source_split_masks;
    std::vector<std::int64_t> source_leaf_values;
    std::vector<std::int64_t> root_source_nodes;
};

class zkboostSharedForestWellFormednessPlonkInstance {
public:
    plonkSetup setup;
    plonkWitness witness;
    std::vector<Fr> public_inputs;

    zkboostSharedForestWellFormednessSummary summary;
    zkboostSharedForestWellFormednessView view;
    zkboostSharedForestCommitments commitments;
};

zkboostSharedForestWellFormednessView build_zkboost_shared_forest_well_formedness_view(
    const forestTrainingSemanticTrace& training);

zkboostSharedForestWellFormednessPlonkInstance
build_zkboost_shared_forest_well_formedness_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedForestCommitments& commitments,
    const kzgPublicParameters& public_parameters);

size_t required_zkboost_shared_forest_well_formedness_plonk_srs_size(
    const forestTrainingSemanticTrace& training);

plonkProof prove_zkboost_shared_forest_well_formedness_plonk(
    const zkboostSharedForestWellFormednessPlonkInstance& instance);

bool verify_zkboost_shared_forest_well_formedness_plonk(
    const zkboostSharedForestWellFormednessPlonkInstance& instance,
    const plonkProof& proof);

bool verify_zkboost_shared_forest_well_formedness_commitment_binding(
    const zkboostSharedForestWellFormednessPlonkInstance& instance);

bool link_zkboost_shared_forest_well_formedness_to_fetch(
    const zkboostSharedForestWellFormednessPlonkInstance& well_formedness,
    const zkboostSharedValidateForestFetchInstance& fetch);

bool link_zkboost_shared_forest_well_formedness_to_splits(
    const zkboostSharedForestWellFormednessPlonkInstance& well_formedness,
    const zkboostSharedSplitPlonkInstance& splits);

bool link_zkboost_shared_forest_well_formedness_to_leaf_values(
    const zkboostSharedForestWellFormednessPlonkInstance& well_formedness,
    const zkboostSharedLeafValuePlonkInstance& leaf_values);
