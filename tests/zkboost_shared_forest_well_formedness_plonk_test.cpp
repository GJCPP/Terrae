#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/forest_well_formedness_plonk.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedForestWellFormednessPlonkTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

namespace {

forestTrainingSemanticTrace pruned_depth_two_trace()
{
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    trace.max_depth = 3;
    trace.tree_count = 1;
    trace.node_count = 5;
    trace.iteration_count = 1;
    trace.class_count = 1;
    trace.query_count = 4;
    trace.feature_count = 1;
    trace.bin_count = 8;

    trace.root_indices = {0};
    trace.split_node_mask = {1, 0, 1, 0, 0};
    trace.left_child_indices = {1, 1, 3, 3, 4};
    trace.right_child_indices = {2, 1, 4, 3, 4};
    trace.node_tree_ids = {1, 1, 1, 1, 1};
    trace.feature_selector = {1, 0, 1, 0, 0};
    trace.thresholds = {2, 0, 5, 0, 0};
    trace.node_leaf_mask = {0, 1, 0, 1, 1};
    trace.node_leaf_values = {0, 7, 0, -3, 4};
    trace.training_inputs = {0, 1, 6, 7};
    trace.base_labels = {1, 1, 0, 0};
    trace.leaf_indices = {1, 1, 3, 4};
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);
    trace.selected_candidate_indices.assign(trace.node_count, 0);
    trace.selected_candidate_indices[0] = 2;
    trace.selected_candidate_indices[2] = 5;
    return trace;
}

zkboostSharedForestWellFormednessPlonkInstance build_instance(
    const kzgPublicParameters& public_params)
{
    const forestTrainingSemanticTrace trace = pruned_depth_two_trace();
    const zkboostSharedForestCommitments commitments =
        commit_zkboost_shared_forest_from_training_trace(public_params, trace);
    return build_zkboost_shared_forest_well_formedness_plonk_instance(
        trace,
        commitments,
        public_params);
}

} // namespace

TEST_F(ZkboostSharedForestWellFormednessPlonkTest, ProvesPrunedSubtreeEncoding)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedForestWellFormednessPlonkInstance instance = build_instance(public_params);

    EXPECT_EQ(instance.summary.internal_slot_count, 3u);
    EXPECT_EQ(instance.summary.leaf_slot_count, 4u);
    EXPECT_EQ(instance.summary.dummy_internal_slot_count, 1u);
    EXPECT_EQ(instance.view.dummy_masks, std::vector<std::int64_t>({0, 1, 0}));
    EXPECT_EQ(instance.view.ensemble.split_thresholds, std::vector<std::int64_t>({2, 0, 5}));
    EXPECT_EQ(instance.view.leaf_source_nodes, std::vector<std::int64_t>({1, 1, 3, 4}));
    EXPECT_EQ(instance.view.source_leaf_values, std::vector<std::int64_t>({7, 7, -3, 4}));
    EXPECT_TRUE(verify_zkboost_shared_forest_well_formedness_commitment_binding(instance));

    const plonkProof proof = prove_zkboost_shared_forest_well_formedness_plonk(instance);
    EXPECT_TRUE(verify_zkboost_shared_forest_well_formedness_plonk(instance, proof));
}

TEST_F(ZkboostSharedForestWellFormednessPlonkTest, RejectsTamperedDummyThreshold)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedForestWellFormednessPlonkInstance instance = build_instance(public_params);
    const plonkProof proof = prove_zkboost_shared_forest_well_formedness_plonk(instance);

    const size_t threshold_offset = instance.summary.internal_slot_count;
    const size_t dummy_slot = 1;
    instance.public_inputs.at(threshold_offset + dummy_slot) = Fr(6);

    EXPECT_FALSE(verify_zkboost_shared_forest_well_formedness_commitment_binding(instance));
    EXPECT_FALSE(verify_zkboost_shared_forest_well_formedness_plonk(instance, proof));
}

TEST_F(ZkboostSharedForestWellFormednessPlonkTest, RejectsTamperedReplicatedLeaf)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedForestWellFormednessPlonkInstance instance = build_instance(public_params);
    const plonkProof proof = prove_zkboost_shared_forest_well_formedness_plonk(instance);

    const size_t leaf_offset = 2 * instance.summary.internal_slot_count;
    instance.public_inputs.at(leaf_offset + 1) = Fr(8);

    EXPECT_FALSE(verify_zkboost_shared_forest_well_formedness_commitment_binding(instance));
    EXPECT_FALSE(verify_zkboost_shared_forest_well_formedness_plonk(instance, proof));
}

TEST_F(ZkboostSharedForestWellFormednessPlonkTest, RejectsTamperedForestCommitment)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedForestWellFormednessPlonkInstance instance = build_instance(public_params);
    const plonkProof proof = prove_zkboost_shared_forest_well_formedness_plonk(instance);

    instance.commitments.com_w = Group1::zero();

    EXPECT_FALSE(verify_zkboost_shared_forest_well_formedness_commitment_binding(instance));
    EXPECT_FALSE(verify_zkboost_shared_forest_well_formedness_plonk(instance, proof));
}
