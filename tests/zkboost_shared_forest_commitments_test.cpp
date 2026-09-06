#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/forest_commitments.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedForestCommitmentsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

namespace {

forestTrainingSemanticTrace two_class_training_trace()
{
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    trace.tree_count = 2;
    trace.node_count = 8;
    trace.iteration_count = 1;
    trace.class_count = 2;
    trace.query_count = 4;
    trace.feature_count = 1;
    trace.bin_count = 8;

    trace.root_indices = {0, 4};
    trace.split_node_mask = {1, 0, 0, 0, 1, 0, 0, 0};
    trace.left_child_indices = {1, 1, 2, 3, 5, 5, 6, 7};
    trace.right_child_indices = {2, 1, 2, 3, 6, 5, 6, 7};
    trace.node_tree_ids = {1, 1, 1, 0, 2, 2, 2, 0};
    trace.feature_selector = {1, 0, 0, 0, 1, 0, 0, 0};
    trace.thresholds = {2, 0, 0, 0, 2, 0, 0, 0};
    trace.node_leaf_mask = {0, 1, 1, 0, 0, 1, 1, 0};
    trace.node_leaf_values = {0, 5, -5, 0, 0, -5, 5, 0};
    trace.training_inputs = {0, 1, 2, 3};
    trace.base_labels = {
        1, 1, 0, 0,
        0, 0, 1, 1,
    };
    trace.leaf_indices = {
        1, 1, 2, 2,
        5, 5, 6, 6,
    };
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);
    trace.selected_candidate_indices.assign(trace.node_count, 0);
    trace.selected_candidate_indices[0] = 1;
    trace.selected_candidate_indices[4] = 1;
    return trace;
}

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

} // namespace

TEST_F(ZkboostSharedForestCommitmentsTest, DerivesPaperStyleTreeEnsembleFromSharedTrace)
{
    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(two_class_training_trace());

    EXPECT_EQ(ensemble.tree_count, 2u);
    EXPECT_EQ(ensemble.max_depth, 1u);
    EXPECT_EQ(ensemble.internal_node_count_per_tree, 1u);
    EXPECT_EQ(ensemble.leaf_count_per_tree, 2u);
    EXPECT_EQ(ensemble.split_features, std::vector<std::int64_t>({0, 0}));
    EXPECT_EQ(ensemble.split_thresholds, std::vector<std::int64_t>({2, 2}));
    EXPECT_EQ(ensemble.leaf_values, std::vector<std::int64_t>({5, -5, -5, 5}));
}

TEST_F(ZkboostSharedForestCommitmentsTest, FillsPrunedSubtreesWithDummySplits)
{
    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(pruned_depth_two_trace());

    EXPECT_EQ(ensemble.max_depth, 2u);
    EXPECT_EQ(ensemble.internal_node_count_per_tree, 3u);
    EXPECT_EQ(ensemble.leaf_count_per_tree, 4u);
    EXPECT_EQ(ensemble.split_features, std::vector<std::int64_t>({0, 0, 0}));
    EXPECT_EQ(ensemble.split_thresholds, std::vector<std::int64_t>({2, 0, 5}));
    EXPECT_EQ(ensemble.leaf_values, std::vector<std::int64_t>({7, 7, -3, 4}));
}

TEST_F(ZkboostSharedForestCommitmentsTest, RejectsLeafBeyondOneBasedSemanticMaxDepth)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.max_depth = 1;

    EXPECT_THROW((void)shared_tree_ensemble_from_training_trace(trace), std::runtime_error);
}

TEST_F(ZkboostSharedForestCommitmentsTest, DerivesAuxiliaryForestColumnsFromTrace)
{
    const zkboostSharedForestColumns forest =
        shared_forest_columns_from_training_trace(two_class_training_trace());

    EXPECT_EQ(forest.tree_count, 2u);
    EXPECT_EQ(forest.node_count, 8u);
    EXPECT_EQ(forest.root_indices, std::vector<std::int64_t>({0, 4}));
    EXPECT_EQ(forest.split_masks, std::vector<std::int64_t>({1, 0, 0, 0, 1, 0, 0, 0}));
    EXPECT_EQ(forest.split_features, std::vector<std::int64_t>({0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(forest.thresholds, std::vector<std::int64_t>({2, 0, 0, 0, 2, 0, 0, 0}));
    EXPECT_EQ(forest.left_children, std::vector<std::int64_t>({1, 1, 2, 3, 5, 5, 6, 7}));
    EXPECT_EQ(forest.right_children, std::vector<std::int64_t>({2, 1, 2, 3, 6, 5, 6, 7}));
    EXPECT_EQ(forest.leaf_values, std::vector<std::int64_t>({0, 5, -5, 0, 0, -5, 5, 0}));
}

TEST_F(ZkboostSharedForestCommitmentsTest, EncodesAndPadsForestColumns)
{
    const zkboostSharedForestColumns forest =
        shared_forest_columns_from_training_trace(two_class_training_trace());
    const std::vector<Fr> encoded = encode_zkboost_shared_forest_column(forest.root_indices);
    const std::vector<Fr> padded = encode_zkboost_shared_forest_column_padded(forest.root_indices);

    ASSERT_EQ(encoded.size(), 2u);
    EXPECT_EQ(canonical_size_t_from_fr(encoded[0], "encoded root"), 0u);
    EXPECT_EQ(canonical_size_t_from_fr(encoded[1], "encoded root"), 4u);
    EXPECT_EQ(padded.size(), 2u);

    const std::vector<Fr> padded_thresholds =
        encode_zkboost_shared_forest_column_padded(forest.thresholds);
    EXPECT_EQ(padded_thresholds.size(), 8u);
}

TEST_F(ZkboostSharedForestCommitmentsTest, RejectsMalformedForestColumns)
{
    zkboostSharedForestColumns forest =
        shared_forest_columns_from_training_trace(two_class_training_trace());
    forest.thresholds[0] = 8;

    EXPECT_THROW(validate_zkboost_shared_forest_columns(forest), std::runtime_error);
}

TEST_F(ZkboostSharedForestCommitmentsTest, CommitsLogicalForestAsInstanceArtifact)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(1024, 1024);
    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(two_class_training_trace());

    const zkboostSharedForestCommitments committed_a =
        commit_zkboost_shared_forest(public_params, ensemble);
    const zkboostSharedForestCommitments committed_b =
        commit_zkboost_shared_forest_from_training_trace(public_params, two_class_training_trace());

    EXPECT_EQ(committed_a.com_f, committed_b.com_f);
    EXPECT_EQ(committed_a.com_t, committed_b.com_t);
    EXPECT_EQ(committed_a.com_w, committed_b.com_w);
}

TEST_F(ZkboostSharedForestCommitmentsTest, ForestCommitmentSrsRequirementSeparatesDeg1AndDeg2)
{
    const zkboostSharedTreeEnsemble ensemble =
        shared_tree_ensemble_from_training_trace(two_class_training_trace());
    const zkboostSharedKzgSrsRequirement requirement =
        required_zkboost_shared_forest_commitment_srs_requirement(ensemble);

    EXPECT_EQ(requirement.deg1_plus_1, 4u);
    EXPECT_EQ(requirement.deg2_plus_1, 3u);
}

TEST_F(ZkboostSharedForestCommitmentsTest, ForestCommitmentsChangeWhenForestChanges)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(1024, 1024);
    forestTrainingSemanticTrace trace = two_class_training_trace();

    const zkboostSharedForestCommitments original =
        commit_zkboost_shared_forest_from_training_trace(public_params, trace);
    trace.node_leaf_values[6] = -5;
    const zkboostSharedForestCommitments changed =
        commit_zkboost_shared_forest_from_training_trace(public_params, trace);

    EXPECT_NE(original.com_w, changed.com_w);
    EXPECT_EQ(original.com_f, changed.com_f);
    EXPECT_EQ(original.com_t, changed.com_t);
}
