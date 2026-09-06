#include "algebra_utils.hpp"
#include "zkboost/shared_trace/validate_splits.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedValidateSplitsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

namespace {

Fr fr_i64(std::int64_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

forestTrainingSemanticTrace demo_split_trace()
{
    return build_demo_training_trace();
}

forestTrainingSemanticTrace two_class_split_trace()
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
    trace.node_leaf_values = {0, 1, -2, 0, 0, -2, 1, 0};

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
    // The mask describes every threshold with non-empty left and right sides,
    // not merely the threshold selected by the prover.
    for (const size_t index : {0u, 1u, 2u, 8u, 18u, 32u, 33u, 34u, 40u, 50u}) {
        trace.candidate_mask[index] = 1;
    }
    trace.selected_candidate_indices = {1, 0, 2, 0, 1, 0, 2, 0};
    return trace;
}

size_t offset(size_t node, size_t bin)
{
    return node * 8 + bin;
}

forestTrainingSemanticTrace first_order_two_class_split_trace()
{
    forestTrainingSemanticTrace trace = two_class_split_trace();
    trace.loss_mode = forestTrainingLoss::FirstOrderRss;
    trace.l2_regularization = 0;
    trace.split_penalty = Fr(0);
    trace.gain_numerator_encoding = fixedPointEncoding::signed_offset_binary(16, 0);
    return trace;
}

} // namespace

TEST_F(ZkboostSharedValidateSplitsTest, DerivesDemoSplitSelection)
{
    const zkboostSharedSplitView view =
        build_zkboost_shared_split_view(demo_split_trace());

    EXPECT_EQ(view.summary.node_count, 4u);
    EXPECT_EQ(view.summary.feature_count, 1u);
    EXPECT_EQ(view.summary.bin_count, 4u);
    EXPECT_EQ(view.summary.candidate_count, 16u);
    EXPECT_EQ(view.summary.split_node_count, 1u);
    EXPECT_EQ(view.selected_features, std::vector<Fr>({fr_i64(0), fr_i64(0), fr_i64(0), fr_i64(0)}));
    EXPECT_EQ(view.selected_bins, std::vector<Fr>({fr_i64(0), fr_i64(0), fr_i64(0), fr_i64(0)}));
    EXPECT_EQ(view.selector[0], fr_i64(1));
}

TEST_F(ZkboostSharedValidateSplitsTest, DerivesTwoClassSplitSelection)
{
    const zkboostSharedSplitView view =
        build_zkboost_shared_split_view(two_class_split_trace());

    EXPECT_EQ(view.summary.node_count, 8u);
    EXPECT_EQ(view.summary.feature_count, 1u);
    EXPECT_EQ(view.summary.bin_count, 8u);
    EXPECT_EQ(view.summary.candidate_count, 64u);
    EXPECT_EQ(view.summary.split_node_count, 2u);
    EXPECT_EQ(view.left_gradients[offset(0, 1)], fr_i64(-4));
    EXPECT_EQ(view.right_gradients[offset(0, 1)], fr_i64(4));
    EXPECT_EQ(view.left_hessians[offset(0, 1)], fr_i64(2));
    EXPECT_EQ(view.right_hessians[offset(0, 1)], fr_i64(2));
    EXPECT_EQ(view.selected_features, std::vector<Fr>({fr_i64(0), fr_i64(0), fr_i64(0), fr_i64(0),
                                                       fr_i64(0), fr_i64(0), fr_i64(0), fr_i64(0)}));
    EXPECT_EQ(view.selected_bins, std::vector<Fr>({fr_i64(1), fr_i64(0), fr_i64(2), fr_i64(0),
                                                   fr_i64(1), fr_i64(0), fr_i64(2), fr_i64(0)}));
    EXPECT_EQ(view.selector[offset(0, 1)], fr_i64(1));
    EXPECT_EQ(view.selector[offset(4, 1)], fr_i64(1));
}

TEST_F(ZkboostSharedValidateSplitsTest, DerivesFirstOrderRssAndSafeMaskedDenominators)
{
    const zkboostSharedSplitView view =
        build_zkboost_shared_split_view(first_order_two_class_split_trace());

    EXPECT_EQ(view.left_gradients[offset(0, 1)], fr_i64(4));
    EXPECT_EQ(view.right_gradients[offset(0, 1)], fr_i64(-4));
    EXPECT_EQ(view.left_hessians[offset(0, 1)], fr_i64(8));
    EXPECT_EQ(view.right_hessians[offset(0, 1)], fr_i64(8));
    EXPECT_EQ(view.left_counts[offset(0, 1)], fr_i64(2));
    EXPECT_EQ(view.right_counts[offset(0, 1)], fr_i64(2));
    EXPECT_EQ(view.gain_numerators[offset(0, 1)], fr_i64(256));
    EXPECT_EQ(view.gain_denominators[offset(0, 1)], fr_i64(64));
    EXPECT_EQ(view.selector[offset(0, 1)], fr_i64(1));
    EXPECT_NE(view.gain_denominators[offset(0, 0)], Fr(0));
}

TEST_F(ZkboostSharedValidateSplitsTest, RejectsSelectedCandidateMismatch)
{
    forestTrainingSemanticTrace trace = two_class_split_trace();
    trace.selected_candidate_indices[0] = 0;
    trace.candidate_mask[0] = 1;

    EXPECT_THROW((void)validate_zkboost_shared_splits(trace), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsTest, RejectsMissingCandidateForSplitNode)
{
    forestTrainingSemanticTrace trace = two_class_split_trace();
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);

    EXPECT_THROW((void)validate_zkboost_shared_splits(trace), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsTest, RejectsFeatureSelectorMismatch)
{
    forestTrainingSemanticTrace trace = two_class_split_trace();
    trace.feature_selector[0] = 0;

    EXPECT_THROW((void)validate_zkboost_shared_splits(trace), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsTest, RejectsThresholdMismatch)
{
    forestTrainingSemanticTrace trace = two_class_split_trace();
    trace.thresholds[0] = 1;

    EXPECT_THROW((void)validate_zkboost_shared_splits(trace), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsTest, RejectsSplitGainBelowStoppingThreshold)
{
    forestTrainingSemanticTrace trace = two_class_split_trace();
    trace.split_gain_threshold = fr_i64(1000);

    EXPECT_THROW((void)validate_zkboost_shared_splits(trace), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsTest, RejectsEarlyLeafWithEligibleSplit)
{
    forestTrainingSemanticTrace trace = demo_split_trace();
    trace.split_node_mask[0] = 0;
    trace.node_leaf_mask[0] = 1;
    trace.feature_selector[0] = 0;

    EXPECT_THROW((void)validate_zkboost_shared_splits(trace), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsTest, AcceptsThresholdStoppedEarlyLeaf)
{
    forestTrainingSemanticTrace trace = demo_split_trace();
    trace.split_node_mask[0] = 0;
    trace.node_leaf_mask[0] = 1;
    trace.feature_selector[0] = 0;
    trace.leaf_indices = {0, 0, 0, 0};
    trace.split_gain_threshold = fr_i64(1000);

    const zkboostSharedSplitView view = build_zkboost_shared_split_view(trace);
    EXPECT_EQ(view.active_node_masks[0], Fr(1));
    EXPECT_EQ(view.leaf_masks[0], Fr(1));
    EXPECT_EQ(view.node_depths[0], Fr(1));
    EXPECT_EQ(view.depth_limit_masks[0], Fr(0));
}

TEST_F(ZkboostSharedValidateSplitsTest, AcceptsCandidateBearingLeafAtMaximumDepth)
{
    forestTrainingSemanticTrace trace = two_class_split_trace();

    const zkboostSharedSplitView view = build_zkboost_shared_split_view(trace);
    EXPECT_EQ(view.active_node_masks[1], Fr(1));
    EXPECT_EQ(view.leaf_masks[1], Fr(1));
    EXPECT_EQ(view.node_depths[1], Fr(2));
    EXPECT_EQ(view.depth_limit_masks[1], Fr(1));
    EXPECT_EQ(view.selector[offset(1, 0)], Fr(1));
}

TEST_F(ZkboostSharedValidateSplitsTest, RejectsBadHistogramShape)
{
    const forestTrainingSemanticTrace trace = two_class_split_trace();
    zkboostSharedHistogramView histograms = build_zkboost_shared_histogram_view(trace);
    histograms.right_statistics[1].pop_back();

    EXPECT_THROW((void)validate_zkboost_shared_splits(trace, histograms), std::runtime_error);
}
