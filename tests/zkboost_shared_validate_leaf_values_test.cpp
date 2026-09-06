#include "algebra_utils.hpp"
#include "zkboost/shared_trace/validate_leaf_values.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedValidateLeafValuesTest : public ::testing::Test {
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

forestTrainingSemanticTrace demo_trace()
{
    return build_demo_training_trace();
}

forestTrainingSemanticTrace two_class_valid_leaf_trace()
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
    trace.node_leaf_values = {0, 0, -1, 0, 0, -1, 0, 0};
    trace.learning_rate_numerator = 1;
    trace.learning_rate_denominator = 2;

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

forestTrainingSemanticTrace first_order_two_class_leaf_trace()
{
    forestTrainingSemanticTrace trace = two_class_valid_leaf_trace();
    trace.loss_mode = forestTrainingLoss::FirstOrderRss;
    trace.l2_regularization = 0;
    trace.leaf_value_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    trace.node_leaf_values = {0, 1, -1, 0, 0, -1, 1, 0};
    return trace;
}

} // namespace

TEST_F(ZkboostSharedValidateLeafValuesTest, DerivesDemoLeafValues)
{
    const zkboostSharedLeafValueView view =
        build_zkboost_shared_leaf_value_view(demo_trace());

    EXPECT_EQ(view.summary.node_count, 4u);
    EXPECT_EQ(view.summary.leaf_node_count, 2u);
    EXPECT_EQ(view.summary.histogram_width, 4u);
    EXPECT_EQ(view.summary.l2_regularization, 1u);
    EXPECT_EQ(view.summary.learning_rate_numerator, 1u);
    EXPECT_EQ(view.summary.learning_rate_denominator, 1u);

    EXPECT_EQ(view.leaf_gradients, std::vector<Fr>({fr_i64(0), fr_i64(0), fr_i64(0), fr_i64(0)}));
    EXPECT_EQ(view.leaf_hessians, std::vector<Fr>({fr_i64(0), fr_i64(0), fr_i64(0), fr_i64(0)}));
    EXPECT_EQ(view.leaf_denominators, std::vector<Fr>({fr_i64(1), fr_i64(1), fr_i64(1), fr_i64(1)}));
    EXPECT_EQ(view.decoded_leaf_values, std::vector<Fr>({fr_i64(0), fr_i64(0), fr_i64(0), fr_i64(0)}));
    EXPECT_EQ(view.leaf_remainders, std::vector<Fr>({fr_i64(0), fr_i64(0), fr_i64(0), fr_i64(0)}));
}

TEST_F(ZkboostSharedValidateLeafValuesTest, DerivesTwoClassLeafValues)
{
    const zkboostSharedLeafValueView view =
        build_zkboost_shared_leaf_value_view(two_class_valid_leaf_trace());

    EXPECT_EQ(view.summary.learning_rate_numerator, 1u);
    EXPECT_EQ(view.summary.learning_rate_denominator, 2u);
    EXPECT_EQ(view.summary.node_count, 8u);
    EXPECT_EQ(view.summary.leaf_node_count, 4u);
    EXPECT_EQ(view.leaf_gradients, std::vector<Fr>({fr_i64(0), fr_i64(-4), fr_i64(4), fr_i64(0),
                                                    fr_i64(0), fr_i64(4), fr_i64(-4), fr_i64(0)}));
    EXPECT_EQ(view.leaf_hessians, std::vector<Fr>({fr_i64(4), fr_i64(2), fr_i64(2), fr_i64(0),
                                                   fr_i64(4), fr_i64(2), fr_i64(2), fr_i64(0)}));
    EXPECT_EQ(view.leaf_denominators, std::vector<Fr>({fr_i64(10), fr_i64(6), fr_i64(6), fr_i64(2),
                                                       fr_i64(10), fr_i64(6), fr_i64(6), fr_i64(2)}));
    EXPECT_EQ(view.decoded_leaf_values, std::vector<Fr>({fr_i64(0), fr_i64(0), fr_i64(-1), fr_i64(0),
                                                         fr_i64(0), fr_i64(-1), fr_i64(0), fr_i64(0)}));
    EXPECT_EQ(view.leaf_remainders, std::vector<Fr>({fr_i64(0), fr_i64(4), fr_i64(2), fr_i64(0),
                                                     fr_i64(0), fr_i64(2), fr_i64(4), fr_i64(0)}));
}

TEST_F(ZkboostSharedValidateLeafValuesTest, DerivesFirstOrderMeanResidualLeafValues)
{
    const zkboostSharedLeafValueView view =
        build_zkboost_shared_leaf_value_view(first_order_two_class_leaf_trace());

    EXPECT_EQ(view.leaf_gradients, std::vector<Fr>({fr_i64(0), fr_i64(4), fr_i64(-4), fr_i64(0),
                                                    fr_i64(0), fr_i64(-4), fr_i64(4), fr_i64(0)}));
    EXPECT_EQ(view.leaf_hessians, std::vector<Fr>({fr_i64(16), fr_i64(8), fr_i64(8), fr_i64(0),
                                                   fr_i64(16), fr_i64(8), fr_i64(8), fr_i64(0)}));
    EXPECT_EQ(view.leaf_denominators, std::vector<Fr>({fr_i64(34), fr_i64(16), fr_i64(16), fr_i64(2),
                                                       fr_i64(34), fr_i64(16), fr_i64(16), fr_i64(2)}));
    EXPECT_EQ(view.decoded_leaf_values, std::vector<Fr>({fr_i64(0), fr_i64(1), fr_i64(-1), fr_i64(0),
                                                         fr_i64(0), fr_i64(-1), fr_i64(1), fr_i64(0)}));
}

TEST_F(ZkboostSharedValidateLeafValuesTest, RejectsCommittedLeafValueMismatch)
{
    forestTrainingSemanticTrace trace = two_class_valid_leaf_trace();
    trace.node_leaf_values[1] = 99;

    EXPECT_THROW(
        (void)validate_zkboost_shared_leaf_values(trace),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateLeafValuesTest, RejectsBadHistogramShape)
{
    const forestTrainingSemanticTrace trace = two_class_valid_leaf_trace();
    zkboostSharedHistogramView histograms = build_zkboost_shared_histogram_view(trace);
    histograms.left_statistics[1].pop_back();

    EXPECT_THROW(
        (void)validate_zkboost_shared_leaf_values(trace, histograms),
        std::runtime_error);
}
