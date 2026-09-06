#include "zkboost/shared_trace/shared_prediction.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedPredictionTest : public ::testing::Test {
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

forestSemanticInferenceQueryBatch two_class_inference_trace()
{
    forestSemanticInferenceQueryBatch trace;
    trace.query_count = 8;
    trace.feature_count = 1;
    trace.query_tree_ids = {1, 1, 1, 1, 2, 2, 2, 2};
    trace.lifted_inputs = {
        0, 1, 2, 3,
        0, 1, 2, 3,
    };
    return trace;
}

forestSemanticInferenceQueryBatch subset_inference_trace()
{
    forestSemanticInferenceQueryBatch trace;
    trace.query_count = 4;
    trace.feature_count = 1;
    trace.query_tree_ids = {1, 1, 2, 2};
    trace.lifted_inputs = {3, 1, 3, 1};
    trace.sample_indices = {3, 1};
    return trace;
}

} // namespace

TEST_F(ZkboostSharedPredictionTest, DerivesDemoPredictionAndDigest)
{
    const zkboostSharedTracePrediction prediction =
        derive_zkboost_shared_trace_prediction(build_demo_training_trace(), build_demo_inference_query_trace());

    EXPECT_EQ(prediction.predicted_labels, std::vector<std::int64_t>({0, 0, 0, 0}));
    EXPECT_EQ(prediction.actual_labels, std::vector<std::int64_t>({0, 0, 0, 0}));
    EXPECT_EQ(prediction.accuracy_numerator, 4u);
    EXPECT_EQ(prediction.accuracy_denominator, 4u);
    EXPECT_EQ(prediction.prediction_digest,
              derive_zkboost_shared_trace_prediction(build_demo_training_trace(),
                                                     build_demo_inference_query_trace()).prediction_digest);
}

TEST_F(ZkboostSharedPredictionTest, DerivesTwoClassPredictionsFromOneTrace)
{
    const zkboostSharedTracePrediction prediction =
        derive_zkboost_shared_trace_prediction(two_class_training_trace(), two_class_inference_trace());

    EXPECT_EQ(prediction.predicted_labels, std::vector<std::int64_t>({0, 0, 1, 1}));
    EXPECT_EQ(prediction.actual_labels, std::vector<std::int64_t>({0, 0, 1, 1}));
    EXPECT_EQ(prediction.tree_scores, std::vector<std::int64_t>({5, 5, -5, -5, -5, -5, 5, 5}));
    EXPECT_EQ(prediction.class_scores, std::vector<std::int64_t>({5, 5, -5, -5, -5, -5, 5, 5}));
    EXPECT_EQ(prediction.predicted_scores, std::vector<std::int64_t>({5, 5, 5, 5}));
    EXPECT_EQ(prediction.class_selector,
              std::vector<std::int64_t>({1, 1, 0, 0, 0, 0, 1, 1}));
    EXPECT_EQ(prediction.accuracy_numerator, 4u);
    EXPECT_EQ(prediction.accuracy_denominator, 4u);
}

TEST_F(ZkboostSharedPredictionTest, PredictionChangesWhenSharedForestChanges)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const zkboostSharedTracePrediction original =
        derive_zkboost_shared_trace_prediction(trace, inference);

    trace.node_leaf_values[6] = -5;
    const zkboostSharedTracePrediction changed =
        derive_zkboost_shared_trace_prediction(trace, inference);

    EXPECT_NE(changed.prediction_digest, original.prediction_digest);
    EXPECT_EQ(changed.predicted_labels, std::vector<std::int64_t>({0, 0, 0, 0}));
    EXPECT_EQ(changed.accuracy_numerator, 2u);
}

TEST_F(ZkboostSharedPredictionTest, RejectsInferenceRowsThatDoNotMatchSharedTrace)
{
    forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    inference.query_tree_ids[4] = 0;

    EXPECT_THROW(
        derive_zkboost_shared_trace_prediction(two_class_training_trace(), inference),
        std::runtime_error);
}

TEST_F(ZkboostSharedPredictionTest, SupportsSubsetInferenceBatchViaSampleIndices)
{
    const zkboostSharedTracePrediction prediction =
        derive_zkboost_shared_trace_prediction(two_class_training_trace(), subset_inference_trace());

    EXPECT_EQ(prediction.predicted_labels, std::vector<std::int64_t>({1, 0}));
    EXPECT_EQ(prediction.actual_labels, std::vector<std::int64_t>({1, 0}));
    EXPECT_EQ(prediction.tree_scores, std::vector<std::int64_t>({-5, 5, 5, -5}));
    EXPECT_EQ(prediction.class_scores, std::vector<std::int64_t>({-5, 5, 5, -5}));
    EXPECT_EQ(prediction.predicted_scores, std::vector<std::int64_t>({5, 5}));
    EXPECT_EQ(prediction.class_selector,
              std::vector<std::int64_t>({0, 1, 1, 0}));
    EXPECT_EQ(prediction.accuracy_numerator, 2u);
    EXPECT_EQ(prediction.accuracy_denominator, 2u);
}
