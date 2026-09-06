#include "bls12_381_wrapper.hpp"
#include "zkboost/io/trace_io.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

zkboostTrainingTrace demo_training_trace()
{
    zkboostTrainingTrace trace;
    trace.tree_shape.max_depth = 2;
    trace.tree_shape.leaf_count = 4;
    trace.tree_shape.internal_node_count = 3;
    trace.tree_shape.node_count = 7;

    trace.hyperparameters.round_count = 2;
    trace.hyperparameters.tree_count = 4;
    trace.hyperparameters.max_depth = 2;
    trace.hyperparameters.bin_count = 8;
    trace.hyperparameters.learning_rate = 1;
    trace.hyperparameters.l2_regularization = 1;

    trace.fixed_point.input_bit_bound = 4;
    trace.fixed_point.full_upper_bound = 8;
    trace.fixed_point.structure_bit_bound = 5;
    trace.fixed_point.score_encoding = fixedPointEncoding::signed_offset_binary(8, 2);
    trace.fixed_point.probability_encoding = fixedPointEncoding::signed_offset_binary(6, 4);
    trace.fixed_point.gradient_encoding = fixedPointEncoding::signed_offset_binary(8, 2);
    trace.fixed_point.hessian_encoding = fixedPointEncoding::signed_offset_binary(6, 4);
    trace.fixed_point.gain_encoding = fixedPointEncoding::signed_offset_binary(10, 2);
    trace.fixed_point.leaf_value_encoding = fixedPointEncoding::signed_offset_binary(8, 2);

    trace.feature_count = 4;
    trace.query_count = 8;
    trace.base_score = 0;
    trace.labels = {0, 1, 0, 1, 0, 1, 0, 1};
    trace.raw_inputs = {
        1, 3, 5, 7,
        3, 5, 7, 9,
        5, 7, 9, 11,
        7, 9, 11, 13,
        9, 11, 13, 17,
        11, 13, 17, 1,
        13, 17, 1, 3,
        17, 1, 3, 5,
    };
    trace.prebin_feature_mins = {1, 1, 1, 1};
    trace.prebin_feature_maxs = {17, 17, 17, 17};
    trace.prebin_deltas = {2, 2, 2, 2};
    trace.prebin_edges = {
        1, 3, 5, 7, 9, 11, 13, 15,
        1, 3, 5, 7, 9, 11, 13, 15,
        1, 3, 5, 7, 9, 11, 13, 15,
        1, 3, 5, 7, 9, 11, 13, 15,
    };
    trace.binned_inputs = {
        0, 1, 2, 3,
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7,
        5, 6, 7, 0,
        6, 7, 0, 1,
        7, 0, 1, 2,
    };
    trace.prebin_bin_ids = trace.binned_inputs;
    trace.prebin_dummy_feature = 0;
    trace.prebin_dummy_bin = 3;
    trace.prebin_dummy_threshold = 7;
    trace.root_indices = {0, 7, 14, 21};
    trace.split_node_mask = {
        1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0,
    };
    trace.left_child_indices = {
        1, 3, 5, 0, 0, 0, 0,
        8, 10, 12, 0, 0, 0, 0,
        15, 17, 19, 0, 0, 0, 0,
        22, 24, 26, 0, 0, 0, 0,
    };
    trace.right_child_indices = {
        2, 4, 6, 0, 0, 0, 0,
        9, 11, 13, 0, 0, 0, 0,
        16, 18, 20, 0, 0, 0, 0,
        23, 25, 27, 0, 0, 0, 0,
    };
    trace.node_split_features = {
        0, 1, 2, 0, 0, 0, 0,
        0, 1, 2, 0, 0, 0, 0,
        0, 1, 2, 0, 0, 0, 0,
        0, 1, 2, 0, 0, 0, 0,
    };
    trace.node_threshold_bins = {
        3, 2, 5, 0, 0, 0, 0,
        3, 2, 5, 0, 0, 0, 0,
        3, 2, 5, 0, 0, 0, 0,
        3, 2, 5, 0, 0, 0, 0,
    };
    trace.node_dummy_mask = {
        0, 0, 1, 0, 0, 1, 1,
        0, 0, 1, 0, 0, 1, 1,
        0, 0, 1, 0, 0, 1, 1,
        0, 0, 1, 0, 0, 1, 1,
    };
    trace.node_leaf_mask = {
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
    };
    trace.leaf_weights = {
        3, -1, 2, 0,
        4, -2, 1, 0,
        5, -2, 1, 1,
        6, -3, 2, 1,
    };
    const std::vector<std::vector<std::int64_t>> local_paths = {
        {0, 1, 3},
        {0, 1, 3},
        {0, 1, 4},
        {0, 1, 4},
        {0, 2, 6},
        {0, 2, 6},
        {0, 2, 5},
        {0, 2, 5},
    };
    const std::vector<std::vector<std::int64_t>> local_decisions = {
        {0, 0},
        {0, 0},
        {0, 1},
        {0, 1},
        {1, 1},
        {1, 1},
        {1, 0},
        {1, 0},
    };
    for (size_t tree = 0; tree < trace.hyperparameters.tree_count; ++tree) {
        const std::int64_t base = static_cast<std::int64_t>(tree * trace.tree_shape.node_count);
        for (size_t query = 0; query < trace.query_count; ++query) {
            trace.leaf_indices.push_back(base + local_paths[query].back());
            for (std::int64_t node : local_paths[query]) {
                trace.path_node_indices.push_back(base + node);
            }
            for (std::int64_t bit : local_decisions[query]) {
                trace.path_decisions.push_back(bit);
            }
        }
    }
    trace.round_start_scores.assign(trace.hyperparameters.round_count * trace.query_count, 0);
    trace.round_probabilities.assign(trace.hyperparameters.round_count * trace.query_count, 0);
    trace.round_gradients.assign(trace.hyperparameters.round_count * trace.query_count, 0);
    trace.round_hessians.assign(trace.hyperparameters.round_count * trace.query_count, 0);
    const size_t hist_count =
        trace.hyperparameters.tree_count * trace.tree_shape.node_count * trace.feature_count *
        trace.hyperparameters.bin_count;
    trace.node_hist_counts.assign(hist_count, 0);
    trace.node_hist_gradients.assign(hist_count, 0);
    trace.node_hist_hessians.assign(hist_count, 0);
    return trace;
}

zkboostInferenceTrace demo_inference_trace()
{
    zkboostInferenceTrace trace;
    trace.query_count = 8;
    trace.feature_count = 4;
    trace.tree_count = 4;
    trace.max_depth = 2;
    trace.query_tree_ids = {0, 1, 2, 3, 0, 1, 2, 3};
    trace.binned_inputs = {
        0, 1, 2, 3,
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7,
        5, 6, 7, 0,
        6, 7, 0, 1,
        7, 0, 1, 2,
    };
    const std::vector<std::vector<std::int64_t>> local_paths = {
        {0, 1, 3},
        {0, 1, 3},
        {0, 1, 4},
        {0, 1, 4},
        {0, 2, 6},
        {0, 2, 6},
        {0, 2, 5},
        {0, 2, 5},
    };
    const std::vector<std::vector<std::int64_t>> local_decisions = {
        {0, 0},
        {0, 0},
        {0, 1},
        {0, 1},
        {1, 1},
        {1, 1},
        {1, 0},
        {1, 0},
    };
    for (size_t tree = 0; tree < trace.tree_count; ++tree) {
        const std::int64_t base = static_cast<std::int64_t>(tree * 7);
        for (size_t query = 0; query < trace.query_count; ++query) {
            trace.leaf_indices.push_back(base + local_paths[query].back());
            for (std::int64_t node : local_paths[query]) {
                trace.path_node_indices.push_back(base + node);
            }
            for (std::int64_t bit : local_decisions[query]) {
                trace.path_decisions.push_back(bit);
            }
        }
    }
    return trace;
}

} // namespace

class ZkboostTraceIoTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ZkboostTraceIoTest, SummarizesAndRoundTrips)
{
    const zkboostTrainingTrace training = demo_training_trace();
    const zkboostInferenceTrace inference = demo_inference_trace();

    const zkboostTraceSummary summary = summarize_zkboost_traces(training, inference);
    const std::string summary_text = zkboost_trace_summary_description(summary);
    const std::string format_text = zkboost_trace_format_description();

    EXPECT_EQ(summary.round_count, 2u);
    EXPECT_EQ(summary.tree_count, 4u);
    EXPECT_EQ(summary.node_count, 28u);
    EXPECT_EQ(summary.leaf_count, 16u);
    EXPECT_EQ(summary.feature_count, 4u);
    EXPECT_EQ(summary.bin_count, 8u);
    EXPECT_EQ(summary.training_query_count, 8u);
    EXPECT_EQ(summary.inference_query_count, 8u);
    EXPECT_NE(summary_text.find("CertXGB trace summary"), std::string::npos);
    EXPECT_NE(format_text.find("zkboost_binary_training_trace_v1"), std::string::npos);

    const std::string training_path = "/tmp/zkboost_training.trace";
    const std::string inference_path = "/tmp/zkboost_inference.trace";
    write_zkboost_training_trace(training_path, training);
    write_zkboost_inference_trace(inference_path, inference);

    const zkboostTrainingTrace loaded_training = read_zkboost_training_trace(training_path);
    const zkboostInferenceTrace loaded_inference = read_zkboost_inference_trace(inference_path);

    EXPECT_EQ(loaded_training.labels, training.labels);
    EXPECT_EQ(loaded_training.node_threshold_bins, training.node_threshold_bins);
    EXPECT_EQ(loaded_training.raw_inputs, training.raw_inputs);
    EXPECT_EQ(loaded_training.prebin_feature_mins, training.prebin_feature_mins);
    EXPECT_EQ(loaded_training.prebin_feature_maxs, training.prebin_feature_maxs);
    EXPECT_EQ(loaded_training.prebin_deltas, training.prebin_deltas);
    EXPECT_EQ(loaded_training.prebin_edges, training.prebin_edges);
    EXPECT_EQ(loaded_training.prebin_bin_ids, training.prebin_bin_ids);
    EXPECT_EQ(loaded_training.prebin_dummy_feature, training.prebin_dummy_feature);
    EXPECT_EQ(loaded_training.prebin_dummy_bin, training.prebin_dummy_bin);
    EXPECT_EQ(loaded_training.prebin_dummy_threshold, training.prebin_dummy_threshold);
    EXPECT_EQ(loaded_training.leaf_weights, training.leaf_weights);
    EXPECT_EQ(loaded_training.leaf_indices, training.leaf_indices);
    EXPECT_EQ(loaded_training.path_node_indices, training.path_node_indices);
    EXPECT_EQ(loaded_training.path_decisions, training.path_decisions);
    EXPECT_EQ(loaded_training.base_score, training.base_score);
    EXPECT_EQ(loaded_training.round_start_scores, training.round_start_scores);
    EXPECT_EQ(loaded_training.round_probabilities, training.round_probabilities);
    EXPECT_EQ(loaded_training.round_gradients, training.round_gradients);
    EXPECT_EQ(loaded_training.round_hessians, training.round_hessians);
    EXPECT_EQ(loaded_inference.query_tree_ids, inference.query_tree_ids);
    EXPECT_EQ(loaded_inference.binned_inputs, inference.binned_inputs);
    EXPECT_EQ(loaded_inference.leaf_indices, inference.leaf_indices);
    EXPECT_EQ(loaded_inference.path_node_indices, inference.path_node_indices);
    EXPECT_EQ(loaded_inference.path_decisions, inference.path_decisions);

    std::remove(training_path.c_str());
    std::remove(inference_path.c_str());
}

TEST_F(ZkboostTraceIoTest, RejectsNonBinaryLabels)
{
    zkboostTrainingTrace training = demo_training_trace();
    training.labels[0] = 2;
    EXPECT_THROW(validate_zkboost_training_trace(training), std::runtime_error);
}
