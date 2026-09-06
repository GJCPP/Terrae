#include "algebra_utils.hpp"
#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/validate_boosting_state_plonk.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"
#include "zkboost/shared_trace/validate_histograms.hpp"
#include "zkboost/shared_trace/validate_histograms_plonk.hpp"
#include "zkboost/shared_trace/weighted_logup.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedValidateHistogramsTest : public ::testing::Test {
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

size_t offset(size_t node, size_t bin)
{
    return node * 8 + bin;
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

} // namespace

TEST_F(ZkboostSharedValidateHistogramsTest, DerivesLeafAndFullHistograms)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const zkboostSharedHistogramView view =
        build_zkboost_shared_histogram_view(training);

    EXPECT_EQ(view.summary.tree_count, 2u);
    EXPECT_EQ(view.summary.node_count, 8u);
    EXPECT_EQ(view.summary.feature_count, 1u);
    EXPECT_EQ(view.summary.bin_count, 8u);
    EXPECT_EQ(view.summary.query_row_count, 8u);
    EXPECT_EQ(view.summary.histogram_width, 8u);
    EXPECT_EQ(view.summary.histogram_size, 64u);
    EXPECT_EQ(view.summary.channel_count, 3u);

    ASSERT_EQ(view.leaf_histograms.size(), 3u);
    ASSERT_EQ(view.full_histograms.size(), 3u);
    EXPECT_EQ(view.query_pseudo_residuals,
              std::vector<Fr>({fr_i64(-2), fr_i64(-2), fr_i64(2), fr_i64(2),
                               fr_i64(2), fr_i64(2), fr_i64(-2), fr_i64(-2)}));

    EXPECT_EQ(view.lifted_leaf_indices,
              std::vector<Fr>({fr_i64(1), fr_i64(1), fr_i64(2), fr_i64(2),
                               fr_i64(5), fr_i64(5), fr_i64(6), fr_i64(6)}));
    EXPECT_EQ(view.flattened_bucket_indices,
              std::vector<Fr>({fr_i64(8), fr_i64(9), fr_i64(18), fr_i64(19),
                               fr_i64(40), fr_i64(41), fr_i64(50), fr_i64(51)}));

    const std::vector<Fr>& count_leaf = view.leaf_histograms[0];
    EXPECT_EQ(count_leaf[offset(1, 0)], fr_i64(1));
    EXPECT_EQ(count_leaf[offset(1, 1)], fr_i64(1));
    EXPECT_EQ(count_leaf[offset(2, 2)], fr_i64(1));
    EXPECT_EQ(count_leaf[offset(2, 3)], fr_i64(1));
    EXPECT_EQ(count_leaf[offset(5, 0)], fr_i64(1));
    EXPECT_EQ(count_leaf[offset(5, 1)], fr_i64(1));
    EXPECT_EQ(count_leaf[offset(6, 2)], fr_i64(1));
    EXPECT_EQ(count_leaf[offset(6, 3)], fr_i64(1));

    const std::vector<Fr>& residual_leaf = view.leaf_histograms[1];
    EXPECT_EQ(residual_leaf[offset(1, 0)], fr_i64(-2));
    EXPECT_EQ(residual_leaf[offset(1, 1)], fr_i64(-2));
    EXPECT_EQ(residual_leaf[offset(2, 2)], fr_i64(2));
    EXPECT_EQ(residual_leaf[offset(2, 3)], fr_i64(2));
    EXPECT_EQ(residual_leaf[offset(5, 0)], fr_i64(2));
    EXPECT_EQ(residual_leaf[offset(5, 1)], fr_i64(2));
    EXPECT_EQ(residual_leaf[offset(6, 2)], fr_i64(-2));
    EXPECT_EQ(residual_leaf[offset(6, 3)], fr_i64(-2));

    EXPECT_EQ(view.full_histograms[0][offset(0, 0)], fr_i64(1));
    EXPECT_EQ(view.full_histograms[0][offset(0, 1)], fr_i64(1));
    EXPECT_EQ(view.full_histograms[0][offset(0, 2)], fr_i64(1));
    EXPECT_EQ(view.full_histograms[0][offset(0, 3)], fr_i64(1));
    EXPECT_EQ(view.full_histograms[1][offset(0, 0)], fr_i64(-2));
    EXPECT_EQ(view.full_histograms[1][offset(0, 2)], fr_i64(2));
    EXPECT_EQ(view.full_histograms[1][offset(4, 0)], fr_i64(2));
    EXPECT_EQ(view.full_histograms[1][offset(4, 2)], fr_i64(-2));
}

TEST_F(ZkboostSharedValidateHistogramsTest, DerivesLeftAndRightStatistics)
{
    const zkboostSharedHistogramView view =
        build_zkboost_shared_histogram_view(two_class_training_trace());

    ASSERT_EQ(view.left_statistics.size(), 3u);
    ASSERT_EQ(view.right_statistics.size(), 3u);

    EXPECT_EQ(view.left_statistics[0][offset(0, 0)], fr_i64(1));
    EXPECT_EQ(view.left_statistics[0][offset(0, 1)], fr_i64(2));
    EXPECT_EQ(view.left_statistics[0][offset(0, 2)], fr_i64(3));
    EXPECT_EQ(view.left_statistics[0][offset(0, 3)], fr_i64(4));
    EXPECT_EQ(view.right_statistics[0][offset(0, 0)], fr_i64(3));
    EXPECT_EQ(view.right_statistics[0][offset(0, 1)], fr_i64(2));
    EXPECT_EQ(view.right_statistics[0][offset(0, 2)], fr_i64(1));
    EXPECT_EQ(view.right_statistics[0][offset(0, 3)], fr_i64(0));
}

TEST_F(ZkboostSharedValidateHistogramsTest, RejectsBadBoostingShape)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    zkboostSharedBoostingStateView boosting =
        build_zkboost_shared_boosting_state_view(training);
    boosting.pseudo_residuals.pop_back();

    EXPECT_THROW(
        (void)validate_zkboost_shared_histograms(training, boosting),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateHistogramsTest, RejectsLeafIndexOutOfRange)
{
    forestTrainingSemanticTrace training = two_class_training_trace();
    training.leaf_indices[0] = 99;

    EXPECT_THROW(
        (void)validate_zkboost_shared_histograms(training),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateHistogramsTest, RejectsTrainingInputBinOutOfRange)
{
    forestTrainingSemanticTrace training = two_class_training_trace();
    training.training_inputs[0] = 99;

    EXPECT_THROW(
        (void)validate_zkboost_shared_histograms(training),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateHistogramsTest, ProvesAndVerifiesHistogramCircuit)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const zkboostSharedHistogramView view =
        build_zkboost_shared_histogram_view(training);
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedHistogramPlonkInstance instance =
        build_zkboost_shared_histogram_plonk_instance(training, public_params);

    EXPECT_EQ(instance.public_inputs, encode_zkboost_shared_histogram_public_inputs(view, training));
    EXPECT_EQ(instance.summary.histogram_size, 64u);

    const plonkProof proof = prove_zkboost_shared_histogram_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_histogram_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateHistogramsTest, ProvesAndVerifiesWeightedLogup)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const zkboostSharedHistogramView view =
        build_zkboost_shared_histogram_view(training);
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance_from_view(
            training,
            view,
            public_params);
    const zkboostSharedWeightedLogupInstance logup_instance =
        build_zkboost_shared_weighted_logup_instance(
            training,
            view,
            public_params);

    EXPECT_EQ(logup_instance.summary.query_domain_size, 8u);
    EXPECT_EQ(logup_instance.summary.histogram_domain_size, 64u);
    EXPECT_TRUE(link_zkboost_shared_histograms_to_weighted_logup(
        histogram_instance,
        logup_instance));

    const SerializedProof proof = prove_zkboost_shared_weighted_logup(logup_instance);
    EXPECT_FALSE(proof.bytes.empty());
    EXPECT_TRUE(verify_zkboost_shared_weighted_logup(logup_instance, proof));

    SerializedProof tampered = proof;
    tampered.bytes.at(tampered.bytes.size() / 2) ^= 1;
    EXPECT_FALSE(verify_zkboost_shared_weighted_logup(logup_instance, tampered));
}

TEST_F(ZkboostSharedValidateHistogramsTest, WeightedLogupLinkRejectsUnrelatedCommitment)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const zkboostSharedHistogramView view = build_zkboost_shared_histogram_view(training);
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance_from_view(training, view, public_params);
    zkboostSharedWeightedLogupInstance logup_instance =
        build_zkboost_shared_weighted_logup_instance(training, view, public_params);

    const Group1 recomputed_commitment = commit_g1_with_public_parameters(
        logup_instance.public_parameters,
        logup_instance.query_indices);
    const Group1 original_commitment = logup_instance.query_indices_commitment;
    ASSERT_EQ(recomputed_commitment, original_commitment);
    logup_instance.query_indices_commitment += public_params.g1_powers.front();
    ASSERT_NE(logup_instance.query_indices_commitment, original_commitment);
    ASSERT_NE(logup_instance.query_indices_commitment, recomputed_commitment);
    EXPECT_FALSE(link_zkboost_shared_histograms_to_weighted_logup(
        histogram_instance,
        logup_instance));
}

TEST_F(ZkboostSharedValidateHistogramsTest, WeightedLogupLinkRejectsTamperedBucketAddress)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const zkboostSharedHistogramView view = build_zkboost_shared_histogram_view(training);
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance_from_view(training, view, public_params);
    const zkboostSharedWeightedLogupInstance logup_instance =
        build_zkboost_shared_weighted_logup_instance(training, view, public_params);

    const size_t flattened_offset =
        3 * histogram_instance.summary.query_row_count +
        12 * histogram_instance.summary.histogram_size +
        histogram_instance.summary.sample_count * histogram_instance.summary.feature_count;
    ASSERT_LT(flattened_offset, histogram_instance.public_inputs.size());
    histogram_instance.public_inputs[flattened_offset] += Fr(1);
    EXPECT_FALSE(link_zkboost_shared_histograms_to_weighted_logup(
        histogram_instance,
        logup_instance));
}

TEST_F(ZkboostSharedValidateHistogramsTest, LinksForestAndBoostingStateToHistograms)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance forest_instance =
        build_zkboost_shared_validate_forest_plonk_instance(
            training,
            two_class_inference_trace(),
            public_params);
    const zkboostSharedBoostingStatePlonkInstance boosting_instance =
        build_zkboost_shared_boosting_state_plonk_instance(training, public_params);
    const zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance(training, public_params);

    EXPECT_TRUE(link_zkboost_shared_validate_forest_to_histograms(
        forest_instance,
        histogram_instance));
    EXPECT_TRUE(link_zkboost_shared_boosting_state_to_histograms(
        boosting_instance,
        histogram_instance));
}

TEST_F(ZkboostSharedValidateHistogramsTest, LinkRejectsTamperedResidualPublicInput)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedBoostingStatePlonkInstance boosting_instance =
        build_zkboost_shared_boosting_state_plonk_instance(training, public_params);
    zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance(training, public_params);

    const size_t residual_offset = histogram_instance.summary.query_row_count;
    ASSERT_LT(residual_offset, histogram_instance.public_inputs.size());
    histogram_instance.public_inputs[residual_offset] += Fr(1);

    EXPECT_FALSE(link_zkboost_shared_boosting_state_to_histograms(
        boosting_instance,
        histogram_instance));
}

TEST_F(ZkboostSharedValidateHistogramsTest, VerifyRejectsTamperedHistogramPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedHistogramPlonkInstance instance =
        build_zkboost_shared_histogram_plonk_instance(two_class_training_trace(), public_params);
    const plonkProof proof = prove_zkboost_shared_histogram_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    const size_t leaf_histogram_offset = instance.summary.query_row_count * 3;
    ASSERT_LT(leaf_histogram_offset, bad_public_inputs.size());
    bad_public_inputs[leaf_histogram_offset] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_histogram_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateHistogramsTest, WeightedLogupRejectsBadLeafHistogramWitness)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    zkboostSharedHistogramView view = build_zkboost_shared_histogram_view(training);
    ASSERT_FALSE(view.leaf_histograms.empty());
    ASSERT_GT(view.leaf_histograms[0].size(), offset(1, 0));
    view.leaf_histograms[0][offset(1, 0)] += Fr(1);

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedWeightedLogupInstance instance =
        build_zkboost_shared_weighted_logup_instance(
            training,
            view,
            public_params);

    EXPECT_THROW((void)prove_zkboost_shared_weighted_logup(instance), std::runtime_error);
}
