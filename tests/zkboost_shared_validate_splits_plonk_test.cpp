#include "algebra_utils.hpp"
#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/validate_histograms_plonk.hpp"
#include "zkboost/shared_trace/validate_splits.hpp"
#include "zkboost/shared_trace/validate_splits_plonk.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedValidateSplitsPlonkTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

namespace {

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
    for (size_t bin = 0; bin < 3; ++bin) {
        trace.candidate_mask[bin] = 1;
        trace.candidate_mask[4 * trace.feature_count * trace.bin_count + bin] = 1;
    }
    trace.candidate_mask[1 * trace.bin_count + 0] = 1;
    trace.candidate_mask[2 * trace.bin_count + 2] = 1;
    trace.candidate_mask[5 * trace.bin_count + 0] = 1;
    trace.candidate_mask[6 * trace.bin_count + 2] = 1;
    trace.selected_candidate_indices.assign(trace.node_count, 0);
    trace.selected_candidate_indices[0] = 1;
    trace.selected_candidate_indices[1] = 0;
    trace.selected_candidate_indices[2] = 2;
    trace.selected_candidate_indices[4] = 1;
    trace.selected_candidate_indices[5] = 0;
    trace.selected_candidate_indices[6] = 2;
    return trace;
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

TEST_F(ZkboostSharedValidateSplitsPlonkTest, ProvesAndVerifiesSplitCircuit)
{
    const forestTrainingSemanticTrace training = two_class_split_trace();
    const zkboostSharedSplitView view = build_zkboost_shared_split_view(training);
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitPlonkInstance instance =
        build_zkboost_shared_split_plonk_instance(training, public_params);

    EXPECT_EQ(instance.public_inputs, encode_zkboost_shared_split_public_inputs(view));
    EXPECT_EQ(instance.summary.candidate_count, 64u);

    const plonkProof proof = prove_zkboost_shared_split_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_split_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, ProvesAndVerifiesFirstOrderRssSplitCircuit)
{
    const forestTrainingSemanticTrace training = first_order_two_class_split_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitPlonkInstance instance =
        build_zkboost_shared_split_plonk_instance(training, public_params);
    const plonkProof proof = prove_zkboost_shared_split_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_split_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, LinksHistogramsToSplits)
{
    const forestTrainingSemanticTrace training = two_class_split_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance(training, public_params);
    const zkboostSharedSplitPlonkInstance split_instance =
        build_zkboost_shared_split_plonk_instance(training, public_params);

    EXPECT_TRUE(link_zkboost_shared_histograms_to_splits(histogram_instance, split_instance));
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, HistogramLinkRejectsTamperedLeftGradient)
{
    const forestTrainingSemanticTrace training = two_class_split_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance(training, public_params);
    zkboostSharedSplitPlonkInstance split_instance =
        build_zkboost_shared_split_plonk_instance(training, public_params);

    ASSERT_FALSE(split_instance.public_inputs.empty());
    split_instance.public_inputs[0] += Fr(1);

    EXPECT_FALSE(link_zkboost_shared_histograms_to_splits(histogram_instance, split_instance));
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, VerifyRejectsTamperedSelectedBinPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitPlonkInstance instance =
        build_zkboost_shared_split_plonk_instance(two_class_split_trace(), public_params);
    const plonkProof proof = prove_zkboost_shared_split_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    const size_t selected_bins_offset = instance.summary.candidate_count * 16 +
                                        instance.summary.node_count * 3;
    ASSERT_LT(selected_bins_offset, bad_public_inputs.size());
    bad_public_inputs[selected_bins_offset] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_split_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, VerifyRejectsTamperedSplitMaskPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitPlonkInstance instance =
        build_zkboost_shared_split_plonk_instance(two_class_split_trace(), public_params);
    const plonkProof proof = prove_zkboost_shared_split_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    const size_t split_mask_offset = instance.summary.candidate_count * 16 +
                                     instance.summary.node_count * 4;
    ASSERT_LT(split_mask_offset, bad_public_inputs.size());
    bad_public_inputs[split_mask_offset] = Fr(0);

    EXPECT_FALSE(verify_zkboost_shared_split_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, LookupTableIsWitnessIndependent)
{
    const forestTrainingSemanticTrace training = two_class_split_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitView original = build_zkboost_shared_split_view(training);
    zkboostSharedSplitView changed = original;
    changed.candidate_masks[0] = Fr(0);
    const auto first = build_zkboost_shared_split_plonk_instance_from_view(
        training, original, public_params);
    const auto second = build_zkboost_shared_split_plonk_instance_from_view(
        training, changed, public_params);
    EXPECT_EQ(first.setup.verification_key.table_1_commitment,
              second.setup.verification_key.table_1_commitment);
    EXPECT_EQ(first.setup.verification_key.table_2_commitment,
              second.setup.verification_key.table_2_commitment);
    EXPECT_EQ(first.setup.verification_key.table_3_commitment,
              second.setup.verification_key.table_3_commitment);
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, RejectsIncompleteCandidateMaskBeforeProving)
{
    forestTrainingSemanticTrace training = two_class_split_trace();
    training.candidate_mask[0] = 0;
    EXPECT_THROW((void)build_zkboost_shared_split_view(training), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, RejectsInvalidLastBinCandidateBeforeProving)
{
    forestTrainingSemanticTrace training = two_class_split_trace();
    training.candidate_mask[training.bin_count - 1] = 1;
    EXPECT_THROW((void)build_zkboost_shared_split_view(training), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, ProverRejectsBadGainNumeratorWitness)
{
    const forestTrainingSemanticTrace training = two_class_split_trace();
    zkboostSharedSplitView view = build_zkboost_shared_split_view(training);
    ASSERT_FALSE(view.gain_numerators.empty());
    view.gain_numerators[1] += Fr(1);

    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitPlonkInstance instance =
        build_zkboost_shared_split_plonk_instance_from_view(training, view, public_params);

    EXPECT_THROW((void)prove_zkboost_shared_split_plonk(instance), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, ProverRejectsBadSelectorWitness)
{
    const forestTrainingSemanticTrace training = two_class_split_trace();
    zkboostSharedSplitView view = build_zkboost_shared_split_view(training);
    ASSERT_GT(view.selector.size(), 1u);
    view.selector[1] = Fr(0);

    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitPlonkInstance instance =
        build_zkboost_shared_split_plonk_instance_from_view(training, view, public_params);

    EXPECT_THROW((void)prove_zkboost_shared_split_plonk(instance), std::runtime_error);
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, ProvesMaximumDepthLeafStoppingRule)
{
    forestTrainingSemanticTrace training = build_demo_training_trace();
    training.training_inputs = {1, 0, 2, 3};
    training.candidate_mask[1] = 1;
    training.candidate_mask[2] = 1;
    training.candidate_mask[2 * training.bin_count + 1] = 1;
    training.candidate_mask[2 * training.bin_count + 2] = 1;
    training.selected_candidate_indices[2] = 1;
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitPlonkInstance instance =
        build_zkboost_shared_split_plonk_instance(training, public_params);
    const plonkProof proof = prove_zkboost_shared_split_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_split_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateSplitsPlonkTest, ProverRejectsDepthLimitMaskOnSplitNode)
{
    const forestTrainingSemanticTrace training = two_class_split_trace();
    zkboostSharedSplitView view = build_zkboost_shared_split_view(training);
    view.depth_limit_masks[0] = Fr(1);

    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedSplitPlonkInstance instance =
        build_zkboost_shared_split_plonk_instance_from_view(training, view, public_params);

    EXPECT_THROW((void)prove_zkboost_shared_split_plonk(instance), std::runtime_error);
}
