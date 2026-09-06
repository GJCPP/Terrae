#include "algebra_utils.hpp"
#include "fixed_point.hpp"
#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/validate_boosting_state.hpp"
#include "zkboost/shared_trace/validate_boosting_state_plonk.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedValidateBoostingStateTest : public ::testing::Test {
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

Fr fr_size(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

std::vector<Fr> fr_i64s(const std::vector<std::int64_t>& values)
{
    std::vector<Fr> out;
    out.reserve(values.size());
    for (std::int64_t value : values) {
        out.push_back(fr_i64(value));
    }
    return out;
}

std::vector<Fr> repeated_fr(const Fr& value, size_t count)
{
    return std::vector<Fr>(count, value);
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

forestTrainingSemanticTrace first_order_two_class_training_trace()
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.loss_mode = forestTrainingLoss::FirstOrderRss;
    trace.l2_regularization = 0;
    return trace;
}

forestTrainingSemanticTrace two_class_varying_denominator_trace()
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.tree_count = 4;
    trace.node_count = 16;
    trace.iteration_count = 2;

    trace.root_indices = {0, 4, 8, 12};
    trace.split_node_mask = {
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
    };
    trace.left_child_indices = {
        1, 1, 2, 3,
        5, 5, 6, 7,
        9, 9, 10, 11,
        13, 13, 14, 15,
    };
    trace.right_child_indices = {
        2, 1, 2, 3,
        6, 5, 6, 7,
        10, 9, 10, 11,
        14, 13, 14, 15,
    };
    trace.node_tree_ids = {
        1, 1, 1, 0,
        2, 2, 2, 0,
        3, 3, 3, 0,
        4, 4, 4, 0,
    };
    trace.feature_selector = {
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
    };
    trace.thresholds = {
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
    };
    trace.node_leaf_mask = {
        0, 1, 1, 0,
        0, 1, 1, 0,
        0, 1, 1, 0,
        0, 1, 1, 0,
    };
    trace.node_leaf_values = {
        0, 1, 5, 0,
        0, 1, 3, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    trace.leaf_indices = {
        1, 1, 2, 2,
        5, 5, 6, 6,
        9, 9, 10, 10,
        13, 13, 14, 14,
    };
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);
    trace.selected_candidate_indices.assign(trace.node_count, 0);
    trace.selected_candidate_indices[0] = 1;
    trace.selected_candidate_indices[4] = 1;
    trace.selected_candidate_indices[8] = 1;
    trace.selected_candidate_indices[12] = 1;
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

} // namespace

TEST_F(ZkboostSharedValidateBoostingStateTest, DerivesTwoClassBoostingStateFromTrace)
{
    const forestTrainingSemanticTrace trace = two_class_training_trace();
    const zkboostSharedBoostingStateView view =
        build_zkboost_shared_boosting_state_view(trace);

    EXPECT_EQ(view.summary.iteration_count, 1u);
    EXPECT_EQ(view.summary.class_count, 2u);
    EXPECT_EQ(view.summary.sample_count, 4u);
    EXPECT_EQ(view.summary.tree_count, 2u);
    EXPECT_EQ(view.summary.cube_size, 8u);
    EXPECT_EQ(view.summary.softmax_scale, 4u);

    EXPECT_EQ(view.tree_scores, fr_i64s({5, 5, -5, -5, -5, -5, 5, 5}));
    EXPECT_EQ(view.cumulative_scores, repeated_fr(fr_i64(0), 8));
    EXPECT_EQ(view.encoded_cumulative_scores, repeated_fr(fr_size(32), 8));
    EXPECT_EQ(view.softmax_shifts, repeated_fr(fr_size(32), 4));
    EXPECT_EQ(view.shift_selector, fr_i64s({1, 1, 1, 1, 0, 0, 0, 0}));
    EXPECT_EQ(view.lifted_softmax_shifts, repeated_fr(fr_size(32), 8));
    EXPECT_EQ(view.encoded_shifted_scores, repeated_fr(fr_size(32), 8));
    EXPECT_EQ(view.softmax_exp_values, repeated_fr(fr_i64(1), 8));
    EXPECT_EQ(view.softmax_denominators, repeated_fr(fr_i64(2), 4));
    EXPECT_EQ(view.lifted_softmax_denominators, repeated_fr(fr_i64(2), 8));
    EXPECT_EQ(view.softmax_probabilities, repeated_fr(fr_size(34), 8));
    EXPECT_EQ(view.softmax_remainders, repeated_fr(fr_i64(0), 8));
    EXPECT_EQ(view.pseudo_residuals, fr_i64s({-2, -2, 2, 2, 2, 2, -2, -2}));
    EXPECT_EQ(view.diagonal_hessians, repeated_fr(fr_i64(1), 8));
    EXPECT_EQ(view.diagonal_hessian_remainders, repeated_fr(fr_i64(0), 8));
    EXPECT_EQ(view.encoded_diagonal_hessians, repeated_fr(fr_size(9), 8));
}

TEST_F(ZkboostSharedValidateBoostingStateTest, ProvesFirstOrderResidualAndCountSurrogateCircuit)
{
    const forestTrainingSemanticTrace training = first_order_two_class_training_trace();
    const zkboostSharedBoostingStateView view =
        build_zkboost_shared_boosting_state_view(training);

    EXPECT_EQ(view.pseudo_residuals, fr_i64s({2, 2, -2, -2, -2, -2, 2, 2}));
    EXPECT_EQ(view.diagonal_hessians, repeated_fr(fr_i64(4), 8));
    EXPECT_EQ(view.diagonal_hessian_remainders, repeated_fr(fr_i64(0), 8));
    EXPECT_EQ(view.encoded_diagonal_hessians, repeated_fr(fr_size(12), 8));

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedBoostingStatePlonkInstance instance =
        build_zkboost_shared_boosting_state_plonk_instance(training, public_params);
    const plonkProof proof = prove_zkboost_shared_boosting_state_plonk(instance);
    EXPECT_TRUE(verify_zkboost_shared_boosting_state_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateBoostingStateTest, LiftsSoftmaxDenominatorsInCubeOrder)
{
    const forestTrainingSemanticTrace trace = two_class_varying_denominator_trace();
    const zkboostSharedBoostingStateView view =
        build_zkboost_shared_boosting_state_view(trace);

    ASSERT_EQ(view.softmax_denominators.size(), 8u);
    ASSERT_EQ(view.lifted_softmax_denominators.size(), 16u);
    EXPECT_NE(view.softmax_denominators[4], view.softmax_denominators[6]);

    EXPECT_EQ(view.lifted_softmax_denominators[8], view.softmax_denominators[4]);
    EXPECT_EQ(view.lifted_softmax_denominators[9], view.softmax_denominators[5]);
    EXPECT_EQ(view.lifted_softmax_denominators[10], view.softmax_denominators[6]);
    EXPECT_EQ(view.lifted_softmax_denominators[11], view.softmax_denominators[7]);
    EXPECT_EQ(view.lifted_softmax_denominators[12], view.softmax_denominators[4]);
    EXPECT_EQ(view.lifted_softmax_denominators[13], view.softmax_denominators[5]);
    EXPECT_EQ(view.lifted_softmax_denominators[14], view.softmax_denominators[6]);
    EXPECT_EQ(view.lifted_softmax_denominators[15], view.softmax_denominators[7]);
}

TEST_F(ZkboostSharedValidateBoostingStateTest, RejectsBadTreeIterationLayout)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.tree_count = 3;

    EXPECT_THROW(
        (void)validate_zkboost_shared_boosting_state(trace),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateBoostingStateTest, RejectsNonOneHotLabels)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.base_labels[0] = 0;

    EXPECT_THROW(
        (void)validate_zkboost_shared_boosting_state(trace),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateBoostingStateTest, RejectsLeafIndexOutOfRange)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.leaf_indices[0] = 99;

    EXPECT_THROW(
        (void)validate_zkboost_shared_boosting_state(trace),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateBoostingStateTest, ProvesAndVerifiesBoostingStateCircuit)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const zkboostSharedBoostingStateView view =
        build_zkboost_shared_boosting_state_view(training);
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedBoostingStatePlonkInstance instance =
        build_zkboost_shared_boosting_state_plonk_instance(training, public_params);

    EXPECT_EQ(instance.public_inputs, encode_zkboost_shared_boosting_state_public_inputs(view));
    EXPECT_EQ(instance.summary.cube_size, 8u);

    const plonkProof proof = prove_zkboost_shared_boosting_state_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_boosting_state_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateBoostingStateTest, ProvesAndVerifiesMultiIterationBoostingStateCircuit)
{
    const forestTrainingSemanticTrace training = two_class_varying_denominator_trace();
    const zkboostSharedBoostingStateView view =
        build_zkboost_shared_boosting_state_view(training);
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedBoostingStatePlonkInstance instance =
        build_zkboost_shared_boosting_state_plonk_instance(training, public_params);

    EXPECT_EQ(instance.public_inputs, encode_zkboost_shared_boosting_state_public_inputs(view));
    EXPECT_EQ(instance.summary.iteration_count, 2u);

    const plonkProof proof = prove_zkboost_shared_boosting_state_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_boosting_state_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateBoostingStateTest, LookupTableIsWitnessIndependent)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedBoostingStateView original =
        build_zkboost_shared_boosting_state_view(training);
    zkboostSharedBoostingStateView changed = original;
    changed.base_labels[0] = Fr(0);
    changed.base_labels[training.query_count] = Fr(1);
    const auto first = build_zkboost_shared_boosting_state_plonk_instance_from_view(
        training, original, public_params);
    const auto second = build_zkboost_shared_boosting_state_plonk_instance_from_view(
        training, changed, public_params);
    EXPECT_EQ(first.setup.verification_key.table_1_commitment,
              second.setup.verification_key.table_1_commitment);
    EXPECT_EQ(first.setup.verification_key.table_2_commitment,
              second.setup.verification_key.table_2_commitment);
    EXPECT_EQ(first.setup.verification_key.table_3_commitment,
              second.setup.verification_key.table_3_commitment);
}

TEST_F(ZkboostSharedValidateBoostingStateTest, LinksValidateForestTreeScoresToBoostingState)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance forest_instance =
        build_zkboost_shared_validate_forest_plonk_instance(training, inference, public_params);
    const zkboostSharedBoostingStatePlonkInstance boosting_instance =
        build_zkboost_shared_boosting_state_plonk_instance(training, public_params);

    EXPECT_TRUE(link_zkboost_shared_validate_forest_to_boosting_state(
        forest_instance,
        boosting_instance));
}

TEST_F(ZkboostSharedValidateBoostingStateTest, LinkRejectsMismatchedTreeScorePublicInput)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance forest_instance =
        build_zkboost_shared_validate_forest_plonk_instance(training, inference, public_params);
    zkboostSharedBoostingStatePlonkInstance boosting_instance =
        build_zkboost_shared_boosting_state_plonk_instance(training, public_params);

    ASSERT_FALSE(boosting_instance.public_inputs.empty());
    boosting_instance.public_inputs[0] += Fr(1);

    EXPECT_FALSE(link_zkboost_shared_validate_forest_to_boosting_state(
        forest_instance,
        boosting_instance));
}

TEST_F(ZkboostSharedValidateBoostingStateTest, VerifyRejectsTamperedResidualPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedBoostingStatePlonkInstance instance =
        build_zkboost_shared_boosting_state_plonk_instance(two_class_training_trace(), public_params);
    const plonkProof proof = prove_zkboost_shared_boosting_state_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    const size_t residual_offset = instance.summary.cube_size;
    ASSERT_LT(residual_offset, bad_public_inputs.size());
    bad_public_inputs[residual_offset] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_boosting_state_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateBoostingStateTest, ProverRejectsBadCumulativeWitness)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    zkboostSharedBoostingStateView view =
        build_zkboost_shared_boosting_state_view(training);
    ASSERT_GT(view.cumulative_scores.size(), 4u);
    view.cumulative_scores[4] += Fr(1);

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedBoostingStatePlonkInstance instance =
        build_zkboost_shared_boosting_state_plonk_instance_from_view(
            training,
            view,
            public_params);

    EXPECT_THROW((void)prove_zkboost_shared_boosting_state_plonk(instance), std::runtime_error);
}
