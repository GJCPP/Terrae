#include "algebra_utils.hpp"
#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"
#include "zkboost/shared_trace/validate_histograms_plonk.hpp"
#include "zkboost/shared_trace/validate_leaf_values.hpp"
#include "zkboost/shared_trace/validate_leaf_values_plonk.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedValidateLeafValuesPlonkTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

namespace {

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

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, ProvesAndVerifiesLeafValueCircuit)
{
    const forestTrainingSemanticTrace training = two_class_valid_leaf_trace();
    const zkboostSharedLeafValueView view =
        build_zkboost_shared_leaf_value_view(training);
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedLeafValuePlonkInstance instance =
        build_zkboost_shared_leaf_value_plonk_instance(training, public_params);

    EXPECT_EQ(instance.public_inputs, encode_zkboost_shared_leaf_value_public_inputs(view));
    EXPECT_EQ(instance.summary.node_count, 8u);

    const plonkProof proof = prove_zkboost_shared_leaf_value_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_leaf_value_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, ProvesAndVerifiesFirstOrderMeanResidualCircuit)
{
    const forestTrainingSemanticTrace training = first_order_two_class_leaf_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedLeafValuePlonkInstance instance =
        build_zkboost_shared_leaf_value_plonk_instance(training, public_params);
    const plonkProof proof = prove_zkboost_shared_leaf_value_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_leaf_value_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, LookupTableIsWitnessIndependent)
{
    const forestTrainingSemanticTrace training = two_class_valid_leaf_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedLeafValueView original = build_zkboost_shared_leaf_value_view(training);
    zkboostSharedLeafValueView changed = original;
    changed.committed_leaf_values[1] += Fr(1);
    const auto first = build_zkboost_shared_leaf_value_plonk_instance_from_view(
        training, original, public_params);
    const auto second = build_zkboost_shared_leaf_value_plonk_instance_from_view(
        training, changed, public_params);
    EXPECT_EQ(first.setup.verification_key.table_1_commitment,
              second.setup.verification_key.table_1_commitment);
    EXPECT_EQ(first.setup.verification_key.table_2_commitment,
              second.setup.verification_key.table_2_commitment);
    EXPECT_EQ(first.setup.verification_key.table_3_commitment,
              second.setup.verification_key.table_3_commitment);
}

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, LinksHistogramsAndForestToLeafValues)
{
    const forestTrainingSemanticTrace training = two_class_valid_leaf_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance(training, public_params);
    const zkboostSharedValidateForestPlonkInstance forest_instance =
        build_zkboost_shared_validate_forest_plonk_instance(
            training,
            two_class_inference_trace(),
            public_params);
    const zkboostSharedLeafValuePlonkInstance leaf_value_instance =
        build_zkboost_shared_leaf_value_plonk_instance(training, public_params);

    EXPECT_TRUE(link_zkboost_shared_histograms_to_leaf_values(
        histogram_instance,
        leaf_value_instance));
    EXPECT_TRUE(link_zkboost_shared_validate_forest_to_leaf_values(
        forest_instance,
        leaf_value_instance));
}

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, HistogramLinkRejectsTamperedGradient)
{
    const forestTrainingSemanticTrace training = two_class_valid_leaf_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedHistogramPlonkInstance histogram_instance =
        build_zkboost_shared_histogram_plonk_instance(training, public_params);
    zkboostSharedLeafValuePlonkInstance leaf_value_instance =
        build_zkboost_shared_leaf_value_plonk_instance(training, public_params);

    ASSERT_FALSE(leaf_value_instance.public_inputs.empty());
    leaf_value_instance.public_inputs[0] += Fr(1);

    EXPECT_FALSE(link_zkboost_shared_histograms_to_leaf_values(
        histogram_instance,
        leaf_value_instance));
}

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, ForestLinkRejectsTamperedLeafValue)
{
    const forestTrainingSemanticTrace training = two_class_valid_leaf_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance forest_instance =
        build_zkboost_shared_validate_forest_plonk_instance(
            training,
            two_class_inference_trace(),
            public_params);
    zkboostSharedLeafValuePlonkInstance leaf_value_instance =
        build_zkboost_shared_leaf_value_plonk_instance(training, public_params);

    const size_t decoded_leaf_offset = 3 * leaf_value_instance.summary.node_count;
    ASSERT_LT(decoded_leaf_offset + 1, leaf_value_instance.public_inputs.size());
    leaf_value_instance.public_inputs[decoded_leaf_offset + 1] += Fr(1);

    EXPECT_FALSE(link_zkboost_shared_validate_forest_to_leaf_values(
        forest_instance,
        leaf_value_instance));
}

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, VerifyRejectsTamperedDecodedLeafPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedLeafValuePlonkInstance instance =
        build_zkboost_shared_leaf_value_plonk_instance(two_class_valid_leaf_trace(), public_params);
    const plonkProof proof = prove_zkboost_shared_leaf_value_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    const size_t decoded_leaf_offset = 3 * instance.summary.node_count;
    ASSERT_LT(decoded_leaf_offset + 1, bad_public_inputs.size());
    bad_public_inputs[decoded_leaf_offset + 1] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_leaf_value_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, ProverRejectsBadDenominatorWitness)
{
    const forestTrainingSemanticTrace training = two_class_valid_leaf_trace();
    zkboostSharedLeafValueView view = build_zkboost_shared_leaf_value_view(training);
    ASSERT_GT(view.leaf_denominators.size(), 1u);
    view.leaf_denominators[1] += Fr(1);

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedLeafValuePlonkInstance instance =
        build_zkboost_shared_leaf_value_plonk_instance_from_view(
            training,
            view,
            public_params);

    EXPECT_THROW((void)prove_zkboost_shared_leaf_value_plonk(instance), std::runtime_error);
}

TEST_F(ZkboostSharedValidateLeafValuesPlonkTest, ProverRejectsBadRemainderWitness)
{
    const forestTrainingSemanticTrace training = two_class_valid_leaf_trace();
    zkboostSharedLeafValueView view = build_zkboost_shared_leaf_value_view(training);
    ASSERT_GT(view.leaf_remainders.size(), 1u);
    view.leaf_remainders[1] += Fr(1);

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedLeafValuePlonkInstance instance =
        build_zkboost_shared_leaf_value_plonk_instance_from_view(
            training,
            view,
            public_params);

    EXPECT_THROW((void)prove_zkboost_shared_leaf_value_plonk(instance), std::runtime_error);
}
