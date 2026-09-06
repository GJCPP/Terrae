#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/validate_forest_fetch.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedValidateForestFetchTest : public ::testing::Test {
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

TEST_F(ZkboostSharedValidateForestFetchTest, ProvesAndVerifiesOpeningBasedFetchProof)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);

    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedValidateForestPlonkInstance routing_instance =
        build_zkboost_shared_validate_forest_plonk_instance_from_view(
            view,
            training.bin_count,
            public_params);
    const zkboostSharedValidateForestFetchInstance fetch_instance =
        build_zkboost_shared_validate_forest_fetch_instance_from_view(
            view,
            shared_tree_ensemble_from_training_trace(training),
            public_params);

    EXPECT_EQ(fetch_instance.summary.split_field_read_count, 2u);
    EXPECT_EQ(fetch_instance.summary.path_read_count, 8u);
    EXPECT_EQ(fetch_instance.summary.leaf_value_read_count, 8u);
    EXPECT_TRUE(link_zkboost_shared_validate_forest_to_fetch(
        routing_instance,
        fetch_instance));

    const SerializedProof proof =
        prove_zkboost_shared_validate_forest_fetch(fetch_instance);

    EXPECT_TRUE(verify_zkboost_shared_validate_forest_fetch(fetch_instance, proof));
    EXPECT_FALSE(proof.bytes.empty());
}

TEST_F(ZkboostSharedValidateForestFetchTest, VerifyRejectsTamperedCommitment)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedValidateForestFetchInstance instance =
        build_zkboost_shared_validate_forest_fetch_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const SerializedProof proof = prove_zkboost_shared_validate_forest_fetch(instance);

    instance.commitments.com_t = Group1::zero();

    EXPECT_FALSE(verify_zkboost_shared_validate_forest_fetch(instance, proof));
}

TEST_F(ZkboostSharedValidateForestFetchTest, LinkRejectsTamperedLinkedInputDigest)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedValidateForestPlonkInstance routing_instance =
        build_zkboost_shared_validate_forest_plonk_instance_from_view(
            view,
            training.bin_count,
            public_params,
            zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);
    zkboostSharedValidateForestFetchInstance fetch_instance =
        build_zkboost_shared_validate_forest_fetch_instance_from_view(
            view,
            shared_tree_ensemble_from_training_trace(training),
            public_params);

    ASSERT_TRUE(link_zkboost_shared_validate_forest_to_fetch(routing_instance, fetch_instance));
    fetch_instance.linked_input_digest = "deadbeef";

    EXPECT_FALSE(link_zkboost_shared_validate_forest_to_fetch(routing_instance, fetch_instance));
}

TEST_F(ZkboostSharedValidateForestFetchTest, ProverRejectsPathFieldNotInCommittedForest)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    ASSERT_FALSE(view.fetch.path_reads.empty());

    view.fetch.path_reads[0].threshold = 0;

    const zkboostSharedValidateForestFetchInstance instance =
        build_zkboost_shared_validate_forest_fetch_instance_from_view(
            view,
            shared_tree_ensemble_from_training_trace(training),
            public_params);

    EXPECT_THROW((void)prove_zkboost_shared_validate_forest_fetch(instance), std::runtime_error);
}

TEST_F(ZkboostSharedValidateForestFetchTest, ProverRejectsLeafValueNotInCommittedForest)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    ASSERT_FALSE(view.fetch.leaf_value_reads.empty());

    view.fetch.leaf_value_reads[0].tree_score = 7;

    const zkboostSharedValidateForestFetchInstance instance =
        build_zkboost_shared_validate_forest_fetch_instance_from_view(
            view,
            shared_tree_ensemble_from_training_trace(training),
            public_params);

    EXPECT_THROW((void)prove_zkboost_shared_validate_forest_fetch(instance), std::runtime_error);
}
