#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/inference_workflow.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedInferenceWorkflowTest : public ::testing::Test {
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

forestSemanticInferenceQueryBatch two_class_subset_inference_trace()
{
    forestSemanticInferenceQueryBatch trace;
    trace.query_count = 4;
    trace.feature_count = 1;
    trace.query_tree_ids = {1, 1, 2, 2};
    trace.lifted_inputs = {3, 1, 3, 1};
    trace.sample_indices = {3, 1};
    return trace;
}

forestTrainingSemanticTrace depth_two_training_trace()
{
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    trace.max_depth = 3;
    trace.tree_count = 1;
    trace.node_count = 7;
    trace.iteration_count = 1;
    trace.class_count = 1;
    trace.query_count = 4;
    trace.feature_count = 1;
    trace.bin_count = 4;
    trace.score_encoding = fixedPointEncoding::signed_offset_binary(7, 0);

    trace.root_indices = {0};
    trace.split_node_mask = {1, 1, 1, 0, 0, 0, 0};
    trace.left_child_indices = {1, 3, 5, 3, 4, 5, 6};
    trace.right_child_indices = {2, 4, 6, 3, 4, 5, 6};
    trace.node_tree_ids = {1, 1, 1, 1, 1, 1, 1};
    trace.feature_selector = {1, 1, 1, 0, 0, 0, 0};
    trace.thresholds = {2, 1, 3, 0, 0, 0, 0};
    trace.node_leaf_mask = {0, 0, 0, 1, 1, 1, 1};
    trace.node_leaf_values = {0, 0, 0, 10, 20, 30, 40};

    trace.training_inputs = {0, 1, 2, 3};
    trace.base_labels = {1, 1, 1, 1};
    trace.leaf_indices = {3, 4, 5, 6};
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);
    trace.selected_candidate_indices.assign(trace.node_count, 0);
    return trace;
}

forestTrainingSemanticTrace depth_one_training_trace()
{
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    trace.tree_count = 1;
    trace.node_count = 4;
    trace.iteration_count = 1;
    trace.class_count = 1;
    trace.query_count = 4;
    trace.feature_count = 1;
    trace.bin_count = 4;
    trace.score_encoding = fixedPointEncoding::signed_offset_binary(7, 0);

    trace.root_indices = {0};
    trace.split_node_mask = {1, 0, 0, 0};
    trace.left_child_indices = {1, 1, 2, 3};
    trace.right_child_indices = {2, 1, 2, 3};
    trace.node_tree_ids = {1, 1, 1, 0};
    trace.feature_selector = {1, 0, 0, 0};
    trace.thresholds = {2, 0, 0, 0};
    trace.node_leaf_mask = {0, 1, 1, 0};
    trace.node_leaf_values = {0, 10, 20, 0};

    trace.training_inputs = {0, 1, 2, 3};
    trace.base_labels = {1, 1, 1, 1};
    trace.leaf_indices = {1, 1, 2, 2};
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);
    trace.selected_candidate_indices.assign(trace.node_count, 0);
    return trace;
}

forestSemanticInferenceQueryBatch depth_two_inference_trace()
{
    forestSemanticInferenceQueryBatch trace;
    trace.query_count = 4;
    trace.feature_count = 1;
    trace.query_tree_ids = {1, 1, 1, 1};
    trace.lifted_inputs = {0, 1, 2, 3};
    return trace;
}

} // namespace

TEST_F(ZkboostSharedInferenceWorkflowTest, CommitsForestAsSeparateStage)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedCommittedForest committed_forest =
        build_zkboost_shared_committed_forest(
            public_params,
            two_class_training_trace());

    EXPECT_EQ(committed_forest.summary.tree_count, 2u);
    EXPECT_EQ(committed_forest.summary.node_count, 8u);
    EXPECT_EQ(committed_forest.summary.max_depth, 1u);
    EXPECT_EQ(committed_forest.summary.split_domain_size, 2u);
    EXPECT_EQ(committed_forest.summary.leaf_domain_size, 4u);
    EXPECT_EQ(committed_forest.summary.commitment_count, 3u);
    EXPECT_GT(committed_forest.summary.commitment_bytes, 0u);
}

TEST_F(ZkboostSharedInferenceWorkflowTest, ProvesAndVerifiesStandaloneInference)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const zkboostSharedCommittedForest committed_forest =
        build_zkboost_shared_committed_forest(public_params, training);
    const zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance_from_committed_forest(
            training,
            inference,
            committed_forest,
            public_params);

    EXPECT_EQ(instance.summary.checked_paths, 8u);
    EXPECT_EQ(instance.prediction.accuracy_numerator, 4u);
    EXPECT_EQ(instance.prediction.accuracy_denominator, 4u);
    EXPECT_FALSE(instance.inference_plonk.public_inputs.empty());
    EXPECT_TRUE(verify_zkboost_shared_inference_links(instance));

    const zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    EXPECT_TRUE(verify_zkboost_shared_inference(instance, proof));
    EXPECT_GT(serialize_plonk_proof(proof.inference_plonk).bytes.size(), 0u);
    EXPECT_GT(serialized_zkboost_shared_inference_proof_bytes(proof), 0u);
}

TEST_F(ZkboostSharedInferenceWorkflowTest, ProvesAndVerifiesStandaloneInferenceWithPreprocessedRoutingInputs)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            training,
            inference,
            public_params,
            zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);

    EXPECT_EQ(instance.inference_plonk.public_input_mode,
              zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);
    EXPECT_TRUE(instance.inference_plonk.public_inputs.empty());
    EXPECT_TRUE(verify_zkboost_shared_inference_links(instance));

    const zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    EXPECT_TRUE(verify_zkboost_shared_inference(instance, proof));
    EXPECT_GT(serialized_zkboost_shared_inference_proof_bytes(proof), 0u);
}

TEST_F(ZkboostSharedInferenceWorkflowTest, UsesSeparateCircuitAndFetchOpeningBatches)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    ASSERT_FALSE(proof.cross_session_opening_batch.bytes.empty());
    ASSERT_FALSE(proof.forest_fetch.bytes.empty());
    EXPECT_TRUE(verify_zkboost_shared_inference(instance, proof));

    zkboostSharedInferenceProof tampered = proof;
    ASSERT_FALSE(tampered.forest_fetch.bytes.empty());
    tampered.forest_fetch.bytes.back() ^= 0x01u;
    EXPECT_FALSE(verify_zkboost_shared_inference(instance, tampered));
}

TEST_F(ZkboostSharedInferenceWorkflowTest, PreprocessedVerifierRejectsTamperedRoutingProof)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params,
            zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);
    const zkboostSharedInferenceVerifierBundle verifier_bundle =
        prepare_zkboost_shared_inference_verifier_bundle(instance);
    zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    EXPECT_TRUE(verify_zkboost_shared_inference_preprocessed(verifier_bundle, proof));

    proof.inference_plonk.a_eval += Fr(1);
    EXPECT_FALSE(verify_zkboost_shared_inference_preprocessed(verifier_bundle, proof));
}

TEST_F(ZkboostSharedInferenceWorkflowTest, PreprocessedVerifierRejectsTamperedPredictionProof)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params,
            zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);
    const zkboostSharedInferenceVerifierBundle verifier_bundle =
        prepare_zkboost_shared_inference_verifier_bundle(instance);
    zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    EXPECT_TRUE(verify_zkboost_shared_inference_preprocessed(verifier_bundle, proof));

    proof.inference_plonk.a_eval += Fr(1);
    EXPECT_FALSE(verify_zkboost_shared_inference_preprocessed(verifier_bundle, proof));
}

TEST_F(ZkboostSharedInferenceWorkflowTest, LinksRejectTamperedPredictionTreeScores)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);

    instance.prediction_view.tree_scores.at(0) += 1;
    EXPECT_FALSE(verify_zkboost_shared_inference_links(instance));
}

TEST_F(ZkboostSharedInferenceWorkflowTest, VerifyRejectsTamperedPredictionPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    instance.inference_plonk.public_inputs.back() += Fr(1);
    EXPECT_FALSE(verify_zkboost_shared_inference(instance, proof));
}

TEST_F(ZkboostSharedInferenceWorkflowTest, InferenceSrsRequirementCountsFetchOpeningPoints)
{
    const zkboostSharedKzgSrsRequirement requirement =
        required_zkboost_shared_inference_srs_requirement(
            two_class_training_trace(),
            two_class_inference_trace());

    EXPECT_GE(requirement.deg2_plus_1, 5u);
    EXPECT_GT(requirement.deg1_plus_1, requirement.deg2_plus_1);
}

TEST_F(ZkboostSharedInferenceWorkflowTest, VerifyRejectsTamperedRoutingPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    ASSERT_FALSE(instance.inference_plonk.public_inputs.empty());
    instance.inference_plonk.public_inputs[0] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_inference(instance, proof));
}

TEST_F(ZkboostSharedInferenceWorkflowTest, VerifyRejectsTamperedFetchCommitment)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    instance.fetch.commitments.com_w = Group1::zero();

    EXPECT_FALSE(verify_zkboost_shared_inference(instance, proof));
}

TEST_F(ZkboostSharedInferenceWorkflowTest, ProvesAndVerifiesSubsetInferenceBatch)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedInferenceInstance instance =
        build_zkboost_shared_inference_instance(
            two_class_training_trace(),
            two_class_subset_inference_trace(),
            public_params);

    EXPECT_EQ(instance.summary.sample_count, 2u);
    EXPECT_EQ(instance.summary.checked_paths, 4u);
    EXPECT_EQ(instance.prediction.accuracy_numerator, 2u);
    EXPECT_EQ(instance.prediction.accuracy_denominator, 2u);
    EXPECT_TRUE(verify_zkboost_shared_inference_links(instance));

    const zkboostSharedInferenceProof proof =
        prove_zkboost_shared_inference(instance);

    EXPECT_TRUE(verify_zkboost_shared_inference(instance, proof));
}

TEST_F(ZkboostSharedInferenceWorkflowTest, InferenceProofBytesStayFlatAcrossDepth)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);

    const zkboostSharedInferenceInstance depth_one =
        build_zkboost_shared_inference_instance(
            depth_one_training_trace(),
            depth_two_inference_trace(),
            public_params);
    const zkboostSharedInferenceInstance depth_two =
        build_zkboost_shared_inference_instance(
            depth_two_training_trace(),
            depth_two_inference_trace(),
            public_params);

    const zkboostSharedInferenceProof depth_one_proof =
        prove_zkboost_shared_inference(depth_one);
    const zkboostSharedInferenceProof depth_two_proof =
        prove_zkboost_shared_inference(depth_two);

    EXPECT_EQ(serialized_zkboost_shared_inference_proof_bytes(depth_one_proof),
              serialized_zkboost_shared_inference_proof_bytes(depth_two_proof));
    EXPECT_TRUE(verify_zkboost_shared_inference(depth_one, depth_one_proof));
    EXPECT_TRUE(verify_zkboost_shared_inference(depth_two, depth_two_proof));
}
