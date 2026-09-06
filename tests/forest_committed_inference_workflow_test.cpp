#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "forest_committed_inference_workflow.hpp"
#include "forest_training_trace_adapter.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

namespace {

forestTrainingSemanticTrace demo_training_trace()
{
    forestTrainingSemanticTrace trace;
    trace.tree_count = 1;
    trace.node_count = 4;
    trace.feature_count = 1;
    trace.bin_count = 4;
    trace.iteration_count = 1;
    trace.class_count = 1;
    trace.query_count = 4;
    trace.max_depth = 2;
    trace.input_bit_bound = 3;
    trace.full_upper_bound = 4;
    trace.structure_bit_bound = 3;
    trace.gradient_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    trace.gain_numerator_encoding = fixedPointEncoding::signed_offset_binary(10, 0);
    trace.quantized_gain_encoding = fixedPointEncoding::signed_offset_binary(10, 2);
    trace.leaf_value_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    trace.score_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    trace.softmax_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    trace.hessian_bit_bound = 3;
    trace.gain_denominator_bit_bound = 3;
    trace.leaf_denominator_bit_bound = 2;
    trace.gain_gap_bit_bound = 2;
    trace.loss_mode = forestTrainingLoss::SecondOrderXgboost;
    trace.l2_regularization = 1;

    trace.root_indices = {0};
    trace.split_node_mask = {1, 0, 0, 0};
    trace.left_child_indices = {1, 0, 0, 0};
    trace.right_child_indices = {3, 0, 0, 0};
    trace.node_tree_ids = {1, 1, 0, 1};
    trace.feature_selector = {1, 0, 0, 0};
    trace.thresholds = {2, 0, 0, 0};
    trace.node_leaf_mask = {0, 1, 0, 1};
    trace.node_leaf_values = {0, 5, 0, 9};

    trace.training_inputs = {1, 0, 2, 3};
    trace.base_labels = {1, 1, 1, 1};
    trace.leaf_indices = {1, 1, 3, 3};
    trace.candidate_mask = {
        0, 0, 1, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    trace.selected_candidate_indices = {2, 0, 0, 0};
    return trace;
}

forestSemanticInferenceQueryBatch demo_inference_queries()
{
    forestSemanticInferenceQueryBatch out;
    out.query_count = 4;
    out.feature_count = 1;
    out.query_tree_ids = {1, 1, 1, 1};
    out.lifted_inputs = {0, 1, 2, 3};
    return out;
}

} // namespace

class ForestCommittedInferenceWorkflowTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestCommittedInferenceWorkflowTest, HelperSupportsProveAndVerifyAgainstForestCommitment)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestCommittedInferenceInputs inputs =
        assemble_forest_committed_inference_inputs_from_trace(demo_training_trace(), demo_inference_queries());
    const std::vector<Fr> split_mask_evals = inputs.committed_forest.split_node_mask.evaluations_on_domain(
        inputs.committed_forest.node_count);
    ASSERT_EQ(split_mask_evals.size(), inputs.committed_forest.node_count);
    EXPECT_EQ(split_mask_evals[0], Fr(1));
    const forestCommittedInferencePublicInputs public_inputs =
        forest_committed_inference_public_inputs_from_inputs(public_params, inputs);

    const ProofArtifact artifact =
        prove_committed_forest_inference_with_runtime(public_params, inputs, "workflow.committed_inference");
    const VerificationResult result =
        verify_committed_forest_inference_with_runtime(public_params,
                                                       artifact,
                                                       public_inputs,
                                                       "workflow.committed_inference");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestCommittedInferenceWorkflowTest, VerifierAcceptsBothRawQueriesAndPreprocessedQueryCommitments)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestCommittedInferenceInputs inputs =
        assemble_forest_committed_inference_inputs_from_trace(demo_training_trace(), demo_inference_queries());
    const forestCommittedInferencePublicInputs preprocessed_public_inputs =
        forest_committed_inference_public_inputs_from_inputs(public_params, inputs);
    forestCommittedInferencePublicInputs raw_public_inputs = preprocessed_public_inputs;
    raw_public_inputs.query_commitments.reset();

    const ProofArtifact artifact =
        prove_committed_forest_inference_with_runtime(public_params, inputs, "workflow.committed_inference");
    const VerificationResult raw_result =
        verify_committed_forest_inference_with_runtime(public_params,
                                                       artifact,
                                                       raw_public_inputs,
                                                       "workflow.committed_inference");
    const VerificationResult preprocessed_result =
        verify_committed_forest_inference_with_runtime(public_params,
                                                       artifact,
                                                       preprocessed_public_inputs,
                                                       "workflow.committed_inference");

    EXPECT_TRUE(raw_result.accepted);
    EXPECT_TRUE(raw_result.proof_fully_consumed);
    EXPECT_TRUE(preprocessed_result.accepted);
    EXPECT_TRUE(preprocessed_result.proof_fully_consumed);
}

TEST_F(ForestCommittedInferenceWorkflowTest, RejectsTooSmallSrsBeforeProving)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(2, 2);
    const forestCommittedInferenceInputs inputs =
        assemble_forest_committed_inference_inputs_from_trace(demo_training_trace(), demo_inference_queries());

    EXPECT_THROW(
        (void)prove_committed_forest_inference_with_runtime(public_params, inputs, "workflow.committed_inference"),
        std::runtime_error);
}
