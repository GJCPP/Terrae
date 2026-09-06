#include "bls12_381_wrapper.hpp"
#include "forest_training_trace_adapter.hpp"
#include "test_kzg_srs.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

std::vector<Fr> evals_on_domain(const poly& polynomial, size_t domain_size)
{
    return polynomial.evaluations_on_domain(domain_size);
}

} // namespace

class ForestTrainingTraceAdapterTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTrainingTraceAdapterTest, AssemblesTrainingInputsFromSemanticTrace)
{
    const forestTrainingSemanticTrace trace = build_demo_training_trace();
    const forestTrainingProofInputs inputs = assemble_forest_training_inputs_from_trace(trace);

    EXPECT_EQ(inputs.tree_count, 1u);
    EXPECT_EQ(inputs.node_count, 4u);
    EXPECT_EQ(inputs.feature_count, 1u);
    EXPECT_EQ(inputs.query_count, 4u);
    EXPECT_EQ(inputs.max_depth, 2u);
    EXPECT_EQ(evals_on_domain(inputs.node_depths, 4),
              std::vector<Fr>({fr_from_int(1), fr_from_int(2), fr_from_int(2), fr_from_int(0)}));

    EXPECT_EQ(evals_on_domain(inputs.query_tree_ids, 4),
              std::vector<Fr>({fr_from_int(1), fr_from_int(1), fr_from_int(1), fr_from_int(1)}));
    EXPECT_EQ(evals_on_domain(inputs.lifted_inputs, 4),
              std::vector<Fr>({fr_from_int(1), fr_from_int(0), fr_from_int(1), fr_from_int(1)}));
    EXPECT_EQ(evals_on_domain(inputs.leaf_indices, 4),
              std::vector<Fr>({fr_from_int(2), fr_from_int(1), fr_from_int(2), fr_from_int(2)}));
    EXPECT_EQ(evals_on_domain(inputs.lower_bounds, 4),
              std::vector<Fr>({fr_from_int(0), fr_from_int(0), fr_from_int(1), fr_from_int(0)}));
    EXPECT_EQ(evals_on_domain(inputs.upper_bounds, 4),
              std::vector<Fr>({fr_from_int(4), fr_from_int(1), fr_from_int(4), fr_from_int(0)}));
    EXPECT_EQ(evals_on_domain(inputs.leaf_lower_bounds, 4),
              std::vector<Fr>({fr_from_int(1), fr_from_int(0), fr_from_int(1), fr_from_int(1)}));
    EXPECT_EQ(evals_on_domain(inputs.leaf_upper_bounds, 4),
              std::vector<Fr>({fr_from_int(4), fr_from_int(1), fr_from_int(4), fr_from_int(4)}));
    EXPECT_EQ(evals_on_domain(inputs.flattened_bucket_indices, 4),
              std::vector<Fr>({fr_from_int(9), fr_from_int(4), fr_from_int(9), fr_from_int(9)}));
    EXPECT_EQ(evals_on_domain(inputs.selected_candidate_indices, 4),
              std::vector<Fr>({fr_from_int(0), fr_from_int(0), fr_from_int(0), fr_from_int(0)}));
}

TEST_F(ForestTrainingTraceAdapterTest, AssemblesFirstOrderRssInputsAndProves)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace trace = build_demo_first_order_training_trace();
    const forestTrainingProofInputs training_inputs =
        assemble_forest_training_inputs_from_trace(trace);

    EXPECT_EQ(training_inputs.loss_mode, forestTrainingLoss::FirstOrderRss);
    EXPECT_EQ(evals_on_domain(training_inputs.pseudo_residuals, 4),
              std::vector<Fr>(4, Fr(0)));
    EXPECT_EQ(evals_on_domain(training_inputs.diagonal_hessians, 4),
              std::vector<Fr>(4, fr_from_int(4)));

    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(
            trace, build_demo_inference_query_trace());
    const forestTrainingThenInferencePreparedRun prepared =
        prove_and_prepare_training_then_inference_with_runtime(
            public_params, inputs, "trace_adapter.first_order_training_then_inference");
    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, prepared.verifier_bundle, "trace_adapter.first_order_training_then_inference");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestTrainingTraceAdapterTest, AssemblesCombinedInputsAndProves)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace trace = build_demo_training_trace();
    const forestSemanticInferenceQueryBatch inference_trace = build_demo_inference_query_trace();

    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(trace, inference_trace);
    const forestTrainingThenInferencePreparedRun prepared =
        prove_and_prepare_training_then_inference_with_runtime(
            public_params, inputs, "trace_adapter.training_then_inference");
    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, prepared.verifier_bundle, "trace_adapter.training_then_inference");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestTrainingTraceAdapterTest, WritesCaseFromTrace)
{
    const forestTrainingSemanticTrace trace = build_demo_training_trace();
    const forestSemanticInferenceQueryBatch inference_trace = build_demo_inference_query_trace();

    const std::string path = "/tmp/forest_training_trace_adapter_test.case";
    write_forest_training_then_inference_case_from_trace(path, trace, inference_trace);
    const forestTrainingThenInferenceInputs loaded =
        read_forest_training_then_inference_case(path);

    EXPECT_EQ(loaded.training_inputs.tree_count, 1u);
    EXPECT_EQ(loaded.training_inputs.query_count, 4u);
    EXPECT_EQ(loaded.inference_queries.query_count, 4u);
    EXPECT_EQ(evals_on_domain(loaded.training_inputs.thresholds, 4),
              std::vector<Fr>({fr_from_int(1), fr_from_int(0), fr_from_int(0), fr_from_int(0)}));

    std::remove(path.c_str());
}

TEST_F(ForestTrainingTraceAdapterTest, RejectsNonOneHotBaseLabels)
{
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    trace.base_labels = {0, 1, 1, 1};

    EXPECT_THROW((void)assemble_forest_training_inputs_from_trace(trace), std::runtime_error);
}
