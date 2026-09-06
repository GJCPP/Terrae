#include "bls12_381_wrapper.hpp"
#include "forest_training_trace_io.hpp"
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

class ForestTrainingTraceIoTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTrainingTraceIoTest, SummarizesAndRoundTripsSemanticTraceCase)
{
    const forestTrainingSemanticTraceCase trace_case = build_demo_training_trace_case();
    const forestTrainingSemanticTraceCaseSummary summary =
        summarize_forest_training_semantic_trace_case(trace_case);
    const std::string summary_text =
        forest_training_semantic_trace_case_summary_description(summary);
    const std::string contract_text =
        forest_training_semantic_trace_contract_description();

    EXPECT_EQ(summary.tree_count, 1u);
    EXPECT_EQ(summary.node_count, 4u);
    EXPECT_EQ(summary.bin_count, 4u);
    EXPECT_EQ(summary.class_count, 1u);
    EXPECT_EQ(summary.exported_leaf_row_count, 2u);
    EXPECT_EQ(summary.trace_leaf_assignment_count, 4u);
    EXPECT_EQ(summary.trace_candidate_count, 16u);
    EXPECT_EQ(summary.active_split_node_count, 1u);
    EXPECT_EQ(summary.active_candidate_count, 1u);
    EXPECT_EQ(summary.leaf_node_count, 2u);
    EXPECT_EQ(summary.derived_training_query_row_count, 4u);
    EXPECT_EQ(summary.derived_bucket_address_count, 4u);
    EXPECT_EQ(summary.derived_boosting_entry_count, 4u);
    EXPECT_NE(summary_text.find("Semantic trace summary"), std::string::npos);
    EXPECT_NE(summary_text.find("active split nodes: 1"), std::string::npos);
    EXPECT_NE(contract_text.find("Required semantic-trace fields"), std::string::npos);
    EXPECT_NE(contract_text.find("Derived internally by the repository"), std::string::npos);

    const std::string path = "/tmp/forest_training_trace_io_test.trace";
    write_forest_training_semantic_trace_case(path, trace_case);
    const forestTrainingSemanticTraceCase loaded =
        read_forest_training_semantic_trace_case(path);

    EXPECT_EQ(loaded.training_trace.tree_count, trace_case.training_trace.tree_count);
    EXPECT_EQ(loaded.training_trace.query_count, trace_case.training_trace.query_count);
    EXPECT_EQ(loaded.training_trace.loss_mode, forestTrainingLoss::SecondOrderXgboost);
    EXPECT_EQ(loaded.inference_queries.query_count, trace_case.inference_queries.query_count);
    EXPECT_EQ(loaded.training_trace.thresholds, trace_case.training_trace.thresholds);
    EXPECT_EQ(loaded.training_trace.leaf_indices, trace_case.training_trace.leaf_indices);
    EXPECT_EQ(loaded.inference_queries.lifted_inputs, trace_case.inference_queries.lifted_inputs);

    std::remove(path.c_str());
}

TEST_F(ForestTrainingTraceIoTest, ReadsSemanticTraceAndProves)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const std::string path = "/tmp/forest_training_trace_io_test.trace";
    write_forest_training_semantic_trace_case(path, build_demo_training_trace_case());

    const forestTrainingThenInferenceInputs inputs =
        read_forest_training_then_inference_inputs_from_trace_case(path);
    const forestTrainingThenInferencePreparedRun prepared =
        prove_and_prepare_training_then_inference_with_runtime(
            public_params, inputs, "test.training_trace_io");
    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(public_params, prepared.verifier_bundle, "test.training_trace_io");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
    EXPECT_EQ(evals_on_domain(inputs.training_inputs.thresholds, 4),
              std::vector<Fr>({fr_from_int(1), fr_from_int(0), fr_from_int(0), fr_from_int(0)}));

    std::remove(path.c_str());
}

TEST_F(ForestTrainingTraceIoTest, RoundTripsSplitTrainingAndInferenceTraceFiles)
{
    const forestTrainingSemanticTrace training_trace = build_demo_training_trace();
    forestSemanticInferenceQueryBatch inference_trace = build_demo_inference_query_trace();
    inference_trace.sample_indices = {3, 1, 2, 0};

    const std::string training_path = "/tmp/forest_training_trace_io_training.trace";
    const std::string inference_path = "/tmp/forest_training_trace_io_inference.trace";
    write_forest_training_semantic_trace(training_path, training_trace);
    write_forest_semantic_inference_query_trace(inference_path, inference_trace);

    const forestTrainingSemanticTrace loaded_training =
        read_forest_training_semantic_trace(training_path);
    const forestSemanticInferenceQueryBatch loaded_inference =
        read_forest_semantic_inference_query_trace(inference_path);

    EXPECT_EQ(loaded_training.root_indices, training_trace.root_indices);
    EXPECT_EQ(loaded_training.thresholds, training_trace.thresholds);
    EXPECT_EQ(loaded_training.node_leaf_mask, training_trace.node_leaf_mask);
    EXPECT_EQ(loaded_inference.query_tree_ids, inference_trace.query_tree_ids);
    EXPECT_EQ(loaded_inference.lifted_inputs, inference_trace.lifted_inputs);
    EXPECT_EQ(loaded_inference.sample_indices, inference_trace.sample_indices);

    std::remove(training_path.c_str());
    std::remove(inference_path.c_str());
}

TEST_F(ForestTrainingTraceIoTest, ReadsTrainingThenInferenceInputsFromSplitTraceFiles)
{
    const forestTrainingSemanticTrace training_trace = build_demo_training_trace();
    const forestSemanticInferenceQueryBatch inference_trace = build_demo_inference_query_trace();

    const std::string training_path = "/tmp/forest_training_trace_io_training_inputs.trace";
    const std::string inference_path = "/tmp/forest_training_trace_io_inference_inputs.trace";
    write_forest_training_semantic_trace(training_path, training_trace);
    write_forest_semantic_inference_query_trace(inference_path, inference_trace);

    const forestTrainingThenInferenceInputs combined_inputs =
        assemble_forest_training_then_inference_inputs_from_trace(training_trace, inference_trace);
    const forestTrainingThenInferenceInputs split_inputs =
        read_forest_training_then_inference_inputs_from_split_traces(training_path, inference_path);

    EXPECT_EQ(evals_on_domain(split_inputs.training_inputs.thresholds, 4),
              evals_on_domain(combined_inputs.training_inputs.thresholds, 4));
    EXPECT_EQ(evals_on_domain(split_inputs.training_inputs.node_leaf_mask, 4),
              evals_on_domain(combined_inputs.training_inputs.node_leaf_mask, 4));
    EXPECT_EQ(evals_on_domain(split_inputs.inference_queries.query_tree_ids, 4),
              evals_on_domain(combined_inputs.inference_queries.query_tree_ids, 4));
    EXPECT_EQ(evals_on_domain(split_inputs.inference_queries.lifted_inputs, 4),
              evals_on_domain(combined_inputs.inference_queries.lifted_inputs, 4));

    std::remove(training_path.c_str());
    std::remove(inference_path.c_str());
}
