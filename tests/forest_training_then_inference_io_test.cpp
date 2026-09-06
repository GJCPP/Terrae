#include "bls12_381_wrapper.hpp"
#include "forest_training_then_inference_io.hpp"
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

class ForestTrainingThenInferenceIoTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTrainingThenInferenceIoTest, SummarizesAndRoundTripsCase)
{
    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = build_demo_training_inputs();
    inputs.inference_queries = build_demo_inference_query_batch();

    const forestTrainingThenInferenceCaseSummary summary =
        summarize_forest_training_then_inference_inputs(inputs);
    EXPECT_EQ(summary.training_tree_count, 1u);
    EXPECT_EQ(summary.exported_leaf_row_count, 2u);
    EXPECT_FALSE(summary.explicit_inference_witness_provided);

    const std::string path = "/tmp/forest_training_then_inference_io_test.case";
    write_forest_training_then_inference_case(path, inputs);
    const forestTrainingThenInferenceInputs loaded =
        read_forest_training_then_inference_case(path);

    EXPECT_EQ(loaded.training_inputs.tree_count, inputs.training_inputs.tree_count);
    EXPECT_EQ(loaded.training_inputs.feature_count, inputs.training_inputs.feature_count);
    EXPECT_EQ(loaded.inference_queries.query_count, inputs.inference_queries.query_count);
    EXPECT_EQ(evals_on_domain(loaded.training_inputs.root_indices, 1),
              evals_on_domain(inputs.training_inputs.root_indices, 1));
    EXPECT_EQ(evals_on_domain(loaded.training_inputs.thresholds, 4),
              evals_on_domain(inputs.training_inputs.thresholds, 4));
    EXPECT_EQ(evals_on_domain(loaded.inference_queries.query_tree_ids, 4),
              evals_on_domain(inputs.inference_queries.query_tree_ids, 4));
    EXPECT_EQ(evals_on_domain(loaded.inference_queries.lifted_inputs, 4),
              evals_on_domain(inputs.inference_queries.lifted_inputs, 4));

    std::remove(path.c_str());
}

TEST_F(ForestTrainingThenInferenceIoTest, RoundTripsCaseWithExplicitInferenceWitness)
{
    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = build_demo_training_inputs();
    inputs.inference_queries = build_demo_inference_query_batch();

    const forestCommittedNodeTable committed_forest =
        forest_committed_node_table_from_training_inputs(inputs.training_inputs);
    const forestExportedLeafTable exported = export_leaf_table_from_committed_forest(committed_forest);
    inputs.explicit_inference_witness =
        derive_exported_inference_witness_batch(committed_forest, exported, inputs.inference_queries);

    const forestTrainingThenInferenceCaseSummary summary =
        summarize_forest_training_then_inference_inputs(inputs);
    EXPECT_TRUE(summary.explicit_inference_witness_provided);

    const std::string path = "/tmp/forest_training_then_inference_io_explicit_test.case";
    write_forest_training_then_inference_case(path, inputs);
    const forestTrainingThenInferenceInputs loaded =
        read_forest_training_then_inference_case(path);

    ASSERT_TRUE(loaded.explicit_inference_witness.has_value());
    EXPECT_EQ(evals_on_domain(loaded.explicit_inference_witness->lower_bounds, 4),
              evals_on_domain(inputs.explicit_inference_witness->lower_bounds, 4));
    EXPECT_EQ(evals_on_domain(loaded.explicit_inference_witness->leaf_indices, 4),
              evals_on_domain(inputs.explicit_inference_witness->leaf_indices, 4));

    std::remove(path.c_str());
}

TEST_F(ForestTrainingThenInferenceIoTest, RoundTripsBundleAndVerifies)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = build_demo_training_inputs();
    inputs.inference_queries = build_demo_inference_query_batch();

    const forestTrainingThenInferenceArtifact artifact =
        prove_training_then_inference_with_runtime(public_params, inputs, "test.training_then_inference_io");

    const std::string path = "/tmp/forest_training_then_inference_io_test.bundle";
    write_forest_training_then_inference_bundle(path, artifact);
    const forestTrainingThenInferenceArtifact loaded =
        read_forest_training_then_inference_bundle(path);
    const forestTrainingThenInferenceVerifierBundle verifier_bundle =
        prepare_training_then_inference_verifier_bundle(
            public_params, inputs, loaded, "test.training_then_inference_io");

    EXPECT_EQ(loaded.well_formedness_artifact.proof.bytes,
              artifact.well_formedness_artifact.proof.bytes);
    EXPECT_EQ(loaded.training_artifact.proof.bytes, artifact.training_artifact.proof.bytes);
    EXPECT_EQ(loaded.inference_artifact.proof.bytes, artifact.inference_artifact.proof.bytes);
    EXPECT_EQ(loaded.exported_leaf_table.row_count, artifact.exported_leaf_table.row_count);
    EXPECT_EQ(evals_on_domain(loaded.exported_leaf_table.leaf_indices, loaded.exported_leaf_table.row_count),
              evals_on_domain(artifact.exported_leaf_table.leaf_indices, artifact.exported_leaf_table.row_count));

    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, verifier_bundle, "test.training_then_inference_io");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);

    std::remove(path.c_str());
}
