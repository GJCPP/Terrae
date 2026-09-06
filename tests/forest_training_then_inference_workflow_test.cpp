#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "forest_training_trace_adapter.hpp"
#include "forest_training_then_inference_workflow.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mcl::bn;

namespace {

poly poly_from_domain_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

poly matrix_poly_from_rows(const std::vector<std::vector<int>>& rows)
{
    std::vector<Fr> evals;
    for (const auto& row : rows) {
        for (int value : row) {
            evals.push_back(fr_from_int(value));
        }
    }
    return poly_from_evals(evals);
}

poly cube_poly_from_slices(const std::vector<std::vector<std::vector<int>>>& cube)
{
    std::vector<Fr> evals;
    for (const auto& outer : cube) {
        for (const auto& middle : outer) {
            for (int value : middle) {
                evals.push_back(fr_from_int(value));
            }
        }
    }
    return poly_from_evals(evals);
}

forestExportedLeafTable swapped_leaf_rows(const forestExportedLeafTable& table)
{
    const std::vector<Fr> tree_evals = table.tree_indices.evaluations_on_domain(table.row_count);
    const std::vector<Fr> lower_evals = table.lower_bounds.evaluations_on_domain(table.row_count * table.feature_count);
    const std::vector<Fr> upper_evals = table.upper_bounds.evaluations_on_domain(table.row_count * table.feature_count);
    const std::vector<Fr> score_evals = table.scores.evaluations_on_domain(table.row_count);
    const std::vector<Fr> leaf_index_evals = table.leaf_indices.evaluations_on_domain(table.row_count);

    std::vector<Fr> swapped_tree_evals;
    std::vector<Fr> swapped_lower_evals;
    std::vector<Fr> swapped_upper_evals;
    std::vector<Fr> swapped_score_evals;
    std::vector<Fr> swapped_leaf_index_evals;
    for (size_t row = table.row_count; row > 0; --row) {
        const size_t source_row = row - 1;
        swapped_tree_evals.push_back(tree_evals[source_row]);
        swapped_score_evals.push_back(score_evals[source_row]);
        swapped_leaf_index_evals.push_back(leaf_index_evals[source_row]);
        for (size_t feature = 0; feature < table.feature_count; ++feature) {
            const size_t index = source_row * table.feature_count + feature;
            swapped_lower_evals.push_back(lower_evals[index]);
            swapped_upper_evals.push_back(upper_evals[index]);
        }
    }

    forestExportedLeafTable out;
    out.tree_indices = poly_from_evals(swapped_tree_evals);
    out.lower_bounds = poly_from_evals(swapped_lower_evals);
    out.upper_bounds = poly_from_evals(swapped_upper_evals);
    out.scores = poly_from_evals(swapped_score_evals);
    out.leaf_indices = poly_from_evals(swapped_leaf_index_evals);
    out.row_count = table.row_count;
    out.feature_count = table.feature_count;
    return out;
}

poly fixed_point_encoded_matrix_poly_from_rows(const std::vector<std::vector<int>>& rows,
                                               const fixedPointEncoding& encoding)
{
    std::vector<Fr> evals;
    for (const auto& row : rows) {
        for (int value : row) {
            evals.push_back(fixed_point_fr_from_size_t(encode_fixed_point_scaled_integer(value, encoding)));
        }
    }
    return poly_from_evals(evals);
}

forestTrainingSemanticTrace valid_training_trace()
{
    forestTrainingSemanticTrace trace;
    trace.tree_count = 1;
    trace.node_count = 4;
    trace.feature_count = 1;
    trace.bin_count = 4;
    trace.iteration_count = 1;
    trace.class_count = 1;
    trace.query_count = 4;
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
    trace.max_depth = 2;

    trace.root_indices = {0};
    trace.split_node_mask = {1, 0, 0, 0};
    trace.left_child_indices = {1, 0, 0, 0};
    trace.right_child_indices = {2, 0, 0, 0};
    trace.node_tree_ids = {1, 1, 1, 0};
    trace.feature_selector = {1, 0, 0, 0};
    trace.thresholds = {1, 0, 0, 0};
    trace.node_leaf_mask = {0, 1, 1, 0};
    trace.node_leaf_values = {0, 0, 0, 0};

    trace.training_inputs = {1, 0, 1, 1};
    trace.base_labels = {1, 1, 1, 1};
    trace.leaf_indices = {2, 1, 2, 2};
    trace.candidate_mask = {
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    trace.selected_candidate_indices = {0, 0, 0, 0};
    return trace;
}

forestTrainingProofInputs valid_training_inputs()
{
    return assemble_forest_training_inputs_from_trace(valid_training_trace());
}

forestSemanticInferenceQueryBatch valid_inference_query_trace()
{
    forestSemanticInferenceQueryBatch out;
    out.query_tree_ids = {1, 1, 1, 1};
    out.lifted_inputs = {1, 0, 1, 1};
    out.query_count = 4;
    out.feature_count = 1;
    return out;
}

forestInferenceQueryBatch valid_inference_queries()
{
    return assemble_forest_inference_query_batch_from_trace(valid_inference_query_trace());
}

} // namespace

class ForestTrainingThenInferenceWorkflowTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTrainingThenInferenceWorkflowTest, DerivesInferenceWitnessFromExportedLeafTable)
{
    const forestCommittedNodeTable committed_forest =
        forest_committed_node_table_from_training_inputs(valid_training_inputs());
    const forestExportedLeafTable exported = export_leaf_table_from_committed_forest(committed_forest);
    const forestExportedInferenceWitnessBatch witness =
        derive_exported_inference_witness_batch(committed_forest, exported, valid_inference_queries());

    EXPECT_EQ(witness.query_count, 4u);
    EXPECT_EQ(witness.lower_bounds.evaluations_on_domain(4), std::vector<Fr>({
                                                          fr_from_int(1),
                                                          fr_from_int(0),
                                                          fr_from_int(1),
                                                          fr_from_int(1),
                                                      }));
    EXPECT_EQ(witness.upper_bounds.evaluations_on_domain(4), std::vector<Fr>({
                                                          fr_from_int(4),
                                                          fr_from_int(1),
                                                          fr_from_int(4),
                                                          fr_from_int(4),
                                                      }));
    EXPECT_EQ(witness.leaf_indices.evaluations_on_domain(4), std::vector<Fr>({
                                                          fr_from_int(2),
                                                          fr_from_int(1),
                                                          fr_from_int(2),
                                                          fr_from_int(2),
                                                      }));
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, ClassificationSrsIncludesRangeTables)
{
    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();

    const kzgSrsRequirement requirement = required_exported_forest_classification_srs(inputs);
    EXPECT_EQ(requirement.deg1_plus_1, 64u);
    EXPECT_EQ(requirement.deg2_plus_1, 64u);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, FallbackInferenceDerivationUsesCommittedTreeTraversal)
{
    const forestCommittedNodeTable committed_forest =
        forest_committed_node_table_from_training_inputs(valid_training_inputs());
    const forestExportedLeafTable exported =
        swapped_leaf_rows(export_leaf_table_from_committed_forest(committed_forest));
    const forestExportedInferenceWitnessBatch witness =
        derive_exported_inference_witness_batch(committed_forest, exported, valid_inference_queries());

    EXPECT_EQ(witness.leaf_indices.evaluations_on_domain(4), std::vector<Fr>({
                                                          fr_from_int(2),
                                                          fr_from_int(1),
                                                          fr_from_int(2),
                                                          fr_from_int(2),
                                                      }));
    EXPECT_EQ(witness.lower_bounds.evaluations_on_domain(4), std::vector<Fr>({
                                                          fr_from_int(1),
                                                          fr_from_int(0),
                                                          fr_from_int(1),
                                                          fr_from_int(1),
                                                      }));
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, TrainingAndWellFormednessProduceSingleProofArtifact)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);
    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();
    const forestExportedInferencePlan plan = assemble_exported_inference_plan(
        public_params, inputs, "workflow.training_with_well_formedness");
    const forestWellFormednessInputs well_formedness_inputs =
        forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs);

    const ProofArtifact artifact =
        prove_forest_training_with_well_formedness_with_runtime(
            public_params,
            inputs.training_inputs,
            well_formedness_inputs,
            plan.committed_forest_public_inputs,
            "workflow.training_with_well_formedness");
    ASSERT_FALSE(artifact.proof.bytes.empty());

    const forestTrainingProofPublicInputs training_public_inputs =
        forest_training_proof_public_inputs_from_inputs(
            plan.committed_forest_public_inputs, inputs.training_inputs);
    const forestWellFormednessPublicInputs well_formedness_public_inputs =
        forest_well_formedness_public_inputs_from_inputs(
            plan.committed_forest_public_inputs, well_formedness_inputs);
    forestTrainingWithWellFormednessVerifierPreprocessing verifier =
        prepare_forest_training_with_well_formedness_verifier_preprocessing(
            public_params,
            artifact,
            training_public_inputs,
            well_formedness_public_inputs,
            "workflow.training_with_well_formedness");
    const VerificationResult result =
        verify_forest_training_with_well_formedness_preprocessed(verifier);
    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, CombinedHelperSupportsProveVerifyWithoutRowCountBasedSplitting)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();

    const forestTrainingThenInferencePreparedRun prepared =
        prove_and_prepare_training_then_inference_with_runtime(
            public_params, inputs, "workflow.training_then_inference");
    EXPECT_FALSE(prepared.artifact.well_formedness_artifact.proof.bytes.empty());
    EXPECT_FALSE(prepared.artifact.training_artifact.proof.bytes.empty());
    EXPECT_FALSE(prepared.artifact.inference_artifact.proof.bytes.empty());

    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, prepared.verifier_bundle, "workflow.training_then_inference");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
    EXPECT_TRUE(result.well_formedness_result.accepted);
    EXPECT_TRUE(result.well_formedness_result.proof_fully_consumed);
    EXPECT_TRUE(result.inference_result.accepted);
    EXPECT_TRUE(result.inference_result.proof_fully_consumed);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, PreparedExportedInferenceVerifierAcceptsValidProof)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();

    const forestTrainingThenInferencePreparedRun prepared =
        prove_and_prepare_training_then_inference_with_runtime(
            public_params, inputs, "workflow.training_then_inference.inference_prepared");
    const forestExportedInferenceBatchShape batch_shape = {
        prepared.exported_inference_plan.witness_batch.query_count,
        prepared.exported_inference_plan.witness_batch.feature_count,
    };

    forestExportedInferenceVerifierPreprocessing verifier =
        prepare_exported_forest_inference_verifier_preprocessing(
            public_params,
            prepared.artifact.inference_artifact,
            prepared.exported_inference_plan.inference_table,
            batch_shape,
            "workflow.training_then_inference.inference_prepared.inference");
    const VerificationResult result = verify_exported_forest_inference_preprocessed(verifier);

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, ExportedInferencePlanReportsCommitForestTimingProfile)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();

    forestExportedInferencePlanProfile profile;
    const forestExportedInferencePlan plan =
        assemble_exported_inference_plan_with_profile(
            public_params, inputs, profile, "workflow.training_then_inference.profile");

    EXPECT_EQ(plan.committed_forest.node_count, inputs.training_inputs.node_count);
    EXPECT_EQ(plan.committed_forest_public_inputs.node_count, inputs.training_inputs.node_count);
    EXPECT_EQ(plan.exported_leaf_table.row_count, 2u);
    EXPECT_GE(profile.total_seconds, 0.0);
    EXPECT_GE(profile.build_committed_forest_table_seconds, 0.0);
    EXPECT_GE(profile.commit_committed_forest_public_inputs_seconds, 0.0);
    EXPECT_DOUBLE_EQ(profile.commit_forest_seconds(),
                     profile.build_committed_forest_table_seconds +
                         profile.commit_committed_forest_public_inputs_seconds);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, CombinedHelperAcceptsExplicitInferenceWitness)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();

    const forestCommittedNodeTable committed_forest =
        forest_committed_node_table_from_training_inputs(inputs.training_inputs);
    const forestExportedLeafTable exported = export_leaf_table_from_committed_forest(committed_forest);
    inputs.explicit_inference_witness =
        derive_exported_inference_witness_batch(committed_forest, exported, inputs.inference_queries);

    const forestTrainingThenInferencePreparedRun prepared =
        prove_and_prepare_training_then_inference_with_runtime(
            public_params, inputs, "workflow.training_then_inference.explicit");
    EXPECT_FALSE(prepared.artifact.inference_artifact.proof.bytes.empty());

    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, prepared.verifier_bundle, "workflow.training_then_inference.explicit");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
    EXPECT_TRUE(result.inference_result.accepted);
    EXPECT_TRUE(result.inference_result.proof_fully_consumed);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, VerifierBundleDoesNotNeedFullTrainingWitnessPolynomials)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();

    const forestTrainingThenInferencePreparedRun prepared =
        prove_and_prepare_training_then_inference_with_runtime(
            public_params, inputs, "workflow.training_then_inference.bundle");

    EXPECT_EQ(prepared.verifier_bundle.inference_batch_shape.query_count, inputs.inference_queries.query_count);
    EXPECT_EQ(prepared.verifier_bundle.inference_batch_shape.feature_count, inputs.inference_queries.feature_count);
    EXPECT_EQ(prepared.artifact.exported_leaf_table.row_count, prepared.exported_inference_plan.exported_leaf_table.row_count);

    const forestTrainingThenInferenceVerificationResult result = verify_training_then_inference_with_runtime(
        public_params, prepared.verifier_bundle, "workflow.training_then_inference.bundle");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, RejectsExplicitWitnessThatDoesNotMatchQueries)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();

    const forestCommittedNodeTable committed_forest =
        forest_committed_node_table_from_training_inputs(inputs.training_inputs);
    const forestExportedLeafTable exported = export_leaf_table_from_committed_forest(committed_forest);
    forestExportedInferenceWitnessBatch witness =
        derive_exported_inference_witness_batch(committed_forest, exported, inputs.inference_queries);
    witness.query_tree_ids = poly_from_domain_ints({0, 0, 0, 0});
    inputs.explicit_inference_witness = std::move(witness);

    EXPECT_THROW(
        (void)prove_training_then_inference_with_runtime(
            public_params, inputs, "workflow.training_then_inference.invalid_explicit"),
        std::runtime_error);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, RejectsTooSmallSrsBeforeProving)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8, 8);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();

    EXPECT_THROW(
        (void)prove_training_then_inference_with_runtime(
            public_params, inputs, "workflow.training_then_inference.small_srs"),
        std::runtime_error);
}


TEST_F(ForestTrainingThenInferenceWorkflowTest, ExportedClassificationSupportsProveAndVerify)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();
    const forestExportedInferencePlan plan =
        assemble_exported_inference_plan(public_params, inputs, "workflow.exported_classification");

    EXPECT_EQ(plan.witness_batch.iteration_count, 1u);
    EXPECT_EQ(plan.witness_batch.class_count, 1u);
    EXPECT_EQ(plan.witness_batch.sample_count, 4u);
    EXPECT_EQ(plan.witness_batch.class_scores.evaluations_on_domain(4),
              plan.witness_batch.scores.evaluations_on_domain(4));
    EXPECT_EQ(plan.witness_batch.class_selector.evaluations_on_domain(4),
              std::vector<Fr>(4, Fr(1)));

    const ProofArtifact artifact = prove_exported_forest_classification_with_runtime(
        public_params,
        plan.classification_table,
        plan.witness_batch,
        "workflow.exported_classification.proof");
    const forestExportedInferenceBatchShape shape = {
        plan.witness_batch.query_count,
        plan.witness_batch.feature_count,
        plan.witness_batch.iteration_count,
        plan.witness_batch.class_count,
        plan.witness_batch.sample_count,
    };
    const VerificationResult result = verify_exported_forest_classification_with_runtime(
        public_params,
        artifact,
        plan.classification_table,
        shape,
        "workflow.exported_classification.proof");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestTrainingThenInferenceWorkflowTest, PreparedExportedClassificationVerifierAcceptsValidProof)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = valid_training_inputs();
    inputs.inference_queries = valid_inference_queries();
    const forestExportedInferencePlan plan =
        assemble_exported_inference_plan(public_params, inputs, "workflow.exported_classification_prepared");

    const ProofArtifact artifact = prove_exported_forest_classification_with_runtime(
        public_params,
        plan.classification_table,
        plan.witness_batch,
        "workflow.exported_classification_prepared.proof");
    const forestExportedInferenceBatchShape shape = {
        plan.witness_batch.query_count,
        plan.witness_batch.feature_count,
        plan.witness_batch.iteration_count,
        plan.witness_batch.class_count,
        plan.witness_batch.sample_count,
    };

    forestExportedClassificationVerifierPreprocessing verifier =
        prepare_exported_forest_classification_verifier_preprocessing(
            public_params,
            artifact,
            plan.classification_table,
            shape,
            "workflow.exported_classification_prepared.proof");
    const VerificationResult result = verify_exported_forest_classification_preprocessed(verifier);

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}
