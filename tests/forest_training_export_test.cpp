#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_inference.hpp"
#include "forest_training_export.hpp"
#include "forest_training_workflow.hpp"
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

forestTrainingProofInputs valid_inputs()
{
    forestTrainingProofInputs inputs;

    inputs.tree_count = 1;
    inputs.node_count = 4;
    inputs.feature_count = 1;
    inputs.bin_count = 4;
    inputs.iteration_count = 1;
    inputs.class_count = 1;
    inputs.query_count = 4;
    inputs.input_bit_bound = 3;
    inputs.full_upper_bound = 4;
    inputs.structure_bit_bound = 3;
    inputs.gradient_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    inputs.gain_numerator_encoding = fixedPointEncoding::signed_offset_binary(10, 0);
    inputs.quantized_gain_encoding = fixedPointEncoding::signed_offset_binary(10, 2);
    inputs.leaf_value_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    inputs.score_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    inputs.hessian_bit_bound = 3;
    inputs.gain_denominator_bit_bound = 3;
    inputs.leaf_denominator_bit_bound = 2;
    inputs.gain_gap_bit_bound = 2;
    inputs.loss_mode = forestTrainingLoss::SecondOrderXgboost;
    inputs.l2_regularization = 1;

    inputs.root_indices = poly_from_domain_ints({0});
    inputs.split_node_mask = poly_from_domain_ints({1, 0, 0, 0});
    inputs.node_depths = poly_from_domain_ints({1, 2, 2, 0});
    inputs.left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    inputs.right_child_indices = poly_from_domain_ints({2, 0, 0, 0});
    inputs.node_tree_ids = poly_from_domain_ints({1, 1, 1, 0});
    inputs.feature_selector = matrix_poly_from_rows({{1}, {0}, {0}, {0}});
    inputs.lower_bounds = matrix_poly_from_rows({{0}, {0}, {2}, {0}});
    inputs.upper_bounds = matrix_poly_from_rows({{4}, {2}, {4}, {0}});
    inputs.left_lower = matrix_poly_from_rows({{0}, {0}, {0}, {0}});
    inputs.left_upper = matrix_poly_from_rows({{2}, {0}, {0}, {0}});
    inputs.right_lower = matrix_poly_from_rows({{2}, {0}, {0}, {0}});
    inputs.right_upper = matrix_poly_from_rows({{4}, {0}, {0}, {0}});
    inputs.thresholds = poly_from_domain_ints({2, 0, 0, 0});

    inputs.query_tree_ids = poly_from_domain_ints({1, 1, 1, 1});
    inputs.lifted_inputs = matrix_poly_from_rows({{1}, {0}, {2}, {3}});
    inputs.leaf_indices = poly_from_domain_ints({1, 1, 2, 2});
    inputs.leaf_lower_bounds = matrix_poly_from_rows({{0}, {0}, {2}, {2}});
    inputs.leaf_upper_bounds = matrix_poly_from_rows({{2}, {2}, {4}, {4}});
    inputs.tree_scores = poly_from_domain_ints({0, 0, 0, 0});
    inputs.training_node_tree_ids = inputs.node_tree_ids;
    inputs.training_node_lower_bounds = inputs.lower_bounds;
    inputs.training_node_upper_bounds = inputs.upper_bounds;
    inputs.node_leaf_mask = poly_from_domain_ints({0, 1, 1, 0});
    inputs.node_leaf_values = poly_from_domain_ints({0, 0, 0, 0});
    inputs.lifted_leaf_indices = matrix_poly_from_rows({{1}, {1}, {2}, {2}});
    inputs.flattened_bucket_indices = matrix_poly_from_rows({{5}, {4}, {10}, {11}});

    inputs.channels[0].query_values = poly_from_domain_ints({1, 1, 1, 1});
    inputs.channels[0].lifted_values = matrix_poly_from_rows({{1}, {1}, {1}, {1}});
    inputs.channels[0].leaf_histograms = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {1, 1, 0, 0},
        {0, 0, 1, 1},
        {0, 0, 0, 0},
    });
    inputs.channels[0].full_histograms = matrix_poly_from_rows({
        {1, 1, 1, 1},
        {1, 1, 0, 0},
        {0, 0, 1, 1},
        {0, 0, 0, 0},
    });
    inputs.channels[0].split_histograms = inputs.channels[0].full_histograms;
    inputs.channels[0].left_statistics = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {1, 2, 2, 2},
        {0, 0, 1, 2},
        {0, 0, 0, 0},
    });
    inputs.channels[0].right_statistics = matrix_poly_from_rows({
        {3, 2, 1, 0},
        {1, 0, 0, 0},
        {2, 2, 1, 0},
        {0, 0, 0, 0},
    });

    for (size_t channel = 1; channel < inputs.channels.size(); ++channel) {
        inputs.channels[channel].query_values = poly_from_domain_ints({0, 0, 0, 0});
        inputs.channels[channel].lifted_values = matrix_poly_from_rows({{0}, {0}, {0}, {0}});
        inputs.channels[channel].leaf_histograms = matrix_poly_from_rows({
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
        });
        inputs.channels[channel].full_histograms = inputs.channels[channel].leaf_histograms;
        inputs.channels[channel].split_histograms = inputs.channels[channel].full_histograms;
        inputs.channels[channel].left_statistics = inputs.channels[channel].full_histograms;
        inputs.channels[channel].right_statistics = inputs.channels[channel].full_histograms;
    }

    inputs.left_gradients = inputs.channels[1].left_statistics;
    inputs.right_gradients = inputs.channels[1].right_statistics;
    inputs.left_hessians = inputs.channels[2].left_statistics;
    inputs.right_hessians = inputs.channels[2].right_statistics;
    inputs.left_gradient_squares = inputs.channels[1].full_histograms;
    inputs.right_gradient_squares = inputs.channels[1].full_histograms;
    inputs.parent_gradient_squares = inputs.channels[1].full_histograms;
    inputs.gain_numerators = inputs.channels[1].full_histograms;
    inputs.gain_denominators = matrix_poly_from_rows({
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1},
    });
    inputs.encoded_gain_numerators = fixed_point_encoded_matrix_poly_from_rows(
        {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, inputs.gain_numerator_encoding);
    inputs.encoded_quantized_gains = fixed_point_encoded_matrix_poly_from_rows(
        {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, inputs.quantized_gain_encoding);
    inputs.gain_remainders = inputs.channels[1].full_histograms;

    inputs.quantized_gains = inputs.channels[1].full_histograms;
    inputs.tie_break_scores = matrix_poly_from_rows({
        {3, 2, 1, 0},
        {3, 2, 1, 0},
        {3, 2, 1, 0},
        {3, 2, 1, 0},
    });
    inputs.candidate_mask = matrix_poly_from_rows({
        {0, 0, 1, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
    });
    inputs.active_node_mask = inputs.split_node_mask;
    inputs.selector = inputs.candidate_mask;
    inputs.selected_gains = poly_from_domain_ints({0, 0, 0, 0});
    inputs.selected_tie_break_scores = poly_from_domain_ints({1, 0, 0, 0});
    inputs.selected_candidate_indices = poly_from_domain_ints({2, 0, 0, 0});
    inputs.selected_features = poly_from_domain_ints({0, 0, 0, 0});
    inputs.selected_bins = poly_from_domain_ints({2, 0, 0, 0});

    inputs.leaf_gradients = poly_from_domain_ints({0, 0, 0, 0});
    inputs.leaf_hessians = poly_from_domain_ints({0, 0, 0, 0});
    inputs.leaf_denominators = poly_from_domain_ints({1, 1, 1, 1});
    inputs.encoded_negative_leaf_gradients =
        fixed_point_encoded_matrix_poly_from_rows({{0}, {0}, {0}, {0}}, inputs.gradient_encoding);
    inputs.encoded_leaf_values =
        fixed_point_encoded_matrix_poly_from_rows({{0}, {0}, {0}, {0}}, inputs.leaf_value_encoding);
    inputs.leaf_remainders = poly_from_domain_ints({0, 0, 0, 0});

    inputs.cumulative_scores = matrix_poly_from_rows({{0, 0, 0, 0}});
    inputs.encoded_cumulative_scores =
        fixed_point_encoded_matrix_poly_from_rows({{0, 0, 0, 0}}, inputs.score_encoding);
    inputs.encoded_shifted_scores =
        fixed_point_encoded_matrix_poly_from_rows({{0, 0, 0, 0}}, inputs.score_encoding);
    inputs.softmax_shifts =
        fixed_point_encoded_matrix_poly_from_rows({{0, 0, 0, 0}}, inputs.score_encoding);
    inputs.shift_selector = cube_poly_from_slices({{{1, 1, 1, 1}}});
    inputs.softmax_probabilities = cube_poly_from_slices({{{1, 1, 1, 1}}});
    inputs.base_labels = matrix_poly_from_rows({{1, 1, 1, 1}});
    inputs.pseudo_residuals = cube_poly_from_slices({{{0, 0, 0, 0}}});
    inputs.diagonal_hessians = cube_poly_from_slices({{{0, 0, 0, 0}}});

    return inputs;
}

} // namespace

class ForestTrainingExportTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTrainingExportTest, ExportLeafTableTracksLeafRows)
{
    const forestCommittedNodeTable committed_forest =
        forest_committed_node_table_from_training_inputs(valid_inputs());
    const forestExportedLeafTable exported = export_leaf_table_from_committed_forest(committed_forest);

    EXPECT_EQ(exported.row_count, 2u);
    EXPECT_EQ(exported.feature_count, 1u);
    EXPECT_EQ(exported.tree_indices.evaluations_on_domain(2), std::vector<Fr>({fr_from_int(1), fr_from_int(1)}));
    EXPECT_EQ(exported.leaf_indices.evaluations_on_domain(2), std::vector<Fr>({fr_from_int(1), fr_from_int(2)}));
}

TEST_F(ForestTrainingExportTest, ExportLeafTablePadsToPowerOfTwoWithUniqueDummyLeafIndices)
{
    forestTrainingProofInputs inputs = valid_inputs();
    inputs.node_leaf_mask = poly_from_domain_ints({0, 1, 1, 1});
    inputs.node_tree_ids = poly_from_domain_ints({1, 1, 1, 1});
    inputs.training_node_tree_ids = inputs.node_tree_ids;
    inputs.node_leaf_values = poly_from_domain_ints({0, 5, 7, 9});
    inputs.training_node_lower_bounds = matrix_poly_from_rows({{0}, {0}, {2}, {3}});
    inputs.training_node_upper_bounds = matrix_poly_from_rows({{4}, {2}, {4}, {4}});

    const forestCommittedNodeTable committed_forest = forest_committed_node_table_from_training_inputs(inputs);
    const forestExportedLeafTable exported = export_leaf_table_from_committed_forest(committed_forest);
    const std::vector<Fr> leaf_index_evals = exported.leaf_indices.evaluations_on_domain(exported.row_count);
    const std::vector<Fr> tree_evals = exported.tree_indices.evaluations_on_domain(exported.row_count);

    ASSERT_EQ(exported.row_count, 4u);
    ASSERT_EQ(leaf_index_evals.size(), 4u);
    EXPECT_EQ(leaf_index_evals[0], fr_from_int(1));
    EXPECT_EQ(leaf_index_evals[1], fr_from_int(2));
    EXPECT_EQ(leaf_index_evals[2], fr_from_int(3));
    EXPECT_EQ(leaf_index_evals[3], fr_from_int(4));
    EXPECT_EQ(tree_evals[0], fr_from_int(1));
    EXPECT_EQ(tree_evals[1], fr_from_int(1));
    EXPECT_EQ(tree_evals[2], fr_from_int(1));
    EXPECT_EQ(tree_evals[3], fr_from_int(0));
}

TEST_F(ForestTrainingExportTest, ExportedInferenceTableSupportsProveAndVerify)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestCommittedNodeTable committed_forest =
        forest_committed_node_table_from_training_inputs(valid_inputs());
    const forestExportedLeafTable exported = export_leaf_table_from_committed_forest(committed_forest);
    const forestInferencePublicTable table = build_forest_inference_public_table_from_exported_forest(
        public_params, "forest_export.inference", exported, 3, 3);

    const poly query_tree_ids = poly_from_domain_ints({1, 1});
    const poly lifted_inputs = matrix_poly_from_rows({{1}, {3}});
    const poly lower_bounds = matrix_poly_from_rows({{0}, {2}});
    const poly upper_bounds = matrix_poly_from_rows({{2}, {4}});
    const poly scores = poly_from_domain_ints({0, 0});
    const poly leaf_indices = poly_from_domain_ints({1, 2});

    ProofRuntime prover = ProofRuntime::prover(public_params);
    EXPECT_TRUE(run_proof_program(prover, [&](ProofProgram& program) {
        auto& context = program.create_forest_inference_context(table);
        context.add_check(WitnessCommitmentInput::prover("forest_export.query_tree_ids", query_tree_ids, 3),
                          WitnessCommitmentInput::prover("forest_export.inputs", lifted_inputs, 3),
                          WitnessCommitmentInput::prover("forest_export.lower_bounds", lower_bounds, 3),
                          WitnessCommitmentInput::prover("forest_export.upper_bounds", upper_bounds, 3),
                          WitnessCommitmentInput::prover("forest_export.scores", scores, 3),
                          WitnessCommitmentInput::prover("forest_export.leaf_indices", leaf_indices, 3),
                          2);
    }));

    const SerializedProof proof = prover.take_proof();
    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_proof_program(verifier, [&](ProofProgram& program) {
        auto& context = program.create_forest_inference_context(table.verifier_view());
        context.add_check(WitnessCommitmentInput::verifier("forest_export.query_tree_ids", 3),
                          WitnessCommitmentInput::verifier("forest_export.inputs", 3),
                          WitnessCommitmentInput::verifier("forest_export.lower_bounds", 3),
                          WitnessCommitmentInput::verifier("forest_export.upper_bounds", 3),
                          WitnessCommitmentInput::verifier("forest_export.scores", 3),
                          WitnessCommitmentInput::verifier("forest_export.leaf_indices", 3),
                          2);
    }));
    EXPECT_TRUE(verifier.proof_fully_consumed());
}
