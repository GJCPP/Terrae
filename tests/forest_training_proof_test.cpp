#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "box_check.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "cube_argmax.hpp"
#include "cube_softmax.hpp"
#include "cube_sum.hpp"
#include "fixed_point.hpp"
#include "fixed_point_division.hpp"
#include "forest_boosting_state.hpp"
#include "forest_histogram.hpp"
#include "forest_histogram_bucket_address.hpp"
#include "forest_histogram_refresh.hpp"
#include "forest_leaf_value.hpp"
#include "forest_depth_consistency.hpp"
#include "forest_quantized_split_gain.hpp"
#include "forest_split_gain.hpp"
#include "forest_split_selection.hpp"
#include "forest_split_statistics.hpp"
#include "forest_structure.hpp"
#include "forest_training_inference.hpp"
#include "forest_training_proof.hpp"
#include "hist_agg.hpp"
#include "index_routing.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "masked_argmax.hpp"
#include "matrix_sum.hpp"
#include "selector_check.hpp"
#include "signed_range_check.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "test_kzg_srs.hpp"
#include "tree_histogram_consistency.hpp"
#include "zero_check.hpp"
#include "../examples/forest_training_demo_data.hpp"

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
    return build_demo_training_inputs();
}

forestTrainingProofWitness valid_witness(kzgProtocol& kzg, const forestTrainingProofInputs& inputs)
{
    const auto commit = [&kzg](const poly& polynomial) { return virCommitment(kzg.commit(polynomial)); };

    forestTrainingProofWitness witness;
    witness.root_indices = commit(inputs.root_indices);
    witness.split_node_mask = commit(inputs.split_node_mask);
    witness.node_leaf_mask = commit(inputs.node_leaf_mask);
    witness.depths = commit(inputs.node_depths);
    witness.left_child_indices = commit(inputs.left_child_indices);
    witness.right_child_indices = commit(inputs.right_child_indices);
    witness.node_tree_ids = commit(inputs.node_tree_ids);
    witness.feature_selector = commit(inputs.feature_selector);
    witness.lower_bounds = commit(inputs.lower_bounds);
    witness.upper_bounds = commit(inputs.upper_bounds);
    witness.left_lower = commit(inputs.left_lower);
    witness.left_upper = commit(inputs.left_upper);
    witness.right_lower = commit(inputs.right_lower);
    witness.right_upper = commit(inputs.right_upper);
    witness.thresholds = commit(inputs.thresholds);

    witness.histogram_refresh.inference.query_tree_ids = commit(inputs.query_tree_ids);
    witness.histogram_refresh.inference.lifted_inputs = commit(inputs.lifted_inputs);
    witness.histogram_refresh.inference.leaf_indices = commit(inputs.leaf_indices);
    witness.histogram_refresh.inference.leaf_lower_bounds = commit(inputs.leaf_lower_bounds);
    witness.histogram_refresh.inference.leaf_upper_bounds = commit(inputs.leaf_upper_bounds);
    witness.histogram_refresh.inference.tree_scores = commit(inputs.tree_scores);
    witness.histogram_refresh.inference.node_tree_ids = commit(inputs.training_node_tree_ids);
    witness.histogram_refresh.inference.node_lower_bounds = commit(inputs.training_node_lower_bounds);
    witness.histogram_refresh.inference.node_upper_bounds = commit(inputs.training_node_upper_bounds);
    witness.histogram_refresh.inference.node_leaf_mask = commit(inputs.node_leaf_mask);
    witness.histogram_refresh.inference.node_leaf_values = commit(inputs.node_leaf_values);
    witness.histogram_refresh.lifted_leaf_indices = commit(inputs.lifted_leaf_indices);
    witness.histogram_refresh.flattened_bucket_indices = commit(inputs.flattened_bucket_indices);
    witness.histogram_refresh.parent_mask = commit(inputs.split_node_mask);
    witness.histogram_refresh.left_child_indices = commit(inputs.left_child_indices);
    witness.histogram_refresh.right_child_indices = commit(inputs.right_child_indices);

    for (const forestTrainingProofChannelInputs& channel : inputs.channels) {
        witness.histogram_refresh.channels.query_values.push_back(commit(channel.query_values));
        witness.histogram_refresh.channels.lifted_values.push_back(commit(channel.lifted_values));
        witness.histogram_refresh.channels.leaf_histograms.push_back(commit(channel.leaf_histograms));
        witness.histogram_refresh.channels.full_histograms.push_back(commit(channel.full_histograms));
        witness.split_statistics.histograms.push_back(commit(channel.split_histograms));
        witness.split_statistics.left_statistics.push_back(commit(channel.left_statistics));
        witness.split_statistics.right_statistics.push_back(commit(channel.right_statistics));
    }

    witness.split_gain.left_gradients = commit(inputs.left_gradients);
    witness.split_gain.right_gradients = commit(inputs.right_gradients);
    witness.split_gain.left_hessians = commit(inputs.left_hessians);
    witness.split_gain.right_hessians = commit(inputs.right_hessians);
    witness.split_gain.left_gradient_squares = commit(inputs.left_gradient_squares);
    witness.split_gain.right_gradient_squares = commit(inputs.right_gradient_squares);
    witness.split_gain.parent_gradient_squares = commit(inputs.parent_gradient_squares);
    witness.split_gain.gain_numerators = commit(inputs.gain_numerators);
    witness.split_gain.gain_denominators = commit(inputs.gain_denominators);

    witness.quantized_split_gain.exact_gain_numerators = witness.split_gain.gain_numerators;
    witness.quantized_split_gain.gain_denominators = witness.split_gain.gain_denominators;
    witness.quantized_split_gain.encoded_gain_numerators = commit(inputs.encoded_gain_numerators);
    witness.quantized_split_gain.encoded_quantized_gains = commit(inputs.encoded_quantized_gains);
    witness.quantized_split_gain.gain_remainders = commit(inputs.gain_remainders);

    witness.split_selection.quantized_gains = commit(inputs.quantized_gains);
    witness.split_selection.tie_break_scores = commit(inputs.tie_break_scores);
    witness.split_selection.candidate_mask = commit(inputs.candidate_mask);
    witness.split_selection.active_node_mask = commit(inputs.active_node_mask);
    witness.split_selection.selector = commit(inputs.selector);
    witness.split_selection.selected_gains = commit(inputs.selected_gains);
    witness.split_selection.selected_tie_break_scores = commit(inputs.selected_tie_break_scores);
    witness.split_selection.selected_candidate_indices = commit(inputs.selected_candidate_indices);
    witness.split_selection.selected_features = commit(inputs.selected_features);
    witness.split_selection.selected_bins = commit(inputs.selected_bins);
    witness.depth_limit_mask = commit(inputs.depth_limit_mask);

    witness.leaf_value.leaf_gradients = commit(inputs.leaf_gradients);
    witness.leaf_value.leaf_hessians = commit(inputs.leaf_hessians);
    witness.leaf_value.leaf_denominators = commit(inputs.leaf_denominators);
    witness.leaf_value.encoded_negative_leaf_gradients = commit(inputs.encoded_negative_leaf_gradients);
    witness.leaf_value.encoded_leaf_values = commit(inputs.encoded_leaf_values);
    witness.leaf_value.leaf_remainders = commit(inputs.leaf_remainders);

    witness.boosting_state.tree_scores = commit(inputs.tree_scores);
    witness.boosting_state.cumulative_scores = commit(inputs.cumulative_scores);
    witness.boosting_state.encoded_cumulative_scores = commit(inputs.encoded_cumulative_scores);
    witness.boosting_state.encoded_shifted_scores = commit(inputs.encoded_shifted_scores);
    witness.boosting_state.softmax_shifts = commit(inputs.softmax_shifts);
    witness.boosting_state.shift_selector = commit(inputs.shift_selector);
    witness.boosting_state.lifted_softmax_shifts = commit(inputs.lifted_softmax_shifts);
    witness.boosting_state.softmax_exp_values = commit(inputs.softmax_exp_values);
    witness.boosting_state.softmax_denominators = commit(inputs.softmax_denominators);
    witness.boosting_state.lifted_softmax_denominators = commit(inputs.lifted_softmax_denominators);
    witness.boosting_state.softmax_probabilities = commit(inputs.softmax_probabilities);
    witness.boosting_state.softmax_remainders = commit(inputs.softmax_remainders);
    witness.boosting_state.base_labels = commit(inputs.base_labels);
    witness.boosting_state.pseudo_residuals = commit(inputs.pseudo_residuals);
    witness.boosting_state.encoded_diagonal_hessians = commit(inputs.encoded_diagonal_hessians);
    witness.boosting_state.diagonal_hessian_remainders = commit(inputs.diagonal_hessian_remainders);
    witness.boosting_state.diagonal_hessians = commit(inputs.diagonal_hessians);

    return witness;
}

} // namespace

class ForestTrainingProofProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTrainingProofProtocolTest, ValidTrainingProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    matrixColumnPrefixSumProtocol column_prefix;
    column_prefix.setup(kzg, zero, column_replication);

    matrixRowPrefixSumProtocol row_prefix;
    row_prefix.setup(kzg, zero);

    matrixColumnSumProtocol column_sum;
    column_sum.setup(kzg, zero);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    boxCheckProtocol box_check;
    box_check.setup(kzg, less);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    rootSelectorConsistencyProtocol root_selector_consistency;
    root_selector_consistency.setup(kzg, zero, hist_agg, index_routing);

    nodeRoleConsistencyProtocol node_role_consistency;
    node_role_consistency.setup(kzg, zero, hist_agg);

    forestDepthConsistencyProtocol forest_depth_consistency;
    forest_depth_consistency.setup(kzg, zero, index_routing);

    parentChildBoxInheritanceProtocol parent_child_box_inheritance;
    parent_child_box_inheritance.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    parentChildTreeConsistencyProtocol parent_child_tree_consistency;
    parent_child_tree_consistency.setup(kzg, zero, index_routing);

    rootBoxRangeProtocol root_box_range;
    root_box_range.setup(kzg, zero, index_routing);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestStructureProtocol forest_structure;
    forest_structure.setup(kzg,
                           zero,
                           column_replication,
                           root_box_range,
                           root_selector_consistency,
                           node_role_consistency,
                           forest_depth_consistency,
                           parent_child_box_inheritance,
                           parent_child_tree_consistency);

    forestTrainingInferenceProtocol training_inference;
    training_inference.setup(kzg, less, box_check, concatenated_lookup);

    forestHistogramBucketAddressProtocol bucket_address;
    bucket_address.setup(zero, column_replication, less);

    forestHistogramProtocol forest_histogram;
    forest_histogram.setup(hist_agg, tree_histogram_consistency);

    forestHistogramRefreshProtocol histogram_refresh;
    histogram_refresh.setup(training_inference, bucket_address, column_replication, forest_histogram);

    forestSplitStatisticsProtocol split_statistics;
    split_statistics.setup(kzg, zero, row_prefix, row_sum, column_replication);

    forestSplitGainProtocol split_gain;
    split_gain.setup(zero, less, signed_range);

    fixedPointDivisionProtocol fixed_division;
    fixed_division.setup(kzg, zero, less, signed_range);

    forestQuantizedSplitGainProtocol quantized_split_gain;
    quantized_split_gain.setup(zero, fixed_division);

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol split_selection;
    split_selection.setup(masked_argmax, row_sum, zero, less);

    forestLeafValueProtocol leaf_value;
    leaf_value.setup(zero, less, fixed_division);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol cube_replication;
    cube_replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, cube_replication, less);

    cubeMiddleSoftmaxProtocol cube_softmax;
    cube_softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, cube_replication);

    forestBoostingStateProtocol boosting_state;
    boosting_state.setup(zero,
                         column_prefix,
                         column_sum,
                         cube_argmax,
                         cube_replication,
                         cube_sum,
                         exp_check,
                         fixed_division,
                         signed_range);

    forestTrainingProofProtocol protocol;
    protocol.setup(histogram_refresh,
                   split_statistics,
                   split_gain,
                   quantized_split_gain,
                   split_selection,
                   leaf_value,
                   boosting_state,
                   row_sum,
                   zero,
                   less);

    const forestTrainingProofInputs inputs = valid_inputs();

    protocol.add_check(valid_witness(kzg, inputs),
                       inputs.tree_count,
                       inputs.node_count,
                       inputs.feature_count,
                       inputs.bin_count,
                       inputs.iteration_count,
                       inputs.class_count,
                       inputs.query_count,
                       inputs.input_bit_bound,
                       inputs.full_upper_bound,
                       inputs.structure_bit_bound,
                       inputs.gradient_encoding,
                       inputs.gain_numerator_encoding,
                       inputs.quantized_gain_encoding,
                       inputs.leaf_value_encoding,
                       inputs.score_encoding,
                       inputs.softmax_encoding,
                       inputs.hessian_bit_bound,
                       inputs.gain_denominator_bit_bound,
                       inputs.leaf_denominator_bit_bound,
                       inputs.gain_gap_bit_bound,
                       inputs.l2_regularization,
                       inputs.split_penalty);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(boosting_state.finalize());
    EXPECT_TRUE(leaf_value.finalize());
    EXPECT_TRUE(split_selection.finalize());
    EXPECT_TRUE(quantized_split_gain.finalize());
    EXPECT_TRUE(split_gain.finalize());
    EXPECT_TRUE(split_statistics.finalize());
    EXPECT_TRUE(histogram_refresh.finalize());
    EXPECT_TRUE(forest_histogram.finalize());
    EXPECT_TRUE(bucket_address.finalize());
    EXPECT_TRUE(training_inference.finalize());
    EXPECT_TRUE(forest_structure.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_TRUE(root_box_range.finalize());
    EXPECT_TRUE(root_selector_consistency.finalize());
    EXPECT_TRUE(node_role_consistency.finalize());
    EXPECT_TRUE(forest_depth_consistency.finalize());
    EXPECT_TRUE(parent_child_box_inheritance.finalize());
    EXPECT_TRUE(parent_child_tree_consistency.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_prefix.finalize());
    EXPECT_TRUE(row_prefix.finalize());
    EXPECT_TRUE(column_sum.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(fixed_division.finalize());
    EXPECT_TRUE(cube_softmax.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(cube_replication.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(box_check.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestTrainingProofProtocolTest, WrongResidualChannelFailsCrossProtocolConsistency)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    matrixColumnPrefixSumProtocol column_prefix;
    column_prefix.setup(kzg, zero, column_replication);

    matrixRowPrefixSumProtocol row_prefix;
    row_prefix.setup(kzg, zero);

    matrixColumnSumProtocol column_sum;
    column_sum.setup(kzg, zero);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    boxCheckProtocol box_check;
    box_check.setup(kzg, less);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    rootSelectorConsistencyProtocol root_selector_consistency;
    root_selector_consistency.setup(kzg, zero, hist_agg, index_routing);

    nodeRoleConsistencyProtocol node_role_consistency;
    node_role_consistency.setup(kzg, zero, hist_agg);

    forestDepthConsistencyProtocol forest_depth_consistency;
    forest_depth_consistency.setup(kzg, zero, index_routing);

    parentChildBoxInheritanceProtocol parent_child_box_inheritance;
    parent_child_box_inheritance.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    parentChildTreeConsistencyProtocol parent_child_tree_consistency;
    parent_child_tree_consistency.setup(kzg, zero, index_routing);

    rootBoxRangeProtocol root_box_range;
    root_box_range.setup(kzg, zero, index_routing);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestStructureProtocol forest_structure;
    forest_structure.setup(kzg,
                           zero,
                           column_replication,
                           root_box_range,
                           root_selector_consistency,
                           node_role_consistency,
                           forest_depth_consistency,
                           parent_child_box_inheritance,
                           parent_child_tree_consistency);

    forestTrainingInferenceProtocol training_inference;
    training_inference.setup(kzg, less, box_check, concatenated_lookup);

    forestHistogramBucketAddressProtocol bucket_address;
    bucket_address.setup(zero, column_replication, less);

    forestHistogramProtocol forest_histogram;
    forest_histogram.setup(hist_agg, tree_histogram_consistency);

    forestHistogramRefreshProtocol histogram_refresh;
    histogram_refresh.setup(training_inference, bucket_address, column_replication, forest_histogram);

    forestSplitStatisticsProtocol split_statistics;
    split_statistics.setup(kzg, zero, row_prefix, row_sum, column_replication);

    forestSplitGainProtocol split_gain;
    split_gain.setup(zero, less, signed_range);

    fixedPointDivisionProtocol fixed_division;
    fixed_division.setup(kzg, zero, less, signed_range);

    forestQuantizedSplitGainProtocol quantized_split_gain;
    quantized_split_gain.setup(zero, fixed_division);

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol split_selection;
    split_selection.setup(masked_argmax, row_sum, zero, less);

    forestLeafValueProtocol leaf_value;
    leaf_value.setup(zero, less, fixed_division);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol cube_replication;
    cube_replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, cube_replication, less);

    cubeMiddleSoftmaxProtocol cube_softmax;
    cube_softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, cube_replication);

    forestBoostingStateProtocol boosting_state;
    boosting_state.setup(zero,
                         column_prefix,
                         column_sum,
                         cube_argmax,
                         cube_replication,
                         cube_sum,
                         exp_check,
                         fixed_division,
                         signed_range);

    forestTrainingProofProtocol protocol;
    protocol.setup(histogram_refresh,
                   split_statistics,
                   split_gain,
                   quantized_split_gain,
                   split_selection,
                   leaf_value,
                   boosting_state,
                   row_sum,
                   zero,
                   less);

    const forestTrainingProofInputs inputs = valid_inputs();
    forestTrainingProofWitness witness = valid_witness(kzg, inputs);
    witness.histogram_refresh.channels.query_values[1] = virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0})));

    protocol.add_check(witness,
                       inputs.tree_count,
                       inputs.node_count,
                       inputs.feature_count,
                       inputs.bin_count,
                       inputs.iteration_count,
                       inputs.class_count,
                       inputs.query_count,
                       inputs.input_bit_bound,
                       inputs.full_upper_bound,
                       inputs.structure_bit_bound,
                       inputs.gradient_encoding,
                       inputs.gain_numerator_encoding,
                       inputs.quantized_gain_encoding,
                       inputs.leaf_value_encoding,
                       inputs.score_encoding,
                       inputs.softmax_encoding,
                       inputs.hessian_bit_bound,
                       inputs.gain_denominator_bit_bound,
                       inputs.leaf_denominator_bit_bound,
                       inputs.gain_gap_bit_bound,
                       inputs.l2_regularization,
                       inputs.split_penalty);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(boosting_state.finalize());
    EXPECT_TRUE(leaf_value.finalize());
    EXPECT_TRUE(split_selection.finalize());
    EXPECT_TRUE(quantized_split_gain.finalize());
    EXPECT_TRUE(split_gain.finalize());
    EXPECT_TRUE(split_statistics.finalize());
    EXPECT_TRUE(histogram_refresh.finalize());
    EXPECT_TRUE(forest_histogram.finalize());
    EXPECT_TRUE(bucket_address.finalize());
    EXPECT_TRUE(training_inference.finalize());
    EXPECT_TRUE(forest_structure.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_TRUE(root_box_range.finalize());
    EXPECT_TRUE(root_selector_consistency.finalize());
    EXPECT_TRUE(node_role_consistency.finalize());
    EXPECT_TRUE(forest_depth_consistency.finalize());
    EXPECT_TRUE(parent_child_box_inheritance.finalize());
    EXPECT_TRUE(parent_child_tree_consistency.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_prefix.finalize());
    EXPECT_TRUE(row_prefix.finalize());
    EXPECT_TRUE(column_sum.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(fixed_division.finalize());
    EXPECT_TRUE(cube_softmax.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(cube_replication.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(box_check.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestTrainingProofProtocolTest, DelegatesNonZeroInternalNodeScoreToWellFormedness)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    matrixColumnPrefixSumProtocol column_prefix;
    column_prefix.setup(kzg, zero, column_replication);

    matrixRowPrefixSumProtocol row_prefix;
    row_prefix.setup(kzg, zero);

    matrixColumnSumProtocol column_sum;
    column_sum.setup(kzg, zero);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    boxCheckProtocol box_check;
    box_check.setup(kzg, less);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    rootSelectorConsistencyProtocol root_selector_consistency;
    root_selector_consistency.setup(kzg, zero, hist_agg, index_routing);

    nodeRoleConsistencyProtocol node_role_consistency;
    node_role_consistency.setup(kzg, zero, hist_agg);

    forestDepthConsistencyProtocol forest_depth_consistency;
    forest_depth_consistency.setup(kzg, zero, index_routing);

    parentChildBoxInheritanceProtocol parent_child_box_inheritance;
    parent_child_box_inheritance.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    parentChildTreeConsistencyProtocol parent_child_tree_consistency;
    parent_child_tree_consistency.setup(kzg, zero, index_routing);

    rootBoxRangeProtocol root_box_range;
    root_box_range.setup(kzg, zero, index_routing);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestStructureProtocol forest_structure;
    forest_structure.setup(kzg,
                           zero,
                           column_replication,
                           root_box_range,
                           root_selector_consistency,
                           node_role_consistency,
                           forest_depth_consistency,
                           parent_child_box_inheritance,
                           parent_child_tree_consistency);

    forestTrainingInferenceProtocol training_inference;
    training_inference.setup(kzg, less, box_check, concatenated_lookup);

    forestHistogramBucketAddressProtocol bucket_address;
    bucket_address.setup(zero, column_replication, less);

    forestHistogramProtocol forest_histogram;
    forest_histogram.setup(hist_agg, tree_histogram_consistency);

    forestHistogramRefreshProtocol histogram_refresh;
    histogram_refresh.setup(training_inference, bucket_address, column_replication, forest_histogram);

    forestSplitStatisticsProtocol split_statistics;
    split_statistics.setup(kzg, zero, row_prefix, row_sum, column_replication);

    forestSplitGainProtocol split_gain;
    split_gain.setup(zero, less, signed_range);

    fixedPointDivisionProtocol fixed_division;
    fixed_division.setup(kzg, zero, less, signed_range);

    forestQuantizedSplitGainProtocol quantized_split_gain;
    quantized_split_gain.setup(zero, fixed_division);

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol split_selection;
    split_selection.setup(masked_argmax, row_sum, zero, less);

    forestLeafValueProtocol leaf_value;
    leaf_value.setup(zero, less, fixed_division);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol cube_replication;
    cube_replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, cube_replication, less);

    cubeMiddleSoftmaxProtocol cube_softmax;
    cube_softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, cube_replication);

    forestBoostingStateProtocol boosting_state;
    boosting_state.setup(zero,
                         column_prefix,
                         column_sum,
                         cube_argmax,
                         cube_replication,
                         cube_sum,
                         exp_check,
                         fixed_division,
                         signed_range);

    forestTrainingProofProtocol protocol;
    protocol.setup(histogram_refresh,
                   split_statistics,
                   split_gain,
                   quantized_split_gain,
                   split_selection,
                   leaf_value,
                   boosting_state,
                   row_sum,
                   zero,
                   less);

    const forestTrainingProofInputs inputs = valid_inputs();
    forestTrainingProofWitness witness = valid_witness(kzg, inputs);
    witness.histogram_refresh.inference.node_leaf_values =
        virCommitment(kzg.commit(poly_from_domain_ints({7, 0, 0, 0})));

    protocol.add_check(witness,
                       inputs.tree_count,
                       inputs.node_count,
                       inputs.feature_count,
                       inputs.bin_count,
                       inputs.iteration_count,
                       inputs.class_count,
                       inputs.query_count,
                       inputs.input_bit_bound,
                       inputs.full_upper_bound,
                       inputs.structure_bit_bound,
                       inputs.gradient_encoding,
                       inputs.gain_numerator_encoding,
                       inputs.quantized_gain_encoding,
                       inputs.leaf_value_encoding,
                       inputs.score_encoding,
                       inputs.softmax_encoding,
                       inputs.hessian_bit_bound,
                       inputs.gain_denominator_bit_bound,
                       inputs.leaf_denominator_bit_bound,
                       inputs.gain_gap_bit_bound,
                       inputs.l2_regularization,
                       inputs.split_penalty);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(boosting_state.finalize());
    EXPECT_TRUE(leaf_value.finalize());
    EXPECT_TRUE(split_selection.finalize());
    EXPECT_TRUE(quantized_split_gain.finalize());
    EXPECT_TRUE(split_gain.finalize());
    EXPECT_TRUE(split_statistics.finalize());
    EXPECT_TRUE(histogram_refresh.finalize());
    EXPECT_TRUE(forest_histogram.finalize());
    EXPECT_TRUE(bucket_address.finalize());
    EXPECT_TRUE(training_inference.finalize());
    EXPECT_TRUE(forest_structure.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_TRUE(root_box_range.finalize());
    EXPECT_TRUE(root_selector_consistency.finalize());
    EXPECT_TRUE(node_role_consistency.finalize());
    EXPECT_TRUE(forest_depth_consistency.finalize());
    EXPECT_TRUE(parent_child_box_inheritance.finalize());
    EXPECT_TRUE(parent_child_tree_consistency.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_prefix.finalize());
    EXPECT_TRUE(row_prefix.finalize());
    EXPECT_TRUE(column_sum.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(fixed_division.finalize());
    EXPECT_TRUE(cube_softmax.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(cube_replication.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(box_check.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
