#pragma once

#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "forest_training_trace_adapter.hpp"
#include "forest_training_trace_io.hpp"
#include "forest_training_then_inference_workflow.hpp"

#include <vector>

using namespace mcl::bn;

inline poly example_demo_poly_from_domain_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

inline poly example_demo_matrix_poly_from_rows(const std::vector<std::vector<int>>& rows)
{
    std::vector<Fr> evals;
    for (const auto& row : rows) {
        for (int value : row) {
            evals.push_back(fr_from_int(value));
        }
    }
    return poly_from_evals(evals);
}

inline poly example_demo_cube_poly_from_slices(const std::vector<std::vector<std::vector<int>>>& cube)
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

inline poly example_demo_fixed_point_encoded_matrix_poly_from_rows(const std::vector<std::vector<int>>& rows,
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

inline forestTrainingSemanticTrace build_demo_training_trace()
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

inline forestTrainingSemanticTrace build_demo_first_order_training_trace()
{
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    trace.loss_mode = forestTrainingLoss::FirstOrderRss;
    trace.l2_regularization = 0;
    trace.split_penalty = Fr(0);
    trace.hessian_bit_bound = 5;
    trace.gain_denominator_bit_bound = 10;
    trace.leaf_denominator_bit_bound = 5;
    return trace;
}

inline forestTrainingProofInputs build_demo_training_inputs()
{
    return assemble_forest_training_inputs_from_trace(build_demo_training_trace());
}

inline forestSemanticInferenceQueryBatch build_demo_inference_query_trace()
{
    forestSemanticInferenceQueryBatch out;
    out.query_tree_ids = {1, 1, 1, 1};
    out.lifted_inputs = {1, 0, 1, 1};
    out.query_count = 4;
    out.feature_count = 1;
    return out;
}

inline forestTrainingSemanticTraceCase build_demo_training_trace_case()
{
    forestTrainingSemanticTraceCase out;
    out.training_trace = build_demo_training_trace();
    out.inference_queries = build_demo_inference_query_trace();
    return out;
}

inline forestInferenceQueryBatch build_demo_inference_query_batch()
{
    return assemble_forest_inference_query_batch_from_trace(build_demo_inference_query_trace());
}
