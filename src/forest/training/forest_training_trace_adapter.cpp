#include "forest_training_trace_adapter.hpp"

#include "algebra_utils.hpp"
#include "cube_softmax.hpp"
#include "fixed_point.hpp"
#include "forest_training_export.hpp"
#include "matrix_utils.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

class DerivedForestBoxes {
public:
    std::vector<std::int64_t> lower_bounds;
    std::vector<std::int64_t> upper_bounds;
    std::vector<std::int64_t> left_lower;
    std::vector<std::int64_t> left_upper;
    std::vector<std::int64_t> right_lower;
    std::vector<std::int64_t> right_upper;
};

class DerivedForestDepths {
public:
    std::vector<std::int64_t> depths;
};

Fr fr_from_int64(std::int64_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

poly poly_from_fr_evals(const std::vector<Fr>& evals)
{
    return poly_from_evals(evals);
}

poly poly_from_int64_values(const std::vector<std::int64_t>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (std::int64_t value : values) {
        evals.push_back(fr_from_int64(value));
    }
    return poly_from_evals(evals);
}

size_t checked_non_negative_size_t(std::int64_t value, const char* context)
{
    if (value < 0) {
        throw std::runtime_error(std::string(context) + ": expected a non-negative integer");
    }
    return static_cast<size_t>(value);
}

size_t checked_size_product(size_t left, size_t right, const char* context)
{
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        throw std::runtime_error(std::string(context) + ": size overflow");
    }
    return left * right;
}

std::int64_t checked_scaled_product(std::int64_t value, size_t multiplier, bool negate, const char* context)
{
    const cpp_int product = (negate ? -cpp_int(value) : cpp_int(value)) * multiplier;
    if (product < cpp_int(std::numeric_limits<std::int64_t>::min()) ||
        product > cpp_int(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error(std::string(context) + ": int64 overflow");
    }
    return product.convert_to<std::int64_t>();
}

void validate_trace(const forestTrainingSemanticTrace& trace)
{
    if (trace.tree_count == 0 || trace.node_count == 0 || trace.feature_count == 0 || trace.bin_count == 0 ||
        trace.iteration_count == 0 || trace.class_count == 0 || trace.query_count == 0) {
        throw std::runtime_error("forest_training_trace_adapter: shape parameters must be non-zero");
    }
    if (!isPowerOfTwo(trace.tree_count) || !isPowerOfTwo(trace.node_count) || !isPowerOfTwo(trace.feature_count) ||
        !isPowerOfTwo(trace.bin_count) || !isPowerOfTwo(trace.query_count)) {
        throw std::runtime_error(
            "forest_training_trace_adapter: tree/node/feature/bin/query counts must be powers of two");
    }
    if (trace.tree_count != trace.iteration_count * trace.class_count) {
        throw std::runtime_error("forest_training_trace_adapter: tree_count must equal iteration_count * class_count");
    }
    if (trace.max_depth == 0) {
        throw std::runtime_error("forest_training_trace_adapter: max_depth must be positive because the root has depth one");
    }

    if (trace.learning_rate_numerator == 0 || trace.learning_rate_denominator == 0) {
        throw std::runtime_error("forest_training_trace_adapter: learning rate terms must be non-zero");
    }
    if (trace.loss_mode == forestTrainingLoss::FirstOrderRss &&
        (trace.l2_regularization != 0 || trace.split_penalty != Fr(0))) {
        throw std::runtime_error(
            "forest_training_trace_adapter: first-order RSS requires zero L2 regularization and split penalty");
    }
    if (trace.loss_mode == forestTrainingLoss::SecondOrderXgboost && trace.l2_regularization == 0) {
        throw std::runtime_error(
            "forest_training_trace_adapter: second-order XGBoost requires non-zero L2 regularization");
    }

    validate_fixed_point_encoding(trace.gradient_encoding, "forest_training_trace_adapter gradient");
    validate_fixed_point_encoding(trace.gain_numerator_encoding, "forest_training_trace_adapter gain numerator");
    validate_fixed_point_encoding(trace.quantized_gain_encoding, "forest_training_trace_adapter quantized gain");
    validate_fixed_point_encoding(trace.leaf_value_encoding, "forest_training_trace_adapter leaf value");
    validate_fixed_point_encoding(trace.score_encoding, "forest_training_trace_adapter score");
    validate_fixed_point_encoding(trace.softmax_encoding, "forest_training_trace_adapter softmax");

    const size_t query_row_count = trace.tree_count * trace.query_count;
    const size_t node_feature_count = trace.node_count * trace.feature_count;
    const size_t candidate_count = trace.node_count * trace.feature_count * trace.bin_count;

    auto require_size = [](const std::vector<std::int64_t>& values, size_t expected, const char* context) {
        if (values.size() != expected) {
            throw std::runtime_error(std::string(context) + ": unexpected trace length");
        }
    };

    require_size(trace.root_indices, trace.tree_count, "forest_training_trace_adapter root_indices");
    require_size(trace.split_node_mask, trace.node_count, "forest_training_trace_adapter split_node_mask");
    require_size(trace.left_child_indices, trace.node_count, "forest_training_trace_adapter left_child_indices");
    require_size(trace.right_child_indices, trace.node_count, "forest_training_trace_adapter right_child_indices");
    require_size(trace.node_tree_ids, trace.node_count, "forest_training_trace_adapter node_tree_ids");
    require_size(trace.feature_selector, node_feature_count, "forest_training_trace_adapter feature_selector");
    require_size(trace.thresholds, trace.node_count, "forest_training_trace_adapter thresholds");
    require_size(trace.node_leaf_mask, trace.node_count, "forest_training_trace_adapter node_leaf_mask");
    require_size(trace.node_leaf_values, trace.node_count, "forest_training_trace_adapter node_leaf_values");
    require_size(
        trace.training_inputs, trace.query_count * trace.feature_count, "forest_training_trace_adapter training_inputs");
    require_size(
        trace.base_labels, trace.class_count * trace.query_count, "forest_training_trace_adapter base_labels");
    require_size(trace.leaf_indices, query_row_count, "forest_training_trace_adapter leaf_indices");
    require_size(trace.candidate_mask, candidate_count, "forest_training_trace_adapter candidate_mask");
    require_size(
        trace.selected_candidate_indices, trace.node_count, "forest_training_trace_adapter selected_candidate_indices");

    for (size_t node = 0; node < trace.node_count; ++node) {
        const size_t tree_id = checked_non_negative_size_t(
            trace.node_tree_ids[node], "forest_training_trace_adapter node_tree_ids");
        const bool is_dummy = trace.split_node_mask[node] == 0 && trace.node_leaf_mask[node] == 0;
        if (is_dummy) {
            if (tree_id != 0) {
                throw std::runtime_error(
                    "forest_training_trace_adapter: dummy node tree ID must be 0");
            }
        } else if (tree_id == 0 || tree_id > trace.tree_count) {
            throw std::runtime_error(
                "forest_training_trace_adapter: real node tree IDs must be in [1, tree_count]");
        }
    }
    for (size_t tree = 0; tree < trace.tree_count; ++tree) {
        const size_t root = checked_non_negative_size_t(
            trace.root_indices[tree], "forest_training_trace_adapter root_indices");
        if (root >= trace.node_count ||
            checked_non_negative_size_t(trace.node_tree_ids[root],
                                        "forest_training_trace_adapter root tree ID") != tree + 1) {
            throw std::runtime_error(
                "forest_training_trace_adapter: root must carry its one-based tree ID");
        }
    }

    for (size_t i = 0; i < trace.base_labels.size(); ++i) {
        const std::int64_t value = trace.base_labels[i];
        if (value != 0 && value != 1) {
            throw std::runtime_error("forest_training_trace_adapter: base_labels must be binary");
        }
    }
    for (size_t query = 0; query < trace.query_count; ++query) {
        std::int64_t column_sum = 0;
        for (size_t cls = 0; cls < trace.class_count; ++cls) {
            column_sum += trace.base_labels[cls * trace.query_count + query];
        }
        if (column_sum != 1) {
            throw std::runtime_error("forest_training_trace_adapter: base_labels columns must sum to one");
        }
    }
}

size_t selected_feature_for_node(const forestTrainingSemanticTrace& trace, size_t node)
{
    std::optional<size_t> selected_feature;
    for (size_t feature = 0; feature < trace.feature_count; ++feature) {
        const size_t offset = node * trace.feature_count + feature;
        const size_t value =
            checked_non_negative_size_t(trace.feature_selector[offset], "forest_training_trace_adapter feature_selector");
        if (value == 0) {
            continue;
        }
        if (selected_feature.has_value()) {
            throw std::runtime_error("forest_training_trace_adapter: split node has multiple selected features");
        }
        selected_feature = feature;
    }
    if (!selected_feature.has_value()) {
        throw std::runtime_error("forest_training_trace_adapter: split node is missing a selected feature");
    }
    return *selected_feature;
}

DerivedForestBoxes derive_forest_boxes(const forestTrainingSemanticTrace& trace)
{
    const size_t node_feature_count = trace.node_count * trace.feature_count;
    DerivedForestBoxes out;
    out.lower_bounds.assign(node_feature_count, 0);
    out.upper_bounds.assign(node_feature_count, 0);
    out.left_lower.assign(node_feature_count, 0);
    out.left_upper.assign(node_feature_count, 0);
    out.right_lower.assign(node_feature_count, 0);
    out.right_upper.assign(node_feature_count, 0);

    std::vector<bool> node_box_assigned(trace.node_count, false);

    auto assign_node_box = [&](size_t node,
                               const std::vector<std::int64_t>& lower,
                               const std::vector<std::int64_t>& upper,
                               const char* context) {
        if (node >= trace.node_count) {
            throw std::runtime_error(std::string(context) + ": node index out of range");
        }
        if (!node_box_assigned[node]) {
            for (size_t feature = 0; feature < trace.feature_count; ++feature) {
                const size_t offset = node * trace.feature_count + feature;
                out.lower_bounds[offset] = lower[feature];
                out.upper_bounds[offset] = upper[feature];
            }
            node_box_assigned[node] = true;
            return;
        }
        for (size_t feature = 0; feature < trace.feature_count; ++feature) {
            const size_t offset = node * trace.feature_count + feature;
            if (out.lower_bounds[offset] != lower[feature] || out.upper_bounds[offset] != upper[feature]) {
                throw std::runtime_error(std::string(context) + ": conflicting node box assignment");
            }
        }
    };

    std::vector<size_t> stack;
    stack.reserve(trace.node_count);

    for (size_t tree = 0; tree < trace.tree_count; ++tree) {
        const size_t root =
            checked_non_negative_size_t(trace.root_indices[tree], "forest_training_trace_adapter root_indices");
        std::vector<std::int64_t> root_lower(trace.feature_count, 0);
        std::vector<std::int64_t> root_upper(trace.feature_count, static_cast<std::int64_t>(trace.full_upper_bound));
        assign_node_box(root, root_lower, root_upper, "forest_training_trace_adapter root");
        stack.push_back(root);
    }

    while (!stack.empty()) {
        const size_t node = stack.back();
        stack.pop_back();

        const size_t split_mask =
            checked_non_negative_size_t(trace.split_node_mask[node], "forest_training_trace_adapter split_node_mask");
        if (split_mask == 0) {
            continue;
        }

        const size_t left_child =
            checked_non_negative_size_t(trace.left_child_indices[node], "forest_training_trace_adapter left_child_indices");
        const size_t right_child =
            checked_non_negative_size_t(trace.right_child_indices[node], "forest_training_trace_adapter right_child_indices");
        const size_t selected_feature = selected_feature_for_node(trace, node);
        const std::int64_t threshold = trace.thresholds[node];

        std::vector<std::int64_t> parent_lower(trace.feature_count, 0);
        std::vector<std::int64_t> parent_upper(trace.feature_count, 0);
        for (size_t feature = 0; feature < trace.feature_count; ++feature) {
            const size_t offset = node * trace.feature_count + feature;
            parent_lower[feature] = out.lower_bounds[offset];
            parent_upper[feature] = out.upper_bounds[offset];
        }

        if (threshold <= parent_lower[selected_feature] || threshold >= parent_upper[selected_feature]) {
            throw std::runtime_error("forest_training_trace_adapter: split threshold falls outside the parent box");
        }

        std::vector<std::int64_t> left_lower = parent_lower;
        std::vector<std::int64_t> left_upper = parent_upper;
        std::vector<std::int64_t> right_lower = parent_lower;
        std::vector<std::int64_t> right_upper = parent_upper;
        left_upper[selected_feature] = threshold;
        right_lower[selected_feature] = threshold;

        for (size_t feature = 0; feature < trace.feature_count; ++feature) {
            const size_t offset = node * trace.feature_count + feature;
            out.left_lower[offset] = left_lower[feature];
            out.left_upper[offset] = left_upper[feature];
            out.right_lower[offset] = right_lower[feature];
            out.right_upper[offset] = right_upper[feature];
        }

        assign_node_box(left_child, left_lower, left_upper, "forest_training_trace_adapter left child");
        assign_node_box(right_child, right_lower, right_upper, "forest_training_trace_adapter right child");
        stack.push_back(left_child);
        stack.push_back(right_child);
    }

    return out;
}

DerivedForestDepths derive_forest_depths(const forestTrainingSemanticTrace& trace)
{
    DerivedForestDepths out;
    out.depths.assign(trace.node_count, 0);
    std::vector<bool> node_depth_assigned(trace.node_count, false);

    auto assign_node_depth = [&](size_t node, std::int64_t depth, const char* context) {
        if (node >= trace.node_count) {
            throw std::runtime_error(std::string(context) + ": node index out of range");
        }
        if (!node_depth_assigned[node]) {
            out.depths[node] = depth;
            node_depth_assigned[node] = true;
            return;
        }
        if (out.depths[node] != depth) {
            throw std::runtime_error(std::string(context) + ": conflicting node depth assignment");
        }
    };

    std::vector<size_t> stack;
    stack.reserve(trace.node_count);
    for (size_t tree = 0; tree < trace.tree_count; ++tree) {
        const size_t root =
            checked_non_negative_size_t(trace.root_indices[tree], "forest_training_trace_adapter root_indices");
        assign_node_depth(root, 1, "forest_training_trace_adapter root depth");
        stack.push_back(root);
    }

    while (!stack.empty()) {
        const size_t node = stack.back();
        stack.pop_back();

        const size_t split_mask =
            checked_non_negative_size_t(trace.split_node_mask[node], "forest_training_trace_adapter split_node_mask");
        if (split_mask == 0) {
            continue;
        }

        const size_t left_child =
            checked_non_negative_size_t(trace.left_child_indices[node], "forest_training_trace_adapter left_child_indices");
        const size_t right_child =
            checked_non_negative_size_t(trace.right_child_indices[node], "forest_training_trace_adapter right_child_indices");
        const std::int64_t child_depth = out.depths[node] + 1;
        assign_node_depth(left_child, child_depth, "forest_training_trace_adapter left child depth");
        assign_node_depth(right_child, child_depth, "forest_training_trace_adapter right child depth");
        stack.push_back(left_child);
        stack.push_back(right_child);
    }

    return out;
}

void validate_semantic_inference_queries(const forestSemanticInferenceQueryBatch& trace)
{
    if (trace.query_count == 0 || trace.feature_count == 0) {
        throw std::runtime_error("forest_training_trace_adapter: inference query shape must be non-zero");
    }
    if (!isPowerOfTwo(trace.query_count) || !isPowerOfTwo(trace.feature_count)) {
        throw std::runtime_error("forest_training_trace_adapter: inference query shape must be powers of two");
    }
    if (trace.query_tree_ids.size() != trace.query_count) {
        throw std::runtime_error("forest_training_trace_adapter: inference query_tree_ids size mismatch");
    }
    if (trace.lifted_inputs.size() != trace.query_count * trace.feature_count) {
        throw std::runtime_error("forest_training_trace_adapter: inference lifted_inputs size mismatch");
    }
    for (std::int64_t sample_index : trace.sample_indices) {
        if (sample_index < 0) {
            throw std::runtime_error("forest_training_trace_adapter: inference sample_indices must be non-negative");
        }
    }
}

std::vector<Fr> repeated_tree_id_evals(size_t tree_count, size_t query_count)
{
    std::vector<Fr> evals;
    evals.reserve(tree_count * query_count);
    for (size_t tree = 0; tree < tree_count; ++tree) {
        for (size_t query = 0; query < query_count; ++query) {
            (void)query;
            evals.push_back(field_from_size_t(tree + 1));
        }
    }
    return evals;
}

std::vector<Fr> repeated_training_input_evals(const forestTrainingSemanticTrace& trace)
{
    std::vector<Fr> evals;
    evals.reserve(trace.tree_count * trace.query_count * trace.feature_count);
    for (size_t tree = 0; tree < trace.tree_count; ++tree) {
        (void)tree;
        for (std::int64_t value : trace.training_inputs) {
            evals.push_back(fr_from_int64(value));
        }
    }
    return evals;
}

size_t candidate_column(size_t feature, size_t bin, size_t bin_count)
{
    return feature * bin_count + bin;
}

std::vector<Fr> derive_leaf_bound_evals(const forestTrainingSemanticTrace& trace,
                                       const DerivedForestBoxes& boxes,
                                       bool upper)
{
    std::vector<Fr> evals;
    evals.reserve(trace.tree_count * trace.query_count * trace.feature_count);
    for (std::int64_t leaf_index_value : trace.leaf_indices) {
        const size_t leaf_index = checked_non_negative_size_t(leaf_index_value, "derive_leaf_bound_evals leaf_index");
        if (leaf_index >= trace.node_count) {
            throw std::runtime_error("derive_leaf_bound_evals: leaf index out of range");
        }
        for (size_t feature = 0; feature < trace.feature_count; ++feature) {
            const size_t offset = leaf_index * trace.feature_count + feature;
            evals.push_back(fr_from_int64(upper ? boxes.upper_bounds[offset] : boxes.lower_bounds[offset]));
        }
    }
    return evals;
}

std::vector<Fr> derive_tree_score_evals(const forestTrainingSemanticTrace& trace)
{
    std::vector<Fr> evals;
    evals.reserve(trace.tree_count * trace.query_count);
    for (std::int64_t leaf_index_value : trace.leaf_indices) {
        const size_t leaf_index =
            checked_non_negative_size_t(leaf_index_value, "derive_tree_score_evals leaf_index");
        if (leaf_index >= trace.node_count) {
            throw std::runtime_error("derive_tree_score_evals: leaf index out of range");
        }
        evals.push_back(fr_from_int64(trace.node_leaf_values[leaf_index]));
    }
    return evals;
}

std::vector<Fr> derive_lifted_leaf_index_evals(const forestTrainingSemanticTrace& trace)
{
    std::vector<Fr> evals;
    evals.reserve(trace.tree_count * trace.query_count * trace.feature_count);
    for (std::int64_t leaf_index_value : trace.leaf_indices) {
        const Fr field_value = fr_from_int64(leaf_index_value);
        for (size_t feature = 0; feature < trace.feature_count; ++feature) {
            (void)feature;
            evals.push_back(field_value);
        }
    }
    return evals;
}

std::vector<Fr> derive_flattened_bucket_index_evals(const forestTrainingSemanticTrace& trace)
{
    std::vector<Fr> evals;
    evals.reserve(trace.tree_count * trace.query_count * trace.feature_count);
    const size_t row_width = trace.feature_count * trace.bin_count;
    for (size_t query_row = 0; query_row < trace.tree_count * trace.query_count; ++query_row) {
        const size_t sample = query_row % trace.query_count;
        const size_t leaf_index =
            checked_non_negative_size_t(trace.leaf_indices[query_row], "derive_flattened_bucket_index_evals leaf");
        for (size_t feature = 0; feature < trace.feature_count; ++feature) {
            const size_t lifted_index = sample * trace.feature_count + feature;
            const size_t bin = checked_non_negative_size_t(
                trace.training_inputs[lifted_index], "derive_flattened_bucket_index_evals bin");
            if (bin >= trace.bin_count) {
                throw std::runtime_error("derive_flattened_bucket_index_evals: feature bin out of range");
            }
            evals.push_back(field_from_size_t(leaf_index * row_width + candidate_column(feature, bin, trace.bin_count)));
        }
    }
    return evals;
}

poly x_power_poly_for_adapter(const poly& base, size_t exponent)
{
    if (exponent == 0) {
        poly constant;
        constant.add_term(0, base.eval(Fr(1)));
        return constant;
    }

    std::vector<Fr> coeffs = base.coefficients();
    if (coeffs.empty()) {
        return poly();
    }

    std::vector<Fr> composed((coeffs.size() - 1) * exponent + 1, Fr(0));
    for (size_t i = 0; i < coeffs.size(); ++i) {
        composed[i * exponent] = coeffs[i];
    }
    return poly_from_coeff(composed);
}

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

Fr fr_from_cpp_int_checked(const cpp_int& value)
{
    const std::string repr = value.convert_to<std::string>();
    Fr out;
    out.setStr(repr, 10);
    if (out.getStr(10) != repr) {
        throw std::runtime_error("forest_training_trace_adapter: non-canonical cpp_int conversion");
    }
    return out;
}

Fr encode_fixed_point_fr_checked(std::int64_t scaled_value,
                                 const fixedPointEncoding& encoding,
                                 const std::string& context)
{
    try {
        return fixed_point_fr_from_size_t(encode_fixed_point_scaled_integer(scaled_value, encoding));
    } catch (const std::exception& e) {
        throw std::runtime_error(context + ": " + e.what());
    }
}

cpp_int canonical_cpp_int_from_fr(const Fr& value, const char* context)
{
    const std::string repr = value.getStr(10);
    cpp_int out(repr);
    if (fr_from_cpp_int_checked(out) != value) {
        throw std::runtime_error(std::string(context) + ": field element is not a canonical integer");
    }
    return out;
}

std::vector<Fr> derive_cumulative_score_evals(const std::vector<Fr>& tree_scores,
                                              size_t iteration_count,
                                              size_t class_count,
                                              size_t query_count)
{
    std::vector<Fr> out(tree_scores.size(), Fr(0));
    const size_t row_width = class_count * query_count;
    std::vector<Fr> running(row_width, Fr(0));
    for (size_t iteration = 0; iteration < iteration_count; ++iteration) {
        for (size_t column = 0; column < row_width; ++column) {
            const size_t index = iteration * row_width + column;
            out[index] = running[column];
            running[column] += tree_scores[index];
        }
    }
    return out;
}

std::vector<Fr> encode_fixed_point_field_vector(const std::vector<Fr>& values,
                                                const fixedPointEncoding& encoding,
                                                const char* context)
{
    std::vector<Fr> out;
    out.reserve(values.size());
    for (const Fr& value : values) {
        const std::int64_t scaled_value = canonical_signed_int64_from_fr(value, context);
        out.push_back(encode_fixed_point_fr_checked(scaled_value, encoding, context));
    }
    return out;
}

struct DerivedBoostingState {
    std::vector<Fr> tree_scores;
    std::vector<Fr> cumulative_scores;
    std::vector<Fr> encoded_cumulative_scores;
    std::vector<Fr> softmax_shifts;
    std::vector<Fr> shift_selector;
    std::vector<Fr> lifted_softmax_shifts;
    std::vector<Fr> encoded_shifted_scores;
    std::vector<Fr> softmax_exp_values;
    std::vector<Fr> softmax_denominators;
    std::vector<Fr> lifted_softmax_denominators;
    std::vector<Fr> softmax_probabilities;
    std::vector<Fr> softmax_remainders;
    std::vector<Fr> pseudo_residuals;
    std::vector<Fr> encoded_diagonal_hessians;
    std::vector<Fr> diagonal_hessian_remainders;
    std::vector<Fr> diagonal_hessians;
};

DerivedBoostingState derive_boosting_state(const forestTrainingSemanticTrace& trace)
{
    DerivedBoostingState out;
    out.tree_scores = derive_tree_score_evals(trace);
    out.cumulative_scores =
        derive_cumulative_score_evals(out.tree_scores, trace.iteration_count, trace.class_count, trace.query_count);
    out.encoded_cumulative_scores =
        encode_fixed_point_field_vector(out.cumulative_scores, trace.score_encoding, "derive_boosting_state cumulative_scores");

    out.softmax_shifts.reserve(trace.iteration_count * trace.query_count);
    out.shift_selector.assign(trace.iteration_count * trace.class_count * trace.query_count, Fr(0));
    for (size_t iteration = 0; iteration < trace.iteration_count; ++iteration) {
        for (size_t query = 0; query < trace.query_count; ++query) {
            size_t best_class = 0;
            Fr best_value = out.encoded_cumulative_scores[iteration * trace.class_count * trace.query_count + query];
            for (size_t cls = 1; cls < trace.class_count; ++cls) {
                const size_t index = iteration * trace.class_count * trace.query_count + cls * trace.query_count + query;
                if (best_value.isLess(out.encoded_cumulative_scores[index])) {
                    best_class = cls;
                    best_value = out.encoded_cumulative_scores[index];
                }
            }
            out.softmax_shifts.push_back(best_value);
            const size_t selector_index =
                iteration * trace.class_count * trace.query_count + best_class * trace.query_count + query;
            out.shift_selector[selector_index] = Fr(1);
        }
    }

    out.lifted_softmax_shifts.reserve(out.encoded_cumulative_scores.size());
    out.encoded_shifted_scores.reserve(out.encoded_cumulative_scores.size());
    for (size_t iteration = 0; iteration < trace.iteration_count; ++iteration) {
        for (size_t cls = 0; cls < trace.class_count; ++cls) {
            for (size_t query = 0; query < trace.query_count; ++query) {
                const size_t cube_index =
                    iteration * trace.class_count * trace.query_count + cls * trace.query_count + query;
                const size_t shift_index = iteration * trace.query_count + query;
                out.lifted_softmax_shifts.push_back(out.softmax_shifts[shift_index]);
                const std::int64_t cumulative_scaled = canonical_signed_int64_from_fr(
                    out.cumulative_scores[cube_index], "derive_boosting_state cumulative_scores");
                const std::int64_t shift_scaled = decode_fixed_point_scaled_integer(
                    fixed_point_size_t_from_fr(out.softmax_shifts[shift_index], "derive_boosting_state softmax_shifts"),
                    trace.score_encoding);
                const std::int64_t shifted_scaled = cumulative_scaled - shift_scaled;
                out.encoded_shifted_scores.push_back(encode_fixed_point_fr_checked(
                    shifted_scaled, trace.score_encoding, "derive_boosting_state encoded_shifted_scores"));
            }
        }
    }

    out.softmax_exp_values.reserve(out.encoded_shifted_scores.size());
    for (const Fr& encoded_shifted_score : out.encoded_shifted_scores) {
        const size_t encoded =
            fixed_point_size_t_from_fr(encoded_shifted_score, "derive_boosting_state encoded_shifted_scores");
        const std::int64_t scaled_score = decode_fixed_point_scaled_integer(encoded, trace.score_encoding);
        out.softmax_exp_values.push_back(
            fr_from_cpp_int_checked(quantized_exp_output(scaled_score, trace.score_encoding.fractional_bits)));
    }

    out.softmax_denominators.reserve(trace.iteration_count * trace.query_count);
    out.lifted_softmax_denominators.reserve(out.softmax_exp_values.size());
    out.softmax_probabilities.assign(out.softmax_exp_values.size(), Fr(0));
    out.softmax_remainders.assign(out.softmax_exp_values.size(), Fr(0));
    const std::int64_t softmax_scale = static_cast<std::int64_t>(size_t(1) << trace.softmax_encoding.fractional_bits);
    const fixedPointEncoding hessian_encoding = fixedPointEncoding::signed_offset_binary(
        std::max(trace.hessian_bit_bound + 1, trace.softmax_encoding.fractional_bits + 1),
        trace.softmax_encoding.fractional_bits);
    auto floor_division = [](std::int64_t numerator, std::int64_t denominator) -> std::pair<std::int64_t, std::int64_t> {
        if (denominator <= 0) {
            throw std::runtime_error("derive_boosting_state: denominator must be positive");
        }
        std::int64_t quotient = numerator / denominator;
        std::int64_t remainder = numerator % denominator;
        if (remainder < 0) {
            quotient -= 1;
            remainder += denominator;
        }
        return {quotient, remainder};
    };
    for (size_t iteration = 0; iteration < trace.iteration_count; ++iteration) {
        for (size_t query = 0; query < trace.query_count; ++query) {
            cpp_int denominator(0);
            for (size_t cls = 0; cls < trace.class_count; ++cls) {
                const size_t index =
                    iteration * trace.class_count * trace.query_count + cls * trace.query_count + query;
                denominator += canonical_cpp_int_from_fr(
                    out.softmax_exp_values[index], "derive_boosting_state softmax_exp_values");
            }
            const Fr denominator_fr = fr_from_cpp_int_checked(denominator);
            out.softmax_denominators.push_back(denominator_fr);
        }
        for (size_t cls = 0; cls < trace.class_count; ++cls) {
            for (size_t query = 0; query < trace.query_count; ++query) {
                const size_t index =
                    iteration * trace.class_count * trace.query_count + cls * trace.query_count + query;
                const Fr denominator_fr = out.softmax_denominators[iteration * trace.query_count + query];
                out.lifted_softmax_denominators.push_back(denominator_fr);
                const std::int64_t denominator_i64 = canonical_signed_int64_from_fr(
                    denominator_fr, "derive_boosting_state softmax_denominators");
                const std::int64_t exp_value = canonical_signed_int64_from_fr(
                    out.softmax_exp_values[index], "derive_boosting_state softmax_exp_values");
                const auto [probability_scaled, probability_remainder] =
                    floor_division(exp_value * softmax_scale, denominator_i64);
                out.softmax_probabilities[index] = encode_fixed_point_fr_checked(
                    probability_scaled, trace.softmax_encoding, "derive_boosting_state softmax_probabilities");
                out.softmax_remainders[index] = field_from_size_t(static_cast<size_t>(probability_remainder));
            }
        }
    }

    const poly base_labels = poly_from_int64_values(trace.base_labels);
    const std::vector<Fr> lifted_labels =
        x_power_poly_for_adapter(base_labels, trace.iteration_count)
            .evaluations_on_domain(trace.iteration_count * trace.class_count * trace.query_count);

    out.pseudo_residuals.assign(out.softmax_probabilities.size(), Fr(0));
    out.encoded_diagonal_hessians.assign(out.softmax_probabilities.size(), Fr(0));
    out.diagonal_hessian_remainders.assign(out.softmax_probabilities.size(), Fr(0));
    out.diagonal_hessians.assign(out.softmax_probabilities.size(), Fr(0));
    for (size_t i = 0; i < out.softmax_probabilities.size(); ++i) {
        const std::int64_t probability_scaled =
            decode_fixed_point_scaled_integer(fixed_point_size_t_from_fr(out.softmax_probabilities[i], "derive_boosting_state softmax_probabilities"),
                                              trace.softmax_encoding);
        const std::int64_t lifted_label_scaled =
            canonical_signed_int64_from_fr(lifted_labels[i], "derive_boosting_state lifted_labels") * softmax_scale;
        const std::int64_t pseudo_residual_scaled =
            trace.loss_mode == forestTrainingLoss::FirstOrderRss
                ? lifted_label_scaled - probability_scaled
                : probability_scaled - lifted_label_scaled;
        out.pseudo_residuals[i] = fr_from_int64(pseudo_residual_scaled);

        std::int64_t hessian_scaled = softmax_scale;
        std::int64_t hessian_remainder = 0;
        if (trace.loss_mode == forestTrainingLoss::SecondOrderXgboost) {
            const auto division =
                floor_division(probability_scaled * (softmax_scale - probability_scaled), softmax_scale);
            hessian_scaled = division.first;
            hessian_remainder = division.second;
        }
        out.diagonal_hessians[i] = fr_from_int64(hessian_scaled);
        out.diagonal_hessian_remainders[i] = field_from_size_t(static_cast<size_t>(hessian_remainder));
        out.encoded_diagonal_hessians[i] = encode_fixed_point_fr_checked(
            hessian_scaled, hessian_encoding, "derive_boosting_state encoded_diagonal_hessians");
    }

    return out;
}

std::vector<Fr> derive_count_query_values(size_t query_row_count)
{
    return std::vector<Fr>(query_row_count, Fr(1));
}

std::vector<std::vector<Fr>> derive_leaf_histograms(const forestTrainingSemanticTrace& trace,
                                                    const std::vector<std::vector<Fr>>& query_channels)
{
    const size_t query_row_count = trace.tree_count * trace.query_count;
    const size_t histogram_width = trace.feature_count * trace.bin_count;
    std::vector<std::vector<Fr>> out(query_channels.size(), std::vector<Fr>(trace.node_count * histogram_width, Fr(0)));

    for (size_t query_row = 0; query_row < query_row_count; ++query_row) {
        const size_t sample = query_row % trace.query_count;
        const size_t leaf_index =
            checked_non_negative_size_t(trace.leaf_indices[query_row], "derive_leaf_histograms leaf_index");
        if (leaf_index >= trace.node_count) {
            throw std::runtime_error("derive_leaf_histograms: leaf index out of range");
        }

        for (size_t feature = 0; feature < trace.feature_count; ++feature) {
            const size_t bin = checked_non_negative_size_t(
                trace.training_inputs[sample * trace.feature_count + feature], "derive_leaf_histograms bin");
            if (bin >= trace.bin_count) {
                throw std::runtime_error("derive_leaf_histograms: bin out of range");
            }
            const size_t offset = leaf_index * histogram_width + candidate_column(feature, bin, trace.bin_count);
            for (size_t channel = 0; channel < query_channels.size(); ++channel) {
                out[channel][offset] += query_channels[channel][query_row];
            }
        }
    }

    return out;
}

std::vector<Fr> full_histogram_for_channel(const forestTrainingSemanticTrace& trace,
                                           const std::vector<Fr>& leaf_histogram)
{
    const size_t width = trace.feature_count * trace.bin_count;
    std::vector<Fr> out(trace.node_count * width, Fr(0));
    std::vector<bool> computed(trace.node_count, false);

    std::function<void(size_t)> compute_row = [&](size_t node) {
        if (computed[node]) {
            return;
        }

        for (size_t col = 0; col < width; ++col) {
            out[node * width + col] = leaf_histogram[node * width + col];
        }

        if (checked_non_negative_size_t(trace.split_node_mask[node], "full_histogram_for_channel split mask") != 0) {
            const size_t left_child =
                checked_non_negative_size_t(trace.left_child_indices[node], "full_histogram_for_channel left child");
            const size_t right_child =
                checked_non_negative_size_t(trace.right_child_indices[node], "full_histogram_for_channel right child");
            if (left_child >= trace.node_count || right_child >= trace.node_count) {
                throw std::runtime_error("full_histogram_for_channel: child index out of range");
            }
            compute_row(left_child);
            compute_row(right_child);
            for (size_t col = 0; col < width; ++col) {
                out[node * width + col] += out[left_child * width + col] + out[right_child * width + col];
            }
        }

        computed[node] = true;
    };

    for (size_t node = 0; node < trace.node_count; ++node) {
        compute_row(node);
    }

    return out;
}

std::vector<Fr> derive_left_statistics(const std::vector<Fr>& histogram,
                                       size_t node_count,
                                       size_t feature_count,
                                       size_t bin_count)
{
    std::vector<Fr> out(node_count * feature_count * bin_count, Fr(0));
    for (size_t node = 0; node < node_count; ++node) {
        for (size_t feature = 0; feature < feature_count; ++feature) {
            Fr running(0);
            for (size_t bin = 0; bin < bin_count; ++bin) {
                const size_t index = node * feature_count * bin_count + feature * bin_count + bin;
                running += histogram[index];
                out[index] = running;
            }
        }
    }
    return out;
}

std::vector<Fr> derive_right_statistics(const std::vector<Fr>& left_statistics,
                                        size_t node_count,
                                        size_t feature_count,
                                        size_t bin_count)
{
    std::vector<Fr> out(node_count * feature_count * bin_count, Fr(0));
    for (size_t node = 0; node < node_count; ++node) {
        for (size_t feature = 0; feature < feature_count; ++feature) {
            const size_t total_index = node * feature_count * bin_count + feature * bin_count + (bin_count - 1);
            const Fr total = left_statistics[total_index];
            for (size_t bin = 0; bin < bin_count; ++bin) {
                const size_t index = node * feature_count * bin_count + feature * bin_count + bin;
                out[index] = total - left_statistics[index];
            }
        }
    }
    return out;
}

struct DerivedGainSelectionState {
    std::vector<Fr> leaf_histograms[3];
    std::vector<Fr> full_histograms[3];
    std::vector<Fr> left_statistics[3];
    std::vector<Fr> right_statistics[3];
    std::vector<Fr> left_gradient_squares;
    std::vector<Fr> right_gradient_squares;
    std::vector<Fr> parent_gradient_squares;
    std::vector<Fr> gain_numerators;
    std::vector<Fr> gain_denominators;
    std::vector<Fr> encoded_gain_numerators;
    std::vector<Fr> encoded_quantized_gains;
    std::vector<Fr> gain_remainders;
    std::vector<Fr> quantized_gains;
    std::vector<Fr> tie_break_scores;
    std::vector<Fr> selector;
    std::vector<Fr> selected_gains;
    std::vector<Fr> selected_tie_break_scores;
    std::vector<Fr> selected_features;
    std::vector<Fr> selected_bins;
    std::vector<Fr> leaf_gradients;
    std::vector<Fr> leaf_hessians;
    std::vector<Fr> leaf_denominators;
    std::vector<Fr> encoded_negative_leaf_gradients;
    std::vector<Fr> encoded_leaf_values;
    std::vector<Fr> leaf_remainders;
};

std::pair<std::int64_t, std::int64_t> floor_division_with_remainder(std::int64_t numerator,
                                                                    std::int64_t denominator)
{
    if (denominator <= 0) {
        throw std::runtime_error("floor_division_with_remainder: denominator must be positive");
    }

    std::int64_t quotient = numerator / denominator;
    std::int64_t remainder = numerator % denominator;
    if (remainder < 0) {
        quotient -= 1;
        remainder += denominator;
    }
    return {quotient, remainder};
}

std::pair<std::int64_t, std::int64_t> quantized_fixed_point_division(std::int64_t numerator,
                                                                     size_t denominator,
                                                                     const fixedPointEncoding& numerator_encoding,
                                                                     const fixedPointEncoding& quotient_encoding)
{
    const size_t scale_power =
        quotient_encoding.fractional_bits - numerator_encoding.fractional_bits;
    if (scale_power >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error("quantized_fixed_point_division: scale power too large");
    }
    const std::int64_t scaled_numerator = numerator * static_cast<std::int64_t>(size_t(1) << scale_power);
    return floor_division_with_remainder(scaled_numerator, static_cast<std::int64_t>(denominator));
}

DerivedGainSelectionState derive_gain_selection_state(const forestTrainingSemanticTrace& trace,
                                                      const DerivedBoostingState& boosting)
{
    DerivedGainSelectionState out;
    const size_t query_row_count = trace.tree_count * trace.query_count;
    const size_t candidate_width = trace.feature_count * trace.bin_count;
    const size_t candidate_domain_size = trace.node_count * candidate_width;
    if (trace.softmax_encoding.fractional_bits >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error(
            "derive_gain_selection_state: softmax fractional bits exceed size_t scale range");
    }
    const Fr first_order_hessian_scale =
        field_from_size_t(size_t(1) << trace.softmax_encoding.fractional_bits);

    std::vector<std::vector<Fr>> query_channels(3);
    query_channels[0] = derive_count_query_values(query_row_count);
    query_channels[1] = boosting.pseudo_residuals;
    query_channels[2] = boosting.diagonal_hessians;

    const std::vector<std::vector<Fr>> leaf_histograms = derive_leaf_histograms(trace, query_channels);
    for (size_t channel = 0; channel < 3; ++channel) {
        out.leaf_histograms[channel] = leaf_histograms[channel];
        out.full_histograms[channel] = full_histogram_for_channel(trace, out.leaf_histograms[channel]);
        out.left_statistics[channel] =
            derive_left_statistics(out.full_histograms[channel], trace.node_count, trace.feature_count, trace.bin_count);
        out.right_statistics[channel] = derive_right_statistics(
            out.left_statistics[channel], trace.node_count, trace.feature_count, trace.bin_count);
    }

    out.left_gradient_squares.assign(candidate_domain_size, Fr(0));
    out.right_gradient_squares.assign(candidate_domain_size, Fr(0));
    out.parent_gradient_squares.assign(candidate_domain_size, Fr(0));
    out.gain_numerators.assign(candidate_domain_size, Fr(0));
    out.gain_denominators.assign(candidate_domain_size, Fr(0));
    out.encoded_gain_numerators.assign(candidate_domain_size, Fr(0));
    out.encoded_quantized_gains.assign(candidate_domain_size, Fr(0));
    out.gain_remainders.assign(candidate_domain_size, Fr(0));
    out.quantized_gains.assign(candidate_domain_size, Fr(0));
    out.tie_break_scores.assign(candidate_domain_size, Fr(0));
    out.selector.assign(candidate_domain_size, Fr(0));
    out.selected_gains.assign(trace.node_count, Fr(0));
    out.selected_tie_break_scores.assign(trace.node_count, Fr(0));
    out.selected_features.assign(trace.node_count, Fr(0));
    out.selected_bins.assign(trace.node_count, Fr(0));

    const Fr lambda = field_from_size_t(trace.l2_regularization);
    const size_t column_count = trace.feature_count * trace.bin_count;
    for (size_t node = 0; node < trace.node_count; ++node) {
        const size_t selected_index =
            checked_non_negative_size_t(trace.selected_candidate_indices[node], "derive_gain_selection_state selected index");
        bool candidate_row = false;
        for (size_t column = 0; column < column_count; ++column) {
            if (trace.candidate_mask[node * column_count + column] != 0) {
                candidate_row = true;
                break;
            }
        }
        if (selected_index >= column_count && candidate_row) {
            throw std::runtime_error("derive_gain_selection_state: selected candidate index out of range");
        }

        for (size_t column = 0; column < column_count; ++column) {
            const size_t index = node * column_count + column;
            const Fr left_gradient = out.left_statistics[1][index];
            const Fr right_gradient = out.right_statistics[1][index];
            const Fr left_hessian = out.left_statistics[2][index];
            const Fr right_hessian = out.right_statistics[2][index];
            const Fr parent_gradient = left_gradient + right_gradient;
            const Fr parent_hessian = left_hessian + right_hessian;

            out.left_gradient_squares[index] = left_gradient * left_gradient;
            out.right_gradient_squares[index] = right_gradient * right_gradient;
            out.parent_gradient_squares[index] = parent_gradient * parent_gradient;

            const Fr inactive_candidate =
                trace.loss_mode == forestTrainingLoss::FirstOrderRss && trace.candidate_mask[index] == 0
                    ? Fr(1)
                    : Fr(0);
            Fr left_den = left_hessian + lambda;
            Fr right_den = right_hessian + lambda;
            Fr parent_den = parent_hessian + lambda;
            Fr denominator_scale(1);
            if (trace.loss_mode == forestTrainingLoss::FirstOrderRss) {
                left_den = out.left_statistics[0][index] + inactive_candidate;
                right_den = out.right_statistics[0][index] + inactive_candidate;
                parent_den = out.left_statistics[0][index] + out.right_statistics[0][index] +
                             inactive_candidate;
                denominator_scale = first_order_hessian_scale;
            }
            out.gain_denominators[index] = left_den * right_den * parent_den * denominator_scale;
            out.gain_numerators[index] =
                out.left_gradient_squares[index] * right_den * parent_den +
                out.right_gradient_squares[index] * left_den * parent_den -
                out.parent_gradient_squares[index] * left_den * right_den -
                trace.split_penalty * out.gain_denominators[index];

            const std::int64_t gain_numerator_int =
                canonical_signed_int64_from_fr(out.gain_numerators[index], "derive_gain_selection_state gain numerator");
            const size_t gain_denominator_int =
                canonical_size_t_from_fr(out.gain_denominators[index], "derive_gain_selection_state gain denominator");
            out.encoded_gain_numerators[index] = encode_fixed_point_fr_checked(
                gain_numerator_int, trace.gain_numerator_encoding, "derive_gain_selection_state encoded_gain_numerators");

            const auto [quantized_gain, remainder] = quantized_fixed_point_division(
                gain_numerator_int, gain_denominator_int, trace.gain_numerator_encoding, trace.quantized_gain_encoding);
            out.quantized_gains[index] = fr_from_int64(quantized_gain);
            out.encoded_quantized_gains[index] = encode_fixed_point_fr_checked(
                quantized_gain, trace.quantized_gain_encoding, "derive_gain_selection_state encoded_quantized_gains");
            out.gain_remainders[index] = field_from_size_t(static_cast<size_t>(remainder));
            out.tie_break_scores[index] =
                out.quantized_gains[index] * field_from_size_t(column_count) + field_from_size_t(column_count - 1 - column);
        }

        if (candidate_row) {
            const size_t selector_index = node * column_count + selected_index;
            out.selector[selector_index] = Fr(1);
            out.selected_gains[node] = out.quantized_gains[selector_index];
            out.selected_tie_break_scores[node] = out.tie_break_scores[selector_index];
            out.selected_features[node] = field_from_size_t(selected_index / trace.bin_count);
            out.selected_bins[node] = field_from_size_t(selected_index % trace.bin_count);
        }
    }

    out.leaf_gradients.assign(trace.node_count, Fr(0));
    out.leaf_hessians.assign(trace.node_count, Fr(0));
    out.leaf_denominators.assign(trace.node_count, Fr(0));
    out.encoded_negative_leaf_gradients.assign(trace.node_count, Fr(0));
    out.encoded_leaf_values.assign(trace.node_count, Fr(0));
    out.leaf_remainders.assign(trace.node_count, Fr(0));
    for (size_t query_row = 0; query_row < query_row_count; ++query_row) {
        const size_t leaf_index =
            checked_non_negative_size_t(trace.leaf_indices[query_row], "derive_gain_selection_state leaf aggregation");
        out.leaf_gradients[leaf_index] += boosting.pseudo_residuals[query_row];
        out.leaf_hessians[leaf_index] += boosting.diagonal_hessians[query_row];
    }
    for (size_t node = 0; node < trace.node_count; ++node) {
        const Fr inactive_node =
            trace.loss_mode == forestTrainingLoss::FirstOrderRss && trace.node_leaf_mask[node] == 0 ? Fr(1) : Fr(0);
        const size_t base_denominator = canonical_size_t_from_fr(
            out.leaf_hessians[node] + field_from_size_t(trace.l2_regularization) + inactive_node,
            "derive_gain_selection_state leaf denominator");
        const size_t denominator = checked_size_product(
            base_denominator,
            trace.learning_rate_denominator,
            "derive_gain_selection_state scaled leaf denominator");
        out.leaf_denominators[node] = field_from_size_t(denominator);

        const std::int64_t leaf_gradient = canonical_signed_int64_from_fr(
            out.leaf_gradients[node], "derive_gain_selection_state leaf gradient");
        const std::int64_t negative_gradient = checked_scaled_product(
            leaf_gradient,
            trace.learning_rate_numerator,
            trace.loss_mode == forestTrainingLoss::SecondOrderXgboost,
            "derive_gain_selection_state scaled leaf gradient");
        out.encoded_negative_leaf_gradients[node] = encode_fixed_point_fr_checked(
            negative_gradient, trace.gradient_encoding, "derive_gain_selection_state encoded_negative_leaf_gradients");

        const auto [leaf_value, remainder] = quantized_fixed_point_division(
            negative_gradient, denominator, trace.gradient_encoding, trace.leaf_value_encoding);
        out.encoded_leaf_values[node] = encode_fixed_point_fr_checked(
            leaf_value, trace.leaf_value_encoding, "derive_gain_selection_state encoded_leaf_values");
        out.leaf_remainders[node] = field_from_size_t(static_cast<size_t>(remainder));
    }

    return out;
}

} // namespace

forestInferenceQueryBatch assemble_forest_inference_query_batch_from_trace(
    const forestSemanticInferenceQueryBatch& trace)
{
    validate_semantic_inference_queries(trace);

    forestInferenceQueryBatch out;
    out.query_tree_ids = poly_from_int64_values(trace.query_tree_ids);
    out.lifted_inputs = poly_from_int64_values(trace.lifted_inputs);
    out.query_count = trace.query_count;
    out.feature_count = trace.feature_count;
    return out;
}

forestTrainingProofInputs assemble_forest_training_inputs_from_trace(
    const forestTrainingSemanticTrace& trace)
{
    validate_trace(trace);

    forestTrainingProofInputs out;
    out.tree_count = trace.tree_count;
    out.node_count = trace.node_count;
    out.feature_count = trace.feature_count;
    out.bin_count = trace.bin_count;
    out.iteration_count = trace.iteration_count;
    out.class_count = trace.class_count;
    out.query_count = trace.query_count;
    out.input_bit_bound = trace.input_bit_bound;
    out.full_upper_bound = trace.full_upper_bound;
    out.structure_bit_bound = trace.structure_bit_bound;
    out.gradient_encoding = trace.gradient_encoding;
    out.gain_numerator_encoding = trace.gain_numerator_encoding;
    out.quantized_gain_encoding = trace.quantized_gain_encoding;
    out.leaf_value_encoding = trace.leaf_value_encoding;
    out.score_encoding = trace.score_encoding;
    out.softmax_encoding = trace.softmax_encoding;
    out.hessian_bit_bound = trace.hessian_bit_bound;
    out.gain_denominator_bit_bound = trace.gain_denominator_bit_bound;
    out.leaf_denominator_bit_bound = trace.leaf_denominator_bit_bound;
    out.gain_gap_bit_bound = trace.gain_gap_bit_bound;
    out.loss_mode = trace.loss_mode;
    out.l2_regularization = trace.l2_regularization;
    out.learning_rate_numerator = trace.learning_rate_numerator;
    out.learning_rate_denominator = trace.learning_rate_denominator;
    out.split_penalty = trace.split_penalty;
    out.split_gain_threshold = trace.split_gain_threshold;
    out.max_depth = trace.max_depth;

    const DerivedForestBoxes boxes = derive_forest_boxes(trace);
    const DerivedForestDepths depths = derive_forest_depths(trace);

    out.root_indices = poly_from_int64_values(trace.root_indices);
    out.split_node_mask = poly_from_int64_values(trace.split_node_mask);
    out.node_depths = poly_from_int64_values(depths.depths);
    out.left_child_indices = poly_from_int64_values(trace.left_child_indices);
    out.right_child_indices = poly_from_int64_values(trace.right_child_indices);
    out.node_tree_ids = poly_from_int64_values(trace.node_tree_ids);
    out.feature_selector = poly_from_int64_values(trace.feature_selector);
    out.lower_bounds = poly_from_int64_values(boxes.lower_bounds);
    out.upper_bounds = poly_from_int64_values(boxes.upper_bounds);
    out.left_lower = poly_from_int64_values(boxes.left_lower);
    out.left_upper = poly_from_int64_values(boxes.left_upper);
    out.right_lower = poly_from_int64_values(boxes.right_lower);
    out.right_upper = poly_from_int64_values(boxes.right_upper);
    out.thresholds = poly_from_int64_values(trace.thresholds);

    const DerivedBoostingState boosting = derive_boosting_state(trace);
    const DerivedGainSelectionState derived = derive_gain_selection_state(trace, boosting);

    for (size_t node = 0; node < trace.node_count; ++node) {
        if (checked_non_negative_size_t(trace.node_leaf_mask[node], "assemble_forest_training_inputs_from_trace leaf mask") == 0) {
            continue;
        }
        const std::int64_t derived_leaf_value = decode_fixed_point_scaled_integer(
            fixed_point_size_t_from_fr(
                derived.encoded_leaf_values[node], "assemble_forest_training_inputs_from_trace encoded_leaf_values"),
            trace.leaf_value_encoding);
        if (derived_leaf_value != trace.node_leaf_values[node]) {
            const std::int64_t derived_leaf_gradient = canonical_signed_int64_from_fr(
                derived.leaf_gradients[node], "assemble_forest_training_inputs_from_trace leaf_gradient");
            const std::int64_t derived_leaf_hessian = canonical_signed_int64_from_fr(
                derived.leaf_hessians[node], "assemble_forest_training_inputs_from_trace leaf_hessian");
            const size_t derived_leaf_denominator = canonical_size_t_from_fr(
                derived.leaf_denominators[node], "assemble_forest_training_inputs_from_trace leaf_denominator");
            throw std::runtime_error("assemble_forest_training_inputs_from_trace: leaf value mismatch at node " +
                                     std::to_string(node) + " (trace=" +
                                     std::to_string(trace.node_leaf_values[node]) + ", derived=" +
                                     std::to_string(derived_leaf_value) + ", leaf_gradient=" +
                                     std::to_string(derived_leaf_gradient) + ", leaf_hessian=" +
                                     std::to_string(derived_leaf_hessian) + ", denominator=" +
                                     std::to_string(derived_leaf_denominator) + ")");
        }
    }

    out.query_tree_ids = poly_from_fr_evals(repeated_tree_id_evals(trace.tree_count, trace.query_count));
    out.lifted_inputs = poly_from_fr_evals(repeated_training_input_evals(trace));
    out.leaf_indices = poly_from_int64_values(trace.leaf_indices);
    out.leaf_lower_bounds = poly_from_fr_evals(derive_leaf_bound_evals(trace, boxes, false));
    out.leaf_upper_bounds = poly_from_fr_evals(derive_leaf_bound_evals(trace, boxes, true));
    out.tree_scores = poly_from_fr_evals(boosting.tree_scores);
    out.training_node_tree_ids = out.node_tree_ids;
    out.training_node_lower_bounds = out.lower_bounds;
    out.training_node_upper_bounds = out.upper_bounds;
    out.node_leaf_mask = poly_from_int64_values(trace.node_leaf_mask);
    out.node_leaf_values = poly_from_int64_values(trace.node_leaf_values);
    out.lifted_leaf_indices = poly_from_fr_evals(derive_lifted_leaf_index_evals(trace));
    out.flattened_bucket_indices = poly_from_fr_evals(derive_flattened_bucket_index_evals(trace));

    out.channels[0].query_values = poly_from_fr_evals(derive_count_query_values(trace.tree_count * trace.query_count));
    out.channels[1].query_values = poly_from_fr_evals(boosting.pseudo_residuals);
    out.channels[2].query_values = poly_from_fr_evals(boosting.diagonal_hessians);
    for (size_t channel = 0; channel < out.channels.size(); ++channel) {
        const std::vector<Fr> query_values = out.channels[channel].query_values.evaluations_on_domain(trace.tree_count * trace.query_count);
        std::vector<Fr> lifted_values;
        lifted_values.reserve(trace.tree_count * trace.query_count * trace.feature_count);
        for (const Fr& value : query_values) {
            for (size_t feature = 0; feature < trace.feature_count; ++feature) {
                (void)feature;
                lifted_values.push_back(value);
            }
        }
        out.channels[channel].lifted_values = poly_from_fr_evals(lifted_values);
        out.channels[channel].leaf_histograms = poly_from_fr_evals(derived.leaf_histograms[channel]);
        out.channels[channel].full_histograms = poly_from_fr_evals(derived.full_histograms[channel]);
        out.channels[channel].split_histograms = out.channels[channel].full_histograms;
        out.channels[channel].left_statistics = poly_from_fr_evals(derived.left_statistics[channel]);
        out.channels[channel].right_statistics = poly_from_fr_evals(derived.right_statistics[channel]);
    }

    out.left_gradients = poly_from_fr_evals(derived.left_statistics[1]);
    out.right_gradients = poly_from_fr_evals(derived.right_statistics[1]);
    out.left_hessians = poly_from_fr_evals(derived.left_statistics[2]);
    out.right_hessians = poly_from_fr_evals(derived.right_statistics[2]);
    out.left_gradient_squares = poly_from_fr_evals(derived.left_gradient_squares);
    out.right_gradient_squares = poly_from_fr_evals(derived.right_gradient_squares);
    out.parent_gradient_squares = poly_from_fr_evals(derived.parent_gradient_squares);
    out.gain_numerators = poly_from_fr_evals(derived.gain_numerators);
    out.gain_denominators = poly_from_fr_evals(derived.gain_denominators);
    out.encoded_gain_numerators = poly_from_fr_evals(derived.encoded_gain_numerators);
    out.encoded_quantized_gains = poly_from_fr_evals(derived.encoded_quantized_gains);
    out.gain_remainders = poly_from_fr_evals(derived.gain_remainders);

    out.quantized_gains = poly_from_fr_evals(derived.quantized_gains);
    out.tie_break_scores = poly_from_fr_evals(derived.tie_break_scores);
    out.candidate_mask = poly_from_int64_values(trace.candidate_mask);
    const size_t candidate_width = trace.feature_count * trace.bin_count;
    std::vector<Fr> candidate_row_mask(trace.node_count, Fr(0));
    for (size_t node = 0; node < trace.node_count; ++node) {
        for (size_t column = 0; column < candidate_width; ++column) {
            if (trace.candidate_mask[node * candidate_width + column] != 0) {
                candidate_row_mask[node] = Fr(1);
                break;
            }
        }
    }
    out.active_node_mask = poly_from_fr_evals(candidate_row_mask);
    std::vector<Fr> depth_limit_mask(trace.node_count, Fr(0));
    for (size_t node = 0; node < trace.node_count; ++node) {
        if (trace.node_leaf_mask[node] != 0 && candidate_row_mask[node] != Fr(0) &&
            checked_non_negative_size_t(depths.depths[node], "assemble_forest_training_inputs_from_trace depth") >=
                trace.max_depth) {
            depth_limit_mask[node] = Fr(1);
        }
    }
    out.depth_limit_mask = poly_from_fr_evals(depth_limit_mask);
    out.selector = poly_from_fr_evals(derived.selector);
    out.selected_gains = poly_from_fr_evals(derived.selected_gains);
    out.selected_tie_break_scores = poly_from_fr_evals(derived.selected_tie_break_scores);
    out.selected_candidate_indices = poly_from_int64_values(trace.selected_candidate_indices);
    out.selected_features = poly_from_fr_evals(derived.selected_features);
    out.selected_bins = poly_from_fr_evals(derived.selected_bins);

    out.leaf_gradients = poly_from_fr_evals(derived.leaf_gradients);
    out.leaf_hessians = poly_from_fr_evals(derived.leaf_hessians);
    out.leaf_denominators = poly_from_fr_evals(derived.leaf_denominators);
    out.encoded_negative_leaf_gradients = poly_from_fr_evals(derived.encoded_negative_leaf_gradients);
    out.encoded_leaf_values = poly_from_fr_evals(derived.encoded_leaf_values);
    out.leaf_remainders = poly_from_fr_evals(derived.leaf_remainders);

    {
        const std::vector<Fr> mask_evals = out.node_leaf_mask.evaluations_on_domain(trace.node_count);
        const std::vector<Fr> node_leaf_value_evals = out.node_leaf_values.evaluations_on_domain(trace.node_count);
        const std::vector<Fr> encoded_leaf_value_evals = out.encoded_leaf_values.evaluations_on_domain(trace.node_count);
        const Fr leaf_offset = fixed_point_offset_fr(trace.leaf_value_encoding);
        for (size_t node = 0; node < trace.node_count; ++node) {
            const Fr zero_eval = mask_evals[node] * (node_leaf_value_evals[node] - (encoded_leaf_value_evals[node] - leaf_offset));
            if (zero_eval != Fr(0)) {
                throw std::runtime_error("assemble_forest_training_inputs_from_trace: node leaf zero relation mismatch at node " +
                                         std::to_string(node));
            }
        }
    }

    out.cumulative_scores = poly_from_fr_evals(boosting.cumulative_scores);
    out.encoded_cumulative_scores = poly_from_fr_evals(boosting.encoded_cumulative_scores);
    out.softmax_shifts = poly_from_fr_evals(boosting.softmax_shifts);
    out.shift_selector = poly_from_fr_evals(boosting.shift_selector);
    out.lifted_softmax_shifts = poly_from_fr_evals(boosting.lifted_softmax_shifts);
    out.encoded_shifted_scores = poly_from_fr_evals(boosting.encoded_shifted_scores);
    out.softmax_exp_values = poly_from_fr_evals(boosting.softmax_exp_values);
    out.softmax_denominators = poly_from_fr_evals(boosting.softmax_denominators);
    out.lifted_softmax_denominators = poly_from_fr_evals(boosting.lifted_softmax_denominators);
    out.softmax_probabilities = poly_from_fr_evals(boosting.softmax_probabilities);
    out.softmax_remainders = poly_from_fr_evals(boosting.softmax_remainders);
    out.base_labels = poly_from_int64_values(trace.base_labels);
    out.pseudo_residuals = poly_from_fr_evals(boosting.pseudo_residuals);
    out.encoded_diagonal_hessians = poly_from_fr_evals(boosting.encoded_diagonal_hessians);
    out.diagonal_hessian_remainders = poly_from_fr_evals(boosting.diagonal_hessian_remainders);
    out.diagonal_hessians = poly_from_fr_evals(boosting.diagonal_hessians);

    return out;
}

forestCommittedInferenceInputs assemble_forest_committed_inference_inputs_from_trace(
    const forestTrainingSemanticTrace& trace,
    const forestSemanticInferenceQueryBatch& inference_queries)
{
    validate_trace(trace);
    const DerivedForestBoxes boxes = derive_forest_boxes(trace);
    const DerivedForestDepths depths = derive_forest_depths(trace);

    forestCommittedInferenceInputs out;
    out.committed_forest.root_indices = poly_from_int64_values(trace.root_indices);
    out.committed_forest.split_node_mask = poly_from_int64_values(trace.split_node_mask);
    out.committed_forest.node_depths = poly_from_int64_values(depths.depths);
    out.committed_forest.left_child_indices = poly_from_int64_values(trace.left_child_indices);
    out.committed_forest.right_child_indices = poly_from_int64_values(trace.right_child_indices);
    out.committed_forest.node_tree_ids = poly_from_int64_values(trace.node_tree_ids);
    out.committed_forest.feature_selector = poly_from_int64_values(trace.feature_selector);
    out.committed_forest.lower_bounds = poly_from_int64_values(boxes.lower_bounds);
    out.committed_forest.upper_bounds = poly_from_int64_values(boxes.upper_bounds);
    out.committed_forest.thresholds = poly_from_int64_values(trace.thresholds);
    out.committed_forest.node_leaf_mask = poly_from_int64_values(trace.node_leaf_mask);
    out.committed_forest.node_leaf_values = poly_from_int64_values(trace.node_leaf_values);
    out.committed_forest.tree_count = trace.tree_count;
    out.committed_forest.node_count = trace.node_count;
    out.committed_forest.feature_count = trace.feature_count;
    out.queries = assemble_forest_inference_query_batch_from_trace(inference_queries);
    out.input_bit_bound = trace.input_bit_bound;
    return out;
}

forestTrainingThenInferenceInputs assemble_forest_training_then_inference_inputs_from_trace(
    const forestTrainingSemanticTrace& trace,
    const forestSemanticInferenceQueryBatch& inference_queries)
{
    forestTrainingThenInferenceInputs out;
    out.training_inputs = assemble_forest_training_inputs_from_trace(trace);
    out.inference_queries = assemble_forest_inference_query_batch_from_trace(inference_queries);
    return out;
}

void write_forest_training_then_inference_case_from_trace(
    const std::string& path,
    const forestTrainingSemanticTrace& trace,
    const forestSemanticInferenceQueryBatch& inference_queries)
{
    write_forest_training_then_inference_case(
        path, assemble_forest_training_then_inference_inputs_from_trace(trace, inference_queries));
}
