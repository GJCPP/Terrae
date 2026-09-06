#include "zkboost/shared_trace/validate_boosting_state.hpp"

#include "algebra_utils.hpp"
#include "fixed_point.hpp"
#include "matrix_utils.hpp"

#include <algorithm>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_validate_boosting_state: " + message);
    }
}

Fr fr_from_int64(std::int64_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
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
    Fr round_trip;
    round_trip.setStr(out.convert_to<std::string>(), 10);
    if (round_trip != value) {
        throw std::runtime_error(std::string(context) + ": field element is not a canonical integer");
    }
    return out;
}

Fr fr_from_cpp_int_checked(const cpp_int& value)
{
    const std::string repr = value.convert_to<std::string>();
    Fr out;
    out.setStr(repr, 10);
    if (out.getStr(10) != repr) {
        throw std::runtime_error("zkboost_shared_validate_boosting_state: non-canonical cpp_int conversion");
    }
    return out;
}

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

std::pair<std::int64_t, std::int64_t> floor_division(std::int64_t numerator, std::int64_t denominator)
{
    require_condition(denominator > 0, "division denominator must be positive");
    std::int64_t quotient = numerator / denominator;
    std::int64_t remainder = numerator % denominator;
    if (remainder < 0) {
        --quotient;
        remainder += denominator;
    }
    return {quotient, remainder};
}

size_t checked_size(std::int64_t value, const std::string& name)
{
    require_condition(value >= 0, name + " must be non-negative");
    return static_cast<size_t>(value);
}

void validate_shapes(const forestTrainingSemanticTrace& trace)
{
    require_condition(trace.iteration_count > 0, "iteration_count must be non-zero");
    require_condition(trace.class_count > 0, "class_count must be non-zero");
    require_condition(trace.query_count > 0, "query_count must be non-zero");
    require_condition(trace.tree_count == trace.iteration_count * trace.class_count,
                      "tree_count must equal iteration_count * class_count");
    require_condition(trace.node_count > 0, "node_count must be non-zero");
    require_condition(trace.leaf_indices.size() == trace.tree_count * trace.query_count,
                      "leaf_indices length mismatch");
    require_condition(trace.node_leaf_values.size() == trace.node_count,
                      "node_leaf_values length mismatch");
    require_condition(trace.base_labels.size() == trace.class_count * trace.query_count,
                      "base_labels length mismatch");
    validate_fixed_point_encoding(trace.score_encoding, "zkboost_shared_validate_boosting_state score");
    validate_fixed_point_encoding(trace.softmax_encoding, "zkboost_shared_validate_boosting_state softmax");
    require_condition(trace.score_encoding.signed_values, "score encoding must be signed");
    require_condition(trace.softmax_encoding.signed_values, "softmax encoding must be signed");
    require_condition(trace.hessian_bit_bound > 0, "hessian_bit_bound must be non-zero");
    require_condition(trace.softmax_encoding.fractional_bits < std::numeric_limits<size_t>::digits,
                      "softmax fractional bits exceed size_t range");

    for (size_t sample = 0; sample < trace.query_count; ++sample) {
        size_t sum = 0;
        for (size_t cls = 0; cls < trace.class_count; ++cls) {
            const std::int64_t label = trace.base_labels.at(cls * trace.query_count + sample);
            require_condition(label == 0 || label == 1, "base_labels must be binary");
            sum += static_cast<size_t>(label);
        }
        require_condition(sum == 1, "base_labels must be one-hot per sample");
    }
}

std::vector<Fr> derive_tree_scores(const forestTrainingSemanticTrace& trace)
{
    std::vector<Fr> out;
    out.reserve(trace.tree_count * trace.query_count);
    for (std::int64_t leaf_index_value : trace.leaf_indices) {
        const size_t leaf_index = checked_size(leaf_index_value, "leaf_indices");
        require_condition(leaf_index < trace.node_count, "leaf index out of range");
        out.push_back(fr_from_int64(trace.node_leaf_values.at(leaf_index)));
    }
    return out;
}

std::vector<Fr> derive_cumulative_scores(const std::vector<Fr>& tree_scores,
                                         size_t iteration_count,
                                         size_t class_count,
                                         size_t query_count)
{
    const size_t row_width = class_count * query_count;
    std::vector<Fr> out(tree_scores.size(), Fr(0));
    std::vector<Fr> running(row_width, Fr(0));
    for (size_t iteration = 0; iteration < iteration_count; ++iteration) {
        for (size_t column = 0; column < row_width; ++column) {
            const size_t index = iteration * row_width + column;
            out[index] = running[column];
            running[column] += tree_scores.at(index);
        }
    }
    return out;
}

std::vector<Fr> encode_fixed_point_vector(const std::vector<Fr>& values,
                                          const fixedPointEncoding& encoding,
                                          const char* context)
{
    std::vector<Fr> out;
    out.reserve(values.size());
    for (const Fr& value : values) {
        out.push_back(encode_fixed_point_fr_checked(canonical_signed_int64_from_fr(value, context),
                                                    encoding,
                                                    context));
    }
    return out;
}

void derive_softmax_shifts(zkboostSharedBoostingStateView& view,
                           const forestTrainingSemanticTrace& trace)
{
    view.softmax_shifts.reserve(trace.iteration_count * trace.query_count);
    view.shift_selector.assign(view.summary.cube_size, Fr(0));
    for (size_t iteration = 0; iteration < trace.iteration_count; ++iteration) {
        for (size_t sample = 0; sample < trace.query_count; ++sample) {
            size_t best_class = 0;
            Fr best_value =
                view.encoded_cumulative_scores.at(iteration * trace.class_count * trace.query_count + sample);
            for (size_t cls = 1; cls < trace.class_count; ++cls) {
                const size_t index =
                    iteration * trace.class_count * trace.query_count + cls * trace.query_count + sample;
                if (best_value.isLess(view.encoded_cumulative_scores.at(index))) {
                    best_class = cls;
                    best_value = view.encoded_cumulative_scores.at(index);
                }
            }
            view.softmax_shifts.push_back(best_value);
            const size_t selector_index =
                iteration * trace.class_count * trace.query_count + best_class * trace.query_count + sample;
            view.shift_selector.at(selector_index) = Fr(1);
        }
    }
}

void derive_shifted_scores_and_exp(zkboostSharedBoostingStateView& view,
                                   const forestTrainingSemanticTrace& trace)
{
    view.lifted_softmax_shifts.reserve(view.summary.cube_size);
    view.encoded_shifted_scores.reserve(view.summary.cube_size);
    view.softmax_exp_values.reserve(view.summary.cube_size);

    for (size_t iteration = 0; iteration < trace.iteration_count; ++iteration) {
        for (size_t cls = 0; cls < trace.class_count; ++cls) {
            for (size_t sample = 0; sample < trace.query_count; ++sample) {
                const size_t cube_index =
                    iteration * trace.class_count * trace.query_count + cls * trace.query_count + sample;
                const size_t shift_index = iteration * trace.query_count + sample;
                const Fr shift = view.softmax_shifts.at(shift_index);
                view.lifted_softmax_shifts.push_back(shift);

                const std::int64_t cumulative_scaled =
                    canonical_signed_int64_from_fr(view.cumulative_scores.at(cube_index),
                                                   "zkboost_shared_validate_boosting_state cumulative_scores");
                const std::int64_t shift_scaled =
                    decode_fixed_point_scaled_integer(
                        fixed_point_size_t_from_fr(shift, "zkboost_shared_validate_boosting_state softmax_shifts"),
                        trace.score_encoding);
                const std::int64_t shifted_scaled = cumulative_scaled - shift_scaled;
                const Fr encoded_shifted = encode_fixed_point_fr_checked(
                    shifted_scaled,
                    trace.score_encoding,
                    "zkboost_shared_validate_boosting_state encoded_shifted_scores");
                view.encoded_shifted_scores.push_back(encoded_shifted);
                view.softmax_exp_values.push_back(
                    fr_from_cpp_int_checked(quantized_exp_output(shifted_scaled,
                                                                 trace.score_encoding.fractional_bits)));
            }
        }
    }
}

void derive_softmax_probabilities(zkboostSharedBoostingStateView& view,
                                  const forestTrainingSemanticTrace& trace)
{
    view.softmax_denominators.reserve(trace.iteration_count * trace.query_count);
    view.lifted_softmax_denominators.reserve(view.summary.cube_size);
    view.softmax_probabilities.assign(view.summary.cube_size, Fr(0));
    view.softmax_remainders.assign(view.summary.cube_size, Fr(0));

    const std::int64_t softmax_scale = static_cast<std::int64_t>(view.summary.softmax_scale);
    for (size_t iteration = 0; iteration < trace.iteration_count; ++iteration) {
        for (size_t sample = 0; sample < trace.query_count; ++sample) {
            cpp_int denominator = 0;
            for (size_t cls = 0; cls < trace.class_count; ++cls) {
                const size_t index =
                    iteration * trace.class_count * trace.query_count + cls * trace.query_count + sample;
                denominator += canonical_cpp_int_from_fr(
                    view.softmax_exp_values.at(index),
                    "zkboost_shared_validate_boosting_state softmax_exp_values");
            }
            require_condition(denominator <= std::numeric_limits<std::int64_t>::max(),
                              "softmax denominator exceeds int64 range");
            const Fr denominator_fr = fr_from_cpp_int_checked(denominator);
            view.softmax_denominators.push_back(denominator_fr);
        }
        for (size_t cls = 0; cls < trace.class_count; ++cls) {
            for (size_t sample = 0; sample < trace.query_count; ++sample) {
                const size_t index =
                    iteration * trace.class_count * trace.query_count + cls * trace.query_count + sample;
                const Fr denominator_fr = view.softmax_denominators.at(iteration * trace.query_count + sample);
                view.lifted_softmax_denominators.push_back(denominator_fr);
                const std::int64_t denominator_i64 =
                    canonical_signed_int64_from_fr(denominator_fr,
                                                   "zkboost_shared_validate_boosting_state softmax_denominators");
                const std::int64_t exp_value =
                    canonical_signed_int64_from_fr(view.softmax_exp_values.at(index),
                                                   "zkboost_shared_validate_boosting_state softmax_exp_values");
                const auto [probability_scaled, remainder] =
                    floor_division(exp_value * softmax_scale, denominator_i64);
                view.softmax_probabilities.at(index) =
                    encode_fixed_point_fr_checked(probability_scaled,
                                                  trace.softmax_encoding,
                                                  "zkboost_shared_validate_boosting_state softmax_probabilities");
                view.softmax_remainders.at(index) = field_from_size_t(static_cast<size_t>(remainder));
            }
        }
    }
}

void derive_residuals_and_hessians(zkboostSharedBoostingStateView& view,
                                   const forestTrainingSemanticTrace& trace)
{
    view.pseudo_residuals.assign(view.summary.cube_size, Fr(0));
    view.encoded_diagonal_hessians.assign(view.summary.cube_size, Fr(0));
    view.diagonal_hessian_remainders.assign(view.summary.cube_size, Fr(0));
    view.diagonal_hessians.assign(view.summary.cube_size, Fr(0));

    const std::int64_t softmax_scale = static_cast<std::int64_t>(view.summary.softmax_scale);
    const fixedPointEncoding hessian_encoding = fixedPointEncoding::signed_offset_binary(
        std::max(trace.hessian_bit_bound + 1, trace.softmax_encoding.fractional_bits + 1),
        trace.softmax_encoding.fractional_bits);

    for (size_t iteration = 0; iteration < trace.iteration_count; ++iteration) {
        for (size_t cls = 0; cls < trace.class_count; ++cls) {
            for (size_t sample = 0; sample < trace.query_count; ++sample) {
                const size_t index =
                    iteration * trace.class_count * trace.query_count + cls * trace.query_count + sample;
                const std::int64_t probability_scaled =
                    decode_fixed_point_scaled_integer(
                        fixed_point_size_t_from_fr(
                            view.softmax_probabilities.at(index),
                            "zkboost_shared_validate_boosting_state softmax_probabilities"),
                        trace.softmax_encoding);
                const std::int64_t label_scaled =
                    trace.base_labels.at(cls * trace.query_count + sample) * softmax_scale;
                const std::int64_t residual_scaled =
                    trace.loss_mode == forestTrainingLoss::FirstOrderRss
                        ? label_scaled - probability_scaled
                        : probability_scaled - label_scaled;
                view.pseudo_residuals.at(index) = fr_from_int64(residual_scaled);

                std::int64_t hessian_scaled = softmax_scale;
                std::int64_t hessian_remainder = 0;
                if (trace.loss_mode == forestTrainingLoss::SecondOrderXgboost) {
                    const auto division =
                        floor_division(probability_scaled * (softmax_scale - probability_scaled),
                                       softmax_scale);
                    hessian_scaled = division.first;
                    hessian_remainder = division.second;
                }
                view.diagonal_hessians.at(index) = fr_from_int64(hessian_scaled);
                view.diagonal_hessian_remainders.at(index) =
                    field_from_size_t(static_cast<size_t>(hessian_remainder));
                view.encoded_diagonal_hessians.at(index) =
                    encode_fixed_point_fr_checked(
                        hessian_scaled,
                        hessian_encoding,
                        "zkboost_shared_validate_boosting_state encoded_diagonal_hessians");
            }
        }
    }
}

} // namespace

zkboostSharedBoostingStateView build_zkboost_shared_boosting_state_view(
    const forestTrainingSemanticTrace& training)
{
    validate_shapes(training);

    zkboostSharedBoostingStateView view;
    view.summary.iteration_count = training.iteration_count;
    view.summary.class_count = training.class_count;
    view.summary.sample_count = training.query_count;
    view.summary.tree_count = training.tree_count;
    view.summary.cube_size = training.iteration_count * training.class_count * training.query_count;
    view.summary.softmax_scale = size_t(1) << training.softmax_encoding.fractional_bits;

    view.tree_scores = derive_tree_scores(training);
    view.base_labels.reserve(training.base_labels.size());
    for (std::int64_t label : training.base_labels) {
        view.base_labels.push_back(fr_from_dec(std::to_string(label)));
    }
    view.cumulative_scores =
        derive_cumulative_scores(view.tree_scores,
                                 training.iteration_count,
                                 training.class_count,
                                 training.query_count);
    view.encoded_cumulative_scores =
        encode_fixed_point_vector(view.cumulative_scores,
                                  training.score_encoding,
                                  "zkboost_shared_validate_boosting_state cumulative_scores");
    derive_softmax_shifts(view, training);
    derive_shifted_scores_and_exp(view, training);
    derive_softmax_probabilities(view, training);
    derive_residuals_and_hessians(view, training);

    return view;
}

zkboostSharedBoostingStateSummary validate_zkboost_shared_boosting_state(
    const forestTrainingSemanticTrace& training)
{
    return build_zkboost_shared_boosting_state_view(training).summary;
}
