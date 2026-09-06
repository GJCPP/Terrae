#include "zkboost/shared_trace/validate_boosting_state_plonk.hpp"

#include "algebra_utils.hpp"
#include "fixed_point.hpp"
#include "zkboost/compiler/generic_circuit.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

constexpr std::int64_t kExpLookupTag = 2000000011LL;
constexpr std::int64_t kProbabilityRangeLookupTag = 2000000014LL;
constexpr std::int64_t kHessianRemainderRangeLookupTag = 2000000015LL;

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_boosting_state_plonk: " + message);
    }
}

std::int64_t checked_i64(size_t value, const std::string& name)
{
    require_condition(value <= static_cast<size_t>(std::numeric_limits<std::int64_t>::max()),
                      name + " exceeds int64 range");
    return static_cast<std::int64_t>(value);
}

std::int64_t checked_i64_from_fr(const Fr& value, const char* context)
{
    return canonical_signed_int64_from_fr(value, context);
}

std::int64_t checked_nonnegative_i64_from_fr(const Fr& value, const char* context)
{
    const size_t parsed = fixed_point_size_t_from_fr(value, context);
    return checked_i64(parsed, context);
}

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

std::int64_t checked_exp_i64(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_int value = quantized_exp_output(scaled_input, fractional_bits);
    require_condition(value <= std::numeric_limits<std::int64_t>::max(),
                      "exp lookup output exceeds int64 range");
    return value.convert_to<std::int64_t>();
}

std::int64_t softmax_scale_from_encoding(const fixedPointEncoding& encoding)
{
    require_condition(encoding.fractional_bits < static_cast<size_t>(std::numeric_limits<std::int64_t>::digits),
                      "softmax fractional bits exceed int64 range");
    return std::int64_t(1) << encoding.fractional_bits;
}

size_t nonnegative_bit_width(size_t maximum)
{
    size_t bits = 1;
    while (maximum > 1) {
        maximum >>= 1;
        ++bits;
    }
    return bits;
}

size_t cube_index(size_t class_count, size_t sample_count, size_t iteration, size_t cls, size_t sample)
{
    return iteration * class_count * sample_count + cls * sample_count + sample;
}

class sharedBoostingStateCircuitCompiler : public zkboostGenericCircuitCompilerBase {
public:
    sharedBoostingStateCircuitCompiler(const forestTrainingSemanticTrace& training,
                                       const zkboostSharedBoostingStateView& view)
        : training_(training)
        , view_(view)
    {
        require_condition(training_.tree_count == view_.summary.tree_count, "tree_count mismatch");
        require_condition(training_.iteration_count == view_.summary.iteration_count,
                          "iteration_count mismatch");
        require_condition(training_.class_count == view_.summary.class_count, "class_count mismatch");
        require_condition(training_.class_count > 0, "class_count must be non-zero");
        require_condition(training_.query_count == view_.summary.sample_count, "sample_count mismatch");
        require_condition(view_.summary.cube_size ==
                              training_.iteration_count * training_.class_count * training_.query_count,
                          "cube_size mismatch");
        require_condition(view_.tree_scores.size() == view_.summary.cube_size,
                          "tree_scores length mismatch");
        require_condition(view_.cumulative_scores.size() == view_.summary.cube_size,
                          "cumulative_scores length mismatch");
        require_condition(view_.encoded_cumulative_scores.size() == view_.summary.cube_size,
                          "encoded_cumulative_scores length mismatch");
        require_condition(view_.softmax_shifts.size() == training_.iteration_count * training_.query_count,
                          "softmax_shifts length mismatch");
        require_condition(view_.shift_selector.size() == view_.summary.cube_size,
                          "shift_selector length mismatch");
        require_condition(view_.lifted_softmax_shifts.size() == view_.summary.cube_size,
                          "lifted_softmax_shifts length mismatch");
        require_condition(view_.encoded_shifted_scores.size() == view_.summary.cube_size,
                          "encoded_shifted_scores length mismatch");
        require_condition(view_.softmax_exp_values.size() == view_.summary.cube_size,
                          "softmax_exp_values length mismatch");
        require_condition(view_.softmax_denominators.size() ==
                              training_.iteration_count * training_.query_count,
                          "softmax_denominators length mismatch");
        require_condition(view_.lifted_softmax_denominators.size() == view_.summary.cube_size,
                          "lifted_softmax_denominators length mismatch");
        require_condition(view_.softmax_probabilities.size() == view_.summary.cube_size,
                          "softmax_probabilities length mismatch");
        require_condition(view_.softmax_remainders.size() == view_.summary.cube_size,
                          "softmax_remainders length mismatch");
        require_condition(view_.pseudo_residuals.size() == view_.summary.cube_size,
                          "pseudo_residuals length mismatch");
        require_condition(view_.diagonal_hessian_remainders.size() == view_.summary.cube_size,
                          "diagonal_hessian_remainders length mismatch");
        require_condition(view_.diagonal_hessians.size() == view_.summary.cube_size,
                          "diagonal_hessians length mismatch");
        require_condition(training_.base_labels.size() == training_.class_count * training_.query_count,
                          "base_labels length mismatch");
        require_condition(view_.base_labels.size() == training_.base_labels.size(),
                          "public base_labels length mismatch");
        validate_fixed_point_encoding(training_.score_encoding, "shared_boosting_state_plonk score");
        validate_fixed_point_encoding(training_.softmax_encoding, "shared_boosting_state_plonk softmax");

        zero_ = constant(0);
        one_ = constant(1);
        score_offset_ = fixed_point_offset_fr(training_.score_encoding);
        add_exp_lookup_table();
        add_range_lookup_tables();
        initialize_public_inputs();
        initialize_private_witnesses();
    }

    void compile()
    {
        compile_labels();
        compile_cumulative_scores();
        compile_softmax_shifts();
        compile_shifted_scores_and_exp();
        compile_probabilities();
        compile_residuals_and_hessians();
    }

private:
    void add_exp_lookup_table()
    {
        const std::int64_t min_score = fixed_point_min_scaled_value(training_.score_encoding);
        for (std::int64_t scaled = min_score; scaled <= 0; ++scaled) {
            const std::int64_t encoded =
                checked_i64(encode_fixed_point_scaled_integer(scaled, training_.score_encoding),
                            "encoded exp input");
            add_lookup_table_row(encoded,
                                 kExpLookupTag,
                                 checked_exp_i64(scaled, training_.score_encoding.fractional_bits));
        }
    }

    void add_range_lookup_tables()
    {
        const std::int64_t softmax_scale = softmax_scale_from_encoding(training_.softmax_encoding);
        for (std::int64_t value = 0; value <= softmax_scale; ++value) {
            add_lookup_table_row(value, kProbabilityRangeLookupTag, 0);
        }
        for (std::int64_t value = 0; value < softmax_scale; ++value) {
            add_lookup_table_row(value, kHessianRemainderRangeLookupTag, 0);
        }
    }

    void initialize_public_inputs()
    {
        public_tree_scores_.reserve(view_.tree_scores.size());
        for (const Fr& value : view_.tree_scores) {
            public_tree_scores_.push_back(
                public_value(checked_i64_from_fr(value, "shared boosting tree_scores")));
        }

        public_pseudo_residuals_.reserve(view_.pseudo_residuals.size());
        for (const Fr& value : view_.pseudo_residuals) {
            public_pseudo_residuals_.push_back(
                public_value(checked_i64_from_fr(value, "shared boosting pseudo_residuals")));
        }

        public_diagonal_hessians_.reserve(view_.diagonal_hessians.size());
        for (const Fr& value : view_.diagonal_hessians) {
            public_diagonal_hessians_.push_back(
                public_value(checked_i64_from_fr(value, "shared boosting diagonal_hessians")));
        }

        public_base_labels_.reserve(view_.base_labels.size());
        for (const Fr& value : view_.base_labels) {
            public_base_labels_.push_back(
                public_value(checked_i64_from_fr(value, "shared boosting base_labels")));
        }
    }

    void initialize_private_witnesses()
    {
        cumulative_scores_.reserve(view_.summary.cube_size);
        encoded_cumulative_scores_.reserve(view_.summary.cube_size);
        shift_selectors_.reserve(view_.summary.cube_size);
        lifted_softmax_shifts_.reserve(view_.summary.cube_size);
        encoded_shifted_scores_.reserve(view_.summary.cube_size);
        softmax_exp_values_.reserve(view_.summary.cube_size);
        lifted_softmax_denominators_.reserve(view_.summary.cube_size);
        probability_scaled_values_.reserve(view_.summary.cube_size);
        softmax_remainders_.reserve(view_.summary.cube_size);
        hessian_remainders_.reserve(view_.summary.cube_size);

        for (size_t i = 0; i < view_.summary.cube_size; ++i) {
            cumulative_scores_.push_back(
                private_value(checked_i64_from_fr(view_.cumulative_scores.at(i),
                                                  "shared boosting cumulative_scores")));
            encoded_cumulative_scores_.push_back(
                private_value(checked_nonnegative_i64_from_fr(view_.encoded_cumulative_scores.at(i),
                                                              "shared boosting encoded_cumulative_scores")));
            shift_selectors_.push_back(
                private_value(checked_i64_from_fr(view_.shift_selector.at(i),
                                                  "shared boosting shift_selector")));
            lifted_softmax_shifts_.push_back(
                private_value(checked_nonnegative_i64_from_fr(view_.lifted_softmax_shifts.at(i),
                                                              "shared boosting lifted_softmax_shifts")));
            encoded_shifted_scores_.push_back(
                private_value(checked_nonnegative_i64_from_fr(view_.encoded_shifted_scores.at(i),
                                                              "shared boosting encoded_shifted_scores")));
            softmax_exp_values_.push_back(
                private_value(checked_i64_from_fr(view_.softmax_exp_values.at(i),
                                                  "shared boosting softmax_exp_values")));
            lifted_softmax_denominators_.push_back(
                private_value(checked_i64_from_fr(view_.lifted_softmax_denominators.at(i),
                                                  "shared boosting lifted_softmax_denominators")));
            const std::int64_t probability_scaled =
                decode_fixed_point_scaled_integer(
                    fixed_point_size_t_from_fr(view_.softmax_probabilities.at(i),
                                               "shared boosting softmax_probabilities"),
                    training_.softmax_encoding);
            probability_scaled_values_.push_back(private_value(probability_scaled));
            softmax_remainders_.push_back(
                private_value(checked_i64_from_fr(view_.softmax_remainders.at(i),
                                                  "shared boosting softmax_remainders")));
            hessian_remainders_.push_back(
                private_value(checked_i64_from_fr(view_.diagonal_hessian_remainders.at(i),
                                                  "shared boosting diagonal_hessian_remainders")));
        }

        softmax_shifts_.reserve(view_.softmax_shifts.size());
        softmax_denominators_.reserve(view_.softmax_denominators.size());
        for (size_t i = 0; i < view_.softmax_shifts.size(); ++i) {
            softmax_shifts_.push_back(
                private_value(checked_nonnegative_i64_from_fr(view_.softmax_shifts.at(i),
                                                              "shared boosting softmax_shifts")));
            softmax_denominators_.push_back(
                private_value(checked_i64_from_fr(view_.softmax_denominators.at(i),
                                                  "shared boosting softmax_denominators")));
        }
    }

    void compile_labels()
    {
        for (size_t sample = 0; sample < training_.query_count; ++sample) {
            zkboostAssignedCircuitValue sum = zero_;
            for (size_t cls = 0; cls < training_.class_count; ++cls) {
                const size_t index = cls * training_.query_count + sample;
                constrain_boolean(public_base_labels_.at(index));
                sum = add_scaled(sum, Fr(1), public_base_labels_.at(index), Fr(1), Fr(0));
            }
            constrain_equal(sum, one_);
        }
    }

    void compile_cumulative_scores()
    {
        const size_t row_width = training_.class_count * training_.query_count;
        for (size_t iteration = 0; iteration < training_.iteration_count; ++iteration) {
            for (size_t cls = 0; cls < training_.class_count; ++cls) {
                for (size_t sample = 0; sample < training_.query_count; ++sample) {
                    const size_t index =
                        cube_index(training_.class_count, training_.query_count, iteration, cls, sample);
                    if (iteration == 0) {
                        constrain_equal(cumulative_scores_.at(index), zero_);
                    } else {
                        const size_t previous_index = index - row_width;
                        const zkboostAssignedCircuitValue expected =
                            add_scaled(cumulative_scores_.at(previous_index),
                                       Fr(1),
                                       public_tree_scores_.at(previous_index),
                                       Fr(1),
                                       Fr(0));
                        constrain_equal(cumulative_scores_.at(index), expected);
                    }

                    const zkboostAssignedCircuitValue expected_encoded =
                        affine(cumulative_scores_.at(index), Fr(1), score_offset_);
                    constrain_equal(encoded_cumulative_scores_.at(index), expected_encoded);
                }
            }
        }
    }

    void compile_softmax_shifts()
    {
        for (size_t iteration = 0; iteration < training_.iteration_count; ++iteration) {
            for (size_t sample = 0; sample < training_.query_count; ++sample) {
                zkboostAssignedCircuitValue selector_sum = constant(0);
                zkboostAssignedCircuitValue selected_sum = constant(0);
                const size_t shift_index = iteration * training_.query_count + sample;
                for (size_t cls = 0; cls < training_.class_count; ++cls) {
                    const size_t index =
                        cube_index(training_.class_count, training_.query_count, iteration, cls, sample);
                    constrain_boolean(shift_selectors_.at(index));

                    const zkboostAssignedCircuitValue selected =
                        multiply(shift_selectors_.at(index), encoded_cumulative_scores_.at(index));
                    if (cls == 0) {
                        selector_sum = shift_selectors_.at(index);
                        selected_sum = selected;
                    } else {
                        selector_sum =
                            add_scaled(selector_sum, Fr(1), shift_selectors_.at(index), Fr(1), Fr(0));
                        selected_sum = add_scaled(selected_sum, Fr(1), selected, Fr(1), Fr(0));
                    }

                    const std::int64_t encoded = checked_nonnegative_i64_from_fr(
                        view_.encoded_cumulative_scores.at(index),
                        "shared boosting encoded_cumulative_scores");
                    const std::int64_t shift = checked_nonnegative_i64_from_fr(
                        view_.softmax_shifts.at(shift_index),
                        "shared boosting softmax_shifts");
                    const zkboostAssignedCircuitValue max_gap =
                        add_scaled(softmax_shifts_.at(shift_index),
                                   Fr(1),
                                   encoded_cumulative_scores_.at(index),
                                   Fr(-1),
                                   Fr(0));
                    constrain_nonnegative_with_limbs(max_gap,
                                                     shift - encoded,
                                                     training_.score_encoding.total_bits);
                }
                constrain_equal(selector_sum, one_);
                constrain_equal(selected_sum, softmax_shifts_.at(shift_index));
            }
        }
    }

    void compile_shifted_scores_and_exp()
    {
        for (size_t iteration = 0; iteration < training_.iteration_count; ++iteration) {
            for (size_t cls = 0; cls < training_.class_count; ++cls) {
                for (size_t sample = 0; sample < training_.query_count; ++sample) {
                    const size_t index =
                        cube_index(training_.class_count, training_.query_count, iteration, cls, sample);
                    const size_t shift_index = iteration * training_.query_count + sample;
                    constrain_equal(lifted_softmax_shifts_.at(index), softmax_shifts_.at(shift_index));

                    const zkboostAssignedCircuitValue expected_shifted =
                        add_scaled(encoded_cumulative_scores_.at(index),
                                   Fr(1),
                                   lifted_softmax_shifts_.at(index),
                                   Fr(-1),
                                   score_offset_);
                    constrain_equal(encoded_shifted_scores_.at(index), expected_shifted);
                    add_lookup(encoded_shifted_scores_.at(index),
                               constant(kExpLookupTag),
                               softmax_exp_values_.at(index));
                }
            }
        }
    }

    void compile_probabilities()
    {
        const std::int64_t softmax_scale = softmax_scale_from_encoding(training_.softmax_encoding);

        for (size_t iteration = 0; iteration < training_.iteration_count; ++iteration) {
            for (size_t sample = 0; sample < training_.query_count; ++sample) {
                const size_t shift_index = iteration * training_.query_count + sample;
                zkboostAssignedCircuitValue denominator_sum =
                    softmax_exp_values_.at(
                        cube_index(training_.class_count,
                                   training_.query_count,
                                   iteration,
                                   0,
                                   sample));
                for (size_t cls = 1; cls < training_.class_count; ++cls) {
                    const size_t index =
                        cube_index(training_.class_count, training_.query_count, iteration, cls, sample);
                    denominator_sum = add_scaled(denominator_sum,
                                                 Fr(1),
                                                 softmax_exp_values_.at(index),
                                                 Fr(1),
                                                 Fr(0));
                }
                constrain_equal(softmax_denominators_.at(shift_index), denominator_sum);

                for (size_t cls = 0; cls < training_.class_count; ++cls) {
                    const size_t index =
                        cube_index(training_.class_count, training_.query_count, iteration, cls, sample);
                    constrain_equal(lifted_softmax_denominators_.at(index),
                                    softmax_denominators_.at(shift_index));
                    add_lookup(probability_scaled_values_.at(index),
                               constant(kProbabilityRangeLookupTag),
                               constant(0));
                    const std::int64_t denominator = checked_i64_from_fr(
                        view_.lifted_softmax_denominators.at(index),
                        "shared boosting lifted_softmax_denominators");
                    const std::int64_t remainder = checked_i64_from_fr(
                        view_.softmax_remainders.at(index),
                        "shared boosting softmax_remainders");
                    require_condition(denominator > 0 && remainder >= 0 && remainder < denominator,
                                      "softmax remainder must be in [0, denominator)");
                    require_condition(
                        training_.class_count <=
                            std::numeric_limits<size_t>::max() /
                                static_cast<size_t>(softmax_scale),
                        "softmax denominator bound overflows size_t");
                    const size_t denominator_bit_bound = nonnegative_bit_width(
                        training_.class_count * static_cast<size_t>(softmax_scale));
                    require_condition(denominator_bit_bound > 0 && denominator_bit_bound <= 63,
                                      "softmax denominator bit bound must be in [1, 63]");
                    constrain_nonnegative_with_limbs(softmax_remainders_.at(index),
                                                     remainder,
                                                     denominator_bit_bound);
                    const zkboostAssignedCircuitValue remainder_gap =
                        add_scaled(lifted_softmax_denominators_.at(index),
                                   Fr(1),
                                   softmax_remainders_.at(index),
                                   Fr(-1),
                                   Fr(-1));
                    constrain_nonnegative_with_limbs(remainder_gap,
                                                     denominator - remainder - 1,
                                                     denominator_bit_bound);

                    const zkboostAssignedCircuitValue denominator_times_probability =
                        multiply(lifted_softmax_denominators_.at(index),
                                 probability_scaled_values_.at(index));
                    const zkboostAssignedCircuitValue quotient_plus_remainder =
                        add_scaled(denominator_times_probability,
                                   Fr(1),
                                   softmax_remainders_.at(index),
                                   Fr(1),
                                   Fr(0));
                    const zkboostAssignedCircuitValue scaled_exp =
                        affine(softmax_exp_values_.at(index), Fr(softmax_scale), Fr(0));
                    constrain_equal(quotient_plus_remainder, scaled_exp);
                }
            }
        }
    }

    void compile_residuals_and_hessians()
    {
        const std::int64_t softmax_scale = softmax_scale_from_encoding(training_.softmax_encoding);
        for (size_t iteration = 0; iteration < training_.iteration_count; ++iteration) {
            for (size_t cls = 0; cls < training_.class_count; ++cls) {
                for (size_t sample = 0; sample < training_.query_count; ++sample) {
                    const size_t index =
                        cube_index(training_.class_count, training_.query_count, iteration, cls, sample);
                    const zkboostAssignedCircuitValue label =
                        public_base_labels_.at(cls * training_.query_count + sample);
                    const zkboostAssignedCircuitValue expected_residual =
                        training_.loss_mode == forestTrainingLoss::FirstOrderRss
                            ? add_scaled(probability_scaled_values_.at(index),
                                         Fr(-1),
                                         label,
                                         Fr(softmax_scale),
                                         Fr(0))
                            : add_scaled(probability_scaled_values_.at(index),
                                         Fr(1),
                                         label,
                                         Fr(-softmax_scale),
                                         Fr(0));
                    constrain_equal(public_pseudo_residuals_.at(index), expected_residual);

                    if (training_.loss_mode == forestTrainingLoss::FirstOrderRss) {
                        constrain_equal(public_diagonal_hessians_.at(index), constant(softmax_scale));
                        constrain_equal(hessian_remainders_.at(index), zero_);
                    } else {
                        const zkboostAssignedCircuitValue scale_minus_probability =
                            affine(probability_scaled_values_.at(index), Fr(-1), Fr(softmax_scale));
                        const zkboostAssignedCircuitValue hessian_product =
                            multiply(probability_scaled_values_.at(index), scale_minus_probability);
                        add_lookup(hessian_remainders_.at(index),
                                   constant(kHessianRemainderRangeLookupTag),
                                   constant(0));
                        const zkboostAssignedCircuitValue scaled_hessian =
                            affine(public_diagonal_hessians_.at(index), Fr(softmax_scale), Fr(0));
                        const zkboostAssignedCircuitValue hessian_quotient_plus_remainder =
                            add_scaled(scaled_hessian,
                                       Fr(1),
                                       hessian_remainders_.at(index),
                                       Fr(1),
                                       Fr(0));
                        constrain_equal(hessian_product, hessian_quotient_plus_remainder);
                    }
                }
            }
        }
    }

    const forestTrainingSemanticTrace& training_;
    const zkboostSharedBoostingStateView& view_;

    zkboostAssignedCircuitValue zero_;
    zkboostAssignedCircuitValue one_;
    Fr score_offset_;

    std::vector<zkboostAssignedCircuitValue> public_tree_scores_;
    std::vector<zkboostAssignedCircuitValue> public_base_labels_;
    std::vector<zkboostAssignedCircuitValue> public_pseudo_residuals_;
    std::vector<zkboostAssignedCircuitValue> public_diagonal_hessians_;

    std::vector<zkboostAssignedCircuitValue> cumulative_scores_;
    std::vector<zkboostAssignedCircuitValue> encoded_cumulative_scores_;
    std::vector<zkboostAssignedCircuitValue> softmax_shifts_;
    std::vector<zkboostAssignedCircuitValue> shift_selectors_;
    std::vector<zkboostAssignedCircuitValue> lifted_softmax_shifts_;
    std::vector<zkboostAssignedCircuitValue> encoded_shifted_scores_;
    std::vector<zkboostAssignedCircuitValue> softmax_exp_values_;
    std::vector<zkboostAssignedCircuitValue> softmax_denominators_;
    std::vector<zkboostAssignedCircuitValue> lifted_softmax_denominators_;
    std::vector<zkboostAssignedCircuitValue> probability_scaled_values_;
    std::vector<zkboostAssignedCircuitValue> softmax_remainders_;
    std::vector<zkboostAssignedCircuitValue> hessian_remainders_;
};

zkboostNamedPublicSpans shared_boosting_state_named_public_spans(
    const zkboostSharedBoostingStateView& view)
{
    const size_t cube = view.summary.cube_size;
    return {
        {"tree_scores", {0, cube}},
        {"pseudo_residuals", {cube, cube}},
        {"diagonal_hessians", {2 * cube, cube}},
        {"base_labels", {3 * cube, view.summary.class_count * view.summary.sample_count}},
    };
}

} // namespace

std::vector<Fr> encode_zkboost_shared_boosting_state_public_inputs(
    const zkboostSharedBoostingStateView& view)
{
    std::vector<Fr> out;
    out.reserve(view.tree_scores.size() + view.pseudo_residuals.size() +
                view.diagonal_hessians.size() + view.base_labels.size());
    out.insert(out.end(), view.tree_scores.begin(), view.tree_scores.end());
    out.insert(out.end(), view.pseudo_residuals.begin(), view.pseudo_residuals.end());
    out.insert(out.end(), view.diagonal_hessians.begin(), view.diagonal_hessians.end());
    out.insert(out.end(), view.base_labels.begin(), view.base_labels.end());
    return out;
}

zkboostSharedBoostingStatePlonkInstance build_zkboost_shared_boosting_state_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const kzgPublicParameters& public_parameters)
{
    const zkboostSharedBoostingStateView view =
        build_zkboost_shared_boosting_state_view(training);
    return build_zkboost_shared_boosting_state_plonk_instance_from_view(
        training,
        view,
        public_parameters);
}

zkboostSharedBoostingStatePlonkInstance build_zkboost_shared_boosting_state_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& view,
    const kzgPublicParameters& public_parameters)
{
    sharedBoostingStateCircuitCompiler compiler(training, view);
    compiler.compile();
    const zkboostCompiledCircuitStatement statement =
        compile_zkboost_generic_circuit_statement("shared_boosting_state",
                                                  compiler.build_circuit(),
                                                  public_parameters,
                                                  shared_boosting_state_named_public_spans(view));

    zkboostSharedBoostingStatePlonkInstance out;
    out.setup = statement.setup;
    out.witness = statement.witness;
    out.public_inputs = statement.public_inputs;
    out.summary = view.summary;
    return out;
}

size_t required_zkboost_shared_boosting_state_plonk_srs_size(
    const forestTrainingSemanticTrace& training)
{
    return required_zkboost_shared_boosting_state_plonk_srs_size_from_view(
        training,
        build_zkboost_shared_boosting_state_view(training));
}

size_t required_zkboost_shared_boosting_state_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& view)
{
    sharedBoostingStateCircuitCompiler compiler(training, view);
    compiler.compile();
    return required_zkboost_generic_circuit_plonk_srs_size(compiler.build_circuit());
}

plonkProof prove_zkboost_shared_boosting_state_plonk(
    const zkboostSharedBoostingStatePlonkInstance& instance)
{
    return prove_plonk(instance.setup.proving_key, instance.witness, instance.public_inputs);
}

bool verify_zkboost_shared_boosting_state_plonk(
    const zkboostSharedBoostingStatePlonkInstance& instance,
    const plonkProof& proof)
{
    return verify_zkboost_shared_boosting_state_plonk(
        instance.setup.verification_key,
        instance.public_inputs,
        proof);
}

bool verify_zkboost_shared_boosting_state_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof)
{
    return verify_plonk(verification_key, public_inputs, proof);
}

bool link_zkboost_shared_validate_forest_to_boosting_state(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedBoostingStatePlonkInstance& boosting_instance)
{
    const size_t forest_tree_score_offset = forest_instance.summary.checked_paths;
    const size_t count = boosting_instance.summary.cube_size;
    if (forest_instance.summary.checked_paths != count) {
        return false;
    }
    if (forest_instance.public_inputs.size() < forest_tree_score_offset + count) {
        return false;
    }
    if (boosting_instance.public_inputs.size() < count) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (forest_instance.public_inputs.at(forest_tree_score_offset + i) !=
            boosting_instance.public_inputs.at(i)) {
            return false;
        }
    }
    return true;
}
