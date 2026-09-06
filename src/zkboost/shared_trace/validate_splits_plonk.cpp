#include "zkboost/shared_trace/validate_splits_plonk.hpp"

#include "algebra_utils.hpp"
#include "fixed_point.hpp"
#include "zkboost/compiler/generic_circuit.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_split_plonk: " + message);
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

size_t candidate_offset(size_t feature_count,
                        size_t bin_count,
                        size_t node,
                        size_t feature,
                        size_t bin)
{
    return node * feature_count * bin_count + feature * bin_count + bin;
}

std::int64_t split_division_scale(const forestTrainingSemanticTrace& training)
{
    require_condition(training.quantized_gain_encoding.fractional_bits >=
                          training.gain_numerator_encoding.fractional_bits,
                      "quantized gain fractional bits must be at least gain numerator fractional bits");
    const size_t scale_power =
        training.quantized_gain_encoding.fractional_bits -
        training.gain_numerator_encoding.fractional_bits;
    require_condition(scale_power < static_cast<size_t>(std::numeric_limits<std::int64_t>::digits),
                      "division scale power exceeds int64 range");
    return std::int64_t(1) << scale_power;
}

void validate_view_shapes(const forestTrainingSemanticTrace& training,
                          const zkboostSharedSplitView& view)
{
    require_condition(training.node_count == view.summary.node_count, "node_count mismatch");
    require_condition(training.feature_count == view.summary.feature_count, "feature_count mismatch");
    require_condition(training.bin_count == view.summary.bin_count, "bin_count mismatch");
    require_condition(training.feature_count > 0, "feature_count must be non-zero");
    require_condition(training.bin_count > 0, "bin_count must be non-zero");
    require_condition(view.summary.candidate_count ==
                          training.node_count * training.feature_count * training.bin_count,
                      "candidate_count mismatch");
    require_condition(training.split_node_mask.size() == training.node_count,
                      "split_node_mask length mismatch");
    require_condition(training.feature_selector.size() == training.node_count * training.feature_count,
                      "feature_selector length mismatch");
    require_condition(training.thresholds.size() == training.node_count,
                      "thresholds length mismatch");
    require_condition(training.candidate_mask.size() == view.summary.candidate_count,
                      "candidate_mask length mismatch");
    require_condition(training.selected_candidate_indices.size() == training.node_count,
                      "selected_candidate_indices length mismatch");

    const size_t c = view.summary.candidate_count;
    require_condition(view.left_counts.size() == c, "left_counts length mismatch");
    require_condition(view.right_counts.size() == c, "right_counts length mismatch");
    require_condition(view.candidate_masks.size() == c, "candidate_masks length mismatch");
    require_condition(view.left_gradients.size() == c, "left_gradients length mismatch");
    require_condition(view.right_gradients.size() == c, "right_gradients length mismatch");
    require_condition(view.left_hessians.size() == c, "left_hessians length mismatch");
    require_condition(view.right_hessians.size() == c, "right_hessians length mismatch");
    require_condition(view.parent_gradients.size() == c, "parent_gradients length mismatch");
    require_condition(view.parent_hessians.size() == c, "parent_hessians length mismatch");
    require_condition(view.gain_numerators.size() == c, "gain_numerators length mismatch");
    require_condition(view.gain_denominators.size() == c, "gain_denominators length mismatch");
    require_condition(view.encoded_gain_numerators.size() == c,
                      "encoded_gain_numerators length mismatch");
    require_condition(view.quantized_gains.size() == c, "quantized_gains length mismatch");
    require_condition(view.encoded_quantized_gains.size() == c,
                      "encoded_quantized_gains length mismatch");
    require_condition(view.gain_remainders.size() == c, "gain_remainders length mismatch");
    require_condition(view.tie_break_scores.size() == c, "tie_break_scores length mismatch");
    require_condition(view.selector.size() == c, "selector length mismatch");

    require_condition(view.selected_gains.size() == training.node_count,
                      "selected_gains length mismatch");
    require_condition(view.selected_tie_break_scores.size() == training.node_count,
                      "selected_tie_break_scores length mismatch");
    require_condition(view.selected_features.size() == training.node_count,
                      "selected_features length mismatch");
    require_condition(view.selected_bins.size() == training.node_count,
                      "selected_bins length mismatch");
    require_condition(view.split_masks.size() == training.node_count,
                      "split_masks length mismatch");
    require_condition(view.active_node_masks.size() == training.node_count,
                      "active_node_masks length mismatch");
    require_condition(view.leaf_masks.size() == training.node_count,
                      "leaf_masks length mismatch");
    require_condition(view.node_depths.size() == training.node_count,
                      "node_depths length mismatch");
    require_condition(view.depth_limit_masks.size() == training.node_count,
                      "depth_limit_masks length mismatch");
    validate_fixed_point_encoding(training.gain_numerator_encoding,
                                  "zkboost_shared_split_plonk gain numerator");
    validate_fixed_point_encoding(training.quantized_gain_encoding,
                                  "zkboost_shared_split_plonk quantized gain");
    require_condition(training.gain_numerator_encoding.signed_values,
                      "gain numerator encoding must be signed");
    require_condition(training.quantized_gain_encoding.signed_values,
                      "quantized gain encoding must be signed");
}

class sharedSplitCircuitCompiler : public zkboostGenericCircuitCompilerBase {
public:
    sharedSplitCircuitCompiler(const forestTrainingSemanticTrace& training,
                               const zkboostSharedSplitView& view)
        : training_(training)
        , view_(view)
    {
        validate_view_shapes(training_, view_);
        require_condition(training_.query_count > 0, "query_count must be non-zero");
        require_condition(training_.gain_denominator_bit_bound > 0 &&
                              training_.gain_denominator_bit_bound <= 63,
                          "gain_denominator_bit_bound must be in [1, 63]");
        require_condition(training_.gain_gap_bit_bound > 0,
                          "gain_gap_bit_bound must be non-zero");
        require_condition(training_.softmax_encoding.fractional_bits <
                              std::numeric_limits<size_t>::digits,
                          "softmax fractional bits exceed size_t scale range");
        gain_numerator_offset_ = fixed_point_offset_fr(training_.gain_numerator_encoding);
        quantized_gain_offset_ = fixed_point_offset_fr(training_.quantized_gain_encoding);
        division_scale_ = split_division_scale(training_);
        add_candidate_nonzero_lookup_table();
        initialize_public_inputs();
    }

    void compile()
    {
        compile_candidate_relations();
        compile_selection_relations();
    }

private:
    static constexpr std::int64_t kCandidateNonzeroLookupTag = 3100000011LL;

    static size_t nonnegative_bit_width(size_t maximum)
    {
        size_t bits = 1;
        while (maximum > 1) {
            maximum >>= 1;
            ++bits;
        }
        return bits;
    }

    void add_candidate_nonzero_lookup_table()
    {
        // This table is determined entirely by the public sample count.  It is
        // independent of all forest/training witnesses.
        for (size_t count = 0; count <= training_.query_count; ++count) {
            add_lookup_table_row(checked_i64(count, "candidate count"),
                                 kCandidateNonzeroLookupTag,
                                 count == 0 ? 0 : 1);
        }
    }

    void append_public_vector(const std::vector<Fr>& values,
                              std::vector<zkboostAssignedCircuitValue>& out,
                              const char* context)
    {
        out.reserve(out.size() + values.size());
        for (const Fr& value : values) {
            out.push_back(public_value(checked_i64_from_fr(value, context)));
        }
    }

    void append_public_encoded_vector(const std::vector<Fr>& values,
                                      std::vector<zkboostAssignedCircuitValue>& out,
                                      const char* context)
    {
        out.reserve(out.size() + values.size());
        for (const Fr& value : values) {
            out.push_back(public_value(checked_nonnegative_i64_from_fr(value, context)));
        }
    }

    void initialize_public_inputs()
    {
        append_public_vector(view_.left_gradients, public_left_gradients_, "split left_gradients");
        append_public_vector(view_.right_gradients, public_right_gradients_, "split right_gradients");
        append_public_vector(view_.left_hessians, public_left_hessians_, "split left_hessians");
        append_public_vector(view_.right_hessians, public_right_hessians_, "split right_hessians");
        append_public_vector(view_.parent_gradients, public_parent_gradients_, "split parent_gradients");
        append_public_vector(view_.parent_hessians, public_parent_hessians_, "split parent_hessians");
        append_public_vector(view_.gain_numerators, public_gain_numerators_, "split gain_numerators");
        append_public_vector(view_.gain_denominators, public_gain_denominators_, "split gain_denominators");
        append_public_encoded_vector(view_.encoded_gain_numerators,
                                     public_encoded_gain_numerators_,
                                     "split encoded_gain_numerators");
        append_public_vector(view_.quantized_gains, public_quantized_gains_, "split quantized_gains");
        append_public_encoded_vector(view_.encoded_quantized_gains,
                                     public_encoded_quantized_gains_,
                                     "split encoded_quantized_gains");
        append_public_vector(view_.tie_break_scores, public_tie_break_scores_, "split tie_break_scores");
        append_public_vector(view_.selector, public_selector_, "split selector");
        append_public_vector(view_.left_counts, public_left_counts_, "split left_counts");
        append_public_vector(view_.right_counts, public_right_counts_, "split right_counts");
        append_public_vector(view_.candidate_masks, public_candidate_masks_, "split candidate_masks");
        append_public_vector(view_.selected_gains, public_selected_gains_, "split selected_gains");
        append_public_vector(view_.selected_tie_break_scores,
                             public_selected_tie_break_scores_,
                             "split selected_tie_break_scores");
        append_public_vector(view_.selected_features, public_selected_features_, "split selected_features");
        append_public_vector(view_.selected_bins, public_selected_bins_, "split selected_bins");
        append_public_vector(view_.split_masks, public_split_masks_, "split split_masks");
        append_public_vector(view_.active_node_masks, public_active_node_masks_, "split active_node_masks");
        append_public_vector(view_.leaf_masks, public_leaf_masks_, "split leaf_masks");
        append_public_vector(view_.node_depths, public_node_depths_, "split node_depths");
        append_public_vector(view_.depth_limit_masks, public_depth_limit_masks_, "split depth_limit_masks");

        private_gain_remainders_.reserve(view_.summary.candidate_count);
        for (const Fr& value : view_.gain_remainders) {
            private_gain_remainders_.push_back(
                private_value(checked_i64_from_fr(value, "split gain_remainders")));
        }
    }

    void compile_candidate_relations()
    {
        const Fr lambda = field_from_size_t(training_.l2_regularization);
        const Fr first_order_hessian_scale =
            field_from_size_t(size_t(1) << training_.softmax_encoding.fractional_bits);
        const Fr column_count_fr = field_from_size_t(training_.feature_count * training_.bin_count);

        for (size_t node = 0; node < training_.node_count; ++node) {
            for (size_t feature = 0; feature < training_.feature_count; ++feature) {
                for (size_t bin = 0; bin < training_.bin_count; ++bin) {
                    const size_t index =
                        candidate_offset(training_.feature_count, training_.bin_count, node, feature, bin);
                    const std::int64_t left_count =
                        checked_nonnegative_i64_from_fr(view_.left_counts.at(index), "split left_counts");
                    const std::int64_t right_count =
                        checked_nonnegative_i64_from_fr(view_.right_counts.at(index), "split right_counts");
                    require_condition(static_cast<size_t>(left_count) <= training_.query_count &&
                                          static_cast<size_t>(right_count) <= training_.query_count,
                                      "candidate count exceeds query_count");
                    const zkboostAssignedCircuitValue left_nonzero = private_value(left_count == 0 ? 0 : 1);
                    const zkboostAssignedCircuitValue right_nonzero = private_value(right_count == 0 ? 0 : 1);
                    add_lookup(public_left_counts_.at(index),
                               constant(kCandidateNonzeroLookupTag),
                               left_nonzero);
                    add_lookup(public_right_counts_.at(index),
                               constant(kCandidateNonzeroLookupTag),
                               right_nonzero);
                    const zkboostAssignedCircuitValue both_nonzero = multiply(left_nonzero, right_nonzero);
                    const zkboostAssignedCircuitValue expected_mask =
                        bin + 1 < training_.bin_count ? both_nonzero : constant(0);
                    constrain_boolean(public_candidate_masks_.at(index));
                    constrain_equal(public_candidate_masks_.at(index), expected_mask);

                    constrain_equal(public_parent_gradients_.at(index),
                                    add_scaled(public_left_gradients_.at(index),
                                               Fr(1),
                                               public_right_gradients_.at(index),
                                               Fr(1),
                                               Fr(0)));
                    constrain_equal(public_parent_hessians_.at(index),
                                    add_scaled(public_left_hessians_.at(index),
                                               Fr(1),
                                               public_right_hessians_.at(index),
                                               Fr(1),
                                               Fr(0)));

                    zkboostAssignedCircuitValue left_den =
                        affine(public_left_hessians_.at(index), Fr(1), lambda);
                    zkboostAssignedCircuitValue right_den =
                        affine(public_right_hessians_.at(index), Fr(1), lambda);
                    zkboostAssignedCircuitValue parent_den =
                        affine(public_parent_hessians_.at(index), Fr(1), lambda);
                    Fr denominator_scale(1);
                    if (training_.loss_mode == forestTrainingLoss::FirstOrderRss) {
                        const zkboostAssignedCircuitValue inactive = one_minus(public_candidate_masks_.at(index));
                        left_den = add_scaled(public_left_counts_.at(index), Fr(1), inactive, Fr(1), Fr(0));
                        right_den = add_scaled(public_right_counts_.at(index), Fr(1), inactive, Fr(1), Fr(0));
                        parent_den = add_scaled(public_left_counts_.at(index),
                                                Fr(1),
                                                public_right_counts_.at(index),
                                                Fr(1),
                                                Fr(0));
                        parent_den = add_scaled(parent_den, Fr(1), inactive, Fr(1), Fr(0));
                        denominator_scale = first_order_hessian_scale;
                    }
                    const zkboostAssignedCircuitValue left_right_den = multiply(left_den, right_den);
                    const zkboostAssignedCircuitValue expected_denominator = affine(
                        multiply(left_right_den, parent_den), denominator_scale, Fr(0));
                    constrain_equal(public_gain_denominators_.at(index), expected_denominator);

                    const zkboostAssignedCircuitValue left_square =
                        multiply(public_left_gradients_.at(index), public_left_gradients_.at(index));
                    const zkboostAssignedCircuitValue right_square =
                        multiply(public_right_gradients_.at(index), public_right_gradients_.at(index));
                    const zkboostAssignedCircuitValue parent_square =
                        multiply(public_parent_gradients_.at(index), public_parent_gradients_.at(index));
                    const zkboostAssignedCircuitValue right_parent_den = multiply(right_den, parent_den);
                    const zkboostAssignedCircuitValue left_parent_den = multiply(left_den, parent_den);
                    const zkboostAssignedCircuitValue left_term = multiply(left_square, right_parent_den);
                    const zkboostAssignedCircuitValue right_term = multiply(right_square, left_parent_den);
                    const zkboostAssignedCircuitValue parent_term = multiply(parent_square, left_right_den);
                    const zkboostAssignedCircuitValue left_plus_right =
                        add_scaled(left_term, Fr(1), right_term, Fr(1), Fr(0));
                    const zkboostAssignedCircuitValue without_parent =
                        add_scaled(left_plus_right, Fr(1), parent_term, Fr(-1), Fr(0));
                    const zkboostAssignedCircuitValue penalty =
                        affine(public_gain_denominators_.at(index), training_.split_penalty, Fr(0));
                    const zkboostAssignedCircuitValue expected_numerator =
                        add_scaled(without_parent, Fr(1), penalty, Fr(-1), Fr(0));
                    constrain_equal(public_gain_numerators_.at(index), expected_numerator);

                    constrain_equal(public_encoded_gain_numerators_.at(index),
                                    affine(public_gain_numerators_.at(index),
                                           Fr(1),
                                           gain_numerator_offset_));
                    constrain_nonnegative_with_limbs(
                        public_encoded_gain_numerators_.at(index),
                        checked_nonnegative_i64_from_fr(view_.encoded_gain_numerators.at(index),
                                                        "split encoded_gain_numerators"),
                        training_.gain_numerator_encoding.total_bits);
                    constrain_equal(public_encoded_quantized_gains_.at(index),
                                    affine(public_quantized_gains_.at(index),
                                           Fr(1),
                                           quantized_gain_offset_));
                    constrain_nonnegative_with_limbs(
                        public_encoded_quantized_gains_.at(index),
                        checked_nonnegative_i64_from_fr(view_.encoded_quantized_gains.at(index),
                                                        "split encoded_quantized_gains"),
                        training_.quantized_gain_encoding.total_bits);

                    const std::int64_t denominator =
                        checked_nonnegative_i64_from_fr(view_.gain_denominators.at(index),
                                                        "split gain_denominators");
                    const std::int64_t remainder =
                        checked_nonnegative_i64_from_fr(view_.gain_remainders.at(index),
                                                        "split gain_remainders");
                    require_condition(denominator > 0 && remainder < denominator,
                                      "split gain remainder must be in [0, denominator)");
                    const size_t division_range_bits = std::max(
                        training_.gain_denominator_bit_bound,
                        training_.gain_numerator_encoding.total_bits);
                    constrain_nonnegative_with_limbs(private_gain_remainders_.at(index),
                                                     remainder,
                                                     division_range_bits);
                    const zkboostAssignedCircuitValue remainder_gap =
                        add_scaled(public_gain_denominators_.at(index),
                                   Fr(1),
                                   private_gain_remainders_.at(index),
                                   Fr(-1),
                                   Fr(-1));
                    constrain_nonnegative_with_limbs(remainder_gap,
                                                     denominator - remainder - 1,
                                                     division_range_bits);
                    const zkboostAssignedCircuitValue denominator_times_gain =
                        multiply(public_gain_denominators_.at(index),
                                 public_quantized_gains_.at(index));
                    const zkboostAssignedCircuitValue quotient_plus_remainder =
                        add_scaled(denominator_times_gain,
                                   Fr(1),
                                   private_gain_remainders_.at(index),
                                   Fr(1),
                                   Fr(0));
                    constrain_equal(quotient_plus_remainder,
                                    affine(public_gain_numerators_.at(index),
                                           Fr(division_scale_),
                                           Fr(0)));

                    const size_t column = feature * training_.bin_count + bin;
                    const Fr tie_break_offset =
                        field_from_size_t(training_.feature_count * training_.bin_count - 1 - column);
                    constrain_equal(public_tie_break_scores_.at(index),
                                    affine(public_quantized_gains_.at(index),
                                           column_count_fr,
                                           tie_break_offset));
                }
            }
        }
    }

    void compile_selection_relations()
    {
        const size_t column_count = training_.feature_count * training_.bin_count;
        const size_t tie_gap_bits = std::max(
            training_.gain_gap_bit_bound + nonnegative_bit_width(column_count) + 1,
            training_.quantized_gain_encoding.total_bits + nonnegative_bit_width(column_count));
        const size_t stopping_gap_bits = std::max(
            training_.gain_gap_bit_bound + 1,
            training_.quantized_gain_encoding.total_bits + 1);
        require_condition(tie_gap_bits <= 63 && stopping_gap_bits <= 63,
                          "split gap bit bound exceeds 63 bits");
        const std::int64_t max_depth = checked_i64(training_.max_depth, "max_depth");
        const std::int64_t split_gain_threshold =
            checked_i64_from_fr(training_.split_gain_threshold, "split_gain_threshold");

        for (size_t node = 0; node < training_.node_count; ++node) {
            const std::int64_t active =
                checked_i64_from_fr(view_.active_node_masks.at(node), "active_node_masks");
            const std::int64_t split =
                checked_i64_from_fr(view_.split_masks.at(node), "split_masks");
            const std::int64_t leaf =
                checked_i64_from_fr(view_.leaf_masks.at(node), "leaf_masks");
            const std::int64_t depth =
                checked_i64_from_fr(view_.node_depths.at(node), "node_depths");
            const std::int64_t depth_limit =
                checked_i64_from_fr(view_.depth_limit_masks.at(node), "depth_limit_masks");
            require_condition((active == 0 || active == 1) &&
                                  (split == 0 || split == 1) &&
                                  (leaf == 0 || leaf == 1) &&
                                  (depth_limit == 0 || depth_limit == 1),
                              "node masks must be binary");
            constrain_boolean(public_active_node_masks_.at(node));
            constrain_boolean(public_split_masks_.at(node));
            constrain_boolean(public_leaf_masks_.at(node));
            constrain_boolean(public_depth_limit_masks_.at(node));
            constrain_equal(multiply(public_split_masks_.at(node), public_leaf_masks_.at(node)),
                            constant(0));
            constrain_nonnegative_at_most_with_limbs(public_node_depths_.at(node),
                                                      depth,
                                                      max_depth);
            const zkboostAssignedCircuitValue reachable =
                add_scaled(public_split_masks_.at(node),
                           Fr(1),
                           public_leaf_masks_.at(node),
                           Fr(1),
                           Fr(0));
            constrain_equal(multiply(public_active_node_masks_.at(node), one_minus(reachable)),
                            constant(0));
            constrain_equal(multiply(public_split_masks_.at(node),
                                     one_minus(public_active_node_masks_.at(node))),
                            constant(0));

            zkboostAssignedCircuitValue selector_sum = constant(0);
            zkboostAssignedCircuitValue selected_gain_sum = constant(0);
            zkboostAssignedCircuitValue selected_tie_break_sum = constant(0);
            zkboostAssignedCircuitValue selected_feature_sum = constant(0);
            zkboostAssignedCircuitValue selected_bin_sum = constant(0);

            for (size_t feature = 0; feature < training_.feature_count; ++feature) {
                for (size_t bin = 0; bin < training_.bin_count; ++bin) {
                    const size_t column = feature * training_.bin_count + bin;
                    const size_t index = node * column_count + column;
                    const std::int64_t mask =
                        checked_i64_from_fr(view_.candidate_masks.at(index), "candidate_masks");
                    const std::int64_t selector =
                        checked_i64_from_fr(view_.selector.at(index), "selector");
                    constrain_boolean(public_selector_.at(index));
                    constrain_equal(multiply(public_selector_.at(index),
                                             one_minus(public_candidate_masks_.at(index))),
                                    constant(0));
                    constrain_equal(multiply(public_candidate_masks_.at(index),
                                             one_minus(public_active_node_masks_.at(node))),
                                    constant(0));

                    selector_sum = add_scaled(selector_sum,
                                              Fr(1),
                                              public_selector_.at(index),
                                              Fr(1),
                                              Fr(0));
                    selected_gain_sum = add_scaled(
                        selected_gain_sum,
                        Fr(1),
                        multiply(public_selector_.at(index), public_quantized_gains_.at(index)),
                        Fr(1),
                        Fr(0));
                    selected_tie_break_sum = add_scaled(
                        selected_tie_break_sum,
                        Fr(1),
                        multiply(public_selector_.at(index), public_tie_break_scores_.at(index)),
                        Fr(1),
                        Fr(0));
                    selected_feature_sum = add_scaled(selected_feature_sum,
                                                      Fr(1),
                                                      public_selector_.at(index),
                                                      Fr(checked_i64(feature, "feature")),
                                                      Fr(0));
                    selected_bin_sum = add_scaled(selected_bin_sum,
                                                  Fr(1),
                                                  public_selector_.at(index),
                                                  Fr(checked_i64(bin, "bin")),
                                                  Fr(0));

                    const zkboostAssignedCircuitValue raw_gap =
                        add_scaled(public_selected_tie_break_scores_.at(node),
                                   Fr(1),
                                   public_tie_break_scores_.at(index),
                                   Fr(-1),
                                   Fr(0));
                    const zkboostAssignedCircuitValue masked_gap =
                        multiply(public_candidate_masks_.at(index), raw_gap);
                    const std::int64_t selected_tie =
                        checked_i64_from_fr(view_.selected_tie_break_scores.at(node),
                                            "selected_tie_break_scores");
                    const std::int64_t candidate_tie =
                        checked_i64_from_fr(view_.tie_break_scores.at(index), "tie_break_scores");
                    const std::int64_t gap_witness = mask * (selected_tie - candidate_tie);
                    constrain_nonnegative_with_limbs(masked_gap, gap_witness, tie_gap_bits);
                    (void)selector;
                }
            }

            constrain_equal(selector_sum, public_active_node_masks_.at(node));
            constrain_equal(public_selected_gains_.at(node), selected_gain_sum);
            constrain_equal(public_selected_tie_break_scores_.at(node), selected_tie_break_sum);
            constrain_equal(public_selected_features_.at(node), selected_feature_sum);
            constrain_equal(public_selected_bins_.at(node), selected_bin_sum);

            const zkboostAssignedCircuitValue split_gain_gap_raw =
                affine(public_selected_gains_.at(node), Fr(1), Fr(-split_gain_threshold));
            const zkboostAssignedCircuitValue split_gain_gap =
                multiply(public_split_masks_.at(node), split_gain_gap_raw);
            const std::int64_t selected_gain =
                checked_i64_from_fr(view_.selected_gains.at(node), "selected_gains");
            constrain_nonnegative_with_limbs(split_gain_gap,
                                             split * (selected_gain - split_gain_threshold),
                                             stopping_gap_bits);

            const zkboostAssignedCircuitValue remaining_depth_raw =
                affine(public_node_depths_.at(node), Fr(-1), Fr(max_depth - 1));
            const zkboostAssignedCircuitValue remaining_depth =
                multiply(public_split_masks_.at(node), remaining_depth_raw);
            constrain_nonnegative_at_most_with_limbs(remaining_depth,
                                                      split * (max_depth - 1 - depth),
                                                      max_depth);

            const zkboostAssignedCircuitValue active_leaf =
                multiply(public_leaf_masks_.at(node), public_active_node_masks_.at(node));
            constrain_equal(multiply(public_depth_limit_masks_.at(node), one_minus(active_leaf)),
                            constant(0));
            const zkboostAssignedCircuitValue depth_limit_difference =
                affine(public_node_depths_.at(node), Fr(1), Fr(-max_depth));
            constrain_equal(multiply(public_depth_limit_masks_.at(node), depth_limit_difference),
                            constant(0));
            const zkboostAssignedCircuitValue threshold_stop_mask =
                multiply(active_leaf, one_minus(public_depth_limit_masks_.at(node)));
            const zkboostAssignedCircuitValue threshold_gap_raw =
                affine(public_selected_gains_.at(node),
                       Fr(-1),
                       Fr(split_gain_threshold - 1));
            const zkboostAssignedCircuitValue threshold_gap =
                multiply(threshold_stop_mask, threshold_gap_raw);
            const std::int64_t threshold_mask_witness = leaf * active * (1 - depth_limit);
            constrain_nonnegative_with_limbs(
                threshold_gap,
                threshold_mask_witness * (split_gain_threshold - selected_gain - 1),
                stopping_gap_bits);
        }
    }

    const forestTrainingSemanticTrace& training_;
    const zkboostSharedSplitView& view_;

    Fr gain_numerator_offset_;
    Fr quantized_gain_offset_;
    std::int64_t division_scale_ = 1;

    std::vector<zkboostAssignedCircuitValue> public_left_gradients_;
    std::vector<zkboostAssignedCircuitValue> public_right_gradients_;
    std::vector<zkboostAssignedCircuitValue> public_left_hessians_;
    std::vector<zkboostAssignedCircuitValue> public_right_hessians_;
    std::vector<zkboostAssignedCircuitValue> public_parent_gradients_;
    std::vector<zkboostAssignedCircuitValue> public_parent_hessians_;
    std::vector<zkboostAssignedCircuitValue> public_gain_numerators_;
    std::vector<zkboostAssignedCircuitValue> public_gain_denominators_;
    std::vector<zkboostAssignedCircuitValue> public_encoded_gain_numerators_;
    std::vector<zkboostAssignedCircuitValue> public_quantized_gains_;
    std::vector<zkboostAssignedCircuitValue> public_encoded_quantized_gains_;
    std::vector<zkboostAssignedCircuitValue> public_tie_break_scores_;
    std::vector<zkboostAssignedCircuitValue> public_selector_;
    std::vector<zkboostAssignedCircuitValue> public_left_counts_;
    std::vector<zkboostAssignedCircuitValue> public_right_counts_;
    std::vector<zkboostAssignedCircuitValue> public_candidate_masks_;
    std::vector<zkboostAssignedCircuitValue> public_selected_gains_;
    std::vector<zkboostAssignedCircuitValue> public_selected_tie_break_scores_;
    std::vector<zkboostAssignedCircuitValue> public_selected_features_;
    std::vector<zkboostAssignedCircuitValue> public_selected_bins_;
    std::vector<zkboostAssignedCircuitValue> public_split_masks_;
    std::vector<zkboostAssignedCircuitValue> public_active_node_masks_;
    std::vector<zkboostAssignedCircuitValue> public_leaf_masks_;
    std::vector<zkboostAssignedCircuitValue> public_node_depths_;
    std::vector<zkboostAssignedCircuitValue> public_depth_limit_masks_;
    std::vector<zkboostAssignedCircuitValue> private_gain_remainders_;
};

zkboostNamedPublicSpans split_named_public_spans(const zkboostSharedSplitView& view)
{
    const size_t c = view.summary.candidate_count;
    const size_t n = view.summary.node_count;
    size_t offset = 0;
    zkboostNamedPublicSpans spans;
    spans["left_gradients"] = {offset, c};
    offset += c;
    spans["right_gradients"] = {offset, c};
    offset += c;
    spans["left_hessians"] = {offset, c};
    offset += c;
    spans["right_hessians"] = {offset, c};
    offset += c;
    spans["parent_gradients"] = {offset, c};
    offset += c;
    spans["parent_hessians"] = {offset, c};
    offset += c;
    spans["gain_numerators"] = {offset, c};
    offset += c;
    spans["gain_denominators"] = {offset, c};
    offset += c;
    spans["encoded_gain_numerators"] = {offset, c};
    offset += c;
    spans["quantized_gains"] = {offset, c};
    offset += c;
    spans["encoded_quantized_gains"] = {offset, c};
    offset += c;
    spans["tie_break_scores"] = {offset, c};
    offset += c;
    spans["selector"] = {offset, c};
    offset += c;
    spans["left_counts"] = {offset, c};
    offset += c;
    spans["right_counts"] = {offset, c};
    offset += c;
    spans["candidate_masks"] = {offset, c};
    offset += c;
    spans["selected_gains"] = {offset, n};
    offset += n;
    spans["selected_tie_break_scores"] = {offset, n};
    offset += n;
    spans["selected_features"] = {offset, n};
    offset += n;
    spans["selected_bins"] = {offset, n};
    offset += n;
    spans["split_masks"] = {offset, n};
    offset += n;
    spans["active_node_masks"] = {offset, n};
    offset += n;
    spans["leaf_masks"] = {offset, n};
    offset += n;
    spans["node_depths"] = {offset, n};
    offset += n;
    spans["depth_limit_masks"] = {offset, n};
    return spans;
}

} // namespace

std::vector<Fr> encode_zkboost_shared_split_public_inputs(
    const zkboostSharedSplitView& view)
{
    std::vector<Fr> out;
    out.reserve(view.summary.candidate_count * 16 + view.summary.node_count * 9);
    out.insert(out.end(), view.left_gradients.begin(), view.left_gradients.end());
    out.insert(out.end(), view.right_gradients.begin(), view.right_gradients.end());
    out.insert(out.end(), view.left_hessians.begin(), view.left_hessians.end());
    out.insert(out.end(), view.right_hessians.begin(), view.right_hessians.end());
    out.insert(out.end(), view.parent_gradients.begin(), view.parent_gradients.end());
    out.insert(out.end(), view.parent_hessians.begin(), view.parent_hessians.end());
    out.insert(out.end(), view.gain_numerators.begin(), view.gain_numerators.end());
    out.insert(out.end(), view.gain_denominators.begin(), view.gain_denominators.end());
    out.insert(out.end(), view.encoded_gain_numerators.begin(), view.encoded_gain_numerators.end());
    out.insert(out.end(), view.quantized_gains.begin(), view.quantized_gains.end());
    out.insert(out.end(), view.encoded_quantized_gains.begin(), view.encoded_quantized_gains.end());
    out.insert(out.end(), view.tie_break_scores.begin(), view.tie_break_scores.end());
    out.insert(out.end(), view.selector.begin(), view.selector.end());
    out.insert(out.end(), view.left_counts.begin(), view.left_counts.end());
    out.insert(out.end(), view.right_counts.begin(), view.right_counts.end());
    out.insert(out.end(), view.candidate_masks.begin(), view.candidate_masks.end());
    out.insert(out.end(), view.selected_gains.begin(), view.selected_gains.end());
    out.insert(out.end(), view.selected_tie_break_scores.begin(), view.selected_tie_break_scores.end());
    out.insert(out.end(), view.selected_features.begin(), view.selected_features.end());
    out.insert(out.end(), view.selected_bins.begin(), view.selected_bins.end());
    out.insert(out.end(), view.split_masks.begin(), view.split_masks.end());
    out.insert(out.end(), view.active_node_masks.begin(), view.active_node_masks.end());
    out.insert(out.end(), view.leaf_masks.begin(), view.leaf_masks.end());
    out.insert(out.end(), view.node_depths.begin(), view.node_depths.end());
    out.insert(out.end(), view.depth_limit_masks.begin(), view.depth_limit_masks.end());
    return out;
}

zkboostSharedSplitPlonkInstance build_zkboost_shared_split_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const kzgPublicParameters& public_parameters)
{
    return build_zkboost_shared_split_plonk_instance(
        training,
        build_zkboost_shared_histogram_view(training),
        public_parameters);
}

zkboostSharedSplitPlonkInstance build_zkboost_shared_split_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms,
    const kzgPublicParameters& public_parameters)
{
    return build_zkboost_shared_split_plonk_instance_from_view(
        training,
        build_zkboost_shared_split_view(training, histograms),
        public_parameters);
}

zkboostSharedSplitPlonkInstance build_zkboost_shared_split_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedSplitView& view,
    const kzgPublicParameters& public_parameters)
{
    sharedSplitCircuitCompiler compiler(training, view);
    compiler.compile();
    const zkboostCompiledCircuitStatement statement =
        compile_zkboost_generic_circuit_statement("shared_validate_splits",
                                                  compiler.build_circuit(),
                                                  public_parameters,
                                                  split_named_public_spans(view));

    zkboostSharedSplitPlonkInstance out;
    out.setup = statement.setup;
    out.witness = statement.witness;
    out.public_inputs = statement.public_inputs;
    out.summary = view.summary;
    return out;
}

size_t required_zkboost_shared_split_plonk_srs_size(
    const forestTrainingSemanticTrace& training)
{
    return required_zkboost_shared_split_plonk_srs_size(
        training,
        build_zkboost_shared_histogram_view(training));
}

size_t required_zkboost_shared_split_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms)
{
    return required_zkboost_shared_split_plonk_srs_size_from_view(
        training,
        build_zkboost_shared_split_view(training, histograms));
}

size_t required_zkboost_shared_split_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedSplitView& view)
{
    sharedSplitCircuitCompiler compiler(training, view);
    compiler.compile();
    return required_zkboost_generic_circuit_plonk_srs_size(compiler.build_circuit());
}

plonkProof prove_zkboost_shared_split_plonk(
    const zkboostSharedSplitPlonkInstance& instance)
{
    return prove_plonk(instance.setup.proving_key, instance.witness, instance.public_inputs);
}

bool verify_zkboost_shared_split_plonk(
    const zkboostSharedSplitPlonkInstance& instance,
    const plonkProof& proof)
{
    return verify_zkboost_shared_split_plonk(
        instance.setup.verification_key,
        instance.public_inputs,
        proof);
}

bool verify_zkboost_shared_split_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof)
{
    return verify_plonk(verification_key, public_inputs, proof);
}

bool link_zkboost_shared_histograms_to_splits(
    const zkboostSharedHistogramPlonkInstance& histogram_instance,
    const zkboostSharedSplitPlonkInstance& split_instance)
{
    const size_t c = split_instance.summary.candidate_count;
    if (histogram_instance.summary.node_count != split_instance.summary.node_count ||
        histogram_instance.summary.feature_count != split_instance.summary.feature_count ||
        histogram_instance.summary.bin_count != split_instance.summary.bin_count ||
        histogram_instance.summary.histogram_size != c) {
        return false;
    }

    const size_t q = histogram_instance.summary.query_row_count;
    const size_t h = histogram_instance.summary.histogram_size;
    const size_t left_statistics_offset = 3 * q + 6 * h;
    const size_t right_statistics_offset = 3 * q + 9 * h;
    const size_t split_left_gradient_offset = 0;
    const size_t split_right_gradient_offset = c;
    const size_t split_left_hessian_offset = 2 * c;
    const size_t split_right_hessian_offset = 3 * c;
    const size_t split_left_count_offset = 13 * c;
    const size_t split_right_count_offset = 14 * c;
    if (histogram_instance.public_inputs.size() < right_statistics_offset + 3 * h ||
        split_instance.public_inputs.size() < 4 * c) {
        return false;
    }

    for (size_t i = 0; i < c; ++i) {
        if (histogram_instance.public_inputs.at(left_statistics_offset + i) !=
            split_instance.public_inputs.at(split_left_count_offset + i)) {
            return false;
        }
        if (histogram_instance.public_inputs.at(right_statistics_offset + i) !=
            split_instance.public_inputs.at(split_right_count_offset + i)) {
            return false;
        }
        if (histogram_instance.public_inputs.at(left_statistics_offset + h + i) !=
            split_instance.public_inputs.at(split_left_gradient_offset + i)) {
            return false;
        }
        if (histogram_instance.public_inputs.at(right_statistics_offset + h + i) !=
            split_instance.public_inputs.at(split_right_gradient_offset + i)) {
            return false;
        }
        if (histogram_instance.public_inputs.at(left_statistics_offset + 2 * h + i) !=
            split_instance.public_inputs.at(split_left_hessian_offset + i)) {
            return false;
        }
        if (histogram_instance.public_inputs.at(right_statistics_offset + 2 * h + i) !=
            split_instance.public_inputs.at(split_right_hessian_offset + i)) {
            return false;
        }
    }
    return true;
}

bool link_zkboost_shared_validate_forest_to_splits(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedSplitPlonkInstance& split_instance)
{
    const size_t n = split_instance.summary.node_count;
    const size_t c = split_instance.summary.candidate_count;
    if (forest_instance.summary.node_count != n ||
        forest_instance.summary.feature_count != split_instance.summary.feature_count) {
        return false;
    }

    const size_t forest_path_count = forest_instance.summary.checked_paths;
    const size_t committed_mask_offset = 2 * forest_path_count;
    const size_t committed_feature_offset = committed_mask_offset + n;
    const size_t committed_threshold_offset = committed_feature_offset + n;
    const size_t split_selected_feature_offset = 16 * c + 2 * n;
    const size_t split_selected_bin_offset = 16 * c + 3 * n;
    const size_t split_mask_offset = 16 * c + 4 * n;
    if (forest_instance.public_inputs.size() < committed_threshold_offset + n ||
        split_instance.public_inputs.size() < split_mask_offset + n) {
        return false;
    }

    const Fr zero(0);
    const Fr one(1);
    for (size_t node = 0; node < n; ++node) {
        const Fr& committed_mask = forest_instance.public_inputs.at(committed_mask_offset + node);
        const Fr& split_mask = split_instance.public_inputs.at(split_mask_offset + node);
        if (committed_mask != split_mask) {
            return false;
        }
        if (split_mask != zero && split_mask != one) {
            return false;
        }
        if (split_mask == zero) {
            continue;
        }
        if (forest_instance.public_inputs.at(committed_feature_offset + node) !=
            split_instance.public_inputs.at(split_selected_feature_offset + node)) {
            return false;
        }
        if (forest_instance.public_inputs.at(committed_threshold_offset + node) !=
            split_instance.public_inputs.at(split_selected_bin_offset + node) + one) {
            return false;
        }
    }
    return true;
}
