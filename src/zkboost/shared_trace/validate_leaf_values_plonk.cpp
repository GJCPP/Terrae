#include "zkboost/shared_trace/validate_leaf_values_plonk.hpp"

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
        throw std::runtime_error("zkboost_shared_leaf_value_plonk: " + message);
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

std::int64_t leaf_value_division_scale(const forestTrainingSemanticTrace& training)
{
    require_condition(training.leaf_value_encoding.fractional_bits >=
                          training.gradient_encoding.fractional_bits,
                      "leaf value fractional bits must be at least gradient fractional bits");
    const size_t scale_power =
        training.leaf_value_encoding.fractional_bits -
        training.gradient_encoding.fractional_bits;
    require_condition(scale_power < static_cast<size_t>(std::numeric_limits<std::int64_t>::digits),
                      "division scale power exceeds int64 range");
    return std::int64_t(1) << scale_power;
}

void validate_view_shapes(const forestTrainingSemanticTrace& training,
                          const zkboostSharedLeafValueView& view)
{
    require_condition(training.node_count == view.summary.node_count, "node_count mismatch");
    require_condition(training.node_leaf_mask.size() == training.node_count,
                      "node_leaf_mask length mismatch");
    require_condition(training.node_leaf_values.size() == training.node_count,
                      "node_leaf_values length mismatch");
    require_condition(view.leaf_gradients.size() == training.node_count,
                      "leaf_gradients length mismatch");
    require_condition(view.leaf_hessians.size() == training.node_count,
                      "leaf_hessians length mismatch");
    require_condition(view.leaf_denominators.size() == training.node_count,
                      "leaf_denominators length mismatch");
    require_condition(view.encoded_negative_leaf_gradients.size() == training.node_count,
                      "encoded_negative_leaf_gradients length mismatch");
    require_condition(view.encoded_leaf_values.size() == training.node_count,
                      "encoded_leaf_values length mismatch");
    require_condition(view.leaf_remainders.size() == training.node_count,
                      "leaf_remainders length mismatch");
    require_condition(view.decoded_leaf_values.size() == training.node_count,
                      "decoded_leaf_values length mismatch");
    require_condition(view.leaf_masks.size() == training.node_count,
                      "leaf_masks length mismatch");
    require_condition(view.committed_leaf_values.size() == training.node_count,
                      "committed_leaf_values length mismatch");
    validate_fixed_point_encoding(training.gradient_encoding,
                                  "zkboost_shared_leaf_value_plonk gradient");
    validate_fixed_point_encoding(training.leaf_value_encoding,
                                  "zkboost_shared_leaf_value_plonk leaf value");
    require_condition(training.gradient_encoding.signed_values,
                      "gradient encoding must be signed");
    require_condition(training.leaf_value_encoding.signed_values,
                      "leaf value encoding must be signed");
    require_condition(training.learning_rate_numerator > 0,
                      "learning_rate_numerator must be non-zero");
    require_condition(training.learning_rate_denominator > 0,
                      "learning_rate_denominator must be non-zero");
}

class sharedLeafValueCircuitCompiler : public zkboostGenericCircuitCompilerBase {
public:
    sharedLeafValueCircuitCompiler(const forestTrainingSemanticTrace& training,
                                   const zkboostSharedLeafValueView& view)
        : training_(training)
        , view_(view)
    {
        validate_view_shapes(training_, view_);
        gradient_offset_ = fixed_point_offset_fr(training_.gradient_encoding);
        leaf_value_offset_ = fixed_point_offset_fr(training_.leaf_value_encoding);
        division_scale_ = leaf_value_division_scale(training_);
        initialize_public_inputs();
    }

    void compile()
    {
        compile_leaf_value_relations();
    }

private:
    void initialize_public_inputs()
    {
        public_leaf_gradients_.reserve(training_.node_count);
        public_leaf_hessians_.reserve(training_.node_count);
        public_leaf_denominators_.reserve(training_.node_count);
        public_decoded_leaf_values_.reserve(training_.node_count);
        public_encoded_leaf_values_.reserve(training_.node_count);
        public_encoded_negative_gradients_.reserve(training_.node_count);
        public_leaf_masks_.reserve(training_.node_count);
        public_committed_leaf_values_.reserve(training_.node_count);
        private_leaf_remainders_.reserve(training_.node_count);

        for (size_t node = 0; node < training_.node_count; ++node) {
            public_leaf_gradients_.push_back(
                public_value(checked_i64_from_fr(view_.leaf_gradients.at(node),
                                                 "leaf value gradients")));
        }
        for (size_t node = 0; node < training_.node_count; ++node) {
            public_leaf_hessians_.push_back(
                public_value(checked_i64_from_fr(view_.leaf_hessians.at(node),
                                                 "leaf value hessians")));
        }
        for (size_t node = 0; node < training_.node_count; ++node) {
            public_leaf_denominators_.push_back(
                public_value(checked_i64_from_fr(view_.leaf_denominators.at(node),
                                                 "leaf value denominators")));
        }
        for (size_t node = 0; node < training_.node_count; ++node) {
            public_decoded_leaf_values_.push_back(
                public_value(checked_i64_from_fr(view_.decoded_leaf_values.at(node),
                                                 "leaf value decoded values")));
        }
        for (size_t node = 0; node < training_.node_count; ++node) {
            public_encoded_leaf_values_.push_back(
                public_value(checked_nonnegative_i64_from_fr(view_.encoded_leaf_values.at(node),
                                                             "leaf value encoded values")));
        }
        for (size_t node = 0; node < training_.node_count; ++node) {
            public_encoded_negative_gradients_.push_back(
                public_value(checked_nonnegative_i64_from_fr(
                    view_.encoded_negative_leaf_gradients.at(node),
                    "leaf value encoded negative gradients")));
        }
        for (size_t node = 0; node < training_.node_count; ++node) {
            public_leaf_masks_.push_back(
                public_value(checked_i64_from_fr(view_.leaf_masks.at(node), "leaf masks")));
        }
        for (size_t node = 0; node < training_.node_count; ++node) {
            public_committed_leaf_values_.push_back(
                public_value(checked_i64_from_fr(view_.committed_leaf_values.at(node),
                                                 "committed leaf values")));
        }
        for (size_t node = 0; node < training_.node_count; ++node) {
            private_leaf_remainders_.push_back(
                private_value(checked_i64_from_fr(view_.leaf_remainders.at(node),
                                                  "leaf value remainders")));
        }
    }

    void compile_leaf_value_relations()
    {
        for (size_t node = 0; node < training_.node_count; ++node) {
            const Fr learning_rate_denominator = field_from_size_t(training_.learning_rate_denominator);
            const std::int64_t leaf_mask = training_.node_leaf_mask.at(node);
            require_condition(leaf_mask == 0 || leaf_mask == 1, "node_leaf_mask must be binary");
            constrain_boolean(public_leaf_masks_.at(node));
            const zkboostAssignedCircuitValue base_denominator =
                affine(public_leaf_hessians_.at(node),
                       learning_rate_denominator,
                       field_from_size_t(training_.l2_regularization) * learning_rate_denominator);
            const zkboostAssignedCircuitValue expected_denominator =
                training_.loss_mode == forestTrainingLoss::FirstOrderRss
                    ? add_scaled(base_denominator,
                                 Fr(1),
                                 one_minus(public_leaf_masks_.at(node)),
                                 learning_rate_denominator,
                                 Fr(0))
                    : base_denominator;
            constrain_equal(public_leaf_denominators_.at(node), expected_denominator);

            const Fr gradient_sign =
                training_.loss_mode == forestTrainingLoss::FirstOrderRss ? Fr(1) : Fr(-1);
            const zkboostAssignedCircuitValue expected_encoded_negative_gradient =
                affine(public_leaf_gradients_.at(node),
                       gradient_sign * field_from_size_t(training_.learning_rate_numerator),
                       gradient_offset_);
            constrain_equal(public_encoded_negative_gradients_.at(node),
                            expected_encoded_negative_gradient);
            constrain_nonnegative_with_limbs(
                public_encoded_negative_gradients_.at(node),
                checked_nonnegative_i64_from_fr(
                    view_.encoded_negative_leaf_gradients.at(node),
                    "leaf value encoded negative gradients"),
                training_.gradient_encoding.total_bits);

            const zkboostAssignedCircuitValue expected_encoded_leaf_value =
                affine(public_decoded_leaf_values_.at(node), Fr(1), leaf_value_offset_);
            constrain_equal(public_encoded_leaf_values_.at(node), expected_encoded_leaf_value);
            constrain_nonnegative_with_limbs(
                public_encoded_leaf_values_.at(node),
                checked_nonnegative_i64_from_fr(view_.encoded_leaf_values.at(node),
                                                "leaf value encoded values"),
                training_.leaf_value_encoding.total_bits);

            const std::int64_t denominator =
                checked_i64_from_fr(view_.leaf_denominators.at(node),
                                    "leaf value denominators");
            const std::int64_t remainder =
                checked_i64_from_fr(view_.leaf_remainders.at(node),
                                    "leaf value remainders");
            require_condition(denominator > 0, "division denominator must be positive");
            require_condition(remainder >= 0 && remainder < denominator,
                              "leaf remainder must be in [0, denominator)");
            require_condition(training_.leaf_denominator_bit_bound > 0 &&
                                  training_.leaf_denominator_bit_bound <= 63,
                              "leaf denominator bit bound must be in [1, 63]");
            const size_t division_range_bits = std::max(
                training_.leaf_denominator_bit_bound,
                training_.gradient_encoding.total_bits);
            constrain_nonnegative_with_limbs(private_leaf_remainders_.at(node),
                                             remainder,
                                             division_range_bits);
            const zkboostAssignedCircuitValue remainder_gap =
                add_scaled(public_leaf_denominators_.at(node),
                           Fr(1),
                           private_leaf_remainders_.at(node),
                           Fr(-1),
                           Fr(-1));
            constrain_nonnegative_with_limbs(remainder_gap,
                                             denominator - remainder - 1,
                                             division_range_bits);

            const zkboostAssignedCircuitValue denominator_times_leaf_value =
                multiply(public_leaf_denominators_.at(node),
                         public_decoded_leaf_values_.at(node));
            const zkboostAssignedCircuitValue quotient_plus_remainder =
                add_scaled(denominator_times_leaf_value,
                           Fr(1),
                           private_leaf_remainders_.at(node),
                           Fr(1),
                           Fr(0));
            const zkboostAssignedCircuitValue scaled_negative_gradient =
                affine(public_leaf_gradients_.at(node),
                       gradient_sign * Fr(division_scale_) *
                           field_from_size_t(training_.learning_rate_numerator),
                       Fr(0));
            constrain_equal(quotient_plus_remainder, scaled_negative_gradient);

            const zkboostAssignedCircuitValue committed_difference =
                add_scaled(public_decoded_leaf_values_.at(node),
                           Fr(1),
                           public_committed_leaf_values_.at(node),
                           Fr(-1),
                           Fr(0));
            constrain_equal(multiply(public_leaf_masks_.at(node), committed_difference),
                            constant(0));
        }
    }

    const forestTrainingSemanticTrace& training_;
    const zkboostSharedLeafValueView& view_;

    Fr gradient_offset_;
    Fr leaf_value_offset_;
    std::int64_t division_scale_ = 1;

    std::vector<zkboostAssignedCircuitValue> public_leaf_gradients_;
    std::vector<zkboostAssignedCircuitValue> public_leaf_hessians_;
    std::vector<zkboostAssignedCircuitValue> public_leaf_denominators_;
    std::vector<zkboostAssignedCircuitValue> public_decoded_leaf_values_;
    std::vector<zkboostAssignedCircuitValue> public_encoded_leaf_values_;
    std::vector<zkboostAssignedCircuitValue> public_encoded_negative_gradients_;
    std::vector<zkboostAssignedCircuitValue> public_leaf_masks_;
    std::vector<zkboostAssignedCircuitValue> public_committed_leaf_values_;
    std::vector<zkboostAssignedCircuitValue> private_leaf_remainders_;
};

zkboostNamedPublicSpans leaf_value_named_public_spans(const zkboostSharedLeafValueView& view)
{
    const size_t n = view.summary.node_count;
    return {
        {"leaf_gradients", {0, n}},
        {"leaf_hessians", {n, n}},
        {"leaf_denominators", {2 * n, n}},
        {"decoded_leaf_values", {3 * n, n}},
        {"encoded_leaf_values", {4 * n, n}},
        {"encoded_negative_leaf_gradients", {5 * n, n}},
        {"leaf_masks", {6 * n, n}},
        {"committed_leaf_values", {7 * n, n}},
    };
}

} // namespace

std::vector<Fr> encode_zkboost_shared_leaf_value_public_inputs(
    const zkboostSharedLeafValueView& view)
{
    std::vector<Fr> out;
    out.reserve(view.summary.node_count * 8);
    out.insert(out.end(), view.leaf_gradients.begin(), view.leaf_gradients.end());
    out.insert(out.end(), view.leaf_hessians.begin(), view.leaf_hessians.end());
    out.insert(out.end(), view.leaf_denominators.begin(), view.leaf_denominators.end());
    out.insert(out.end(), view.decoded_leaf_values.begin(), view.decoded_leaf_values.end());
    out.insert(out.end(), view.encoded_leaf_values.begin(), view.encoded_leaf_values.end());
    out.insert(out.end(),
               view.encoded_negative_leaf_gradients.begin(),
               view.encoded_negative_leaf_gradients.end());
    out.insert(out.end(), view.leaf_masks.begin(), view.leaf_masks.end());
    out.insert(out.end(), view.committed_leaf_values.begin(), view.committed_leaf_values.end());
    return out;
}

zkboostSharedLeafValuePlonkInstance build_zkboost_shared_leaf_value_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const kzgPublicParameters& public_parameters)
{
    return build_zkboost_shared_leaf_value_plonk_instance(
        training,
        build_zkboost_shared_histogram_view(training),
        public_parameters);
}

zkboostSharedLeafValuePlonkInstance build_zkboost_shared_leaf_value_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms,
    const kzgPublicParameters& public_parameters)
{
    return build_zkboost_shared_leaf_value_plonk_instance_from_view(
        training,
        build_zkboost_shared_leaf_value_view(training, histograms),
        public_parameters);
}

zkboostSharedLeafValuePlonkInstance build_zkboost_shared_leaf_value_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedLeafValueView& view,
    const kzgPublicParameters& public_parameters)
{
    sharedLeafValueCircuitCompiler compiler(training, view);
    compiler.compile();
    const zkboostCompiledCircuitStatement statement =
        compile_zkboost_generic_circuit_statement("shared_leaf_values",
                                                  compiler.build_circuit(),
                                                  public_parameters,
                                                  leaf_value_named_public_spans(view));

    zkboostSharedLeafValuePlonkInstance out;
    out.setup = statement.setup;
    out.witness = statement.witness;
    out.public_inputs = statement.public_inputs;
    out.summary = view.summary;
    return out;
}

size_t required_zkboost_shared_leaf_value_plonk_srs_size(
    const forestTrainingSemanticTrace& training)
{
    return required_zkboost_shared_leaf_value_plonk_srs_size(
        training,
        build_zkboost_shared_histogram_view(training));
}

size_t required_zkboost_shared_leaf_value_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms)
{
    return required_zkboost_shared_leaf_value_plonk_srs_size_from_view(
        training,
        build_zkboost_shared_leaf_value_view(training, histograms));
}

size_t required_zkboost_shared_leaf_value_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedLeafValueView& view)
{
    sharedLeafValueCircuitCompiler compiler(training, view);
    compiler.compile();
    return required_zkboost_generic_circuit_plonk_srs_size(compiler.build_circuit());
}

plonkProof prove_zkboost_shared_leaf_value_plonk(
    const zkboostSharedLeafValuePlonkInstance& instance)
{
    return prove_plonk(instance.setup.proving_key, instance.witness, instance.public_inputs);
}

bool verify_zkboost_shared_leaf_value_plonk(
    const zkboostSharedLeafValuePlonkInstance& instance,
    const plonkProof& proof)
{
    return verify_zkboost_shared_leaf_value_plonk(
        instance.setup.verification_key,
        instance.public_inputs,
        proof);
}

bool verify_zkboost_shared_leaf_value_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof)
{
    return verify_plonk(verification_key, public_inputs, proof);
}

bool link_zkboost_shared_histograms_to_leaf_values(
    const zkboostSharedHistogramPlonkInstance& histogram_instance,
    const zkboostSharedLeafValuePlonkInstance& leaf_value_instance)
{
    const size_t node_count = leaf_value_instance.summary.node_count;
    if (histogram_instance.summary.node_count != node_count) {
        return false;
    }
    if (histogram_instance.summary.histogram_width != leaf_value_instance.summary.histogram_width) {
        return false;
    }
    if (histogram_instance.summary.bin_count == 0) {
        return false;
    }

    const size_t q = histogram_instance.summary.query_row_count;
    const size_t h = histogram_instance.summary.histogram_size;
    const size_t left_statistics_offset = 3 * q + 6 * h;
    const size_t gradient_offset = 0;
    const size_t hessian_offset = node_count;
    if (histogram_instance.public_inputs.size() < left_statistics_offset + 3 * h ||
        leaf_value_instance.public_inputs.size() < 2 * node_count) {
        return false;
    }

    for (size_t node = 0; node < node_count; ++node) {
        const size_t histogram_index =
            node * histogram_instance.summary.histogram_width +
            (histogram_instance.summary.bin_count - 1);
        const size_t histogram_gradient =
            left_statistics_offset + h + histogram_index;
        const size_t histogram_hessian =
            left_statistics_offset + 2 * h + histogram_index;
        if (histogram_instance.public_inputs.at(histogram_gradient) !=
            leaf_value_instance.public_inputs.at(gradient_offset + node)) {
            return false;
        }
        if (histogram_instance.public_inputs.at(histogram_hessian) !=
            leaf_value_instance.public_inputs.at(hessian_offset + node)) {
            return false;
        }
    }
    return true;
}

bool link_zkboost_shared_validate_forest_to_leaf_values(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedLeafValuePlonkInstance& leaf_value_instance)
{
    const size_t count = forest_instance.summary.checked_paths;
    const size_t node_count = leaf_value_instance.summary.node_count;
    const size_t forest_tree_score_offset = count;
    const size_t leaf_decoded_offset = 3 * node_count;
    if (forest_instance.public_inputs.size() < forest_tree_score_offset + count ||
        leaf_value_instance.public_inputs.size() < leaf_decoded_offset + node_count) {
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        const size_t leaf = canonical_size_t_from_fr(forest_instance.public_inputs.at(i),
                                                     "shared forest leaf index");
        if (leaf >= node_count) {
            return false;
        }
        if (forest_instance.public_inputs.at(forest_tree_score_offset + i) !=
            leaf_value_instance.public_inputs.at(leaf_decoded_offset + leaf)) {
            return false;
        }
    }
    return true;
}
