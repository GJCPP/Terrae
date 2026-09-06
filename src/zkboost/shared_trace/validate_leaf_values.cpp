#include "zkboost/shared_trace/validate_leaf_values.hpp"

#include "algebra_utils.hpp"
#include "fixed_point.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using boost::multiprecision::cpp_int;

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_validate_leaf_values: " + message);
    }
}

size_t checked_size(const Fr& value, const std::string& name)
{
    return canonical_size_t_from_fr(value, name.c_str());
}

std::int64_t checked_i64(const Fr& value, const std::string& name)
{
    return canonical_signed_int64_from_fr(value, name.c_str());
}

size_t checked_size_product(size_t left, size_t right, const std::string& name)
{
    require_condition(left == 0 || right <= std::numeric_limits<size_t>::max() / left,
                      name + " exceeds size_t range");
    return left * right;
}

std::int64_t checked_scaled_product(std::int64_t value,
                                    size_t multiplier,
                                    bool negate,
                                    const std::string& name)
{
    const cpp_int product = (negate ? -cpp_int(value) : cpp_int(value)) * multiplier;
    require_condition(product >= cpp_int(std::numeric_limits<std::int64_t>::min()) &&
                          product <= cpp_int(std::numeric_limits<std::int64_t>::max()),
                      name + " exceeds int64 range");
    return product.convert_to<std::int64_t>();
}

Fr fr_from_i64(std::int64_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

Fr encode_fixed_point_checked(std::int64_t scaled_value,
                              const fixedPointEncoding& encoding,
                              const std::string& context)
{
    try {
        return fixed_point_fr_from_size_t(encode_fixed_point_scaled_integer(scaled_value, encoding));
    } catch (const std::exception& e) {
        throw std::runtime_error(context + ": " + e.what());
    }
}

std::pair<std::int64_t, std::int64_t> floor_division(std::int64_t numerator,
                                                     std::int64_t denominator)
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

std::pair<std::int64_t, std::int64_t> quantized_fixed_point_division(
    std::int64_t numerator,
    size_t denominator,
    const fixedPointEncoding& numerator_encoding,
    const fixedPointEncoding& quotient_encoding)
{
    require_condition(quotient_encoding.fractional_bits >= numerator_encoding.fractional_bits,
                      "leaf value fractional bits must be at least gradient fractional bits");
    const size_t scale_power =
        quotient_encoding.fractional_bits - numerator_encoding.fractional_bits;
    require_condition(scale_power < static_cast<size_t>(std::numeric_limits<std::int64_t>::digits),
                      "division scale power exceeds int64 range");
    const std::int64_t scale = std::int64_t(1) << scale_power;
    return floor_division(numerator * scale, static_cast<std::int64_t>(denominator));
}

void validate_shapes(const forestTrainingSemanticTrace& training,
                     const zkboostSharedHistogramView& histograms)
{
    require_condition(training.node_count > 0, "node_count must be non-zero");
    require_condition(training.feature_count > 0, "feature_count must be non-zero");
    require_condition(training.bin_count > 0, "bin_count must be non-zero");
    require_condition(training.node_leaf_mask.size() == training.node_count,
                      "node_leaf_mask length mismatch");
    require_condition(training.node_leaf_values.size() == training.node_count,
                      "node_leaf_values length mismatch");
    require_condition(histograms.summary.node_count == training.node_count,
                      "histogram node_count mismatch");
    require_condition(histograms.summary.feature_count == training.feature_count,
                      "histogram feature_count mismatch");
    require_condition(histograms.summary.bin_count == training.bin_count,
                      "histogram bin_count mismatch");
    require_condition(histograms.summary.histogram_width == training.feature_count * training.bin_count,
                      "histogram width mismatch");
    require_condition(histograms.left_statistics.size() == 3,
                      "left_statistics channel mismatch");
    require_condition(histograms.left_statistics[1].size() == histograms.summary.histogram_size,
                      "gradient left_statistics length mismatch");
    require_condition(histograms.left_statistics[2].size() == histograms.summary.histogram_size,
                      "hessian left_statistics length mismatch");
    validate_fixed_point_encoding(training.gradient_encoding,
                                  "zkboost_shared_validate_leaf_values gradient");
    validate_fixed_point_encoding(training.leaf_value_encoding,
                                  "zkboost_shared_validate_leaf_values leaf value");
    require_condition(training.gradient_encoding.signed_values,
                      "gradient encoding must be signed");
    require_condition(training.leaf_value_encoding.signed_values,
                      "leaf value encoding must be signed");
    require_condition(training.learning_rate_numerator > 0,
                      "learning_rate_numerator must be non-zero");
    require_condition(training.learning_rate_denominator > 0,
                      "learning_rate_denominator must be non-zero");
    if (training.loss_mode == forestTrainingLoss::FirstOrderRss) {
        require_condition(training.l2_regularization == 0,
                          "first-order RSS requires zero L2 regularization");
    } else {
        require_condition(training.l2_regularization > 0,
                          "second-order XGBoost requires non-zero L2 regularization");
    }
}

Fr node_total_from_left_statistics(const zkboostSharedHistogramView& histograms,
                                   size_t channel,
                                   size_t node)
{
    // Feature 0 is enough because histogram channels are replicated across features.
    const size_t index = node * histograms.summary.histogram_width +
                         (histograms.summary.bin_count - 1);
    return histograms.left_statistics.at(channel).at(index);
}

} // namespace

zkboostSharedLeafValueView build_zkboost_shared_leaf_value_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms)
{
    validate_shapes(training, histograms);

    zkboostSharedLeafValueView view;
    view.summary.node_count = training.node_count;
    view.summary.histogram_width = histograms.summary.histogram_width;
    view.summary.l2_regularization = training.l2_regularization;
    view.summary.learning_rate_numerator = training.learning_rate_numerator;
    view.summary.learning_rate_denominator = training.learning_rate_denominator;

    view.leaf_gradients.assign(training.node_count, Fr(0));
    view.leaf_hessians.assign(training.node_count, Fr(0));
    view.leaf_denominators.assign(training.node_count, Fr(0));
    view.encoded_negative_leaf_gradients.assign(training.node_count, Fr(0));
    view.encoded_leaf_values.assign(training.node_count, Fr(0));
    view.leaf_remainders.assign(training.node_count, Fr(0));
    view.decoded_leaf_values.assign(training.node_count, Fr(0));
    view.leaf_masks.reserve(training.node_count);
    view.committed_leaf_values.reserve(training.node_count);
    for (size_t node = 0; node < training.node_count; ++node) {
        view.leaf_masks.push_back(fr_from_i64(training.node_leaf_mask.at(node)));
        view.committed_leaf_values.push_back(fr_from_i64(training.node_leaf_values.at(node)));
    }

    for (size_t node = 0; node < training.node_count; ++node) {
        const std::int64_t is_leaf_i64 = training.node_leaf_mask.at(node);
        require_condition(is_leaf_i64 == 0 || is_leaf_i64 == 1, "node_leaf_mask must be binary");
        const size_t is_leaf = static_cast<size_t>(is_leaf_i64);
        view.summary.leaf_node_count += is_leaf;

        view.leaf_gradients.at(node) = node_total_from_left_statistics(histograms, 1, node);
        view.leaf_hessians.at(node) = node_total_from_left_statistics(histograms, 2, node);

        const size_t hessian = checked_size(view.leaf_hessians.at(node), "leaf_hessians");
        const size_t inactive_node =
            training.loss_mode == forestTrainingLoss::FirstOrderRss && is_leaf == 0 ? 1 : 0;
        const size_t denominator = checked_size_product(
            hessian + training.l2_regularization + inactive_node,
            training.learning_rate_denominator,
            "scaled leaf denominator");
        view.leaf_denominators.at(node) = field_from_size_t(denominator);

        const std::int64_t gradient = checked_i64(view.leaf_gradients.at(node), "leaf_gradients");
        const std::int64_t negative_gradient = checked_scaled_product(
            gradient,
            training.learning_rate_numerator,
            training.loss_mode == forestTrainingLoss::SecondOrderXgboost,
            "scaled leaf gradient");
        view.encoded_negative_leaf_gradients.at(node) =
            encode_fixed_point_checked(negative_gradient,
                                       training.gradient_encoding,
                                       "encoded_negative_leaf_gradients");

        const auto [leaf_value, remainder] =
            quantized_fixed_point_division(negative_gradient,
                                           denominator,
                                           training.gradient_encoding,
                                           training.leaf_value_encoding);
        view.decoded_leaf_values.at(node) = fr_from_i64(leaf_value);
        view.encoded_leaf_values.at(node) =
            encode_fixed_point_checked(leaf_value,
                                       training.leaf_value_encoding,
                                       "encoded_leaf_values");
        view.leaf_remainders.at(node) = field_from_size_t(static_cast<size_t>(remainder));

        if (is_leaf != 0) {
            require_condition(training.node_leaf_values.at(node) == leaf_value,
                              "committed leaf value does not match derived leaf value");
        }
    }

    return view;
}

zkboostSharedLeafValueView build_zkboost_shared_leaf_value_view(
    const forestTrainingSemanticTrace& training)
{
    return build_zkboost_shared_leaf_value_view(
        training,
        build_zkboost_shared_histogram_view(training));
}

zkboostSharedLeafValueSummary validate_zkboost_shared_leaf_values(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms)
{
    return build_zkboost_shared_leaf_value_view(training, histograms).summary;
}

zkboostSharedLeafValueSummary validate_zkboost_shared_leaf_values(
    const forestTrainingSemanticTrace& training)
{
    return build_zkboost_shared_leaf_value_view(training).summary;
}
