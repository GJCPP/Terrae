#include "zkboost/shared_trace/validate_splits.hpp"

#include "algebra_utils.hpp"
#include "fixed_point.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_validate_splits: " + message);
    }
}

size_t checked_size(std::int64_t value, const std::string& name)
{
    require_condition(value >= 0, name + " must be non-negative");
    return static_cast<size_t>(value);
}

size_t checked_size_from_fr(const Fr& value, const std::string& name)
{
    return canonical_size_t_from_fr(value, name.c_str());
}

std::int64_t checked_i64_from_fr(const Fr& value, const std::string& name)
{
    return canonical_signed_int64_from_fr(value, name.c_str());
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
                      "quotient fractional bits must be at least numerator fractional bits");
    const size_t scale_power =
        quotient_encoding.fractional_bits - numerator_encoding.fractional_bits;
    require_condition(scale_power < static_cast<size_t>(std::numeric_limits<std::int64_t>::digits),
                      "division scale power exceeds int64 range");
    const std::int64_t scale = std::int64_t(1) << scale_power;
    return floor_division(numerator * scale, static_cast<std::int64_t>(denominator));
}

size_t candidate_offset(size_t feature_count,
                        size_t bin_count,
                        size_t node,
                        size_t feature,
                        size_t bin)
{
    return node * feature_count * bin_count + feature * bin_count + bin;
}

size_t node_feature_offset(const forestTrainingSemanticTrace& training, size_t node, size_t feature)
{
    return node * training.feature_count + feature;
}

void validate_shapes(const forestTrainingSemanticTrace& training,
                     const zkboostSharedHistogramView& histograms)
{
    require_condition(training.node_count > 0, "node_count must be non-zero");
    require_condition(training.feature_count > 0, "feature_count must be non-zero");
    require_condition(training.bin_count > 0, "bin_count must be non-zero");
    require_condition(histograms.summary.node_count == training.node_count,
                      "histogram node_count mismatch");
    require_condition(histograms.summary.feature_count == training.feature_count,
                      "histogram feature_count mismatch");
    require_condition(histograms.summary.bin_count == training.bin_count,
                      "histogram bin_count mismatch");
    require_condition(histograms.summary.histogram_width == training.feature_count * training.bin_count,
                      "histogram_width mismatch");
    require_condition(histograms.left_statistics.size() == 3, "left_statistics channel mismatch");
    require_condition(histograms.right_statistics.size() == 3, "right_statistics channel mismatch");
    require_condition(histograms.left_statistics[1].size() == histograms.summary.histogram_size,
                      "left gradient statistics length mismatch");
    require_condition(histograms.left_statistics[2].size() == histograms.summary.histogram_size,
                      "left hessian statistics length mismatch");
    require_condition(histograms.right_statistics[1].size() == histograms.summary.histogram_size,
                      "right gradient statistics length mismatch");
    require_condition(histograms.right_statistics[2].size() == histograms.summary.histogram_size,
                      "right hessian statistics length mismatch");
    require_condition(training.split_node_mask.size() == training.node_count,
                      "split_node_mask length mismatch");
    require_condition(training.feature_selector.size() == training.node_count * training.feature_count,
                      "feature_selector length mismatch");
    require_condition(training.thresholds.size() == training.node_count,
                      "thresholds length mismatch");
    require_condition(training.candidate_mask.size() == training.node_count * histograms.summary.histogram_width,
                      "candidate_mask length mismatch");
    require_condition(training.selected_candidate_indices.size() == training.node_count,
                      "selected_candidate_indices length mismatch");
    require_condition(training.root_indices.size() == training.tree_count,
                      "root_indices length mismatch");
    require_condition(training.left_child_indices.size() == training.node_count,
                      "left_child_indices length mismatch");
    require_condition(training.right_child_indices.size() == training.node_count,
                      "right_child_indices length mismatch");
    require_condition(training.node_tree_ids.size() == training.node_count,
                      "node_tree_ids length mismatch");
    require_condition(training.node_leaf_mask.size() == training.node_count,
                      "node_leaf_mask length mismatch");
    require_condition(training.max_depth > 0, "max_depth must be positive");
    validate_fixed_point_encoding(training.gain_numerator_encoding,
                                  "zkboost_shared_validate_splits gain numerator");
    validate_fixed_point_encoding(training.quantized_gain_encoding,
                                  "zkboost_shared_validate_splits quantized gain");
    require_condition(training.gain_numerator_encoding.signed_values,
                      "gain numerator encoding must be signed");
    require_condition(training.quantized_gain_encoding.signed_values,
                      "quantized gain encoding must be signed");
    if (training.loss_mode == forestTrainingLoss::FirstOrderRss) {
        require_condition(training.l2_regularization == 0 && training.split_penalty == Fr(0),
                          "first-order RSS requires zero L2 regularization and split penalty");
    } else {
        require_condition(training.l2_regularization > 0,
                          "second-order XGBoost requires non-zero L2 regularization");
    }
}

std::vector<size_t> derive_one_based_node_depths(const forestTrainingSemanticTrace& training)
{
    std::vector<size_t> depths(training.node_count, 0);
    std::vector<bool> visiting(training.node_count, false);
    std::function<void(size_t, size_t, size_t)> visit =
        [&](size_t tree, size_t node, size_t depth) {
            require_condition(node < training.node_count, "forest child index out of range");
            require_condition(checked_size(training.node_tree_ids.at(node), "node_tree_ids") == tree + 1,
                              "reachable node belongs to another tree");
            require_condition(!visiting.at(node), "forest contains a cycle");
            if (depths.at(node) != 0) {
                require_condition(depths.at(node) == depth,
                                  "node is reachable at inconsistent depths");
                return;
            }

            visiting.at(node) = true;
            depths.at(node) = depth;
            const std::int64_t split = training.split_node_mask.at(node);
            const std::int64_t leaf = training.node_leaf_mask.at(node);
            require_condition(split == 0 || split == 1, "split_node_mask must be binary");
            require_condition(leaf == 0 || leaf == 1, "node_leaf_mask must be binary");
            require_condition(split + leaf == 1,
                              "reachable node must be exactly split or leaf");
            if (split != 0) {
                visit(tree,
                      checked_size(training.left_child_indices.at(node), "left_child_indices"),
                      depth + 1);
                visit(tree,
                      checked_size(training.right_child_indices.at(node), "right_child_indices"),
                      depth + 1);
            }
            visiting.at(node) = false;
        };

    for (size_t tree = 0; tree < training.tree_count; ++tree) {
        visit(tree,
              checked_size(training.root_indices.at(tree), "root_indices"),
              1);
    }
    return depths;
}

size_t selected_feature_for_split_node(const forestTrainingSemanticTrace& training, size_t node)
{
    size_t selected = training.feature_count;
    for (size_t feature = 0; feature < training.feature_count; ++feature) {
        const std::int64_t value =
            training.feature_selector.at(node_feature_offset(training, node, feature));
        require_condition(value == 0 || value == 1, "feature_selector must be binary");
        if (value == 0) {
            continue;
        }
        require_condition(selected == training.feature_count,
                          "split node has multiple selected features");
        selected = feature;
    }
    require_condition(selected < training.feature_count, "split node has no selected feature");
    return selected;
}

void validate_inactive_feature_selector(const forestTrainingSemanticTrace& training, size_t node)
{
    for (size_t feature = 0; feature < training.feature_count; ++feature) {
        const std::int64_t value =
            training.feature_selector.at(node_feature_offset(training, node, feature));
        require_condition(value == 0 || value == 1, "feature_selector must be binary");
        require_condition(value == 0, "inactive node must not select a split feature");
    }
}

bool candidate_is_better(const Fr& candidate_score,
                         size_t candidate_column,
                         const Fr& best_score,
                         size_t best_column)
{
    const std::int64_t candidate_i64 =
        checked_i64_from_fr(candidate_score, "candidate tie_break_scores");
    const std::int64_t best_i64 =
        checked_i64_from_fr(best_score, "best tie_break_scores");
    if (candidate_i64 != best_i64) {
        return candidate_i64 > best_i64;
    }
    return candidate_column < best_column;
}

} // namespace

zkboostSharedSplitView build_zkboost_shared_split_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms)
{
    validate_shapes(training, histograms);

    zkboostSharedSplitView view;
    view.summary.node_count = training.node_count;
    view.summary.feature_count = training.feature_count;
    view.summary.bin_count = training.bin_count;
    view.summary.candidate_count = training.node_count * histograms.summary.histogram_width;

    view.left_counts.assign(view.summary.candidate_count, Fr(0));
    view.right_counts.assign(view.summary.candidate_count, Fr(0));
    view.candidate_masks.assign(view.summary.candidate_count, Fr(0));
    view.left_gradients.assign(view.summary.candidate_count, Fr(0));
    view.right_gradients.assign(view.summary.candidate_count, Fr(0));
    view.left_hessians.assign(view.summary.candidate_count, Fr(0));
    view.right_hessians.assign(view.summary.candidate_count, Fr(0));
    view.parent_gradients.assign(view.summary.candidate_count, Fr(0));
    view.parent_hessians.assign(view.summary.candidate_count, Fr(0));
    view.left_gradient_squares.assign(view.summary.candidate_count, Fr(0));
    view.right_gradient_squares.assign(view.summary.candidate_count, Fr(0));
    view.parent_gradient_squares.assign(view.summary.candidate_count, Fr(0));
    view.gain_numerators.assign(view.summary.candidate_count, Fr(0));
    view.gain_denominators.assign(view.summary.candidate_count, Fr(0));
    view.encoded_gain_numerators.assign(view.summary.candidate_count, Fr(0));
    view.quantized_gains.assign(view.summary.candidate_count, Fr(0));
    view.encoded_quantized_gains.assign(view.summary.candidate_count, Fr(0));
    view.gain_remainders.assign(view.summary.candidate_count, Fr(0));
    view.tie_break_scores.assign(view.summary.candidate_count, Fr(0));
    view.selector.assign(view.summary.candidate_count, Fr(0));
    view.selected_gains.assign(training.node_count, Fr(0));
    view.selected_tie_break_scores.assign(training.node_count, Fr(0));
    view.selected_features.assign(training.node_count, Fr(0));
    view.selected_bins.assign(training.node_count, Fr(0));
    view.split_masks.assign(training.node_count, Fr(0));
    view.active_node_masks.assign(training.node_count, Fr(0));
    view.leaf_masks.assign(training.node_count, Fr(0));
    view.node_depths.assign(training.node_count, Fr(0));
    view.depth_limit_masks.assign(training.node_count, Fr(0));

    const Fr lambda = field_from_size_t(training.l2_regularization);
    require_condition(training.softmax_encoding.fractional_bits < std::numeric_limits<size_t>::digits,
                      "softmax fractional bits exceed size_t scale range");
    const Fr first_order_hessian_scale =
        field_from_size_t(size_t(1) << training.softmax_encoding.fractional_bits);
    const size_t column_count = training.feature_count * training.bin_count;
    const std::vector<size_t> node_depths = derive_one_based_node_depths(training);
    const std::int64_t split_gain_threshold =
        checked_i64_from_fr(training.split_gain_threshold, "split_gain_threshold");
    for (size_t node = 0; node < training.node_count; ++node) {
        const std::int64_t split = training.split_node_mask.at(node);
        const std::int64_t leaf = training.node_leaf_mask.at(node);
        require_condition(split == 0 || split == 1, "split_node_mask must be binary");
        require_condition(leaf == 0 || leaf == 1, "node_leaf_mask must be binary");
        require_condition(split + leaf <= 1, "node cannot be both split and leaf");
        view.split_masks.at(node) = field_from_size_t(static_cast<size_t>(split));
        view.leaf_masks.at(node) = field_from_size_t(static_cast<size_t>(leaf));
        view.node_depths.at(node) = field_from_size_t(node_depths.at(node));
        if (split != 0) {
            ++view.summary.split_node_count;
        } else {
            validate_inactive_feature_selector(training, node);
        }

        size_t best_column = column_count;
        Fr best_score(0);
        bool has_candidate = false;

        for (size_t feature = 0; feature < training.feature_count; ++feature) {
            for (size_t bin = 0; bin < training.bin_count; ++bin) {
                const size_t index =
                    candidate_offset(training.feature_count, training.bin_count, node, feature, bin);
                view.left_counts.at(index) = histograms.left_statistics.at(0).at(index);
                view.right_counts.at(index) = histograms.right_statistics.at(0).at(index);
                const std::int64_t candidate_mask = training.candidate_mask.at(index);
                require_condition(candidate_mask == 0 || candidate_mask == 1,
                                  "candidate_mask must be binary");
                const bool expected_candidate =
                    bin + 1 < training.bin_count &&
                    view.left_counts.at(index) != Fr(0) &&
                    view.right_counts.at(index) != Fr(0);
                require_condition(candidate_mask == (expected_candidate ? 1 : 0),
                                  "candidate_mask does not match threshold/count eligibility");
                view.candidate_masks.at(index) = field_from_size_t(static_cast<size_t>(candidate_mask));
                view.left_gradients.at(index) = histograms.left_statistics.at(1).at(index);
                view.right_gradients.at(index) = histograms.right_statistics.at(1).at(index);
                view.left_hessians.at(index) = histograms.left_statistics.at(2).at(index);
                view.right_hessians.at(index) = histograms.right_statistics.at(2).at(index);
                view.parent_gradients.at(index) =
                    view.left_gradients.at(index) + view.right_gradients.at(index);
                view.parent_hessians.at(index) =
                    view.left_hessians.at(index) + view.right_hessians.at(index);

                view.left_gradient_squares.at(index) =
                    view.left_gradients.at(index) * view.left_gradients.at(index);
                view.right_gradient_squares.at(index) =
                    view.right_gradients.at(index) * view.right_gradients.at(index);
                view.parent_gradient_squares.at(index) =
                    view.parent_gradients.at(index) * view.parent_gradients.at(index);

                const Fr inactive_candidate =
                    training.loss_mode == forestTrainingLoss::FirstOrderRss &&
                            training.candidate_mask.at(index) == 0
                        ? Fr(1)
                        : Fr(0);
                Fr left_den = view.left_hessians.at(index) + lambda;
                Fr right_den = view.right_hessians.at(index) + lambda;
                Fr parent_den = view.parent_hessians.at(index) + lambda;
                Fr denominator_scale(1);
                if (training.loss_mode == forestTrainingLoss::FirstOrderRss) {
                    left_den = view.left_counts.at(index) + inactive_candidate;
                    right_den = view.right_counts.at(index) + inactive_candidate;
                    parent_den = view.left_counts.at(index) + view.right_counts.at(index) +
                                 inactive_candidate;
                    denominator_scale = first_order_hessian_scale;
                }
                checked_size_from_fr(left_den, "left gain denominator");
                checked_size_from_fr(right_den, "right gain denominator");
                checked_size_from_fr(parent_den, "parent gain denominator");
                view.gain_denominators.at(index) =
                    left_den * right_den * parent_den * denominator_scale;
                view.gain_numerators.at(index) =
                    view.left_gradient_squares.at(index) * right_den * parent_den +
                    view.right_gradient_squares.at(index) * left_den * parent_den -
                    view.parent_gradient_squares.at(index) * left_den * right_den -
                    training.split_penalty * view.gain_denominators.at(index);

                const std::int64_t numerator =
                    checked_i64_from_fr(view.gain_numerators.at(index), "gain_numerators");
                const size_t denominator =
                    checked_size_from_fr(view.gain_denominators.at(index), "gain_denominators");
                require_condition(denominator > 0, "gain denominator must be positive");
                view.encoded_gain_numerators.at(index) =
                    encode_fixed_point_checked(numerator,
                                               training.gain_numerator_encoding,
                                               "encoded_gain_numerators");

                const auto [quantized_gain, remainder] =
                    quantized_fixed_point_division(numerator,
                                                   denominator,
                                                   training.gain_numerator_encoding,
                                                   training.quantized_gain_encoding);
                view.quantized_gains.at(index) = fr_from_i64(quantized_gain);
                view.encoded_quantized_gains.at(index) =
                    encode_fixed_point_checked(quantized_gain,
                                               training.quantized_gain_encoding,
                                               "encoded_quantized_gains");
                view.gain_remainders.at(index) =
                    field_from_size_t(static_cast<size_t>(remainder));
                const size_t column = feature * training.bin_count + bin;
                view.tie_break_scores.at(index) =
                    view.quantized_gains.at(index) * field_from_size_t(column_count) +
                    field_from_size_t(column_count - 1 - column);

                if (candidate_mask != 0) {
                    if (!has_candidate ||
                        candidate_is_better(view.tie_break_scores.at(index), column, best_score, best_column)) {
                        best_score = view.tie_break_scores.at(index);
                        best_column = column;
                        has_candidate = true;
                    }
                }
            }
        }

        view.active_node_masks.at(node) = has_candidate ? Fr(1) : Fr(0);
        require_condition(split == 0 || has_candidate,
                          "active split node has no candidate");
        if (!has_candidate) {
            continue;
        }

        const size_t selected =
            checked_size(training.selected_candidate_indices.at(node), "selected_candidate_indices");
        require_condition(selected < column_count, "selected_candidate_indices entry out of range");
        require_condition(selected == best_column,
                          "selected candidate is not the masked argmax winner");

        const size_t selector_index = node * column_count + selected;
        require_condition(training.candidate_mask.at(selector_index) == 1,
                          "selected candidate is not enabled by candidate_mask");
        view.selector.at(selector_index) = Fr(1);
        view.selected_gains.at(node) = view.quantized_gains.at(selector_index);
        view.selected_tie_break_scores.at(node) = view.tie_break_scores.at(selector_index);
        view.selected_features.at(node) = field_from_size_t(selected / training.bin_count);
        view.selected_bins.at(node) = field_from_size_t(selected % training.bin_count);

        const std::int64_t selected_gain =
            checked_i64_from_fr(view.selected_gains.at(node), "selected_gains");
        require_condition(node_depths.at(node) > 0,
                          "candidate row must belong to a reachable node");
        require_condition(node_depths.at(node) <= training.max_depth,
                          "active node exceeds max_depth");
        if (split == 0) {
            require_condition(leaf == 1,
                              "candidate-bearing non-split node must be a leaf");
            const bool at_depth_limit = node_depths.at(node) >= training.max_depth;
            view.depth_limit_masks.at(node) = at_depth_limit ? Fr(1) : Fr(0);
            require_condition(at_depth_limit || selected_gain < split_gain_threshold,
                              "leaf with an eligible candidate violates the stopping rule");
            continue;
        }

        require_condition(leaf == 0, "split node cannot be a leaf");
        require_condition(node_depths.at(node) < training.max_depth,
                          "split node cannot be at max_depth");
        require_condition(selected_gain >= split_gain_threshold,
                          "split gain is below split_gain_threshold");
        const size_t selected_feature = selected_feature_for_split_node(training, node);
        require_condition(selected_feature == selected / training.bin_count,
                          "feature_selector does not match selected candidate");
        require_condition(training.thresholds.at(node) >= 0, "split threshold must be non-negative");
        require_condition(static_cast<size_t>(training.thresholds.at(node)) == selected % training.bin_count + 1,
                          "threshold does not match selected candidate bin");
    }

    return view;
}

zkboostSharedSplitView build_zkboost_shared_split_view(
    const forestTrainingSemanticTrace& training)
{
    return build_zkboost_shared_split_view(training, build_zkboost_shared_histogram_view(training));
}

zkboostSharedSplitSummary validate_zkboost_shared_splits(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& histograms)
{
    return build_zkboost_shared_split_view(training, histograms).summary;
}

zkboostSharedSplitSummary validate_zkboost_shared_splits(
    const forestTrainingSemanticTrace& training)
{
    return build_zkboost_shared_split_view(training).summary;
}
