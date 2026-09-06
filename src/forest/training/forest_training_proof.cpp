#include "forest_training_proof.hpp"

#include "lagrange_utils.hpp"
#include "matrix_utils.hpp"

#include <limits>
#include <stdexcept>

namespace {

virCommitment first_feature_mask_expression(size_t node_count, size_t feature_count, size_t bin_count)
{
    const size_t domain_size = node_count * feature_count * bin_count;
    std::vector<size_t> positions;
    positions.reserve(node_count * bin_count);
    for (size_t row = 0; row < node_count; ++row) {
        for (size_t bin = 0; bin < bin_count; ++bin) {
            positions.push_back((row * feature_count * bin_count) + bin);
        }
    }
    return subgroup_lagrange_selector_expression(domain_size, positions);
}

void validate_forest_training_proof_parameters(const forestTrainingProofWitness& witness,
                                               size_t tree_count,
                                               size_t node_count,
                                               size_t feature_count,
                                               size_t bin_count,
                                               size_t query_count,
                                               size_t input_bit_bound,
                                               size_t full_upper_bound,
                                               size_t structure_bit_bound,
                                               size_t hessian_bit_bound,
                                               size_t gain_denominator_bit_bound,
                                               size_t leaf_denominator_bit_bound,
                                               size_t gain_gap_bit_bound)
{
    if (!isPowerOfTwo(tree_count) || !isPowerOfTwo(node_count) || !isPowerOfTwo(feature_count) || !isPowerOfTwo(bin_count) ||
        !isPowerOfTwo(query_count) || tree_count == 0 || node_count == 0 || feature_count == 0 || bin_count == 0 ||
        query_count == 0) {
        throw std::runtime_error(
            "forestTrainingProofProtocol: tree/node/feature/bin/query counts must be non-zero powers of two");
    }
    if (input_bit_bound == 0 || full_upper_bound == 0 || structure_bit_bound == 0 || hessian_bit_bound == 0 ||
        gain_denominator_bit_bound == 0 || leaf_denominator_bit_bound == 0 || gain_gap_bit_bound == 0) {
        throw std::runtime_error("forestTrainingProofProtocol: bit bounds and full_upper_bound must be non-zero");
    }
    if (witness.histogram_refresh.channels.query_values.size() != 3 ||
        witness.histogram_refresh.channels.lifted_values.size() != 3 ||
        witness.histogram_refresh.channels.leaf_histograms.size() != 3 ||
        witness.histogram_refresh.channels.full_histograms.size() != 3 ||
        witness.split_statistics.histograms.size() != 3 || witness.split_statistics.left_statistics.size() != 3 ||
        witness.split_statistics.right_statistics.size() != 3) {
        throw std::runtime_error("forestTrainingProofProtocol: expected exactly three histogram/statistic channels");
    }
}

} // namespace

void forestTrainingProofProtocol::add_check(const forestTrainingProofWitness& witness,
                                            size_t tree_count,
                                            size_t node_count,
                                            size_t feature_count,
                                            size_t bin_count,
                                            size_t iteration_count,
                                            size_t class_count,
                                            size_t query_count,
                                            size_t input_bit_bound,
                                            size_t full_upper_bound,
                                            size_t structure_bit_bound,
                                            const fixedPointEncoding& gradient_encoding,
                                            const fixedPointEncoding& gain_numerator_encoding,
                                            const fixedPointEncoding& quantized_gain_encoding,
                                            const fixedPointEncoding& leaf_value_encoding,
                                            const fixedPointEncoding& score_encoding,
                                            const fixedPointEncoding& softmax_encoding,
                                            size_t hessian_bit_bound,
                                            size_t gain_denominator_bit_bound,
                                            size_t leaf_denominator_bit_bound,
                                            size_t gain_gap_bit_bound,
                                            size_t l2_regularization,
                                            const Fr& split_penalty,
                                            const Fr& split_gain_threshold,
                                            size_t max_depth,
                                            size_t learning_rate_numerator,
                                            size_t learning_rate_denominator,
                                            forestTrainingLoss loss_mode)
{
    if (!is_setup) {
        throw std::runtime_error("forestTrainingProofProtocol not setup");
    }
    if (params_.forest_histogram_refresh == nullptr || params_.forest_split_statistics == nullptr ||
        params_.forest_split_gain == nullptr ||
        params_.forest_quantized_split_gain == nullptr || params_.forest_split_selection == nullptr ||
        params_.forest_leaf_value == nullptr || params_.forest_boosting_state == nullptr || params_.row_sum == nullptr ||
        params_.zero_check == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("forestTrainingProofProtocol dependencies not setup");
    }

    validate_forest_training_proof_parameters(witness,
                                              tree_count,
                                              node_count,
                                              feature_count,
                                              bin_count,
                                              query_count,
                                              input_bit_bound,
                                              full_upper_bound,
                                              structure_bit_bound,
                                              hessian_bit_bound,
                                              gain_denominator_bit_bound,
                                              leaf_denominator_bit_bound,
                                              gain_gap_bit_bound);
    validate_fixed_point_encoding(softmax_encoding, "forestTrainingProofProtocol softmax encoding");
    if (softmax_encoding.fractional_bits >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error(
            "forestTrainingProofProtocol: softmax fractional bits exceed size_t scale range");
    }
    if (tree_count != iteration_count * class_count) {
        throw std::runtime_error(
            "forestTrainingProofProtocol: tree_count must equal iteration_count * class_count");
    }

    const size_t query_row_count = tree_count * query_count;

    witnesses_.push_back(witness);
    checks_.push_back({tree_count,
                       node_count,
                       feature_count,
                       bin_count,
                       query_count,
                       query_row_count,
                       iteration_count,
                       class_count,
                       input_bit_bound,
                       full_upper_bound,
                       structure_bit_bound,
                       gradient_encoding,
                       gain_numerator_encoding,
                       quantized_gain_encoding,
                       leaf_value_encoding,
                       score_encoding,
                       softmax_encoding,
                       hessian_bit_bound,
                       gain_denominator_bit_bound,
                       leaf_denominator_bit_bound,
                       gain_gap_bit_bound,
                       l2_regularization,
                       split_penalty,
                       split_gain_threshold,
                       max_depth,
                       learning_rate_numerator,
                       learning_rate_denominator,
                       loss_mode});
}

bool forestTrainingProofProtocol::finalize()
{
    begin_finalize("forestTrainingProofProtocol");

    auto add_zero = [this](const virCommitment& expression, size_t n, const std::string& label) {
        if (is_zero_virtual_commitment(expression)) {
            return;
        }
        virCommitment copy = expression;
        params_.zero_check->add_check(copy, n, label);
    };
    auto add_zero_difference = [&](const virCommitment& left,
                                   const virCommitment& right,
                                   size_t n,
                                   const std::string& label) {
        if (same_virtual_commitment_alias(left, right)) {
            return;
        }
        add_zero(left - right, n, label);
    };

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestTrainingProofCheckRecord& record = checks_[i];
        const forestTrainingProofWitness& witness = witnesses_[i];

        const size_t histogram_width = record.feature_count * record.bin_count;
        const size_t histogram_domain_size = record.node_count * histogram_width;
        const size_t candidate_domain_size = histogram_domain_size;
        const virCommitment feature_indices =
            matrix_column_index_expression(record.node_count, record.feature_count);
        const virCommitment first_feature_mask =
            first_feature_mask_expression(record.node_count, record.feature_count, record.bin_count);

        params_.forest_histogram_refresh->add_check_with_certified_parent_mask(
            witness.histogram_refresh,
            record.query_row_count,
            record.node_count,
            record.feature_count,
            record.bin_count,
            record.input_bit_bound);

        params_.forest_split_statistics->add_check(
            witness.split_statistics, record.node_count, record.feature_count, record.bin_count);

        forestSplitGainWitness split_gain = witness.split_gain;
        split_gain.left_counts = witness.split_statistics.left_statistics[0];
        split_gain.right_counts = witness.split_statistics.right_statistics[0];
        params_.forest_split_gain->add_check(split_gain,
                                             record.node_count,
                                             record.feature_count,
                                             record.bin_count,
                                             record.gradient_encoding,
                                             record.hessian_bit_bound,
                                             record.gain_denominator_bit_bound,
                                             record.l2_regularization,
                                             record.split_penalty,
                                             record.loss_mode,
                                             witness.split_selection.candidate_mask,
                                             size_t(1) << record.softmax_encoding.fractional_bits);

        params_.forest_quantized_split_gain->add_masked_check(witness.quantized_split_gain,
                                                              witness.split_selection.candidate_mask,
                                                              record.node_count,
                                                              record.feature_count,
                                                              record.bin_count,
                                                              record.gain_numerator_encoding,
                                                              record.quantized_gain_encoding,
                                                              record.gain_denominator_bit_bound);

        forestSplitSelectionWitness split_selection = witness.split_selection;
        split_selection.left_counts = witness.split_statistics.left_statistics[0];
        split_selection.right_counts = witness.split_statistics.right_statistics[0];
        params_.forest_split_selection->add_check(split_selection,
                                                  record.node_count,
                                                  record.feature_count,
                                                  record.bin_count,
                                                  record.query_count,
                                                  record.gain_gap_bit_bound);

        params_.forest_leaf_value->add_check(witness.leaf_value,
                                             record.node_count,
                                             record.gradient_encoding,
                                             record.leaf_value_encoding,
                                             record.hessian_bit_bound,
                                             record.leaf_denominator_bit_bound,
                                             record.l2_regularization,
                                             record.learning_rate_numerator,
                                             record.learning_rate_denominator,
                                             record.loss_mode,
                                             witness.node_leaf_mask);

        params_.forest_boosting_state->add_check(witness.boosting_state,
                                                 record.iteration_count,
                                                 record.class_count,
                                                 record.query_count,
                                                 record.score_encoding,
                                                 record.softmax_encoding,
                                                 record.hessian_bit_bound,
                                                 record.loss_mode);

        add_zero_difference(witness.histogram_refresh.inference.node_tree_ids,
                            witness.node_tree_ids,
                            record.node_count,
                            "forest_training_proof.node_tree_ids");
        add_zero_difference(witness.histogram_refresh.inference.node_lower_bounds,
                            witness.lower_bounds,
                            record.node_count * record.feature_count,
                            "forest_training_proof.node_lower_bounds");
        add_zero_difference(witness.histogram_refresh.inference.node_upper_bounds,
                            witness.upper_bounds,
                            record.node_count * record.feature_count,
                            "forest_training_proof.node_upper_bounds");
        add_zero_difference(witness.histogram_refresh.parent_mask,
                            witness.split_node_mask,
                            record.node_count,
                            "forest_training_proof.parent_mask");
        add_zero_difference(witness.histogram_refresh.left_child_indices,
                            witness.left_child_indices,
                            record.node_count,
                            "forest_training_proof.left_child_indices");
        add_zero_difference(witness.histogram_refresh.right_child_indices,
                            witness.right_child_indices,
                            record.node_count,
                            "forest_training_proof.right_child_indices");
        add_zero_difference(witness.boosting_state.tree_scores,
                            witness.histogram_refresh.inference.tree_scores,
                            record.query_row_count,
                            "forest_training_proof.tree_scores");

        virCommitment count_channel_one = witness.histogram_refresh.channels.query_values[0] - virCommitment(Fr(1));
        add_zero(count_channel_one, record.query_row_count, "forest_training_proof.count_channel");
        add_zero_difference(witness.histogram_refresh.channels.query_values[1],
                            witness.boosting_state.pseudo_residuals,
                            record.query_row_count,
                            "forest_training_proof.residual_channel");
        add_zero_difference(witness.histogram_refresh.channels.query_values[2],
                            witness.boosting_state.diagonal_hessians,
                            record.query_row_count,
                            "forest_training_proof.hessian_channel");

        for (size_t channel = 0; channel < 3; ++channel) {
            add_zero_difference(witness.split_statistics.histograms[channel],
                                witness.histogram_refresh.channels.full_histograms[channel],
                                histogram_domain_size,
                                "forest_training_proof.split_histograms.channel" + std::to_string(channel));
        }

        add_zero_difference(witness.split_gain.left_gradients,
                            witness.split_statistics.left_statistics[1],
                            candidate_domain_size,
                            "forest_training_proof.left_gradients");
        add_zero_difference(witness.split_gain.right_gradients,
                            witness.split_statistics.right_statistics[1],
                            candidate_domain_size,
                            "forest_training_proof.right_gradients");
        add_zero_difference(witness.split_gain.left_hessians,
                            witness.split_statistics.left_statistics[2],
                            candidate_domain_size,
                            "forest_training_proof.left_hessians");
        add_zero_difference(witness.split_gain.right_hessians,
                            witness.split_statistics.right_statistics[2],
                            candidate_domain_size,
                            "forest_training_proof.right_hessians");

        add_zero_difference(witness.quantized_split_gain.exact_gain_numerators,
                            witness.split_gain.gain_numerators,
                            candidate_domain_size,
                            "forest_training_proof.gain_numerators");
        add_zero_difference(witness.quantized_split_gain.gain_denominators,
                            witness.split_gain.gain_denominators,
                            candidate_domain_size,
                            "forest_training_proof.gain_denominators");

        virCommitment decoded_quantized_gain_zero =
            witness.split_selection.quantized_gains -
            (witness.quantized_split_gain.encoded_quantized_gains -
             virCommitment(fixed_point_offset_fr(record.quantized_gain_encoding)));
        add_zero(decoded_quantized_gain_zero, candidate_domain_size, "forest_training_proof.quantized_gains");

        virCommitment split_node_has_candidate_zero =
            witness.split_node_mask * (witness.split_selection.active_node_mask - virCommitment(Fr(1)));
        add_zero(split_node_has_candidate_zero, record.node_count, "forest_training_proof.split_node_has_candidate");

        const size_t stopping_gap_bit_bound = record.gain_gap_bit_bound + 1;
        params_.less_check->add_masked_check(witness.split_selection.selected_gains -
                                                 virCommitment(record.split_gain_threshold),
                                             witness.split_node_mask,
                                             stopping_gap_bit_bound,
                                             record.node_count);
        const virCommitment leaf_candidate_mask =
            witness.histogram_refresh.inference.node_leaf_mask * witness.split_selection.active_node_mask;
        const virCommitment depth_limit_boolean =
            witness.depth_limit_mask * (witness.depth_limit_mask - virCommitment(Fr(1)));
        add_zero(depth_limit_boolean, record.node_count, "forest_training_proof.depth_limit_mask_boolean");
        const virCommitment depth_limit_scope =
            witness.depth_limit_mask * (virCommitment(Fr(1)) - leaf_candidate_mask);
        add_zero(depth_limit_scope, record.node_count, "forest_training_proof.depth_limit_mask_scope");
        params_.less_check->add_masked_check(witness.depths -
                                                 virCommitment(field_from_size_t(record.max_depth)),
                                             witness.depth_limit_mask,
                                             record.structure_bit_bound,
                                             record.node_count);
        const virCommitment threshold_stop_mask =
            leaf_candidate_mask * (virCommitment(Fr(1)) - witness.depth_limit_mask);
        params_.less_check->add_masked_check(virCommitment(record.split_gain_threshold) -
                                                 witness.split_selection.selected_gains - virCommitment(Fr(1)),
                                             threshold_stop_mask,
                                             stopping_gap_bit_bound,
                                             record.node_count);

        const virCommitment selected_split_feature =
            witness.split_node_mask * witness.split_selection.selected_features;
        params_.row_sum->add_check(
            witness.feature_selector * feature_indices, selected_split_feature, record.node_count, record.feature_count);
        virCommitment selected_bin_zero =
            witness.split_node_mask *
            (witness.thresholds - witness.split_selection.selected_bins - virCommitment(Fr(1)));
        add_zero(selected_bin_zero, record.node_count, "forest_training_proof.selected_bins");

        params_.row_sum->add_check(witness.histogram_refresh.channels.leaf_histograms[1] * first_feature_mask,
                                   witness.leaf_value.leaf_gradients,
                                   record.node_count,
                                   histogram_width);
        params_.row_sum->add_check(witness.histogram_refresh.channels.leaf_histograms[2] * first_feature_mask,
                                   witness.leaf_value.leaf_hessians,
                                   record.node_count,
                                   histogram_width);

        virCommitment decoded_leaf_value_zero =
            witness.histogram_refresh.inference.node_leaf_mask *
            (witness.histogram_refresh.inference.node_leaf_values -
             (witness.leaf_value.encoded_leaf_values - virCommitment(fixed_point_offset_fr(record.leaf_value_encoding))));
        add_zero(decoded_leaf_value_zero, record.node_count, "forest_training_proof.node_leaf_values");
    }

    return true;
}
