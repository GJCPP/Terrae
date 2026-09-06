#include "forest_split_gain.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

void validate_split_gain_parameters(size_t node_count,
                                    size_t feature_count,
                                    size_t bin_count,
                                    const fixedPointEncoding& gradient_encoding,
                                    size_t hessian_bit_bound,
                                    size_t denominator_bit_bound,
                                    size_t l2_regularization,
                                    forestTrainingLoss loss_mode,
                                    const Fr& split_penalty,
                                    size_t first_order_hessian_scale)
{
    validate_matrix_shape(node_count, feature_count * bin_count, "forestSplitGainProtocol candidate tensors");
    validate_fixed_point_encoding(gradient_encoding, "forestSplitGainProtocol");
    if (!gradient_encoding.signed_values) {
        throw std::runtime_error("forestSplitGainProtocol: gradient encoding must be signed");
    }
    if (hessian_bit_bound == 0) {
        throw std::runtime_error("forestSplitGainProtocol: hessian bit bound must be non-zero");
    }
    if (denominator_bit_bound == 0) {
        throw std::runtime_error("forestSplitGainProtocol: denominator bit bound must be non-zero");
    }
    if (denominator_bit_bound < hessian_bit_bound) {
        throw std::runtime_error("forestSplitGainProtocol: denominator bit bound must cover the Hessian bound");
    }
    if (loss_mode == forestTrainingLoss::FirstOrderRss) {
        if (first_order_hessian_scale == 0) {
            throw std::runtime_error(
                "forestSplitGainProtocol: first-order Hessian scale must be non-zero");
        }
        if (l2_regularization != 0 || split_penalty != Fr(0)) {
            throw std::runtime_error(
                "forestSplitGainProtocol: first-order RSS requires zero L2 regularization and split penalty");
        }
    } else if (l2_regularization == 0) {
        throw std::runtime_error("forestSplitGainProtocol: second-order XGBoost requires non-zero L2 regularization");
    }
}

} // namespace

void forestSplitGainProtocol::add_check(const forestSplitGainWitness& witness,
                                        size_t node_count,
                                        size_t feature_count,
                                        size_t bin_count,
                                        const fixedPointEncoding& gradient_encoding,
                                        size_t hessian_bit_bound,
                                        size_t denominator_bit_bound,
                                        size_t l2_regularization,
                                        const Fr& split_penalty)
{
    add_check(witness,
              node_count,
              feature_count,
              bin_count,
              gradient_encoding,
              hessian_bit_bound,
              denominator_bit_bound,
              l2_regularization,
              split_penalty,
              forestTrainingLoss::SecondOrderXgboost,
              virCommitment(Fr(0)),
              1);
}

void forestSplitGainProtocol::add_check(const forestSplitGainWitness& witness,
                                        size_t node_count,
                                        size_t feature_count,
                                        size_t bin_count,
                                        const fixedPointEncoding& gradient_encoding,
                                        size_t hessian_bit_bound,
                                        size_t denominator_bit_bound,
                                        size_t l2_regularization,
                                        const Fr& split_penalty,
                                        forestTrainingLoss loss_mode,
                                        const virCommitment& candidate_mask,
                                        size_t first_order_hessian_scale)
{
    if (!is_setup) {
        throw std::runtime_error("forestSplitGainProtocol not setup");
    }
    if (params_.zero_check == nullptr || params_.less_check == nullptr || params_.signed_range_check == nullptr) {
        throw std::runtime_error("forestSplitGainProtocol dependencies not setup");
    }

    validate_split_gain_parameters(node_count,
                                   feature_count,
                                   bin_count,
                                   gradient_encoding,
                                   hessian_bit_bound,
                                   denominator_bit_bound,
                                   l2_regularization,
                                   loss_mode,
                                   split_penalty,
                                   first_order_hessian_scale);

    witnesses_.push_back(witness);
    checks_.push_back(
        {node_count, feature_count, bin_count, gradient_encoding, hessian_bit_bound, denominator_bit_bound,
         l2_regularization, split_penalty, loss_mode, first_order_hessian_scale});
    candidate_masks_.push_back(candidate_mask);
}

bool forestSplitGainProtocol::finalize()
{
    begin_finalize("forestSplitGainProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestSplitGainCheckRecord& record = checks_[i];
        const forestSplitGainWitness& witness = witnesses_[i];
        const size_t domain_size = record.node_count * record.feature_count * record.bin_count;
        const virCommitment gradient_offset(fixed_point_offset_fr(record.gradient_encoding));
        const virCommitment lambda(field_from_size_t(record.l2_regularization));
        const virCommitment split_penalty(record.split_penalty);

        const virCommitment parent_gradients = witness.left_gradients + witness.right_gradients;
        const virCommitment parent_hessians = witness.left_hessians + witness.right_hessians;
        // candidate_mask is already constrained by split selection. Reuse the
        // same commitment only to make inactive RSS denominators non-zero; do
        // not prove its Booleanity a second time here.
        virCommitment left_denominators = witness.left_hessians + lambda;
        virCommitment right_denominators = witness.right_hessians + lambda;
        virCommitment parent_denominators = parent_hessians + lambda;
        virCommitment denominator_scale(Fr(1));
        if (record.loss_mode == forestTrainingLoss::FirstOrderRss) {
            const virCommitment inactive_candidate = virCommitment(Fr(1)) - candidate_masks_[i];
            left_denominators = witness.left_counts + inactive_candidate;
            right_denominators = witness.right_counts + inactive_candidate;
            parent_denominators = witness.left_counts + witness.right_counts + inactive_candidate;
            denominator_scale = virCommitment(field_from_size_t(record.first_order_hessian_scale));
        }

        params_.signed_range_check->add_check(witness.left_gradients + gradient_offset, record.gradient_encoding, domain_size);
        params_.signed_range_check->add_check(
            witness.right_gradients + gradient_offset, record.gradient_encoding, domain_size);
        params_.signed_range_check->add_check(parent_gradients + gradient_offset, record.gradient_encoding, domain_size);

        params_.less_check->add_check(witness.left_hessians, record.hessian_bit_bound, domain_size);
        params_.less_check->add_check(witness.right_hessians, record.hessian_bit_bound, domain_size);
        params_.less_check->add_check(parent_hessians, record.denominator_bit_bound, domain_size);

        params_.less_check->add_check(left_denominators, record.denominator_bit_bound, domain_size);
        params_.less_check->add_check(right_denominators, record.denominator_bit_bound, domain_size);
        params_.less_check->add_check(parent_denominators, record.denominator_bit_bound, domain_size);
        if (params_.nonzero_check != nullptr) {
            params_.nonzero_check->add_check(left_denominators, domain_size, "forest_split_gain.left_denominator_nonzero");
            params_.nonzero_check->add_check(right_denominators, domain_size, "forest_split_gain.right_denominator_nonzero");
            params_.nonzero_check->add_check(parent_denominators, domain_size, "forest_split_gain.parent_denominator_nonzero");
        } else {
            params_.less_check->add_check(
                left_denominators - virCommitment(Fr(1)), record.denominator_bit_bound, domain_size);
            params_.less_check->add_check(
                right_denominators - virCommitment(Fr(1)), record.denominator_bit_bound, domain_size);
            params_.less_check->add_check(
                parent_denominators - virCommitment(Fr(1)), record.denominator_bit_bound, domain_size);
        }

        virCommitment left_square_zero = witness.left_gradient_squares - witness.left_gradients * witness.left_gradients;
        virCommitment right_square_zero =
            witness.right_gradient_squares - witness.right_gradients * witness.right_gradients;
        virCommitment parent_square_zero = witness.parent_gradient_squares - parent_gradients * parent_gradients;
        params_.zero_check->add_check(left_square_zero, domain_size);
        params_.zero_check->add_check(right_square_zero, domain_size);
        params_.zero_check->add_check(parent_square_zero, domain_size);

        const virCommitment expected_gain_denominator =
            left_denominators * right_denominators * parent_denominators * denominator_scale;
        virCommitment gain_denominator_zero = witness.gain_denominators - expected_gain_denominator;
        params_.zero_check->add_check(gain_denominator_zero, domain_size);

        const virCommitment expected_gain_numerator =
            witness.left_gradient_squares * right_denominators * parent_denominators +
            witness.right_gradient_squares * left_denominators * parent_denominators -
            witness.parent_gradient_squares * left_denominators * right_denominators -
            split_penalty * witness.gain_denominators;
        virCommitment gain_numerator_zero = witness.gain_numerators - expected_gain_numerator;
        params_.zero_check->add_check(gain_numerator_zero, domain_size);
    }

    return true;
}
