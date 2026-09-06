#include "forest_leaf_value.hpp"

#include "algebra_utils.hpp"
#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

void validate_leaf_value_parameters(size_t node_count,
                                    const fixedPointEncoding& gradient_encoding,
                                    const fixedPointEncoding& leaf_value_encoding,
                                    size_t hessian_bit_bound,
                                    size_t denominator_bit_bound,
                                    size_t learning_rate_numerator,
                                    size_t learning_rate_denominator)
{
    if (!isPowerOfTwo(node_count) || node_count == 0) {
        throw std::runtime_error("forestLeafValueProtocol: node_count must be a non-zero power of two");
    }
    validate_fixed_point_encoding(gradient_encoding, "forestLeafValueProtocol");
    validate_fixed_point_encoding(leaf_value_encoding, "forestLeafValueProtocol");
    if (!gradient_encoding.signed_values || !leaf_value_encoding.signed_values) {
        throw std::runtime_error("forestLeafValueProtocol: encodings must be signed");
    }
    if (hessian_bit_bound == 0 || denominator_bit_bound == 0) {
        throw std::runtime_error("forestLeafValueProtocol: bit bounds must be non-zero");
    }
    if (learning_rate_numerator == 0 || learning_rate_denominator == 0) {
        throw std::runtime_error("forestLeafValueProtocol: learning rate terms must be non-zero");
    }
}

} // namespace

void forestLeafValueProtocol::add_check(const forestLeafValueWitness& witness,
                                        size_t node_count,
                                        const fixedPointEncoding& gradient_encoding,
                                        const fixedPointEncoding& leaf_value_encoding,
                                        size_t hessian_bit_bound,
                                        size_t denominator_bit_bound,
                                        size_t l2_regularization,
                                        size_t learning_rate_numerator,
                                        size_t learning_rate_denominator)
{
    add_check(witness,
              node_count,
              gradient_encoding,
              leaf_value_encoding,
              hessian_bit_bound,
              denominator_bit_bound,
              l2_regularization,
              learning_rate_numerator,
              learning_rate_denominator,
              forestTrainingLoss::SecondOrderXgboost,
              virCommitment(Fr(0)));
}

void forestLeafValueProtocol::add_check(const forestLeafValueWitness& witness,
                                        size_t node_count,
                                        const fixedPointEncoding& gradient_encoding,
                                        const fixedPointEncoding& leaf_value_encoding,
                                        size_t hessian_bit_bound,
                                        size_t denominator_bit_bound,
                                        size_t l2_regularization,
                                        size_t learning_rate_numerator,
                                        size_t learning_rate_denominator,
                                        forestTrainingLoss loss_mode,
                                        const virCommitment& node_leaf_mask)
{
    if (!is_setup) {
        throw std::runtime_error("forestLeafValueProtocol not setup");
    }
    if (params_.zero_check == nullptr || params_.less_check == nullptr || params_.fixed_point_division == nullptr) {
        throw std::runtime_error("forestLeafValueProtocol dependencies not setup");
    }

    validate_leaf_value_parameters(
        node_count,
        gradient_encoding,
        leaf_value_encoding,
        hessian_bit_bound,
        denominator_bit_bound,
        learning_rate_numerator,
        learning_rate_denominator);

    witnesses_.push_back(witness);
    checks_.push_back({node_count,
                       gradient_encoding,
                       leaf_value_encoding,
                       hessian_bit_bound,
                       denominator_bit_bound,
                       l2_regularization,
                       learning_rate_numerator,
                       learning_rate_denominator,
                       loss_mode});
    node_leaf_masks_.push_back(node_leaf_mask);
}

bool forestLeafValueProtocol::finalize()
{
    begin_finalize("forestLeafValueProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestLeafValueCheckRecord& record = checks_[i];
        const forestLeafValueWitness& witness = witnesses_[i];

        const virCommitment scaled_gradient =
            witness.leaf_gradients * virCommitment(field_from_size_t(record.learning_rate_numerator));
        virCommitment negative_gradient_zero =
            witness.encoded_negative_leaf_gradients -
            virCommitment(fixed_point_offset_fr(record.gradient_encoding)) +
            (record.loss_mode == forestTrainingLoss::FirstOrderRss
                 ? virCommitment(Fr(0)) - scaled_gradient
                 : scaled_gradient);
        params_.zero_check->add_check(negative_gradient_zero, record.node_count);

        params_.less_check->add_check(witness.leaf_hessians, record.hessian_bit_bound, record.node_count);
        params_.less_check->add_check(witness.leaf_denominators, record.denominator_bit_bound, record.node_count);

        // node_leaf_mask is owned by the committed-forest well-formedness
        // statement. The training proof only reuses it as a safe-denominator
        // selector and does not duplicate the mask's Boolean constraint.
        const virCommitment inactive_node =
            record.loss_mode == forestTrainingLoss::FirstOrderRss
                ? virCommitment(Fr(1)) - node_leaf_masks_[i]
                : virCommitment(Fr(0));
        virCommitment denominator_zero =
            witness.leaf_denominators -
            (witness.leaf_hessians + virCommitment(field_from_size_t(record.l2_regularization)) + inactive_node) *
                virCommitment(field_from_size_t(record.learning_rate_denominator));
        params_.zero_check->add_check(denominator_zero, record.node_count);

        params_.fixed_point_division->add_check(witness.encoded_negative_leaf_gradients,
                                                witness.leaf_denominators,
                                                witness.encoded_leaf_values,
                                                witness.leaf_remainders,
                                                record.gradient_encoding,
                                                record.leaf_value_encoding,
                                                record.denominator_bit_bound,
                                                record.node_count);
    }

    return true;
}
