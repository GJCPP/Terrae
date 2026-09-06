#include "forest_quantized_split_gain.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

void validate_quantized_split_gain_parameters(size_t node_count,
                                              size_t feature_count,
                                              size_t bin_count,
                                              const fixedPointEncoding& numerator_encoding,
                                              const fixedPointEncoding& quantized_gain_encoding,
                                              size_t denominator_bit_bound)
{
    validate_matrix_shape(node_count, feature_count * bin_count, "forestQuantizedSplitGainProtocol candidate tensors");
    validate_fixed_point_encoding(numerator_encoding, "forestQuantizedSplitGainProtocol");
    validate_fixed_point_encoding(quantized_gain_encoding, "forestQuantizedSplitGainProtocol");
    if (!numerator_encoding.signed_values || !quantized_gain_encoding.signed_values) {
        throw std::runtime_error("forestQuantizedSplitGainProtocol: encodings must be signed");
    }
    if (denominator_bit_bound == 0) {
        throw std::runtime_error("forestQuantizedSplitGainProtocol: denominator bit bound must be non-zero");
    }
}

} // namespace

void forestQuantizedSplitGainProtocol::add_check(const forestQuantizedSplitGainWitness& witness,
                                                 size_t node_count,
                                                 size_t feature_count,
                                                 size_t bin_count,
                                                 const fixedPointEncoding& numerator_encoding,
                                                 const fixedPointEncoding& quantized_gain_encoding,
                                                 size_t denominator_bit_bound)
{
    add_masked_check(witness,
                     virCommitment(Fr(1)),
                     node_count,
                     feature_count,
                     bin_count,
                     numerator_encoding,
                     quantized_gain_encoding,
                     denominator_bit_bound);
    checks_.back().uses_mask = false;
}

void forestQuantizedSplitGainProtocol::add_masked_check(const forestQuantizedSplitGainWitness& witness,
                                                        const virCommitment& candidate_mask,
                                                        size_t node_count,
                                                        size_t feature_count,
                                                        size_t bin_count,
                                                        const fixedPointEncoding& numerator_encoding,
                                                        const fixedPointEncoding& quantized_gain_encoding,
                                                        size_t denominator_bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("forestQuantizedSplitGainProtocol not setup");
    }
    if (params_.zero_check == nullptr || params_.fixed_point_division == nullptr) {
        throw std::runtime_error("forestQuantizedSplitGainProtocol dependencies not setup");
    }

    validate_quantized_split_gain_parameters(
        node_count, feature_count, bin_count, numerator_encoding, quantized_gain_encoding, denominator_bit_bound);

    witnesses_.push_back(witness);
    candidate_masks_.push_back(candidate_mask);
    checks_.push_back({node_count,
                       feature_count,
                       bin_count,
                       numerator_encoding,
                       quantized_gain_encoding,
                       denominator_bit_bound,
                       true});
}

bool forestQuantizedSplitGainProtocol::finalize()
{
    begin_finalize("forestQuantizedSplitGainProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestQuantizedSplitGainCheckRecord& record = checks_[i];
        const forestQuantizedSplitGainWitness& witness = witnesses_[i];
        const size_t domain_size = record.node_count * record.feature_count * record.bin_count;

        virCommitment numerator_decode_zero =
            witness.encoded_gain_numerators - virCommitment(fixed_point_offset_fr(record.numerator_encoding)) -
            witness.exact_gain_numerators;
        if (record.uses_mask) {
            numerator_decode_zero = candidate_masks_[i] * numerator_decode_zero;
            params_.fixed_point_division->add_masked_check(witness.encoded_gain_numerators,
                                                           witness.gain_denominators,
                                                           witness.encoded_quantized_gains,
                                                           witness.gain_remainders,
                                                           candidate_masks_[i],
                                                           record.numerator_encoding,
                                                           record.quantized_gain_encoding,
                                                           record.denominator_bit_bound,
                                                           domain_size);
        } else {
            params_.fixed_point_division->add_check(witness.encoded_gain_numerators,
                                                   witness.gain_denominators,
                                                   witness.encoded_quantized_gains,
                                                   witness.gain_remainders,
                                                   record.numerator_encoding,
                                                   record.quantized_gain_encoding,
                                                   record.denominator_bit_bound,
                                                   domain_size);
        }

        params_.zero_check->add_check(numerator_decode_zero, domain_size);
    }

    return true;
}
