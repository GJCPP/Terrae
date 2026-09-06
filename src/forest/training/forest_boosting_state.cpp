#include "forest_boosting_state.hpp"

#include "matrix_utils.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <limits>
#include <string>
#include <stdexcept>

namespace {

bool forest_boosting_state_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_FOREST_BOOSTING_STATE");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void forest_boosting_state_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = forest_boosting_state_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "forest_boosting_state_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("forest_boosting_state_profile", label, "", elapsed.count(), "ok");
}

void validate_forest_boosting_state_parameters(size_t iteration_count,
                                               size_t class_count,
                                               size_t query_count,
                                               const fixedPointEncoding& score_encoding,
                                               const fixedPointEncoding& softmax_encoding,
                                               size_t hessian_bit_bound)
{
    validate_matrix_shape(iteration_count, class_count * query_count, "forestBoostingStateProtocol");
    validate_matrix_shape(class_count, query_count, "forestBoostingStateProtocol");
    validate_fixed_point_encoding(score_encoding, "forestBoostingStateProtocol");
    validate_fixed_point_encoding(softmax_encoding, "forestBoostingStateProtocol");
    if (!score_encoding.signed_values) {
        throw std::runtime_error("forestBoostingStateProtocol: score encoding must be signed");
    }
    if (!softmax_encoding.signed_values) {
        throw std::runtime_error("forestBoostingStateProtocol: softmax encoding must be signed");
    }
    if (hessian_bit_bound == 0) {
        throw std::runtime_error("forestBoostingStateProtocol: hessian bit bound must be non-zero");
    }
}

fixedPointEncoding unsigned_channel_as_signed_encoding(size_t bit_bound, size_t fractional_bits, const char* context)
{
    if (bit_bound == 0) {
        throw std::runtime_error(std::string(context) + ": bit bound must be non-zero");
    }
    const size_t total_bits = std::max(bit_bound + 1, fractional_bits + 1);
    return fixedPointEncoding::signed_offset_binary(total_bits, fractional_bits);
}

size_t non_negative_bit_bound_from_encoding(const fixedPointEncoding& encoding)
{
    return encoding.total_bits - 1;
}

size_t floor_log2_power_of_two(size_t value)
{
    size_t out = 0;
    while ((size_t(1) << out) < value) {
        ++out;
    }
    return out;
}

void add_labeled_zero(zeroProtocol& zero, const virCommitment& expression, size_t n, const std::string& label)
{
    virCommitment copy = expression;
    zero.add_check(copy, n, label);
}

} // namespace

void forestBoostingStateProtocol::add_check(const forestBoostingStateWitness& witness,
                                            size_t iteration_count,
                                            size_t class_count,
                                            size_t query_count,
                                            const fixedPointEncoding& score_encoding,
                                            const fixedPointEncoding& softmax_encoding,
                                            size_t hessian_bit_bound,
                                            forestTrainingLoss loss_mode)
{
    if (!is_setup) {
        throw std::runtime_error("forestBoostingStateProtocol not setup");
    }
    if (params_.zero_check == nullptr || params_.column_prefix == nullptr || params_.column_sum == nullptr ||
        params_.cube_argmax == nullptr || params_.replication == nullptr || params_.cube_sum == nullptr ||
        params_.exp_check == nullptr || params_.fixed_point_division == nullptr || params_.signed_range_check == nullptr) {
        throw std::runtime_error("forestBoostingStateProtocol dependencies not setup");
    }

    validate_forest_boosting_state_parameters(
        iteration_count, class_count, query_count, score_encoding, softmax_encoding, hessian_bit_bound);

    witnesses_.push_back(witness);
    checks_.push_back(
        {iteration_count, class_count, query_count, score_encoding, softmax_encoding, hessian_bit_bound, loss_mode});
}

bool forestBoostingStateProtocol::finalize()
{
    begin_finalize("forestBoostingStateProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestBoostingStateCheckRecord& record = checks_[i];
        const forestBoostingStateWitness& witness = witnesses_[i];
        const size_t matrix_column_count = record.class_count * record.query_count;
        const size_t cube_domain_size = record.iteration_count * record.class_count * record.query_count;
        const size_t score_bit_bound = fixed_point_encoded_bit_bound(record.score_encoding);
        const size_t softmax_scale = size_t(1) << record.softmax_encoding.fractional_bits;
        const size_t softmax_scale_bit_bound = record.softmax_encoding.fractional_bits + 1;
        const fixedPointEncoding exp_output_encoding =
            unsigned_channel_as_signed_encoding(softmax_scale_bit_bound, 0, "forestBoostingStateProtocol exp output");
        const size_t exp_output_bit_bound = non_negative_bit_bound_from_encoding(exp_output_encoding);
        const size_t softmax_denominator_bit_bound =
            exp_output_bit_bound + floor_log2_power_of_two(record.class_count) + 1;
        const fixedPointEncoding hessian_encoding =
            unsigned_channel_as_signed_encoding(record.hessian_bit_bound, record.softmax_encoding.fractional_bits,
                                                "forestBoostingStateProtocol hessian");
        const fixedPointEncoding hessian_product_encoding =
            unsigned_channel_as_signed_encoding(record.hessian_bit_bound + record.softmax_encoding.fractional_bits + 1,
                                                record.softmax_encoding.fractional_bits,
                                                "forestBoostingStateProtocol hessian product");
        const virCommitment score_offset(fixed_point_offset_fr(record.score_encoding));
        const virCommitment softmax_offset(fixed_point_offset_fr(record.softmax_encoding));
        const virCommitment hessian_offset(fixed_point_offset_fr(hessian_encoding));
        const virCommitment hessian_product_offset(fixed_point_offset_fr(hessian_product_encoding));
        const virCommitment softmax_scale_commitment(field_from_size_t(softmax_scale));

        const virCommitment inclusive_scores = witness.cumulative_scores + witness.tree_scores;
        forest_boosting_state_profile_scope("column_prefix.add_check", [&] {
            params_.column_prefix->add_check(
                witness.tree_scores, inclusive_scores, record.iteration_count, matrix_column_count);
        });

        virCommitment encoded_score_zero = witness.encoded_cumulative_scores - score_offset - witness.cumulative_scores;
        forest_boosting_state_profile_scope("zero.encoded_cumulative_scores", [&] {
            add_labeled_zero(*params_.zero_check,
                             encoded_score_zero,
                             cube_domain_size,
                             "forest_boosting_state.encoded_cumulative_scores");
        });

        forest_boosting_state_profile_scope("cube_argmax.add_check", [&] {
            params_.cube_argmax->add_check(witness.encoded_cumulative_scores,
                                           witness.softmax_shifts,
                                           witness.lifted_softmax_shifts,
                                           witness.shift_selector,
                                           record.iteration_count,
                                           record.class_count,
                                           record.query_count,
                                           score_bit_bound);
        });
        virCommitment shifted_score_zero =
            witness.encoded_shifted_scores - witness.encoded_cumulative_scores + witness.lifted_softmax_shifts - score_offset;
        forest_boosting_state_profile_scope("zero.encoded_shifted_scores", [&] {
            add_labeled_zero(*params_.zero_check,
                             shifted_score_zero,
                             cube_domain_size,
                             "forest_boosting_state.encoded_shifted_scores");
        });
        forest_boosting_state_profile_scope("signed_range.encoded_shifted_scores", [&] {
            params_.signed_range_check->add_check(
                witness.encoded_shifted_scores, record.score_encoding, cube_domain_size);
        });

        forest_boosting_state_profile_scope("exp_check.add_non_positive_check", [&] {
            params_.exp_check->add_non_positive_check(
                witness.encoded_shifted_scores, witness.softmax_exp_values, record.score_encoding, cube_domain_size);
        });
        forest_boosting_state_profile_scope("cube_sum.softmax_denominators", [&] {
            params_.cube_sum->add_check(witness.softmax_exp_values,
                                        witness.softmax_denominators,
                                        record.iteration_count,
                                        record.class_count,
                                        record.query_count);
        });
        forest_boosting_state_profile_scope("replication.softmax_denominators", [&] {
            params_.replication->add_check(witness.softmax_denominators,
                                           witness.lifted_softmax_denominators,
                                           record.iteration_count,
                                           record.class_count,
                                           record.query_count);
        });

        const virCommitment encoded_softmax_exp_values = witness.softmax_exp_values + virCommitment(fixed_point_offset_fr(exp_output_encoding));
        forest_boosting_state_profile_scope("fixed_point_division.softmax_probabilities", [&] {
            params_.fixed_point_division->add_check(encoded_softmax_exp_values,
                                                    witness.lifted_softmax_denominators,
                                                    witness.softmax_probabilities,
                                                    witness.softmax_remainders,
                                                    exp_output_encoding,
                                                    record.softmax_encoding,
                                                    softmax_denominator_bit_bound,
                                                    cube_domain_size);
        });

        virCommitment label_boolean_zero = witness.base_labels * (witness.base_labels - virCommitment(Fr(1)));
        forest_boosting_state_profile_scope("zero.base_labels_boolean", [&] {
            params_.zero_check->add_check(label_boolean_zero, matrix_column_count);
        });
        forest_boosting_state_profile_scope("column_sum.base_labels", [&] {
            params_.column_sum->add_check(
                witness.base_labels, virCommitment(Fr(1)), record.class_count, record.query_count);
        });

        const virCommitment lifted_labels = x_pow(witness.base_labels, record.iteration_count);
        const virCommitment decoded_softmax_probabilities = witness.softmax_probabilities - softmax_offset;
        virCommitment residual_zero =
            record.loss_mode == forestTrainingLoss::FirstOrderRss
                ? witness.pseudo_residuals + decoded_softmax_probabilities - lifted_labels * softmax_scale_commitment
                : witness.pseudo_residuals - decoded_softmax_probabilities + lifted_labels * softmax_scale_commitment;
        forest_boosting_state_profile_scope("zero.pseudo_residuals", [&] {
            add_labeled_zero(
                *params_.zero_check, residual_zero, cube_domain_size, "forest_boosting_state.pseudo_residuals");
        });

        if (record.loss_mode == forestTrainingLoss::FirstOrderRss) {
            const virCommitment encoded_hessian_zero =
                witness.encoded_diagonal_hessians - hessian_offset - softmax_scale_commitment;
            forest_boosting_state_profile_scope("zero.encoded_diagonal_hessians", [&] {
                add_labeled_zero(*params_.zero_check,
                                 encoded_hessian_zero,
                                 cube_domain_size,
                                 "forest_boosting_state.encoded_diagonal_hessians");
            });
            forest_boosting_state_profile_scope("zero.diagonal_hessian_remainders", [&] {
                add_labeled_zero(*params_.zero_check,
                                 witness.diagonal_hessian_remainders,
                                 cube_domain_size,
                                 "forest_boosting_state.diagonal_hessian_remainders");
            });
            const virCommitment hessian_zero = witness.diagonal_hessians - softmax_scale_commitment;
            forest_boosting_state_profile_scope("zero.diagonal_hessians", [&] {
                add_labeled_zero(
                    *params_.zero_check, hessian_zero, cube_domain_size, "forest_boosting_state.diagonal_hessians");
            });
        } else {
            const virCommitment hessian_product =
                decoded_softmax_probabilities * (softmax_scale_commitment - decoded_softmax_probabilities);
            forest_boosting_state_profile_scope("fixed_point_division.diagonal_hessians", [&] {
                params_.fixed_point_division->add_check(hessian_product + hessian_product_offset,
                                                        softmax_scale_commitment,
                                                        witness.encoded_diagonal_hessians,
                                                        witness.diagonal_hessian_remainders,
                                                        hessian_product_encoding,
                                                        hessian_encoding,
                                                        softmax_scale_bit_bound,
                                                        cube_domain_size);
            });

            const virCommitment hessian_zero =
                witness.diagonal_hessians - (witness.encoded_diagonal_hessians - hessian_offset);
            forest_boosting_state_profile_scope("zero.diagonal_hessians", [&] {
                add_labeled_zero(
                    *params_.zero_check, hessian_zero, cube_domain_size, "forest_boosting_state.diagonal_hessians");
            });
        }
    }

    return true;
}
