#include "cube_softmax.hpp"

#include "algebra_utils.hpp"
#include "cube_argmax.hpp"
#include "matrix_utils.hpp"
#include "terrae_env.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

bool cube_replication_failure_debug_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_DEBUG_CUBE_REPLICATION_FAILURE");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

void validate_cube_shape(size_t outer_count, size_t middle_count, size_t inner_count, const char* context)
{
    if (!isPowerOfTwo(outer_count) || !isPowerOfTwo(middle_count) || !isPowerOfTwo(inner_count) ||
        outer_count == 0 || middle_count == 0 || inner_count == 0) {
        throw std::runtime_error(std::string(context) +
                                 ": outer_count, middle_count, and inner_count must be non-zero powers of two");
    }
}

size_t cube_domain_size(size_t outer_count, size_t middle_count, size_t inner_count)
{
    return outer_count * middle_count * inner_count;
}

size_t matrix_domain_size(size_t row_count, size_t column_count)
{
    return row_count * column_count;
}

std::vector<size_t> block_start_rows(size_t outer_count, size_t middle_count)
{
    std::vector<size_t> rows;
    rows.reserve(outer_count);
    for (size_t outer = 0; outer < outer_count; ++outer) {
        rows.push_back(outer * middle_count);
    }
    return rows;
}

virCommitment cube_boundary_selector_transition_expression(size_t outer_count,
                                                           size_t middle_count,
                                                           size_t inner_count)
{
    validate_cube_shape(
        outer_count, middle_count, inner_count, "cube_boundary_selector_transition_expression");

    const size_t domain_size = cube_domain_size(outer_count, middle_count, inner_count);
    const size_t block_size = middle_count * inner_count;
    const virCommitment block_start_first_column =
        x_pow(subgroup_zero_lagrange_expression(block_size), outer_count);
    const Fr next_row_shift = pow_fr(inverse_root_of_unity(domain_size), inner_count);
    return block_start_first_column - x_scale(block_start_first_column, next_row_shift);
}

void add_cube_boundary_selector_constraints(zeroProtocol& zero,
                                            const virCommitment& boundary_selector,
                                            size_t outer_count,
                                            size_t middle_count,
                                            size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cubeMiddleReplicationProtocol");

    const size_t domain_size = cube_domain_size(outer_count, middle_count, inner_count);
    if (middle_count == 1) {
        virCommitment all_one_zero = boundary_selector - virCommitment(Fr(1));
        zero.add_check(all_one_zero, domain_size);
        return;
    }

    const virCommitment transition =
        cube_boundary_selector_transition_expression(outer_count, middle_count, inner_count);
    virCommitment step_zero =
        boundary_selector - x_scale(boundary_selector, inverse_root_of_unity(domain_size)) - transition;
    zero.add_check(step_zero, domain_size);

    virCommitment anchor_zero = boundary_selector - virCommitment(Fr(1));
    zero.add_check(anchor_zero, 1);
}

void add_cube_middle_replication_constraints(zeroProtocol& zero,
                                             const virCommitment& lifted,
                                             const virCommitment& boundary_selector,
                                             size_t outer_count,
                                             size_t middle_count,
                                             size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cubeMiddleReplicationProtocol");

    const size_t row_count = outer_count * middle_count;
    const size_t domain_size = row_count * inner_count;
    const Fr previous_row_shift = inverse_root_of_unity(row_count);
    const virCommitment non_boundary_selector = virCommitment(Fr(1)) - boundary_selector;

    virCommitment step_zero = (lifted - x_scale(lifted, previous_row_shift)) * non_boundary_selector;
    zero.add_check(step_zero, domain_size);
}

void add_cube_lookup_row_index_constraints(zeroProtocol& zero,
                                           const virCommitment& row_index,
                                           size_t outer_count,
                                           size_t middle_count,
                                           size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cubeMiddleReplicationProtocol");

    const size_t table_row_count = outer_count * middle_count;
    const size_t query_domain_size = outer_count * inner_count;
    const Fr previous_row_shift = inverse_root_of_unity(outer_count);
    const Fr row_ratio = subgroup_label(table_row_count, middle_count);

    virCommitment row_transition_zero =
        row_index - scale_by_constant(x_scale(row_index, previous_row_shift), row_ratio);
    zero.add_check(row_transition_zero, query_domain_size);
}

size_t checked_exp_table_row_count(size_t quantization_bits)
{
    const size_t max_safe_quantization_bits = Fr::getBitSize() - 3;
    if (quantization_bits > max_safe_quantization_bits) {
        throw std::runtime_error(
            "cubeMiddleSoftmaxProtocol: quantization bits exceed the safely representable integer range of Fr");
    }
    if (quantization_bits >= std::numeric_limits<size_t>::digits - 1) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol: quantization bits are too large for size_t domain sizing");
    }
    return size_t(1) << quantization_bits;
}

void validate_softmax_parameters(size_t quantization_bits,
                                 size_t outer_count,
                                 size_t middle_count,
                                 size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cubeMiddleSoftmaxProtocol");
    checked_exp_table_row_count(quantization_bits);
}

void validate_softmax_parameters(const fixedPointEncoding& score_encoding,
                                 size_t outer_count,
                                 size_t middle_count,
                                 size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cubeMiddleSoftmaxProtocol");
    validate_fixed_point_encoding(score_encoding, "cubeMiddleSoftmaxProtocol");
    if (!score_encoding.signed_values) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol: fixed-point overload expects a signed encoding");
    }
    if (score_encoding.total_bits >= std::numeric_limits<size_t>::digits - 1) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol: score encoding is too wide for size_t domain sizing");
    }
}

Fr fr_from_cpp_int(const cpp_int& value)
{
    if (value < 0) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol: expected a non-negative exp value");
    }

    const std::string repr = value.convert_to<std::string>();
    Fr out;
    out.setStr(repr, 10);
    if (out.getStr(10) != repr) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol: exp value does not fit canonically in Fr");
    }
    return out;
}

cpp_int quantized_exp_output(size_t input, size_t quantization_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(quantization_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

poly quantized_exp_cube_polynomial(const std::vector<Fr>& score_evals,
                                   size_t quantization_bits,
                                   const char* context)
{
    const size_t table_row_count = checked_exp_table_row_count(quantization_bits);
    std::vector<Fr> exp_evals;
    exp_evals.reserve(score_evals.size());

    for (const Fr& score : score_evals) {
        const size_t input = fixed_point_size_t_from_fr(score, context);
        if (input >= table_row_count) {
            throw std::runtime_error(std::string(context) + ": input value exceeds the exp table range");
        }
        exp_evals.push_back(fr_from_cpp_int(quantized_exp_output(input, quantization_bits)));
    }

    return poly_from_evals(exp_evals);
}

poly quantized_exp_cube_polynomial(const std::vector<Fr>& score_evals,
                                   const fixedPointEncoding& score_encoding,
                                   const char* context)
{
    validate_fixed_point_encoding(score_encoding, context);
    if (!score_encoding.signed_values) {
        throw std::runtime_error(std::string(context) + ": expected a signed score encoding");
    }

    const size_t table_row_count = size_t(1) << score_encoding.total_bits;
    std::vector<Fr> exp_evals;
    exp_evals.reserve(score_evals.size());

    for (const Fr& score : score_evals) {
        // Each score evaluation is an encoded fixed-point witness. Decode it before
        // applying the quantized exp rule, but keep the exp outputs as unsigned field elements.
        const size_t encoded_score = fixed_point_size_t_from_fr(score, context);
        if (encoded_score >= table_row_count) {
            throw std::runtime_error(std::string(context) + ": encoded score exceeds the signed exp table range");
        }

        const std::int64_t scaled_score = decode_fixed_point_scaled_integer(encoded_score, score_encoding);
        exp_evals.push_back(fr_from_cpp_int(quantized_exp_output(scaled_score, score_encoding.fractional_bits)));
    }

    return poly_from_evals(exp_evals);
}

} // namespace

poly cube_middle_replication_polynomial(const poly& base,
                                        size_t outer_count,
                                        size_t middle_count,
                                        size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cube_middle_replication_polynomial");

    const std::vector<Fr> base_evals = base.evaluations_on_domain(matrix_domain_size(outer_count, inner_count));
    std::vector<Fr> lifted(cube_domain_size(outer_count, middle_count, inner_count), Fr(0));

    for (size_t outer = 0; outer < outer_count; ++outer) {
        for (size_t middle = 0; middle < middle_count; ++middle) {
            const size_t lifted_row = outer * middle_count + middle;
            for (size_t inner = 0; inner < inner_count; ++inner) {
                lifted[matrix_index(lifted_row, inner, inner_count)] =
                    base_evals[matrix_index(outer, inner, inner_count)];
            }
        }
    }

    return poly_from_evals(lifted);
}

void cubeMiddleReplicationVerifier::add_check(const virCommitment& base,
                                              const virCommitment& lifted,
                                              const cubeMiddleReplicationCheckRecord& record)
{
    bases.push_back(base);
    lifted_values.push_back(lifted);
    checks.push_back(record);
}

void cubeMiddleReplicationVerifier::add_direct_opening_check(Fr opened_value, Fr expected_value)
{
    direct_opening_checks.emplace_back(opened_value, expected_value);
}

bool cubeMiddleReplicationVerifier::finalize() const
{
    for (const auto& [opened_value, expected_value] : direct_opening_checks) {
        if (opened_value != expected_value) {
            return false;
        }
    }
    return true;
}

cubeMiddleReplicationCheckRecord cubeMiddleReplicationProver::add_check(size_t outer_count,
                                                                        size_t middle_count,
                                                                        size_t inner_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("cubeMiddleReplicationProver not setup");
    }

    cubeMiddleReplicationCheckRecord record;
    record.outer_count = outer_count;
    record.middle_count = middle_count;
    record.inner_count = inner_count;
    record.boundary_selector_commitment = matrix_row_subset_selector_commitment(
        *params->kzg, outer_count * middle_count, inner_count, block_start_rows(outer_count, middle_count));
    checks.push_back(record);
    return record;
}

void cubeMiddleReplicationProtocol::add_check(const virCommitment& base,
                                              const virCommitment& lifted,
                                              size_t outer_count,
                                              size_t middle_count,
                                              size_t inner_count)
{
    if (!is_setup) {
        throw std::runtime_error("cubeMiddleReplicationProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.table_lookup == nullptr) {
        throw std::runtime_error("cubeMiddleReplicationProtocol dependencies not setup");
    }

    validate_cube_shape(outer_count, middle_count, inner_count, "cubeMiddleReplicationProtocol");

    const cubeMiddleReplicationCheckRecord record = prover_.add_check(outer_count, middle_count, inner_count);
    verifier_.add_check(base, lifted, record);

    const virCommitment boundary_selector(record.boundary_selector_commitment);
    add_cube_boundary_selector_constraints(
        *params_.zero_check, boundary_selector, outer_count, middle_count, inner_count);
    add_cube_middle_replication_constraints(
        *params_.zero_check, lifted, boundary_selector, outer_count, middle_count, inner_count);

    const size_t row_count = outer_count * middle_count;
    const tableLookupWitness witness = tableLookupWitness::from_positions(row_count, block_start_rows(outer_count, middle_count));
    tableLookupCheckRecord lookup_record;
    if (cube_replication_failure_debug_enabled()) {
        const std::vector<Fr> base_evals =
            params_.kzg->evaluations(base, outer_count * inner_count, kzgProtocol::EvaluationMode::PreferEval);
        const std::vector<Fr> lifted_evals =
            params_.kzg->evaluations(lifted, row_count * inner_count, kzgProtocol::EvaluationMode::PreferEval);
        size_t mismatch_count = 0;
        std::vector<std::tuple<size_t, size_t, Fr, Fr>> first_mismatches;
        for (size_t outer = 0; outer < outer_count; ++outer) {
            const size_t lifted_row = outer * middle_count;
            for (size_t inner = 0; inner < inner_count; ++inner) {
                const size_t base_index = matrix_index(outer, inner, inner_count);
                const size_t lifted_index = matrix_index(lifted_row, inner, inner_count);
                if (base_evals[base_index] == lifted_evals[lifted_index]) {
                    continue;
                }
                ++mismatch_count;
                if (first_mismatches.size() < 8) {
                    first_mismatches.emplace_back(outer, inner, base_evals[base_index], lifted_evals[lifted_index]);
                }
            }
        }
        if (mismatch_count != 0) {
            std::cerr << "cube_replication_debug lookup_mismatch"
                      << " check_index=" << params_.check_count
                      << " outer_count=" << outer_count
                      << " middle_count=" << middle_count
                      << " inner_count=" << inner_count
                      << " mismatch_count=" << mismatch_count;
            for (const auto& [outer, inner, base_value, lifted_value] : first_mismatches) {
                std::cerr << " sample=" << outer << "," << inner
                          << ":base=" << base_value.getStr(10)
                          << ",lifted=" << lifted_value.getStr(10);
            }
            std::cerr << "\n";
        }
    }
    params_.table_lookup->add_check(base, lifted, outer_count, row_count, inner_count, witness, lookup_record);
    add_cube_lookup_row_index_constraints(
        *params_.zero_check, virCommitment(lookup_record.row_index_commitment), outer_count, middle_count, inner_count);

    const Fr first_row_label = subgroup_label(row_count, 0);
    const Fr opened_row_index = params_.kzg->open(lookup_record.row_index_commitment, Fr(1), kzgOpeningRootRole::FixedEvaluation);
    verifier_.add_direct_opening_check(opened_row_index, first_row_label);
    ++params_.check_count;
}

bool cubeMiddleReplicationProtocol::finalize()
{
    begin_finalize("cubeMiddleReplicationProtocol");
    return verifier_.finalize();
}

void cubeMiddleSoftmaxVerifier::add_check(const virCommitment& scores,
                                          const virCommitment& softmax,
                                          const cubeMiddleSoftmaxCheckRecord& record)
{
    this->scores.push_back(scores);
    softmax_values.push_back(softmax);
    checks.push_back(record);
}

bool cubeMiddleSoftmaxVerifier::finalize() const
{
    return true;
}

cubeMiddleSoftmaxCheckRecord cubeMiddleSoftmaxProver::add_check(const virCommitment& scores,
                                                                size_t quantization_bits,
                                                                size_t outer_count,
                                                                size_t middle_count,
                                                                size_t inner_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("cubeMiddleSoftmaxProver not setup");
    }

    const size_t full_domain_size = cube_domain_size(outer_count, middle_count, inner_count);
    const std::vector<Fr> score_evals = params->kzg->evaluations(scores, full_domain_size);
    const poly exp_poly = quantized_exp_cube_polynomial(score_evals, quantization_bits, "cubeMiddleSoftmaxProver");
    const poly denominator_poly = cube_middle_sum_polynomial(exp_poly, outer_count, middle_count, inner_count);
    const poly lifted_denominator_poly =
        cube_middle_replication_polynomial(denominator_poly, outer_count, middle_count, inner_count);

    cubeMiddleSoftmaxCheckRecord record;
    record.quantization_bits = quantization_bits;
    record.outer_count = outer_count;
    record.middle_count = middle_count;
    record.inner_count = inner_count;
    record.exp_commitment = params->kzg->commit(exp_poly, full_domain_size - 1);
    record.denominator_commitment = params->kzg->commit(denominator_poly, outer_count * inner_count - 1);
    record.lifted_denominator_commitment = params->kzg->commit(lifted_denominator_poly, full_domain_size - 1);
    checks.push_back(record);
    return record;
}

cubeMiddleSoftmaxCheckRecord cubeMiddleSoftmaxProver::add_check(const virCommitment& scores,
                                                                const fixedPointEncoding& score_encoding,
                                                                size_t outer_count,
                                                                size_t middle_count,
                                                                size_t inner_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("cubeMiddleSoftmaxProver not setup");
    }

    const size_t full_domain_size = cube_domain_size(outer_count, middle_count, inner_count);
    const std::vector<Fr> score_evals = params->kzg->evaluations(scores, full_domain_size);
    // Prover-side witness construction mirrors the unsigned path, except the exp witness
    // is derived from decoded signed scores rather than directly from raw non-negative inputs.
    const poly exp_poly =
        quantized_exp_cube_polynomial(score_evals, score_encoding, "cubeMiddleSoftmaxProver");
    const poly denominator_poly = cube_middle_sum_polynomial(exp_poly, outer_count, middle_count, inner_count);
    const poly lifted_denominator_poly =
        cube_middle_replication_polynomial(denominator_poly, outer_count, middle_count, inner_count);

    cubeMiddleSoftmaxCheckRecord record;
    record.quantization_bits = score_encoding.fractional_bits;
    record.outer_count = outer_count;
    record.middle_count = middle_count;
    record.inner_count = inner_count;
    record.uses_signed_score_encoding = true;
    record.score_encoding = score_encoding;
    record.exp_commitment = params->kzg->commit(exp_poly, full_domain_size - 1);
    record.denominator_commitment = params->kzg->commit(denominator_poly, outer_count * inner_count - 1);
    record.lifted_denominator_commitment = params->kzg->commit(lifted_denominator_poly, full_domain_size - 1);
    checks.push_back(record);
    return record;
}

cubeMiddleSoftmaxCheckRecord cubeMiddleSoftmaxProver::add_shifted_check(const virCommitment& shifts,
                                                                        const fixedPointEncoding& score_encoding,
                                                                        size_t outer_count,
                                                                        size_t middle_count,
                                                                        size_t inner_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("cubeMiddleSoftmaxProver not setup");
    }

    const size_t shift_domain_size = matrix_domain_size(outer_count, inner_count);
    const std::vector<Fr> shift_evals = params->kzg->evaluations(shifts, shift_domain_size);
    const poly shift_poly = poly_from_evals(shift_evals);
    const poly lifted_shift_poly = cube_middle_replication_polynomial(shift_poly, outer_count, middle_count, inner_count);

    cubeMiddleSoftmaxCheckRecord record;
    record.quantization_bits = score_encoding.fractional_bits;
    record.outer_count = outer_count;
    record.middle_count = middle_count;
    record.inner_count = inner_count;
    record.uses_signed_score_encoding = true;
    record.uses_explicit_shift = true;
    record.score_encoding = score_encoding;
    record.lifted_shift_commitment = params->kzg->commit(
        lifted_shift_poly, cube_domain_size(outer_count, middle_count, inner_count) - 1);
    checks.push_back(record);
    return record;
}

void cubeMiddleSoftmaxProtocol::add_check(const virCommitment& scores,
                                          const virCommitment& softmax,
                                          size_t quantization_bits,
                                          size_t outer_count,
                                          size_t middle_count,
                                          size_t inner_count)
{
    if (!is_setup) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.exp_check == nullptr ||
        params_.cube_sum == nullptr || params_.replication == nullptr) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol dependencies not setup");
    }

    validate_softmax_parameters(quantization_bits, outer_count, middle_count, inner_count);

    scores_.push_back(scores);
    shifted_scores_.push_back(virCommitment(Fr(0)));
    shifts_.push_back(virCommitment(Fr(0)));
    shift_selectors_.push_back(virCommitment(Fr(0)));
    softmax_values_.push_back(softmax);
    cubeMiddleSoftmaxCheckSpec spec;
    spec.quantization_bits = quantization_bits;
    spec.outer_count = outer_count;
    spec.middle_count = middle_count;
    spec.inner_count = inner_count;
    checks_.push_back(spec);
}

void cubeMiddleSoftmaxProtocol::add_check(const virCommitment& scores,
                                          const virCommitment& softmax,
                                          const fixedPointEncoding& score_encoding,
                                          size_t outer_count,
                                          size_t middle_count,
                                          size_t inner_count)
{
    if (!is_setup) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.signed_range_check == nullptr ||
        params_.exp_check == nullptr || params_.cube_sum == nullptr || params_.replication == nullptr) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol dependencies not setup");
    }

    validate_softmax_parameters(score_encoding, outer_count, middle_count, inner_count);

    scores_.push_back(scores);
    shifted_scores_.push_back(virCommitment(Fr(0)));
    shifts_.push_back(virCommitment(Fr(0)));
    shift_selectors_.push_back(virCommitment(Fr(0)));
    softmax_values_.push_back(softmax);
    cubeMiddleSoftmaxCheckSpec spec;
    spec.quantization_bits = score_encoding.fractional_bits;
    spec.outer_count = outer_count;
    spec.middle_count = middle_count;
    spec.inner_count = inner_count;
    spec.uses_signed_score_encoding = true;
    spec.score_encoding = score_encoding;
    checks_.push_back(spec);
}

void cubeMiddleSoftmaxProtocol::add_shifted_check(const virCommitment& scores,
                                                  const virCommitment& shifted_scores,
                                                  const virCommitment& shifts,
                                                  const virCommitment& softmax,
                                                  const fixedPointEncoding& score_encoding,
                                                  size_t outer_count,
                                                  size_t middle_count,
                                                  size_t inner_count)
{
    if (!is_setup) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.signed_range_check == nullptr ||
        params_.exp_check == nullptr || params_.cube_sum == nullptr || params_.replication == nullptr) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol dependencies not setup");
    }

    validate_softmax_parameters(score_encoding, outer_count, middle_count, inner_count);

    scores_.push_back(scores);
    shifted_scores_.push_back(shifted_scores);
    shifts_.push_back(shifts);
    shift_selectors_.push_back(virCommitment(Fr(0)));
    softmax_values_.push_back(softmax);
    cubeMiddleSoftmaxCheckSpec spec;
    spec.quantization_bits = score_encoding.fractional_bits;
    spec.outer_count = outer_count;
    spec.middle_count = middle_count;
    spec.inner_count = inner_count;
    spec.uses_signed_score_encoding = true;
    spec.uses_explicit_shift = true;
    spec.score_encoding = score_encoding;
    checks_.push_back(spec);
}

void cubeMiddleSoftmaxProtocol::add_row_max_shifted_check(const virCommitment& scores,
                                                          const virCommitment& shifted_scores,
                                                          const virCommitment& shifts,
                                                          const virCommitment& shift_selector,
                                                          const virCommitment& softmax,
                                                          const fixedPointEncoding& score_encoding,
                                                          size_t outer_count,
                                                          size_t middle_count,
                                                          size_t inner_count)
{
    if (!is_setup) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.signed_range_check == nullptr ||
        params_.cube_middle_argmax == nullptr || params_.exp_check == nullptr || params_.cube_sum == nullptr ||
        params_.replication == nullptr) {
        throw std::runtime_error("cubeMiddleSoftmaxProtocol dependencies not setup");
    }

    validate_softmax_parameters(score_encoding, outer_count, middle_count, inner_count);

    scores_.push_back(scores);
    shifted_scores_.push_back(shifted_scores);
    shifts_.push_back(shifts);
    shift_selectors_.push_back(shift_selector);
    softmax_values_.push_back(softmax);
    cubeMiddleSoftmaxCheckSpec spec;
    spec.quantization_bits = score_encoding.fractional_bits;
    spec.outer_count = outer_count;
    spec.middle_count = middle_count;
    spec.inner_count = inner_count;
    spec.uses_signed_score_encoding = true;
    spec.uses_explicit_shift = true;
    spec.uses_row_max_shift = true;
    spec.score_encoding = score_encoding;
    checks_.push_back(spec);
}

bool cubeMiddleSoftmaxProtocol::finalize()
{
    begin_finalize("cubeMiddleSoftmaxProtocol");

    if (checks_.empty()) {
        return true;
    }

    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    for (size_t i = 0; i < checks_.size(); ++i) {
        const cubeMiddleSoftmaxCheckSpec& spec = checks_[i];
        const size_t full_domain_size = cube_domain_size(spec.outer_count, spec.middle_count, spec.inner_count);
        cubeMiddleSoftmaxCheckRecord record;

        if (proof_session && proof_session->mode() == ProofMode::Verify) {
            record.quantization_bits = spec.quantization_bits;
            record.outer_count = spec.outer_count;
            record.middle_count = spec.middle_count;
            record.inner_count = spec.inner_count;
            record.uses_signed_score_encoding = spec.uses_signed_score_encoding;
            record.uses_explicit_shift = spec.uses_explicit_shift;
            record.uses_row_max_shift = spec.uses_row_max_shift;
            record.score_encoding = spec.score_encoding;

            const Group1 exp_group = proof_session->transport_group1({});
            record.exp_commitment = params_.kzg->adopt_commitment(exp_group);
            params_.kzg->prove_degree(record.exp_commitment, full_domain_size - 1);

            const Group1 denominator_group = proof_session->transport_group1({});
            record.denominator_commitment = params_.kzg->adopt_commitment(denominator_group);
            params_.kzg->prove_degree(record.denominator_commitment, spec.outer_count * spec.inner_count - 1);

            const Group1 lifted_denominator_group = proof_session->transport_group1({});
            record.lifted_denominator_commitment = params_.kzg->adopt_commitment(lifted_denominator_group);
            params_.kzg->prove_degree(record.lifted_denominator_commitment, full_domain_size - 1);

            if (spec.uses_explicit_shift) {
                const Group1 lifted_shift_group = proof_session->transport_group1({});
                record.lifted_shift_commitment = params_.kzg->adopt_commitment(lifted_shift_group);
                params_.kzg->prove_degree(record.lifted_shift_commitment, full_domain_size - 1);
            }
        } else {
            if (spec.uses_explicit_shift) {
                record = prover_.add_check(shifted_scores_[i],
                                           spec.score_encoding,
                                           spec.outer_count,
                                           spec.middle_count,
                                           spec.inner_count);
            } else if (spec.uses_signed_score_encoding) {
                record = prover_.add_check(scores_[i],
                                           spec.score_encoding,
                                           spec.outer_count,
                                           spec.middle_count,
                                           spec.inner_count);
            } else {
                record = prover_.add_check(
                    scores_[i], spec.quantization_bits, spec.outer_count, spec.middle_count, spec.inner_count);
            }
            if (spec.uses_explicit_shift) {
                const cubeMiddleSoftmaxCheckRecord shift_record = prover_.add_shifted_check(
                    shifts_[i], spec.score_encoding, spec.outer_count, spec.middle_count, spec.inner_count);
                record.uses_explicit_shift = true;
                record.uses_row_max_shift = spec.uses_row_max_shift;
                record.lifted_shift_commitment = shift_record.lifted_shift_commitment;
            }
            if (proof_session && proof_session->mode() == ProofMode::Prove) {
                const Group1 exp_group = proof_session->transport_group1([&] { return record.exp_commitment.com; });
                if (exp_group != record.exp_commitment.com) {
                    throw std::runtime_error("cubeMiddleSoftmaxProtocol: transported exp commitment mismatch");
                }

                const Group1 denominator_group =
                    proof_session->transport_group1([&] { return record.denominator_commitment.com; });
                if (denominator_group != record.denominator_commitment.com) {
                    throw std::runtime_error("cubeMiddleSoftmaxProtocol: transported denominator commitment mismatch");
                }

                const Group1 lifted_denominator_group =
                    proof_session->transport_group1([&] { return record.lifted_denominator_commitment.com; });
                if (lifted_denominator_group != record.lifted_denominator_commitment.com) {
                    throw std::runtime_error(
                        "cubeMiddleSoftmaxProtocol: transported lifted denominator commitment mismatch");
                }

                if (spec.uses_explicit_shift) {
                    const Group1 lifted_shift_group =
                        proof_session->transport_group1([&] { return record.lifted_shift_commitment.com; });
                    if (lifted_shift_group != record.lifted_shift_commitment.com) {
                        throw std::runtime_error("cubeMiddleSoftmaxProtocol: transported lifted shift commitment mismatch");
                    }
                }
            }
        }

        verifier_.add_check(scores_[i], softmax_values_[i], record);

        const virCommitment exp_values(record.exp_commitment);
        const virCommitment denominator(record.denominator_commitment);
        const virCommitment lifted_denominator(record.lifted_denominator_commitment);

        if (spec.uses_signed_score_encoding) {
            params_.signed_range_check->add_check(scores_[i], spec.score_encoding, full_domain_size);
        }

        if (spec.uses_explicit_shift) {
            const size_t shift_domain_size = matrix_domain_size(spec.outer_count, spec.inner_count);
            const virCommitment lifted_shift(record.lifted_shift_commitment);
            params_.signed_range_check->add_check(shifts_[i], spec.score_encoding, shift_domain_size);
            params_.signed_range_check->add_check(shifted_scores_[i], spec.score_encoding, full_domain_size);
            params_.replication->add_check(
                shifts_[i], lifted_shift, spec.outer_count, spec.middle_count, spec.inner_count);

            virCommitment shift_relation_zero =
                shifted_scores_[i] - scores_[i] + lifted_shift - virCommitment(fixed_point_offset_fr(spec.score_encoding));
            params_.zero_check->add_check(shift_relation_zero, full_domain_size);

            if (spec.uses_row_max_shift) {
                params_.cube_middle_argmax->add_check(scores_[i],
                                                      shifts_[i],
                                                      shift_selectors_[i],
                                                      spec.outer_count,
                                                      spec.middle_count,
                                                      spec.inner_count,
                                                      spec.score_encoding.total_bits);
            }
        }

        if (spec.uses_signed_score_encoding) {
            if (spec.uses_explicit_shift) {
                params_.exp_check->add_non_positive_check(
                    shifted_scores_[i], exp_values, spec.score_encoding, full_domain_size);
            } else {
                params_.exp_check->add_check(scores_[i], exp_values, spec.score_encoding, full_domain_size);
            }
        } else {
            params_.exp_check->add_check(scores_[i], exp_values, spec.quantization_bits, full_domain_size);
        }
        params_.cube_sum->add_check(
            exp_values, denominator, spec.outer_count, spec.middle_count, spec.inner_count);
        params_.replication->add_check(
            denominator, lifted_denominator, spec.outer_count, spec.middle_count, spec.inner_count);

        virCommitment softmax_zero = softmax_values_[i] * lifted_denominator - exp_values;
        params_.zero_check->add_check(softmax_zero, full_domain_size);
    }

    return verifier_.finalize();
}
