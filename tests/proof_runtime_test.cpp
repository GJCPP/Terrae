#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "test_kzg_srs.hpp"
#include "proof_runtime.hpp"
#include "../examples/forest_training_demo_data.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <vector>

using namespace mcl::bn;

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

poly poly_from_domain_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

poly matrix_poly_from_rows(const std::vector<std::vector<int>>& rows)
{
    std::vector<Fr> evals;
    for (const auto& row : rows) {
        for (int value : row) {
            evals.push_back(fr_from_int(value));
        }
    }
    return poly_from_evals(evals);
}

poly fixed_point_encoded_matrix_poly_from_rows(const std::vector<std::vector<int>>& rows,
                                               const fixedPointEncoding& encoding)
{
    std::vector<Fr> evals;
    for (const auto& row : rows) {
        for (int value : row) {
            evals.push_back(fixed_point_fr_from_size_t(encode_fixed_point_scaled_integer(value, encoding)));
        }
    }
    return poly_from_evals(evals);
}

std::vector<std::vector<int>> square_matrix(const std::vector<std::vector<int>>& matrix)
{
    std::vector<std::vector<int>> out = matrix;
    for (auto& row : out) {
        for (int& value : row) {
            value *= value;
        }
    }
    return out;
}

std::vector<std::vector<int>> add_matrices(const std::vector<std::vector<int>>& lhs,
                                           const std::vector<std::vector<int>>& rhs)
{
    std::vector<std::vector<int>> out = lhs;
    for (size_t row = 0; row < out.size(); ++row) {
        for (size_t col = 0; col < out[row].size(); ++col) {
            out[row][col] += rhs[row][col];
        }
    }
    return out;
}

std::vector<std::vector<int>> gain_denominator_matrix(const std::vector<std::vector<int>>& left_hessians,
                                                      const std::vector<std::vector<int>>& right_hessians,
                                                      int lambda)
{
    std::vector<std::vector<int>> out = left_hessians;
    for (size_t row = 0; row < out.size(); ++row) {
        for (size_t col = 0; col < out[row].size(); ++col) {
            const int left_den = left_hessians[row][col] + lambda;
            const int right_den = right_hessians[row][col] + lambda;
            const int parent_den = left_hessians[row][col] + right_hessians[row][col] + lambda;
            out[row][col] = left_den * right_den * parent_den;
        }
    }
    return out;
}

std::vector<std::vector<int>> gain_numerator_matrix(const std::vector<std::vector<int>>& left_gradients,
                                                    const std::vector<std::vector<int>>& right_gradients,
                                                    const std::vector<std::vector<int>>& left_hessians,
                                                    const std::vector<std::vector<int>>& right_hessians,
                                                    int lambda,
                                                    int split_penalty)
{
    std::vector<std::vector<int>> out = left_gradients;
    for (size_t row = 0; row < out.size(); ++row) {
        for (size_t col = 0; col < out[row].size(); ++col) {
            const int left_gradient = left_gradients[row][col];
            const int right_gradient = right_gradients[row][col];
            const int parent_gradient = left_gradient + right_gradient;
            const int left_den = left_hessians[row][col] + lambda;
            const int right_den = right_hessians[row][col] + lambda;
            const int parent_den = left_hessians[row][col] + right_hessians[row][col] + lambda;
            const int gain_denominator = left_den * right_den * parent_den;

            out[row][col] = left_gradient * left_gradient * right_den * parent_den +
                            right_gradient * right_gradient * left_den * parent_den -
                            parent_gradient * parent_gradient * left_den * right_den -
                            split_penalty * gain_denominator;
        }
    }
    return out;
}

int floor_div_scaled(int numerator, int denominator, int scale)
{
    const int scaled = numerator * scale;
    int quotient = scaled / denominator;
    const int remainder = scaled % denominator;
    if (remainder != 0 && ((remainder > 0) != (denominator > 0))) {
        --quotient;
    }
    return quotient;
}

std::vector<std::vector<int>> quantized_gain_matrix(const std::vector<std::vector<int>>& numerators,
                                                    const std::vector<std::vector<int>>& denominators,
                                                    int fractional_bits)
{
    const int scale = 1 << fractional_bits;
    std::vector<std::vector<int>> out = numerators;
    for (size_t row = 0; row < out.size(); ++row) {
        for (size_t col = 0; col < out[row].size(); ++col) {
            out[row][col] = floor_div_scaled(numerators[row][col], denominators[row][col], scale);
        }
    }
    return out;
}

std::vector<std::vector<int>> quantized_remainder_matrix(const std::vector<std::vector<int>>& numerators,
                                                         const std::vector<std::vector<int>>& denominators,
                                                         int fractional_bits)
{
    const int scale = 1 << fractional_bits;
    std::vector<std::vector<int>> out = numerators;
    for (size_t row = 0; row < out.size(); ++row) {
        for (size_t col = 0; col < out[row].size(); ++col) {
            const int quotient = floor_div_scaled(numerators[row][col], denominators[row][col], scale);
            out[row][col] = numerators[row][col] * scale - denominators[row][col] * quotient;
        }
    }
    return out;
}

poly cube_poly_from_slices(const std::vector<std::vector<std::vector<int>>>& cube)
{
    std::vector<Fr> evals;
    for (const auto& outer : cube) {
        for (const auto& middle : outer) {
            for (int value : middle) {
                evals.push_back(fr_from_int(value));
            }
        }
    }
    return poly_from_evals(evals);
}

size_t quantized_exp_output(size_t input, size_t quantization_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(quantization_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>().convert_to<size_t>();
}

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

Fr fr_from_cpp_int_checked(const cpp_int& value)
{
    const std::string repr = value.convert_to<std::string>();
    Fr out;
    out.setStr(repr, 10);
    EXPECT_EQ(out.getStr(10), repr);
    return out;
}

poly cube_middle_softmax_polynomial_for_test(const poly& scores,
                                             size_t quantization_bits,
                                             size_t outer_count,
                                             size_t middle_count,
                                             size_t inner_count)
{
    const std::vector<Fr> score_evals = scores.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> exp_evals;
    exp_evals.reserve(score_evals.size());
    for (const Fr& score : score_evals) {
        exp_evals.push_back(fr_from_int(static_cast<int>(
            quantized_exp_output(static_cast<size_t>(std::stoull(score.getStr(10))), quantization_bits))));
    }

    const poly exp_values = poly_from_evals(exp_evals);
    const poly denominator = cube_middle_sum_polynomial(exp_values, outer_count, middle_count, inner_count);
    const poly lifted_denominator =
        cube_middle_replication_polynomial(denominator, outer_count, middle_count, inner_count);

    const std::vector<Fr> lifted_denominator_evals =
        lifted_denominator.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> softmax_evals(exp_evals.size(), Fr(0));
    for (size_t i = 0; i < exp_evals.size(); ++i) {
        Fr inverse;
        Fr::inv(inverse, lifted_denominator_evals[i]);
        softmax_evals[i] = exp_evals[i] * inverse;
    }
    return poly_from_evals(softmax_evals);
}

poly exp_outputs_for_signed_scores(const std::vector<std::int64_t>& scaled_scores, const fixedPointEncoding& encoding)
{
    std::vector<Fr> evals;
    evals.reserve(scaled_scores.size());
    for (std::int64_t scaled_score : scaled_scores) {
        evals.push_back(fr_from_cpp_int_checked(quantized_exp_output(scaled_score, encoding.fractional_bits)));
    }
    return poly_from_evals(evals);
}

poly cube_middle_softmax_polynomial_for_test(const poly& encoded_scores,
                                             const fixedPointEncoding& encoding,
                                             size_t outer_count,
                                             size_t middle_count,
                                             size_t inner_count)
{
    const std::vector<Fr> score_evals = encoded_scores.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> exp_evals;
    exp_evals.reserve(score_evals.size());
    for (const Fr& encoded_score : score_evals) {
        const size_t encoded = fixed_point_size_t_from_fr(encoded_score, "proof_runtime_test");
        const std::int64_t scaled_score = decode_fixed_point_scaled_integer(encoded, encoding);
        exp_evals.push_back(fr_from_cpp_int_checked(quantized_exp_output(scaled_score, encoding.fractional_bits)));
    }

    const poly exp_values = poly_from_evals(exp_evals);
    const poly denominator = cube_middle_sum_polynomial(exp_values, outer_count, middle_count, inner_count);
    const poly lifted_denominator =
        cube_middle_replication_polynomial(denominator, outer_count, middle_count, inner_count);

    const std::vector<Fr> lifted_denominator_evals =
        lifted_denominator.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> softmax_evals(exp_evals.size(), Fr(0));
    for (size_t i = 0; i < exp_evals.size(); ++i) {
        Fr inverse;
        Fr::inv(inverse, lifted_denominator_evals[i]);
        softmax_evals[i] = exp_evals[i] * inverse;
    }
    return poly_from_evals(softmax_evals);
}

poly shifted_scores_polynomial_for_test(const poly& encoded_scores,
                                        const poly& encoded_shifts,
                                        const fixedPointEncoding& encoding,
                                        size_t outer_count,
                                        size_t middle_count,
                                        size_t inner_count)
{
    const std::vector<Fr> score_evals =
        encoded_scores.evaluations_on_domain(outer_count * middle_count * inner_count);
    const poly lifted_shifts = cube_middle_replication_polynomial(encoded_shifts, outer_count, middle_count, inner_count);
    const std::vector<Fr> lifted_shift_evals =
        lifted_shifts.evaluations_on_domain(outer_count * middle_count * inner_count);

    std::vector<Fr> shifted_evals;
    shifted_evals.reserve(score_evals.size());
    const Fr offset = fixed_point_offset_fr(encoding);
    for (size_t i = 0; i < score_evals.size(); ++i) {
        shifted_evals.push_back(score_evals[i] - lifted_shift_evals[i] + offset);
    }
    return poly_from_evals(shifted_evals);
}

poly x_power_poly_for_test(const poly& base, size_t exponent)
{
    if (exponent == 0) {
        poly constant;
        constant.add_term(0, base.eval(Fr(1)));
        return constant;
    }

    std::vector<Fr> coeffs = base.coefficients();
    if (coeffs.empty()) {
        return poly();
    }

    std::vector<Fr> composed((coeffs.size() - 1) * exponent + 1, Fr(0));
    for (size_t i = 0; i < coeffs.size(); ++i) {
        composed[i * exponent] = coeffs[i];
    }
    return poly_from_coeff(composed);
}

poly pseudo_residual_poly_from_labels_and_softmax(const poly& base_labels,
                                                  const poly& softmax,
                                                  size_t iteration_count,
                                                  size_t class_count,
                                                  size_t query_count)
{
    const std::vector<Fr> label_evals =
        x_power_poly_for_test(base_labels, iteration_count)
            .evaluations_on_domain(iteration_count * class_count * query_count);
    const std::vector<Fr> softmax_evals = softmax.evaluations_on_domain(iteration_count * class_count * query_count);
    std::vector<Fr> residual_evals(softmax_evals.size(), Fr(0));
    for (size_t i = 0; i < residual_evals.size(); ++i) {
        residual_evals[i] = softmax_evals[i] - label_evals[i];
    }
    return poly_from_evals(residual_evals);
}

poly diagonal_hessian_poly_from_softmax(const poly& softmax,
                                        size_t iteration_count,
                                        size_t class_count,
                                        size_t query_count)
{
    const std::vector<Fr> softmax_evals = softmax.evaluations_on_domain(iteration_count * class_count * query_count);
    std::vector<Fr> hessian_evals(softmax_evals.size(), Fr(0));
    for (size_t i = 0; i < hessian_evals.size(); ++i) {
        hessian_evals[i] = softmax_evals[i] * (Fr(1) - softmax_evals[i]);
    }
    return poly_from_evals(hessian_evals);
}

poly fixed_point_encoded_cube_poly_from_slices(const std::vector<std::vector<std::vector<int>>>& cube,
                                               const fixedPointEncoding& encoding)
{
    std::vector<Fr> evals;
    for (const auto& outer : cube) {
        for (const auto& middle : outer) {
            for (int value : middle) {
                evals.push_back(fixed_point_fr_from_size_t(encode_fixed_point_scaled_integer(value, encoding)));
            }
        }
    }
    return poly_from_evals(evals);
}

struct FixedPointBoostingWitnessPolys {
    poly cumulative_scores;
    poly encoded_cumulative_scores;
    poly softmax_shifts;
    poly shift_selector;
    poly lifted_softmax_shifts;
    poly encoded_shifted_scores;
    poly softmax_exp_values;
    poly softmax_denominators;
    poly lifted_softmax_denominators;
    poly softmax_probabilities;
    poly softmax_remainders;
    poly pseudo_residuals;
    poly encoded_diagonal_hessians;
    poly diagonal_hessian_remainders;
    poly diagonal_hessians;
    size_t hessian_bit_bound;
};

FixedPointBoostingWitnessPolys derive_fixed_point_boosting_witness_for_test(
    const std::vector<std::vector<int>>& tree_score_rows,
    const std::vector<std::vector<int>>& base_label_rows,
    const fixedPointEncoding& score_encoding,
    const fixedPointEncoding& softmax_encoding)
{
    const size_t iteration_count = tree_score_rows.size();
    const size_t row_width = tree_score_rows.front().size();
    const size_t class_count = base_label_rows.size();
    const size_t query_count = row_width / class_count;
    const int softmax_scale = 1 << softmax_encoding.fractional_bits;
    const std::vector<Fr> lifted_label_evals =
        x_power_poly_for_test(matrix_poly_from_rows(base_label_rows), iteration_count)
            .evaluations_on_domain(iteration_count * row_width);

    std::vector<std::vector<int>> cumulative_rows = tree_score_rows;
    for (size_t iteration = 1; iteration < iteration_count; ++iteration) {
        for (size_t column = 0; column < row_width; ++column) {
            cumulative_rows[iteration][column] += cumulative_rows[iteration - 1][column];
        }
    }

    std::vector<std::vector<int>> shift_rows(iteration_count, std::vector<int>(query_count, 0));
    std::vector<std::vector<std::vector<int>>> shift_selector(
        iteration_count, std::vector<std::vector<int>>(class_count, std::vector<int>(query_count, 0)));
    std::vector<std::vector<int>> shifted_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> exp_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> denominator_rows(iteration_count, std::vector<int>(query_count, 0));
    std::vector<std::vector<int>> lifted_denominator_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> probability_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> probability_remainder_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> pseudo_residual_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> hessian_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> hessian_remainder_rows(iteration_count, std::vector<int>(row_width, 0));

    auto floor_division = [](int numerator, int denominator) -> std::pair<int, int> {
        int quotient = numerator / denominator;
        int remainder = numerator % denominator;
        if (remainder < 0) {
            --quotient;
            remainder += denominator;
        }
        return {quotient, remainder};
    };

    int max_abs_hessian = 0;
    for (size_t iteration = 0; iteration < iteration_count; ++iteration) {
        for (size_t query = 0; query < query_count; ++query) {
            size_t best_class = 0;
            int best_score = cumulative_rows[iteration][query];
            for (size_t cls = 1; cls < class_count; ++cls) {
                const int candidate = cumulative_rows[iteration][cls * query_count + query];
                if (candidate > best_score) {
                    best_class = cls;
                    best_score = candidate;
                }
            }
            shift_rows[iteration][query] = best_score;
            shift_selector[iteration][best_class][query] = 1;

            int denominator = 0;
            for (size_t cls = 0; cls < class_count; ++cls) {
                const size_t column = cls * query_count + query;
                const int shifted = cumulative_rows[iteration][column] - best_score;
                shifted_rows[iteration][column] = shifted;
                const int exp_value =
                    quantized_exp_output(static_cast<std::int64_t>(shifted), softmax_encoding.fractional_bits)
                        .convert_to<int>();
                exp_rows[iteration][column] = exp_value;
                denominator += exp_value;
            }
            denominator_rows[iteration][query] = denominator;

            for (size_t cls = 0; cls < class_count; ++cls) {
                const size_t column = cls * query_count + query;
                lifted_denominator_rows[iteration][column] = denominator;
                const auto [probability_scaled, probability_remainder] =
                    floor_division(exp_rows[iteration][column] * softmax_scale, denominator);
                probability_rows[iteration][column] = probability_scaled;
                probability_remainder_rows[iteration][column] = probability_remainder;

                const size_t flat_index = iteration * row_width + column;
                const int label_scaled =
                    canonical_signed_int64_from_fr(lifted_label_evals[flat_index], "proof_runtime_test lifted_labels") *
                    softmax_scale;
                pseudo_residual_rows[iteration][column] = probability_scaled - label_scaled;

                const auto [hessian_scaled, hessian_remainder] =
                    floor_division(probability_scaled * (softmax_scale - probability_scaled), softmax_scale);
                hessian_rows[iteration][column] = hessian_scaled;
                hessian_remainder_rows[iteration][column] = hessian_remainder;
                max_abs_hessian = std::max(max_abs_hessian, std::abs(hessian_scaled));
            }
        }
    }

    size_t hessian_bit_bound = 1;
    while ((size_t(1) << hessian_bit_bound) <= static_cast<size_t>(max_abs_hessian)) {
        ++hessian_bit_bound;
    }
    const fixedPointEncoding hessian_encoding = fixedPointEncoding::signed_offset_binary(
        std::max(hessian_bit_bound + 1, softmax_encoding.fractional_bits + 1), softmax_encoding.fractional_bits);

    std::vector<std::vector<int>> lifted_shift_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> encoded_shifted_rows(iteration_count, std::vector<int>(row_width, 0));
    std::vector<std::vector<int>> encoded_hessian_rows(iteration_count, std::vector<int>(row_width, 0));
    for (size_t iteration = 0; iteration < iteration_count; ++iteration) {
        for (size_t cls = 0; cls < class_count; ++cls) {
            for (size_t query = 0; query < query_count; ++query) {
                const size_t column = cls * query_count + query;
                lifted_shift_rows[iteration][column] = shift_rows[iteration][query];
                encoded_shifted_rows[iteration][column] = shifted_rows[iteration][column];
                encoded_hessian_rows[iteration][column] = hessian_rows[iteration][column];
            }
        }
    }

    FixedPointBoostingWitnessPolys out{
        matrix_poly_from_rows(cumulative_rows),
        fixed_point_encoded_matrix_poly_from_rows(cumulative_rows, score_encoding),
        fixed_point_encoded_matrix_poly_from_rows(shift_rows, score_encoding),
        cube_poly_from_slices(shift_selector),
        fixed_point_encoded_matrix_poly_from_rows(lifted_shift_rows, score_encoding),
        fixed_point_encoded_matrix_poly_from_rows(encoded_shifted_rows, score_encoding),
        matrix_poly_from_rows(exp_rows),
        matrix_poly_from_rows(denominator_rows),
        matrix_poly_from_rows(lifted_denominator_rows),
        fixed_point_encoded_matrix_poly_from_rows(probability_rows, softmax_encoding),
        matrix_poly_from_rows(probability_remainder_rows),
        matrix_poly_from_rows(pseudo_residual_rows),
        fixed_point_encoded_matrix_poly_from_rows(encoded_hessian_rows, hessian_encoding),
        matrix_poly_from_rows(hessian_remainder_rows),
        matrix_poly_from_rows(hessian_rows),
        hessian_bit_bound};
    return out;
}

poly histogram_poly_from_inputs_for_test(const std::vector<int>& indices,
                                         const std::vector<int>& values,
                                         size_t bucket_count)
{
    std::vector<Fr> histogram(bucket_count, Fr(0));
    for (size_t i = 0; i < indices.size(); ++i) {
        Fr::add(histogram[static_cast<size_t>(indices[i])], histogram[static_cast<size_t>(indices[i])], fr_from_int(values[i]));
    }
    return poly_from_evals(histogram);
}

bool run_lookup_program(ProofRuntime& runtime,
                        const ProtocolPolynomialInput& query,
                        const ProtocolPolynomialInput& table,
                        size_t query_domain_size,
                        size_t table_domain_size)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.lookup().add_check(
            program.bind_virtual(query), program.bind_virtual(table), query_domain_size, table_domain_size);
    });
}

bool run_cq_lookup_program(ProofRuntime& runtime,
                           const ProtocolPolynomialInput& query,
                           const CqLookupPublicInput& table,
                           size_t query_domain_size,
                           size_t table_domain_size)
{
    return std::visit(
        [&](const auto& table_input) {
            using TableInput = std::decay_t<decltype(table_input)>;
            return run_proof_program(runtime, [&](ProofProgram& program) {
                auto& context = [&]() -> auto& {
                    if constexpr (std::is_same_v<TableInput, PublicPolynomialInput>) {
                        return program.create_cq_lookup_context(table_input, table_domain_size);
                    } else {
                        return program.create_cq_lookup_context(table_input);
                    }
                }();
                context.add_check(query, query_domain_size);
            });
        },
        table);
}

bool run_cq_table_lookup_program(ProofRuntime& runtime,
                                 const ProtocolPolynomialInput& query,
                                 const CqTableLookupPublicInput& table,
                                 size_t query_row_count,
                                 size_t table_row_count,
                                 size_t column_count)
{
    return std::visit(
        [&](const auto& table_input) {
            using TableInput = std::decay_t<decltype(table_input)>;
            return run_proof_program(runtime, [&](ProofProgram& program) {
                auto& context = [&]() -> auto& {
                    if constexpr (std::is_same_v<TableInput, PublicPolynomialInput>) {
                        return program.create_cq_table_lookup_context(table_input, table_row_count, column_count);
                    } else {
                        return program.create_cq_table_lookup_context(table_input);
                    }
                }();
                context.add_check(query, query_row_count);
            });
        },
        table);
}

bool run_cq_concatenated_table_lookup_program(ProofRuntime& runtime,
                                              const std::vector<ProtocolPolynomialInput>& queries,
                                              const cqConcatenatedTableLookupPublicTable& table,
                                              size_t query_row_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        auto& context = program.create_cq_concatenated_table_lookup_context(table);
        context.add_check(queries, query_row_count);
    });
}

bool run_forest_row_fetch_program(ProofRuntime& runtime,
                                  const forestRowFetchPublicTable& table,
                                  const ProtocolPolynomialInput& tree_indices,
                                  const ProtocolPolynomialInput& lower_bounds,
                                  const ProtocolPolynomialInput& upper_bounds,
                                  const ProtocolPolynomialInput& scores,
                                  const std::optional<ProtocolPolynomialInput>& leaf_indices,
                                  size_t query_row_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        auto& context = program.create_forest_row_fetch_context(table);
        if (leaf_indices.has_value()) {
            context.add_check(tree_indices, lower_bounds, upper_bounds, scores, *leaf_indices, query_row_count);
            return;
        }
        context.add_check(tree_indices, lower_bounds, upper_bounds, scores, query_row_count);
    });
}

bool run_box_check_program(ProofRuntime& runtime,
                           const ProtocolPolynomialInput& values,
                           const ProtocolPolynomialInput& lower_bounds,
                           const ProtocolPolynomialInput& upper_bounds,
                           size_t row_count,
                           size_t column_count,
                           size_t bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.box_check().add_check(program.bind_virtual(values),
                                      program.bind_virtual(lower_bounds),
                                      program.bind_virtual(upper_bounds),
                                      row_count,
                                      column_count,
                                      bit_bound);
    });
}

bool run_forest_inference_program(ProofRuntime& runtime,
                                  const forestInferencePublicTable& table,
                                  const ProtocolPolynomialInput& tree_indices,
                                  const ProtocolPolynomialInput& lifted_inputs,
                                  const ProtocolPolynomialInput& lower_bounds,
                                  const ProtocolPolynomialInput& upper_bounds,
                                  const ProtocolPolynomialInput& scores,
                                  const std::optional<ProtocolPolynomialInput>& leaf_indices,
                                  size_t query_row_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        auto& context = program.create_forest_inference_context(table);
        if (leaf_indices.has_value()) {
            context.add_check(
                tree_indices, lifted_inputs, lower_bounds, upper_bounds, scores, *leaf_indices, query_row_count);
            return;
        }
        context.add_check(tree_indices, lifted_inputs, lower_bounds, upper_bounds, scores, query_row_count);
    });
}

bool run_forest_prediction_program(ProofRuntime& runtime,
                                   const forestPredictionPublicTable& table,
                                   const ProtocolPolynomialInput& scores,
                                   const ProtocolPolynomialInput& class_scores,
                                   const ProtocolPolynomialInput& predicted_scores,
                                   const ProtocolPolynomialInput& class_selector,
                                   size_t query_row_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        auto& context = program.create_forest_prediction_context(table);
        context.add_check(scores, class_scores, predicted_scores, class_selector, query_row_count);
    });
}

bool run_forest_tree_layout_program(ProofRuntime& runtime,
                                    const forestTreeLayoutPublicTable& table,
                                    const ProtocolPolynomialInput& tree_indices,
                                    size_t query_row_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        auto& context = program.create_forest_tree_layout_context(table);
        context.add_check(tree_indices, query_row_count);
    });
}

bool run_parent_child_box_inheritance_program(ProofRuntime& runtime,
                                              const ProtocolPolynomialInput& feature_selector,
                                              const std::optional<ProtocolPolynomialInput>& maybe_mask,
                                              const ProtocolPolynomialInput& left_child_indices,
                                              const ProtocolPolynomialInput& right_child_indices,
                                              const ProtocolPolynomialInput& parent_lower,
                                              const ProtocolPolynomialInput& parent_upper,
                                              const ProtocolPolynomialInput& left_lower,
                                              const ProtocolPolynomialInput& left_upper,
                                              const ProtocolPolynomialInput& right_lower,
                                              const ProtocolPolynomialInput& right_upper,
                                              const ProtocolPolynomialInput& thresholds,
                                              size_t node_count,
                                              size_t feature_count,
                                              size_t bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        if (maybe_mask.has_value()) {
            program.parent_child_box_inheritance().add_masked_check(program.bind_virtual(feature_selector),
                                                                    program.bind_virtual(*maybe_mask),
                                                                    program.bind_virtual(left_child_indices),
                                                                    program.bind_virtual(right_child_indices),
                                                                    program.bind_virtual(parent_lower),
                                                                    program.bind_virtual(parent_upper),
                                                                    program.bind_virtual(left_lower),
                                                                    program.bind_virtual(left_upper),
                                                                    program.bind_virtual(right_lower),
                                                                    program.bind_virtual(right_upper),
                                                                    program.bind_virtual(thresholds),
                                                                    node_count,
                                                                    feature_count,
                                                                    bit_bound);
            return;
        }

        program.parent_child_box_inheritance().add_check(program.bind_virtual(feature_selector),
                                                         program.bind_virtual(left_child_indices),
                                                         program.bind_virtual(right_child_indices),
                                                         program.bind_virtual(parent_lower),
                                                         program.bind_virtual(parent_upper),
                                                         program.bind_virtual(left_lower),
                                                         program.bind_virtual(left_upper),
                                                         program.bind_virtual(right_lower),
                                                         program.bind_virtual(right_upper),
                                                         program.bind_virtual(thresholds),
                                                         node_count,
                                                         feature_count,
                                                         bit_bound);
        });
}

bool run_root_selector_consistency_program(ProofRuntime& runtime,
                                           const ProtocolPolynomialInput& root_indices,
                                           const ProtocolPolynomialInput& node_tree_ids,
                                           size_t tree_count,
                                           size_t node_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.root_selector_consistency().add_check(
            program.bind_virtual(root_indices), program.bind_virtual(node_tree_ids), tree_count, node_count);
    });
}

bool run_parent_child_tree_consistency_program(ProofRuntime& runtime,
                                               const ProtocolPolynomialInput& left_child_indices,
                                               const ProtocolPolynomialInput& right_child_indices,
                                               const std::optional<ProtocolPolynomialInput>& mask,
                                               const ProtocolPolynomialInput& node_tree_ids,
                                               size_t node_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        if (mask.has_value()) {
            program.parent_child_tree_consistency().add_masked_check(program.bind_virtual(left_child_indices),
                                                                     program.bind_virtual(right_child_indices),
                                                                     program.bind_virtual(*mask),
                                                                     program.bind_virtual(node_tree_ids),
                                                                     node_count);
            return;
        }

        program.parent_child_tree_consistency().add_check(program.bind_virtual(left_child_indices),
                                                          program.bind_virtual(right_child_indices),
                                                          program.bind_virtual(node_tree_ids),
                                                          node_count);
    });
}

bool run_node_role_consistency_program(ProofRuntime& runtime,
                                       const ProtocolPolynomialInput& root_indices,
                                       const ProtocolPolynomialInput& node_leaf_mask,
                                       const ProtocolPolynomialInput& left_child_indices,
                                       const ProtocolPolynomialInput& right_child_indices,
                                       const std::optional<ProtocolPolynomialInput>& parent_mask,
                                       size_t tree_count,
                                       size_t node_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        if (parent_mask.has_value()) {
            program.node_role_consistency().add_masked_check(program.bind_virtual(root_indices),
                                                             program.bind_virtual(node_leaf_mask),
                                                             program.bind_virtual(left_child_indices),
                                                             program.bind_virtual(right_child_indices),
                                                             program.bind_virtual(*parent_mask),
                                                             tree_count,
                                                             node_count);
            return;
        }

        program.node_role_consistency().add_check(program.bind_virtual(root_indices),
                                                  program.bind_virtual(node_leaf_mask),
                                                  program.bind_virtual(left_child_indices),
                                                  program.bind_virtual(right_child_indices),
                                                  tree_count,
                                                  node_count);
    });
}

bool run_forest_structure_program(ProofRuntime& runtime,
                                  const ProtocolPolynomialInput& root_indices,
                                  const ProtocolPolynomialInput& parent_mask,
                                  const ProtocolPolynomialInput& left_child_indices,
                                  const ProtocolPolynomialInput& right_child_indices,
                                  const ProtocolPolynomialInput& node_tree_ids,
                                  const ProtocolPolynomialInput& feature_selector,
                                  const ProtocolPolynomialInput& lower_bounds,
                                  const ProtocolPolynomialInput& upper_bounds,
                                  const ProtocolPolynomialInput& left_lower,
                                  const ProtocolPolynomialInput& left_upper,
                                  const ProtocolPolynomialInput& right_lower,
                                  const ProtocolPolynomialInput& right_upper,
                                  const ProtocolPolynomialInput& thresholds,
                                  size_t tree_count,
                                  size_t node_count,
                                  size_t feature_count,
                                  size_t full_upper_bound,
                                  size_t bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.forest_structure().add_masked_check(program.bind_virtual(root_indices),
                                                    program.bind_virtual(parent_mask),
                                                    program.bind_virtual(left_child_indices),
                                                    program.bind_virtual(right_child_indices),
                                                    program.bind_virtual(node_tree_ids),
                                                    program.bind_virtual(feature_selector),
                                                    program.bind_virtual(lower_bounds),
                                                    program.bind_virtual(upper_bounds),
                                                    program.bind_virtual(left_lower),
                                                    program.bind_virtual(left_upper),
                                                    program.bind_virtual(right_lower),
                                                    program.bind_virtual(right_upper),
                                                    program.bind_virtual(thresholds),
                                                    tree_count,
                                                    node_count,
                                                    feature_count,
                                                    full_upper_bound,
                                                    bit_bound);
    });
}

bool run_forest_classification_program(ProofRuntime& runtime,
                                       const forestClassificationPublicTable& table,
                                       const ProtocolPolynomialInput& tree_indices,
                                       const ProtocolPolynomialInput& lifted_inputs,
                                       const ProtocolPolynomialInput& lower_bounds,
                                       const ProtocolPolynomialInput& upper_bounds,
                                       const ProtocolPolynomialInput& scores,
                                       const std::optional<ProtocolPolynomialInput>& leaf_indices,
                                       const ProtocolPolynomialInput& class_scores,
                                       const ProtocolPolynomialInput& predicted_scores,
                                       const ProtocolPolynomialInput& class_selector,
                                       size_t query_row_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        auto& context = program.create_forest_classification_context(table);
        if (leaf_indices.has_value()) {
            context.add_check(tree_indices,
                              lifted_inputs,
                              lower_bounds,
                              upper_bounds,
                              scores,
                              *leaf_indices,
                              class_scores,
                              predicted_scores,
                              class_selector,
                              query_row_count);
            return;
        }
        context.add_check(tree_indices,
                          lifted_inputs,
                          lower_bounds,
                          upper_bounds,
                          scores,
                          class_scores,
                          predicted_scores,
                          class_selector,
                          query_row_count);
    });
}

bool run_concatenated_table_lookup_program(ProofRuntime& runtime,
                                           const std::vector<ProtocolPolynomialInput>& queries,
                                           const std::vector<ProtocolPolynomialInput>& tables,
                                           const std::vector<size_t>& column_counts,
                                           size_t query_row_count,
                                           size_t table_row_count)
{
    if (queries.size() != tables.size() || queries.size() != column_counts.size()) {
        throw std::runtime_error("run_concatenated_table_lookup_program: block metadata size mismatch");
    }

    return run_proof_program(runtime, [&](ProofProgram& program) {
        std::vector<concatenatedTableLookupBlock> blocks;
        blocks.reserve(queries.size());
        for (size_t i = 0; i < queries.size(); ++i) {
            blocks.emplace_back(
                program.bind_virtual(queries[i]), program.bind_virtual(tables[i]), column_counts[i]);
        }
        program.concatenated_table_lookup().add_check(blocks, query_row_count, table_row_count);
    });
}

bool run_hist_agg_program(ProofRuntime& runtime,
                          const ProtocolPolynomialInput& indices,
                          const ProtocolPolynomialInput& values,
                          const ProtocolPolynomialInput& histogram,
                          size_t query_domain_size,
                          size_t table_domain_size)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.hist_agg().add_check(program.bind_virtual(indices),
                                     program.bind_virtual(values),
                                     program.bind_virtual(histogram),
                                     query_domain_size,
                                     table_domain_size);
    });
}

bool run_tree_histogram_consistency_program(ProofRuntime& runtime,
                                            const ProtocolPolynomialInput& full_histogram,
                                            const ProtocolPolynomialInput& leaf_histogram,
                                            const ProtocolPolynomialInput& parent_mask,
                                            const ProtocolPolynomialInput& left_child_indices,
                                            const ProtocolPolynomialInput& right_child_indices,
                                            size_t node_count,
                                            size_t histogram_width)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.tree_histogram_consistency().add_masked_check(program.bind_virtual(full_histogram),
                                                              program.bind_virtual(leaf_histogram),
                                                              program.bind_virtual(parent_mask),
                                                              program.bind_virtual(left_child_indices),
                                                              program.bind_virtual(right_child_indices),
                                                              node_count,
                                                              histogram_width);
    });
}

bool run_forest_histogram_bucket_address_program(ProofRuntime& runtime,
                                                 const ProtocolPolynomialInput& leaf_indices,
                                                 const ProtocolPolynomialInput& lifted_leaf_indices,
                                                 const ProtocolPolynomialInput& lifted_inputs,
                                                 const ProtocolPolynomialInput& flattened_bucket_indices,
                                                 size_t query_row_count,
                                                 size_t feature_count,
                                                 size_t bin_count,
                                                 size_t node_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestHistogramBucketAddressWitness witness;
        witness.leaf_indices = program.bind_virtual(leaf_indices);
        witness.lifted_leaf_indices = program.bind_virtual(lifted_leaf_indices);
        witness.lifted_inputs = program.bind_virtual(lifted_inputs);
        witness.flattened_bucket_indices = program.bind_virtual(flattened_bucket_indices);

        program.forest_histogram_bucket_address().add_check(
            witness, query_row_count, feature_count, bin_count, node_count);
    });
}

bool run_forest_training_inference_program(ProofRuntime& runtime,
                                           const ProtocolPolynomialInput& query_tree_ids,
                                           const ProtocolPolynomialInput& lifted_inputs,
                                           const ProtocolPolynomialInput& leaf_indices,
                                           const ProtocolPolynomialInput& leaf_lower_bounds,
                                           const ProtocolPolynomialInput& leaf_upper_bounds,
                                           const ProtocolPolynomialInput& tree_scores,
                                           const ProtocolPolynomialInput& node_tree_ids,
                                           const ProtocolPolynomialInput& node_lower_bounds,
                                           const ProtocolPolynomialInput& node_upper_bounds,
                                           const ProtocolPolynomialInput& node_leaf_mask,
                                           const ProtocolPolynomialInput& node_leaf_values,
                                           size_t query_row_count,
                                           size_t node_count,
                                           size_t feature_count,
                                           size_t input_bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestTrainingInferenceWitness witness;
        witness.query_tree_ids = program.bind_virtual(query_tree_ids);
        witness.lifted_inputs = program.bind_virtual(lifted_inputs);
        witness.leaf_indices = program.bind_virtual(leaf_indices);
        witness.leaf_lower_bounds = program.bind_virtual(leaf_lower_bounds);
        witness.leaf_upper_bounds = program.bind_virtual(leaf_upper_bounds);
        witness.tree_scores = program.bind_virtual(tree_scores);
        witness.node_tree_ids = program.bind_virtual(node_tree_ids);
        witness.node_lower_bounds = program.bind_virtual(node_lower_bounds);
        witness.node_upper_bounds = program.bind_virtual(node_upper_bounds);
        witness.node_leaf_mask = program.bind_virtual(node_leaf_mask);
        witness.node_leaf_values = program.bind_virtual(node_leaf_values);

        program.forest_training_inference().add_check(
            witness, query_row_count, node_count, feature_count, input_bit_bound);
    });
}

bool run_forest_histogram_refresh_program(ProofRuntime& runtime,
                                          const ProtocolPolynomialInput& query_tree_ids,
                                          const ProtocolPolynomialInput& lifted_inputs,
                                          const ProtocolPolynomialInput& leaf_indices,
                                          const ProtocolPolynomialInput& leaf_lower_bounds,
                                          const ProtocolPolynomialInput& leaf_upper_bounds,
                                          const ProtocolPolynomialInput& tree_scores,
                                          const ProtocolPolynomialInput& node_tree_ids,
                                          const ProtocolPolynomialInput& node_lower_bounds,
                                          const ProtocolPolynomialInput& node_upper_bounds,
                                          const ProtocolPolynomialInput& node_leaf_mask,
                                          const ProtocolPolynomialInput& node_leaf_values,
                                          const ProtocolPolynomialInput& lifted_leaf_indices,
                                          const ProtocolPolynomialInput& flattened_bucket_indices,
                                          const std::vector<ProtocolPolynomialInput>& query_values,
                                          const std::vector<ProtocolPolynomialInput>& lifted_values,
                                          const std::vector<ProtocolPolynomialInput>& leaf_histograms,
                                          const std::vector<ProtocolPolynomialInput>& full_histograms,
                                          const ProtocolPolynomialInput& parent_mask,
                                          const ProtocolPolynomialInput& left_child_indices,
                                          const ProtocolPolynomialInput& right_child_indices,
                                          size_t query_row_count,
                                          size_t node_count,
                                          size_t feature_count,
                                          size_t bin_count,
                                          size_t input_bit_bound)
{
    if (query_values.size() != lifted_values.size() || query_values.size() != leaf_histograms.size() ||
        query_values.size() != full_histograms.size()) {
        throw std::runtime_error("run_forest_histogram_refresh_program: channel metadata size mismatch");
    }

    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestHistogramRefreshWitness witness;
        witness.inference.query_tree_ids = program.bind_virtual(query_tree_ids);
        witness.inference.lifted_inputs = program.bind_virtual(lifted_inputs);
        witness.inference.leaf_indices = program.bind_virtual(leaf_indices);
        witness.inference.leaf_lower_bounds = program.bind_virtual(leaf_lower_bounds);
        witness.inference.leaf_upper_bounds = program.bind_virtual(leaf_upper_bounds);
        witness.inference.tree_scores = program.bind_virtual(tree_scores);
        witness.inference.node_tree_ids = program.bind_virtual(node_tree_ids);
        witness.inference.node_lower_bounds = program.bind_virtual(node_lower_bounds);
        witness.inference.node_upper_bounds = program.bind_virtual(node_upper_bounds);
        witness.inference.node_leaf_mask = program.bind_virtual(node_leaf_mask);
        witness.inference.node_leaf_values = program.bind_virtual(node_leaf_values);
        witness.lifted_leaf_indices = program.bind_virtual(lifted_leaf_indices);
        witness.flattened_bucket_indices = program.bind_virtual(flattened_bucket_indices);
        witness.parent_mask = program.bind_virtual(parent_mask);
        witness.left_child_indices = program.bind_virtual(left_child_indices);
        witness.right_child_indices = program.bind_virtual(right_child_indices);

        witness.channels.query_values.reserve(query_values.size());
        witness.channels.lifted_values.reserve(query_values.size());
        witness.channels.leaf_histograms.reserve(query_values.size());
        witness.channels.full_histograms.reserve(query_values.size());
        for (size_t i = 0; i < query_values.size(); ++i) {
            witness.channels.query_values.push_back(program.bind_virtual(query_values[i]));
            witness.channels.lifted_values.push_back(program.bind_virtual(lifted_values[i]));
            witness.channels.leaf_histograms.push_back(program.bind_virtual(leaf_histograms[i]));
            witness.channels.full_histograms.push_back(program.bind_virtual(full_histograms[i]));
        }

        program.forest_histogram_refresh().add_check(
            witness, query_row_count, node_count, feature_count, bin_count, input_bit_bound);
    });
}

bool run_forest_histogram_program(ProofRuntime& runtime,
                                  const ProtocolPolynomialInput& flattened_bucket_indices,
                                  const ProtocolPolynomialInput& values,
                                  const ProtocolPolynomialInput& leaf_histogram,
                                  const ProtocolPolynomialInput& full_histogram,
                                  const ProtocolPolynomialInput& parent_mask,
                                  const ProtocolPolynomialInput& left_child_indices,
                                  const ProtocolPolynomialInput& right_child_indices,
                                  size_t query_domain_size,
                                  size_t node_count,
                                  size_t histogram_width)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.forest_histogram().add_masked_check(program.bind_virtual(flattened_bucket_indices),
                                                    program.bind_virtual(values),
                                                    program.bind_virtual(leaf_histogram),
                                                    program.bind_virtual(full_histogram),
                                                    program.bind_virtual(parent_mask),
                                                    program.bind_virtual(left_child_indices),
                                                    program.bind_virtual(right_child_indices),
                                                    query_domain_size,
                                                    node_count,
                                                    histogram_width);
    });
}

bool run_forest_histogram_multi_program(ProofRuntime& runtime,
                                        const ProtocolPolynomialInput& flattened_bucket_indices,
                                        const std::vector<ProtocolPolynomialInput>& values,
                                        const std::vector<ProtocolPolynomialInput>& leaf_histograms,
                                        const std::vector<ProtocolPolynomialInput>& full_histograms,
                                        const ProtocolPolynomialInput& parent_mask,
                                        const ProtocolPolynomialInput& left_child_indices,
                                        const ProtocolPolynomialInput& right_child_indices,
                                        size_t query_domain_size,
                                        size_t node_count,
                                        size_t histogram_width)
{
    if (values.size() != leaf_histograms.size() || values.size() != full_histograms.size()) {
        throw std::runtime_error("run_forest_histogram_multi_program: channel metadata size mismatch");
    }

    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestHistogramChannelBatch channels;
        channels.values.reserve(values.size());
        channels.leaf_histograms.reserve(values.size());
        channels.full_histograms.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            channels.values.push_back(program.bind_virtual(values[i]));
            channels.leaf_histograms.push_back(program.bind_virtual(leaf_histograms[i]));
            channels.full_histograms.push_back(program.bind_virtual(full_histograms[i]));
        }

        program.forest_histogram().add_masked_check(program.bind_virtual(flattened_bucket_indices),
                                                    channels,
                                                    program.bind_virtual(parent_mask),
                                                    program.bind_virtual(left_child_indices),
                                                    program.bind_virtual(right_child_indices),
                                                    query_domain_size,
                                                    node_count,
                                                    histogram_width);
    });
}

bool run_forest_split_statistics_multi_program(ProofRuntime& runtime,
                                               const std::vector<ProtocolPolynomialInput>& histograms,
                                               const std::vector<ProtocolPolynomialInput>& left_statistics,
                                               const std::vector<ProtocolPolynomialInput>& right_statistics,
                                               size_t node_count,
                                               size_t feature_count,
                                               size_t bin_count)
{
    if (histograms.size() != left_statistics.size() || histograms.size() != right_statistics.size()) {
        throw std::runtime_error("run_forest_split_statistics_multi_program: channel metadata size mismatch");
    }

    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestSplitStatisticsChannelBatch channels;
        channels.histograms.reserve(histograms.size());
        channels.left_statistics.reserve(histograms.size());
        channels.right_statistics.reserve(histograms.size());
        for (size_t i = 0; i < histograms.size(); ++i) {
            channels.histograms.push_back(program.bind_virtual(histograms[i]));
            channels.left_statistics.push_back(program.bind_virtual(left_statistics[i]));
            channels.right_statistics.push_back(program.bind_virtual(right_statistics[i]));
        }

        program.forest_split_statistics().add_check(channels, node_count, feature_count, bin_count);
    });
}

bool run_forest_split_gain_program(ProofRuntime& runtime,
                                   const ProtocolPolynomialInput& left_gradients,
                                   const ProtocolPolynomialInput& right_gradients,
                                   const ProtocolPolynomialInput& left_hessians,
                                   const ProtocolPolynomialInput& right_hessians,
                                   const ProtocolPolynomialInput& left_gradient_squares,
                                   const ProtocolPolynomialInput& right_gradient_squares,
                                   const ProtocolPolynomialInput& parent_gradient_squares,
                                   const ProtocolPolynomialInput& gain_numerators,
                                   const ProtocolPolynomialInput& gain_denominators,
                                   size_t node_count,
                                   size_t feature_count,
                                   size_t bin_count,
                                   const fixedPointEncoding& gradient_encoding,
                                   size_t hessian_bit_bound,
                                   size_t denominator_bit_bound,
                                   size_t l2_regularization,
                                   const Fr& split_penalty)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestSplitGainWitness witness;
        witness.left_gradients = program.bind_virtual(left_gradients);
        witness.right_gradients = program.bind_virtual(right_gradients);
        witness.left_hessians = program.bind_virtual(left_hessians);
        witness.right_hessians = program.bind_virtual(right_hessians);
        witness.left_gradient_squares = program.bind_virtual(left_gradient_squares);
        witness.right_gradient_squares = program.bind_virtual(right_gradient_squares);
        witness.parent_gradient_squares = program.bind_virtual(parent_gradient_squares);
        witness.gain_numerators = program.bind_virtual(gain_numerators);
        witness.gain_denominators = program.bind_virtual(gain_denominators);

        program.forest_split_gain().add_check(witness,
                                              node_count,
                                              feature_count,
                                              bin_count,
                                              gradient_encoding,
                                              hessian_bit_bound,
                                              denominator_bit_bound,
                                              l2_regularization,
                                              split_penalty);
    });
}

bool run_forest_quantized_split_gain_program(ProofRuntime& runtime,
                                             const ProtocolPolynomialInput& exact_gain_numerators,
                                             const ProtocolPolynomialInput& gain_denominators,
                                             const ProtocolPolynomialInput& encoded_gain_numerators,
                                             const ProtocolPolynomialInput& encoded_quantized_gains,
                                             const ProtocolPolynomialInput& gain_remainders,
                                             size_t node_count,
                                             size_t feature_count,
                                             size_t bin_count,
                                             const fixedPointEncoding& numerator_encoding,
                                             const fixedPointEncoding& quantized_gain_encoding,
                                             size_t denominator_bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestQuantizedSplitGainWitness witness;
        witness.exact_gain_numerators = program.bind_virtual(exact_gain_numerators);
        witness.gain_denominators = program.bind_virtual(gain_denominators);
        witness.encoded_gain_numerators = program.bind_virtual(encoded_gain_numerators);
        witness.encoded_quantized_gains = program.bind_virtual(encoded_quantized_gains);
        witness.gain_remainders = program.bind_virtual(gain_remainders);

        program.forest_quantized_split_gain().add_check(
            witness, node_count, feature_count, bin_count, numerator_encoding, quantized_gain_encoding, denominator_bit_bound);
    });
}

bool run_forest_split_selection_program(ProofRuntime& runtime,
                                        const ProtocolPolynomialInput& quantized_gains,
                                        const ProtocolPolynomialInput& tie_break_scores,
                                        const ProtocolPolynomialInput& candidate_mask,
                                        const ProtocolPolynomialInput& active_node_mask,
                                        const ProtocolPolynomialInput& selector,
                                        const ProtocolPolynomialInput& selected_gains,
                                        const ProtocolPolynomialInput& selected_tie_break_scores,
                                        const ProtocolPolynomialInput& selected_candidate_indices,
                                        const ProtocolPolynomialInput& selected_features,
                                        const ProtocolPolynomialInput& selected_bins,
                                        size_t node_count,
                                        size_t feature_count,
                                        size_t bin_count,
                                        size_t gain_gap_bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestSplitSelectionWitness witness;
        witness.quantized_gains = program.bind_virtual(quantized_gains);
        witness.tie_break_scores = program.bind_virtual(tie_break_scores);
        witness.candidate_mask = program.bind_virtual(candidate_mask);
        witness.left_counts = witness.candidate_mask;
        witness.right_counts = witness.candidate_mask;
        witness.active_node_mask = program.bind_virtual(active_node_mask);
        witness.selector = program.bind_virtual(selector);
        witness.selected_gains = program.bind_virtual(selected_gains);
        witness.selected_tie_break_scores = program.bind_virtual(selected_tie_break_scores);
        witness.selected_candidate_indices = program.bind_virtual(selected_candidate_indices);
        witness.selected_features = program.bind_virtual(selected_features);
        witness.selected_bins = program.bind_virtual(selected_bins);

        program.forest_split_selection().add_check(
            witness, node_count, feature_count, bin_count, feature_count * bin_count, gain_gap_bit_bound);
    });
}

bool run_forest_split_update_program(ProofRuntime& runtime,
                                     const ProtocolPolynomialInput& root_indices,
                                     const ProtocolPolynomialInput& active_node_mask,
                                     const ProtocolPolynomialInput& current_leaf_mask,
                                     const ProtocolPolynomialInput& current_internal_mask,
                                     const ProtocolPolynomialInput& next_leaf_mask,
                                     const ProtocolPolynomialInput& next_internal_mask,
                                     const ProtocolPolynomialInput& next_active_node_mask,
                                     const ProtocolPolynomialInput& left_child_occupancy,
                                     const ProtocolPolynomialInput& right_child_occupancy,
                                     const ProtocolPolynomialInput& left_child_indices,
                                     const ProtocolPolynomialInput& right_child_indices,
                                     const ProtocolPolynomialInput& node_tree_ids,
                                     const ProtocolPolynomialInput& selected_features,
                                     const ProtocolPolynomialInput& selected_bins,
                                     const ProtocolPolynomialInput& feature_selector,
                                     const ProtocolPolynomialInput& lower_bounds,
                                     const ProtocolPolynomialInput& upper_bounds,
                                     const ProtocolPolynomialInput& left_lower,
                                     const ProtocolPolynomialInput& left_upper,
                                     const ProtocolPolynomialInput& right_lower,
                                     const ProtocolPolynomialInput& right_upper,
                                     const ProtocolPolynomialInput& thresholds,
                                     size_t tree_count,
                                     size_t node_count,
                                     size_t feature_count,
                                     size_t full_upper_bound,
                                     size_t bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestSplitUpdateWitness witness;
        witness.root_indices = program.bind_virtual(root_indices);
        witness.active_node_mask = program.bind_virtual(active_node_mask);
        witness.current_leaf_mask = program.bind_virtual(current_leaf_mask);
        witness.current_internal_mask = program.bind_virtual(current_internal_mask);
        witness.next_leaf_mask = program.bind_virtual(next_leaf_mask);
        witness.next_internal_mask = program.bind_virtual(next_internal_mask);
        witness.next_active_node_mask = program.bind_virtual(next_active_node_mask);
        witness.left_child_occupancy = program.bind_virtual(left_child_occupancy);
        witness.right_child_occupancy = program.bind_virtual(right_child_occupancy);
        witness.left_child_indices = program.bind_virtual(left_child_indices);
        witness.right_child_indices = program.bind_virtual(right_child_indices);
        witness.node_tree_ids = program.bind_virtual(node_tree_ids);
        witness.selected_features = program.bind_virtual(selected_features);
        witness.selected_bins = program.bind_virtual(selected_bins);
        witness.feature_selector = program.bind_virtual(feature_selector);
        witness.lower_bounds = program.bind_virtual(lower_bounds);
        witness.upper_bounds = program.bind_virtual(upper_bounds);
        witness.left_lower = program.bind_virtual(left_lower);
        witness.left_upper = program.bind_virtual(left_upper);
        witness.right_lower = program.bind_virtual(right_lower);
        witness.right_upper = program.bind_virtual(right_upper);
        witness.thresholds = program.bind_virtual(thresholds);

        program.forest_split_update().add_check(
            witness, tree_count, node_count, feature_count, full_upper_bound, bit_bound);
    });
}

bool run_forest_leaf_value_program(ProofRuntime& runtime,
                                   const ProtocolPolynomialInput& leaf_gradients,
                                   const ProtocolPolynomialInput& leaf_hessians,
                                   const ProtocolPolynomialInput& leaf_denominators,
                                   const ProtocolPolynomialInput& encoded_negative_leaf_gradients,
                                   const ProtocolPolynomialInput& encoded_leaf_values,
                                   const ProtocolPolynomialInput& leaf_remainders,
                                   size_t node_count,
                                   const fixedPointEncoding& gradient_encoding,
                                   const fixedPointEncoding& leaf_value_encoding,
                                   size_t hessian_bit_bound,
                                   size_t denominator_bit_bound,
                                   size_t l2_regularization)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestLeafValueWitness witness;
        witness.leaf_gradients = program.bind_virtual(leaf_gradients);
        witness.leaf_hessians = program.bind_virtual(leaf_hessians);
        witness.leaf_denominators = program.bind_virtual(leaf_denominators);
        witness.encoded_negative_leaf_gradients = program.bind_virtual(encoded_negative_leaf_gradients);
        witness.encoded_leaf_values = program.bind_virtual(encoded_leaf_values);
        witness.leaf_remainders = program.bind_virtual(leaf_remainders);

        program.forest_leaf_value().add_check(witness,
                                              node_count,
                                              gradient_encoding,
                                              leaf_value_encoding,
                                              hessian_bit_bound,
                                              denominator_bit_bound,
                                              l2_regularization);
    });
}

bool run_forest_boosting_state_program(ProofRuntime& runtime,
                                       const ProtocolPolynomialInput& tree_scores,
                                       const ProtocolPolynomialInput& cumulative_scores,
                                       const ProtocolPolynomialInput& encoded_cumulative_scores,
                                       const ProtocolPolynomialInput& encoded_shifted_scores,
                                       const ProtocolPolynomialInput& softmax_shifts,
                                       const ProtocolPolynomialInput& shift_selector,
                                       const ProtocolPolynomialInput& lifted_softmax_shifts,
                                       const ProtocolPolynomialInput& softmax_exp_values,
                                       const ProtocolPolynomialInput& softmax_denominators,
                                       const ProtocolPolynomialInput& lifted_softmax_denominators,
                                       const ProtocolPolynomialInput& softmax_probabilities,
                                       const ProtocolPolynomialInput& softmax_remainders,
                                       const ProtocolPolynomialInput& base_labels,
                                       const ProtocolPolynomialInput& pseudo_residuals,
                                       const ProtocolPolynomialInput& encoded_diagonal_hessians,
                                       const ProtocolPolynomialInput& diagonal_hessian_remainders,
                                       const ProtocolPolynomialInput& diagonal_hessians,
                                       size_t iteration_count,
                                       size_t class_count,
                                       size_t query_count,
                                       const fixedPointEncoding& score_encoding,
                                       const fixedPointEncoding& softmax_encoding,
                                       size_t hessian_bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestBoostingStateWitness witness;
        witness.tree_scores = program.bind_virtual(tree_scores);
        witness.cumulative_scores = program.bind_virtual(cumulative_scores);
        witness.encoded_cumulative_scores = program.bind_virtual(encoded_cumulative_scores);
        witness.encoded_shifted_scores = program.bind_virtual(encoded_shifted_scores);
        witness.softmax_shifts = program.bind_virtual(softmax_shifts);
        witness.shift_selector = program.bind_virtual(shift_selector);
        witness.lifted_softmax_shifts = program.bind_virtual(lifted_softmax_shifts);
        witness.softmax_exp_values = program.bind_virtual(softmax_exp_values);
        witness.softmax_denominators = program.bind_virtual(softmax_denominators);
        witness.lifted_softmax_denominators = program.bind_virtual(lifted_softmax_denominators);
        witness.softmax_probabilities = program.bind_virtual(softmax_probabilities);
        witness.softmax_remainders = program.bind_virtual(softmax_remainders);
        witness.base_labels = program.bind_virtual(base_labels);
        witness.pseudo_residuals = program.bind_virtual(pseudo_residuals);
        witness.encoded_diagonal_hessians = program.bind_virtual(encoded_diagonal_hessians);
        witness.diagonal_hessian_remainders = program.bind_virtual(diagonal_hessian_remainders);
        witness.diagonal_hessians = program.bind_virtual(diagonal_hessians);

        program.forest_boosting_state().add_check(
            witness, iteration_count, class_count, query_count, score_encoding, softmax_encoding, hessian_bit_bound);
    });
}

bool run_forest_training_proof_program(ProofRuntime& runtime,
                                       const ProtocolPolynomialInput& root_indices,
                                       const ProtocolPolynomialInput& split_node_mask,
                                       const ProtocolPolynomialInput& node_depths,
                                       const ProtocolPolynomialInput& left_child_indices,
                                       const ProtocolPolynomialInput& right_child_indices,
                                       const ProtocolPolynomialInput& node_tree_ids,
                                       const ProtocolPolynomialInput& feature_selector,
                                       const ProtocolPolynomialInput& lower_bounds,
                                       const ProtocolPolynomialInput& upper_bounds,
                                       const ProtocolPolynomialInput& left_lower,
                                       const ProtocolPolynomialInput& left_upper,
                                       const ProtocolPolynomialInput& right_lower,
                                       const ProtocolPolynomialInput& right_upper,
                                       const ProtocolPolynomialInput& thresholds,
                                       const ProtocolPolynomialInput& query_tree_ids,
                                       const ProtocolPolynomialInput& lifted_inputs,
                                       const ProtocolPolynomialInput& leaf_indices,
                                       const ProtocolPolynomialInput& leaf_lower_bounds,
                                       const ProtocolPolynomialInput& leaf_upper_bounds,
                                       const ProtocolPolynomialInput& tree_scores,
                                       const ProtocolPolynomialInput& training_node_tree_ids,
                                       const ProtocolPolynomialInput& training_node_lower_bounds,
                                       const ProtocolPolynomialInput& training_node_upper_bounds,
                                       const ProtocolPolynomialInput& node_leaf_mask,
                                       const ProtocolPolynomialInput& node_leaf_values,
                                       const ProtocolPolynomialInput& lifted_leaf_indices,
                                       const ProtocolPolynomialInput& flattened_bucket_indices,
                                       const std::vector<ProtocolPolynomialInput>& query_values,
                                       const std::vector<ProtocolPolynomialInput>& lifted_values,
                                       const std::vector<ProtocolPolynomialInput>& leaf_histograms,
                                       const std::vector<ProtocolPolynomialInput>& full_histograms,
                                       const std::vector<ProtocolPolynomialInput>& split_histograms,
                                       const std::vector<ProtocolPolynomialInput>& left_statistics,
                                       const std::vector<ProtocolPolynomialInput>& right_statistics,
                                       const ProtocolPolynomialInput& left_gradients,
                                       const ProtocolPolynomialInput& right_gradients,
                                       const ProtocolPolynomialInput& left_hessians,
                                       const ProtocolPolynomialInput& right_hessians,
                                       const ProtocolPolynomialInput& left_gradient_squares,
                                       const ProtocolPolynomialInput& right_gradient_squares,
                                       const ProtocolPolynomialInput& parent_gradient_squares,
                                       const ProtocolPolynomialInput& gain_numerators,
                                       const ProtocolPolynomialInput& gain_denominators,
                                       const ProtocolPolynomialInput& encoded_gain_numerators,
                                       const ProtocolPolynomialInput& encoded_quantized_gains,
                                       const ProtocolPolynomialInput& gain_remainders,
                                       const ProtocolPolynomialInput& quantized_gains,
                                       const ProtocolPolynomialInput& tie_break_scores,
                                       const ProtocolPolynomialInput& candidate_mask,
                                       const ProtocolPolynomialInput& active_node_mask,
                                       const ProtocolPolynomialInput& selector,
                                       const ProtocolPolynomialInput& selected_gains,
                                       const ProtocolPolynomialInput& selected_tie_break_scores,
                                       const ProtocolPolynomialInput& selected_candidate_indices,
                                       const ProtocolPolynomialInput& selected_features,
                                       const ProtocolPolynomialInput& selected_bins,
                                       const ProtocolPolynomialInput& leaf_gradients,
                                       const ProtocolPolynomialInput& leaf_hessians,
                                       const ProtocolPolynomialInput& leaf_denominators,
                                       const ProtocolPolynomialInput& encoded_negative_leaf_gradients,
                                       const ProtocolPolynomialInput& encoded_leaf_values,
                                       const ProtocolPolynomialInput& leaf_remainders,
                                       const ProtocolPolynomialInput& cumulative_scores,
                                       const ProtocolPolynomialInput& encoded_cumulative_scores,
                                       const ProtocolPolynomialInput& encoded_shifted_scores,
                                       const ProtocolPolynomialInput& softmax_shifts,
                                       const ProtocolPolynomialInput& shift_selector,
                                       const ProtocolPolynomialInput& lifted_softmax_shifts,
                                       const ProtocolPolynomialInput& softmax_exp_values,
                                       const ProtocolPolynomialInput& softmax_denominators,
                                       const ProtocolPolynomialInput& lifted_softmax_denominators,
                                       const ProtocolPolynomialInput& softmax_probabilities,
                                       const ProtocolPolynomialInput& softmax_remainders,
                                       const ProtocolPolynomialInput& base_labels,
                                       const ProtocolPolynomialInput& pseudo_residuals,
                                       const ProtocolPolynomialInput& encoded_diagonal_hessians,
                                       const ProtocolPolynomialInput& diagonal_hessian_remainders,
                                       const ProtocolPolynomialInput& diagonal_hessians,
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
                                       const Fr& split_penalty)
{
    if (query_values.size() != 3 || lifted_values.size() != 3 || leaf_histograms.size() != 3 ||
        full_histograms.size() != 3 || split_histograms.size() != 3 || left_statistics.size() != 3 ||
        right_statistics.size() != 3) {
        throw std::runtime_error("run_forest_training_proof_program: expected exactly three channels");
    }

    return run_proof_program(runtime, [&](ProofProgram& program) {
        forestTrainingProofWitness witness;
        witness.root_indices = program.bind_virtual(root_indices);
        witness.split_node_mask = program.bind_virtual(split_node_mask);
        witness.node_leaf_mask = program.bind_virtual(node_leaf_mask);
        witness.depths = program.bind_virtual(node_depths);
        witness.left_child_indices = program.bind_virtual(left_child_indices);
        witness.right_child_indices = program.bind_virtual(right_child_indices);
        witness.node_tree_ids = program.bind_virtual(node_tree_ids);
        witness.feature_selector = program.bind_virtual(feature_selector);
        witness.lower_bounds = program.bind_virtual(lower_bounds);
        witness.upper_bounds = program.bind_virtual(upper_bounds);
        witness.left_lower = program.bind_virtual(left_lower);
        witness.left_upper = program.bind_virtual(left_upper);
        witness.right_lower = program.bind_virtual(right_lower);
        witness.right_upper = program.bind_virtual(right_upper);
        witness.thresholds = program.bind_virtual(thresholds);

        witness.histogram_refresh.inference.query_tree_ids = program.bind_virtual(query_tree_ids);
        witness.histogram_refresh.inference.lifted_inputs = program.bind_virtual(lifted_inputs);
        witness.histogram_refresh.inference.leaf_indices = program.bind_virtual(leaf_indices);
        witness.histogram_refresh.inference.leaf_lower_bounds = program.bind_virtual(leaf_lower_bounds);
        witness.histogram_refresh.inference.leaf_upper_bounds = program.bind_virtual(leaf_upper_bounds);
        witness.histogram_refresh.inference.tree_scores = program.bind_virtual(tree_scores);
        witness.histogram_refresh.inference.node_tree_ids = program.bind_virtual(training_node_tree_ids);
        witness.histogram_refresh.inference.node_lower_bounds = program.bind_virtual(training_node_lower_bounds);
        witness.histogram_refresh.inference.node_upper_bounds = program.bind_virtual(training_node_upper_bounds);
        witness.histogram_refresh.inference.node_leaf_mask = program.bind_virtual(node_leaf_mask);
        witness.histogram_refresh.inference.node_leaf_values = program.bind_virtual(node_leaf_values);
        witness.histogram_refresh.lifted_leaf_indices = program.bind_virtual(lifted_leaf_indices);
        witness.histogram_refresh.flattened_bucket_indices = program.bind_virtual(flattened_bucket_indices);
        witness.histogram_refresh.parent_mask = program.bind_virtual(split_node_mask);
        witness.histogram_refresh.left_child_indices = program.bind_virtual(left_child_indices);
        witness.histogram_refresh.right_child_indices = program.bind_virtual(right_child_indices);

        for (size_t i = 0; i < 3; ++i) {
            witness.histogram_refresh.channels.query_values.push_back(program.bind_virtual(query_values[i]));
            witness.histogram_refresh.channels.lifted_values.push_back(program.bind_virtual(lifted_values[i]));
            witness.histogram_refresh.channels.leaf_histograms.push_back(program.bind_virtual(leaf_histograms[i]));
            witness.histogram_refresh.channels.full_histograms.push_back(program.bind_virtual(full_histograms[i]));
            witness.split_statistics.histograms.push_back(program.bind_virtual(split_histograms[i]));
            witness.split_statistics.left_statistics.push_back(program.bind_virtual(left_statistics[i]));
            witness.split_statistics.right_statistics.push_back(program.bind_virtual(right_statistics[i]));
        }

        witness.split_gain.left_gradients = program.bind_virtual(left_gradients);
        witness.split_gain.right_gradients = program.bind_virtual(right_gradients);
        witness.split_gain.left_hessians = program.bind_virtual(left_hessians);
        witness.split_gain.right_hessians = program.bind_virtual(right_hessians);
        witness.split_gain.left_gradient_squares = program.bind_virtual(left_gradient_squares);
        witness.split_gain.right_gradient_squares = program.bind_virtual(right_gradient_squares);
        witness.split_gain.parent_gradient_squares = program.bind_virtual(parent_gradient_squares);
        witness.split_gain.gain_numerators = program.bind_virtual(gain_numerators);
        witness.split_gain.gain_denominators = program.bind_virtual(gain_denominators);

        witness.quantized_split_gain.exact_gain_numerators = witness.split_gain.gain_numerators;
        witness.quantized_split_gain.gain_denominators = witness.split_gain.gain_denominators;
        witness.quantized_split_gain.encoded_gain_numerators = program.bind_virtual(encoded_gain_numerators);
        witness.quantized_split_gain.encoded_quantized_gains = program.bind_virtual(encoded_quantized_gains);
        witness.quantized_split_gain.gain_remainders = program.bind_virtual(gain_remainders);

        witness.split_selection.quantized_gains = program.bind_virtual(quantized_gains);
        witness.split_selection.tie_break_scores = program.bind_virtual(tie_break_scores);
        witness.split_selection.candidate_mask = program.bind_virtual(candidate_mask);
        witness.split_selection.active_node_mask = program.bind_virtual(active_node_mask);
        witness.split_selection.selector = program.bind_virtual(selector);
        witness.split_selection.selected_gains = program.bind_virtual(selected_gains);
        witness.split_selection.selected_tie_break_scores = program.bind_virtual(selected_tie_break_scores);
        witness.split_selection.selected_candidate_indices = program.bind_virtual(selected_candidate_indices);
        witness.split_selection.selected_features = program.bind_virtual(selected_features);
        witness.split_selection.selected_bins = program.bind_virtual(selected_bins);

        witness.leaf_value.leaf_gradients = program.bind_virtual(leaf_gradients);
        witness.leaf_value.leaf_hessians = program.bind_virtual(leaf_hessians);
        witness.leaf_value.leaf_denominators = program.bind_virtual(leaf_denominators);
        witness.leaf_value.encoded_negative_leaf_gradients = program.bind_virtual(encoded_negative_leaf_gradients);
        witness.leaf_value.encoded_leaf_values = program.bind_virtual(encoded_leaf_values);
        witness.leaf_value.leaf_remainders = program.bind_virtual(leaf_remainders);

        witness.boosting_state.tree_scores = program.bind_virtual(tree_scores);
        witness.boosting_state.cumulative_scores = program.bind_virtual(cumulative_scores);
        witness.boosting_state.encoded_cumulative_scores = program.bind_virtual(encoded_cumulative_scores);
        witness.boosting_state.encoded_shifted_scores = program.bind_virtual(encoded_shifted_scores);
        witness.boosting_state.softmax_shifts = program.bind_virtual(softmax_shifts);
        witness.boosting_state.shift_selector = program.bind_virtual(shift_selector);
        witness.boosting_state.lifted_softmax_shifts = program.bind_virtual(lifted_softmax_shifts);
        witness.boosting_state.softmax_exp_values = program.bind_virtual(softmax_exp_values);
        witness.boosting_state.softmax_denominators = program.bind_virtual(softmax_denominators);
        witness.boosting_state.lifted_softmax_denominators = program.bind_virtual(lifted_softmax_denominators);
        witness.boosting_state.softmax_probabilities = program.bind_virtual(softmax_probabilities);
        witness.boosting_state.softmax_remainders = program.bind_virtual(softmax_remainders);
        witness.boosting_state.base_labels = program.bind_virtual(base_labels);
        witness.boosting_state.pseudo_residuals = program.bind_virtual(pseudo_residuals);
        witness.boosting_state.encoded_diagonal_hessians = program.bind_virtual(encoded_diagonal_hessians);
        witness.boosting_state.diagonal_hessian_remainders = program.bind_virtual(diagonal_hessian_remainders);
        witness.boosting_state.diagonal_hessians = program.bind_virtual(diagonal_hessians);

        program.forest_training_proof().add_check(witness,
                                                  tree_count,
                                                  node_count,
                                                  feature_count,
                                                  bin_count,
                                                  iteration_count,
                                                  class_count,
                                                  query_count,
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
                                                  split_penalty);
    });
}

bool run_selector_program(ProofRuntime& runtime,
                          const ProtocolPolynomialInput& selector_matrix,
                          const std::optional<ProtocolPolynomialInput>& maybe_mask,
                          size_t row_count,
                          size_t column_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        const virCommitment selector_commitment = program.bind_virtual(selector_matrix);
        if (maybe_mask.has_value()) {
            program.selector_check().add_masked_one_hot_rows_check(
                selector_commitment, program.bind_virtual(*maybe_mask), row_count, column_count);
            return;
        }
        program.selector_check().add_one_hot_rows_check(selector_commitment, row_count, column_count);
    });
}

bool run_argmax_program(ProofRuntime& runtime,
                        const ProtocolPolynomialInput& matrix,
                        const ProtocolPolynomialInput& max_values,
                        const ProtocolPolynomialInput& selector,
                        size_t row_count,
                        size_t column_count,
                        size_t bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.argmax().add_check(program.bind_virtual(matrix),
                                   program.bind_virtual(max_values),
                                   program.bind_virtual(selector),
                                   row_count,
                                   column_count,
                                   bit_bound);
    });
}

bool run_signed_range_program(ProofRuntime& runtime,
                              const ProtocolPolynomialInput& encoded_values,
                              const fixedPointEncoding& encoding,
                              size_t domain_size)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.signed_range_check().add_check(program.bind_virtual(encoded_values), encoding, domain_size);
    });
}

bool run_rational_program(ProofRuntime& runtime,
                          const ProtocolPolynomialInput& numerator,
                          const ProtocolPolynomialInput& denominator,
                          const ProtocolPolynomialInput& quotient,
                          const std::optional<ProtocolPolynomialInput>& maybe_mask,
                          size_t domain_size)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        if (maybe_mask.has_value()) {
            program.rational_check().add_masked_check(program.bind_virtual(numerator),
                                                      program.bind_virtual(denominator),
                                                      program.bind_virtual(quotient),
                                                      program.bind_virtual(*maybe_mask),
                                                      domain_size);
            return;
        }

        program.rational_check().add_check(
            program.bind_virtual(numerator), program.bind_virtual(denominator), program.bind_virtual(quotient), domain_size);
    });
}

bool run_fixed_point_division_program(ProofRuntime& runtime,
                                      const ProtocolPolynomialInput& encoded_numerator,
                                      const ProtocolPolynomialInput& denominator,
                                      const ProtocolPolynomialInput& encoded_quotient,
                                      const ProtocolPolynomialInput& remainder,
                                      const std::optional<ProtocolPolynomialInput>& maybe_mask,
                                      const fixedPointEncoding& numerator_encoding,
                                      const fixedPointEncoding& quotient_encoding,
                                      size_t denominator_bit_bound,
                                      size_t domain_size)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        if (maybe_mask.has_value()) {
            program.fixed_point_division().add_masked_check(program.bind_virtual(encoded_numerator),
                                                            program.bind_virtual(denominator),
                                                            program.bind_virtual(encoded_quotient),
                                                            program.bind_virtual(remainder),
                                                            program.bind_virtual(*maybe_mask),
                                                            numerator_encoding,
                                                            quotient_encoding,
                                                            denominator_bit_bound,
                                                            domain_size);
            return;
        }

        program.fixed_point_division().add_check(program.bind_virtual(encoded_numerator),
                                                 program.bind_virtual(denominator),
                                                 program.bind_virtual(encoded_quotient),
                                                 program.bind_virtual(remainder),
                                                 numerator_encoding,
                                                 quotient_encoding,
                                                 denominator_bit_bound,
                                                 domain_size);
    });
}

bool run_signed_denominator_fixed_point_division_program(ProofRuntime& runtime,
                                                         const ProtocolPolynomialInput& encoded_numerator,
                                                         const ProtocolPolynomialInput& encoded_denominator,
                                                         const ProtocolPolynomialInput& denominator_abs,
                                                         const ProtocolPolynomialInput& denominator_is_negative,
                                                         const ProtocolPolynomialInput& encoded_quotient,
                                                         const ProtocolPolynomialInput& remainder_abs,
                                                         const std::optional<ProtocolPolynomialInput>& maybe_mask,
                                                         const fixedPointEncoding& numerator_encoding,
                                                         const fixedPointEncoding& denominator_encoding,
                                                         const fixedPointEncoding& quotient_encoding,
                                                         size_t denominator_bit_bound,
                                                         size_t domain_size)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        if (maybe_mask.has_value()) {
            program.fixed_point_division().add_masked_check(program.bind_virtual(encoded_numerator),
                                                            program.bind_virtual(encoded_denominator),
                                                            program.bind_virtual(denominator_abs),
                                                            program.bind_virtual(denominator_is_negative),
                                                            program.bind_virtual(encoded_quotient),
                                                            program.bind_virtual(remainder_abs),
                                                            program.bind_virtual(*maybe_mask),
                                                            numerator_encoding,
                                                            denominator_encoding,
                                                            quotient_encoding,
                                                            denominator_bit_bound,
                                                            domain_size);
            return;
        }

        program.fixed_point_division().add_check(program.bind_virtual(encoded_numerator),
                                                 program.bind_virtual(encoded_denominator),
                                                 program.bind_virtual(denominator_abs),
                                                 program.bind_virtual(denominator_is_negative),
                                                 program.bind_virtual(encoded_quotient),
                                                 program.bind_virtual(remainder_abs),
                                                 numerator_encoding,
                                                 denominator_encoding,
                                                 quotient_encoding,
                                                 denominator_bit_bound,
                                                 domain_size);
    });
}

bool run_signed_exp_program(ProofRuntime& runtime,
                            const ProtocolPolynomialInput& encoded_values,
                            const ProtocolPolynomialInput& exp_values,
                            const fixedPointEncoding& encoding,
                            size_t domain_size)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.exp_check().add_check(
            program.bind_virtual(encoded_values), program.bind_virtual(exp_values), encoding, domain_size);
    });
}

bool run_shifted_cube_softmax_program(ProofRuntime& runtime,
                                      const ProtocolPolynomialInput& scores,
                                      const ProtocolPolynomialInput& shifted_scores,
                                      const ProtocolPolynomialInput& shifts,
                                      const ProtocolPolynomialInput& softmax,
                                      const fixedPointEncoding& encoding,
                                      size_t outer_count,
                                      size_t middle_count,
                                      size_t inner_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.cube_middle_softmax().add_shifted_check(program.bind_virtual(scores),
                                                        program.bind_virtual(shifted_scores),
                                                        program.bind_virtual(shifts),
                                                        program.bind_virtual(softmax),
                                                        encoding,
                                                        outer_count,
                                                        middle_count,
                                                        inner_count);
    });
}

bool run_cube_middle_argmax_program(ProofRuntime& runtime,
                                    const ProtocolPolynomialInput& scores,
                                    const ProtocolPolynomialInput& max_values,
                                    const ProtocolPolynomialInput& selector,
                                    size_t outer_count,
                                    size_t middle_count,
                                    size_t inner_count,
                                    size_t bit_bound)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.cube_middle_argmax().add_check(program.bind_virtual(scores),
                                               program.bind_virtual(max_values),
                                               program.bind_virtual(selector),
                                               outer_count,
                                               middle_count,
                                               inner_count,
                                               bit_bound);
    });
}

bool run_row_max_shifted_cube_softmax_program(ProofRuntime& runtime,
                                              const ProtocolPolynomialInput& scores,
                                              const ProtocolPolynomialInput& shifted_scores,
                                              const ProtocolPolynomialInput& shifts,
                                              const ProtocolPolynomialInput& shift_selector,
                                              const ProtocolPolynomialInput& softmax,
                                              const fixedPointEncoding& encoding,
                                              size_t outer_count,
                                              size_t middle_count,
                                              size_t inner_count)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        program.cube_middle_softmax().add_row_max_shifted_check(program.bind_virtual(scores),
                                                                program.bind_virtual(shifted_scores),
                                                                program.bind_virtual(shifts),
                                                                program.bind_virtual(shift_selector),
                                                                program.bind_virtual(softmax),
                                                                encoding,
                                                                outer_count,
                                                                middle_count,
                                                                inner_count);
    });
}

} // namespace

class ProofRuntimeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ProofRuntimeTest, LookupProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly query = poly_from_domain_ints({5, 7, 7, 11});
    const poly table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(512, 512));
    EXPECT_TRUE(run_lookup_program(prover,
                                   WitnessCommitmentInput::prover("lookup.query", query, 3),
                                   PublicPolynomialInput("lookup.table", table, 7),
                                   4,
                                   8));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_lookup_program(verifier,
                                   WitnessCommitmentInput::verifier("lookup.query", 3),
                                   PublicPolynomialInput("lookup.table", table, 7),
                                   4,
                                   8));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, LookupProgramSupportsVerifierUsingPublicCommitments)
{
    const poly query = poly_from_domain_ints({5, 7, 7, 11});
    const poly table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});

    kzgProtocol public_commitment_builder;
    public_commitment_builder.setup(testing_kzg_public_parameters(512, 512));
    const Group1 table_commitment = public_commitment_builder.commit_public("lookup.table", table, 7).com;

    const ProofArtifact artifact = prove_with_runtime(testing_kzg_public_parameters(512, 512), [&](ProofRuntime& runtime) {
        return run_lookup_program(runtime,
                                  WitnessCommitmentInput::prover("lookup.query", query, 3),
                                  PublicCommitmentInput::prover("lookup.table", table, 7),
                                  4,
                                  8);
    });

    const VerificationResult result = verify_with_runtime(testing_kzg_public_parameters(512, 512), artifact, [&](ProofRuntime& runtime) {
        return run_lookup_program(runtime,
                                  WitnessCommitmentInput::verifier("lookup.query", 3),
                                  PublicCommitmentInput::verifier("lookup.table", table_commitment, 7),
                                  4,
                                  8);
    });
    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ProofRuntimeTest, ConcatenatedTableLookupProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly query_tree = matrix_poly_from_rows({
        {2},
        {4},
    });
    const poly table_tree = matrix_poly_from_rows({
        {1},
        {2},
        {3},
        {4},
    });
    const poly query_left_box = matrix_poly_from_rows({
        {20, 21},
        {40, 41},
    });
    const poly table_left_box = matrix_poly_from_rows({
        {10, 11},
        {20, 21},
        {30, 31},
        {40, 41},
    });
    const poly query_score = matrix_poly_from_rows({
        {200},
        {400},
    });
    const poly table_score = matrix_poly_from_rows({
        {100},
        {200},
        {300},
        {400},
    });

    const std::vector<ProtocolPolynomialInput> prover_queries = {
        WitnessCommitmentInput::prover("concat.query_tree", query_tree, 1),
        WitnessCommitmentInput::prover("concat.query_left_box", query_left_box, 3),
        WitnessCommitmentInput::prover("concat.query_score", query_score, 1),
    };
    const std::vector<ProtocolPolynomialInput> prover_tables = {
        PublicPolynomialInput("concat.table_tree", table_tree, 3),
        PublicPolynomialInput("concat.table_left_box", table_left_box, 7),
        PublicPolynomialInput("concat.table_score", table_score, 3),
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_concatenated_table_lookup_program(
        prover, prover_queries, prover_tables, {1, 2, 1}, 2, 4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    const std::vector<ProtocolPolynomialInput> verifier_queries = {
        WitnessCommitmentInput::verifier("concat.query_tree", 1),
        WitnessCommitmentInput::verifier("concat.query_left_box", 3),
        WitnessCommitmentInput::verifier("concat.query_score", 1),
    };
    const std::vector<ProtocolPolynomialInput> verifier_tables = {
        PublicPolynomialInput("concat.table_tree", table_tree, 3),
        PublicPolynomialInput("concat.table_left_box", table_left_box, 7),
        PublicPolynomialInput("concat.table_score", table_score, 3),
    };

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_concatenated_table_lookup_program(
        verifier, verifier_queries, verifier_tables, {1, 2, 1}, 2, 4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ConcatenatedTableLookupProgramSupportsVerifierUsingPublicCommitments)
{
    const poly query_tree = matrix_poly_from_rows({
        {2},
        {4},
    });
    const poly table_tree = matrix_poly_from_rows({
        {1},
        {2},
        {3},
        {4},
    });
    const poly query_left_box = matrix_poly_from_rows({
        {20, 21},
        {40, 41},
    });
    const poly table_left_box = matrix_poly_from_rows({
        {10, 11},
        {20, 21},
        {30, 31},
        {40, 41},
    });
    const poly query_score = matrix_poly_from_rows({
        {200},
        {400},
    });
    const poly table_score = matrix_poly_from_rows({
        {100},
        {200},
        {300},
        {400},
    });

    kzgProtocol public_commitment_builder;
    public_commitment_builder.setup(testing_kzg_public_parameters(4096, 4096));
    const Group1 tree_commitment = public_commitment_builder.commit_public("concat.table_tree", table_tree, 3).com;
    const Group1 left_box_commitment =
        public_commitment_builder.commit_public("concat.table_left_box", table_left_box, 7).com;
    const Group1 score_commitment = public_commitment_builder.commit_public("concat.table_score", table_score, 3).com;

    const ProofArtifact artifact = prove_with_runtime(testing_kzg_public_parameters(4096, 4096), [&](ProofRuntime& runtime) {
        return run_concatenated_table_lookup_program(
            runtime,
            {WitnessCommitmentInput::prover("concat.query_tree", query_tree, 1),
             WitnessCommitmentInput::prover("concat.query_left_box", query_left_box, 3),
             WitnessCommitmentInput::prover("concat.query_score", query_score, 1)},
            {PublicCommitmentInput::prover("concat.table_tree", table_tree, 3),
             PublicCommitmentInput::prover("concat.table_left_box", table_left_box, 7),
             PublicCommitmentInput::prover("concat.table_score", table_score, 3)},
            {1, 2, 1},
            2,
            4);
    });

    const VerificationResult result =
        verify_with_runtime(testing_kzg_public_parameters(4096, 4096), artifact, [&](ProofRuntime& runtime) {
            return run_concatenated_table_lookup_program(
                runtime,
                {WitnessCommitmentInput::verifier("concat.query_tree", 1),
                 WitnessCommitmentInput::verifier("concat.query_left_box", 3),
                 WitnessCommitmentInput::verifier("concat.query_score", 1)},
                {PublicCommitmentInput::verifier("concat.table_tree", tree_commitment, 3),
                 PublicCommitmentInput::verifier("concat.table_left_box", left_box_commitment, 7),
                 PublicCommitmentInput::verifier("concat.table_score", score_commitment, 3)},
                {1, 2, 1},
                2,
                4);
        });

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ProofRuntimeTest, CqConcatenatedTableLookupProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const std::vector<poly> tables = {
        matrix_poly_from_rows({{1}, {2}, {3}, {4}}),
        matrix_poly_from_rows({{10, 11}, {20, 21}, {30, 31}, {40, 41}}),
        matrix_poly_from_rows({{100}, {200}, {300}, {400}}),
    };
    const std::vector<size_t> column_counts = {1, 2, 1};
    const cqConcatenatedTableLookupPublicTable public_table = build_cq_concatenated_table_lookup_public_table(
        testing_kzg_public_parameters(4096, 4096), "cq_concat.table", tables, 4, column_counts, 15);

    const std::vector<ProtocolPolynomialInput> prover_queries = {
        WitnessCommitmentInput::prover("cq_concat.query_tree", matrix_poly_from_rows({{2}, {4}}), 1),
        WitnessCommitmentInput::prover("cq_concat.query_box", matrix_poly_from_rows({{20, 21}, {40, 41}}), 3),
        WitnessCommitmentInput::prover("cq_concat.query_score", matrix_poly_from_rows({{200}, {400}}), 1),
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_cq_concatenated_table_lookup_program(prover, prover_queries, public_table, 2));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    const std::vector<ProtocolPolynomialInput> verifier_queries = {
        WitnessCommitmentInput::verifier("cq_concat.query_tree", 1),
        WitnessCommitmentInput::verifier("cq_concat.query_box", 3),
        WitnessCommitmentInput::verifier("cq_concat.query_score", 1),
    };
    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_cq_concatenated_table_lookup_program(
        verifier, verifier_queries, public_table.verifier_view(), 2));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestRowFetchProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const forestRowFetchPublicTable table = build_forest_row_fetch_public_table(
        testing_kzg_public_parameters(4096, 4096),
        "forest.fetch.table",
        matrix_poly_from_rows({{0}, {1}, {2}, {3}}),
        matrix_poly_from_rows({{10, 11}, {20, 21}, {30, 31}, {40, 41}}),
        matrix_poly_from_rows({{12, 13}, {22, 23}, {32, 33}, {42, 43}}),
        matrix_poly_from_rows({{100}, {200}, {300}, {400}}),
        4,
        2,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}}),
        15);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_row_fetch_program(
        prover,
        table,
        WitnessCommitmentInput::prover("forest.fetch.query_tree", matrix_poly_from_rows({{1}, {3}}), 1),
        WitnessCommitmentInput::prover("forest.fetch.query_lower", matrix_poly_from_rows({{20, 21}, {40, 41}}), 3),
        WitnessCommitmentInput::prover("forest.fetch.query_upper", matrix_poly_from_rows({{22, 23}, {42, 43}}), 3),
        WitnessCommitmentInput::prover("forest.fetch.query_score", matrix_poly_from_rows({{200}, {400}}), 1),
        WitnessCommitmentInput::prover("forest.fetch.query_leaf", matrix_poly_from_rows({{8}, {10}}), 1),
        2));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_row_fetch_program(verifier,
                                             table.verifier_view(),
                                             WitnessCommitmentInput::verifier("forest.fetch.query_tree", 1),
                                             WitnessCommitmentInput::verifier("forest.fetch.query_lower", 3),
                                             WitnessCommitmentInput::verifier("forest.fetch.query_upper", 3),
                                             WitnessCommitmentInput::verifier("forest.fetch.query_score", 1),
                                             WitnessCommitmentInput::verifier("forest.fetch.query_leaf", 1),
                                             2));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestInferenceProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const forestInferencePublicTable table = build_forest_inference_public_table(
        testing_kzg_public_parameters(4096, 4096),
        "forest.infer.table",
        matrix_poly_from_rows({{0}, {1}, {2}, {3}}),
        matrix_poly_from_rows({{10, 11}, {20, 21}, {30, 31}, {40, 41}}),
        matrix_poly_from_rows({{12, 13}, {22, 23}, {32, 33}, {42, 43}}),
        matrix_poly_from_rows({{100}, {200}, {300}, {400}}),
        4,
        2,
        6,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}}),
        15);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_inference_program(
        prover,
        table,
        WitnessCommitmentInput::prover("forest.infer.query_tree", matrix_poly_from_rows({{1}, {3}}), 1),
        WitnessCommitmentInput::prover("forest.infer.inputs", matrix_poly_from_rows({{20, 21}, {40, 41}}), 3),
        WitnessCommitmentInput::prover("forest.infer.query_lower", matrix_poly_from_rows({{20, 21}, {40, 41}}), 3),
        WitnessCommitmentInput::prover("forest.infer.query_upper", matrix_poly_from_rows({{22, 23}, {42, 43}}), 3),
        WitnessCommitmentInput::prover("forest.infer.query_score", matrix_poly_from_rows({{200}, {400}}), 1),
        WitnessCommitmentInput::prover("forest.infer.query_leaf", matrix_poly_from_rows({{8}, {10}}), 1),
        2));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_inference_program(verifier,
                                             table.verifier_view(),
                                             WitnessCommitmentInput::verifier("forest.infer.query_tree", 1),
                                             WitnessCommitmentInput::verifier("forest.infer.inputs", 3),
                                             WitnessCommitmentInput::verifier("forest.infer.query_lower", 3),
                                             WitnessCommitmentInput::verifier("forest.infer.query_upper", 3),
                                             WitnessCommitmentInput::verifier("forest.infer.query_score", 1),
                                             WitnessCommitmentInput::verifier("forest.infer.query_leaf", 1),
                                             2));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestPredictionProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const forestPredictionPublicTable table = build_forest_prediction_public_table(2, 2, 3);
    const poly query_scores = matrix_poly_from_rows({{1}, {2}, {3}, {1}, {2}, {1}, {1}, {4}});
    const poly class_scores = matrix_poly_from_rows({{3, 3}, {4, 5}});
    const poly predicted_scores = matrix_poly_from_rows({{4, 5}});
    const poly class_selector = matrix_poly_from_rows({{0, 0}, {1, 1}});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_prediction_program(prover,
                                              table,
                                              WitnessCommitmentInput::prover("forest_prediction.scores", query_scores, 7),
                                              WitnessCommitmentInput::prover(
                                                  "forest_prediction.class_scores", class_scores, 3),
                                              WitnessCommitmentInput::prover(
                                                  "forest_prediction.predicted_scores", predicted_scores, 1),
                                              WitnessCommitmentInput::prover(
                                                  "forest_prediction.class_selector", class_selector, 3),
                                              8));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_prediction_program(verifier,
                                              table.verifier_view(),
                                              WitnessCommitmentInput::verifier("forest_prediction.scores", 7),
                                              WitnessCommitmentInput::verifier("forest_prediction.class_scores", 3),
                                              WitnessCommitmentInput::verifier("forest_prediction.predicted_scores", 1),
                                              WitnessCommitmentInput::verifier("forest_prediction.class_selector", 3),
                                              8));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestTreeLayoutProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const forestTreeLayoutPublicTable table =
        build_forest_tree_layout_public_table("forest.tree_layout.table", 2, 2);
    const poly query_tree = matrix_poly_from_rows({{1}, {1}, {2}, {2}, {3}, {3}, {4}, {4}});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_tree_layout_program(
        prover, table, WitnessCommitmentInput::prover("forest.tree_layout.query_tree", query_tree, 7), 8));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_tree_layout_program(
        verifier, table.verifier_view(), WitnessCommitmentInput::verifier("forest.tree_layout.query_tree", 7), 8));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ParentChildBoxInheritanceProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly selector = matrix_poly_from_rows({
        {1, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    const poly right_child_indices = poly_from_domain_ints({2, 0, 0, 0});
    const poly parent_lower = matrix_poly_from_rows({
        {1, 10},
        {1, 10},
        {3, 10},
        {0, 0},
    });
    const poly parent_upper = matrix_poly_from_rows({
        {5, 20},
        {3, 20},
        {5, 20},
        {0, 0},
    });
    const poly left_lower = matrix_poly_from_rows({
        {1, 10},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly left_upper = matrix_poly_from_rows({
        {3, 20},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly right_lower = matrix_poly_from_rows({
        {3, 10},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly right_upper = matrix_poly_from_rows({
        {5, 20},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly thresholds = poly_from_domain_ints({3, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_parent_child_box_inheritance_program(
        prover,
        WitnessCommitmentInput::prover("pcbi.selector", selector, 7),
        WitnessCommitmentInput::prover("pcbi.mask", mask, 3),
        WitnessCommitmentInput::prover("pcbi.left_child_indices", left_child_indices, 3),
        WitnessCommitmentInput::prover("pcbi.right_child_indices", right_child_indices, 3),
        WitnessCommitmentInput::prover("pcbi.parent_lower", parent_lower, 7),
        WitnessCommitmentInput::prover("pcbi.parent_upper", parent_upper, 7),
        WitnessCommitmentInput::prover("pcbi.left_lower", left_lower, 7),
        WitnessCommitmentInput::prover("pcbi.left_upper", left_upper, 7),
        WitnessCommitmentInput::prover("pcbi.right_lower", right_lower, 7),
        WitnessCommitmentInput::prover("pcbi.right_upper", right_upper, 7),
        WitnessCommitmentInput::prover("pcbi.thresholds", thresholds, 3),
        4,
        2,
        6));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_parent_child_box_inheritance_program(
        verifier,
        WitnessCommitmentInput::verifier("pcbi.selector", 7),
        WitnessCommitmentInput::verifier("pcbi.mask", 3),
        WitnessCommitmentInput::verifier("pcbi.left_child_indices", 3),
        WitnessCommitmentInput::verifier("pcbi.right_child_indices", 3),
        WitnessCommitmentInput::verifier("pcbi.parent_lower", 7),
        WitnessCommitmentInput::verifier("pcbi.parent_upper", 7),
        WitnessCommitmentInput::verifier("pcbi.left_lower", 7),
        WitnessCommitmentInput::verifier("pcbi.left_upper", 7),
        WitnessCommitmentInput::verifier("pcbi.right_lower", 7),
        WitnessCommitmentInput::verifier("pcbi.right_upper", 7),
        WitnessCommitmentInput::verifier("pcbi.thresholds", 3),
        4,
        2,
        6));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, RootSelectorConsistencyProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly root_indices = poly_from_domain_ints({1, 3});
    const poly node_tree_ids = poly_from_domain_ints({1, 1, 2, 2});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_root_selector_consistency_program(prover,
                                                      WitnessCommitmentInput::prover(
                                                          "root_selector.root_indices", root_indices, 1),
                                                      WitnessCommitmentInput::prover(
                                                          "root_selector.node_tree_ids", node_tree_ids, 3),
                                                      2,
                                                      4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_root_selector_consistency_program(
        verifier,
        WitnessCommitmentInput::verifier("root_selector.root_indices", 1),
        WitnessCommitmentInput::verifier("root_selector.node_tree_ids", 3),
        2,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ParentChildTreeConsistencyProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly left_child_indices = poly_from_domain_ints({1, 0, 3, 0});
    const poly right_child_indices = poly_from_domain_ints({0, 0, 3, 0});
    const poly mask = poly_from_domain_ints({1, 0, 1, 0});
    const poly node_tree_ids = poly_from_domain_ints({1, 1, 2, 2});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_parent_child_tree_consistency_program(
        prover,
        WitnessCommitmentInput::prover("pctc.left_indices", left_child_indices, 3),
        WitnessCommitmentInput::prover("pctc.right_indices", right_child_indices, 3),
        WitnessCommitmentInput::prover("pctc.mask", mask, 3),
        WitnessCommitmentInput::prover("pctc.node_tree_ids", node_tree_ids, 3),
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_parent_child_tree_consistency_program(
        verifier,
        WitnessCommitmentInput::verifier("pctc.left_indices", 3),
        WitnessCommitmentInput::verifier("pctc.right_indices", 3),
        WitnessCommitmentInput::verifier("pctc.mask", 3),
        WitnessCommitmentInput::verifier("pctc.node_tree_ids", 3),
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, NodeRoleConsistencyProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly root_indices = poly_from_domain_ints({0, 3});
    const poly node_leaf_mask = poly_from_domain_ints({0, 1, 1, 1});
    const poly left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    const poly right_child_indices = poly_from_domain_ints({2, 0, 0, 0});
    const poly parent_mask = poly_from_domain_ints({1, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_node_role_consistency_program(
        prover,
        WitnessCommitmentInput::prover("nrc.root_indices", root_indices, 1),
        WitnessCommitmentInput::prover("nrc.node_leaf_mask", node_leaf_mask, 3),
        WitnessCommitmentInput::prover("nrc.left_indices", left_child_indices, 3),
        WitnessCommitmentInput::prover("nrc.right_indices", right_child_indices, 3),
        WitnessCommitmentInput::prover("nrc.parent_mask", parent_mask, 3),
        2,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_node_role_consistency_program(
        verifier,
        WitnessCommitmentInput::verifier("nrc.root_indices", 1),
        WitnessCommitmentInput::verifier("nrc.node_leaf_mask", 3),
        WitnessCommitmentInput::verifier("nrc.left_indices", 3),
        WitnessCommitmentInput::verifier("nrc.right_indices", 3),
        WitnessCommitmentInput::verifier("nrc.parent_mask", 3),
        2,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestStructureProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly root_indices = poly_from_domain_ints({0, 3});
    const poly parent_mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    const poly right_child_indices = poly_from_domain_ints({2, 0, 0, 0});
    const poly node_tree_ids = poly_from_domain_ints({1, 1, 1, 2});
    const poly feature_selector = matrix_poly_from_rows({
        {1, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly lower_bounds = matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {2, 0},
        {0, 0},
    });
    const poly upper_bounds = matrix_poly_from_rows({
        {8, 8},
        {2, 8},
        {8, 8},
        {8, 8},
    });
    const poly left_lower = matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly left_upper = matrix_poly_from_rows({
        {2, 8},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly right_lower = matrix_poly_from_rows({
        {2, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly right_upper = matrix_poly_from_rows({
        {8, 8},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly thresholds = poly_from_domain_ints({2, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_structure_program(prover,
                                             WitnessCommitmentInput::prover("fs.root_indices", root_indices, 1),
                                             WitnessCommitmentInput::prover("fs.parent_mask", parent_mask, 3),
                                             WitnessCommitmentInput::prover("fs.left_child_indices", left_child_indices, 3),
                                             WitnessCommitmentInput::prover("fs.right_child_indices", right_child_indices, 3),
                                             WitnessCommitmentInput::prover("fs.node_tree_ids", node_tree_ids, 3),
                                             WitnessCommitmentInput::prover("fs.feature_selector", feature_selector, 7),
                                             WitnessCommitmentInput::prover("fs.lower_bounds", lower_bounds, 7),
                                             WitnessCommitmentInput::prover("fs.upper_bounds", upper_bounds, 7),
                                             WitnessCommitmentInput::prover("fs.left_lower", left_lower, 7),
                                             WitnessCommitmentInput::prover("fs.left_upper", left_upper, 7),
                                             WitnessCommitmentInput::prover("fs.right_lower", right_lower, 7),
                                             WitnessCommitmentInput::prover("fs.right_upper", right_upper, 7),
                                             WitnessCommitmentInput::prover("fs.thresholds", thresholds, 3),
                                             2,
                                             4,
                                             2,
                                             8,
                                             4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_structure_program(verifier,
                                             WitnessCommitmentInput::verifier("fs.root_indices", 1),
                                             WitnessCommitmentInput::verifier("fs.parent_mask", 3),
                                             WitnessCommitmentInput::verifier("fs.left_child_indices", 3),
                                             WitnessCommitmentInput::verifier("fs.right_child_indices", 3),
                                             WitnessCommitmentInput::verifier("fs.node_tree_ids", 3),
                                             WitnessCommitmentInput::verifier("fs.feature_selector", 7),
                                             WitnessCommitmentInput::verifier("fs.lower_bounds", 7),
                                             WitnessCommitmentInput::verifier("fs.upper_bounds", 7),
                                             WitnessCommitmentInput::verifier("fs.left_lower", 7),
                                             WitnessCommitmentInput::verifier("fs.left_upper", 7),
                                             WitnessCommitmentInput::verifier("fs.right_lower", 7),
                                             WitnessCommitmentInput::verifier("fs.right_upper", 7),
                                             WitnessCommitmentInput::verifier("fs.thresholds", 3),
                                             2,
                                             4,
                                             2,
                                             8,
                                             4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, TreeHistogramConsistencyProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly full_histogram = matrix_poly_from_rows({
        {6, 8, 10, 12},
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {0, 0, 0, 0},
    });
    const poly leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {0, 0, 0, 0},
    });
    const poly parent_mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    const poly right_child_indices = poly_from_domain_ints({2, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_tree_histogram_consistency_program(
        prover,
        WitnessCommitmentInput::prover("tree_hist.full", full_histogram, 15),
        WitnessCommitmentInput::prover("tree_hist.leaf", leaf_histogram, 15),
        WitnessCommitmentInput::prover("tree_hist.parent_mask", parent_mask, 3),
        WitnessCommitmentInput::prover("tree_hist.left_child", left_child_indices, 3),
        WitnessCommitmentInput::prover("tree_hist.right_child", right_child_indices, 3),
        4,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_tree_histogram_consistency_program(
        verifier,
        WitnessCommitmentInput::verifier("tree_hist.full", 15),
        WitnessCommitmentInput::verifier("tree_hist.leaf", 15),
        WitnessCommitmentInput::verifier("tree_hist.parent_mask", 3),
        WitnessCommitmentInput::verifier("tree_hist.left_child", 3),
        WitnessCommitmentInput::verifier("tree_hist.right_child", 3),
        4,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestHistogramProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly flattened_bucket_indices = poly_from_domain_ints({4, 5, 8, 11});
    const poly values = poly_from_domain_ints({1, 2, 5, 8});
    const poly leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {1, 2, 0, 0},
        {5, 0, 0, 8},
        {0, 0, 0, 0},
    });
    const poly full_histogram = matrix_poly_from_rows({
        {6, 2, 0, 8},
        {1, 2, 0, 0},
        {5, 0, 0, 8},
        {0, 0, 0, 0},
    });
    const poly parent_mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    const poly right_child_indices = poly_from_domain_ints({2, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_histogram_program(
        prover,
        WitnessCommitmentInput::prover("forest_hist.indices", flattened_bucket_indices, 3),
        WitnessCommitmentInput::prover("forest_hist.values", values, 3),
        WitnessCommitmentInput::prover("forest_hist.leaf", leaf_histogram, 15),
        WitnessCommitmentInput::prover("forest_hist.full", full_histogram, 15),
        WitnessCommitmentInput::prover("forest_hist.parent_mask", parent_mask, 3),
        WitnessCommitmentInput::prover("forest_hist.left_child", left_child_indices, 3),
        WitnessCommitmentInput::prover("forest_hist.right_child", right_child_indices, 3),
        4,
        4,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_histogram_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_hist.indices", 3),
        WitnessCommitmentInput::verifier("forest_hist.values", 3),
        WitnessCommitmentInput::verifier("forest_hist.leaf", 15),
        WitnessCommitmentInput::verifier("forest_hist.full", 15),
        WitnessCommitmentInput::verifier("forest_hist.parent_mask", 3),
        WitnessCommitmentInput::verifier("forest_hist.left_child", 3),
        WitnessCommitmentInput::verifier("forest_hist.right_child", 3),
        4,
        4,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestHistogramBucketAddressProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly leaf_indices = poly_from_domain_ints({1, 2, 0, 3});
    const poly lifted_leaf_indices = matrix_poly_from_rows({
        {1, 1},
        {2, 2},
        {0, 0},
        {3, 3},
    });
    const poly lifted_inputs = matrix_poly_from_rows({
        {0, 3},
        {1, 2},
        {2, 0},
        {3, 1},
    });
    const poly flattened_bucket_indices = matrix_poly_from_rows({
        {8, 15},
        {17, 22},
        {2, 4},
        {27, 29},
    });

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_histogram_bucket_address_program(
        prover,
        WitnessCommitmentInput::prover("forest_hist_addr.leaf_indices", leaf_indices, 3),
        WitnessCommitmentInput::prover("forest_hist_addr.lifted_leaf_indices", lifted_leaf_indices, 7),
        WitnessCommitmentInput::prover("forest_hist_addr.lifted_inputs", lifted_inputs, 7),
        WitnessCommitmentInput::prover("forest_hist_addr.flattened_bucket_indices", flattened_bucket_indices, 7),
        4,
        2,
        4,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_histogram_bucket_address_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_hist_addr.leaf_indices", 3),
        WitnessCommitmentInput::verifier("forest_hist_addr.lifted_leaf_indices", 7),
        WitnessCommitmentInput::verifier("forest_hist_addr.lifted_inputs", 7),
        WitnessCommitmentInput::verifier("forest_hist_addr.flattened_bucket_indices", 7),
        4,
        2,
        4,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestHistogramMultiChannelProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly flattened_bucket_indices = poly_from_domain_ints({4, 5, 8, 11});
    const poly count_values = poly_from_domain_ints({1, 1, 1, 1});
    const poly grad_values = poly_from_domain_ints({10, 20, 30, 40});
    const poly hess_values = poly_from_domain_ints({3, 5, 7, 11});
    const poly count_leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {1, 1, 0, 0},
        {1, 0, 0, 1},
        {0, 0, 0, 0},
    });
    const poly grad_leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {10, 20, 0, 0},
        {30, 0, 0, 40},
        {0, 0, 0, 0},
    });
    const poly hess_leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {3, 5, 0, 0},
        {7, 0, 0, 11},
        {0, 0, 0, 0},
    });
    const poly count_full_histogram = matrix_poly_from_rows({
        {2, 1, 0, 1},
        {1, 1, 0, 0},
        {1, 0, 0, 1},
        {0, 0, 0, 0},
    });
    const poly grad_full_histogram = matrix_poly_from_rows({
        {40, 20, 0, 40},
        {10, 20, 0, 0},
        {30, 0, 0, 40},
        {0, 0, 0, 0},
    });
    const poly hess_full_histogram = matrix_poly_from_rows({
        {10, 5, 0, 11},
        {3, 5, 0, 0},
        {7, 0, 0, 11},
        {0, 0, 0, 0},
    });
    const poly parent_mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    const poly right_child_indices = poly_from_domain_ints({2, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_histogram_multi_program(
        prover,
        WitnessCommitmentInput::prover("forest_hist_multi.indices", flattened_bucket_indices, 3),
        {
            WitnessCommitmentInput::prover("forest_hist_multi.count_values", count_values, 3),
            WitnessCommitmentInput::prover("forest_hist_multi.grad_values", grad_values, 3),
            WitnessCommitmentInput::prover("forest_hist_multi.hess_values", hess_values, 3),
        },
        {
            WitnessCommitmentInput::prover("forest_hist_multi.count_leaf", count_leaf_histogram, 15),
            WitnessCommitmentInput::prover("forest_hist_multi.grad_leaf", grad_leaf_histogram, 15),
            WitnessCommitmentInput::prover("forest_hist_multi.hess_leaf", hess_leaf_histogram, 15),
        },
        {
            WitnessCommitmentInput::prover("forest_hist_multi.count_full", count_full_histogram, 15),
            WitnessCommitmentInput::prover("forest_hist_multi.grad_full", grad_full_histogram, 15),
            WitnessCommitmentInput::prover("forest_hist_multi.hess_full", hess_full_histogram, 15),
        },
        WitnessCommitmentInput::prover("forest_hist_multi.parent_mask", parent_mask, 3),
        WitnessCommitmentInput::prover("forest_hist_multi.left_child", left_child_indices, 3),
        WitnessCommitmentInput::prover("forest_hist_multi.right_child", right_child_indices, 3),
        4,
        4,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_histogram_multi_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_hist_multi.indices", 3),
        {
            WitnessCommitmentInput::verifier("forest_hist_multi.count_values", 3),
            WitnessCommitmentInput::verifier("forest_hist_multi.grad_values", 3),
            WitnessCommitmentInput::verifier("forest_hist_multi.hess_values", 3),
        },
        {
            WitnessCommitmentInput::verifier("forest_hist_multi.count_leaf", 15),
            WitnessCommitmentInput::verifier("forest_hist_multi.grad_leaf", 15),
            WitnessCommitmentInput::verifier("forest_hist_multi.hess_leaf", 15),
        },
        {
            WitnessCommitmentInput::verifier("forest_hist_multi.count_full", 15),
            WitnessCommitmentInput::verifier("forest_hist_multi.grad_full", 15),
            WitnessCommitmentInput::verifier("forest_hist_multi.hess_full", 15),
        },
        WitnessCommitmentInput::verifier("forest_hist_multi.parent_mask", 3),
        WitnessCommitmentInput::verifier("forest_hist_multi.left_child", 3),
        WitnessCommitmentInput::verifier("forest_hist_multi.right_child", 3),
        4,
        4,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestSplitStatisticsMultiChannelProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly count_histogram = matrix_poly_from_rows({
        {1, 1, 1, 1, 0, 1, 0, 1},
        {2, 0, 2, 0, 1, 1, 1, 1},
    });
    const poly grad_histogram = matrix_poly_from_rows({
        {1, 2, 4, 8, 3, 5, 7, 9},
        {0, 1, 0, 1, 2, 0, 2, 0},
    });
    const poly hess_histogram = matrix_poly_from_rows({
        {2, 3, 5, 7, 11, 13, 17, 19},
        {23, 29, 31, 37, 41, 43, 47, 53},
    });
    const poly count_left_statistics = matrix_poly_from_rows({
        {1, 2, 3, 4, 0, 1, 1, 2},
        {2, 2, 4, 4, 1, 2, 3, 4},
    });
    const poly grad_left_statistics = matrix_poly_from_rows({
        {1, 3, 7, 15, 3, 8, 15, 24},
        {0, 1, 1, 2, 2, 2, 4, 4},
    });
    const poly hess_left_statistics = matrix_poly_from_rows({
        {2, 5, 10, 17, 11, 24, 41, 60},
        {23, 52, 83, 120, 41, 84, 131, 184},
    });
    const poly count_right_statistics = matrix_poly_from_rows({
        {3, 2, 1, 0, 2, 1, 1, 0},
        {2, 2, 0, 0, 3, 2, 1, 0},
    });
    const poly grad_right_statistics = matrix_poly_from_rows({
        {14, 12, 8, 0, 21, 16, 9, 0},
        {2, 1, 1, 0, 2, 2, 0, 0},
    });
    const poly hess_right_statistics = matrix_poly_from_rows({
        {15, 12, 7, 0, 49, 36, 19, 0},
        {97, 68, 37, 0, 143, 100, 53, 0},
    });

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_split_statistics_multi_program(
        prover,
        {
            WitnessCommitmentInput::prover("forest_split_stats.count_hist", count_histogram, 15),
            WitnessCommitmentInput::prover("forest_split_stats.grad_hist", grad_histogram, 15),
            WitnessCommitmentInput::prover("forest_split_stats.hess_hist", hess_histogram, 15),
        },
        {
            WitnessCommitmentInput::prover("forest_split_stats.count_left", count_left_statistics, 15),
            WitnessCommitmentInput::prover("forest_split_stats.grad_left", grad_left_statistics, 15),
            WitnessCommitmentInput::prover("forest_split_stats.hess_left", hess_left_statistics, 15),
        },
        {
            WitnessCommitmentInput::prover("forest_split_stats.count_right", count_right_statistics, 15),
            WitnessCommitmentInput::prover("forest_split_stats.grad_right", grad_right_statistics, 15),
            WitnessCommitmentInput::prover("forest_split_stats.hess_right", hess_right_statistics, 15),
        },
        2,
        2,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_split_statistics_multi_program(
        verifier,
        {
            WitnessCommitmentInput::verifier("forest_split_stats.count_hist", 15),
            WitnessCommitmentInput::verifier("forest_split_stats.grad_hist", 15),
            WitnessCommitmentInput::verifier("forest_split_stats.hess_hist", 15),
        },
        {
            WitnessCommitmentInput::verifier("forest_split_stats.count_left", 15),
            WitnessCommitmentInput::verifier("forest_split_stats.grad_left", 15),
            WitnessCommitmentInput::verifier("forest_split_stats.hess_left", 15),
        },
        {
            WitnessCommitmentInput::verifier("forest_split_stats.count_right", 15),
            WitnessCommitmentInput::verifier("forest_split_stats.grad_right", 15),
            WitnessCommitmentInput::verifier("forest_split_stats.hess_right", 15),
        },
        2,
        2,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestSplitGainProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const std::vector<std::vector<int>> left_gradients = {
        {1, -2, 0, 3, -1, 2, -3, 1},
        {2, -1, 1, -2, 0, 3, -1, 2},
    };
    const std::vector<std::vector<int>> right_gradients = {
        {2, 1, -1, 0, 3, -2, 1, -1},
        {-1, 2, 0, 1, 2, -1, 3, 0},
    };
    const std::vector<std::vector<int>> left_hessians = {
        {1, 2, 1, 3, 2, 1, 2, 1},
        {1, 1, 2, 2, 1, 2, 1, 3},
    };
    const std::vector<std::vector<int>> right_hessians = {
        {1, 1, 2, 1, 2, 1, 1, 2},
        {2, 1, 1, 1, 1, 1, 2, 1},
    };
    const int lambda = 2;
    const int split_penalty = 1;

    const std::vector<std::vector<int>> parent_gradients = add_matrices(left_gradients, right_gradients);
    const std::vector<std::vector<int>> left_squares = square_matrix(left_gradients);
    const std::vector<std::vector<int>> right_squares = square_matrix(right_gradients);
    const std::vector<std::vector<int>> parent_squares = square_matrix(parent_gradients);
    const std::vector<std::vector<int>> gain_numerators =
        gain_numerator_matrix(left_gradients, right_gradients, left_hessians, right_hessians, lambda, split_penalty);
    const std::vector<std::vector<int>> gain_denominators =
        gain_denominator_matrix(left_hessians, right_hessians, lambda);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_split_gain_program(
        prover,
        WitnessCommitmentInput::prover("forest_split_gain.left_grad", matrix_poly_from_rows(left_gradients), 15),
        WitnessCommitmentInput::prover("forest_split_gain.right_grad", matrix_poly_from_rows(right_gradients), 15),
        WitnessCommitmentInput::prover("forest_split_gain.left_hess", matrix_poly_from_rows(left_hessians), 15),
        WitnessCommitmentInput::prover("forest_split_gain.right_hess", matrix_poly_from_rows(right_hessians), 15),
        WitnessCommitmentInput::prover("forest_split_gain.left_sq", matrix_poly_from_rows(left_squares), 15),
        WitnessCommitmentInput::prover("forest_split_gain.right_sq", matrix_poly_from_rows(right_squares), 15),
        WitnessCommitmentInput::prover("forest_split_gain.parent_sq", matrix_poly_from_rows(parent_squares), 15),
        WitnessCommitmentInput::prover("forest_split_gain.gain_num", matrix_poly_from_rows(gain_numerators), 15),
        WitnessCommitmentInput::prover("forest_split_gain.gain_den", matrix_poly_from_rows(gain_denominators), 15),
        2,
        2,
        4,
        fixedPointEncoding::signed_offset_binary(6),
        4,
        5,
        lambda,
        fr_from_int(split_penalty)));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_split_gain_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_split_gain.left_grad", 15),
        WitnessCommitmentInput::verifier("forest_split_gain.right_grad", 15),
        WitnessCommitmentInput::verifier("forest_split_gain.left_hess", 15),
        WitnessCommitmentInput::verifier("forest_split_gain.right_hess", 15),
        WitnessCommitmentInput::verifier("forest_split_gain.left_sq", 15),
        WitnessCommitmentInput::verifier("forest_split_gain.right_sq", 15),
        WitnessCommitmentInput::verifier("forest_split_gain.parent_sq", 15),
        WitnessCommitmentInput::verifier("forest_split_gain.gain_num", 15),
        WitnessCommitmentInput::verifier("forest_split_gain.gain_den", 15),
        2,
        2,
        4,
        fixedPointEncoding::signed_offset_binary(6),
        4,
        5,
        lambda,
        fr_from_int(split_penalty)));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestQuantizedSplitGainProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const std::vector<std::vector<int>> gain_numerators = {
        {-30, -120, -5, -45, -108, -72, -240, -45},
        {-27, -80, -16, -108, -48, -180, -108, -75},
    };
    const std::vector<std::vector<int>> gain_denominators = {
        {18, 48, 18, 60, 48, 27, 48, 27},
        {36, 18, 48, 36, 18, 48, 48, 60},
    };

    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(10, 0);
    const fixedPointEncoding quantized_gain_encoding = fixedPointEncoding::signed_offset_binary(10, 2);
    const std::vector<std::vector<int>> quantized_gains =
        quantized_gain_matrix(gain_numerators, gain_denominators, quantized_gain_encoding.fractional_bits);
    const std::vector<std::vector<int>> remainders =
        quantized_remainder_matrix(gain_numerators, gain_denominators, quantized_gain_encoding.fractional_bits);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_quantized_split_gain_program(
        prover,
        WitnessCommitmentInput::prover("forest_quant_gain.gain_num", matrix_poly_from_rows(gain_numerators), 15),
        WitnessCommitmentInput::prover("forest_quant_gain.gain_den", matrix_poly_from_rows(gain_denominators), 15),
        WitnessCommitmentInput::prover(
            "forest_quant_gain.encoded_gain_num", fixed_point_encoded_matrix_poly_from_rows(gain_numerators, numerator_encoding), 15),
        WitnessCommitmentInput::prover("forest_quant_gain.quantized_gain",
                                       fixed_point_encoded_matrix_poly_from_rows(quantized_gains, quantized_gain_encoding),
                                       15),
        WitnessCommitmentInput::prover("forest_quant_gain.remainder", matrix_poly_from_rows(remainders), 15),
        2,
        2,
        4,
        numerator_encoding,
        quantized_gain_encoding,
        6));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_quantized_split_gain_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_quant_gain.gain_num", 15),
        WitnessCommitmentInput::verifier("forest_quant_gain.gain_den", 15),
        WitnessCommitmentInput::verifier("forest_quant_gain.encoded_gain_num", 15),
        WitnessCommitmentInput::verifier("forest_quant_gain.quantized_gain", 15),
        WitnessCommitmentInput::verifier("forest_quant_gain.remainder", 15),
        2,
        2,
        4,
        numerator_encoding,
        quantized_gain_encoding,
        6));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestSplitSelectionProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const std::vector<std::vector<int>> quantized_gains = {
        {1, 4, 2, 3, 5, 7, 6, 0},
        {2, 6, 1, 4, 3, 0, 5, 2},
    };
    const std::vector<std::vector<int>> candidate_mask = {
        {0, 1, 1, 0, 1, 1, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 0},
    };
    const std::vector<std::vector<int>> selector = {
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_split_selection_program(
        prover,
        WitnessCommitmentInput::prover("forest_split_select.quantized_gains", matrix_poly_from_rows(quantized_gains), 15),
        WitnessCommitmentInput::prover("forest_split_select.tie_break_scores",
                                       matrix_poly_from_rows({
                                           {15, 38, 21, 28, 43, 58, 49, 0},
                                           {23, 54, 13, 36, 27, 2, 41, 16},
                                       }),
                                       15),
        WitnessCommitmentInput::prover("forest_split_select.candidate_mask", matrix_poly_from_rows(candidate_mask), 15),
        WitnessCommitmentInput::prover("forest_split_select.active_row_mask", poly_from_domain_ints({1, 1}), 1),
        WitnessCommitmentInput::prover("forest_split_select.selector", matrix_poly_from_rows(selector), 15),
        WitnessCommitmentInput::prover("forest_split_select.selected_gains", poly_from_domain_ints({7, 6}), 1),
        WitnessCommitmentInput::prover("forest_split_select.selected_tie_break_scores", poly_from_domain_ints({58, 54}), 1),
        WitnessCommitmentInput::prover("forest_split_select.selected_candidate_indices", poly_from_domain_ints({5, 1}), 1),
        WitnessCommitmentInput::prover("forest_split_select.selected_features", poly_from_domain_ints({1, 0}), 1),
        WitnessCommitmentInput::prover("forest_split_select.selected_bins", poly_from_domain_ints({1, 1}), 1),
        2,
        2,
        4,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_split_selection_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_split_select.quantized_gains", 15),
        WitnessCommitmentInput::verifier("forest_split_select.tie_break_scores", 15),
        WitnessCommitmentInput::verifier("forest_split_select.candidate_mask", 15),
        WitnessCommitmentInput::verifier("forest_split_select.active_row_mask", 1),
        WitnessCommitmentInput::verifier("forest_split_select.selector", 15),
        WitnessCommitmentInput::verifier("forest_split_select.selected_gains", 1),
        WitnessCommitmentInput::verifier("forest_split_select.selected_tie_break_scores", 1),
        WitnessCommitmentInput::verifier("forest_split_select.selected_candidate_indices", 1),
        WitnessCommitmentInput::verifier("forest_split_select.selected_features", 1),
        WitnessCommitmentInput::verifier("forest_split_select.selected_bins", 1),
        2,
        2,
        4,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestSplitUpdateProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly root_indices = poly_from_domain_ints({0});
    const poly active_node_mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly current_leaf_mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly current_internal_mask = poly_from_domain_ints({0, 0, 0, 0});
    const poly next_leaf_mask = poly_from_domain_ints({0, 1, 1, 0});
    const poly next_internal_mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly next_active_node_mask = poly_from_domain_ints({0, 1, 1, 0});
    const poly left_child_occupancy = poly_from_domain_ints({0, 1, 0, 0});
    const poly right_child_occupancy = poly_from_domain_ints({0, 0, 1, 0});
    const poly left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    const poly right_child_indices = poly_from_domain_ints({2, 0, 0, 0});
    const poly node_tree_ids = poly_from_domain_ints({1, 1, 1, 0});
    const poly selected_features = poly_from_domain_ints({0, 0, 0, 0});
    const poly selected_bins = poly_from_domain_ints({1, 0, 0, 0});
    const poly feature_selector = matrix_poly_from_rows({
        {1, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly lower_bounds = matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {2, 0},
        {0, 0},
    });
    const poly upper_bounds = matrix_poly_from_rows({
        {4, 4},
        {2, 4},
        {4, 4},
        {0, 0},
    });
    const poly left_lower = matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly left_upper = matrix_poly_from_rows({
        {2, 4},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly right_lower = matrix_poly_from_rows({
        {2, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly right_upper = matrix_poly_from_rows({
        {4, 4},
        {0, 0},
        {0, 0},
        {0, 0},
    });
    const poly thresholds = poly_from_domain_ints({2, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_split_update_program(
        prover,
        WitnessCommitmentInput::prover("forest_split_update.root_indices", root_indices, 0),
        WitnessCommitmentInput::prover("forest_split_update.active_node_mask", active_node_mask, 3),
        WitnessCommitmentInput::prover("forest_split_update.current_leaf_mask", current_leaf_mask, 3),
        WitnessCommitmentInput::prover("forest_split_update.current_internal_mask", current_internal_mask, 3),
        WitnessCommitmentInput::prover("forest_split_update.next_leaf_mask", next_leaf_mask, 3),
        WitnessCommitmentInput::prover("forest_split_update.next_internal_mask", next_internal_mask, 3),
        WitnessCommitmentInput::prover("forest_split_update.next_active_node_mask", next_active_node_mask, 3),
        WitnessCommitmentInput::prover("forest_split_update.left_child_occupancy", left_child_occupancy, 3),
        WitnessCommitmentInput::prover("forest_split_update.right_child_occupancy", right_child_occupancy, 3),
        WitnessCommitmentInput::prover("forest_split_update.left_child_indices", left_child_indices, 3),
        WitnessCommitmentInput::prover("forest_split_update.right_child_indices", right_child_indices, 3),
        WitnessCommitmentInput::prover("forest_split_update.node_tree_ids", node_tree_ids, 3),
        WitnessCommitmentInput::prover("forest_split_update.selected_features", selected_features, 3),
        WitnessCommitmentInput::prover("forest_split_update.selected_bins", selected_bins, 3),
        WitnessCommitmentInput::prover("forest_split_update.feature_selector", feature_selector, 7),
        WitnessCommitmentInput::prover("forest_split_update.lower_bounds", lower_bounds, 7),
        WitnessCommitmentInput::prover("forest_split_update.upper_bounds", upper_bounds, 7),
        WitnessCommitmentInput::prover("forest_split_update.left_lower", left_lower, 7),
        WitnessCommitmentInput::prover("forest_split_update.left_upper", left_upper, 7),
        WitnessCommitmentInput::prover("forest_split_update.right_lower", right_lower, 7),
        WitnessCommitmentInput::prover("forest_split_update.right_upper", right_upper, 7),
        WitnessCommitmentInput::prover("forest_split_update.thresholds", thresholds, 3),
        1,
        4,
        2,
        4,
        3));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_split_update_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_split_update.root_indices", 0),
        WitnessCommitmentInput::verifier("forest_split_update.active_node_mask", 3),
        WitnessCommitmentInput::verifier("forest_split_update.current_leaf_mask", 3),
        WitnessCommitmentInput::verifier("forest_split_update.current_internal_mask", 3),
        WitnessCommitmentInput::verifier("forest_split_update.next_leaf_mask", 3),
        WitnessCommitmentInput::verifier("forest_split_update.next_internal_mask", 3),
        WitnessCommitmentInput::verifier("forest_split_update.next_active_node_mask", 3),
        WitnessCommitmentInput::verifier("forest_split_update.left_child_occupancy", 3),
        WitnessCommitmentInput::verifier("forest_split_update.right_child_occupancy", 3),
        WitnessCommitmentInput::verifier("forest_split_update.left_child_indices", 3),
        WitnessCommitmentInput::verifier("forest_split_update.right_child_indices", 3),
        WitnessCommitmentInput::verifier("forest_split_update.node_tree_ids", 3),
        WitnessCommitmentInput::verifier("forest_split_update.selected_features", 3),
        WitnessCommitmentInput::verifier("forest_split_update.selected_bins", 3),
        WitnessCommitmentInput::verifier("forest_split_update.feature_selector", 7),
        WitnessCommitmentInput::verifier("forest_split_update.lower_bounds", 7),
        WitnessCommitmentInput::verifier("forest_split_update.upper_bounds", 7),
        WitnessCommitmentInput::verifier("forest_split_update.left_lower", 7),
        WitnessCommitmentInput::verifier("forest_split_update.left_upper", 7),
        WitnessCommitmentInput::verifier("forest_split_update.right_lower", 7),
        WitnessCommitmentInput::verifier("forest_split_update.right_upper", 7),
        WitnessCommitmentInput::verifier("forest_split_update.thresholds", 3),
        1,
        4,
        2,
        4,
        3));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestLeafValueProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const fixedPointEncoding gradient_encoding = fixedPointEncoding::signed_offset_binary(8, 0);
    const fixedPointEncoding leaf_value_encoding = fixedPointEncoding::signed_offset_binary(10, 2);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_leaf_value_program(
        prover,
        WitnessCommitmentInput::prover("forest_leaf_value.gradients", poly_from_domain_ints({-6, 4, -2, 0}), 3),
        WitnessCommitmentInput::prover("forest_leaf_value.hessians", poly_from_domain_ints({3, 5, 1, 2}), 3),
        WitnessCommitmentInput::prover("forest_leaf_value.denominators", poly_from_domain_ints({5, 7, 3, 4}), 3),
        WitnessCommitmentInput::prover("forest_leaf_value.encoded_neg_grad",
                                       fixed_point_encoded_poly_from_scaled_integers({6, -4, 2, 0}, gradient_encoding),
                                       3),
        WitnessCommitmentInput::prover("forest_leaf_value.encoded_values",
                                       fixed_point_encoded_poly_from_scaled_integers({4, -3, 2, 0}, leaf_value_encoding),
                                       3),
        WitnessCommitmentInput::prover("forest_leaf_value.remainders", poly_from_domain_ints({4, 5, 2, 0}), 3),
        4,
        gradient_encoding,
        leaf_value_encoding,
        3,
        3,
        2));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_leaf_value_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_leaf_value.gradients", 3),
        WitnessCommitmentInput::verifier("forest_leaf_value.hessians", 3),
        WitnessCommitmentInput::verifier("forest_leaf_value.denominators", 3),
        WitnessCommitmentInput::verifier("forest_leaf_value.encoded_neg_grad", 3),
        WitnessCommitmentInput::verifier("forest_leaf_value.encoded_values", 3),
        WitnessCommitmentInput::verifier("forest_leaf_value.remainders", 3),
        4,
        gradient_encoding,
        leaf_value_encoding,
        3,
        3,
        2));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestTrainingInferenceProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly query_tree_ids = poly_from_domain_ints({1, 1, 2, 2});
    const poly lifted_inputs = matrix_poly_from_rows({
        {1, 1},
        {3, 2},
        {1, 1},
        {3, 1},
    });
    const poly leaf_indices = poly_from_domain_ints({1, 1, 2, 3});
    const poly leaf_lower_bounds = matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {2, 0},
    });
    const poly leaf_upper_bounds = matrix_poly_from_rows({
        {4, 4},
        {4, 4},
        {2, 4},
        {4, 4},
    });
    const poly tree_scores = poly_from_domain_ints({5, 5, 7, 9});
    const poly node_tree_ids = poly_from_domain_ints({1, 1, 2, 2});
    const poly node_lower_bounds = matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {2, 0},
    });
    const poly node_upper_bounds = matrix_poly_from_rows({
        {4, 4},
        {4, 4},
        {2, 4},
        {4, 4},
    });
    const poly node_leaf_mask = poly_from_domain_ints({0, 1, 1, 1});
    const poly node_leaf_values = poly_from_domain_ints({0, 5, 7, 9});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_training_inference_program(
        prover,
        WitnessCommitmentInput::prover("forest_train_infer.query_tree_ids", query_tree_ids, 3),
        WitnessCommitmentInput::prover("forest_train_infer.lifted_inputs", lifted_inputs, 7),
        WitnessCommitmentInput::prover("forest_train_infer.leaf_indices", leaf_indices, 3),
        WitnessCommitmentInput::prover("forest_train_infer.leaf_lower_bounds", leaf_lower_bounds, 7),
        WitnessCommitmentInput::prover("forest_train_infer.leaf_upper_bounds", leaf_upper_bounds, 7),
        WitnessCommitmentInput::prover("forest_train_infer.tree_scores", tree_scores, 3),
        WitnessCommitmentInput::prover("forest_train_infer.node_tree_ids", node_tree_ids, 3),
        WitnessCommitmentInput::prover("forest_train_infer.node_lower_bounds", node_lower_bounds, 7),
        WitnessCommitmentInput::prover("forest_train_infer.node_upper_bounds", node_upper_bounds, 7),
        WitnessCommitmentInput::prover("forest_train_infer.node_leaf_mask", node_leaf_mask, 3),
        WitnessCommitmentInput::prover("forest_train_infer.node_leaf_values", node_leaf_values, 3),
        4,
        4,
        2,
        3));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_training_inference_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_train_infer.query_tree_ids", 3),
        WitnessCommitmentInput::verifier("forest_train_infer.lifted_inputs", 7),
        WitnessCommitmentInput::verifier("forest_train_infer.leaf_indices", 3),
        WitnessCommitmentInput::verifier("forest_train_infer.leaf_lower_bounds", 7),
        WitnessCommitmentInput::verifier("forest_train_infer.leaf_upper_bounds", 7),
        WitnessCommitmentInput::verifier("forest_train_infer.tree_scores", 3),
        WitnessCommitmentInput::verifier("forest_train_infer.node_tree_ids", 3),
        WitnessCommitmentInput::verifier("forest_train_infer.node_lower_bounds", 7),
        WitnessCommitmentInput::verifier("forest_train_infer.node_upper_bounds", 7),
        WitnessCommitmentInput::verifier("forest_train_infer.node_leaf_mask", 3),
        WitnessCommitmentInput::verifier("forest_train_infer.node_leaf_values", 3),
        4,
        4,
        2,
        3));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestHistogramRefreshProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly query_tree_ids = poly_from_domain_ints({1, 1, 1, 1});
    const poly lifted_inputs = matrix_poly_from_rows({
        {1, 1},
        {0, 2},
        {2, 1},
        {3, 3},
    });
    const poly leaf_indices = poly_from_domain_ints({1, 1, 2, 2});
    const poly leaf_lower_bounds = matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {2, 0},
        {2, 0},
    });
    const poly leaf_upper_bounds = matrix_poly_from_rows({
        {2, 4},
        {2, 4},
        {4, 4},
        {4, 4},
    });
    const poly tree_scores = poly_from_domain_ints({5, 5, 7, 7});
    const poly node_tree_ids = poly_from_domain_ints({1, 1, 1, 0});
    const poly node_lower_bounds = matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {2, 0},
        {0, 0},
    });
    const poly node_upper_bounds = matrix_poly_from_rows({
        {4, 4},
        {2, 4},
        {4, 4},
        {0, 0},
    });
    const poly node_leaf_mask = poly_from_domain_ints({0, 1, 1, 0});
    const poly node_leaf_values = poly_from_domain_ints({0, 5, 7, 0});
    const poly lifted_leaf_indices = matrix_poly_from_rows({
        {1, 1},
        {1, 1},
        {2, 2},
        {2, 2},
    });
    const poly flattened_bucket_indices = matrix_poly_from_rows({
        {9, 13},
        {8, 14},
        {18, 21},
        {19, 23},
    });
    const poly count_values = poly_from_domain_ints({1, 1, 1, 1});
    const poly lifted_count_values = matrix_poly_from_rows({
        {1, 1},
        {1, 1},
        {1, 1},
        {1, 1},
    });
    const poly lifted_score_values = matrix_poly_from_rows({
        {5, 5},
        {5, 5},
        {7, 7},
        {7, 7},
    });
    const poly count_leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 1, 1, 0},
        {0, 0, 1, 1, 0, 1, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0},
    });
    const poly score_leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0, 0, 0, 0, 0},
        {5, 5, 0, 0, 0, 5, 5, 0},
        {0, 0, 7, 7, 0, 7, 0, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
    });
    const poly count_full_histogram = matrix_poly_from_rows({
        {1, 1, 1, 1, 0, 2, 1, 1},
        {1, 1, 0, 0, 0, 1, 1, 0},
        {0, 0, 1, 1, 0, 1, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0},
    });
    const poly score_full_histogram = matrix_poly_from_rows({
        {5, 5, 7, 7, 0, 12, 5, 7},
        {5, 5, 0, 0, 0, 5, 5, 0},
        {0, 0, 7, 7, 0, 7, 0, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
    });
    const poly parent_mask = poly_from_domain_ints({1, 0, 0, 0});
    const poly left_child_indices = poly_from_domain_ints({1, 0, 0, 0});
    const poly right_child_indices = poly_from_domain_ints({2, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_histogram_refresh_program(
        prover,
        WitnessCommitmentInput::prover("forest_hist_refresh.query_tree_ids", query_tree_ids, 3),
        WitnessCommitmentInput::prover("forest_hist_refresh.lifted_inputs", lifted_inputs, 7),
        WitnessCommitmentInput::prover("forest_hist_refresh.leaf_indices", leaf_indices, 3),
        WitnessCommitmentInput::prover("forest_hist_refresh.leaf_lower_bounds", leaf_lower_bounds, 7),
        WitnessCommitmentInput::prover("forest_hist_refresh.leaf_upper_bounds", leaf_upper_bounds, 7),
        WitnessCommitmentInput::prover("forest_hist_refresh.tree_scores", tree_scores, 3),
        WitnessCommitmentInput::prover("forest_hist_refresh.node_tree_ids", node_tree_ids, 3),
        WitnessCommitmentInput::prover("forest_hist_refresh.node_lower_bounds", node_lower_bounds, 7),
        WitnessCommitmentInput::prover("forest_hist_refresh.node_upper_bounds", node_upper_bounds, 7),
        WitnessCommitmentInput::prover("forest_hist_refresh.node_leaf_mask", node_leaf_mask, 3),
        WitnessCommitmentInput::prover("forest_hist_refresh.node_leaf_values", node_leaf_values, 3),
        WitnessCommitmentInput::prover("forest_hist_refresh.lifted_leaf_indices", lifted_leaf_indices, 7),
        WitnessCommitmentInput::prover("forest_hist_refresh.flattened_bucket_indices", flattened_bucket_indices, 7),
        {
            WitnessCommitmentInput::prover("forest_hist_refresh.count_values", count_values, 3),
            WitnessCommitmentInput::prover("forest_hist_refresh.score_values", tree_scores, 3),
        },
        {
            WitnessCommitmentInput::prover("forest_hist_refresh.lifted_count_values", lifted_count_values, 7),
            WitnessCommitmentInput::prover("forest_hist_refresh.lifted_score_values", lifted_score_values, 7),
        },
        {
            WitnessCommitmentInput::prover("forest_hist_refresh.count_leaf", count_leaf_histogram, 31),
            WitnessCommitmentInput::prover("forest_hist_refresh.score_leaf", score_leaf_histogram, 31),
        },
        {
            WitnessCommitmentInput::prover("forest_hist_refresh.count_full", count_full_histogram, 31),
            WitnessCommitmentInput::prover("forest_hist_refresh.score_full", score_full_histogram, 31),
        },
        WitnessCommitmentInput::prover("forest_hist_refresh.parent_mask", parent_mask, 3),
        WitnessCommitmentInput::prover("forest_hist_refresh.left_child_indices", left_child_indices, 3),
        WitnessCommitmentInput::prover("forest_hist_refresh.right_child_indices", right_child_indices, 3),
        4,
        4,
        2,
        4,
        3));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_histogram_refresh_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_hist_refresh.query_tree_ids", 3),
        WitnessCommitmentInput::verifier("forest_hist_refresh.lifted_inputs", 7),
        WitnessCommitmentInput::verifier("forest_hist_refresh.leaf_indices", 3),
        WitnessCommitmentInput::verifier("forest_hist_refresh.leaf_lower_bounds", 7),
        WitnessCommitmentInput::verifier("forest_hist_refresh.leaf_upper_bounds", 7),
        WitnessCommitmentInput::verifier("forest_hist_refresh.tree_scores", 3),
        WitnessCommitmentInput::verifier("forest_hist_refresh.node_tree_ids", 3),
        WitnessCommitmentInput::verifier("forest_hist_refresh.node_lower_bounds", 7),
        WitnessCommitmentInput::verifier("forest_hist_refresh.node_upper_bounds", 7),
        WitnessCommitmentInput::verifier("forest_hist_refresh.node_leaf_mask", 3),
        WitnessCommitmentInput::verifier("forest_hist_refresh.node_leaf_values", 3),
        WitnessCommitmentInput::verifier("forest_hist_refresh.lifted_leaf_indices", 7),
        WitnessCommitmentInput::verifier("forest_hist_refresh.flattened_bucket_indices", 7),
        {
            WitnessCommitmentInput::verifier("forest_hist_refresh.count_values", 3),
            WitnessCommitmentInput::verifier("forest_hist_refresh.score_values", 3),
        },
        {
            WitnessCommitmentInput::verifier("forest_hist_refresh.lifted_count_values", 7),
            WitnessCommitmentInput::verifier("forest_hist_refresh.lifted_score_values", 7),
        },
        {
            WitnessCommitmentInput::verifier("forest_hist_refresh.count_leaf", 31),
            WitnessCommitmentInput::verifier("forest_hist_refresh.score_leaf", 31),
        },
        {
            WitnessCommitmentInput::verifier("forest_hist_refresh.count_full", 31),
            WitnessCommitmentInput::verifier("forest_hist_refresh.score_full", 31),
        },
        WitnessCommitmentInput::verifier("forest_hist_refresh.parent_mask", 3),
        WitnessCommitmentInput::verifier("forest_hist_refresh.left_child_indices", 3),
        WitnessCommitmentInput::verifier("forest_hist_refresh.right_child_indices", 3),
        4,
        4,
        2,
        4,
        3));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestBoostingStateProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const forestTrainingProofInputs inputs = build_demo_training_inputs();

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_boosting_state_program(
        prover,
        WitnessCommitmentInput::prover("forest_boost.tree_scores", inputs.tree_scores, 7),
        WitnessCommitmentInput::prover("forest_boost.cumulative_scores", inputs.cumulative_scores, 7),
        WitnessCommitmentInput::prover("forest_boost.encoded_cumulative_scores", inputs.encoded_cumulative_scores, 7),
        WitnessCommitmentInput::prover("forest_boost.encoded_shifted_scores", inputs.encoded_shifted_scores, 7),
        WitnessCommitmentInput::prover("forest_boost.softmax_shifts", inputs.softmax_shifts, 7),
        WitnessCommitmentInput::prover("forest_boost.shift_selector", inputs.shift_selector, 7),
        WitnessCommitmentInput::prover("forest_boost.lifted_softmax_shifts", inputs.lifted_softmax_shifts, 7),
        WitnessCommitmentInput::prover("forest_boost.softmax_exp_values", inputs.softmax_exp_values, 15),
        WitnessCommitmentInput::prover("forest_boost.softmax_denominators", inputs.softmax_denominators, 15),
        WitnessCommitmentInput::prover("forest_boost.lifted_softmax_denominators", inputs.lifted_softmax_denominators, 15),
        WitnessCommitmentInput::prover("forest_boost.softmax_probabilities", inputs.softmax_probabilities, 7),
        WitnessCommitmentInput::prover("forest_boost.softmax_remainders", inputs.softmax_remainders, 15),
        WitnessCommitmentInput::prover("forest_boost.base_labels", inputs.base_labels, 7),
        WitnessCommitmentInput::prover("forest_boost.pseudo_residuals", inputs.pseudo_residuals, 7),
        WitnessCommitmentInput::prover("forest_boost.encoded_diagonal_hessians", inputs.encoded_diagonal_hessians, 7),
        WitnessCommitmentInput::prover("forest_boost.diagonal_hessian_remainders", inputs.diagonal_hessian_remainders, 7),
        WitnessCommitmentInput::prover("forest_boost.diagonal_hessians", inputs.diagonal_hessians, 7),
        inputs.iteration_count,
        inputs.class_count,
        inputs.query_count,
        inputs.score_encoding,
        inputs.softmax_encoding,
        inputs.hessian_bit_bound));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_boosting_state_program(
        verifier,
        WitnessCommitmentInput::verifier("forest_boost.tree_scores", 7),
        WitnessCommitmentInput::verifier("forest_boost.cumulative_scores", 7),
        WitnessCommitmentInput::verifier("forest_boost.encoded_cumulative_scores", 7),
        WitnessCommitmentInput::verifier("forest_boost.encoded_shifted_scores", 7),
        WitnessCommitmentInput::verifier("forest_boost.softmax_shifts", 7),
        WitnessCommitmentInput::verifier("forest_boost.shift_selector", 7),
        WitnessCommitmentInput::verifier("forest_boost.lifted_softmax_shifts", 7),
        WitnessCommitmentInput::verifier("forest_boost.softmax_exp_values", 15),
        WitnessCommitmentInput::verifier("forest_boost.softmax_denominators", 15),
        WitnessCommitmentInput::verifier("forest_boost.lifted_softmax_denominators", 15),
        WitnessCommitmentInput::verifier("forest_boost.softmax_probabilities", 7),
        WitnessCommitmentInput::verifier("forest_boost.softmax_remainders", 15),
        WitnessCommitmentInput::verifier("forest_boost.base_labels", 7),
        WitnessCommitmentInput::verifier("forest_boost.pseudo_residuals", 7),
        WitnessCommitmentInput::verifier("forest_boost.encoded_diagonal_hessians", 7),
        WitnessCommitmentInput::verifier("forest_boost.diagonal_hessian_remainders", 7),
        WitnessCommitmentInput::verifier("forest_boost.diagonal_hessians", 7),
        inputs.iteration_count,
        inputs.class_count,
        inputs.query_count,
        inputs.score_encoding,
        inputs.softmax_encoding,
        inputs.hessian_bit_bound));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestTrainingProofProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const forestTrainingProofInputs inputs = build_demo_training_inputs();

    auto run = [&](ProofRuntime& runtime, bool prover_mode) {
        const auto witness = [prover_mode](const std::string& label, const poly& polynomial, size_t bit_bound) -> ProtocolPolynomialInput {
            if (prover_mode) {
                return WitnessCommitmentInput::prover(label, polynomial, bit_bound);
            }
            return WitnessCommitmentInput::verifier(label, bit_bound);
        };

        return run_forest_training_proof_program(
            runtime,
            witness("forest_training_proof.root_indices", inputs.root_indices, 1),
            witness("forest_training_proof.split_node_mask", inputs.split_node_mask, 3),
            witness("forest_training_proof.node_depths", inputs.node_depths, 3),
            witness("forest_training_proof.left_child_indices", inputs.left_child_indices, 3),
            witness("forest_training_proof.right_child_indices", inputs.right_child_indices, 3),
            witness("forest_training_proof.node_tree_ids", inputs.node_tree_ids, 3),
            witness("forest_training_proof.feature_selector", inputs.feature_selector, 3),
            witness("forest_training_proof.lower_bounds", inputs.lower_bounds, 3),
            witness("forest_training_proof.upper_bounds", inputs.upper_bounds, 3),
            witness("forest_training_proof.left_lower", inputs.left_lower, 3),
            witness("forest_training_proof.left_upper", inputs.left_upper, 3),
            witness("forest_training_proof.right_lower", inputs.right_lower, 3),
            witness("forest_training_proof.right_upper", inputs.right_upper, 3),
            witness("forest_training_proof.thresholds", inputs.thresholds, 3),
            witness("forest_training_proof.query_tree_ids", inputs.query_tree_ids, 3),
            witness("forest_training_proof.lifted_inputs", inputs.lifted_inputs, 3),
            witness("forest_training_proof.leaf_indices", inputs.leaf_indices, 3),
            witness("forest_training_proof.leaf_lower_bounds", inputs.leaf_lower_bounds, 3),
            witness("forest_training_proof.leaf_upper_bounds", inputs.leaf_upper_bounds, 3),
            witness("forest_training_proof.tree_scores", inputs.tree_scores, 3),
            witness("forest_training_proof.training_node_tree_ids", inputs.training_node_tree_ids, 3),
            witness("forest_training_proof.training_node_lower_bounds", inputs.training_node_lower_bounds, 3),
            witness("forest_training_proof.training_node_upper_bounds", inputs.training_node_upper_bounds, 3),
            witness("forest_training_proof.node_leaf_mask", inputs.node_leaf_mask, 3),
            witness("forest_training_proof.node_leaf_values", inputs.node_leaf_values, 3),
            witness("forest_training_proof.lifted_leaf_indices", inputs.lifted_leaf_indices, 3),
            witness("forest_training_proof.flattened_bucket_indices", inputs.flattened_bucket_indices, 3),
            {
                witness("forest_training_proof.count_query_values", inputs.channels[0].query_values, 3),
                witness("forest_training_proof.grad_query_values", inputs.channels[1].query_values, 3),
                witness("forest_training_proof.hess_query_values", inputs.channels[2].query_values, 3),
            },
            {
                witness("forest_training_proof.count_lifted_values", inputs.channels[0].lifted_values, 3),
                witness("forest_training_proof.grad_lifted_values", inputs.channels[1].lifted_values, 3),
                witness("forest_training_proof.hess_lifted_values", inputs.channels[2].lifted_values, 3),
            },
            {
                witness("forest_training_proof.count_leaf_histogram", inputs.channels[0].leaf_histograms, 15),
                witness("forest_training_proof.grad_leaf_histogram", inputs.channels[1].leaf_histograms, 15),
                witness("forest_training_proof.hess_leaf_histogram", inputs.channels[2].leaf_histograms, 15),
            },
            {
                witness("forest_training_proof.count_full_histogram", inputs.channels[0].full_histograms, 15),
                witness("forest_training_proof.grad_full_histogram", inputs.channels[1].full_histograms, 15),
                witness("forest_training_proof.hess_full_histogram", inputs.channels[2].full_histograms, 15),
            },
            {
                witness("forest_training_proof.count_stats_histogram", inputs.channels[0].split_histograms, 15),
                witness("forest_training_proof.grad_stats_histogram", inputs.channels[1].split_histograms, 15),
                witness("forest_training_proof.hess_stats_histogram", inputs.channels[2].split_histograms, 15),
            },
            {
                witness("forest_training_proof.count_left_statistics", inputs.channels[0].left_statistics, 15),
                witness("forest_training_proof.grad_left_statistics", inputs.channels[1].left_statistics, 15),
                witness("forest_training_proof.hess_left_statistics", inputs.channels[2].left_statistics, 15),
            },
            {
                witness("forest_training_proof.count_right_statistics", inputs.channels[0].right_statistics, 15),
                witness("forest_training_proof.grad_right_statistics", inputs.channels[1].right_statistics, 15),
                witness("forest_training_proof.hess_right_statistics", inputs.channels[2].right_statistics, 15),
            },
            witness("forest_training_proof.left_gradients", inputs.left_gradients, 15),
            witness("forest_training_proof.right_gradients", inputs.right_gradients, 15),
            witness("forest_training_proof.left_hessians", inputs.left_hessians, 15),
            witness("forest_training_proof.right_hessians", inputs.right_hessians, 15),
            witness("forest_training_proof.left_gradient_squares", inputs.left_gradient_squares, 15),
            witness("forest_training_proof.right_gradient_squares", inputs.right_gradient_squares, 15),
            witness("forest_training_proof.parent_gradient_squares", inputs.parent_gradient_squares, 15),
            witness("forest_training_proof.gain_numerators", inputs.gain_numerators, 15),
            witness("forest_training_proof.gain_denominators", inputs.gain_denominators, 15),
            witness("forest_training_proof.encoded_gain_numerators", inputs.encoded_gain_numerators, 15),
            witness("forest_training_proof.encoded_quantized_gains", inputs.encoded_quantized_gains, 15),
            witness("forest_training_proof.gain_remainders", inputs.gain_remainders, 15),
            witness("forest_training_proof.quantized_gains", inputs.quantized_gains, 15),
            witness("forest_training_proof.tie_break_scores", inputs.tie_break_scores, 15),
            witness("forest_training_proof.candidate_mask", inputs.candidate_mask, 15),
            witness("forest_training_proof.active_node_mask", inputs.active_node_mask, 3),
            witness("forest_training_proof.selector", inputs.selector, 15),
            witness("forest_training_proof.selected_gains", inputs.selected_gains, 3),
            witness("forest_training_proof.selected_tie_break_scores", inputs.selected_tie_break_scores, 3),
            witness("forest_training_proof.selected_candidate_indices", inputs.selected_candidate_indices, 3),
            witness("forest_training_proof.selected_features", inputs.selected_features, 3),
            witness("forest_training_proof.selected_bins", inputs.selected_bins, 3),
            witness("forest_training_proof.leaf_gradients", inputs.leaf_gradients, 3),
            witness("forest_training_proof.leaf_hessians", inputs.leaf_hessians, 3),
            witness("forest_training_proof.leaf_denominators", inputs.leaf_denominators, 3),
            witness("forest_training_proof.encoded_negative_leaf_gradients", inputs.encoded_negative_leaf_gradients, 7),
            witness("forest_training_proof.encoded_leaf_values", inputs.encoded_leaf_values, 7),
            witness("forest_training_proof.leaf_remainders", inputs.leaf_remainders, 3),
            witness("forest_training_proof.cumulative_scores", inputs.cumulative_scores, 7),
            witness("forest_training_proof.encoded_cumulative_scores", inputs.encoded_cumulative_scores, 7),
            witness("forest_training_proof.encoded_shifted_scores", inputs.encoded_shifted_scores, 7),
            witness("forest_training_proof.softmax_shifts", inputs.softmax_shifts, 7),
            witness("forest_training_proof.shift_selector", inputs.shift_selector, 7),
            witness("forest_training_proof.lifted_softmax_shifts", inputs.lifted_softmax_shifts, 7),
            witness("forest_training_proof.softmax_exp_values", inputs.softmax_exp_values, 15),
            witness("forest_training_proof.softmax_denominators", inputs.softmax_denominators, 15),
            witness("forest_training_proof.lifted_softmax_denominators", inputs.lifted_softmax_denominators, 15),
            witness("forest_training_proof.softmax_probabilities", inputs.softmax_probabilities, 7),
            witness("forest_training_proof.softmax_remainders", inputs.softmax_remainders, 15),
            witness("forest_training_proof.base_labels", inputs.base_labels, 7),
            witness("forest_training_proof.pseudo_residuals", inputs.pseudo_residuals, 7),
            witness("forest_training_proof.encoded_diagonal_hessians", inputs.encoded_diagonal_hessians, 7),
            witness("forest_training_proof.diagonal_hessian_remainders", inputs.diagonal_hessian_remainders, 7),
            witness("forest_training_proof.diagonal_hessians", inputs.diagonal_hessians, 7),
            inputs.tree_count,
            inputs.node_count,
            inputs.feature_count,
            inputs.bin_count,
            inputs.iteration_count,
            inputs.class_count,
            inputs.query_count,
            inputs.input_bit_bound,
            inputs.full_upper_bound,
            inputs.structure_bit_bound,
            inputs.gradient_encoding,
            inputs.gain_numerator_encoding,
            inputs.quantized_gain_encoding,
            inputs.leaf_value_encoding,
            inputs.score_encoding,
            inputs.softmax_encoding,
            inputs.hessian_bit_bound,
            inputs.gain_denominator_bit_bound,
            inputs.leaf_denominator_bit_bound,
            inputs.gain_gap_bit_bound,
            inputs.l2_regularization,
            inputs.split_penalty);
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run(prover, true));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run(verifier, false));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ForestClassificationProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const forestClassificationPublicTable table = build_forest_classification_public_table(
        testing_kzg_public_parameters(4096, 4096),
        "forest.class.table",
        matrix_poly_from_rows({{1}, {1}, {2}, {2}, {3}, {3}, {4}, {4}}),
        matrix_poly_from_rows(
            {{10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}}),
        matrix_poly_from_rows(
            {{12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}}),
        matrix_poly_from_rows({{1}, {2}, {3}, {1}, {2}, {1}, {1}, {4}}),
        8,
        2,
        6,
        2,
        2,
        3,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}}),
        15);

    const poly query_tree = matrix_poly_from_rows({{1}, {1}, {2}, {2}, {3}, {3}, {4}, {4}});
    const poly lifted_inputs = matrix_poly_from_rows(
        {{10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}});
    const poly query_lower = matrix_poly_from_rows(
        {{10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}});
    const poly query_upper = matrix_poly_from_rows(
        {{12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}});
    const poly query_scores = matrix_poly_from_rows({{1}, {2}, {3}, {1}, {2}, {1}, {1}, {4}});
    const poly query_leaf = matrix_poly_from_rows({{7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}});
    const poly class_scores = matrix_poly_from_rows({{3, 3}, {4, 5}});
    const poly predicted_scores = matrix_poly_from_rows({{4, 5}});
    const poly class_selector = matrix_poly_from_rows({{0, 0}, {1, 1}});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_forest_classification_program(
        prover,
        table,
        WitnessCommitmentInput::prover("forest.class.query_tree", query_tree, 7),
        WitnessCommitmentInput::prover("forest.class.inputs", lifted_inputs, 15),
        WitnessCommitmentInput::prover("forest.class.query_lower", query_lower, 15),
        WitnessCommitmentInput::prover("forest.class.query_upper", query_upper, 15),
        WitnessCommitmentInput::prover("forest.class.query_score", query_scores, 7),
        WitnessCommitmentInput::prover("forest.class.query_leaf", query_leaf, 7),
        WitnessCommitmentInput::prover("forest.class.class_scores", class_scores, 3),
        WitnessCommitmentInput::prover("forest.class.predicted_scores", predicted_scores, 1),
        WitnessCommitmentInput::prover("forest.class.class_selector", class_selector, 3),
        8));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_forest_classification_program(verifier,
                                                  table.verifier_view(),
                                                  WitnessCommitmentInput::verifier("forest.class.query_tree", 7),
                                                  WitnessCommitmentInput::verifier("forest.class.inputs", 15),
                                                  WitnessCommitmentInput::verifier("forest.class.query_lower", 15),
                                                  WitnessCommitmentInput::verifier("forest.class.query_upper", 15),
                                                  WitnessCommitmentInput::verifier("forest.class.query_score", 7),
                                                  WitnessCommitmentInput::verifier("forest.class.query_leaf", 7),
                                                  WitnessCommitmentInput::verifier("forest.class.class_scores", 3),
                                                  WitnessCommitmentInput::verifier(
                                                      "forest.class.predicted_scores", 1),
                                                  WitnessCommitmentInput::verifier("forest.class.class_selector", 3),
                                                  8));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, BoxCheckProgramSupportsSharedProtocolSequenceAcrossModes)
{
    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(2048, 2048));
    EXPECT_TRUE(run_box_check_program(prover,
                                      WitnessCommitmentInput::prover(
                                          "box.values", matrix_poly_from_rows({{5, 8}, {2, 7}}), 3),
                                      WitnessCommitmentInput::prover(
                                          "box.lower", matrix_poly_from_rows({{4, 8}, {1, 6}}), 3),
                                      WitnessCommitmentInput::prover(
                                          "box.upper", matrix_poly_from_rows({{7, 9}, {3, 8}}), 3),
                                      2,
                                      2,
                                      4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_box_check_program(verifier,
                                      WitnessCommitmentInput::verifier("box.values", 3),
                                      WitnessCommitmentInput::verifier("box.lower", 3),
                                      WitnessCommitmentInput::verifier("box.upper", 3),
                                      2,
                                      2,
                                      4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, LessCheckProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly three_bit_values = poly_from_domain_ints({0, 3, 5, 7});
    const poly two_bit_values = poly_from_domain_ints({0, 1, 2, 3});

    const auto run_sequence = [&](ProofRuntime& runtime,
                                  const std::optional<poly>& maybe_three_bit_values,
                                  const std::optional<poly>& maybe_two_bit_values) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            program.less_check().add_check(
                program.bind_virtual(WitnessCommitmentInput::from_optional("less.three_bit", maybe_three_bit_values, 3)),
                3,
                4);
            program.less_check().add_check(
                program.bind_virtual(WitnessCommitmentInput::from_optional("less.two_bit", maybe_two_bit_values, 3)),
                2,
                4);
        });
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(512, 512));
    EXPECT_TRUE(run_sequence(prover, three_bit_values, two_bit_values));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, SelectorProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly selector_matrix = matrix_poly_from_rows({
        {1, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1},
    });

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(2048, 2048));
    EXPECT_TRUE(run_selector_program(prover,
                                     WitnessCommitmentInput::prover("selector.matrix", selector_matrix, 15),
                                     std::nullopt,
                                     4,
                                     4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_selector_program(verifier,
                                     WitnessCommitmentInput::verifier("selector.matrix", 15),
                                     std::nullopt,
                                     4,
                                     4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ArgmaxProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly matrix = matrix_poly_from_rows({
        {1, 5, 2, 4},
        {3, 2, 7, 6},
        {0, 8, 1, 2},
        {9, 3, 4, 5},
    });
    const poly max_values = poly_from_domain_ints({5, 7, 8, 9});
    const poly selector = matrix_poly_from_rows({
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {1, 0, 0, 0},
    });

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_argmax_program(prover,
                                   WitnessCommitmentInput::prover("argmax.matrix", matrix, 15),
                                   WitnessCommitmentInput::prover("argmax.max_values", max_values, 3),
                                   WitnessCommitmentInput::prover("argmax.selector", selector, 15),
                                   4,
                                   4,
                                   4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_argmax_program(verifier,
                                   WitnessCommitmentInput::verifier("argmax.matrix", 15),
                                   WitnessCommitmentInput::verifier("argmax.max_values", 3),
                                   WitnessCommitmentInput::verifier("argmax.selector", 15),
                                   4,
                                   4,
                                   4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, SignedRangeProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly encoded_values = fixed_point_encoded_poly_from_scaled_integers({-8, -1, 0, 7}, encoding);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(512, 512));
    EXPECT_TRUE(run_signed_range_program(
        prover, WitnessCommitmentInput::prover("signed_range.values", encoded_values, 3), encoding, 4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_signed_range_program(verifier, WitnessCommitmentInput::verifier("signed_range.values", 3), encoding, 4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, FixedPointDivisionProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({-3, 5, -7, 0}, numerator_encoding);
    const poly denominators = poly_from_domain_ints({2, 0, 3, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-6, 17, -10, 0}, quotient_encoding);
    const poly remainders = poly_from_domain_ints({0, 31, 2, 0});
    const poly mask = poly_from_domain_ints({1, 0, 1, 1});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(1024, 1024));
    EXPECT_TRUE(run_fixed_point_division_program(
        prover,
        WitnessCommitmentInput::prover("fixed_division.numerator", numerators, 3),
        WitnessCommitmentInput::prover("fixed_division.denominator", denominators, 3),
        WitnessCommitmentInput::prover("fixed_division.quotient", quotients, 3),
        WitnessCommitmentInput::prover("fixed_division.remainder", remainders, 5),
        WitnessCommitmentInput::prover("fixed_division.mask", mask, 3),
        numerator_encoding,
        quotient_encoding,
        3,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_fixed_point_division_program(verifier,
                                                 WitnessCommitmentInput::verifier("fixed_division.numerator", 3),
                                                 WitnessCommitmentInput::verifier("fixed_division.denominator", 3),
                                                 WitnessCommitmentInput::verifier("fixed_division.quotient", 3),
                                                 WitnessCommitmentInput::verifier("fixed_division.remainder", 5),
                                                 WitnessCommitmentInput::verifier("fixed_division.mask", 3),
                                                 numerator_encoding,
                                                 quotient_encoding,
                                                 3,
                                                 4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, SignedDenominatorFixedPointDivisionProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding denominator_encoding = fixedPointEncoding::signed_offset_binary(5, 1);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({7, -7, -3, 0}, numerator_encoding);
    const poly denominators = fixed_point_encoded_poly_from_scaled_integers({-3, -5, 5, -1}, denominator_encoding);
    const poly denominator_abs = poly_from_domain_ints({3, 5, 5, 1});
    const poly denominator_is_negative = poly_from_domain_ints({1, 1, 0, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-19, 11, -5, 0}, quotient_encoding);
    const poly remainder_abs = poly_from_domain_ints({1, 1, 1, 0});
    const poly mask = poly_from_domain_ints({1, 1, 1, 1});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(1024, 1024));
    EXPECT_TRUE(run_signed_denominator_fixed_point_division_program(
        prover,
        WitnessCommitmentInput::prover("fixed_division_signed.numerator", numerators, 3),
        WitnessCommitmentInput::prover("fixed_division_signed.denominator", denominators, 3),
        WitnessCommitmentInput::prover("fixed_division_signed.denominator_abs", denominator_abs, 3),
        WitnessCommitmentInput::prover("fixed_division_signed.denominator_is_negative", denominator_is_negative, 3),
        WitnessCommitmentInput::prover("fixed_division_signed.quotient", quotients, 3),
        WitnessCommitmentInput::prover("fixed_division_signed.remainder_abs", remainder_abs, 3),
        WitnessCommitmentInput::prover("fixed_division_signed.mask", mask, 3),
        numerator_encoding,
        denominator_encoding,
        quotient_encoding,
        3,
        4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_signed_denominator_fixed_point_division_program(
        verifier,
        WitnessCommitmentInput::verifier("fixed_division_signed.numerator", 3),
        WitnessCommitmentInput::verifier("fixed_division_signed.denominator", 3),
        WitnessCommitmentInput::verifier("fixed_division_signed.denominator_abs", 3),
        WitnessCommitmentInput::verifier("fixed_division_signed.denominator_is_negative", 3),
        WitnessCommitmentInput::verifier("fixed_division_signed.quotient", 3),
        WitnessCommitmentInput::verifier("fixed_division_signed.remainder_abs", 3),
        WitnessCommitmentInput::verifier("fixed_division_signed.mask", 3),
        numerator_encoding,
        denominator_encoding,
        quotient_encoding,
        3,
        4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, DivCheckProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly divisible_by_four = poly_from_domain_ints({0, 4, 8, 12});
    const poly divisible_by_two = poly_from_domain_ints({0, 2, 4, 6});

    const auto run_sequence = [&](ProofRuntime& runtime,
                                  const std::optional<poly>& maybe_divisible_by_four,
                                  const std::optional<poly>& maybe_divisible_by_two) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            program.div_check().add_check(
                program.bind_virtual(WitnessCommitmentInput::from_optional("div.by_four", maybe_divisible_by_four, 3)),
                4,
                2,
                4);
            program.div_check().add_check(
                program.bind_virtual(WitnessCommitmentInput::from_optional("div.by_two", maybe_divisible_by_two, 3)),
                3,
                1,
                4);
        });
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(512, 512));
    EXPECT_TRUE(run_sequence(prover, divisible_by_four, divisible_by_two));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, RationalCheckProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly numerator = poly_from_domain_ints({6, 10, 999, 0});
    const poly denominator = poly_from_domain_ints({2, 5, 0, 1});
    const poly quotient = poly_from_domain_ints({3, 2, 17, 0});
    const poly mask = poly_from_domain_ints({1, 1, 0, 1});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(512, 512));
    EXPECT_TRUE(run_rational_program(prover,
                                     WitnessCommitmentInput::prover("rational.numerator", numerator, 3),
                                     WitnessCommitmentInput::prover("rational.denominator", denominator, 3),
                                     WitnessCommitmentInput::prover("rational.quotient", quotient, 4),
                                     WitnessCommitmentInput::prover("rational.mask", mask, 3),
                                     4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_rational_program(verifier,
                                     WitnessCommitmentInput::verifier("rational.numerator", 3),
                                     WitnessCommitmentInput::verifier("rational.denominator", 3),
                                     WitnessCommitmentInput::verifier("rational.quotient", 4),
                                     WitnessCommitmentInput::verifier("rational.mask", 3),
                                     4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ExpCheckProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly exp_input = poly_from_domain_ints({0, 1, 2, 3});
    const poly exp_output = poly_from_domain_ints({8, 9, 10, 11});

    const auto run_sequence = [&](ProofRuntime& runtime,
                                  const std::optional<poly>& maybe_exp_input,
                                  const std::optional<poly>& maybe_exp_output) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            program.exp_check().add_check(
                program.bind_virtual(WitnessCommitmentInput::from_optional("exp.input", maybe_exp_input, 3)),
                program.bind_virtual(WitnessCommitmentInput::from_optional("exp.output", maybe_exp_output, 3)),
                3,
                4);
        });
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(2048, 2048));
    EXPECT_TRUE(run_sequence(prover, exp_input, exp_output));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, SignedExpCheckProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const std::vector<std::int64_t> scaled_scores = {-4, -1, 0, 3};
    const poly encoded_scores = fixed_point_encoded_poly_from_scaled_integers(scaled_scores, encoding);
    const poly exp_output = exp_outputs_for_signed_scores(scaled_scores, encoding);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(2048, 2048));
    EXPECT_TRUE(run_signed_exp_program(prover,
                                       WitnessCommitmentInput::prover("signed_exp.input", encoded_scores, 3),
                                       WitnessCommitmentInput::prover("signed_exp.output", exp_output, 7),
                                       encoding,
                                       4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_signed_exp_program(verifier,
                                       WitnessCommitmentInput::verifier("signed_exp.input", 3),
                                       WitnessCommitmentInput::verifier("signed_exp.output", 7),
                                       encoding,
                                       4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ProgramSupportsSharedProtocolSequenceAcrossModes)
{
    const poly lookup_query = poly_from_domain_ints({5, 7, 7, 11});
    const poly lookup_table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});

    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    const poly row_sum = matrix_row_sum_polynomial(matrix, 2, 4);

    const poly cube = cube_poly_from_slices({
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    });
    const poly cube_sum = cube_middle_sum_polynomial(cube, 2, 2, 2);

    const auto run_sequence = [&](ProofRuntime& runtime,
                                  const std::optional<poly>& maybe_lookup_query,
                                  const std::optional<poly>& maybe_matrix,
                                  const std::optional<poly>& maybe_cube) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            program.lookup().add_check(program.bind_virtual(
                                           WitnessCommitmentInput::from_optional("lookup.query", maybe_lookup_query, 3)),
                                       program.bind_virtual(PublicPolynomialInput("lookup.table", lookup_table, 7)),
                                       4,
                                       8);

            program.row_sum().add_check(program.bind_virtual(
                                            WitnessCommitmentInput::from_optional("matrix.input", maybe_matrix, 7)),
                                        program.bind_virtual(PublicPolynomialInput("matrix.row_sum", row_sum, 1)),
                                        2,
                                        4);

            program.cube_sum().add_check(
                program.bind_virtual(WitnessCommitmentInput::from_optional("cube.input", maybe_cube, 7)),
                program.bind_virtual(PublicPolynomialInput("cube.sum", cube_sum, 3)),
                2,
                2,
                2);
        });
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_sequence(prover, lookup_query, matrix, cube));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt, std::nullopt, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ProgramSupportsCubeMiddleSoftmaxAcrossModes)
{
    const poly scores = cube_poly_from_slices({
        {
            {0, 1},
            {2, 3},
        },
        {
            {1, 2},
            {3, 4},
        },
    });
    const poly softmax = cube_middle_softmax_polynomial_for_test(scores, 3, 2, 2, 2);

    const auto run_sequence = [&](ProofRuntime& runtime,
                                  const std::optional<poly>& maybe_scores,
                                  const std::optional<poly>& maybe_softmax) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            program.cube_middle_softmax().add_check(
                program.bind_virtual(WitnessCommitmentInput::from_optional("cube.softmax.scores", maybe_scores, 7)),
                program.bind_virtual(
                    WitnessCommitmentInput::from_optional("cube.softmax.output", maybe_softmax, 7)),
                3,
                2,
                2,
                2);
        });
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_sequence(prover, scores, softmax));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ProgramSupportsSignedCubeMiddleSoftmaxAcrossModes)
{
    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({-4, 0, 2, 6, -3, 1, 3, 7}, encoding);
    const poly softmax = cube_middle_softmax_polynomial_for_test(scores, encoding, 2, 2, 2);

    const auto run_sequence = [&](ProofRuntime& runtime,
                                  const std::optional<poly>& maybe_scores,
                                  const std::optional<poly>& maybe_softmax) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            program.cube_middle_softmax().add_check(
                program.bind_virtual(WitnessCommitmentInput::from_optional("cube.softmax.signed_scores", maybe_scores, 7)),
                program.bind_virtual(
                    WitnessCommitmentInput::from_optional("cube.softmax.signed_output", maybe_softmax, 7)),
                encoding,
                2,
                2,
                2);
        });
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_sequence(prover, scores, softmax));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ProgramSupportsShiftedCubeMiddleSoftmaxAcrossModes)
{
    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({1, 4, 3, 2, 0, 5, 6, 1}, encoding);
    const poly shifts = fixed_point_encoded_poly_from_scaled_integers({3, 4, 6, 5}, encoding);
    const poly shifted_scores = shifted_scores_polynomial_for_test(scores, shifts, encoding, 2, 2, 2);
    const poly softmax = cube_middle_softmax_polynomial_for_test(shifted_scores, encoding, 2, 2, 2);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_shifted_cube_softmax_program(prover,
                                                WitnessCommitmentInput::prover("cube.shifted_softmax.scores", scores, 7),
                                                WitnessCommitmentInput::prover(
                                                    "cube.shifted_softmax.shifted_scores", shifted_scores, 7),
                                                WitnessCommitmentInput::prover("cube.shifted_softmax.shifts", shifts, 3),
                                                WitnessCommitmentInput::prover("cube.shifted_softmax.output", softmax, 7),
                                                encoding,
                                                2,
                                                2,
                                                2));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_shifted_cube_softmax_program(verifier,
                                                WitnessCommitmentInput::verifier("cube.shifted_softmax.scores", 7),
                                                WitnessCommitmentInput::verifier(
                                                    "cube.shifted_softmax.shifted_scores", 7),
                                                WitnessCommitmentInput::verifier("cube.shifted_softmax.shifts", 3),
                                                WitnessCommitmentInput::verifier("cube.shifted_softmax.output", 7),
                                                encoding,
                                                2,
                                                2,
                                                2));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ProgramSupportsCubeMiddleArgmaxAcrossModes)
{
    const poly scores = cube_poly_from_slices({
        {
            {1, 4},
            {3, 2},
        },
        {
            {0, 5},
            {6, 1},
        },
    });
    const poly max_values = matrix_poly_from_rows({
        {3, 4},
        {6, 5},
    });
    const poly selector = cube_poly_from_slices({
        {
            {0, 1},
            {1, 0},
        },
        {
            {0, 1},
            {1, 0},
        },
    });

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_cube_middle_argmax_program(prover,
                                               WitnessCommitmentInput::prover("cube.argmax.scores", scores, 7),
                                               WitnessCommitmentInput::prover("cube.argmax.max_values", max_values, 3),
                                               WitnessCommitmentInput::prover("cube.argmax.selector", selector, 7),
                                               2,
                                               2,
                                               2,
                                               4));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_cube_middle_argmax_program(verifier,
                                               WitnessCommitmentInput::verifier("cube.argmax.scores", 7),
                                               WitnessCommitmentInput::verifier("cube.argmax.max_values", 3),
                                               WitnessCommitmentInput::verifier("cube.argmax.selector", 7),
                                               2,
                                               2,
                                               2,
                                               4));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ProgramSupportsRowMaxShiftedCubeMiddleSoftmaxAcrossModes)
{
    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({1, 4, 3, 2, 0, 5, 6, 1}, encoding);
    const poly shifts = fixed_point_encoded_poly_from_scaled_integers({3, 4, 6, 5}, encoding);
    const poly shift_selector = cube_poly_from_slices({
        {
            {0, 1},
            {1, 0},
        },
        {
            {0, 1},
            {1, 0},
        },
    });
    const poly shifted_scores = shifted_scores_polynomial_for_test(scores, shifts, encoding, 2, 2, 2);
    const poly softmax = cube_middle_softmax_polynomial_for_test(shifted_scores, encoding, 2, 2, 2);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_row_max_shifted_cube_softmax_program(
        prover,
        WitnessCommitmentInput::prover("cube.row_max_shifted_softmax.scores", scores, 7),
        WitnessCommitmentInput::prover("cube.row_max_shifted_softmax.shifted_scores", shifted_scores, 7),
        WitnessCommitmentInput::prover("cube.row_max_shifted_softmax.shifts", shifts, 3),
        WitnessCommitmentInput::prover("cube.row_max_shifted_softmax.selector", shift_selector, 7),
        WitnessCommitmentInput::prover("cube.row_max_shifted_softmax.output", softmax, 7),
        encoding,
        2,
        2,
        2));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_row_max_shifted_cube_softmax_program(
        verifier,
        WitnessCommitmentInput::verifier("cube.row_max_shifted_softmax.scores", 7),
        WitnessCommitmentInput::verifier("cube.row_max_shifted_softmax.shifted_scores", 7),
        WitnessCommitmentInput::verifier("cube.row_max_shifted_softmax.shifts", 3),
        WitnessCommitmentInput::verifier("cube.row_max_shifted_softmax.selector", 7),
        WitnessCommitmentInput::verifier("cube.row_max_shifted_softmax.output", 7),
        encoding,
        2,
        2,
        2));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ProgramSupportsHistAggAcrossModes)
{
    const poly indices = poly_from_domain_ints({0, 2, 1, 2, 3, 3, 0, 1});
    const poly values = poly_from_domain_ints({5, 7, 11, 13, 17, 19, 23, 29});
    const poly histogram =
        histogram_poly_from_inputs_for_test({0, 2, 1, 2, 3, 3, 0, 1}, {5, 7, 11, 13, 17, 19, 23, 29}, 4);

    const auto run_sequence = [&](ProofRuntime& runtime,
                                  const std::optional<poly>& maybe_indices,
                                  const std::optional<poly>& maybe_values,
                                  const std::optional<poly>& maybe_histogram) {
        return run_hist_agg_program(
            runtime,
            WitnessCommitmentInput::from_optional("hist.indices", maybe_indices, 7),
            WitnessCommitmentInput::from_optional("hist.values", maybe_values, 7),
            WitnessCommitmentInput::from_optional("hist.result", maybe_histogram, 3),
            8,
            4);
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_sequence(prover, indices, values, histogram));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt, std::nullopt, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ProgramSupportsCqLookupContextAcrossModes)
{
    const poly table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    const poly query = poly_from_domain_ints({5, 7, 7, 11});

    const auto run_sequence = [&](ProofRuntime& runtime, const std::optional<poly>& maybe_query) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            auto& context = program.create_cq_lookup_context(PublicPolynomialInput("cq.table", table, 7), 8);
            context.add_check(WitnessCommitmentInput::from_optional("cq.query", maybe_query, 3), 4);
        });
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(1024, 1024));
    EXPECT_TRUE(run_sequence(prover, query));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, CqLookupVerifierIgnoresTableEvaluationsInArtifact)
{
    const poly table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    const poly query = poly_from_domain_ints({5, 7, 7, 11});

    kzgProtocol setup_kzg;
    setup_kzg.setup(testing_kzg_public_parameters(1024, 1024));
    const kzgPublicParameters public_params = setup_kzg.public_parameters();
    const cqLookupPublicTable public_table =
        build_cq_lookup_public_table(public_params, "cq.table", table, 8, 7);

    const ProofArtifact artifact = prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_cq_lookup_program(
            runtime, WitnessCommitmentInput::prover("cq.query", query, 3), public_table, 4, 8);
    });

    cqLookupPublicTable verifier_table = public_table;
    verifier_table.table_evals = std::make_shared<std::vector<Fr>>(8, fr_from_int(999));

    const VerificationResult result = verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_cq_lookup_program(
            runtime, WitnessCommitmentInput::verifier("cq.query", 3), verifier_table, 4, 8);
    });
    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ProofRuntimeTest, ProgramSupportsCqTableLookupContextAcrossModes)
{
    const poly table = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    const poly query = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });

    const auto run_sequence = [&](ProofRuntime& runtime, const std::optional<poly>& maybe_query) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            auto& context =
                program.create_cq_table_lookup_context(PublicPolynomialInput("cq_table.table", table, 7), 4, 2);
            context.add_check(WitnessCommitmentInput::from_optional("cq_table.query", maybe_query, 3), 2);
        });
    };

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(2048, 2048));
    EXPECT_TRUE(run_sequence(prover, query));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_sequence(verifier, std::nullopt));
    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, CqTableLookupVerifierIgnoresTableEvaluationsInArtifact)
{
    const poly table = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    const poly query = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });

    kzgProtocol setup_kzg;
    setup_kzg.setup(testing_kzg_public_parameters(2048, 2048));
    const kzgPublicParameters public_params = setup_kzg.public_parameters();
    const cqTableLookupPublicTable public_table =
        build_cq_table_lookup_public_table(public_params, "cq_table.table", table, 4, 2, 7);

    const ProofArtifact artifact = prove_with_runtime(public_params, [&](ProofRuntime& runtime) {
        return run_cq_table_lookup_program(
            runtime, WitnessCommitmentInput::prover("cq_table.query", query, 3), public_table, 2, 4, 2);
    });

    cqTableLookupPublicTable verifier_table = public_table;
    verifier_table.table_evals = std::make_shared<std::vector<Fr>>(8, fr_from_int(999));

    const VerificationResult result = verify_with_runtime(public_params, artifact, [&](ProofRuntime& runtime) {
        return run_cq_table_lookup_program(
            runtime, WitnessCommitmentInput::verifier("cq_table.query", 3), verifier_table, 2, 4, 2);
    });
    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ProofRuntimeTest, ProgramSupportsVerifierUsingPublicCommitmentsInsteadOfPlainPolynomials)
{
    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    const poly row_sum = matrix_row_sum_polynomial(matrix, 2, 4);

    const poly lookup_query = poly_from_domain_ints({5, 7, 7, 11});
    const poly lookup_table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});

    const poly selected_rows = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });
    const poly allowed_rows = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));

    kzgProtocol public_commitment_builder;
    public_commitment_builder.setup(prover.public_parameters());
    const Group1 row_sum_commitment =
        public_commitment_builder.commit_public("public.row_sum", row_sum, 1).com;
    const Group1 lookup_table_commitment =
        public_commitment_builder.commit_public("public.lookup_table", lookup_table, 7).com;
    const Group1 allowed_rows_commitment =
        public_commitment_builder.commit_public("public.allowed_rows", allowed_rows, 7).com;

    EXPECT_TRUE(run_proof_program(prover, [&](ProofProgram& program) {
        program.row_sum().add_check(
            program.bind_virtual(WitnessCommitmentInput::prover("witness.matrix", matrix, 7)),
            program.bind_virtual(PublicCommitmentInput::prover("public.row_sum", row_sum, 1)),
            2,
            4);

        program.lookup().add_check(
            program.bind_virtual(WitnessCommitmentInput::prover("witness.lookup_query", lookup_query, 3)),
            program.bind_virtual(PublicCommitmentInput::prover("public.lookup_table", lookup_table, 7)),
            4,
            8);

        program.table_lookup().add_check(
            program.bind_virtual(WitnessCommitmentInput::prover("witness.selected_rows", selected_rows, 3)),
            program.bind_virtual(PublicCommitmentInput::prover("public.allowed_rows", allowed_rows, 7)),
            2,
            4,
            2);
    }));

    const SerializedProof proof = prover.take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    EXPECT_TRUE(run_proof_program(verifier, [&](ProofProgram& program) {
        program.row_sum().add_check(
            program.bind_virtual(WitnessCommitmentInput::verifier("witness.matrix", 7)),
            program.bind_virtual(PublicCommitmentInput::verifier("public.row_sum", row_sum_commitment, 1)),
            2,
            4);

        program.lookup().add_check(
            program.bind_virtual(WitnessCommitmentInput::verifier("witness.lookup_query", 3)),
            program.bind_virtual(PublicCommitmentInput::verifier("public.lookup_table", lookup_table_commitment, 7)),
            4,
            8);

        program.table_lookup().add_check(
            program.bind_virtual(WitnessCommitmentInput::verifier("witness.selected_rows", 3)),
            program.bind_virtual(PublicCommitmentInput::verifier("public.allowed_rows", allowed_rows_commitment, 7)),
            2,
            4,
            2);
    }));

    EXPECT_TRUE(verifier.proof_fully_consumed());
    EXPECT_EQ(prover.transcript_state(), verifier.transcript_state());
}

TEST_F(ProofRuntimeTest, ApplicationApiSupportsCustomProgramSequence)
{
    const poly query = poly_from_domain_ints({5, 7, 7, 11});
    const poly table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    const poly row_sum = matrix_row_sum_polynomial(matrix, 2, 4);

    kzgProtocol public_commitment_builder;
    public_commitment_builder.setup(testing_kzg_public_parameters(2048, 2048));
    const Group1 table_commitment = public_commitment_builder.commit_public("app.table", table, 7).com;
    const Group1 row_sum_commitment = public_commitment_builder.commit_public("app.row_sum", row_sum, 1).com;

    const ProofArtifact artifact = prove_with_runtime(testing_kzg_public_parameters(2048, 2048), [&](ProofRuntime& runtime) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            program.lookup().add_check(
                program.bind_virtual(WitnessCommitmentInput::prover("app.query", query, 3)),
                program.bind_virtual(PublicCommitmentInput::prover("app.table", table, 7)),
                4,
                8);
            program.row_sum().add_check(
                program.bind_virtual(WitnessCommitmentInput::prover("app.matrix", matrix, 7)),
                program.bind_virtual(PublicCommitmentInput::prover("app.row_sum", row_sum, 1)),
                2,
                4);
        });
    });

    const VerificationResult result = verify_with_runtime(testing_kzg_public_parameters(2048, 2048), artifact, [&](ProofRuntime& runtime) {
        return run_proof_program(runtime, [&](ProofProgram& program) {
            program.lookup().add_check(
                program.bind_virtual(WitnessCommitmentInput::verifier("app.query", 3)),
                program.bind_virtual(PublicCommitmentInput::verifier("app.table", table_commitment, 7)),
                4,
                8);
            program.row_sum().add_check(
                program.bind_virtual(WitnessCommitmentInput::verifier("app.matrix", 7)),
                program.bind_virtual(PublicCommitmentInput::verifier("app.row_sum", row_sum_commitment, 1)),
                2,
                4);
        });
    });

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}
