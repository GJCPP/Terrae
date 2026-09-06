#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "cube_argmax.hpp"
#include "cube_softmax.hpp"
#include "cube_sum.hpp"
#include "fixed_point.hpp"
#include "forest_boosting_state.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "matrix_sum.hpp"
#include "signed_range_check.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "test_kzg_srs.hpp"
#include "zero_check.hpp"
#include "../examples/forest_training_demo_data.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using namespace mcl::bn;

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

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

Fr fr_from_cpp_int_checked(const cpp_int& value)
{
    const std::string repr = value.convert_to<std::string>();
    Fr out;
    out.setStr(repr, 10);
    if (out.getStr(10) != repr) {
        throw std::runtime_error("forest_boosting_state_test: non-canonical cpp_int conversion");
    }
    return out;
}

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
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
        const size_t encoded = fixed_point_size_t_from_fr(encoded_score, "forest_boosting_state_test");
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
    std::vector<Fr> residuals(softmax_evals.size(), Fr(0));
    for (size_t i = 0; i < residuals.size(); ++i) {
        residuals[i] = softmax_evals[i] - label_evals[i];
    }
    return poly_from_evals(residuals);
}

poly diagonal_hessian_poly_from_softmax(const poly& softmax,
                                        size_t iteration_count,
                                        size_t class_count,
                                        size_t query_count)
{
    const std::vector<Fr> softmax_evals = softmax.evaluations_on_domain(iteration_count * class_count * query_count);
    std::vector<Fr> hessians(softmax_evals.size(), Fr(0));
    for (size_t i = 0; i < hessians.size(); ++i) {
        hessians[i] = softmax_evals[i] * (Fr(1) - softmax_evals[i]);
    }
    return poly_from_evals(hessians);
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
                const int label_scaled = canonical_signed_int64_from_fr(
                                              lifted_label_evals[flat_index], "forest_boosting_state_test lifted_labels") *
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
    for (size_t iteration = 0; iteration < iteration_count; ++iteration) {
        for (size_t cls = 0; cls < class_count; ++cls) {
            for (size_t query = 0; query < query_count; ++query) {
                const size_t column = cls * query_count + query;
                lifted_shift_rows[iteration][column] = shift_rows[iteration][query];
            }
        }
    }

    FixedPointBoostingWitnessPolys out{
        matrix_poly_from_rows(cumulative_rows),
        fixed_point_encoded_matrix_poly_from_rows(cumulative_rows, score_encoding),
        fixed_point_encoded_matrix_poly_from_rows(shift_rows, score_encoding),
        cube_poly_from_slices(shift_selector),
        fixed_point_encoded_matrix_poly_from_rows(lifted_shift_rows, score_encoding),
        fixed_point_encoded_matrix_poly_from_rows(shifted_rows, score_encoding),
        matrix_poly_from_rows(exp_rows),
        matrix_poly_from_rows(denominator_rows),
        matrix_poly_from_rows(lifted_denominator_rows),
        fixed_point_encoded_matrix_poly_from_rows(probability_rows, softmax_encoding),
        matrix_poly_from_rows(probability_remainder_rows),
        matrix_poly_from_rows(pseudo_residual_rows),
        fixed_point_encoded_matrix_poly_from_rows(hessian_rows, hessian_encoding),
        matrix_poly_from_rows(hessian_remainder_rows),
        matrix_poly_from_rows(hessian_rows),
        hessian_bit_bound};
    return out;
}

} // namespace

class ForestBoostingStateProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestBoostingStateProtocolTest, ValidBoostingStateFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    fixedPointDivisionProtocol fixed_division;
    fixed_division.setup(kzg, zero, less, signed_range);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol cube_replication;
    cube_replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, cube_replication, less);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    matrixColumnPrefixSumProtocol column_prefix;
    column_prefix.setup(kzg, zero, column_replication);

    matrixColumnSumProtocol column_sum;
    column_sum.setup(kzg, zero);

    forestBoostingStateProtocol protocol;
    protocol.setup(zero,
                   column_prefix,
                   column_sum,
                   cube_argmax,
                   cube_replication,
                   cube_sum,
                   exp_check,
                   fixed_division,
                   signed_range);

    const forestTrainingProofInputs inputs = build_demo_training_inputs();

    forestBoostingStateWitness witness;
    witness.tree_scores = virCommitment(kzg.commit(inputs.tree_scores));
    witness.cumulative_scores = virCommitment(kzg.commit(inputs.cumulative_scores));
    witness.encoded_cumulative_scores = virCommitment(kzg.commit(inputs.encoded_cumulative_scores));
    witness.encoded_shifted_scores = virCommitment(kzg.commit(inputs.encoded_shifted_scores));
    witness.softmax_shifts = virCommitment(kzg.commit(inputs.softmax_shifts));
    witness.shift_selector = virCommitment(kzg.commit(inputs.shift_selector));
    witness.lifted_softmax_shifts = virCommitment(kzg.commit(inputs.lifted_softmax_shifts));
    witness.softmax_exp_values = virCommitment(kzg.commit(inputs.softmax_exp_values));
    witness.softmax_denominators = virCommitment(kzg.commit(inputs.softmax_denominators));
    witness.lifted_softmax_denominators = virCommitment(kzg.commit(inputs.lifted_softmax_denominators));
    witness.softmax_probabilities = virCommitment(kzg.commit(inputs.softmax_probabilities));
    witness.softmax_remainders = virCommitment(kzg.commit(inputs.softmax_remainders));
    witness.base_labels = virCommitment(kzg.commit(inputs.base_labels));
    witness.pseudo_residuals = virCommitment(kzg.commit(inputs.pseudo_residuals));
    witness.encoded_diagonal_hessians = virCommitment(kzg.commit(inputs.encoded_diagonal_hessians));
    witness.diagonal_hessian_remainders = virCommitment(kzg.commit(inputs.diagonal_hessian_remainders));
    witness.diagonal_hessians = virCommitment(kzg.commit(inputs.diagonal_hessians));

    protocol.add_check(witness,
                       inputs.iteration_count,
                       inputs.class_count,
                       inputs.query_count,
                       inputs.score_encoding,
                       inputs.softmax_encoding,
                       inputs.hessian_bit_bound);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(cube_replication.finalize());
    EXPECT_TRUE(column_prefix.finalize());
    EXPECT_TRUE(column_sum.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestBoostingStateProtocolTest, WrongPseudoResidualFailsZeroCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    fixedPointDivisionProtocol fixed_division;
    fixed_division.setup(kzg, zero, less, signed_range);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol cube_replication;
    cube_replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, cube_replication, less);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    matrixColumnPrefixSumProtocol column_prefix;
    column_prefix.setup(kzg, zero, column_replication);

    matrixColumnSumProtocol column_sum;
    column_sum.setup(kzg, zero);

    forestBoostingStateProtocol protocol;
    protocol.setup(zero,
                   column_prefix,
                   column_sum,
                   cube_argmax,
                   cube_replication,
                   cube_sum,
                   exp_check,
                   fixed_division,
                   signed_range);

    const forestTrainingProofInputs inputs = build_demo_training_inputs();
    std::vector<Fr> wrong_residual_evals = inputs.pseudo_residuals.evaluations_on_domain(inputs.tree_count * inputs.query_count);
    Fr::add(wrong_residual_evals[3], wrong_residual_evals[3], Fr(1));
    const poly wrong_pseudo_residuals = poly_from_evals(wrong_residual_evals);

    forestBoostingStateWitness witness;
    witness.tree_scores = virCommitment(kzg.commit(inputs.tree_scores));
    witness.cumulative_scores = virCommitment(kzg.commit(inputs.cumulative_scores));
    witness.encoded_cumulative_scores = virCommitment(kzg.commit(inputs.encoded_cumulative_scores));
    witness.encoded_shifted_scores = virCommitment(kzg.commit(inputs.encoded_shifted_scores));
    witness.softmax_shifts = virCommitment(kzg.commit(inputs.softmax_shifts));
    witness.shift_selector = virCommitment(kzg.commit(inputs.shift_selector));
    witness.lifted_softmax_shifts = virCommitment(kzg.commit(inputs.lifted_softmax_shifts));
    witness.softmax_exp_values = virCommitment(kzg.commit(inputs.softmax_exp_values));
    witness.softmax_denominators = virCommitment(kzg.commit(inputs.softmax_denominators));
    witness.lifted_softmax_denominators = virCommitment(kzg.commit(inputs.lifted_softmax_denominators));
    witness.softmax_probabilities = virCommitment(kzg.commit(inputs.softmax_probabilities));
    witness.softmax_remainders = virCommitment(kzg.commit(inputs.softmax_remainders));
    witness.base_labels = virCommitment(kzg.commit(inputs.base_labels));
    witness.pseudo_residuals = virCommitment(kzg.commit(wrong_pseudo_residuals));
    witness.encoded_diagonal_hessians = virCommitment(kzg.commit(inputs.encoded_diagonal_hessians));
    witness.diagonal_hessian_remainders = virCommitment(kzg.commit(inputs.diagonal_hessian_remainders));
    witness.diagonal_hessians = virCommitment(kzg.commit(inputs.diagonal_hessians));

    protocol.add_check(witness,
                       inputs.iteration_count,
                       inputs.class_count,
                       inputs.query_count,
                       inputs.score_encoding,
                       inputs.softmax_encoding,
                       inputs.hessian_bit_bound);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(cube_replication.finalize());
    EXPECT_TRUE(column_prefix.finalize());
    EXPECT_TRUE(column_sum.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
