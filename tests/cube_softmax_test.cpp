#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "cube_argmax.hpp"
#include "cube_softmax.hpp"
#include "kzg.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "test_kzg_srs.hpp"
#include "zero_check.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <gtest/gtest.h>

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

poly perturb_evaluation(const poly& p, size_t domain_size, size_t index, const Fr& delta)
{
    std::vector<Fr> evals = p.evaluations_on_domain(domain_size);
    Fr::add(evals[index], evals[index], delta);
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

poly quantized_exp_cube_polynomial_for_test(const poly& scores,
                                            size_t quantization_bits,
                                            size_t outer_count,
                                            size_t middle_count,
                                            size_t inner_count)
{
    const std::vector<Fr> score_evals = scores.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> exp_evals;
    exp_evals.reserve(score_evals.size());
    for (const Fr& score : score_evals) {
        const size_t input = static_cast<size_t>(std::stoull(score.getStr(10)));
        exp_evals.push_back(fr_from_int(static_cast<int>(quantized_exp_output(input, quantization_bits))));
    }
    return poly_from_evals(exp_evals);
}

poly quantized_exp_cube_polynomial_for_test(const poly& encoded_scores,
                                            const fixedPointEncoding& score_encoding,
                                            size_t outer_count,
                                            size_t middle_count,
                                            size_t inner_count)
{
    const std::vector<Fr> score_evals =
        encoded_scores.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> exp_evals;
    exp_evals.reserve(score_evals.size());
    for (const Fr& encoded_score : score_evals) {
        const size_t encoded = fixed_point_size_t_from_fr(encoded_score, "cube_softmax_test");
        const std::int64_t scaled_score = decode_fixed_point_scaled_integer(encoded, score_encoding);
        exp_evals.push_back(fr_from_cpp_int_checked(quantized_exp_output(scaled_score, score_encoding.fractional_bits)));
    }
    return poly_from_evals(exp_evals);
}

poly cube_middle_softmax_polynomial_for_test(const poly& scores,
                                             size_t quantization_bits,
                                             size_t outer_count,
                                             size_t middle_count,
                                             size_t inner_count)
{
    const poly exp_values =
        quantized_exp_cube_polynomial_for_test(scores, quantization_bits, outer_count, middle_count, inner_count);
    const poly denominator = cube_middle_sum_polynomial(exp_values, outer_count, middle_count, inner_count);
    const poly lifted_denominator =
        cube_middle_replication_polynomial(denominator, outer_count, middle_count, inner_count);

    const std::vector<Fr> exp_evals = exp_values.evaluations_on_domain(outer_count * middle_count * inner_count);
    const std::vector<Fr> denominator_evals =
        lifted_denominator.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> softmax_evals(exp_evals.size(), Fr(0));
    for (size_t i = 0; i < exp_evals.size(); ++i) {
        Fr inverse;
        Fr::inv(inverse, denominator_evals[i]);
        softmax_evals[i] = exp_evals[i] * inverse;
    }
    return poly_from_evals(softmax_evals);
}

poly cube_middle_softmax_polynomial_for_test(const poly& encoded_scores,
                                             const fixedPointEncoding& score_encoding,
                                             size_t outer_count,
                                             size_t middle_count,
                                             size_t inner_count)
{
    const poly exp_values = quantized_exp_cube_polynomial_for_test(
        encoded_scores, score_encoding, outer_count, middle_count, inner_count);
    const poly denominator = cube_middle_sum_polynomial(exp_values, outer_count, middle_count, inner_count);
    const poly lifted_denominator =
        cube_middle_replication_polynomial(denominator, outer_count, middle_count, inner_count);

    const std::vector<Fr> exp_evals = exp_values.evaluations_on_domain(outer_count * middle_count * inner_count);
    const std::vector<Fr> denominator_evals =
        lifted_denominator.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> softmax_evals(exp_evals.size(), Fr(0));
    for (size_t i = 0; i < exp_evals.size(); ++i) {
        Fr inverse;
        Fr::inv(inverse, denominator_evals[i]);
        softmax_evals[i] = exp_evals[i] * inverse;
    }
    return poly_from_evals(softmax_evals);
}

poly shifted_scores_polynomial_for_test(const poly& encoded_scores,
                                        const poly& encoded_shifts,
                                        const fixedPointEncoding& score_encoding,
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
    const Fr offset = fixed_point_offset_fr(score_encoding);
    for (size_t i = 0; i < score_evals.size(); ++i) {
        shifted_evals.push_back(score_evals[i] - lifted_shift_evals[i] + offset);
    }
    return poly_from_evals(shifted_evals);
}

poly cube_middle_shifted_softmax_polynomial_for_test(const poly& shifted_scores,
                                                     const fixedPointEncoding& score_encoding,
                                                     size_t outer_count,
                                                     size_t middle_count,
                                                     size_t inner_count)
{
    return cube_middle_softmax_polynomial_for_test(
        shifted_scores, score_encoding, outer_count, middle_count, inner_count);
}

} // namespace

class CubeSoftmaxProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(CubeSoftmaxProtocolTest, HelperReplicationPolynomialMatchesExpectedCube)
{
    const poly base = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
    });
    const poly lifted = cube_middle_replication_polynomial(base, 2, 2, 2);

    const std::vector<Fr> expected = {
        fr_from_int(1),
        fr_from_int(2),
        fr_from_int(1),
        fr_from_int(2),
        fr_from_int(3),
        fr_from_int(4),
        fr_from_int(3),
        fr_from_int(4),
    };
    EXPECT_EQ(lifted.evaluations_on_domain(8), expected);
}

TEST_F(CubeSoftmaxProtocolTest, ValidCubeMiddleReplicationProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    cubeMiddleReplicationProtocol protocol;
    protocol.setup(kzg, zero, table_lookup);

    const poly base = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
    });
    const poly lifted = cube_middle_replication_polynomial(base, 2, 2, 2);

    protocol.add_check(virCommitment(kzg.commit(base)), virCommitment(kzg.commit(lifted)), 2, 2, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, ReplicationFailsWhenBlockRowsAreNotConstant)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    cubeMiddleReplicationProtocol protocol;
    protocol.setup(kzg, zero, table_lookup);

    const poly base = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
    });
    poly lifted = cube_middle_replication_polynomial(base, 2, 2, 2);
    lifted = perturb_evaluation(lifted, 8, 3, fr_from_int(1));

    protocol.add_check(virCommitment(kzg.commit(base)), virCommitment(kzg.commit(lifted)), 2, 2, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, ReplicationFailsWhenBoundaryRowsDoNotMatchBase)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    cubeMiddleReplicationProtocol protocol;
    protocol.setup(kzg, zero, table_lookup);

    const poly base = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
    });
    const poly wrong_lifted = cube_poly_from_slices({
        {
            {1, 2},
            {1, 2},
        },
        {
            {9, 10},
            {9, 10},
        },
    });

    protocol.add_check(virCommitment(kzg.commit(base)), virCommitment(kzg.commit(wrong_lifted)), 2, 2, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, ValidCubeMiddleSoftmaxProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, replication, less);

    cubeMiddleSoftmaxProtocol softmax;
    softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, replication);

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
    const poly softmax_values = cube_middle_softmax_polynomial_for_test(scores, 3, 2, 2, 2);

    softmax.add_check(virCommitment(kzg.commit(scores)), virCommitment(kzg.commit(softmax_values)), 3, 2, 2, 2);

    EXPECT_TRUE(softmax.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, ValidSignedCubeMiddleSoftmaxProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, replication, less);

    cubeMiddleSoftmaxProtocol softmax;
    softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, replication);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({-4, 0, 2, 6, -3, 1, 3, 7}, encoding);
    const poly softmax_values = cube_middle_softmax_polynomial_for_test(scores, encoding, 2, 2, 2);

    softmax.add_check(
        virCommitment(kzg.commit(scores)), virCommitment(kzg.commit(softmax_values)), encoding, 2, 2, 2);

    EXPECT_TRUE(softmax.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, ValidShiftedCubeMiddleSoftmaxProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, replication, less);

    cubeMiddleSoftmaxProtocol softmax;
    softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, replication);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({1, 4, 3, 2, 0, 5, 6, 1}, encoding);
    const poly shifts = fixed_point_encoded_poly_from_scaled_integers({3, 4, 6, 5}, encoding);
    const poly shifted_scores = shifted_scores_polynomial_for_test(scores, shifts, encoding, 2, 2, 2);
    const poly softmax_values = cube_middle_shifted_softmax_polynomial_for_test(shifted_scores, encoding, 2, 2, 2);

    softmax.add_shifted_check(virCommitment(kzg.commit(scores)),
                              virCommitment(kzg.commit(shifted_scores)),
                              virCommitment(kzg.commit(shifts)),
                              virCommitment(kzg.commit(softmax_values)),
                              encoding,
                              2,
                              2,
                              2);

    EXPECT_TRUE(softmax.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, ValidRowMaxShiftedCubeMiddleSoftmaxProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, replication, less);

    cubeMiddleSoftmaxProtocol softmax;
    softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, replication);

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
    const poly softmax_values = cube_middle_shifted_softmax_polynomial_for_test(shifted_scores, encoding, 2, 2, 2);

    softmax.add_row_max_shifted_check(virCommitment(kzg.commit(scores)),
                                      virCommitment(kzg.commit(shifted_scores)),
                                      virCommitment(kzg.commit(shifts)),
                                      virCommitment(kzg.commit(shift_selector)),
                                      virCommitment(kzg.commit(softmax_values)),
                                      encoding,
                                      2,
                                      2,
                                      2);

    EXPECT_TRUE(softmax.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, ShiftedCubeMiddleSoftmaxFailsWhenShiftRelationIsBroken)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, replication, less);

    cubeMiddleSoftmaxProtocol softmax;
    softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, replication);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({1, 4, 3, 2, 0, 5, 6, 1}, encoding);
    const poly shifts = fixed_point_encoded_poly_from_scaled_integers({3, 4, 6, 5}, encoding);
    poly shifted_scores = shifted_scores_polynomial_for_test(scores, shifts, encoding, 2, 2, 2);
    shifted_scores = perturb_evaluation(shifted_scores, 8, 3, fr_from_int(1));
    const poly softmax_values = cube_middle_shifted_softmax_polynomial_for_test(shifted_scores, encoding, 2, 2, 2);

    softmax.add_shifted_check(virCommitment(kzg.commit(scores)),
                              virCommitment(kzg.commit(shifted_scores)),
                              virCommitment(kzg.commit(shifts)),
                              virCommitment(kzg.commit(softmax_values)),
                              encoding,
                              2,
                              2,
                              2);

    EXPECT_TRUE(softmax.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, RowMaxShiftedCubeMiddleSoftmaxFailsWhenShiftIsNotRowMax)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, replication, less);

    cubeMiddleSoftmaxProtocol softmax;
    softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, replication);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({1, 4, 3, 2, 0, 5, 6, 1}, encoding);
    poly shifts = fixed_point_encoded_poly_from_scaled_integers({3, 4, 6, 5}, encoding);
    shifts = perturb_evaluation(shifts, 4, 0, fr_from_int(1));
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
    const poly softmax_values = cube_middle_shifted_softmax_polynomial_for_test(shifted_scores, encoding, 2, 2, 2);

    softmax.add_row_max_shifted_check(virCommitment(kzg.commit(scores)),
                                      virCommitment(kzg.commit(shifted_scores)),
                                      virCommitment(kzg.commit(shifts)),
                                      virCommitment(kzg.commit(shift_selector)),
                                      virCommitment(kzg.commit(softmax_values)),
                                      encoding,
                                      2,
                                      2,
                                      2);

    EXPECT_TRUE(softmax.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, WrongSoftmaxOutputFailsFinalZeroCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol cube_argmax;
    cube_argmax.setup(kzg, zero, cube_sum, replication, less);

    cubeMiddleSoftmaxProtocol softmax;
    softmax.setup(kzg, zero, signed_range, cube_argmax, exp_check, cube_sum, replication);

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
    poly softmax_values = cube_middle_softmax_polynomial_for_test(scores, 3, 2, 2, 2);
    softmax_values = perturb_evaluation(softmax_values, 8, 5, fr_from_int(1));

    softmax.add_check(virCommitment(kzg.commit(scores)), virCommitment(kzg.commit(softmax_values)), 3, 2, 2, 2);

    EXPECT_TRUE(softmax.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_argmax.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSoftmaxProtocolTest, CubeMiddleSoftmaxProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(4096, 4096));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_lookup;
    prover_lookup.setup(prover_kzg, prover_zero, prover_sum);

    tableLookupProtocol prover_table_lookup(prover_lookup);
    prover_table_lookup.setup(prover_kzg, prover_zero, prover_sum);

    lessCheckProtocol prover_less;
    prover_less.setup(prover_kzg, prover_lookup);

    expCheckProtocol prover_exp_check;
    prover_exp_check.setup(prover_kzg, prover_table_lookup);

    signedRangeCheckProtocol prover_signed_range;
    prover_signed_range.setup(prover_kzg, prover_less);

    cubeSumProtocol prover_cube_sum(prover_table_lookup);
    prover_cube_sum.setup(prover_kzg, prover_zero);

    cubeMiddleReplicationProtocol prover_replication;
    prover_replication.setup(prover_kzg, prover_zero, prover_table_lookup);

    cubeMiddleArgmaxProtocol prover_cube_argmax;
    prover_cube_argmax.setup(prover_kzg, prover_zero, prover_cube_sum, prover_replication, prover_less);

    cubeMiddleSoftmaxProtocol prover_softmax;
    prover_softmax.setup(
        prover_kzg, prover_zero, prover_signed_range, prover_cube_argmax, prover_exp_check, prover_cube_sum, prover_replication);

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
    const poly softmax_values = cube_middle_softmax_polynomial_for_test(scores, 3, 2, 2, 2);
    const kzgCommitment prover_scores_commitment = prover_kzg.commit(scores);
    const kzgCommitment prover_softmax_commitment = prover_kzg.commit(softmax_values);

    prover_softmax.add_check(
        virCommitment(prover_scores_commitment), virCommitment(prover_softmax_commitment), 3, 2, 2, 2);

    EXPECT_TRUE(prover_softmax.finalize());
    EXPECT_TRUE(prover_replication.finalize());
    EXPECT_TRUE(prover_cube_argmax.finalize());
    EXPECT_TRUE(prover_signed_range.finalize());
    EXPECT_TRUE(prover_exp_check.finalize());
    EXPECT_TRUE(prover_less.finalize());
    EXPECT_TRUE(prover_cube_sum.finalize());
    EXPECT_TRUE(prover_table_lookup.finalize());
    EXPECT_TRUE(prover_lookup.finalize());
    EXPECT_TRUE(prover_sum.finalize());
    EXPECT_TRUE(prover_zero.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    zeroProtocol verifier_zero;
    verifier_zero.setup(verifier_kzg);

    sumProtocol verifier_sum;
    verifier_sum.setup(verifier_kzg, verifier_zero);

    lookupProtocol verifier_lookup;
    verifier_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    tableLookupProtocol verifier_table_lookup(verifier_lookup);
    verifier_table_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    lessCheckProtocol verifier_less;
    verifier_less.setup(verifier_kzg, verifier_lookup);

    expCheckProtocol verifier_exp_check;
    verifier_exp_check.setup(verifier_kzg, verifier_table_lookup);

    signedRangeCheckProtocol verifier_signed_range;
    verifier_signed_range.setup(verifier_kzg, verifier_less);

    cubeSumProtocol verifier_cube_sum(verifier_table_lookup);
    verifier_cube_sum.setup(verifier_kzg, verifier_zero);

    cubeMiddleReplicationProtocol verifier_replication;
    verifier_replication.setup(verifier_kzg, verifier_zero, verifier_table_lookup);

    cubeMiddleArgmaxProtocol verifier_cube_argmax;
    verifier_cube_argmax.setup(verifier_kzg, verifier_zero, verifier_cube_sum, verifier_replication, verifier_less);

    cubeMiddleSoftmaxProtocol verifier_softmax;
    verifier_softmax.setup(
        verifier_kzg, verifier_zero, verifier_signed_range, verifier_cube_argmax, verifier_exp_check, verifier_cube_sum, verifier_replication);

    const kzgCommitment verifier_scores_commitment = verifier_kzg.adopt_commitment(prover_scores_commitment.com);
    const kzgCommitment verifier_softmax_commitment = verifier_kzg.adopt_commitment(prover_softmax_commitment.com);

    verifier_softmax.add_check(
        virCommitment(verifier_scores_commitment), virCommitment(verifier_softmax_commitment), 3, 2, 2, 2);

    EXPECT_TRUE(verifier_softmax.finalize());
    EXPECT_TRUE(verifier_replication.finalize());
    EXPECT_TRUE(verifier_cube_argmax.finalize());
    EXPECT_TRUE(verifier_signed_range.finalize());
    EXPECT_TRUE(verifier_exp_check.finalize());
    EXPECT_TRUE(verifier_less.finalize());
    EXPECT_TRUE(verifier_cube_sum.finalize());
    EXPECT_TRUE(verifier_table_lookup.finalize());
    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(CubeSoftmaxProtocolTest, SignedCubeMiddleSoftmaxProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(4096, 4096));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_lookup;
    prover_lookup.setup(prover_kzg, prover_zero, prover_sum);

    tableLookupProtocol prover_table_lookup(prover_lookup);
    prover_table_lookup.setup(prover_kzg, prover_zero, prover_sum);

    lessCheckProtocol prover_less;
    prover_less.setup(prover_kzg, prover_lookup);

    expCheckProtocol prover_exp_check;
    prover_exp_check.setup(prover_kzg, prover_table_lookup);

    signedRangeCheckProtocol prover_signed_range;
    prover_signed_range.setup(prover_kzg, prover_less);

    cubeSumProtocol prover_cube_sum(prover_table_lookup);
    prover_cube_sum.setup(prover_kzg, prover_zero);

    cubeMiddleReplicationProtocol prover_replication;
    prover_replication.setup(prover_kzg, prover_zero, prover_table_lookup);

    cubeMiddleArgmaxProtocol prover_cube_argmax;
    prover_cube_argmax.setup(prover_kzg, prover_zero, prover_cube_sum, prover_replication, prover_less);

    cubeMiddleSoftmaxProtocol prover_softmax;
    prover_softmax.setup(
        prover_kzg, prover_zero, prover_signed_range, prover_cube_argmax, prover_exp_check, prover_cube_sum, prover_replication);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({-4, 0, 2, 6, -3, 1, 3, 7}, encoding);
    const poly softmax_values = cube_middle_softmax_polynomial_for_test(scores, encoding, 2, 2, 2);
    const kzgCommitment prover_scores_commitment = prover_kzg.commit(scores);
    const kzgCommitment prover_softmax_commitment = prover_kzg.commit(softmax_values);

    prover_softmax.add_check(
        virCommitment(prover_scores_commitment), virCommitment(prover_softmax_commitment), encoding, 2, 2, 2);

    EXPECT_TRUE(prover_softmax.finalize());
    EXPECT_TRUE(prover_replication.finalize());
    EXPECT_TRUE(prover_cube_argmax.finalize());
    EXPECT_TRUE(prover_signed_range.finalize());
    EXPECT_TRUE(prover_exp_check.finalize());
    EXPECT_TRUE(prover_less.finalize());
    EXPECT_TRUE(prover_cube_sum.finalize());
    EXPECT_TRUE(prover_table_lookup.finalize());
    EXPECT_TRUE(prover_lookup.finalize());
    EXPECT_TRUE(prover_sum.finalize());
    EXPECT_TRUE(prover_zero.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    zeroProtocol verifier_zero;
    verifier_zero.setup(verifier_kzg);

    sumProtocol verifier_sum;
    verifier_sum.setup(verifier_kzg, verifier_zero);

    lookupProtocol verifier_lookup;
    verifier_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    tableLookupProtocol verifier_table_lookup(verifier_lookup);
    verifier_table_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    lessCheckProtocol verifier_less;
    verifier_less.setup(verifier_kzg, verifier_lookup);

    expCheckProtocol verifier_exp_check;
    verifier_exp_check.setup(verifier_kzg, verifier_table_lookup);

    signedRangeCheckProtocol verifier_signed_range;
    verifier_signed_range.setup(verifier_kzg, verifier_less);

    cubeSumProtocol verifier_cube_sum(verifier_table_lookup);
    verifier_cube_sum.setup(verifier_kzg, verifier_zero);

    cubeMiddleReplicationProtocol verifier_replication;
    verifier_replication.setup(verifier_kzg, verifier_zero, verifier_table_lookup);

    cubeMiddleArgmaxProtocol verifier_cube_argmax;
    verifier_cube_argmax.setup(verifier_kzg, verifier_zero, verifier_cube_sum, verifier_replication, verifier_less);

    cubeMiddleSoftmaxProtocol verifier_softmax;
    verifier_softmax.setup(
        verifier_kzg, verifier_zero, verifier_signed_range, verifier_cube_argmax, verifier_exp_check, verifier_cube_sum, verifier_replication);

    const kzgCommitment verifier_scores_commitment = verifier_kzg.adopt_commitment(prover_scores_commitment.com);
    const kzgCommitment verifier_softmax_commitment = verifier_kzg.adopt_commitment(prover_softmax_commitment.com);

    verifier_softmax.add_check(
        virCommitment(verifier_scores_commitment), virCommitment(verifier_softmax_commitment), encoding, 2, 2, 2);

    EXPECT_TRUE(verifier_softmax.finalize());
    EXPECT_TRUE(verifier_replication.finalize());
    EXPECT_TRUE(verifier_cube_argmax.finalize());
    EXPECT_TRUE(verifier_signed_range.finalize());
    EXPECT_TRUE(verifier_exp_check.finalize());
    EXPECT_TRUE(verifier_less.finalize());
    EXPECT_TRUE(verifier_cube_sum.finalize());
    EXPECT_TRUE(verifier_table_lookup.finalize());
    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(CubeSoftmaxProtocolTest, ShiftedCubeMiddleSoftmaxProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(4096, 4096));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_lookup;
    prover_lookup.setup(prover_kzg, prover_zero, prover_sum);

    tableLookupProtocol prover_table_lookup(prover_lookup);
    prover_table_lookup.setup(prover_kzg, prover_zero, prover_sum);

    lessCheckProtocol prover_less;
    prover_less.setup(prover_kzg, prover_lookup);

    expCheckProtocol prover_exp_check;
    prover_exp_check.setup(prover_kzg, prover_table_lookup);

    signedRangeCheckProtocol prover_signed_range;
    prover_signed_range.setup(prover_kzg, prover_less);

    cubeSumProtocol prover_cube_sum(prover_table_lookup);
    prover_cube_sum.setup(prover_kzg, prover_zero);

    cubeMiddleReplicationProtocol prover_replication;
    prover_replication.setup(prover_kzg, prover_zero, prover_table_lookup);

    cubeMiddleArgmaxProtocol prover_cube_argmax;
    prover_cube_argmax.setup(prover_kzg, prover_zero, prover_cube_sum, prover_replication, prover_less);

    cubeMiddleSoftmaxProtocol prover_softmax;
    prover_softmax.setup(
        prover_kzg, prover_zero, prover_signed_range, prover_cube_argmax, prover_exp_check, prover_cube_sum, prover_replication);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly scores =
        fixed_point_encoded_poly_from_scaled_integers({1, 4, 3, 2, 0, 5, 6, 1}, encoding);
    const poly shifts = fixed_point_encoded_poly_from_scaled_integers({3, 4, 6, 5}, encoding);
    const poly shifted_scores = shifted_scores_polynomial_for_test(scores, shifts, encoding, 2, 2, 2);
    const poly softmax_values = cube_middle_shifted_softmax_polynomial_for_test(shifted_scores, encoding, 2, 2, 2);
    const kzgCommitment prover_scores_commitment = prover_kzg.commit(scores);
    const kzgCommitment prover_shifted_scores_commitment = prover_kzg.commit(shifted_scores);
    const kzgCommitment prover_shifts_commitment = prover_kzg.commit(shifts);
    const kzgCommitment prover_softmax_commitment = prover_kzg.commit(softmax_values);

    prover_softmax.add_shifted_check(virCommitment(prover_scores_commitment),
                                     virCommitment(prover_shifted_scores_commitment),
                                     virCommitment(prover_shifts_commitment),
                                     virCommitment(prover_softmax_commitment),
                                     encoding,
                                     2,
                                     2,
                                     2);

    EXPECT_TRUE(prover_softmax.finalize());
    EXPECT_TRUE(prover_replication.finalize());
    EXPECT_TRUE(prover_cube_argmax.finalize());
    EXPECT_TRUE(prover_signed_range.finalize());
    EXPECT_TRUE(prover_exp_check.finalize());
    EXPECT_TRUE(prover_less.finalize());
    EXPECT_TRUE(prover_cube_sum.finalize());
    EXPECT_TRUE(prover_table_lookup.finalize());
    EXPECT_TRUE(prover_lookup.finalize());
    EXPECT_TRUE(prover_sum.finalize());
    EXPECT_TRUE(prover_zero.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    zeroProtocol verifier_zero;
    verifier_zero.setup(verifier_kzg);

    sumProtocol verifier_sum;
    verifier_sum.setup(verifier_kzg, verifier_zero);

    lookupProtocol verifier_lookup;
    verifier_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    tableLookupProtocol verifier_table_lookup(verifier_lookup);
    verifier_table_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    lessCheckProtocol verifier_less;
    verifier_less.setup(verifier_kzg, verifier_lookup);

    expCheckProtocol verifier_exp_check;
    verifier_exp_check.setup(verifier_kzg, verifier_table_lookup);

    signedRangeCheckProtocol verifier_signed_range;
    verifier_signed_range.setup(verifier_kzg, verifier_less);

    cubeSumProtocol verifier_cube_sum(verifier_table_lookup);
    verifier_cube_sum.setup(verifier_kzg, verifier_zero);

    cubeMiddleReplicationProtocol verifier_replication;
    verifier_replication.setup(verifier_kzg, verifier_zero, verifier_table_lookup);

    cubeMiddleArgmaxProtocol verifier_cube_argmax;
    verifier_cube_argmax.setup(verifier_kzg, verifier_zero, verifier_cube_sum, verifier_replication, verifier_less);

    cubeMiddleSoftmaxProtocol verifier_softmax;
    verifier_softmax.setup(
        verifier_kzg, verifier_zero, verifier_signed_range, verifier_cube_argmax, verifier_exp_check, verifier_cube_sum, verifier_replication);

    const kzgCommitment verifier_scores_commitment = verifier_kzg.adopt_commitment(prover_scores_commitment.com);
    const kzgCommitment verifier_shifted_scores_commitment =
        verifier_kzg.adopt_commitment(prover_shifted_scores_commitment.com);
    const kzgCommitment verifier_shifts_commitment = verifier_kzg.adopt_commitment(prover_shifts_commitment.com);
    const kzgCommitment verifier_softmax_commitment = verifier_kzg.adopt_commitment(prover_softmax_commitment.com);

    verifier_softmax.add_shifted_check(virCommitment(verifier_scores_commitment),
                                       virCommitment(verifier_shifted_scores_commitment),
                                       virCommitment(verifier_shifts_commitment),
                                       virCommitment(verifier_softmax_commitment),
                                       encoding,
                                       2,
                                       2,
                                       2);

    EXPECT_TRUE(verifier_softmax.finalize());
    EXPECT_TRUE(verifier_replication.finalize());
    EXPECT_TRUE(verifier_cube_argmax.finalize());
    EXPECT_TRUE(verifier_signed_range.finalize());
    EXPECT_TRUE(verifier_exp_check.finalize());
    EXPECT_TRUE(verifier_less.finalize());
    EXPECT_TRUE(verifier_cube_sum.finalize());
    EXPECT_TRUE(verifier_table_lookup.finalize());
    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(CubeSoftmaxProtocolTest, RowMaxShiftedCubeMiddleSoftmaxProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(4096, 4096));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_lookup;
    prover_lookup.setup(prover_kzg, prover_zero, prover_sum);

    tableLookupProtocol prover_table_lookup(prover_lookup);
    prover_table_lookup.setup(prover_kzg, prover_zero, prover_sum);

    lessCheckProtocol prover_less;
    prover_less.setup(prover_kzg, prover_lookup);

    expCheckProtocol prover_exp_check;
    prover_exp_check.setup(prover_kzg, prover_table_lookup);

    signedRangeCheckProtocol prover_signed_range;
    prover_signed_range.setup(prover_kzg, prover_less);

    cubeSumProtocol prover_cube_sum(prover_table_lookup);
    prover_cube_sum.setup(prover_kzg, prover_zero);

    cubeMiddleReplicationProtocol prover_replication;
    prover_replication.setup(prover_kzg, prover_zero, prover_table_lookup);

    cubeMiddleArgmaxProtocol prover_cube_argmax;
    prover_cube_argmax.setup(prover_kzg, prover_zero, prover_cube_sum, prover_replication, prover_less);

    cubeMiddleSoftmaxProtocol prover_softmax;
    prover_softmax.setup(
        prover_kzg, prover_zero, prover_signed_range, prover_cube_argmax, prover_exp_check, prover_cube_sum, prover_replication);

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
    const poly softmax_values = cube_middle_shifted_softmax_polynomial_for_test(shifted_scores, encoding, 2, 2, 2);
    const kzgCommitment prover_scores_commitment = prover_kzg.commit(scores);
    const kzgCommitment prover_shifted_scores_commitment = prover_kzg.commit(shifted_scores);
    const kzgCommitment prover_shifts_commitment = prover_kzg.commit(shifts);
    const kzgCommitment prover_shift_selector_commitment = prover_kzg.commit(shift_selector);
    const kzgCommitment prover_softmax_commitment = prover_kzg.commit(softmax_values);

    prover_softmax.add_row_max_shifted_check(virCommitment(prover_scores_commitment),
                                             virCommitment(prover_shifted_scores_commitment),
                                             virCommitment(prover_shifts_commitment),
                                             virCommitment(prover_shift_selector_commitment),
                                             virCommitment(prover_softmax_commitment),
                                             encoding,
                                             2,
                                             2,
                                             2);

    EXPECT_TRUE(prover_softmax.finalize());
    EXPECT_TRUE(prover_replication.finalize());
    EXPECT_TRUE(prover_cube_argmax.finalize());
    EXPECT_TRUE(prover_signed_range.finalize());
    EXPECT_TRUE(prover_exp_check.finalize());
    EXPECT_TRUE(prover_less.finalize());
    EXPECT_TRUE(prover_cube_sum.finalize());
    EXPECT_TRUE(prover_table_lookup.finalize());
    EXPECT_TRUE(prover_lookup.finalize());
    EXPECT_TRUE(prover_sum.finalize());
    EXPECT_TRUE(prover_zero.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    zeroProtocol verifier_zero;
    verifier_zero.setup(verifier_kzg);

    sumProtocol verifier_sum;
    verifier_sum.setup(verifier_kzg, verifier_zero);

    lookupProtocol verifier_lookup;
    verifier_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    tableLookupProtocol verifier_table_lookup(verifier_lookup);
    verifier_table_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    lessCheckProtocol verifier_less;
    verifier_less.setup(verifier_kzg, verifier_lookup);

    expCheckProtocol verifier_exp_check;
    verifier_exp_check.setup(verifier_kzg, verifier_table_lookup);

    signedRangeCheckProtocol verifier_signed_range;
    verifier_signed_range.setup(verifier_kzg, verifier_less);

    cubeSumProtocol verifier_cube_sum(verifier_table_lookup);
    verifier_cube_sum.setup(verifier_kzg, verifier_zero);

    cubeMiddleReplicationProtocol verifier_replication;
    verifier_replication.setup(verifier_kzg, verifier_zero, verifier_table_lookup);

    cubeMiddleArgmaxProtocol verifier_cube_argmax;
    verifier_cube_argmax.setup(verifier_kzg, verifier_zero, verifier_cube_sum, verifier_replication, verifier_less);

    cubeMiddleSoftmaxProtocol verifier_softmax;
    verifier_softmax.setup(
        verifier_kzg, verifier_zero, verifier_signed_range, verifier_cube_argmax, verifier_exp_check, verifier_cube_sum, verifier_replication);

    const kzgCommitment verifier_scores_commitment = verifier_kzg.adopt_commitment(prover_scores_commitment.com);
    const kzgCommitment verifier_shifted_scores_commitment =
        verifier_kzg.adopt_commitment(prover_shifted_scores_commitment.com);
    const kzgCommitment verifier_shifts_commitment = verifier_kzg.adopt_commitment(prover_shifts_commitment.com);
    const kzgCommitment verifier_shift_selector_commitment =
        verifier_kzg.adopt_commitment(prover_shift_selector_commitment.com);
    const kzgCommitment verifier_softmax_commitment = verifier_kzg.adopt_commitment(prover_softmax_commitment.com);

    verifier_softmax.add_row_max_shifted_check(virCommitment(verifier_scores_commitment),
                                               virCommitment(verifier_shifted_scores_commitment),
                                               virCommitment(verifier_shifts_commitment),
                                               virCommitment(verifier_shift_selector_commitment),
                                               virCommitment(verifier_softmax_commitment),
                                               encoding,
                                               2,
                                               2,
                                               2);

    EXPECT_TRUE(verifier_softmax.finalize());
    EXPECT_TRUE(verifier_replication.finalize());
    EXPECT_TRUE(verifier_cube_argmax.finalize());
    EXPECT_TRUE(verifier_signed_range.finalize());
    EXPECT_TRUE(verifier_exp_check.finalize());
    EXPECT_TRUE(verifier_less.finalize());
    EXPECT_TRUE(verifier_cube_sum.finalize());
    EXPECT_TRUE(verifier_table_lookup.finalize());
    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}
