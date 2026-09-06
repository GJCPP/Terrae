#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "fixed_point_division.hpp"
#include "forest_quantized_split_gain.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "signed_range_check.hpp"
#include "sum_check.hpp"
#include "test_kzg_srs.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mcl::bn;

namespace {

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

} // namespace

class ForestQuantizedSplitGainProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestQuantizedSplitGainProtocolTest, ValidQuantizedGainFinalize)
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

    forestQuantizedSplitGainProtocol protocol;
    protocol.setup(zero, fixed_division);

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

    forestQuantizedSplitGainWitness witness;
    witness.exact_gain_numerators = virCommitment(kzg.commit(matrix_poly_from_rows(gain_numerators)));
    witness.gain_denominators = virCommitment(kzg.commit(matrix_poly_from_rows(gain_denominators)));
    witness.encoded_gain_numerators =
        virCommitment(kzg.commit(fixed_point_encoded_matrix_poly_from_rows(gain_numerators, numerator_encoding)));
    witness.encoded_quantized_gains =
        virCommitment(kzg.commit(fixed_point_encoded_matrix_poly_from_rows(quantized_gains, quantized_gain_encoding)));
    witness.gain_remainders = virCommitment(kzg.commit(matrix_poly_from_rows(remainders)));

    protocol.add_check(witness, 2, 2, 4, numerator_encoding, quantized_gain_encoding, 6);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(fixed_division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestQuantizedSplitGainProtocolTest, WrongQuantizedGainFailsZeroCheck)
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

    forestQuantizedSplitGainProtocol protocol;
    protocol.setup(zero, fixed_division);

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
    std::vector<std::vector<int>> wrong_quantized_gains =
        quantized_gain_matrix(gain_numerators, gain_denominators, quantized_gain_encoding.fractional_bits);
    wrong_quantized_gains[0][1] += 1;
    const std::vector<std::vector<int>> remainders =
        quantized_remainder_matrix(gain_numerators, gain_denominators, quantized_gain_encoding.fractional_bits);

    forestQuantizedSplitGainWitness witness;
    witness.exact_gain_numerators = virCommitment(kzg.commit(matrix_poly_from_rows(gain_numerators)));
    witness.gain_denominators = virCommitment(kzg.commit(matrix_poly_from_rows(gain_denominators)));
    witness.encoded_gain_numerators =
        virCommitment(kzg.commit(fixed_point_encoded_matrix_poly_from_rows(gain_numerators, numerator_encoding)));
    witness.encoded_quantized_gains = virCommitment(
        kzg.commit(fixed_point_encoded_matrix_poly_from_rows(wrong_quantized_gains, quantized_gain_encoding)));
    witness.gain_remainders = virCommitment(kzg.commit(matrix_poly_from_rows(remainders)));

    protocol.add_check(witness, 2, 2, 4, numerator_encoding, quantized_gain_encoding, 6);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(fixed_division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
