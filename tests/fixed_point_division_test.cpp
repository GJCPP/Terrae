#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "fixed_point_division.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
#include "signed_range_check.hpp"
#include "sum_check.hpp"
#include "test_kzg_srs.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mcl::bn;

namespace {

poly poly_from_domain_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

} // namespace

class FixedPointDivisionProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(FixedPointDivisionProtocolTest, AddCheckRejectsInvalidParameters)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

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

    fixedPointDivisionProtocol division;
    division.setup(kzg, zero, less, signed_range);

    const virCommitment one(fr_from_int(1));
    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 2);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 1);

    EXPECT_THROW(
        division.add_check(one, one, one, one, numerator_encoding, quotient_encoding, 4, 4), std::runtime_error);
    EXPECT_THROW(division.add_check(one,
                                    one,
                                    one,
                                    one,
                                    fixedPointEncoding::signed_offset_binary(5, 0),
                                    fixedPointEncoding::signed_offset_binary(6, 2),
                                    0,
                                    4),
                 std::runtime_error);
    EXPECT_THROW(division.add_check(one,
                                    one,
                                    one,
                                    one,
                                    fixedPointEncoding::signed_offset_binary(5, 0),
                                    fixedPointEncoding::signed_offset_binary(6, 2),
                                    4,
                                    3),
                 std::runtime_error);
    EXPECT_THROW(division.add_check(one,
                                    one,
                                    one,
                                    one,
                                    one,
                                    one,
                                    fixedPointEncoding::signed_offset_binary(5, 2),
                                    fixedPointEncoding::signed_offset_binary(5, 0),
                                    fixedPointEncoding::signed_offset_binary(6, 1),
                                    4,
                                    4),
                 std::runtime_error);
}

TEST_F(FixedPointDivisionProtocolTest, ValidUnmaskedCheckFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

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

    fixedPointDivisionProtocol division;
    division.setup(kzg, zero, less, signed_range);

    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({-3, 5, -7, 0}, numerator_encoding);
    const poly denominators = poly_from_domain_ints({2, 4, 3, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-6, 5, -10, 0}, quotient_encoding);
    const poly remainders = poly_from_domain_ints({0, 0, 2, 0});

    division.add_check(virCommitment(kzg.commit(numerators)),
                       virCommitment(kzg.commit(denominators)),
                       virCommitment(kzg.commit(quotients)),
                       virCommitment(kzg.commit(remainders)),
                       numerator_encoding,
                       quotient_encoding,
                       3,
                       4);

    EXPECT_TRUE(division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(FixedPointDivisionProtocolTest, WrongQuotientFailsZeroCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

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

    fixedPointDivisionProtocol division;
    division.setup(kzg, zero, less, signed_range);

    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({-3, 5, -7, 0}, numerator_encoding);
    const poly denominators = poly_from_domain_ints({2, 4, 3, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-6, 5, -9, 0}, quotient_encoding);
    const poly remainders = poly_from_domain_ints({0, 0, 2, 0});

    division.add_check(virCommitment(kzg.commit(numerators)),
                       virCommitment(kzg.commit(denominators)),
                       virCommitment(kzg.commit(quotients)),
                       virCommitment(kzg.commit(remainders)),
                       numerator_encoding,
                       quotient_encoding,
                       3,
                       4);

    EXPECT_TRUE(division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(FixedPointDivisionProtocolTest, RemainderNotLessThanDenominatorFailsRangeCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

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

    fixedPointDivisionProtocol division;
    division.setup(kzg, zero, less, signed_range);

    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({-7, 5, -3, 0}, numerator_encoding);
    const poly denominators = poly_from_domain_ints({3, 4, 2, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-11, 5, -6, 0}, quotient_encoding);
    const poly remainders = poly_from_domain_ints({5, 0, 0, 0});

    division.add_check(virCommitment(kzg.commit(numerators)),
                       virCommitment(kzg.commit(denominators)),
                       virCommitment(kzg.commit(quotients)),
                       virCommitment(kzg.commit(remainders)),
                       numerator_encoding,
                       quotient_encoding,
                       3,
                       4);

    EXPECT_TRUE(division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(FixedPointDivisionProtocolTest, MaskedCheckAllowsDisabledZeroDenominatorRows)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

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

    fixedPointDivisionProtocol division;
    division.setup(kzg, zero, less, signed_range);

    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({-3, 5, -7, 0}, numerator_encoding);
    const poly denominators = poly_from_domain_ints({2, 0, 3, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-6, 17, -10, 0}, quotient_encoding);
    const poly remainders = poly_from_domain_ints({0, 31, 2, 0});
    const poly mask = poly_from_domain_ints({1, 0, 1, 1});

    division.add_masked_check(virCommitment(kzg.commit(numerators)),
                              virCommitment(kzg.commit(denominators)),
                              virCommitment(kzg.commit(quotients)),
                              virCommitment(kzg.commit(remainders)),
                              virCommitment(kzg.commit(mask)),
                              numerator_encoding,
                              quotient_encoding,
                              3,
                              4);

    EXPECT_TRUE(division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(FixedPointDivisionProtocolTest, ReductionDoesNotWaitForFinalize)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

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

    fixedPointDivisionProtocol division;
    division.setup(kzg, zero, less, signed_range);

    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({-3, 5, -7, 0}, numerator_encoding);
    const poly denominators = poly_from_domain_ints({2, 4, 3, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-6, 5, -9, 0}, quotient_encoding);
    const poly remainders = poly_from_domain_ints({0, 0, 2, 0});

    division.add_check(virCommitment(kzg.commit(numerators)),
                       virCommitment(kzg.commit(denominators)),
                       virCommitment(kzg.commit(quotients)),
                       virCommitment(kzg.commit(remainders)),
                       numerator_encoding,
                       quotient_encoding,
                       3,
                       4);

    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(division.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(FixedPointDivisionProtocolTest, ValidSignedDenominatorCheckFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

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

    fixedPointDivisionProtocol division;
    division.setup(kzg, zero, less, signed_range);

    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding denominator_encoding = fixedPointEncoding::signed_offset_binary(5, 1);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({7, -7, -3, 0}, numerator_encoding);
    const poly denominators = fixed_point_encoded_poly_from_scaled_integers({-3, -5, 5, -1}, denominator_encoding);
    const poly denominator_abs = poly_from_domain_ints({3, 5, 5, 1});
    const poly denominator_is_negative = poly_from_domain_ints({1, 1, 0, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-19, 11, -5, 0}, quotient_encoding);
    const poly remainder_abs = poly_from_domain_ints({1, 1, 1, 0});

    division.add_check(virCommitment(kzg.commit(numerators)),
                       virCommitment(kzg.commit(denominators)),
                       virCommitment(kzg.commit(denominator_abs)),
                       virCommitment(kzg.commit(denominator_is_negative)),
                       virCommitment(kzg.commit(quotients)),
                       virCommitment(kzg.commit(remainder_abs)),
                       numerator_encoding,
                       denominator_encoding,
                       quotient_encoding,
                       3,
                       4);

    EXPECT_TRUE(division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(FixedPointDivisionProtocolTest, SignedDenominatorFloorWitnessFailsWhenRemainderMagnitudeIsWrong)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

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

    fixedPointDivisionProtocol division;
    division.setup(kzg, zero, less, signed_range);

    const fixedPointEncoding numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding denominator_encoding = fixedPointEncoding::signed_offset_binary(5, 1);
    const fixedPointEncoding quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly numerators = fixed_point_encoded_poly_from_scaled_integers({7, -7, -3, 0}, numerator_encoding);
    const poly denominators = fixed_point_encoded_poly_from_scaled_integers({-3, -5, 5, -1}, denominator_encoding);
    const poly denominator_abs = poly_from_domain_ints({3, 5, 5, 1});
    const poly denominator_is_negative = poly_from_domain_ints({1, 1, 0, 1});
    const poly quotients = fixed_point_encoded_poly_from_scaled_integers({-19, 11, -5, 0}, quotient_encoding);
    const poly remainder_abs = poly_from_domain_ints({1, 0, 1, 0});

    division.add_check(virCommitment(kzg.commit(numerators)),
                       virCommitment(kzg.commit(denominators)),
                       virCommitment(kzg.commit(denominator_abs)),
                       virCommitment(kzg.commit(denominator_is_negative)),
                       virCommitment(kzg.commit(quotients)),
                       virCommitment(kzg.commit(remainder_abs)),
                       numerator_encoding,
                       denominator_encoding,
                       quotient_encoding,
                       3,
                       4);

    EXPECT_TRUE(division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
