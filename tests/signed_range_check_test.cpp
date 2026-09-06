#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
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

class SignedRangeCheckProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(SignedRangeCheckProtocolTest, EncodingHelpersRoundTripScaledIntegers)
{
    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);

    EXPECT_EQ(fixed_point_min_scaled_value(encoding), -8);
    EXPECT_EQ(fixed_point_max_scaled_value(encoding), 7);

    EXPECT_EQ(encode_fixed_point_scaled_integer(-8, encoding), 0u);
    EXPECT_EQ(encode_fixed_point_scaled_integer(-1, encoding), 7u);
    EXPECT_EQ(encode_fixed_point_scaled_integer(0, encoding), 8u);
    EXPECT_EQ(encode_fixed_point_scaled_integer(7, encoding), 15u);

    EXPECT_EQ(decode_fixed_point_scaled_integer(0, encoding), -8);
    EXPECT_EQ(decode_fixed_point_scaled_integer(7, encoding), -1);
    EXPECT_EQ(decode_fixed_point_scaled_integer(8, encoding), 0);
    EXPECT_EQ(decode_fixed_point_scaled_integer(15, encoding), 7);
}

TEST_F(SignedRangeCheckProtocolTest, AddCheckRejectsInvalidInputs)
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

    fixedPointEncoding invalid_encoding = fixedPointEncoding::signed_offset_binary(4, 1);
    invalid_encoding.signed_values = false;

    const virCommitment values(fr_from_int(0));
    EXPECT_THROW(signed_range.add_check(values, invalid_encoding, 4), std::runtime_error);
    EXPECT_THROW(signed_range.add_check(values, fixedPointEncoding::signed_offset_binary(4, 4), 4), std::runtime_error);
    EXPECT_THROW(signed_range.add_check(values, fixedPointEncoding::signed_offset_binary(4, 1), 3), std::runtime_error);
}

TEST_F(SignedRangeCheckProtocolTest, ValidEncodedSignedValuesFinalize)
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

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const poly encoded = fixed_point_encoded_poly_from_scaled_integers({-8, -1, 0, 7}, encoding);

    signed_range.add_check(virCommitment(kzg.commit(encoded)), encoding, 4);

    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SignedRangeCheckProtocolTest, OutOfRangeEncodedValueFailsLookupReduction)
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

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 1);
    const poly encoded = poly_from_domain_ints({0, 16, 8, 15});

    signed_range.add_check(virCommitment(kzg.commit(encoded)), encoding, 4);

    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SignedRangeCheckProtocolTest, AddCheckImmediatelyReducesToLessCheck)
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

    {
        signedRangeCheckProtocol signed_range;
        signed_range.setup(kzg, less);

        const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 1);
        const poly encoded = poly_from_domain_ints({0, 16, 8, 15});

        signed_range.add_check(virCommitment(kzg.commit(encoded)), encoding, 4);

        EXPECT_TRUE(less.finalize());
        EXPECT_FALSE(lookup.finalize());
    }

    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
