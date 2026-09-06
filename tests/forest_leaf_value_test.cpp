#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "fixed_point_division.hpp"
#include "forest_leaf_value.hpp"
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

class ForestLeafValueProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestLeafValueProtocolTest, ValidLeafValuesWithHalfLearningRateFinalize)
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

    forestLeafValueProtocol protocol;
    protocol.setup(zero, less, fixed_division);

    const fixedPointEncoding gradient_encoding = fixedPointEncoding::signed_offset_binary(8, 0);
    const fixedPointEncoding leaf_value_encoding = fixedPointEncoding::signed_offset_binary(10, 2);

    forestLeafValueWitness witness;
    witness.leaf_gradients = virCommitment(kzg.commit(poly_from_domain_ints({-6, 4, -2, 0})));
    witness.leaf_hessians = virCommitment(kzg.commit(poly_from_domain_ints({3, 5, 1, 2})));
    witness.leaf_denominators = virCommitment(kzg.commit(poly_from_domain_ints({10, 14, 6, 8})));
    witness.encoded_negative_leaf_gradients =
        virCommitment(kzg.commit(fixed_point_encoded_poly_from_scaled_integers({6, -4, 2, 0}, gradient_encoding)));
    witness.encoded_leaf_values =
        virCommitment(kzg.commit(fixed_point_encoded_poly_from_scaled_integers({2, -2, 1, 0}, leaf_value_encoding)));
    witness.leaf_remainders = virCommitment(kzg.commit(poly_from_domain_ints({4, 12, 2, 0})));

    protocol.add_check(witness, 4, gradient_encoding, leaf_value_encoding, 3, 4, 2, 1, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(fixed_division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestLeafValueProtocolTest, WrongLeafValueFailsZeroCheck)
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

    forestLeafValueProtocol protocol;
    protocol.setup(zero, less, fixed_division);

    const fixedPointEncoding gradient_encoding = fixedPointEncoding::signed_offset_binary(8, 0);
    const fixedPointEncoding leaf_value_encoding = fixedPointEncoding::signed_offset_binary(10, 2);

    forestLeafValueWitness witness;
    witness.leaf_gradients = virCommitment(kzg.commit(poly_from_domain_ints({-6, 4, -2, 0})));
    witness.leaf_hessians = virCommitment(kzg.commit(poly_from_domain_ints({3, 5, 1, 2})));
    witness.leaf_denominators = virCommitment(kzg.commit(poly_from_domain_ints({5, 7, 3, 4})));
    witness.encoded_negative_leaf_gradients =
        virCommitment(kzg.commit(fixed_point_encoded_poly_from_scaled_integers({6, -4, 2, 0}, gradient_encoding)));
    witness.encoded_leaf_values =
        virCommitment(kzg.commit(fixed_point_encoded_poly_from_scaled_integers({4, -2, 2, 0}, leaf_value_encoding)));
    witness.leaf_remainders = virCommitment(kzg.commit(poly_from_domain_ints({4, 5, 2, 0})));

    protocol.add_check(witness, 4, gradient_encoding, leaf_value_encoding, 3, 3, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(fixed_division.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
