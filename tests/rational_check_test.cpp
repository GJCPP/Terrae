#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"
#include "polynomial.hpp"
#include "rational_check.hpp"
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

class RationalCheckProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(RationalCheckProtocolTest, AddCheckRejectsInvalidDomainSize)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    zeroProtocol zero;
    zero.setup(kzg);

    rationalCheckProtocol rational;
    rational.setup(kzg, zero);

    const virCommitment one(fr_from_int(1));
    EXPECT_THROW(rational.add_check(one, one, one, 3), std::runtime_error);
    EXPECT_THROW(rational.add_masked_check(one, one, one, one, 0), std::runtime_error);
}

TEST_F(RationalCheckProtocolTest, ValidUnmaskedCheckFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    rationalCheckProtocol rational;
    rational.setup(kzg, zero);

    const poly numerator = poly_from_domain_ints({6, 10, 21, 0});
    const poly denominator = poly_from_domain_ints({2, 5, 7, 1});
    const poly quotient = poly_from_domain_ints({3, 2, 3, 0});

    rational.add_check(
        virCommitment(kzg.commit(numerator)), virCommitment(kzg.commit(denominator)), virCommitment(kzg.commit(quotient)), 4);

    EXPECT_TRUE(rational.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(RationalCheckProtocolTest, WrongQuotientFailsZeroCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    rationalCheckProtocol rational;
    rational.setup(kzg, zero);

    const poly numerator = poly_from_domain_ints({6, 10, 21, 0});
    const poly denominator = poly_from_domain_ints({2, 5, 7, 1});
    const poly quotient = poly_from_domain_ints({3, 2, 4, 0});

    rational.add_check(
        virCommitment(kzg.commit(numerator)), virCommitment(kzg.commit(denominator)), virCommitment(kzg.commit(quotient)), 4);

    EXPECT_TRUE(rational.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(RationalCheckProtocolTest, MaskedCheckAllowsDisabledRows)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    rationalCheckProtocol rational;
    rational.setup(kzg, zero);

    const poly numerator = poly_from_domain_ints({6, 10, 999, 0});
    const poly denominator = poly_from_domain_ints({2, 5, 0, 1});
    const poly quotient = poly_from_domain_ints({3, 2, 17, 0});
    const poly mask = poly_from_domain_ints({1, 1, 0, 1});

    rational.add_masked_check(virCommitment(kzg.commit(numerator)),
                              virCommitment(kzg.commit(denominator)),
                              virCommitment(kzg.commit(quotient)),
                              virCommitment(kzg.commit(mask)),
                              4);

    EXPECT_TRUE(rational.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(RationalCheckProtocolTest, NonBooleanMaskFailsZeroCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    rationalCheckProtocol rational;
    rational.setup(kzg, zero);

    const poly numerator = poly_from_domain_ints({6, 10, 21, 0});
    const poly denominator = poly_from_domain_ints({2, 5, 7, 1});
    const poly quotient = poly_from_domain_ints({3, 2, 3, 0});
    const poly mask = poly_from_domain_ints({1, 2, 1, 0});

    rational.add_masked_check(virCommitment(kzg.commit(numerator)),
                              virCommitment(kzg.commit(denominator)),
                              virCommitment(kzg.commit(quotient)),
                              virCommitment(kzg.commit(mask)),
                              4);

    EXPECT_TRUE(rational.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(RationalCheckProtocolTest, ReductionDoesNotWaitForFinalize)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    rationalCheckProtocol rational;
    rational.setup(kzg, zero);

    const poly numerator = poly_from_domain_ints({6, 10, 21, 0});
    const poly denominator = poly_from_domain_ints({2, 5, 7, 1});
    const poly quotient = poly_from_domain_ints({3, 2, 4, 0});

    rational.add_check(
        virCommitment(kzg.commit(numerator)), virCommitment(kzg.commit(denominator)), virCommitment(kzg.commit(quotient)), 4);

    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(rational.finalize());
    EXPECT_TRUE(kzg.finalize());
}
