#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
#include "shuffle.hpp"
#include "sum_check.hpp"
#include "test_kzg_srs.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
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

class ShuffleProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ShuffleProtocolTest, AddCheckRejectsInvalidShape)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    shuffleProtocol shuffle;
    shuffle.setup(kzg, lookup);

    const virCommitment c(fr_from_int(0));
    EXPECT_THROW(shuffle.add_check(std::vector<virCommitment>{}, std::vector<virCommitment>{c}, 4), std::runtime_error);
    EXPECT_THROW(shuffle.add_check({c}, {c, c}, 4), std::runtime_error);
    EXPECT_THROW(shuffle.add_check({c}, {c}, 3), std::runtime_error);
}

TEST_F(ShuffleProtocolTest, ValidRowPermutationFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    shuffleProtocol shuffle;
    shuffle.setup(kzg, lookup);

    const poly lhs_col0 = poly_from_domain_ints({1, 2, 3, 4});
    const poly lhs_col1 = poly_from_domain_ints({10, 20, 30, 40});
    const poly rhs_col0 = poly_from_domain_ints({3, 1, 4, 2});
    const poly rhs_col1 = poly_from_domain_ints({30, 10, 40, 20});

    shuffle.add_check({virCommitment(kzg.commit(lhs_col0)), virCommitment(kzg.commit(lhs_col1))},
                      {virCommitment(kzg.commit(rhs_col0)), virCommitment(kzg.commit(rhs_col1))},
                      4);

    EXPECT_TRUE(shuffle.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ShuffleProtocolTest, WrongPermutationFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    shuffleProtocol shuffle;
    shuffle.setup(kzg, lookup);

    const poly lhs_col0 = poly_from_domain_ints({1, 2, 3, 4});
    const poly lhs_col1 = poly_from_domain_ints({10, 20, 30, 40});
    const poly rhs_col0 = poly_from_domain_ints({3, 1, 4, 2});
    const poly rhs_col1 = poly_from_domain_ints({30, 10, 999, 20});

    shuffle.add_check({virCommitment(kzg.commit(lhs_col0)), virCommitment(kzg.commit(lhs_col1))},
                      {virCommitment(kzg.commit(rhs_col0)), virCommitment(kzg.commit(rhs_col1))},
                      4);

    EXPECT_TRUE(shuffle.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
