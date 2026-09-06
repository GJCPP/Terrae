#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "plonk/blinding.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

class PlonkBlindingTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkBlindingTest, BlindingPreservesSubgroupEvaluations)
{
    const poly base = poly_from_evals(std::vector<Fr>{Fr(1), Fr(2), Fr(3), Fr(4)});
    const poly blinded = blind_plonk_polynomial_on_subgroup(base, 4, {Fr(9), Fr(10)});

    EXPECT_EQ(blinded.evaluations_on_domain(4), base.evaluations_on_domain(4));
    EXPECT_NE(blinded.eval(Fr(5)), base.eval(Fr(5)));
}
