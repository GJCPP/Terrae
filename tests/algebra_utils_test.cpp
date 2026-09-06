#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"

#include <gtest/gtest.h>

namespace {

class AlgebraUtilsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

} // namespace

TEST_F(AlgebraUtilsTest, CanonicalSignedInt64FromFrSupportsNegativeValues)
{
    EXPECT_EQ(canonical_signed_int64_from_fr(fr_from_int(7), "algebra_utils_test"), 7);
    EXPECT_EQ(canonical_signed_int64_from_fr(fr_from_int(-1), "algebra_utils_test"), -1);
    EXPECT_EQ(canonical_signed_int64_from_fr(fr_from_int(-9), "algebra_utils_test"), -9);
}
