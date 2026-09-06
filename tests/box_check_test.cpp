#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "box_check.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
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

} // namespace

class BoxCheckProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(BoxCheckProtocolTest, ValidBoxesFinalize)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    boxCheckProtocol box;
    box.setup(kzg, less);

    box.add_check(virCommitment(kzg.commit(matrix_poly_from_rows({{5, 8}, {2, 7}}))),
                  virCommitment(kzg.commit(matrix_poly_from_rows({{4, 8}, {1, 6}}))),
                  virCommitment(kzg.commit(matrix_poly_from_rows({{7, 9}, {3, 8}}))),
                  2,
                  2,
                  4);

    EXPECT_TRUE(box.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(BoxCheckProtocolTest, RejectsValuesOutsideHalfOpenInterval)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    boxCheckProtocol box;
    box.setup(kzg, less);

    box.add_check(virCommitment(kzg.commit(matrix_poly_from_rows({{5, 9}, {2, 7}}))),
                  virCommitment(kzg.commit(matrix_poly_from_rows({{4, 8}, {1, 6}}))),
                  virCommitment(kzg.commit(matrix_poly_from_rows({{7, 9}, {3, 8}}))),
                  2,
                  2,
                  4);

    EXPECT_TRUE(box.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(BoxCheckProtocolTest, AddCheckImmediatelyReducesToLessCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    {
        boxCheckProtocol box;
        box.setup(kzg, less);

        box.add_check(virCommitment(kzg.commit(matrix_poly_from_rows({{5, 9}, {2, 7}}))),
                      virCommitment(kzg.commit(matrix_poly_from_rows({{4, 8}, {1, 6}}))),
                      virCommitment(kzg.commit(matrix_poly_from_rows({{7, 9}, {3, 8}}))),
                      2,
                      2,
                      4);

        EXPECT_TRUE(less.finalize());
        EXPECT_FALSE(lookup.finalize());
    }

    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
