#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "matrix_sum.hpp"
#include "selector_check.hpp"
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

class SelectorCheckProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(SelectorCheckProtocolTest, BooleanCheckAcceptsBinaryWitness)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector;
    selector.setup(kzg, zero, row_sum);

    const poly values = poly_from_domain_ints({0, 1, 1, 0, 1, 0, 0, 1});
    selector.add_boolean_check(virCommitment(kzg.commit(values)), 8);

    EXPECT_TRUE(selector.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SelectorCheckProtocolTest, BooleanCheckRejectsNonBinaryWitness)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector;
    selector.setup(kzg, zero, row_sum);

    const poly values = poly_from_domain_ints({0, 1, 2, 0, 1, 0, 0, 1});
    selector.add_boolean_check(virCommitment(kzg.commit(values)), 8);

    EXPECT_TRUE(selector.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SelectorCheckProtocolTest, OneHotRowsAcceptValidSelector)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector;
    selector.setup(kzg, zero, row_sum);

    const poly matrix = matrix_poly_from_rows({
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1},
        {0, 0, 1, 0},
    });
    selector.add_one_hot_rows_check(virCommitment(kzg.commit(matrix)), 4, 4);

    EXPECT_TRUE(selector.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SelectorCheckProtocolTest, OneHotRowsRejectRowWithTwoOnes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector;
    selector.setup(kzg, zero, row_sum);

    const poly matrix = matrix_poly_from_rows({
        {1, 0, 0, 0},
        {0, 1, 1, 0},
        {0, 0, 0, 1},
        {0, 0, 1, 0},
    });
    selector.add_one_hot_rows_check(virCommitment(kzg.commit(matrix)), 4, 4);

    EXPECT_TRUE(selector.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SelectorCheckProtocolTest, MaskedOneHotRowsSupportDisabledRows)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector;
    selector.setup(kzg, zero, row_sum);

    const poly matrix = matrix_poly_from_rows({
        {1, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 0},
    });
    const poly mask = poly_from_domain_ints({1, 0, 1, 0});
    selector.add_masked_one_hot_rows_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(mask)), 4, 4);

    EXPECT_TRUE(selector.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SelectorCheckProtocolTest, MaskedOneHotRowsRejectMaskMismatch)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector;
    selector.setup(kzg, zero, row_sum);

    const poly matrix = matrix_poly_from_rows({
        {1, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 0},
    });
    const poly mask = poly_from_domain_ints({1, 1, 1, 0});
    selector.add_masked_one_hot_rows_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(mask)), 4, 4);

    EXPECT_TRUE(selector.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
