#include "algebra_utils.hpp"
#include "argmax.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "less_check.hpp"
#include "matrix_sum.hpp"
#include "selector_check.hpp"
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

class ArgmaxProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ArgmaxProtocolTest, ValidArgmaxProofFinalizes)
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

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    argmaxProtocol argmax;
    argmax.setup(kzg, selector_check, row_sum, column_replication, less);

    const poly matrix = matrix_poly_from_rows({
        {1, 5, 2, 4},
        {3, 2, 7, 6},
        {0, 8, 1, 2},
        {9, 3, 4, 5},
    });
    const poly max_values = poly_from_domain_ints({5, 7, 8, 9});
    const poly selector = matrix_poly_from_rows({
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {1, 0, 0, 0},
    });

    argmax.add_check(
        virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(max_values)), virCommitment(kzg.commit(selector)), 4, 4, 4);

    EXPECT_TRUE(argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ArgmaxProtocolTest, RejectsSelectorThatIsNotOneHot)
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

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    argmaxProtocol argmax;
    argmax.setup(kzg, selector_check, row_sum, column_replication, less);

    const poly matrix = matrix_poly_from_rows({
        {1, 5, 2, 4},
        {3, 2, 7, 6},
        {0, 8, 1, 2},
        {9, 3, 4, 5},
    });
    const poly max_values = poly_from_domain_ints({5, 7, 8, 9});
    const poly selector = matrix_poly_from_rows({
        {0, 1, 1, 0},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {1, 0, 0, 0},
    });

    argmax.add_check(
        virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(max_values)), virCommitment(kzg.commit(selector)), 4, 4, 4);

    EXPECT_TRUE(argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ArgmaxProtocolTest, RejectsClaimThatIsNotMaximum)
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

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    argmaxProtocol argmax;
    argmax.setup(kzg, selector_check, row_sum, column_replication, less);

    const poly matrix = matrix_poly_from_rows({
        {1, 5, 2, 4},
        {3, 2, 7, 6},
        {0, 8, 1, 2},
        {9, 3, 4, 5},
    });
    const poly wrong_max_values = poly_from_domain_ints({4, 7, 8, 9});
    const poly selector = matrix_poly_from_rows({
        {0, 0, 0, 1},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {1, 0, 0, 0},
    });

    argmax.add_check(virCommitment(kzg.commit(matrix)),
                     virCommitment(kzg.commit(wrong_max_values)),
                     virCommitment(kzg.commit(selector)),
                     4,
                     4,
                     4);

    EXPECT_TRUE(argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
