#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "forest_split_statistics.hpp"
#include "kzg.hpp"
#include "matrix_sum.hpp"
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

class ForestSplitStatisticsProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestSplitStatisticsProtocolTest, ValidSingleChannelSplitStatisticsFinalize)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowPrefixSumProtocol row_prefix;
    row_prefix.setup(kzg, zero);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    forestSplitStatisticsProtocol protocol;
    protocol.setup(kzg, zero, row_prefix, row_sum, column_replication);

    const poly histogram = matrix_poly_from_rows({
        {1, 2, 4, 8, 3, 5, 7, 9},
        {0, 1, 0, 1, 2, 0, 2, 0},
    });
    const poly left_statistics = matrix_poly_from_rows({
        {1, 3, 7, 15, 3, 8, 15, 24},
        {0, 1, 1, 2, 2, 2, 4, 4},
    });
    const poly right_statistics = matrix_poly_from_rows({
        {14, 12, 8, 0, 21, 16, 9, 0},
        {2, 1, 1, 0, 2, 2, 0, 0},
    });

    protocol.add_check(virCommitment(kzg.commit(histogram)),
                       virCommitment(kzg.commit(left_statistics)),
                       virCommitment(kzg.commit(right_statistics)),
                       2,
                       2,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(row_prefix.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitStatisticsProtocolTest, RejectsWrongLeftPrefixStatistics)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowPrefixSumProtocol row_prefix;
    row_prefix.setup(kzg, zero);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    forestSplitStatisticsProtocol protocol;
    protocol.setup(kzg, zero, row_prefix, row_sum, column_replication);

    const poly histogram = matrix_poly_from_rows({
        {1, 2, 4, 8, 3, 5, 7, 9},
        {0, 1, 0, 1, 2, 0, 2, 0},
    });
    const poly wrong_left_statistics = matrix_poly_from_rows({
        {1, 3, 8, 15, 3, 8, 15, 24},
        {0, 1, 1, 2, 2, 2, 4, 4},
    });
    const poly right_statistics = matrix_poly_from_rows({
        {14, 12, 8, 0, 21, 16, 9, 0},
        {2, 1, 1, 0, 2, 2, 0, 0},
    });

    protocol.add_check(virCommitment(kzg.commit(histogram)),
                       virCommitment(kzg.commit(wrong_left_statistics)),
                       virCommitment(kzg.commit(right_statistics)),
                       2,
                       2,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(row_prefix.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitStatisticsProtocolTest, ValidThreeChannelSplitStatisticsFinalize)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    matrixRowPrefixSumProtocol row_prefix;
    row_prefix.setup(kzg, zero);

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    forestSplitStatisticsProtocol protocol;
    protocol.setup(kzg, zero, row_prefix, row_sum, column_replication);

    forestSplitStatisticsChannelBatch channels;
    channels.histograms = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {1, 1, 1, 1, 0, 1, 0, 1},
            {2, 0, 2, 0, 1, 1, 1, 1},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {1, 2, 4, 8, 3, 5, 7, 9},
            {0, 1, 0, 1, 2, 0, 2, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {2, 3, 5, 7, 11, 13, 17, 19},
            {23, 29, 31, 37, 41, 43, 47, 53},
        }))),
    };
    channels.left_statistics = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {1, 2, 3, 4, 0, 1, 1, 2},
            {2, 2, 4, 4, 1, 2, 3, 4},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {1, 3, 7, 15, 3, 8, 15, 24},
            {0, 1, 1, 2, 2, 2, 4, 4},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {2, 5, 10, 17, 11, 24, 41, 60},
            {23, 52, 83, 120, 41, 84, 131, 184},
        }))),
    };
    channels.right_statistics = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {3, 2, 1, 0, 2, 1, 1, 0},
            {2, 2, 0, 0, 3, 2, 1, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {14, 12, 8, 0, 21, 16, 9, 0},
            {2, 1, 1, 0, 2, 2, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {15, 12, 7, 0, 49, 36, 19, 0},
            {97, 68, 37, 0, 143, 100, 53, 0},
        }))),
    };

    protocol.add_check(channels, 2, 2, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(row_prefix.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
