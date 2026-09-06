#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_legality.hpp"

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

class ForestLegalityTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestLegalityTest, BuilderAcceptsCoveredDisjointForest)
{
    const forestLegalityPublicTable table = build_forest_legality_public_table(
        matrix_poly_from_rows({{1}, {1}, {2}, {2}}),
        matrix_poly_from_rows({{10, 11}, {20, 21}, {30, 31}, {40, 41}}),
        matrix_poly_from_rows({{12, 13}, {22, 23}, {32, 33}, {42, 43}}),
        4,
        2,
        2,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}}));

    EXPECT_EQ(table.tree_count, 2u);
    EXPECT_EQ(table.box_consistency_table.row_count, 4u);
    EXPECT_EQ(table.box_consistency_table.feature_count, 2u);
}

TEST_F(ForestLegalityTest, BuilderRejectsOutOfRangeTreeIndex)
{
    EXPECT_THROW(build_forest_legality_public_table(matrix_poly_from_rows({{1}, {3}}),
                                                    matrix_poly_from_rows({{10, 11}, {20, 21}}),
                                                    matrix_poly_from_rows({{12, 13}, {22, 23}}),
                                                    2,
                                                    2,
                                                    2,
                                                    std::nullopt),
                 std::runtime_error);
}

TEST_F(ForestLegalityTest, BuilderRejectsMissingTreeCoverage)
{
    EXPECT_THROW(build_forest_legality_public_table(matrix_poly_from_rows({{1}, {1}}),
                                                    matrix_poly_from_rows({{10, 11}, {20, 21}}),
                                                    matrix_poly_from_rows({{12, 13}, {22, 23}}),
                                                    2,
                                                    2,
                                                    2,
                                                    std::nullopt),
                 std::runtime_error);
}

TEST_F(ForestLegalityTest, BuilderAllowsPaddingSentinelTreeRows)
{
    const forestLegalityPublicTable table = build_forest_legality_public_table(
        matrix_poly_from_rows({{1}, {2}, {0}, {0}}),
        matrix_poly_from_rows({{10, 11}, {30, 31}, {0, 0}, {0, 0}}),
        matrix_poly_from_rows({{12, 13}, {32, 33}, {0, 0}, {0, 0}}),
        4,
        2,
        2,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}}));

    EXPECT_EQ(table.tree_count, 2u);
    EXPECT_EQ(table.box_consistency_table.row_count, 4u);
    EXPECT_TRUE(table.box_consistency_table.includes_leaf_index);
}
