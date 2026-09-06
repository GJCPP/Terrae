#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_box_consistency.hpp"

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

class ForestBoxConsistencyTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestBoxConsistencyTest, BuilderAcceptsDisjointBoxesWithinEachTree)
{
    const forestBoxConsistencyPublicTable table = build_forest_box_consistency_public_table(
        matrix_poly_from_rows({{0}, {0}, {1}, {1}}),
        matrix_poly_from_rows({{10, 11}, {20, 21}, {30, 31}, {40, 41}}),
        matrix_poly_from_rows({{12, 13}, {22, 23}, {32, 33}, {42, 43}}),
        4,
        2,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}}));

    EXPECT_EQ(table.row_count, 4u);
    EXPECT_EQ(table.feature_count, 2u);
    EXPECT_TRUE(table.includes_leaf_index);
}

TEST_F(ForestBoxConsistencyTest, BuilderRejectsOverlappingBoxesWithinOneTree)
{
    EXPECT_THROW(build_forest_box_consistency_public_table(matrix_poly_from_rows({{0}, {0}}),
                                                           matrix_poly_from_rows({{10, 10}, {11, 11}}),
                                                           matrix_poly_from_rows({{13, 13}, {14, 14}}),
                                                           2,
                                                           2,
                                                           std::nullopt),
                 std::runtime_error);
}

TEST_F(ForestBoxConsistencyTest, BuilderRejectsDuplicateLeafIndicesWithinOneTree)
{
    EXPECT_THROW(build_forest_box_consistency_public_table(matrix_poly_from_rows({{0}, {0}}),
                                                           matrix_poly_from_rows({{10, 10}, {20, 20}}),
                                                           matrix_poly_from_rows({{11, 11}, {21, 21}}),
                                                           2,
                                                           2,
                                                           matrix_poly_from_rows({{5}, {5}})),
                 std::runtime_error);
}
