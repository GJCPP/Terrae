#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "forest_split_gain.hpp"
#include "kzg.hpp"
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

std::vector<std::vector<int>> square_matrix(const std::vector<std::vector<int>>& matrix)
{
    std::vector<std::vector<int>> out = matrix;
    for (auto& row : out) {
        for (int& value : row) {
            value *= value;
        }
    }
    return out;
}

std::vector<std::vector<int>> add_matrices(const std::vector<std::vector<int>>& lhs,
                                           const std::vector<std::vector<int>>& rhs)
{
    std::vector<std::vector<int>> out = lhs;
    for (size_t row = 0; row < out.size(); ++row) {
        for (size_t col = 0; col < out[row].size(); ++col) {
            out[row][col] += rhs[row][col];
        }
    }
    return out;
}

std::vector<std::vector<int>> gain_denominator_matrix(const std::vector<std::vector<int>>& left_hessians,
                                                      const std::vector<std::vector<int>>& right_hessians,
                                                      int lambda)
{
    std::vector<std::vector<int>> out = left_hessians;
    for (size_t row = 0; row < out.size(); ++row) {
        for (size_t col = 0; col < out[row].size(); ++col) {
            const int left_den = left_hessians[row][col] + lambda;
            const int right_den = right_hessians[row][col] + lambda;
            const int parent_den = left_hessians[row][col] + right_hessians[row][col] + lambda;
            out[row][col] = left_den * right_den * parent_den;
        }
    }
    return out;
}

std::vector<std::vector<int>> gain_numerator_matrix(const std::vector<std::vector<int>>& left_gradients,
                                                    const std::vector<std::vector<int>>& right_gradients,
                                                    const std::vector<std::vector<int>>& left_hessians,
                                                    const std::vector<std::vector<int>>& right_hessians,
                                                    int lambda,
                                                    int split_penalty)
{
    std::vector<std::vector<int>> out = left_gradients;
    for (size_t row = 0; row < out.size(); ++row) {
        for (size_t col = 0; col < out[row].size(); ++col) {
            const int left_gradient = left_gradients[row][col];
            const int right_gradient = right_gradients[row][col];
            const int parent_gradient = left_gradient + right_gradient;
            const int left_den = left_hessians[row][col] + lambda;
            const int right_den = right_hessians[row][col] + lambda;
            const int parent_den = left_hessians[row][col] + right_hessians[row][col] + lambda;
            const int gain_denominator = left_den * right_den * parent_den;

            out[row][col] = left_gradient * left_gradient * right_den * parent_den +
                            right_gradient * right_gradient * left_den * parent_den -
                            parent_gradient * parent_gradient * left_den * right_den -
                            split_penalty * gain_denominator;
        }
    }
    return out;
}

} // namespace

class ForestSplitGainProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestSplitGainProtocolTest, ValidRegularizedGainFinalize)
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

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    forestSplitGainProtocol protocol;
    protocol.setup(zero, less, signed_range);

    const std::vector<std::vector<int>> left_gradients = {
        {1, -2, 0, 3, -1, 2, -3, 1},
        {2, -1, 1, -2, 0, 3, -1, 2},
    };
    const std::vector<std::vector<int>> right_gradients = {
        {2, 1, -1, 0, 3, -2, 1, -1},
        {-1, 2, 0, 1, 2, -1, 3, 0},
    };
    const std::vector<std::vector<int>> left_hessians = {
        {1, 2, 1, 3, 2, 1, 2, 1},
        {1, 1, 2, 2, 1, 2, 1, 3},
    };
    const std::vector<std::vector<int>> right_hessians = {
        {1, 1, 2, 1, 2, 1, 1, 2},
        {2, 1, 1, 1, 1, 1, 2, 1},
    };
    const int lambda = 2;
    const int split_penalty = 1;

    const std::vector<std::vector<int>> parent_gradients = add_matrices(left_gradients, right_gradients);
    const std::vector<std::vector<int>> left_squares = square_matrix(left_gradients);
    const std::vector<std::vector<int>> right_squares = square_matrix(right_gradients);
    const std::vector<std::vector<int>> parent_squares = square_matrix(parent_gradients);
    const std::vector<std::vector<int>> gain_numerators =
        gain_numerator_matrix(left_gradients, right_gradients, left_hessians, right_hessians, lambda, split_penalty);
    const std::vector<std::vector<int>> gain_denominators =
        gain_denominator_matrix(left_hessians, right_hessians, lambda);

    forestSplitGainWitness witness;
    witness.left_gradients = virCommitment(kzg.commit(matrix_poly_from_rows(left_gradients)));
    witness.right_gradients = virCommitment(kzg.commit(matrix_poly_from_rows(right_gradients)));
    witness.left_hessians = virCommitment(kzg.commit(matrix_poly_from_rows(left_hessians)));
    witness.right_hessians = virCommitment(kzg.commit(matrix_poly_from_rows(right_hessians)));
    witness.left_gradient_squares = virCommitment(kzg.commit(matrix_poly_from_rows(left_squares)));
    witness.right_gradient_squares = virCommitment(kzg.commit(matrix_poly_from_rows(right_squares)));
    witness.parent_gradient_squares = virCommitment(kzg.commit(matrix_poly_from_rows(parent_squares)));
    witness.gain_numerators = virCommitment(kzg.commit(matrix_poly_from_rows(gain_numerators)));
    witness.gain_denominators = virCommitment(kzg.commit(matrix_poly_from_rows(gain_denominators)));

    protocol.add_check(
        witness, 2, 2, 4, fixedPointEncoding::signed_offset_binary(6), 4, 5, lambda, fr_from_int(split_penalty));

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitGainProtocolTest, WrongGainNumeratorFailsZeroCheck)
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

    signedRangeCheckProtocol signed_range;
    signed_range.setup(kzg, less);

    forestSplitGainProtocol protocol;
    protocol.setup(zero, less, signed_range);

    const std::vector<std::vector<int>> left_gradients = {
        {1, -2, 0, 3, -1, 2, -3, 1},
        {2, -1, 1, -2, 0, 3, -1, 2},
    };
    const std::vector<std::vector<int>> right_gradients = {
        {2, 1, -1, 0, 3, -2, 1, -1},
        {-1, 2, 0, 1, 2, -1, 3, 0},
    };
    const std::vector<std::vector<int>> left_hessians = {
        {1, 2, 1, 3, 2, 1, 2, 1},
        {1, 1, 2, 2, 1, 2, 1, 3},
    };
    const std::vector<std::vector<int>> right_hessians = {
        {1, 1, 2, 1, 2, 1, 1, 2},
        {2, 1, 1, 1, 1, 1, 2, 1},
    };
    const int lambda = 2;
    const int split_penalty = 1;

    const std::vector<std::vector<int>> parent_gradients = add_matrices(left_gradients, right_gradients);
    const std::vector<std::vector<int>> left_squares = square_matrix(left_gradients);
    const std::vector<std::vector<int>> right_squares = square_matrix(right_gradients);
    const std::vector<std::vector<int>> parent_squares = square_matrix(parent_gradients);
    std::vector<std::vector<int>> wrong_gain_numerators =
        gain_numerator_matrix(left_gradients, right_gradients, left_hessians, right_hessians, lambda, split_penalty);
    wrong_gain_numerators[0][1] += 1;
    const std::vector<std::vector<int>> gain_denominators =
        gain_denominator_matrix(left_hessians, right_hessians, lambda);

    forestSplitGainWitness witness;
    witness.left_gradients = virCommitment(kzg.commit(matrix_poly_from_rows(left_gradients)));
    witness.right_gradients = virCommitment(kzg.commit(matrix_poly_from_rows(right_gradients)));
    witness.left_hessians = virCommitment(kzg.commit(matrix_poly_from_rows(left_hessians)));
    witness.right_hessians = virCommitment(kzg.commit(matrix_poly_from_rows(right_hessians)));
    witness.left_gradient_squares = virCommitment(kzg.commit(matrix_poly_from_rows(left_squares)));
    witness.right_gradient_squares = virCommitment(kzg.commit(matrix_poly_from_rows(right_squares)));
    witness.parent_gradient_squares = virCommitment(kzg.commit(matrix_poly_from_rows(parent_squares)));
    witness.gain_numerators = virCommitment(kzg.commit(matrix_poly_from_rows(wrong_gain_numerators)));
    witness.gain_denominators = virCommitment(kzg.commit(matrix_poly_from_rows(gain_denominators)));

    protocol.add_check(
        witness, 2, 2, 4, fixedPointEncoding::signed_offset_binary(6), 4, 5, lambda, fr_from_int(split_penalty));

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(signed_range.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
