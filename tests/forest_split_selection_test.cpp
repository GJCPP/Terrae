#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "forest_split_selection.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "masked_argmax.hpp"
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

class ForestSplitSelectionProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestSplitSelectionProtocolTest, ValidSplitSelectionFinalizes)
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

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol protocol;
    protocol.setup(masked_argmax, row_sum, zero, less);

    forestSplitSelectionWitness witness;
    witness.quantized_gains = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 4, 2, 3, 5, 7, 6, 0},
        {2, 6, 1, 4, 3, 0, 5, 2},
    })));
    witness.tie_break_scores = virCommitment(kzg.commit(matrix_poly_from_rows({
        {15, 38, 21, 28, 43, 58, 49, 0},
        {23, 54, 13, 36, 27, 2, 41, 16},
    })));
    witness.candidate_mask = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 1, 1, 0, 1, 1, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 0},
    })));
    witness.left_counts = witness.candidate_mask;
    witness.right_counts = witness.candidate_mask;
    witness.active_node_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));
    witness.selector = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
    })));
    witness.selected_gains = virCommitment(kzg.commit(poly_from_domain_ints({7, 6})));
    witness.selected_tie_break_scores = virCommitment(kzg.commit(poly_from_domain_ints({58, 54})));
    witness.selected_candidate_indices = virCommitment(kzg.commit(poly_from_domain_ints({5, 1})));
    witness.selected_features = virCommitment(kzg.commit(poly_from_domain_ints({1, 0})));
    witness.selected_bins = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));

    protocol.add_check(witness, 2, 2, 4, 8, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitSelectionProtocolTest, EmptyChildCannotBeMarkedAsCandidate)
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

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol protocol;
    protocol.setup(masked_argmax, row_sum, zero, less);

    forestSplitSelectionWitness witness;
    witness.quantized_gains = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 4, 2, 3, 5, 7, 6, 0},
        {2, 6, 1, 4, 3, 0, 5, 2},
    })));
    witness.tie_break_scores = virCommitment(kzg.commit(matrix_poly_from_rows({
        {15, 38, 21, 28, 43, 58, 49, 0},
        {23, 54, 13, 36, 27, 2, 41, 16},
    })));
    witness.candidate_mask = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 1, 1, 0, 1, 1, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 0},
    })));
    witness.left_counts = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0, 1, 0, 1, 1, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 0},
    })));
    witness.right_counts = witness.candidate_mask;
    witness.active_node_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));
    witness.selector = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
    })));
    witness.selected_gains = virCommitment(kzg.commit(poly_from_domain_ints({7, 6})));
    witness.selected_tie_break_scores = virCommitment(kzg.commit(poly_from_domain_ints({58, 54})));
    witness.selected_candidate_indices = virCommitment(kzg.commit(poly_from_domain_ints({5, 1})));
    witness.selected_features = virCommitment(kzg.commit(poly_from_domain_ints({1, 0})));
    witness.selected_bins = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));

    protocol.add_check(witness, 2, 2, 4, 8, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitSelectionProtocolTest, HiddenValidCandidateFailsCompleteness)
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

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol protocol;
    protocol.setup(masked_argmax, row_sum, zero, less);

    forestSplitSelectionWitness witness;
    witness.quantized_gains = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 4, 2, 3, 5, 7, 6, 0},
        {2, 6, 1, 4, 3, 0, 5, 2},
    })));
    witness.tie_break_scores = virCommitment(kzg.commit(matrix_poly_from_rows({
        {15, 38, 21, 28, 43, 58, 49, 0},
        {23, 54, 13, 36, 27, 2, 41, 16},
    })));
    witness.candidate_mask = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 1, 1, 0, 1, 1, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 0},
    })));
    witness.left_counts = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 1, 1, 0, 1, 1, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 0},
    })));
    witness.right_counts = witness.left_counts;
    witness.active_node_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));
    witness.selector = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
    })));
    witness.selected_gains = virCommitment(kzg.commit(poly_from_domain_ints({7, 6})));
    witness.selected_tie_break_scores = virCommitment(kzg.commit(poly_from_domain_ints({58, 54})));
    witness.selected_candidate_indices = virCommitment(kzg.commit(poly_from_domain_ints({5, 1})));
    witness.selected_features = virCommitment(kzg.commit(poly_from_domain_ints({1, 0})));
    witness.selected_bins = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));

    protocol.add_check(witness, 2, 2, 4, 8, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitSelectionProtocolTest, FinalBinCannotBeMarkedAsCandidate)
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

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol protocol;
    protocol.setup(masked_argmax, row_sum, zero, less);

    forestSplitSelectionWitness witness;
    witness.quantized_gains = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 4, 2, 3, 5, 7, 6, 0},
        {2, 6, 1, 4, 3, 0, 5, 2},
    })));
    witness.tie_break_scores = virCommitment(kzg.commit(matrix_poly_from_rows({
        {15, 38, 21, 28, 43, 58, 49, 0},
        {23, 54, 13, 36, 27, 2, 41, 16},
    })));
    witness.candidate_mask = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 1, 1, 1, 1, 1, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 0},
    })));
    witness.left_counts = witness.candidate_mask;
    witness.right_counts = witness.candidate_mask;
    witness.active_node_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));
    witness.selector = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
    })));
    witness.selected_gains = virCommitment(kzg.commit(poly_from_domain_ints({7, 6})));
    witness.selected_tie_break_scores = virCommitment(kzg.commit(poly_from_domain_ints({58, 54})));
    witness.selected_candidate_indices = virCommitment(kzg.commit(poly_from_domain_ints({5, 1})));
    witness.selected_features = virCommitment(kzg.commit(poly_from_domain_ints({1, 0})));
    witness.selected_bins = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));

    protocol.add_check(witness, 2, 2, 4, 8, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitSelectionProtocolTest, WrongSelectedBinFailsConsistency)
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

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol protocol;
    protocol.setup(masked_argmax, row_sum, zero, less);

    forestSplitSelectionWitness witness;
    witness.quantized_gains = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 4, 2, 3, 5, 7, 6, 0},
        {2, 6, 1, 4, 3, 0, 5, 2},
    })));
    witness.tie_break_scores = virCommitment(kzg.commit(matrix_poly_from_rows({
        {15, 38, 21, 28, 43, 58, 49, 0},
        {23, 54, 13, 36, 27, 2, 41, 16},
    })));
    witness.candidate_mask = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 1, 1, 0, 1, 1, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 0},
    })));
    witness.left_counts = witness.candidate_mask;
    witness.right_counts = witness.candidate_mask;
    witness.active_node_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 1})));
    witness.selector = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0},
    })));
    witness.selected_gains = virCommitment(kzg.commit(poly_from_domain_ints({7, 6})));
    witness.selected_tie_break_scores = virCommitment(kzg.commit(poly_from_domain_ints({58, 54})));
    witness.selected_candidate_indices = virCommitment(kzg.commit(poly_from_domain_ints({5, 1})));
    witness.selected_features = virCommitment(kzg.commit(poly_from_domain_ints({1, 0})));
    witness.selected_bins = virCommitment(kzg.commit(poly_from_domain_ints({2, 1})));

    protocol.add_check(witness, 2, 2, 4, 8, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitSelectionProtocolTest, TieBreakPrefersSmallerCandidateIndex)
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

    maskedArgmaxProtocol masked_argmax;
    masked_argmax.setup(kzg, zero, selector_check, row_sum, column_replication, less);

    forestSplitSelectionProtocol protocol;
    protocol.setup(masked_argmax, row_sum, zero, less);

    forestSplitSelectionWitness witness;
    witness.quantized_gains = virCommitment(kzg.commit(matrix_poly_from_rows({
        {5, 7, 7, 3, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
    })));
    witness.tie_break_scores = virCommitment(kzg.commit(matrix_poly_from_rows({
        {47, 62, 61, 28, 3, 2, 1, 0},
        {7, 6, 5, 4, 3, 2, 1, 0},
    })));
    witness.candidate_mask = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
    })));
    witness.left_counts = witness.candidate_mask;
    witness.right_counts = witness.candidate_mask;
    witness.active_node_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 0})));
    witness.selector = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
    })));
    witness.selected_gains = virCommitment(kzg.commit(poly_from_domain_ints({7, 0})));
    witness.selected_tie_break_scores = virCommitment(kzg.commit(poly_from_domain_ints({62, 0})));
    witness.selected_candidate_indices = virCommitment(kzg.commit(poly_from_domain_ints({1, 0})));
    witness.selected_features = virCommitment(kzg.commit(poly_from_domain_ints({0, 0})));
    witness.selected_bins = virCommitment(kzg.commit(poly_from_domain_ints({1, 0})));

    protocol.add_check(witness, 2, 2, 4, 8, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(masked_argmax.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
