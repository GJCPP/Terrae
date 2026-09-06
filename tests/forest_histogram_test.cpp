#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "forest_histogram.hpp"
#include "hist_agg.hpp"
#include "index_routing.hpp"
#include "lookup.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "test_kzg_srs.hpp"
#include "tree_histogram_consistency.hpp"
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

class ForestHistogramProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestHistogramProtocolTest, ValidMaskedForestHistogramFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestHistogramProtocol protocol;
    protocol.setup(hist_agg, tree_histogram_consistency);

    const poly flattened_bucket_indices = poly_from_domain_ints({4, 5, 8, 11});
    const poly values = poly_from_domain_ints({1, 2, 5, 8});
    const poly leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {1, 2, 0, 0},
        {5, 0, 0, 8},
        {0, 0, 0, 0},
    });
    const poly full_histogram = matrix_poly_from_rows({
        {6, 2, 0, 8},
        {1, 2, 0, 0},
        {5, 0, 0, 8},
        {0, 0, 0, 0},
    });

    protocol.add_masked_check(virCommitment(kzg.commit(flattened_bucket_indices)),
                              virCommitment(kzg.commit(values)),
                              virCommitment(kzg.commit(leaf_histogram)),
                              virCommitment(kzg.commit(full_histogram)),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              4,
                              4,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestHistogramProtocolTest, RejectsWrongLeafHistogramAggregation)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestHistogramProtocol protocol;
    protocol.setup(hist_agg, tree_histogram_consistency);

    const poly flattened_bucket_indices = poly_from_domain_ints({4, 5, 8, 11});
    const poly values = poly_from_domain_ints({1, 2, 5, 8});
    const poly wrong_leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {1, 3, 0, 0},
        {5, 0, 0, 8},
        {0, 0, 0, 0},
    });
    const poly full_histogram = matrix_poly_from_rows({
        {6, 3, 0, 8},
        {1, 3, 0, 0},
        {5, 0, 0, 8},
        {0, 0, 0, 0},
    });

    protocol.add_masked_check(virCommitment(kzg.commit(flattened_bucket_indices)),
                              virCommitment(kzg.commit(values)),
                              virCommitment(kzg.commit(wrong_leaf_histogram)),
                              virCommitment(kzg.commit(full_histogram)),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              4,
                              4,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_FALSE(hist_agg.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestHistogramProtocolTest, RejectsWrongFullHistogramConsistency)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestHistogramProtocol protocol;
    protocol.setup(hist_agg, tree_histogram_consistency);

    const poly flattened_bucket_indices = poly_from_domain_ints({4, 5, 8, 11});
    const poly values = poly_from_domain_ints({1, 2, 5, 8});
    const poly leaf_histogram = matrix_poly_from_rows({
        {0, 0, 0, 0},
        {1, 2, 0, 0},
        {5, 0, 0, 8},
        {0, 0, 0, 0},
    });
    const poly wrong_full_histogram = matrix_poly_from_rows({
        {7, 2, 0, 8},
        {1, 2, 0, 0},
        {5, 0, 0, 8},
        {0, 0, 0, 0},
    });

    protocol.add_masked_check(virCommitment(kzg.commit(flattened_bucket_indices)),
                              virCommitment(kzg.commit(values)),
                              virCommitment(kzg.commit(leaf_histogram)),
                              virCommitment(kzg.commit(wrong_full_histogram)),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              4,
                              4,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestHistogramProtocolTest, ValidThreeChannelForestHistogramFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestHistogramProtocol protocol;
    protocol.setup(hist_agg, tree_histogram_consistency);

    forestHistogramChannelBatch channels;
    channels.values = {
        virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 1}))),
        virCommitment(kzg.commit(poly_from_domain_ints({10, 20, 30, 40}))),
        virCommitment(kzg.commit(poly_from_domain_ints({3, 5, 7, 11}))),
    };
    channels.leaf_histograms = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {0, 0, 0, 0},
            {1, 1, 0, 0},
            {1, 0, 0, 1},
            {0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {0, 0, 0, 0},
            {10, 20, 0, 0},
            {30, 0, 0, 40},
            {0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {0, 0, 0, 0},
            {3, 5, 0, 0},
            {7, 0, 0, 11},
            {0, 0, 0, 0},
        }))),
    };
    channels.full_histograms = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {2, 1, 0, 1},
            {1, 1, 0, 0},
            {1, 0, 0, 1},
            {0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {40, 20, 0, 40},
            {10, 20, 0, 0},
            {30, 0, 0, 40},
            {0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {10, 5, 0, 11},
            {3, 5, 0, 0},
            {7, 0, 0, 11},
            {0, 0, 0, 0},
        }))),
    };

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({4, 5, 8, 11}))),
                              channels,
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              4,
                              4,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestHistogramProtocolTest, RejectsWrongChannelInsideThreeChannelBatch)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestHistogramProtocol protocol;
    protocol.setup(hist_agg, tree_histogram_consistency);

    forestHistogramChannelBatch channels;
    channels.values = {
        virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 1}))),
        virCommitment(kzg.commit(poly_from_domain_ints({10, 20, 30, 40}))),
        virCommitment(kzg.commit(poly_from_domain_ints({3, 5, 7, 11}))),
    };
    channels.leaf_histograms = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {0, 0, 0, 0},
            {1, 1, 0, 0},
            {1, 0, 0, 1},
            {0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {0, 0, 0, 0},
            {10, 20, 0, 0},
            {30, 0, 0, 40},
            {0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {0, 0, 0, 0},
            {3, 6, 0, 0},
            {7, 0, 0, 11},
            {0, 0, 0, 0},
        }))),
    };
    channels.full_histograms = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {2, 1, 0, 1},
            {1, 1, 0, 0},
            {1, 0, 0, 1},
            {0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {40, 20, 0, 40},
            {10, 20, 0, 0},
            {30, 0, 0, 40},
            {0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {10, 6, 0, 11},
            {3, 6, 0, 0},
            {7, 0, 0, 11},
            {0, 0, 0, 0},
        }))),
    };

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({4, 5, 8, 11}))),
                              channels,
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              4,
                              4,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_FALSE(hist_agg.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
