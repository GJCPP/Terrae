#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "box_check.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "forest_histogram.hpp"
#include "forest_histogram_bucket_address.hpp"
#include "forest_histogram_refresh.hpp"
#include "forest_training_inference.hpp"
#include "hist_agg.hpp"
#include "index_routing.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
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

forestHistogramRefreshWitness valid_witness(kzgProtocol& kzg)
{
    forestHistogramRefreshWitness witness;
    witness.inference.query_tree_ids = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 1})));
    witness.inference.lifted_inputs = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 1},
        {0, 2},
        {2, 1},
        {3, 3},
    })));
    witness.inference.leaf_indices = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 2, 2})));
    witness.inference.leaf_lower_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {2, 0},
        {2, 0},
    })));
    witness.inference.leaf_upper_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {2, 4},
        {2, 4},
        {4, 4},
        {4, 4},
    })));
    witness.inference.tree_scores = virCommitment(kzg.commit(poly_from_domain_ints({5, 5, 7, 7})));
    witness.inference.node_tree_ids = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 0})));
    witness.inference.node_lower_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {2, 0},
        {0, 0},
    })));
    witness.inference.node_upper_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {4, 4},
        {2, 4},
        {4, 4},
        {0, 0},
    })));
    witness.inference.node_leaf_mask = virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 0})));
    witness.inference.node_leaf_values = virCommitment(kzg.commit(poly_from_domain_ints({0, 5, 7, 0})));

    witness.lifted_leaf_indices = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 1},
        {1, 1},
        {2, 2},
        {2, 2},
    })));
    witness.flattened_bucket_indices = virCommitment(kzg.commit(matrix_poly_from_rows({
        {9, 13},
        {8, 14},
        {18, 21},
        {19, 23},
    })));

    witness.channels.query_values = {
        virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 1}))),
        witness.inference.tree_scores,
    };
    witness.channels.lifted_values = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {1, 1},
            {1, 1},
            {1, 1},
            {1, 1},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {5, 5},
            {5, 5},
            {7, 7},
            {7, 7},
        }))),
    };
    witness.channels.leaf_histograms = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 0, 0, 0, 1, 1, 0},
            {0, 0, 1, 1, 0, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {0, 0, 0, 0, 0, 0, 0, 0},
            {5, 5, 0, 0, 0, 5, 5, 0},
            {0, 0, 7, 7, 0, 7, 0, 7},
            {0, 0, 0, 0, 0, 0, 0, 0},
        }))),
    };
    witness.channels.full_histograms = {
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {1, 1, 1, 1, 0, 2, 1, 1},
            {1, 1, 0, 0, 0, 1, 1, 0},
            {0, 0, 1, 1, 0, 1, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0},
        }))),
        virCommitment(kzg.commit(matrix_poly_from_rows({
            {5, 5, 7, 7, 0, 12, 5, 7},
            {5, 5, 0, 0, 0, 5, 5, 0},
            {0, 0, 7, 7, 0, 7, 0, 7},
            {0, 0, 0, 0, 0, 0, 0, 0},
        }))),
    };

    witness.parent_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0})));
    witness.left_child_indices = virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0})));
    witness.right_child_indices = virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0})));
    return witness;
}

} // namespace

class ForestHistogramRefreshProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestHistogramRefreshProtocolTest, ValidHistogramRefreshFinalizes)
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

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    boxCheckProtocol box_check;
    box_check.setup(kzg, less);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestTrainingInferenceProtocol training_inference;
    training_inference.setup(kzg, less, box_check, concatenated_lookup);

    forestHistogramBucketAddressProtocol bucket_address;
    bucket_address.setup(zero, column_replication, less);

    forestHistogramProtocol forest_histogram;
    forest_histogram.setup(hist_agg, tree_histogram_consistency);

    forestHistogramRefreshProtocol protocol;
    protocol.setup(training_inference, bucket_address, column_replication, forest_histogram);

    protocol.add_check(valid_witness(kzg), 4, 4, 2, 4, 3);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(forest_histogram.finalize());
    EXPECT_TRUE(bucket_address.finalize());
    EXPECT_TRUE(training_inference.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(box_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestHistogramRefreshProtocolTest, WrongLiftedChannelFailsReplication)
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

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    boxCheckProtocol box_check;
    box_check.setup(kzg, less);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    treeHistogramConsistencyProtocol tree_histogram_consistency;
    tree_histogram_consistency.setup(kzg, zero, index_routing);

    forestTrainingInferenceProtocol training_inference;
    training_inference.setup(kzg, less, box_check, concatenated_lookup);

    forestHistogramBucketAddressProtocol bucket_address;
    bucket_address.setup(zero, column_replication, less);

    forestHistogramProtocol forest_histogram;
    forest_histogram.setup(hist_agg, tree_histogram_consistency);

    forestHistogramRefreshProtocol protocol;
    protocol.setup(training_inference, bucket_address, column_replication, forest_histogram);

    forestHistogramRefreshWitness witness = valid_witness(kzg);
    witness.channels.lifted_values[1] = virCommitment(kzg.commit(matrix_poly_from_rows({
        {5, 5},
        {5, 6},
        {7, 7},
        {7, 7},
    })));
    witness.channels.leaf_histograms[1] = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0, 0, 0, 0, 0, 0, 0},
        {5, 5, 0, 0, 0, 5, 6, 0},
        {0, 0, 7, 7, 0, 7, 0, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
    })));
    witness.channels.full_histograms[1] = virCommitment(kzg.commit(matrix_poly_from_rows({
        {5, 5, 7, 7, 0, 12, 6, 7},
        {5, 5, 0, 0, 0, 5, 6, 0},
        {0, 0, 7, 7, 0, 7, 0, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
    })));

    protocol.add_check(witness, 4, 4, 2, 4, 3);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(forest_histogram.finalize());
    EXPECT_TRUE(bucket_address.finalize());
    EXPECT_TRUE(training_inference.finalize());
    EXPECT_TRUE(tree_histogram_consistency.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(box_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
