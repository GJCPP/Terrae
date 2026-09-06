#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "box_check.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "forest_depth_consistency.hpp"
#include "forest_split_update.hpp"
#include "hist_agg.hpp"
#include "index_routing.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "matrix_sum.hpp"
#include "parent_child_box_inheritance.hpp"
#include "parent_child_tree_consistency.hpp"
#include "root_box_range.hpp"
#include "root_selector_consistency.hpp"
#include "selector_check.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
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

forestSplitUpdateWitness valid_witness(kzgProtocol& kzg)
{
    forestSplitUpdateWitness witness;
    witness.root_indices = virCommitment(kzg.commit(poly_from_domain_ints({0})));
    witness.active_node_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0})));
    witness.current_leaf_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0})));
    witness.current_internal_mask = virCommitment(kzg.commit(poly_from_domain_ints({0, 0, 0, 0})));
    witness.next_leaf_mask = virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 0})));
    witness.next_internal_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0})));
    witness.next_active_node_mask = virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 0})));
    witness.left_child_occupancy = virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 0, 0})));
    witness.right_child_occupancy = virCommitment(kzg.commit(poly_from_domain_ints({0, 0, 1, 0})));

    witness.left_child_indices = virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0})));
    witness.right_child_indices = virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0})));
    witness.node_tree_ids = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 0})));
    witness.selected_features = virCommitment(kzg.commit(poly_from_domain_ints({0, 0, 0, 0})));
    witness.selected_bins = virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0})));
    witness.feature_selector = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    })));
    witness.lower_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {2, 0},
        {0, 0},
    })));
    witness.upper_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {4, 4},
        {2, 4},
        {4, 4},
        {0, 0},
    })));
    witness.left_lower = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    })));
    witness.left_upper = virCommitment(kzg.commit(matrix_poly_from_rows({
        {2, 4},
        {0, 0},
        {0, 0},
        {0, 0},
    })));
    witness.right_lower = virCommitment(kzg.commit(matrix_poly_from_rows({
        {2, 0},
        {0, 0},
        {0, 0},
        {0, 0},
    })));
    witness.right_upper = virCommitment(kzg.commit(matrix_poly_from_rows({
        {4, 4},
        {0, 0},
        {0, 0},
        {0, 0},
    })));
    witness.thresholds = virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0})));
    return witness;
}

} // namespace

class ForestSplitUpdateProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestSplitUpdateProtocolTest, ValidSplitUpdateFinalizes)
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

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    selectorCheckProtocol selector;
    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);
    selector.setup(kzg, zero, row_sum);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    rootSelectorConsistencyProtocol root_selector;
    root_selector.setup(kzg, zero, hist_agg, index_routing);

    nodeRoleConsistencyProtocol node_role;
    node_role.setup(kzg, zero, hist_agg);

    forestDepthConsistencyProtocol forest_depth_consistency;
    forest_depth_consistency.setup(kzg, zero, index_routing);

    rootBoxRangeProtocol root_box_range;
    root_box_range.setup(kzg, zero, index_routing);

    parentChildTreeConsistencyProtocol parent_child_tree;
    parent_child_tree.setup(kzg, zero, index_routing);

    parentChildBoxInheritanceProtocol parent_child_box;
    parent_child_box.setup(kzg, zero, selector, column_replication, index_routing, less);

    forestStructureProtocol forest_structure;
    forest_structure.setup(kzg,
                           zero,
                           column_replication,
                           root_box_range,
                           root_selector,
                           node_role,
                           forest_depth_consistency,
                           parent_child_box,
                           parent_child_tree);

    forestSplitUpdateProtocol protocol;
    protocol.setup(forest_structure, hist_agg, row_sum, zero);

    protocol.add_check(valid_witness(kzg), 1, 4, 2, 4, 3);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(forest_structure.finalize());
    EXPECT_TRUE(parent_child_box.finalize());
    EXPECT_TRUE(parent_child_tree.finalize());
    EXPECT_TRUE(root_box_range.finalize());
    EXPECT_TRUE(node_role.finalize());
    EXPECT_TRUE(forest_depth_consistency.finalize());
    EXPECT_TRUE(root_selector.finalize());
    EXPECT_TRUE(selector.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestSplitUpdateProtocolTest, WrongNextLeafMaskFailsZeroCheck)
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

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    selectorCheckProtocol selector;
    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);
    selector.setup(kzg, zero, row_sum);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    rootSelectorConsistencyProtocol root_selector;
    root_selector.setup(kzg, zero, hist_agg, index_routing);

    nodeRoleConsistencyProtocol node_role;
    node_role.setup(kzg, zero, hist_agg);

    forestDepthConsistencyProtocol forest_depth_consistency;
    forest_depth_consistency.setup(kzg, zero, index_routing);

    rootBoxRangeProtocol root_box_range;
    root_box_range.setup(kzg, zero, index_routing);

    parentChildTreeConsistencyProtocol parent_child_tree;
    parent_child_tree.setup(kzg, zero, index_routing);

    parentChildBoxInheritanceProtocol parent_child_box;
    parent_child_box.setup(kzg, zero, selector, column_replication, index_routing, less);

    forestStructureProtocol forest_structure;
    forest_structure.setup(kzg,
                           zero,
                           column_replication,
                           root_box_range,
                           root_selector,
                           node_role,
                           forest_depth_consistency,
                           parent_child_box,
                           parent_child_tree);

    forestSplitUpdateProtocol protocol;
    protocol.setup(forest_structure, hist_agg, row_sum, zero);

    forestSplitUpdateWitness witness = valid_witness(kzg);
    witness.next_leaf_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 0})));

    protocol.add_check(witness, 1, 4, 2, 4, 3);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(forest_structure.finalize());
    EXPECT_TRUE(parent_child_box.finalize());
    EXPECT_TRUE(parent_child_tree.finalize());
    EXPECT_TRUE(root_box_range.finalize());
    EXPECT_TRUE(node_role.finalize());
    EXPECT_TRUE(forest_depth_consistency.finalize());
    EXPECT_TRUE(root_selector.finalize());
    EXPECT_TRUE(selector.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
