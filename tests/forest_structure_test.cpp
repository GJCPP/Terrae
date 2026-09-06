#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "forest_structure.hpp"
#include "forest_depth_consistency.hpp"
#include "hist_agg.hpp"
#include "index_routing.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "matrix_sum.hpp"
#include "node_role_consistency.hpp"
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

} // namespace

class ForestStructureProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestStructureProtocolTest, ValidMaskedForestStructureFinalizes)
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

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    rootSelectorConsistencyProtocol root_selector_consistency;
    root_selector_consistency.setup(kzg, zero, hist_agg, index_routing);

    nodeRoleConsistencyProtocol node_role_consistency;
    node_role_consistency.setup(kzg, zero, hist_agg);

    forestDepthConsistencyProtocol forest_depth_consistency;
    forest_depth_consistency.setup(kzg, zero, index_routing);

    parentChildBoxInheritanceProtocol parent_child_box_inheritance;
    parent_child_box_inheritance.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    parentChildTreeConsistencyProtocol parent_child_tree_consistency;
    parent_child_tree_consistency.setup(kzg, zero, index_routing);

    rootBoxRangeProtocol root_box_range;
    root_box_range.setup(kzg, zero, index_routing);

    forestStructureProtocol protocol;
    protocol.setup(kzg,
                   zero,
                   column_replication,
                   root_box_range,
                   root_selector_consistency,
                   node_role_consistency,
                   forest_depth_consistency,
                   parent_child_box_inheritance,
                   parent_child_tree_consistency);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 3}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 1}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 2, 2, 1}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 2}))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {0, 0},
                                  {0, 0},
                                  {2, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {8, 8},
                                  {2, 8},
                                  {8, 8},
                                  {8, 8},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {2, 8},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {2, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {8, 8},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              2,
                              4,
                              2,
                              8,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(root_box_range.finalize());
    EXPECT_TRUE(root_selector_consistency.finalize());
    EXPECT_TRUE(node_role_consistency.finalize());
    EXPECT_TRUE(forest_depth_consistency.finalize());
    EXPECT_TRUE(parent_child_box_inheritance.finalize());
    EXPECT_TRUE(parent_child_tree_consistency.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestStructureProtocolTest, RejectsLeafWithNonZeroChildIdentifier)
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

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    rootSelectorConsistencyProtocol root_selector_consistency;
    root_selector_consistency.setup(kzg, zero, hist_agg, index_routing);

    nodeRoleConsistencyProtocol node_role_consistency;
    node_role_consistency.setup(kzg, zero, hist_agg);

    forestDepthConsistencyProtocol forest_depth_consistency;
    forest_depth_consistency.setup(kzg, zero, index_routing);

    parentChildBoxInheritanceProtocol parent_child_box_inheritance;
    parent_child_box_inheritance.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    parentChildTreeConsistencyProtocol parent_child_tree_consistency;
    parent_child_tree_consistency.setup(kzg, zero, index_routing);

    rootBoxRangeProtocol root_box_range;
    root_box_range.setup(kzg, zero, index_routing);

    forestStructureProtocol protocol;
    protocol.setup(kzg,
                   zero,
                   column_replication,
                   root_box_range,
                   root_selector_consistency,
                   node_role_consistency,
                   forest_depth_consistency,
                   parent_child_box_inheritance,
                   parent_child_tree_consistency);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 3}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 1}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 2, 2, 1}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 2}))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {0, 0},
                                  {0, 0},
                                  {2, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {8, 8},
                                  {2, 8},
                                  {8, 8},
                                  {8, 8},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {2, 8},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {2, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {8, 8},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              2,
                              4,
                              2,
                              8,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(root_box_range.finalize());
    EXPECT_TRUE(root_selector_consistency.finalize());
    EXPECT_TRUE(node_role_consistency.finalize());
    EXPECT_TRUE(forest_depth_consistency.finalize());
    EXPECT_TRUE(parent_child_box_inheritance.finalize());
    EXPECT_TRUE(parent_child_tree_consistency.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestStructureProtocolTest, RejectsNonZeroGhostNodeAttributes)
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

    matrixRowSumProtocol row_sum;
    row_sum.setup(kzg, zero);

    selectorCheckProtocol selector_check;
    selector_check.setup(kzg, zero, row_sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    histAggProtocol hist_agg(table_lookup);
    hist_agg.setup(kzg, zero, sum);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    rootSelectorConsistencyProtocol root_selector_consistency;
    root_selector_consistency.setup(kzg, zero, hist_agg, index_routing);

    nodeRoleConsistencyProtocol node_role_consistency;
    node_role_consistency.setup(kzg, zero, hist_agg);

    forestDepthConsistencyProtocol forest_depth_consistency;
    forest_depth_consistency.setup(kzg, zero, index_routing);

    parentChildBoxInheritanceProtocol parent_child_box_inheritance;
    parent_child_box_inheritance.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    parentChildTreeConsistencyProtocol parent_child_tree_consistency;
    parent_child_tree_consistency.setup(kzg, zero, index_routing);

    rootBoxRangeProtocol root_box_range;
    root_box_range.setup(kzg, zero, index_routing);

    forestStructureProtocol protocol;
    protocol.setup(kzg,
                   zero,
                   column_replication,
                   root_box_range,
                   root_selector_consistency,
                   node_role_consistency,
                   forest_depth_consistency,
                   parent_child_box_inheritance,
                   parent_child_tree_consistency);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 2, 2, 1}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 2}))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {0, 0},
                                  {0, 0},
                                  {2, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {8, 8},
                                  {2, 8},
                                  {8, 8},
                                  {8, 8},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {2, 8},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {2, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {8, 8},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              1,
                              4,
                              2,
                              8,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(root_box_range.finalize());
    EXPECT_TRUE(root_selector_consistency.finalize());
    EXPECT_TRUE(node_role_consistency.finalize());
    EXPECT_TRUE(forest_depth_consistency.finalize());
    EXPECT_TRUE(parent_child_box_inheritance.finalize());
    EXPECT_TRUE(parent_child_tree_consistency.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
