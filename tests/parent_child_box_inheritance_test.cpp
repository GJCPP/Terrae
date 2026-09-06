#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "index_routing.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "matrix_sum.hpp"
#include "parent_child_box_inheritance.hpp"
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

class ParentChildBoxInheritanceProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ParentChildBoxInheritanceProtocolTest, ValidInheritanceProofFinalizes)
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

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    parentChildBoxInheritanceProtocol protocol;
    protocol.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    protocol.add_masked_check(virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {1, 10},
                           {1, 10},
                           {3, 10},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {5, 20},
                           {3, 20},
                           {5, 20},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {1, 10},
                           {0, 0},
                           {0, 0},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {3, 20},
                           {0, 0},
                           {0, 0},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {3, 10},
                           {0, 0},
                           {0, 0},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {5, 20},
                           {0, 0},
                           {0, 0},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(poly_from_domain_ints({3, 0, 0, 0}))),
                              4,
                              2,
                              6);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ParentChildBoxInheritanceProtocolTest, RejectsNonSelectedFeatureMutation)
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

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    parentChildBoxInheritanceProtocol protocol;
    protocol.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    protocol.add_masked_check(virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {1, 10},
                           {1, 10},
                           {3, 10},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {5, 20},
                           {3, 20},
                           {5, 20},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {1, 10},
                           {0, 0},
                           {0, 0},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {3, 19},
                           {0, 0},
                           {0, 0},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {3, 10},
                           {0, 0},
                           {0, 0},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({
                           {5, 20},
                           {0, 0},
                           {0, 0},
                           {0, 0},
                       }))),
                       virCommitment(kzg.commit(poly_from_domain_ints({3, 0, 0, 0}))),
                              4,
                              2,
                              6);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ParentChildBoxInheritanceProtocolTest, MaskedRowsCanBeLeftUnconstrained)
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

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    parentChildBoxInheritanceProtocol protocol;
    protocol.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    protocol.add_masked_check(virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 10},
                                  {1, 10},
                                  {3, 10},
                                  {100, 200},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {5, 20},
                                  {3, 20},
                                  {5, 20},
                                  {50, 60},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 10},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {3, 20},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {3, 10},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {5, 20},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({3, 1, 0, 0}))),
                              4,
                              2,
                              8);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ParentChildBoxInheritanceProtocolTest, RejectsThresholdOnParentBoundaryViaChildNonEmptiness)
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

    tableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_lookup;
    concatenated_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_lookup);

    parentChildBoxInheritanceProtocol protocol;
    protocol.setup(kzg, zero, selector_check, column_replication, index_routing, less);

    protocol.add_masked_check(virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 10},
                                  {1, 10},
                                  {1, 10},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {5, 20},
                                  {1, 20},
                                  {5, 20},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 10},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 20},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {1, 10},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(matrix_poly_from_rows({
                                  {5, 20},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                              }))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              4,
                              2,
                              6);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(selector_check.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(row_sum.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
