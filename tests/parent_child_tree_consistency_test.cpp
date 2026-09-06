#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "index_routing.hpp"
#include "lookup.hpp"
#include "parent_child_tree_consistency.hpp"
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

} // namespace

class ParentChildTreeConsistencyProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ParentChildTreeConsistencyProtocolTest, ValidMaskedConsistencyProofFinalizes)
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

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    parentChildTreeConsistencyProtocol protocol;
    protocol.setup(kzg, zero, index_routing);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 3, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 0, 3, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 1, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 0, 1, 1}))),
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ParentChildTreeConsistencyProtocolTest, RejectsChildFromDifferentTree)
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

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    parentChildTreeConsistencyProtocol protocol;
    protocol.setup(kzg, zero, index_routing);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 3, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 1, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 0, 1, 1}))),
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ParentChildTreeConsistencyProtocolTest, UnmaskedConsistencyProofFinalizes)
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

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    concatenatedTableLookupProtocol concatenated_table_lookup;
    concatenated_table_lookup.setup(kzg, table_lookup);

    indexRoutingProtocol index_routing;
    index_routing.setup(kzg, zero, column_replication, concatenated_table_lookup);

    parentChildTreeConsistencyProtocol protocol;
    protocol.setup(kzg, zero, index_routing);

    protocol.add_check(virCommitment(kzg.commit(poly_from_domain_ints({1, 2, 3, 0}))),
                       virCommitment(kzg.commit(poly_from_domain_ints({2, 1, 0, 3}))),
                       virCommitment(kzg.commit(poly_from_domain_ints({0, 0, 0, 0}))),
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
