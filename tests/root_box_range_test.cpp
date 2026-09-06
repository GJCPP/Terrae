#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "index_routing.hpp"
#include "lookup.hpp"
#include "root_box_range.hpp"
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

class RootBoxRangeProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(RootBoxRangeProtocolTest, ValidRootBoxesFinalize)
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

    rootBoxRangeProtocol protocol;
    protocol.setup(kzg, zero, index_routing);

    protocol.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 3}))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({{0, 0}, {1, 0}, {2, 1}, {0, 0}}))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({{8, 8}, {2, 8}, {8, 8}, {8, 8}}))),
                       2,
                       4,
                       2,
                       8);

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

TEST_F(RootBoxRangeProtocolTest, RejectsNonFullRootUpper)
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

    rootBoxRangeProtocol protocol;
    protocol.setup(kzg, zero, index_routing);

    protocol.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 3}))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({{0, 0}, {1, 0}, {2, 1}, {0, 0}}))),
                       virCommitment(kzg.commit(matrix_poly_from_rows({{7, 8}, {2, 8}, {8, 8}, {8, 8}}))),
                       2,
                       4,
                       2,
                       8);

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
