#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "concatenated_table_lookup.hpp"
#include "index_routing.hpp"
#include "lookup.hpp"
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

class IndexRoutingProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(IndexRoutingProtocolTest, ValidRoutingProofFinalizes)
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

    const poly row_indices = poly_from_domain_ints({1, 3, 0, 2});
    const poly table = matrix_poly_from_rows({
        {10, 11, 12, 13},
        {20, 21, 22, 23},
        {30, 31, 32, 33},
        {40, 41, 42, 43},
    });
    const poly output = matrix_poly_from_rows({
        {20, 21, 22, 23},
        {40, 41, 42, 43},
        {10, 11, 12, 13},
        {30, 31, 32, 33},
    });

    index_routing.add_check(virCommitment(kzg.commit(row_indices)),
                            virCommitment(kzg.commit(table)),
                            virCommitment(kzg.commit(output)),
                            4,
                            4,
                            4);

    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(IndexRoutingProtocolTest, RejectsOutputThatDoesNotMatchSelectedRows)
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

    const poly row_indices = poly_from_domain_ints({1, 3, 0, 2});
    const poly table = matrix_poly_from_rows({
        {10, 11, 12, 13},
        {20, 21, 22, 23},
        {30, 31, 32, 33},
        {40, 41, 42, 43},
    });
    const poly wrong_output = matrix_poly_from_rows({
        {20, 21, 22, 23},
        {40, 41, 42, 43},
        {30, 31, 32, 33},
        {30, 31, 32, 33},
    });

    index_routing.add_check(virCommitment(kzg.commit(row_indices)),
                            virCommitment(kzg.commit(table)),
                            virCommitment(kzg.commit(wrong_output)),
                            4,
                            4,
                            4);

    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(IndexRoutingProtocolTest, MaskedRoutingAllowsDisabledRowsOnlyWhenOutputIsZero)
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

    const poly row_indices = poly_from_domain_ints({1, 3, 0, 2});
    const poly mask = poly_from_domain_ints({1, 0, 1, 0});
    const poly table = matrix_poly_from_rows({
        {10, 11, 12, 13},
        {20, 21, 22, 23},
        {30, 31, 32, 33},
        {40, 41, 42, 43},
    });
    const poly output = matrix_poly_from_rows({
        {20, 21, 22, 23},
        {0, 0, 0, 0},
        {10, 11, 12, 13},
        {0, 0, 0, 0},
    });

    index_routing.add_masked_check(virCommitment(kzg.commit(row_indices)),
                                   virCommitment(kzg.commit(mask)),
                                   virCommitment(kzg.commit(table)),
                                   virCommitment(kzg.commit(output)),
                                   4,
                                   4,
                                   4);

    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(IndexRoutingProtocolTest, MaskedRoutingRejectsNonZeroDisabledOutput)
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

    const poly row_indices = poly_from_domain_ints({1, 3, 0, 2});
    const poly mask = poly_from_domain_ints({1, 0, 1, 0});
    const poly table = matrix_poly_from_rows({
        {10, 11, 12, 13},
        {20, 21, 22, 23},
        {30, 31, 32, 33},
        {40, 41, 42, 43},
    });
    const poly wrong_output = matrix_poly_from_rows({
        {20, 21, 22, 23},
        {10, 11, 12, 13},
        {10, 11, 12, 13},
        {0, 0, 0, 0},
    });

    index_routing.add_masked_check(virCommitment(kzg.commit(row_indices)),
                                   virCommitment(kzg.commit(mask)),
                                   virCommitment(kzg.commit(table)),
                                   virCommitment(kzg.commit(wrong_output)),
                                   4,
                                   4,
                                   4);

    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(IndexRoutingProtocolTest, RejectsNonBooleanMaskEvenWhenMaskedRouteAlgebraMatches)
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

    const poly row_indices = poly_from_domain_ints({0, 3, 0, 2});
    const poly mask = poly_from_domain_ints({2, 0, 1, 0});
    const poly table = matrix_poly_from_rows({
        {10, 11},
        {20, 21},
        {30, 31},
        {40, 41},
    });
    const poly output = matrix_poly_from_rows({
        {20, 22},
        {0, 0},
        {10, 11},
        {0, 0},
    });

    index_routing.add_masked_check(virCommitment(kzg.commit(row_indices)),
                                   virCommitment(kzg.commit(mask)),
                                   virCommitment(kzg.commit(table)),
                                   virCommitment(kzg.commit(output)),
                                   4,
                                   4,
                                   2);

    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(IndexRoutingProtocolTest, MergedChecksStillRejectMismatchInEitherRoute)
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

    const virCommitment row_indices(kzg.commit(poly_from_domain_ints({1, 3, 0, 2})));
    const virCommitment first_table(kzg.commit(poly_from_domain_ints({10, 20, 30, 40})));
    const virCommitment first_output(kzg.commit(poly_from_domain_ints({20, 40, 10, 30})));
    const virCommitment second_table(kzg.commit(poly_from_domain_ints({1, 2, 3, 4})));
    const virCommitment wrong_second_output(kzg.commit(poly_from_domain_ints({2, 4, 99, 3})));

    index_routing.add_check(row_indices, first_table, first_output, 4, 4, 1);
    index_routing.add_check(row_indices, second_table, wrong_second_output, 4, 4, 1);

    EXPECT_TRUE(index_routing.finalize());
    EXPECT_TRUE(concatenated_table_lookup.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
