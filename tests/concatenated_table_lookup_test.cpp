#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "concatenated_table_lookup.hpp"
#include "lookup.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
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

} // namespace

class ConcatenatedTableLookupProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

class TestableTableLookupProtocol : public tableLookupProtocol {
public:
    using tableLookupProtocol::tableLookupProtocol;

    const tableLookupVerifier& verifier_ref() const { return verifier; }
};

TEST_F(ConcatenatedTableLookupProtocolTest, ValidConcatenatedLookupFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    TestableTableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    concatenatedTableLookupProtocol protocol;
    protocol.setup(kzg, table_lookup);

    const poly tree = matrix_poly_from_rows({
        {1},
        {2},
        {3},
        {4},
    });
    const poly left_box = matrix_poly_from_rows({
        {10, 11},
        {20, 21},
        {30, 31},
        {40, 41},
    });
    const poly right_box = matrix_poly_from_rows({
        {12, 13},
        {22, 23},
        {32, 33},
        {42, 43},
    });
    const poly score = matrix_poly_from_rows({
        {100},
        {200},
        {300},
        {400},
    });

    const poly query_tree = matrix_poly_from_rows({
        {2},
        {4},
    });
    const poly query_left_box = matrix_poly_from_rows({
        {20, 21},
        {40, 41},
    });
    const poly query_right_box = matrix_poly_from_rows({
        {22, 23},
        {42, 43},
    });
    const poly query_score = matrix_poly_from_rows({
        {200},
        {400},
    });

    protocol.add_check({concatenatedTableLookupBlock(
                            virCommitment(kzg.commit(query_tree)), virCommitment(kzg.commit(tree)), 1),
                        concatenatedTableLookupBlock(virCommitment(kzg.commit(query_left_box)),
                                                    virCommitment(kzg.commit(left_box)),
                                                    2),
                        concatenatedTableLookupBlock(virCommitment(kzg.commit(query_right_box)),
                                                    virCommitment(kzg.commit(right_box)),
                                                    2),
                        concatenatedTableLookupBlock(
                            virCommitment(kzg.commit(query_score)), virCommitment(kzg.commit(score)), 1)},
                       2,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_EQ(table_lookup.verifier_ref().checks.size(), 1u);
    EXPECT_EQ(table_lookup.verifier_ref().checks.front().column_count, 2u);
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ConcatenatedTableLookupProtocolTest, RejectsRowsThatDoNotComeFromOneSharedTableRow)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    TestableTableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    concatenatedTableLookupProtocol protocol;
    protocol.setup(kzg, table_lookup);

    const poly tree = matrix_poly_from_rows({
        {1},
        {2},
        {3},
        {4},
    });
    const poly left_box = matrix_poly_from_rows({
        {10, 11},
        {20, 21},
        {30, 31},
        {40, 41},
    });

    const poly inconsistent_tree_query = matrix_poly_from_rows({
        {2},
        {4},
    });
    const poly inconsistent_left_box_query = matrix_poly_from_rows({
        {10, 11},
        {40, 41},
    });

    protocol.add_check({concatenatedTableLookupBlock(virCommitment(kzg.commit(inconsistent_tree_query)),
                                                    virCommitment(kzg.commit(tree)),
                                                    1),
                        concatenatedTableLookupBlock(virCommitment(kzg.commit(inconsistent_left_box_query)),
                                                    virCommitment(kzg.commit(left_box)),
                                                    2)},
                       2,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_EQ(table_lookup.verifier_ref().checks.size(), 1u);
    EXPECT_EQ(table_lookup.verifier_ref().checks.front().column_count, 2u);
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ConcatenatedTableLookupProtocolTest, SupportsExplicitSharedWitness)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    TestableTableLookupProtocol table_lookup(lookup);
    table_lookup.setup(kzg, zero, sum);

    concatenatedTableLookupProtocol protocol;
    protocol.setup(kzg, table_lookup);

    const poly tree = matrix_poly_from_rows({
        {1},
        {2},
        {3},
        {4},
    });
    const poly score = matrix_poly_from_rows({
        {100},
        {200},
        {300},
        {400},
    });
    const poly query_tree = matrix_poly_from_rows({
        {2},
        {4},
    });
    const poly query_score = matrix_poly_from_rows({
        {200},
        {400},
    });

    const tableLookupWitness witness = tableLookupWitness::from_positions(4, {1, 3});
    protocol.add_check({concatenatedTableLookupBlock(
                            virCommitment(kzg.commit(query_tree)), virCommitment(kzg.commit(tree)), 1),
                        concatenatedTableLookupBlock(
                            virCommitment(kzg.commit(query_score)), virCommitment(kzg.commit(score)), 1)},
                       2,
                       4,
                       witness);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_EQ(table_lookup.verifier_ref().checks.size(), 1u);
    EXPECT_EQ(table_lookup.verifier_ref().checks.front().column_count, 1u);
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
