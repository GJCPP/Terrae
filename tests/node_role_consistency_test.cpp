#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "hist_agg.hpp"
#include "lookup.hpp"
#include "node_role_consistency.hpp"
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

class NodeRoleConsistencyProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(NodeRoleConsistencyProtocolTest, ValidMaskedNodeRolesFinalize)
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

    nodeRoleConsistencyProtocol protocol;
    protocol.setup(kzg, zero, hist_agg);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 3}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 1}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              2,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(NodeRoleConsistencyProtocolTest, RejectsChildReusedAcrossParents)
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

    nodeRoleConsistencyProtocol protocol;
    protocol.setup(kzg, zero, hist_agg);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 3}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 0, 1}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 1, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 2, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 1, 0}))),
                              2,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(NodeRoleConsistencyProtocolTest, RejectsRootAlsoUsedAsChild)
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

    nodeRoleConsistencyProtocol protocol;
    protocol.setup(kzg, zero, hist_agg);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 3}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 0, 1}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 1, 0}))),
                              2,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(NodeRoleConsistencyProtocolTest, RejectsMissingLeafCoverage)
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

    nodeRoleConsistencyProtocol protocol;
    protocol.setup(kzg, zero, hist_agg);

    protocol.add_masked_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 3}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({2, 0, 0, 0}))),
                              virCommitment(kzg.commit(poly_from_domain_ints({1, 0, 0, 0}))),
                              2,
                              4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(hist_agg.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
