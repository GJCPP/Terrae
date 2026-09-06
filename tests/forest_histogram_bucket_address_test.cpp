#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "column_replication.hpp"
#include "forest_histogram_bucket_address.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "sum_check.hpp"
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

class ForestHistogramBucketAddressProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestHistogramBucketAddressProtocolTest, ValidBucketAddressesFinalize)
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

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    forestHistogramBucketAddressProtocol protocol;
    protocol.setup(zero, column_replication, less);

    forestHistogramBucketAddressWitness witness;
    witness.leaf_indices = virCommitment(kzg.commit(poly_from_domain_ints({1, 2, 0, 3})));
    witness.lifted_leaf_indices = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 1},
        {2, 2},
        {0, 0},
        {3, 3},
    })));
    witness.lifted_inputs = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 3},
        {1, 2},
        {2, 0},
        {3, 1},
    })));
    witness.flattened_bucket_indices = virCommitment(kzg.commit(matrix_poly_from_rows({
        {8, 15},
        {17, 22},
        {2, 4},
        {27, 29},
    })));

    protocol.add_check(witness, 4, 2, 4, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestHistogramBucketAddressProtocolTest, WrongFlattenedBucketAddressFailsZeroCheck)
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

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    forestHistogramBucketAddressProtocol protocol;
    protocol.setup(zero, column_replication, less);

    forestHistogramBucketAddressWitness witness;
    witness.leaf_indices = virCommitment(kzg.commit(poly_from_domain_ints({1, 2, 0, 3})));
    witness.lifted_leaf_indices = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 1},
        {2, 2},
        {0, 0},
        {3, 3},
    })));
    witness.lifted_inputs = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 3},
        {1, 2},
        {2, 0},
        {3, 1},
    })));
    witness.flattened_bucket_indices = virCommitment(kzg.commit(matrix_poly_from_rows({
        {8, 15},
        {17, 21},
        {2, 4},
        {27, 29},
    })));

    protocol.add_check(witness, 4, 2, 4, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
