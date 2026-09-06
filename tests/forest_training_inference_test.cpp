#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "box_check.hpp"
#include "concatenated_table_lookup.hpp"
#include "forest_training_inference.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
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

forestTrainingInferenceWitness valid_witness(kzgProtocol& kzg)
{
    forestTrainingInferenceWitness witness;
    witness.query_tree_ids = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 2, 2})));
    witness.lifted_inputs = virCommitment(kzg.commit(matrix_poly_from_rows({
        {1, 1},
        {3, 2},
        {1, 1},
        {3, 1},
    })));
    witness.leaf_indices = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 2, 3})));
    witness.leaf_lower_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {2, 0},
    })));
    witness.leaf_upper_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {4, 4},
        {4, 4},
        {2, 4},
        {4, 4},
    })));
    witness.tree_scores = virCommitment(kzg.commit(poly_from_domain_ints({5, 5, 7, 9})));

    witness.node_tree_ids = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 2, 2})));
    witness.node_lower_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {2, 0},
    })));
    witness.node_upper_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {4, 4},
        {4, 4},
        {2, 4},
        {4, 4},
    })));
    witness.node_leaf_mask = virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 1, 1})));
    witness.node_leaf_values = virCommitment(kzg.commit(poly_from_domain_ints({0, 5, 7, 9})));
    return witness;
}

} // namespace

class ForestTrainingInferenceProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTrainingInferenceProtocolTest, ValidWitnessForestInferenceFinalizes)
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

    boxCheckProtocol box_check;
    box_check.setup(kzg, less);

    forestTrainingInferenceProtocol protocol;
    protocol.setup(kzg, less, box_check, concatenated_lookup);

    protocol.add_check(valid_witness(kzg), 4, 4, 2, 3);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(box_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestTrainingInferenceProtocolTest, RejectsInternalRowClaimedAsLeaf)
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

    boxCheckProtocol box_check;
    box_check.setup(kzg, less);

    forestTrainingInferenceProtocol protocol;
    protocol.setup(kzg, less, box_check, concatenated_lookup);

    forestTrainingInferenceWitness witness = valid_witness(kzg);
    witness.leaf_indices = virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 2, 3})));
    witness.leaf_lower_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {2, 0},
    })));
    witness.leaf_upper_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {4, 4},
        {4, 4},
        {2, 4},
        {4, 4},
    })));
    witness.tree_scores = virCommitment(kzg.commit(poly_from_domain_ints({0, 5, 7, 9})));

    protocol.add_check(witness, 4, 4, 2, 3);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(box_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ForestTrainingInferenceProtocolTest, RejectsLeafIndexOutsideNodeTableWithoutSeparateRangeCheck)
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

    boxCheckProtocol box_check;
    box_check.setup(kzg, less);

    forestTrainingInferenceProtocol protocol;
    protocol.setup(kzg, less, box_check, concatenated_lookup);

    forestTrainingInferenceWitness witness = valid_witness(kzg);
    witness.leaf_indices = virCommitment(kzg.commit(poly_from_domain_ints({4, 1, 2, 3})));
    witness.node_leaf_mask = virCommitment(kzg.commit(poly_from_domain_ints({1, 1, 1, 1})));
    witness.leaf_lower_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {0, 0},
        {0, 0},
        {0, 0},
        {2, 0},
    })));
    witness.leaf_upper_bounds = virCommitment(kzg.commit(matrix_poly_from_rows({
        {4, 4},
        {4, 4},
        {2, 4},
        {4, 4},
    })));
    witness.tree_scores = virCommitment(kzg.commit(poly_from_domain_ints({0, 5, 7, 9})));

    protocol.add_check(witness, 4, 4, 2, 3);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(box_check.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(concatenated_lookup.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
