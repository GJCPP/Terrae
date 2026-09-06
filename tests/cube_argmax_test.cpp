#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "cube_argmax.hpp"
#include "kzg.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
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

poly cube_poly_from_slices(const std::vector<std::vector<std::vector<int>>>& cube)
{
    std::vector<Fr> evals;
    for (const auto& outer : cube) {
        for (const auto& middle : outer) {
            for (int value : middle) {
                evals.push_back(fr_from_int(value));
            }
        }
    }
    return poly_from_evals(evals);
}

poly perturb_evaluation(const poly& p, size_t domain_size, size_t index, const Fr& delta)
{
    std::vector<Fr> evals = p.evaluations_on_domain(domain_size);
    Fr::add(evals[index], evals[index], delta);
    return poly_from_evals(evals);
}

} // namespace

class CubeMiddleArgmaxProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(CubeMiddleArgmaxProtocolTest, ValidProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol protocol;
    protocol.setup(kzg, zero, cube_sum, replication, less);

    const poly scores = cube_poly_from_slices({
        {
            {1, 4},
            {3, 2},
        },
        {
            {0, 5},
            {6, 1},
        },
    });
    const poly max_values = matrix_poly_from_rows({
        {3, 4},
        {6, 5},
    });
    const poly selector = cube_poly_from_slices({
        {
            {0, 1},
            {1, 0},
        },
        {
            {0, 1},
            {1, 0},
        },
    });

    protocol.add_check(virCommitment(kzg.commit(scores)),
                       virCommitment(kzg.commit(max_values)),
                       virCommitment(kzg.commit(selector)),
                       2,
                       2,
                       2,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeMiddleArgmaxProtocolTest, WrongSelectorBreaksProof)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, reduced_lookup);

    cubeSumProtocol cube_sum(table_lookup);
    cube_sum.setup(kzg, zero);

    cubeMiddleReplicationProtocol replication;
    replication.setup(kzg, zero, table_lookup);

    cubeMiddleArgmaxProtocol protocol;
    protocol.setup(kzg, zero, cube_sum, replication, less);

    const poly scores = cube_poly_from_slices({
        {
            {1, 4},
            {3, 2},
        },
        {
            {0, 5},
            {6, 1},
        },
    });
    const poly max_values = matrix_poly_from_rows({
        {3, 4},
        {6, 5},
    });
    poly selector = cube_poly_from_slices({
        {
            {0, 1},
            {1, 0},
        },
        {
            {0, 1},
            {1, 0},
        },
    });
    selector = perturb_evaluation(selector, 8, 0, fr_from_int(1));

    protocol.add_check(virCommitment(kzg.commit(scores)),
                       virCommitment(kzg.commit(max_values)),
                       virCommitment(kzg.commit(selector)),
                       2,
                       2,
                       2,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(replication.finalize());
    EXPECT_TRUE(cube_sum.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeMiddleArgmaxProtocolTest, ProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(2048, 2048));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_lookup;
    prover_lookup.setup(prover_kzg, prover_zero, prover_sum);

    tableLookupProtocol prover_table_lookup(prover_lookup);
    prover_table_lookup.setup(prover_kzg, prover_zero, prover_sum);

    lessCheckProtocol prover_less;
    prover_less.setup(prover_kzg, prover_lookup);

    cubeSumProtocol prover_cube_sum(prover_table_lookup);
    prover_cube_sum.setup(prover_kzg, prover_zero);

    cubeMiddleReplicationProtocol prover_replication;
    prover_replication.setup(prover_kzg, prover_zero, prover_table_lookup);

    cubeMiddleArgmaxProtocol prover_protocol;
    prover_protocol.setup(prover_kzg, prover_zero, prover_cube_sum, prover_replication, prover_less);

    const poly scores = cube_poly_from_slices({
        {
            {1, 4},
            {3, 2},
        },
        {
            {0, 5},
            {6, 1},
        },
    });
    const poly max_values = matrix_poly_from_rows({
        {3, 4},
        {6, 5},
    });
    const poly selector = cube_poly_from_slices({
        {
            {0, 1},
            {1, 0},
        },
        {
            {0, 1},
            {1, 0},
        },
    });
    const kzgCommitment prover_scores_commitment = prover_kzg.commit(scores);
    const kzgCommitment prover_max_commitment = prover_kzg.commit(max_values);
    const kzgCommitment prover_selector_commitment = prover_kzg.commit(selector);

    prover_protocol.add_check(virCommitment(prover_scores_commitment),
                              virCommitment(prover_max_commitment),
                              virCommitment(prover_selector_commitment),
                              2,
                              2,
                              2,
                              4);

    EXPECT_TRUE(prover_protocol.finalize());
    EXPECT_TRUE(prover_replication.finalize());
    EXPECT_TRUE(prover_cube_sum.finalize());
    EXPECT_TRUE(prover_less.finalize());
    EXPECT_TRUE(prover_table_lookup.finalize());
    EXPECT_TRUE(prover_lookup.finalize());
    EXPECT_TRUE(prover_sum.finalize());
    EXPECT_TRUE(prover_zero.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    zeroProtocol verifier_zero;
    verifier_zero.setup(verifier_kzg);

    sumProtocol verifier_sum;
    verifier_sum.setup(verifier_kzg, verifier_zero);

    lookupProtocol verifier_lookup;
    verifier_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    tableLookupProtocol verifier_table_lookup(verifier_lookup);
    verifier_table_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    lessCheckProtocol verifier_less;
    verifier_less.setup(verifier_kzg, verifier_lookup);

    cubeSumProtocol verifier_cube_sum(verifier_table_lookup);
    verifier_cube_sum.setup(verifier_kzg, verifier_zero);

    cubeMiddleReplicationProtocol verifier_replication;
    verifier_replication.setup(verifier_kzg, verifier_zero, verifier_table_lookup);

    cubeMiddleArgmaxProtocol verifier_protocol;
    verifier_protocol.setup(verifier_kzg, verifier_zero, verifier_cube_sum, verifier_replication, verifier_less);

    const kzgCommitment verifier_scores_commitment = verifier_kzg.adopt_commitment(prover_scores_commitment.com);
    const kzgCommitment verifier_max_commitment = verifier_kzg.adopt_commitment(prover_max_commitment.com);
    const kzgCommitment verifier_selector_commitment = verifier_kzg.adopt_commitment(prover_selector_commitment.com);

    verifier_protocol.add_check(virCommitment(verifier_scores_commitment),
                                virCommitment(verifier_max_commitment),
                                virCommitment(verifier_selector_commitment),
                                2,
                                2,
                                2,
                                4);

    EXPECT_TRUE(verifier_protocol.finalize());
    EXPECT_TRUE(verifier_replication.finalize());
    EXPECT_TRUE(verifier_cube_sum.finalize());
    EXPECT_TRUE(verifier_less.finalize());
    EXPECT_TRUE(verifier_table_lookup.finalize());
    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}
