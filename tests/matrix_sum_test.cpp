#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "kzg.hpp"
#include "matrix_sum.hpp"
#include "polynomial.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
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

std::vector<Fr> evaluations(const poly& p, size_t domain_size)
{
    return p.evaluations_on_domain(domain_size);
}

poly random_matrix_poly(std::mt19937& rng, size_t row_count, size_t column_count, int coeff_abs_max)
{
    std::uniform_int_distribution<int> coeff_dist(-coeff_abs_max, coeff_abs_max);

    std::vector<Fr> evals(row_count * column_count, Fr(0));
    for (Fr& value : evals) {
        value = fr_from_int(coeff_dist(rng));
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

class MatrixSumProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(MatrixSumProtocolTest, HelperPolynomialsMatchExpectedRowsAndColumns)
{
    const poly matrix = matrix_poly_from_rows({
        {2, 4},
        {6, 8},
    });

    const poly column_sum = matrix_column_sum_polynomial(matrix, 2, 2);
    const poly column_prefix = matrix_column_prefix_sum_polynomial(matrix, 2, 2);
    const poly row_sum = matrix_row_sum_polynomial(matrix, 2, 2);
    const poly row_prefix = matrix_row_prefix_sum_polynomial(matrix, 2, 2);

    const std::vector<Fr> expected_column_sum = {fr_from_int(8), fr_from_int(12)};
    const std::vector<Fr> expected_column_prefix = {
        fr_from_int(2),
        fr_from_int(4),
        fr_from_int(8),
        fr_from_int(12),
    };
    const std::vector<Fr> expected_row_sum = {fr_from_int(6), fr_from_int(14)};
    const std::vector<Fr> expected_row_prefix = {
        fr_from_int(2),
        fr_from_int(6),
        fr_from_int(6),
        fr_from_int(14),
    };

    EXPECT_EQ(evaluations(column_sum, 2), expected_column_sum);
    EXPECT_EQ(evaluations(column_prefix, 4), expected_column_prefix);
    EXPECT_EQ(evaluations(row_sum, 2), expected_row_sum);
    EXPECT_EQ(evaluations(row_prefix, 4), expected_row_prefix);
}

TEST_F(MatrixSumProtocolTest, RejectsNonPowerOfTwoShapes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    matrixColumnPrefixSumProtocol column_prefix_protocol;
    column_prefix_protocol.setup(kzg, zero, column_replication);

    matrixRowPrefixSumProtocol row_prefix_protocol;
    row_prefix_protocol.setup(kzg, zero);

    matrixColumnSumProtocol column_protocol;
    column_protocol.setup(kzg, zero);

    matrixRowSumProtocol row_protocol;
    row_protocol.setup(kzg, zero);

    const virCommitment matrix(fr_from_int(0));
    EXPECT_THROW(column_prefix_protocol.add_check(matrix, matrix, 3, 2), std::runtime_error);
    EXPECT_THROW(column_prefix_protocol.add_check(matrix, matrix, 2, 3), std::runtime_error);
    EXPECT_THROW(row_prefix_protocol.add_check(matrix, matrix, 3, 2), std::runtime_error);
    EXPECT_THROW(row_prefix_protocol.add_check(matrix, matrix, 2, 3), std::runtime_error);
    EXPECT_THROW(column_protocol.add_check(matrix, matrix, 3, 2), std::runtime_error);
    EXPECT_THROW(column_protocol.add_check(matrix, matrix, 2, 3), std::runtime_error);
    EXPECT_THROW(row_protocol.add_check(matrix, matrix, 3, 2), std::runtime_error);
    EXPECT_THROW(row_protocol.add_check(matrix, matrix, 2, 3), std::runtime_error);
}

TEST_F(MatrixSumProtocolTest, ValidColumnPrefixSumProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    const poly matrix = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    });
    const poly prefix = matrix_column_prefix_sum_polynomial(matrix, 4, 2);

    matrixColumnPrefixSumProtocol protocol;
    protocol.setup(kzg, zero, column_replication);
    protocol.add_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(prefix)), 4, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, InvalidColumnPrefixSumProofFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    const poly matrix = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    });
    const poly wrong_prefix = poly_from_evals({
        fr_from_int(1),
        fr_from_int(2),
        fr_from_int(4),
        fr_from_int(7),
        fr_from_int(9),
        fr_from_int(12),
        fr_from_int(16),
        fr_from_int(20),
    });

    matrixColumnPrefixSumProtocol protocol;
    protocol.setup(kzg, zero, column_replication);
    protocol.add_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(wrong_prefix)), 4, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, ValidRowPrefixSumProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    const poly prefix = matrix_row_prefix_sum_polynomial(matrix, 2, 4);

    matrixRowPrefixSumProtocol protocol;
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(prefix)), 2, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, InvalidRowPrefixSumProofFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    const poly wrong_prefix = poly_from_evals({
        fr_from_int(1),
        fr_from_int(3),
        fr_from_int(6),
        fr_from_int(11),
        fr_from_int(5),
        fr_from_int(11),
        fr_from_int(19),
        fr_from_int(26),
    });

    matrixRowPrefixSumProtocol protocol;
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(wrong_prefix)), 2, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, ValidColumnSumProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly matrix = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    });
    const poly column_sum = matrix_column_sum_polynomial(matrix, 4, 2);

    matrixColumnSumProtocol protocol;
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(column_sum)), 4, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, InvalidColumnSumProofFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly matrix = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    });
    const poly wrong_column_sum = poly_from_evals({
        fr_from_int(15),
        fr_from_int(20),
    });

    matrixColumnSumProtocol protocol;
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(wrong_column_sum)), 4, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, ColumnSumProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(1024, 1024));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    const poly matrix = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    });
    const poly column_sum = matrix_column_sum_polynomial(matrix, 4, 2);
    const kzgCommitment prover_matrix_commitment = prover_kzg.commit(matrix);
    const kzgCommitment prover_sum_commitment = prover_kzg.commit(column_sum);

    matrixColumnSumProtocol prover_protocol;
    prover_protocol.setup(prover_kzg, prover_zero);
    prover_protocol.add_check(virCommitment(prover_matrix_commitment), virCommitment(prover_sum_commitment), 4, 2);

    EXPECT_TRUE(prover_protocol.finalize());
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

    matrixColumnSumProtocol verifier_protocol;
    verifier_protocol.setup(verifier_kzg, verifier_zero);

    const kzgCommitment verifier_matrix_commitment = verifier_kzg.adopt_commitment(prover_matrix_commitment.com);
    const kzgCommitment verifier_sum_commitment = verifier_kzg.adopt_commitment(prover_sum_commitment.com);
    verifier_protocol.add_check(
        virCommitment(verifier_matrix_commitment), virCommitment(verifier_sum_commitment), 4, 2);

    EXPECT_TRUE(verifier_protocol.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(MatrixSumProtocolTest, ValidRowSumProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    const poly row_sum = matrix_row_sum_polynomial(matrix, 2, 4);

    matrixRowSumProtocol protocol;
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(row_sum)), 2, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, InvalidRowSumProofFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    const poly wrong_row_sum = poly_from_evals({
        fr_from_int(11),
        fr_from_int(26),
    });

    matrixRowSumProtocol protocol;
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(matrix)), virCommitment(kzg.commit(wrong_row_sum)), 2, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, RowSumProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(1024, 1024));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    const poly row_sum = matrix_row_sum_polynomial(matrix, 2, 4);
    const kzgCommitment prover_matrix_commitment = prover_kzg.commit(matrix);
    const kzgCommitment prover_sum_commitment = prover_kzg.commit(row_sum);

    matrixRowSumProtocol prover_protocol;
    prover_protocol.setup(prover_kzg, prover_zero);
    prover_protocol.add_check(virCommitment(prover_matrix_commitment), virCommitment(prover_sum_commitment), 2, 4);

    EXPECT_TRUE(prover_protocol.finalize());
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

    matrixRowSumProtocol verifier_protocol;
    verifier_protocol.setup(verifier_kzg, verifier_zero);

    const kzgCommitment verifier_matrix_commitment = verifier_kzg.adopt_commitment(prover_matrix_commitment.com);
    const kzgCommitment verifier_sum_commitment = verifier_kzg.adopt_commitment(prover_sum_commitment.com);
    verifier_protocol.add_check(
        virCommitment(verifier_matrix_commitment), virCommitment(verifier_sum_commitment), 2, 4);

    EXPECT_TRUE(verifier_protocol.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(MatrixSumProtocolTest, SupportsBatchedPrefixAndSumChecks)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    columnReplicationProtocol column_replication;
    column_replication.setup(kzg, zero);

    matrixColumnPrefixSumProtocol column_prefix_protocol;
    column_prefix_protocol.setup(kzg, zero, column_replication);

    matrixRowPrefixSumProtocol row_prefix_protocol;
    row_prefix_protocol.setup(kzg, zero);

    matrixColumnSumProtocol column_protocol;
    column_protocol.setup(kzg, zero);

    matrixRowSumProtocol row_protocol;
    row_protocol.setup(kzg, zero);

    const poly matrix_a = matrix_poly_from_rows({
        {2, 3},
        {5, 7},
        {11, 13},
        {17, 19},
    });
    const poly matrix_b = matrix_poly_from_rows({
        {23, 29, 31, 37},
        {41, 43, 47, 53},
    });

    column_prefix_protocol.add_check(
        virCommitment(kzg.commit(matrix_a)),
        virCommitment(kzg.commit(matrix_column_prefix_sum_polynomial(matrix_a, 4, 2))),
        4,
        2);
    row_prefix_protocol.add_check(
        virCommitment(kzg.commit(matrix_b)),
        virCommitment(kzg.commit(matrix_row_prefix_sum_polynomial(matrix_b, 2, 4))),
        2,
        4);
    column_protocol.add_check(
        virCommitment(kzg.commit(matrix_a)),
        virCommitment(kzg.commit(matrix_column_sum_polynomial(matrix_a, 4, 2))),
        4,
        2);
    row_protocol.add_check(
        virCommitment(kzg.commit(matrix_b)),
        virCommitment(kzg.commit(matrix_row_sum_polynomial(matrix_b, 2, 4))),
        2,
        4);

    EXPECT_TRUE(column_prefix_protocol.finalize());
    EXPECT_TRUE(row_prefix_protocol.finalize());
    EXPECT_TRUE(column_protocol.finalize());
    EXPECT_TRUE(row_protocol.finalize());
    EXPECT_TRUE(column_replication.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(MatrixSumProtocolTest, RandomizedValidProofsPass)
{
    std::mt19937 rng(20260331);
    const std::vector<std::pair<size_t, size_t>> shapes = {
        {2, 2},
        {4, 2},
        {2, 4},
        {4, 4},
        {8, 2},
    };
    std::uniform_int_distribution<int> shape_dist(0, static_cast<int>(shapes.size()) - 1);
    std::uniform_int_distribution<int> checks_dist(2, 5);

    for (int round = 0; round < 16; ++round) {
        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(4096, 4096));

        zeroProtocol zero;
        zero.setup(kzg);

        columnReplicationProtocol column_replication;
        column_replication.setup(kzg, zero);

        matrixColumnPrefixSumProtocol column_prefix_protocol;
        column_prefix_protocol.setup(kzg, zero, column_replication);

        matrixRowPrefixSumProtocol row_prefix_protocol;
        row_prefix_protocol.setup(kzg, zero);

        matrixColumnSumProtocol column_protocol;
        column_protocol.setup(kzg, zero);

        matrixRowSumProtocol row_protocol;
        row_protocol.setup(kzg, zero);

        const int checks = checks_dist(rng);
        for (int i = 0; i < checks; ++i) {
            const auto [row_count, column_count] = shapes[static_cast<size_t>(shape_dist(rng))];
            const poly matrix = random_matrix_poly(rng, row_count, column_count, 30);
            const poly column_prefix = matrix_column_prefix_sum_polynomial(matrix, row_count, column_count);
            const poly row_prefix = matrix_row_prefix_sum_polynomial(matrix, row_count, column_count);
            const poly column_sum = matrix_column_sum_polynomial(matrix, row_count, column_count);
            const poly row_sum = matrix_row_sum_polynomial(matrix, row_count, column_count);

            const virCommitment matrix_com(kzg.commit(matrix));
            column_prefix_protocol.add_check(
                matrix_com, virCommitment(kzg.commit(column_prefix)), row_count, column_count);
            row_prefix_protocol.add_check(
                matrix_com, virCommitment(kzg.commit(row_prefix)), row_count, column_count);
            column_protocol.add_check(
                matrix_com, virCommitment(kzg.commit(column_sum)), row_count, column_count);
            row_protocol.add_check(
                matrix_com, virCommitment(kzg.commit(row_sum)), row_count, column_count);
        }

        EXPECT_TRUE(column_prefix_protocol.finalize()) << "round=" << round;
        EXPECT_TRUE(row_prefix_protocol.finalize()) << "round=" << round;
        EXPECT_TRUE(column_protocol.finalize()) << "round=" << round;
        EXPECT_TRUE(row_protocol.finalize()) << "round=" << round;
        EXPECT_TRUE(column_replication.finalize()) << "round=" << round;
        EXPECT_TRUE(zero.finalize()) << "round=" << round;
        EXPECT_TRUE(kzg.finalize()) << "round=" << round;
    }
}

TEST_F(MatrixSumProtocolTest, RandomizedInvalidClaimsFail)
{
    std::mt19937 rng(20260401);
    const std::vector<std::pair<size_t, size_t>> shapes = {
        {2, 2},
        {4, 2},
        {2, 4},
        {4, 4},
    };
    std::uniform_int_distribution<int> shape_dist(0, static_cast<int>(shapes.size()) - 1);
    std::uniform_int_distribution<int> protocol_dist(0, 3);
    std::uniform_int_distribution<int> delta_dist(1, 9);

    for (int round = 0; round < 16; ++round) {
        const auto [row_count, column_count] = shapes[static_cast<size_t>(shape_dist(rng))];
        const size_t matrix_domain_size = row_count * column_count;

        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(4096, 4096));

        zeroProtocol zero;
        zero.setup(kzg);

        columnReplicationProtocol column_replication;
        column_replication.setup(kzg, zero);

        const poly matrix = random_matrix_poly(rng, row_count, column_count, 25);
        const virCommitment matrix_com(kzg.commit(matrix));
        const Fr delta = fr_from_int(delta_dist(rng));
        const int protocol_kind = protocol_dist(rng);

        if (protocol_kind == 0) {
            matrixColumnPrefixSumProtocol protocol;
            protocol.setup(kzg, zero, column_replication);
            std::uniform_int_distribution<size_t> index_dist(0, matrix_domain_size - 1);
            const poly wrong_claim = perturb_evaluation(
                matrix_column_prefix_sum_polynomial(matrix, row_count, column_count),
                matrix_domain_size,
                index_dist(rng),
                delta);
            protocol.add_check(matrix_com, virCommitment(kzg.commit(wrong_claim)), row_count, column_count);
            EXPECT_TRUE(protocol.finalize()) << "round=" << round;
        } else if (protocol_kind == 1) {
            matrixRowPrefixSumProtocol protocol;
            protocol.setup(kzg, zero);
            std::uniform_int_distribution<size_t> index_dist(0, matrix_domain_size - 1);
            const poly wrong_claim = perturb_evaluation(
                matrix_row_prefix_sum_polynomial(matrix, row_count, column_count),
                matrix_domain_size,
                index_dist(rng),
                delta);
            protocol.add_check(matrix_com, virCommitment(kzg.commit(wrong_claim)), row_count, column_count);
            EXPECT_TRUE(protocol.finalize()) << "round=" << round;
        } else if (protocol_kind == 2) {
            matrixColumnSumProtocol protocol;
            protocol.setup(kzg, zero);
            std::uniform_int_distribution<size_t> index_dist(0, column_count - 1);
            const poly wrong_claim = perturb_evaluation(
                matrix_column_sum_polynomial(matrix, row_count, column_count),
                column_count,
                index_dist(rng),
                delta);
            protocol.add_check(matrix_com, virCommitment(kzg.commit(wrong_claim)), row_count, column_count);
            EXPECT_TRUE(protocol.finalize()) << "round=" << round;
        } else {
            matrixRowSumProtocol protocol;
            protocol.setup(kzg, zero);
            std::uniform_int_distribution<size_t> index_dist(0, row_count - 1);
            const poly wrong_claim = perturb_evaluation(
                matrix_row_sum_polynomial(matrix, row_count, column_count),
                row_count,
                index_dist(rng),
                delta);
            protocol.add_check(matrix_com, virCommitment(kzg.commit(wrong_claim)), row_count, column_count);
            EXPECT_TRUE(protocol.finalize()) << "round=" << round;
        }

        EXPECT_TRUE(column_replication.finalize()) << "round=" << round;
        EXPECT_FALSE(zero.finalize()) << "round=" << round;
        EXPECT_TRUE(kzg.finalize()) << "round=" << round;
    }
}
