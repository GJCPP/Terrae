#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "kzg.hpp"
#include "polynomial.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>

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

} // namespace

class TestableTableLookupProtocol : public tableLookupProtocol {
public:
    using tableLookupProtocol::tableLookupProtocol;

    const tableLookupVerifier& verifier_ref() const { return verifier; }
};

class TableLookupProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(TableLookupProtocolTest, RejectsNonPowerOfTwoShapes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, sum);

    const virCommitment matrix(fr_from_int(0));
    EXPECT_THROW(protocol.add_check(matrix, matrix, 3, 4, 2), std::runtime_error);
    EXPECT_THROW(protocol.add_check(matrix, matrix, 2, 4, 3), std::runtime_error);
}

TEST_F(TableLookupProtocolTest, ValidMatrixLookupFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });

    tableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, sum);
    protocol.add_check(virCommitment(kzg.commit(query_poly)),
                       virCommitment(kzg.commit(table_poly)),
                       2,
                       4,
                       2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(TableLookupProtocolTest, InvalidMatrixLookupFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 29},
    });

    tableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, sum);
    protocol.add_check(virCommitment(kzg.commit(query_poly)),
                       virCommitment(kzg.commit(table_poly)),
                       2,
                       4,
                       2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(TableLookupProtocolTest, SupportsBatchedChecks)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    const poly table_poly = matrix_poly_from_rows({
        {2, 3},
        {5, 7},
        {11, 13},
        {17, 19},
    });

    tableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, sum);
    protocol.add_check(virCommitment(kzg.commit(matrix_poly_from_rows({{2, 3}, {11, 13}}))),
                       virCommitment(kzg.commit(table_poly)),
                       2,
                       4,
                       2);
    protocol.add_check(virCommitment(kzg.commit(matrix_poly_from_rows({{17, 19}}))),
                       virCommitment(kzg.commit(table_poly)),
                       1,
                       4,
                       2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(TableLookupProtocolTest, SupportsExplicitWitnessInput)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });

    const tableLookupWitness witness = tableLookupWitness::from_positions(4, {1, 3});

    tableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, sum);
    protocol.add_check(virCommitment(kzg.commit(query_poly)),
                       virCommitment(kzg.commit(table_poly)),
                       2,
                       4,
                       2,
                       witness);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(TableLookupProtocolTest, SupportsExplicitWitnessInputWithWiderRows)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5, 7, 11},
        {13, 17, 19, 23},
        {29, 31, 37, 41},
        {43, 47, 53, 59},
    });
    const poly query_poly = matrix_poly_from_rows({
        {13, 17, 19, 23},
        {43, 47, 53, 59},
    });

    const tableLookupWitness witness = tableLookupWitness::from_positions(4, {1, 3});

    tableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, sum);
    protocol.add_check(virCommitment(kzg.commit(query_poly)),
                       virCommitment(kzg.commit(table_poly)),
                       2,
                       4,
                       4,
                       witness);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(TableLookupProtocolTest, ProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(1024, 1024));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_reduced_lookup;
    prover_reduced_lookup.setup(prover_kzg, prover_zero, prover_sum);

    tableLookupProtocol prover_protocol(prover_reduced_lookup);
    prover_protocol.setup(prover_kzg, prover_zero, prover_sum);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });
    const kzgCommitment prover_query_commitment = prover_kzg.commit(query_poly);
    const kzgCommitment prover_table_commitment = prover_kzg.commit(table_poly);

    prover_protocol.add_check(virCommitment(prover_query_commitment),
                              virCommitment(prover_table_commitment),
                              2,
                              4,
                              2);

    EXPECT_TRUE(prover_protocol.finalize());
    EXPECT_TRUE(prover_reduced_lookup.finalize());
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

    lookupProtocol verifier_reduced_lookup;
    verifier_reduced_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    tableLookupProtocol verifier_protocol(verifier_reduced_lookup);
    verifier_protocol.setup(verifier_kzg, verifier_zero, verifier_sum);

    const kzgCommitment verifier_query_commitment = verifier_kzg.adopt_commitment(prover_query_commitment.com);
    const kzgCommitment verifier_table_commitment = verifier_kzg.adopt_commitment(prover_table_commitment.com);
    verifier_protocol.add_check(virCommitment(verifier_query_commitment),
                                virCommitment(verifier_table_commitment),
                                2,
                                4,
                                2);

    EXPECT_TRUE(verifier_protocol.finalize());
    EXPECT_TRUE(verifier_reduced_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(TableLookupProtocolTest, ChallengeComesFromTranscript)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    const virCommitment table_commitment(kzg.commit(matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    })));
    const virCommitment query_commitment(kzg.commit(matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    })));

    TestableTableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, sum);

    ProofTranscript expected = kzg.transcript();
    const Fr expected_rho = expected.squeeze_nonzero_fr("table_lookup.rho");

    protocol.add_check(query_commitment, table_commitment, 2, 4, 2);

    ASSERT_EQ(protocol.verifier_ref().checks.size(), 1u);
    EXPECT_EQ(protocol.verifier_ref().checks[0].rho, expected_rho);
}

TEST_F(TableLookupProtocolTest, ProofModeBindsRowIndexBeforeRho)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));
    auto prove_session = ProofSession::prover();
    kzg.set_proof_session(prove_session);

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    TestableTableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, sum);

    const virCommitment table_commitment(kzg.commit(matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    })));
    const virCommitment query_commitment(kzg.commit(matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    })));

    protocol.add_check(query_commitment, table_commitment, 2, 4, 2);

    ASSERT_EQ(protocol.verifier_ref().checks.size(), 1u);
    const tableLookupCheckRecord& record = protocol.verifier_ref().checks[0];

    ProofTranscript expected;
    expected.absorb_string("kzg.commitment", transcript_representation(table_commitment.com));
    expected.absorb_string("kzg.commitment", transcript_representation(query_commitment.com));
    expected.absorb_string("table_lookup.row_index", proof_encoding(record.row_index_commitment.com));
    expected.absorb_string("kzg.commitment", transcript_representation(record.row_index_commitment));
    expected.absorb_string("kzg.degree.commitment", transcript_representation(record.row_index_commitment));
    expected.absorb_size("kzg.degree.bound", 3);
    const Fr expected_rho = expected.squeeze_nonzero_fr("table_lookup.rho");
    EXPECT_EQ(record.rho, expected_rho);
}
