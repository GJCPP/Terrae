#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "kzg.hpp"
#include "polynomial.hpp"
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

class TestableCqTableLookupProtocol : public cqTableLookupProtocol {
public:
    using cqTableLookupProtocol::cqTableLookupProtocol;

    const cqTableLookupVerifier& verifier_ref() const { return verifier; }
};

class CqTableLookupProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(CqTableLookupProtocolTest, RejectsNonPowerOfTwoShapes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    cqTableLookupTable table;
    const poly table_poly = matrix_poly_from_rows({
        {1, 2},
        {3, 4},
    });

    EXPECT_THROW(table.setup(kzg, virCommitment(kzg.commit(table_poly)), 3, 2), std::runtime_error);
    EXPECT_THROW(table.setup(kzg, virCommitment(kzg.commit(table_poly)), 2, 3), std::runtime_error);
}

TEST_F(CqTableLookupProtocolTest, ValidMatrixLookupFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    cqTableLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(table_poly)), 4, 2);

    cqLookupProtocol reduced_lookup;
    cqTableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, table);

    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });
    const virCommitment query_commitment(kzg.commit(query_poly));
    protocol.add_check(query_commitment, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqTableLookupProtocolTest, InvalidMatrixLookupFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    cqTableLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(table_poly)), 4, 2);

    cqLookupProtocol reduced_lookup;
    cqTableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, table);

    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 29},
    });
    protocol.add_check(virCommitment(kzg.commit(query_poly)), 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqTableLookupProtocolTest, ProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(2048, 2048));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    const kzgCommitment prover_table_commitment = prover_kzg.commit(table_poly);
    cqTableLookupTable prover_table;
    prover_table.setup(prover_kzg, virCommitment(prover_table_commitment), 4, 2);

    cqLookupProtocol prover_reduced_lookup;
    cqTableLookupProtocol prover_protocol(prover_reduced_lookup);
    prover_protocol.setup(prover_kzg, prover_zero, prover_table);

    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });
    const kzgCommitment prover_query_commitment = prover_kzg.commit(query_poly);
    prover_protocol.add_check(virCommitment(prover_query_commitment), 2);

    EXPECT_TRUE(prover_protocol.finalize());
    EXPECT_TRUE(prover_reduced_lookup.finalize());
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

    const kzgCommitment verifier_table_commitment = verifier_kzg.commit(table_poly);
    cqTableLookupTable verifier_table;
    verifier_table.setup(verifier_kzg, virCommitment(verifier_table_commitment), 4, 2);

    cqLookupProtocol verifier_reduced_lookup;
    cqTableLookupProtocol verifier_protocol(verifier_reduced_lookup);
    verifier_protocol.setup(verifier_kzg, verifier_zero, verifier_table);

    const kzgCommitment verifier_query_commitment = verifier_kzg.commit(query_poly);
    verifier_protocol.add_check(virCommitment(verifier_query_commitment), 2);

    EXPECT_TRUE(verifier_protocol.finalize());
    EXPECT_TRUE(verifier_reduced_lookup.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(CqTableLookupProtocolTest, SupportsBatchedChecks)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly table_poly = matrix_poly_from_rows({
        {2, 3},
        {5, 7},
        {11, 13},
        {17, 19},
    });
    cqTableLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(table_poly)), 4, 2);

    cqLookupProtocol reduced_lookup;
    cqTableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, table);
    protocol.add_check(virCommitment(kzg.commit(matrix_poly_from_rows({{2, 3}, {11, 13}}))), 2);
    protocol.add_check(virCommitment(kzg.commit(matrix_poly_from_rows({{17, 19}}))), 1);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqTableLookupProtocolTest, SupportsMoreQueryRowsThanTableRows)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly table_poly = matrix_poly_from_rows({
        {2, 3},
        {5, 7},
        {11, 13},
        {17, 19},
    });
    cqTableLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(table_poly)), 4, 2);

    cqLookupProtocol reduced_lookup;
    cqTableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, table);
    protocol.add_check(virCommitment(kzg.commit(matrix_poly_from_rows({
                           {2, 3},
                           {5, 7},
                           {11, 13},
                           {17, 19},
                           {2, 3},
                           {5, 7},
                           {11, 13},
                           {17, 19},
                       }))),
                       8);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqTableLookupProtocolTest, SupportsExplicitWitnessInput)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    cqTableLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(table_poly)), 4, 2);

    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });
    const tableLookupWitness witness = tableLookupWitness::from_positions(4, {1, 3});

    cqLookupProtocol reduced_lookup;
    cqTableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, table);
    protocol.add_check(virCommitment(kzg.commit(query_poly)), 2, witness);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqTableLookupProtocolTest, ProofModeBindsRowIndexBeforeRho)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));
    auto prove_session = ProofSession::prover();
    kzg.set_proof_session(prove_session);

    zeroProtocol zero;
    zero.setup(kzg);

    const poly table_poly = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    cqTableLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(table_poly)), 4, 2);

    cqLookupProtocol reduced_lookup;
    TestableCqTableLookupProtocol protocol(reduced_lookup);
    protocol.setup(kzg, zero, table);

    const poly query_poly = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });
    const virCommitment query_commitment(kzg.commit(query_poly));
    protocol.add_check(query_commitment, 2);

    ASSERT_EQ(protocol.verifier_ref().checks.size(), 1u);
    const cqTableLookupCheckRecord& record = protocol.verifier_ref().checks[0];

    ProofTranscript expected;
    expected.absorb_string("kzg.commitment", transcript_representation(table.table_commitment.com));
    expected.absorb_string("kzg.commitment", transcript_representation(query_commitment.com));
    expected.absorb_string("cq_table_lookup.row_index", proof_encoding(record.row_index_commitment.com));
    expected.absorb_string("kzg.commitment", transcript_representation(record.row_index_commitment));
    expected.absorb_string("kzg.degree.commitment", transcript_representation(record.row_index_commitment));
    expected.absorb_size("kzg.degree.bound", 3);
    const Fr expected_rho = expected.squeeze_nonzero_fr("cq_table_lookup.rho");
    EXPECT_EQ(record.rho, expected_rho);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
