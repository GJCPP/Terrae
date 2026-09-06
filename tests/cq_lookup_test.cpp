#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "cq_lookup.hpp"
#include "kzg.hpp"
#include "polynomial.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
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

class TestableCqLookupProtocol : public cqLookupProtocol {
public:
    const cqLookupVerifier& verifier_ref() const { return verifier; }
};

class TestableZeroProtocolForCqLookup : public zeroProtocol {
public:
    const zeroVerifier& verifier_ref() const { return verifier; }
};

class CqLookupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(CqLookupTest, TableSetupRejectsDuplicateValues)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    cqLookupTable table;
    const poly t = poly_from_domain_ints({3, 5, 5, 11});

    EXPECT_THROW(table.setup(kzg, virCommitment(kzg.commit(t)), 4), std::runtime_error);
}

TEST_F(CqLookupTest, ValidLookupFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    const poly t = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    cqLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(t)), 8);

    cqLookupProtocol protocol;
    protocol.setup(kzg, table);

    const poly f = poly_from_domain_ints({5, 7, 7, 11});
    protocol.add_check(virCommitment(kzg.commit(f)), 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqLookupTest, InvalidQueryFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    const poly t = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    cqLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(t)), 8);

    cqLookupProtocol protocol;
    protocol.setup(kzg, table);

    const poly f = poly_from_domain_ints({5, 7, 29, 11});
    protocol.add_check(virCommitment(kzg.commit(f)), 4);

    EXPECT_FALSE(protocol.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqLookupTest, ProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(512, 512));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    const poly table_poly = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    const kzgCommitment prover_table_commitment = prover_kzg.commit(table_poly);
    cqLookupTable prover_table;
    prover_table.setup(prover_kzg, virCommitment(prover_table_commitment), 8);

    cqLookupProtocol prover_protocol;
    prover_protocol.setup(prover_kzg, prover_table);

    const poly query_poly = poly_from_domain_ints({5, 7, 7, 11});
    const kzgCommitment prover_query_commitment = prover_kzg.commit(query_poly);
    prover_protocol.add_check(virCommitment(prover_query_commitment), 4);

    EXPECT_TRUE(prover_protocol.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    const kzgCommitment verifier_table_commitment = verifier_kzg.commit(table_poly);
    cqLookupTable verifier_table;
    verifier_table.setup(verifier_kzg, virCommitment(verifier_table_commitment), 8);

    cqLookupProtocol verifier_protocol;
    verifier_protocol.setup(verifier_kzg, verifier_table);

    const kzgCommitment verifier_query_commitment = verifier_kzg.adopt_commitment(prover_query_commitment.com);
    verifier_protocol.add_check(virCommitment(verifier_query_commitment), 4);

    EXPECT_TRUE(verifier_protocol.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(CqLookupTest, SupportsBatchedChecks)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    const poly t = poly_from_domain_ints({2, 3, 5, 7, 11, 13, 17, 19});
    cqLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(t)), 8);

    cqLookupProtocol protocol;
    protocol.setup(kzg, table);

    protocol.add_check(virCommitment(kzg.commit(poly_from_domain_ints({2, 3}))), 2);
    protocol.add_check(virCommitment(kzg.commit(poly_from_domain_ints({11, 13, 17, 19}))), 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqLookupTest, SupportsQueryDomainLargerThanTableDomain)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    const poly t = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    cqLookupTable table;
    table.setup(kzg, virCommitment(kzg.commit(t)), 8);

    cqLookupProtocol protocol;
    protocol.setup(kzg, table);

    const poly f = poly_from_domain_ints({5, 7, 7, 11, 13, 17, 19, 23, 5, 7, 7, 11, 13, 17, 19, 23});
    protocol.add_check(virCommitment(kzg.commit(f)), 16);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CqLookupTest, ChallengesComeFromTranscript)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    const virCommitment table_commitment(kzg.commit(poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23})));
    cqLookupTable table;
    table.setup(kzg, table_commitment, 8);

    TestableCqLookupProtocol protocol;
    protocol.setup(kzg, table);

    const virCommitment query_commitment(kzg.commit(poly_from_domain_ints({5, 7, 7, 11})));
    ProofTranscript expected = kzg.transcript();

    protocol.add_check(query_commitment, 4);
    ASSERT_TRUE(protocol.finalize());

    ASSERT_EQ(protocol.verifier_ref().checks.size(), 1u);
    const cqLookupCheckRecord& record = protocol.verifier_ref().checks[0];

    const Fr expected_beta = expected.squeeze_fr("cq_lookup.beta");
    expected.absorb_string("kzg.commitment", transcript_representation(record.b0_commitment));
    expected.absorb_string("kzg.degree.commitment", transcript_representation(record.b0_commitment));
    expected.absorb_size("kzg.degree.bound", 2);
    expected.absorb_string("kzg.commitment", transcript_representation(record.qb_commitment));
    const Fr expected_gamma = expected.squeeze_fr("cq_lookup.gamma");

    EXPECT_EQ(record.beta, expected_beta);
    EXPECT_EQ(record.gamma, expected_gamma);
}

TEST_F(CqLookupTest, ProofModeBindsMultiplicityBeforeBetaAndBetaWitnessesBeforeGamma)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));
    auto prove_session = ProofSession::prover();
    kzg.set_proof_session(prove_session);

    const virCommitment table_commitment(kzg.commit(poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23})));
    cqLookupTable table;
    table.setup(kzg, table_commitment, 8);

    TestableCqLookupProtocol protocol;
    protocol.setup(kzg, table);

    const virCommitment query_commitment(kzg.commit(poly_from_domain_ints({5, 7, 7, 11})));

    protocol.add_check(query_commitment, 4);
    ASSERT_TRUE(protocol.finalize());

    ASSERT_EQ(protocol.verifier_ref().checks.size(), 1u);
    const cqLookupCheckRecord& record = protocol.verifier_ref().checks[0];

    ProofTranscript expected;
    expected.absorb_string("kzg.commitment", transcript_representation(table_commitment.com));
    expected.absorb_string("kzg.commitment", transcript_representation(query_commitment.com));
    expected.absorb_string("cq_lookup.multiplicity", proof_encoding(record.multiplicity_commitment));
    const Fr expected_beta = expected.squeeze_fr("cq_lookup.beta");
    expected.absorb_string("cq_lookup.b0", proof_encoding(record.b0_commitment.com));
    expected.absorb_string("kzg.commitment", transcript_representation(record.b0_commitment));
    expected.absorb_string("kzg.degree.commitment", transcript_representation(record.b0_commitment));
    expected.absorb_size("kzg.degree.bound", 2);
    expected.absorb_string("cq_lookup.qb", proof_encoding(record.qb_commitment.com));
    expected.absorb_string("kzg.commitment", transcript_representation(record.qb_commitment));
    expected.absorb_string("cq_lookup.support_valid", proof_encoding(record.support_valid));
    expected.absorb_string("cq_lookup.denominators_valid", proof_encoding(record.denominators_valid));
    expected.absorb_string("cq_lookup.quotient_valid", proof_encoding(record.quotient_valid));
    expected.absorb_string("cq_lookup.a", proof_encoding(record.a_commitment));
    expected.absorb_string("cq_lookup.qa", proof_encoding(record.qa_commitment));
    expected.absorb_string("cq_lookup.a0_quotient", proof_encoding(record.a0_quotient_commitment));
    expected.absorb_string("cq_lookup.a0", proof_encoding(record.a0));
    const Fr expected_gamma = expected.squeeze_fr("cq_lookup.gamma");

    EXPECT_EQ(record.beta, expected_beta);
    EXPECT_EQ(record.gamma, expected_gamma);
}

TEST_F(CqLookupTest, BatchedChecksShareBetaAndGammaAfterFinalize)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    const virCommitment table_commitment(kzg.commit(poly_from_domain_ints({2, 3, 5, 7, 11, 13, 17, 19})));
    cqLookupTable table;
    table.setup(kzg, table_commitment, 8);

    TestableCqLookupProtocol protocol;
    protocol.setup(kzg, table);

    protocol.add_check(virCommitment(kzg.commit(poly_from_domain_ints({2, 3}))), 2);
    protocol.add_check(virCommitment(kzg.commit(poly_from_domain_ints({11, 13, 17, 19}))), 4);

    ASSERT_TRUE(protocol.finalize());
    ASSERT_EQ(protocol.verifier_ref().checks.size(), 2u);
    EXPECT_EQ(protocol.verifier_ref().checks[0].beta, protocol.verifier_ref().checks[1].beta);
    EXPECT_EQ(protocol.verifier_ref().checks[0].gamma, protocol.verifier_ref().checks[1].gamma);
}

TEST_F(CqLookupTest, PreparedOpeningCanBeSharedWithZeroCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    const virCommitment table_commitment(kzg.commit(poly_from_domain_ints({2, 3, 5, 7, 11, 13, 17, 19})));
    cqLookupTable table;
    table.setup(kzg, table_commitment, 8);

    TestableCqLookupProtocol lookup_protocol;
    lookup_protocol.setup(kzg, table);
    lookup_protocol.add_check(virCommitment(kzg.commit(poly_from_domain_ints({2, 3, 5, 7}))), 4);

    TestableZeroProtocolForCqLookup zero_protocol;
    zero_protocol.setup(kzg);
    const poly zero_poly = poly_from_domain_ints({1, 2, 3}) * vanishing_poly(4);
    virCommitment zero_statement(kzg.commit(zero_poly));
    zero_protocol.add_check(zero_statement, 4);

    ASSERT_TRUE(lookup_protocol.prepare_opening());
    ASSERT_TRUE(zero_protocol.prepare_opening());

    const Fr shared_opening = kzg.shared_opening_challenge("protocol.opening.stage0", true);
    ASSERT_TRUE(lookup_protocol.complete_opening(shared_opening));
    ASSERT_TRUE(zero_protocol.complete_opening(shared_opening));

    ASSERT_EQ(lookup_protocol.verifier_ref().checks.size(), 1u);
    EXPECT_EQ(lookup_protocol.verifier_ref().checks[0].gamma, shared_opening);
    EXPECT_EQ(zero_protocol.verifier_ref().x, shared_opening);
    EXPECT_TRUE(kzg.finalize());
}
