#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "proof_session.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

class ProofSessionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ProofSessionTest, RoundTripsScalarAndGroupMessages)
{
    auto prover = ProofSession::prover();
    const Fr expected_scalar = fr_from_int(42);
    const Group1 expected_group = Group1::one();
    const bool expected_flag = true;

    EXPECT_EQ(prover->exchange_fr("field", [&] { return expected_scalar; }), expected_scalar);
    EXPECT_EQ(prover->exchange_group1("group", [&] { return expected_group; }), expected_group);
    EXPECT_EQ(prover->exchange_bool("flag", [&] { return expected_flag; }), expected_flag);

    const SerializedProof proof = prover->take_proof();
    auto verifier = ProofSession::verifier(proof);

    EXPECT_EQ(verifier->exchange_fr("field", {}), expected_scalar);
    EXPECT_EQ(verifier->exchange_group1("group", {}), expected_group);
    EXPECT_EQ(verifier->exchange_bool("flag", {}), expected_flag);
    EXPECT_TRUE(verifier->proof_fully_consumed());
    EXPECT_EQ(prover->transcript().state(), verifier->transcript().state());
}

TEST_F(ProofSessionTest, RawTransportDoesNotTouchTranscript)
{
    auto prover = ProofSession::prover();
    const auto initial_state = prover->transcript().state();
    const Group1 expected_group = Group1::one();

    EXPECT_EQ(prover->transport_group1([&] { return expected_group; }), expected_group);
    const SerializedProof proof = prover->take_proof();

    auto verifier = ProofSession::verifier(proof);
    EXPECT_EQ(verifier->transport_group1({}), expected_group);
    EXPECT_TRUE(verifier->proof_fully_consumed());
    EXPECT_EQ(prover->transcript().state(), initial_state);
    EXPECT_EQ(verifier->transcript().state(), initial_state);
}

TEST_F(ProofSessionTest, LegacyTextProofRoundTrips)
{
    auto prover = ProofSession::prover(std::make_shared<ProofTranscript>(), ProofEncoding::LegacyText);
    const Fr expected_scalar = fr_from_int(42);
    const Group1 expected_group = Group1::one();

    EXPECT_EQ(prover->exchange_fr("field", [&] { return expected_scalar; }), expected_scalar);
    EXPECT_EQ(prover->transport_group1([&] { return expected_group; }), expected_group);

    const SerializedProof proof = prover->take_proof();
    EXPECT_FALSE(has_binary_proof_magic(proof));

    auto verifier = ProofSession::verifier(proof);
    EXPECT_EQ(verifier->exchange_fr("field", {}), expected_scalar);
    EXPECT_EQ(verifier->transport_group1({}), expected_group);
    EXPECT_TRUE(verifier->proof_fully_consumed());
    EXPECT_EQ(prover->transcript().state(), verifier->transcript().state());
}

TEST_F(ProofSessionTest, CountOnlySessionSynthesizesDeterministicValuesWithoutProofBytes)
{
    auto counter = ProofSession::counter();
    const auto initial_state = counter->transcript().state();

    EXPECT_TRUE(counter->count_only());
    EXPECT_EQ(counter->mode(), ProofMode::Verify);
    EXPECT_EQ(counter->exchange_fr("field", {}), Fr(1));
    EXPECT_EQ(counter->exchange_group1("group", {}), Group1::zero());
    EXPECT_TRUE(counter->exchange_bool("flag", {}));
    EXPECT_EQ(counter->exchange_group1_vector("groups", 3, {}).size(), 3);
    EXPECT_EQ(counter->transport_group1({}), Group1::zero());
    EXPECT_TRUE(counter->proof_fully_consumed());
    EXPECT_NE(counter->transcript().state(), initial_state);
}
