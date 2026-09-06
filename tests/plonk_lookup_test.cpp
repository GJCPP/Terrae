#include "bls12_381_wrapper.hpp"
#include "plonk/plonk.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

namespace {

struct LookupCircuitFixture {
    plonkSetup setup;
    plonkVariable left;
    plonkVariable right;
    plonkVariable output;
};

LookupCircuitFixture build_lookup_fixture()
{
    plonkCircuitBuilder builder;
    LookupCircuitFixture fixture;
    fixture.left = builder.add_private_variable();
    fixture.right = builder.add_private_variable();
    fixture.output = builder.add_private_variable();

    builder.add_lookup_table_row(Fr(1), Fr(2), Fr(3));
    builder.add_lookup_table_row(Fr(4), Fr(5), Fr(9));
    builder.add_lookup_table_row(Fr(7), Fr(8), Fr(15));
    builder.add_lookup(fixture.left, fixture.right, fixture.output);
    builder.expose_public(fixture.output);

    fixture.setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(256, 256));
    return fixture;
}

plonkWitness lookup_witness(const LookupCircuitFixture& fixture,
                            const Fr& left,
                            const Fr& right,
                            const Fr& output)
{
    plonkWitness witness;
    witness.set(fixture.left, left);
    witness.set(fixture.right, right);
    witness.set(fixture.output, output);
    return witness;
}

struct LookupTranscriptChallenges {
    Fr lookup_compression = Fr(0);
    Fr beta = Fr(0);
    Fr gamma = Fr(0);
    Fr delta = Fr(0);
    Fr epsilon = Fr(0);
    Fr alpha = Fr(0);
    Fr zeta = Fr(0);
    Fr v = Fr(0);
    Fr u = Fr(0);
};

LookupTranscriptChallenges derive_lookup_challenges(const plonkVerificationKey& verification_key,
                                                    const std::vector<Fr>& public_inputs,
                                                    const plonkProof& proof)
{
    LookupTranscriptChallenges out;
    ProofTranscript transcript = make_plonk_transcript(verification_key, public_inputs);
    absorb_plonk_witness_commitments(transcript, proof);
    out.lookup_compression = sample_plonk_lookup_compression_challenge(transcript);
    absorb_plonk_lookup_commitments(transcript, proof);
    std::tie(out.beta, out.gamma) = sample_plonk_permutation_challenges(transcript);
    std::tie(out.delta, out.epsilon) = sample_plonk_lookup_grand_product_challenges(transcript);
    absorb_plonk_permutation_commitments(transcript, proof);
    out.alpha = sample_plonk_alpha(transcript);
    absorb_plonk_quotient_commitments(transcript, proof);
    out.zeta = sample_plonk_zeta(transcript, verification_key.n);
    absorb_plonk_evaluations(transcript, proof);
    std::tie(out.v, out.u) = sample_plonk_opening_challenges(transcript);
    return out;
}

} // namespace

class PlonkLookupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkLookupTest, PreprocessCommitsLookupSelectorAndTableColumns)
{
    const LookupCircuitFixture fixture = build_lookup_fixture();

    ASSERT_EQ(fixture.setup.proving_key.n, 4u);
    ASSERT_EQ(fixture.setup.proving_key.qK_evals.size(), fixture.setup.proving_key.n);
    ASSERT_EQ(fixture.setup.proving_key.table_1_evals.size(), fixture.setup.proving_key.n);
    ASSERT_EQ(fixture.setup.proving_key.table_2_evals.size(), fixture.setup.proving_key.n);
    ASSERT_EQ(fixture.setup.proving_key.table_3_evals.size(), fixture.setup.proving_key.n);

    EXPECT_EQ(fixture.setup.proving_key.qK_evals[0], Fr(1));
    EXPECT_EQ(fixture.setup.proving_key.qK_evals[1], Fr(0));
    EXPECT_EQ(fixture.setup.proving_key.table_1_evals[0], Fr(1));
    EXPECT_EQ(fixture.setup.proving_key.table_2_evals[1], Fr(5));
    EXPECT_EQ(fixture.setup.proving_key.table_3_evals[2], Fr(15));
}

TEST_F(PlonkLookupTest, ProveAndVerifyLookupCircuit)
{
    const LookupCircuitFixture fixture = build_lookup_fixture();
    const plonkWitness witness = lookup_witness(fixture, Fr(4), Fr(5), Fr(9));

    const plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(9)});
    const LookupTranscriptChallenges challenges =
        derive_lookup_challenges(fixture.setup.verification_key, {Fr(9)}, proof);

    EXPECT_EQ(proof.r_eval, Fr(0));
    EXPECT_TRUE(verify_plonk_batched_opening_pairing(fixture.setup.verification_key.public_parameters,
                                                     fixture.setup.verification_key,
                                                     {Fr(9)},
                                                     proof,
                                                     challenges.lookup_compression,
                                                     challenges.beta,
                                                     challenges.gamma,
                                                     challenges.delta,
                                                     challenges.epsilon,
                                                     challenges.alpha,
                                                     challenges.zeta,
                                                     challenges.v,
                                                     challenges.u));
    EXPECT_TRUE(verify_plonk(fixture.setup.verification_key, {Fr(9)}, proof));
    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(8)}, proof));
}

TEST_F(PlonkLookupTest, SerializedLookupProofRoundTrips)
{
    const LookupCircuitFixture fixture = build_lookup_fixture();
    const plonkWitness witness = lookup_witness(fixture, Fr(7), Fr(8), Fr(15));

    const SerializedProof serialized =
        prove_plonk_serialized(fixture.setup.proving_key, witness, {Fr(15)});

    EXPECT_TRUE(verify_plonk_serialized(fixture.setup.verification_key, {Fr(15)}, serialized));
    EXPECT_FALSE(verify_plonk_serialized(fixture.setup.verification_key, {Fr(9)}, serialized));
}

TEST_F(PlonkLookupTest, ProverRejectsWitnessOutsidePublicTable)
{
    const LookupCircuitFixture fixture = build_lookup_fixture();
    const plonkWitness witness = lookup_witness(fixture, Fr(4), Fr(5), Fr(10));

    EXPECT_THROW((void)prove_plonk(fixture.setup.proving_key, witness, {Fr(10)}), std::runtime_error);
}

TEST_F(PlonkLookupTest, VerifyRejectsTamperedLookupCommitment)
{
    const LookupCircuitFixture fixture = build_lookup_fixture();
    const plonkWitness witness = lookup_witness(fixture, Fr(1), Fr(2), Fr(3));
    plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(3)});

    proof.f_commitment.value += Group1::one();

    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(3)}, proof));
}
