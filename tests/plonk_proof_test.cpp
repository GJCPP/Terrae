#include "bls12_381_wrapper.hpp"
#include "plonk/plonk.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

namespace {

struct MultiplicationCircuitFixture {
    plonkSetup setup;
    plonkVariable x;
    plonkVariable y;
    plonkVariable product;
};

struct ChainedCircuitFixture {
    plonkSetup setup;
    plonkVariable x;
    plonkVariable y;
    plonkVariable z;
    plonkVariable xy;
    plonkVariable yz;
    plonkVariable sum;
    plonkVariable constant_five;
    plonkVariable shifted;
    plonkVariable shifted_alias;
};

MultiplicationCircuitFixture build_multiplication_fixture()
{
    plonkCircuitBuilder builder;
    MultiplicationCircuitFixture fixture;
    fixture.x = builder.add_private_variable();
    fixture.y = builder.add_private_variable();
    fixture.product = builder.add_multiplication(fixture.x, fixture.y);
    builder.expose_public(fixture.product);
    fixture.setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(256, 256));
    return fixture;
}

plonkWitness multiplication_witness(const MultiplicationCircuitFixture& fixture,
                                    const Fr& x,
                                    const Fr& y,
                                    const Fr& product)
{
    plonkWitness witness;
    witness.set(fixture.x, x);
    witness.set(fixture.y, y);
    witness.set(fixture.product, product);
    return witness;
}

ChainedCircuitFixture build_chained_fixture()
{
    plonkCircuitBuilder builder;
    ChainedCircuitFixture fixture;
    fixture.x = builder.add_private_variable();
    fixture.y = builder.add_private_variable();
    fixture.z = builder.add_private_variable();
    fixture.xy = builder.add_multiplication(fixture.x, fixture.y);
    fixture.yz = builder.add_multiplication(fixture.y, fixture.z);
    fixture.sum = builder.add_addition(fixture.xy, fixture.yz);
    fixture.constant_five = builder.add_constant(Fr(5));
    fixture.shifted = builder.add_addition(fixture.sum, fixture.constant_five);
    fixture.shifted_alias = builder.add_private_variable();
    builder.constrain_equal(fixture.shifted, fixture.shifted_alias);
    builder.expose_public(fixture.shifted_alias);
    builder.expose_public(fixture.y);
    fixture.setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(256, 256));
    return fixture;
}

plonkWitness chained_witness(const ChainedCircuitFixture& fixture,
                             const Fr& x,
                             const Fr& y,
                             const Fr& z)
{
    plonkWitness witness;
    const Fr xy = x * y;
    const Fr yz = y * z;
    const Fr sum = xy + yz;
    const Fr shifted = sum + Fr(5);
    witness.set(fixture.x, x);
    witness.set(fixture.y, y);
    witness.set(fixture.z, z);
    witness.set(fixture.xy, xy);
    witness.set(fixture.yz, yz);
    witness.set(fixture.sum, sum);
    witness.set(fixture.constant_five, Fr(5));
    witness.set(fixture.shifted, shifted);
    witness.set(fixture.shifted_alias, shifted);
    return witness;
}

} // namespace

class PlonkProofTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkProofTest, ProveAndVerifyMultiplicationCircuit)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));

    const plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(12)});

    EXPECT_TRUE(verify_plonk(fixture.setup.verification_key, {Fr(12)}, proof));
}

TEST_F(PlonkProofTest, ProveAndVerifyPaddedMultiGateCircuitWithSharedVariables)
{
    const ChainedCircuitFixture fixture = build_chained_fixture();
    const plonkWitness witness = chained_witness(fixture, Fr(2), Fr(3), Fr(4));

    ASSERT_EQ(fixture.setup.proving_key.circuit.gates.size(), 7u);
    ASSERT_EQ(fixture.setup.proving_key.n, 8u);

    const plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(23), Fr(3)});

    EXPECT_TRUE(verify_plonk(fixture.setup.verification_key, {Fr(23), Fr(3)}, proof));
    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(23), Fr(4)}, proof));
}

TEST_F(PlonkProofTest, SerializedProofRoundTripsAndVerifies)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    const plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(12)});

    const SerializedProof serialized = serialize_plonk_proof(proof);
    const plonkProof decoded = deserialize_plonk_proof(serialized);

    EXPECT_FALSE(serialized.bytes.empty());
    EXPECT_EQ(serialized.bytes.size(),
              4u + 13u * G1::getSerializedByteSize() + 15u * Fr::getByteSize());
    ASSERT_GE(serialized.bytes.size(), 4u);
    EXPECT_EQ(serialized.bytes[0], static_cast<uint8_t>('P'));
    EXPECT_EQ(serialized.bytes[1], static_cast<uint8_t>('L'));
    EXPECT_EQ(serialized.bytes[2], static_cast<uint8_t>('B'));
    EXPECT_EQ(serialized.bytes[3], static_cast<uint8_t>('1'));
    EXPECT_EQ(decoded, proof);
    EXPECT_TRUE(verify_plonk(fixture.setup.verification_key, {Fr(12)}, decoded));
}

TEST_F(PlonkProofTest, SerializedProveAndVerifyWrapper)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));

    const SerializedProof serialized =
        prove_plonk_serialized(fixture.setup.proving_key, witness, {Fr(12)});

    EXPECT_TRUE(verify_plonk_serialized(fixture.setup.verification_key, {Fr(12)}, serialized));
    EXPECT_FALSE(verify_plonk_serialized(fixture.setup.verification_key, {Fr(13)}, serialized));
}

TEST_F(PlonkProofTest, DeserializeRejectsTrailingBytes)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    SerializedProof serialized =
        prove_plonk_serialized(fixture.setup.proving_key, witness, {Fr(12)});

    serialized.bytes.push_back(0);

    EXPECT_THROW((void)deserialize_plonk_proof(serialized), std::runtime_error);
    EXPECT_FALSE(verify_plonk_serialized(fixture.setup.verification_key, {Fr(12)}, serialized));
}

TEST_F(PlonkProofTest, DeserializeRejectsWrongProofHeader)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    SerializedProof serialized =
        prove_plonk_serialized(fixture.setup.proving_key, witness, {Fr(12)});

    ASSERT_GE(serialized.bytes.size(), 4u);
    serialized.bytes[3] ^= 0x01u;

    EXPECT_THROW((void)deserialize_plonk_proof(serialized), std::runtime_error);
    EXPECT_FALSE(verify_plonk_serialized(fixture.setup.verification_key, {Fr(12)}, serialized));
}

TEST_F(PlonkProofTest, VerifyRejectsTamperedSerializedProof)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    SerializedProof serialized =
        prove_plonk_serialized(fixture.setup.proving_key, witness, {Fr(12)});

    ASSERT_FALSE(serialized.bytes.empty());
    serialized.bytes.back() ^= 0x01u;

    EXPECT_FALSE(verify_plonk_serialized(fixture.setup.verification_key, {Fr(12)}, serialized));
}

TEST_F(PlonkProofTest, VerifyRejectsTamperedPublicInput)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    const plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(12)});

    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(13)}, proof));
}

TEST_F(PlonkProofTest, VerifyRejectsTamperedEvaluation)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(12)});

    proof.a_eval += Fr(1);

    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(12)}, proof));
}

TEST_F(PlonkProofTest, VerifyRejectsTamperedWitnessCommitment)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(12)});

    proof.a_commitment.value += Group1::one();

    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(12)}, proof));
}

TEST_F(PlonkProofTest, VerifyRejectsTamperedPermutationCommitment)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(12)});

    proof.z_commitment.value += Group1::one();

    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(12)}, proof));
}

TEST_F(PlonkProofTest, VerifyRejectsTamperedQuotientCommitment)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(12)});

    proof.t_mid_commitment.value += Group1::one();

    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(12)}, proof));
}

TEST_F(PlonkProofTest, VerifyRejectsTamperedOpeningCommitment)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    plonkProof proof = prove_plonk(fixture.setup.proving_key, witness, {Fr(12)});

    proof.w_zeta_commitment.value += Group1::one();

    EXPECT_FALSE(verify_plonk(fixture.setup.verification_key, {Fr(12)}, proof));
}

TEST_F(PlonkProofTest, ProverRejectsUnsatisfiedCircuit)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(13));

    EXPECT_THROW((void)prove_plonk(fixture.setup.proving_key, witness, {Fr(13)}), std::runtime_error);
}
