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

plonkSetup build_addition_public_fixture()
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable sum = builder.add_addition(x, y);
    builder.expose_public(sum);
    return preprocess_plonk(builder.build(), testing_kzg_public_parameters(256, 256));
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

} // namespace

class PlonkBundleTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkBundleTest, ProofBundleRoundTripsAndVerifies)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));

    const plonkProofBundle bundle = prove_plonk_bundle(fixture.setup.proving_key, witness, {Fr(12)});
    const SerializedProof serialized = serialize_plonk_proof_bundle(bundle);
    const plonkProofBundle decoded = deserialize_plonk_proof_bundle(serialized);

    EXPECT_EQ(decoded, bundle);
    EXPECT_FALSE(serialized.bytes.empty());
    ASSERT_GE(serialized.bytes.size(), 4u);
    EXPECT_EQ(serialized.bytes[0], static_cast<uint8_t>('P'));
    EXPECT_EQ(serialized.bytes[1], static_cast<uint8_t>('L'));
    EXPECT_EQ(serialized.bytes[2], static_cast<uint8_t>('B'));
    EXPECT_EQ(serialized.bytes[3], static_cast<uint8_t>('B'));
    EXPECT_TRUE(verify_plonk_bundle(fixture.setup.verification_key, {Fr(12)}, decoded));
    EXPECT_TRUE(verify_plonk_serialized_bundle(fixture.setup.verification_key, {Fr(12)}, serialized));
}

TEST_F(PlonkBundleTest, VerifyRejectsWrongPublicInputCount)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    plonkProofBundle bundle = prove_plonk_bundle(fixture.setup.proving_key, witness, {Fr(12)});

    bundle.public_input_count += 1;

    EXPECT_FALSE(verify_plonk_bundle(fixture.setup.verification_key, {Fr(12)}, bundle));
}

TEST_F(PlonkBundleTest, VerifyRejectsWrongVerificationKeyDigest)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkSetup addition_setup = build_addition_public_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    const plonkProofBundle bundle = prove_plonk_bundle(fixture.setup.proving_key, witness, {Fr(12)});

    EXPECT_FALSE(verify_plonk_bundle(addition_setup.verification_key, {Fr(12)}, bundle));
}

TEST_F(PlonkBundleTest, DeserializeRejectsTrailingBytes)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    SerializedProof serialized =
        serialize_plonk_proof_bundle(prove_plonk_bundle(fixture.setup.proving_key, witness, {Fr(12)}));

    serialized.bytes.push_back(0);

    EXPECT_THROW((void)deserialize_plonk_proof_bundle(serialized), std::runtime_error);
    EXPECT_FALSE(verify_plonk_serialized_bundle(fixture.setup.verification_key, {Fr(12)}, serialized));
}

TEST_F(PlonkBundleTest, DeserializeRejectsWrongBundleHeader)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    SerializedProof serialized =
        serialize_plonk_proof_bundle(prove_plonk_bundle(fixture.setup.proving_key, witness, {Fr(12)}));

    ASSERT_GE(serialized.bytes.size(), 4u);
    serialized.bytes[3] ^= 0x01u;

    EXPECT_THROW((void)deserialize_plonk_proof_bundle(serialized), std::runtime_error);
    EXPECT_FALSE(verify_plonk_serialized_bundle(fixture.setup.verification_key, {Fr(12)}, serialized));
}

TEST_F(PlonkBundleTest, VerifyRejectsTamperedSerializedBundle)
{
    const MultiplicationCircuitFixture fixture = build_multiplication_fixture();
    const plonkWitness witness = multiplication_witness(fixture, Fr(3), Fr(4), Fr(12));
    SerializedProof serialized =
        serialize_plonk_proof_bundle(prove_plonk_bundle(fixture.setup.proving_key, witness, {Fr(12)}));

    ASSERT_FALSE(serialized.bytes.empty());
    serialized.bytes.back() ^= 0x01u;

    EXPECT_FALSE(verify_plonk_serialized_bundle(fixture.setup.verification_key, {Fr(12)}, serialized));
}
