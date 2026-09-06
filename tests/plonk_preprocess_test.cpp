#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "plonk/preprocess.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>
#include <set>

using namespace mcl::bn;

class PlonkPreprocessTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkPreprocessTest, BuildsSelectorAndSigmaPolynomials)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    const plonkCircuit circuit = builder.build();
    const plonkSetup setup = preprocess_plonk(circuit, testing_kzg_public_parameters(64, 64));
    const plonkProvingKey& pk = setup.proving_key;

    ASSERT_EQ(pk.n, 2u);
    EXPECT_EQ(pk.qM_evals[0], Fr(1));
    EXPECT_EQ(pk.qO_evals[0], -Fr(1));
    EXPECT_EQ(pk.qO_evals[1], Fr(1));

    const Fr row1_left = plonk_wire_label(pk.n, pk.omega, pk.k1, pk.k2, {1, plonkWireColumn::Left});
    EXPECT_EQ(pk.sigma3_evals[0], row1_left);
    EXPECT_EQ(pk.sigma1.evaluations_on_domain(pk.n), pk.sigma1_evals);
    EXPECT_EQ(pk.sigma2.evaluations_on_domain(pk.n), pk.sigma2_evals);
    EXPECT_EQ(pk.sigma3.evaluations_on_domain(pk.n), pk.sigma3_evals);
}

TEST_F(PlonkPreprocessTest, IdentityCosetsAreDisjoint)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    builder.expose_public(x);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(64, 64));
    const plonkProvingKey& pk = setup.proving_key;

    std::set<Fr> labels;
    for (size_t row = 0; row < pk.n; ++row) {
        EXPECT_TRUE(labels.insert(plonk_wire_label(pk.n, pk.omega, pk.k1, pk.k2, {row, plonkWireColumn::Left})).second);
        EXPECT_TRUE(labels.insert(plonk_wire_label(pk.n, pk.omega, pk.k1, pk.k2, {row, plonkWireColumn::Right})).second);
        EXPECT_TRUE(labels.insert(plonk_wire_label(pk.n, pk.omega, pk.k1, pk.k2, {row, plonkWireColumn::Output})).second);
    }
}

TEST_F(PlonkPreprocessTest, DerivesVerificationKeyFromProvingKey)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(64, 64));
    const plonkVerificationKey derived = derive_plonk_verification_key(setup.proving_key);

    EXPECT_EQ(derived.n, setup.verification_key.n);
    EXPECT_EQ(derived.omega, setup.verification_key.omega);
    EXPECT_EQ(derived.k1, setup.verification_key.k1);
    EXPECT_EQ(derived.k2, setup.verification_key.k2);
    EXPECT_EQ(derived.public_input_rows, setup.verification_key.public_input_rows);
    EXPECT_EQ(derived.qM_commitment, setup.verification_key.qM_commitment);
    EXPECT_EQ(derived.qL_commitment, setup.verification_key.qL_commitment);
    EXPECT_EQ(derived.qR_commitment, setup.verification_key.qR_commitment);
    EXPECT_EQ(derived.qO_commitment, setup.verification_key.qO_commitment);
    EXPECT_EQ(derived.qC_commitment, setup.verification_key.qC_commitment);
    EXPECT_EQ(derived.sigma1_commitment, setup.verification_key.sigma1_commitment);
    EXPECT_EQ(derived.sigma2_commitment, setup.verification_key.sigma2_commitment);
    EXPECT_EQ(derived.sigma3_commitment, setup.verification_key.sigma3_commitment);
}
