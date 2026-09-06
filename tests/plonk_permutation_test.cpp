#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "plonk/permutation.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

class PlonkPermutationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkPermutationTest, GrandProductAcceptsConsistentCopyAssignments)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(64, 64));
    plonkWitness witness;
    witness.set(x, Fr(3));
    witness.set(y, Fr(4));
    witness.set(product, Fr(12));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);

    const plonkPermutationCheck check =
        check_plonk_permutation_argument(setup.proving_key, wires, Fr(7), Fr(11));
    EXPECT_TRUE(check.satisfied);
    ASSERT_EQ(check.z_evals.size(), setup.proving_key.n);
    EXPECT_EQ(check.z_evals.front(), Fr(1));
}

TEST_F(PlonkPermutationTest, GrandProductRejectsInconsistentCopyAssignments)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    builder.constrain_equal(x, y);
    builder.expose_public(x);
    builder.expose_public(y);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(64, 64));
    plonkWitness witness;
    witness.set(x, Fr(5));
    witness.set(y, Fr(7));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);

    EXPECT_FALSE(check_plonk_permutation_argument(setup.proving_key, wires, Fr(7), Fr(11)).satisfied);
}
