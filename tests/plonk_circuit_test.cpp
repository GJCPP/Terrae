#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "plonk/circuit.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

class PlonkCircuitTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkCircuitTest, BuilderCreatesArithmeticAndPublicInputRows)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    const plonkVariable sum = builder.add_addition(product, y);
    builder.expose_public(sum);

    const plonkCircuit circuit = builder.build();
    ASSERT_EQ(circuit.variable_count, 4u);
    ASSERT_EQ(circuit.gates.size(), 3u);
    ASSERT_EQ(circuit.public_input_rows.size(), 1u);
    EXPECT_EQ(circuit.public_input_rows[0], 2u);
    EXPECT_EQ(circuit.domain_size(), 4u);

    EXPECT_EQ(circuit.gates[0].qM, Fr(1));
    EXPECT_EQ(circuit.gates[0].qO, -Fr(1));
    EXPECT_EQ(circuit.gates[1].qL, Fr(1));
    EXPECT_EQ(circuit.gates[1].qR, Fr(1));
    EXPECT_EQ(circuit.gates[1].qO, -Fr(1));
    EXPECT_EQ(circuit.gates[2].qO, Fr(1));
}

TEST_F(PlonkCircuitTest, ConstrainEqualMergesVariableRoots)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    builder.constrain_equal(x, y);

    const plonkCircuit circuit = builder.build();
    EXPECT_EQ(circuit.canonical_variable(x), circuit.canonical_variable(y));
}
