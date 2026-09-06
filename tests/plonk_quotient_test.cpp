#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "plonk/permutation.hpp"
#include "plonk/quotient.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

class PlonkQuotientTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkQuotientTest, GateConstraintsAcceptValidWitnessAndPublicInput)
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

    EXPECT_TRUE(check_plonk_gate_constraints(setup.proving_key, wires, {Fr(12)}).satisfied);
    EXPECT_FALSE(check_plonk_gate_constraints(setup.proving_key, wires, {Fr(13)}).satisfied);
}

TEST_F(PlonkQuotientTest, GateConstraintsRejectInvalidWitness)
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
    witness.set(product, Fr(11));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);

    EXPECT_FALSE(check_plonk_gate_constraints(setup.proving_key, wires, {Fr(11)}).satisfied);
}

TEST_F(PlonkQuotientTest, QuotientDividesForValidCircuit)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(128, 128));
    plonkWitness witness;
    witness.set(x, Fr(3));
    witness.set(y, Fr(4));
    witness.set(product, Fr(12));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);
    const std::vector<Fr> z_evals =
        build_plonk_grand_product_evaluations(setup.proving_key, wires, Fr(7), Fr(11));
    const plonkLookupPolynomials lookup =
        build_plonk_lookup_polynomials(setup.proving_key, wires, Fr(17));
    const std::vector<Fr> z_lookup_evals =
        build_plonk_lookup_grand_product_evaluations(lookup, setup.proving_key.omega, Fr(19), Fr(23));

    const plonkQuotientPolynomials quotient =
        build_plonk_quotient_polynomials(setup.proving_key,
                                         wires,
                                         {Fr(12)},
                                         z_evals,
                                         lookup,
                                         z_lookup_evals,
                                         Fr(17),
                                         Fr(7),
                                         Fr(11),
                                         Fr(19),
                                         Fr(23),
                                         Fr(13));

    EXPECT_TRUE(quotient.divisible);
    EXPECT_TRUE(quotient.remainder.is_zero());

    poly x_to_n;
    x_to_n.add_term(setup.proving_key.n, Fr(1));
    poly x_to_2n;
    x_to_2n.add_term(2 * setup.proving_key.n, Fr(1));
    const poly reconstructed = quotient.t_lo + quotient.t_mid * x_to_n + quotient.t_hi * x_to_2n;
    EXPECT_EQ(reconstructed.coefficients(), quotient.quotient.coefficients());
}

TEST_F(PlonkQuotientTest, QuotientRemainderRejectsTamperedPublicInput)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(128, 128));
    plonkWitness witness;
    witness.set(x, Fr(3));
    witness.set(y, Fr(4));
    witness.set(product, Fr(12));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);
    const std::vector<Fr> z_evals =
        build_plonk_grand_product_evaluations(setup.proving_key, wires, Fr(7), Fr(11));
    const plonkLookupPolynomials lookup =
        build_plonk_lookup_polynomials(setup.proving_key, wires, Fr(17));
    const std::vector<Fr> z_lookup_evals =
        build_plonk_lookup_grand_product_evaluations(lookup, setup.proving_key.omega, Fr(19), Fr(23));

    const plonkQuotientPolynomials quotient =
        build_plonk_quotient_polynomials(setup.proving_key,
                                         wires,
                                         {Fr(13)},
                                         z_evals,
                                         lookup,
                                         z_lookup_evals,
                                         Fr(17),
                                         Fr(7),
                                         Fr(11),
                                         Fr(19),
                                         Fr(23),
                                         Fr(13));

    EXPECT_FALSE(quotient.divisible);
    EXPECT_FALSE(quotient.remainder.is_zero());
}

TEST_F(PlonkQuotientTest, QuotientRemainderRejectsTamperedCopyConstraint)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    builder.constrain_equal(x, y);
    builder.expose_public(x);
    builder.expose_public(y);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(128, 128));
    plonkWitness witness;
    witness.set(x, Fr(5));
    witness.set(y, Fr(7));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);
    const std::vector<Fr> z_evals =
        build_plonk_grand_product_evaluations(setup.proving_key, wires, Fr(7), Fr(11));
    const plonkLookupPolynomials lookup =
        build_plonk_lookup_polynomials(setup.proving_key, wires, Fr(17));
    const std::vector<Fr> z_lookup_evals =
        build_plonk_lookup_grand_product_evaluations(lookup, setup.proving_key.omega, Fr(19), Fr(23));

    const plonkQuotientPolynomials quotient =
        build_plonk_quotient_polynomials(setup.proving_key,
                                         wires,
                                         {Fr(5), Fr(7)},
                                         z_evals,
                                         lookup,
                                         z_lookup_evals,
                                         Fr(17),
                                         Fr(7),
                                         Fr(11),
                                         Fr(19),
                                         Fr(23),
                                         Fr(13));

    EXPECT_FALSE(quotient.divisible);
    EXPECT_FALSE(quotient.remainder.is_zero());
}
