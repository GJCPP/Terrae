#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "plonk/linearisation.hpp"
#include "plonk/permutation.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

class PlonkLinearisationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkLinearisationTest, LinearisationEvaluatesToQuotientIdentity)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(256, 256));
    plonkWitness witness;
    witness.set(x, Fr(3));
    witness.set(y, Fr(4));
    witness.set(product, Fr(12));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);

    const Fr beta(7);
    const Fr gamma(11);
    const Fr lookup_compression(17);
    const Fr delta(19);
    const Fr epsilon(23);
    const Fr alpha(13);
    const Fr zeta(5);
    const plonkLookupPolynomials lookup =
        build_plonk_lookup_polynomials(setup.proving_key, wires, lookup_compression);
    const std::vector<Fr> z_evals =
        build_plonk_grand_product_evaluations(setup.proving_key, wires, beta, gamma);
    const std::vector<Fr> z_lookup_evals =
        build_plonk_lookup_grand_product_evaluations(lookup, setup.proving_key.omega, delta, epsilon);
    const plonkQuotientPolynomials quotient =
        build_plonk_quotient_polynomials(setup.proving_key,
                                         wires,
                                         {Fr(12)},
                                         z_evals,
                                         lookup,
                                         z_lookup_evals,
                                         lookup_compression,
                                         beta,
                                         gamma,
                                         delta,
                                         epsilon,
                                         alpha);
    ASSERT_TRUE(quotient.divisible);

    const plonkLinearisationPolynomial linearised =
        build_plonk_linearisation_polynomial(setup.proving_key,
                                             quotient,
                                             lookup_compression,
                                             beta,
                                             gamma,
                                             delta,
                                             epsilon,
                                             alpha,
                                             zeta);

    const Fr gate_eval =
        setup.proving_key.qM.eval(zeta) * linearised.a_eval * linearised.b_eval +
        setup.proving_key.qL.eval(zeta) * linearised.a_eval +
        setup.proving_key.qR.eval(zeta) * linearised.b_eval +
        setup.proving_key.qO.eval(zeta) * linearised.c_eval +
        setup.proving_key.qC.eval(zeta) + quotient.public_input.eval(zeta);
    const Fr left_identity = linearised.a_eval + beta * zeta + gamma;
    const Fr right_identity = linearised.b_eval + beta * setup.proving_key.k1 * zeta + gamma;
    const Fr output_identity = linearised.c_eval + beta * setup.proving_key.k2 * zeta + gamma;
    const Fr left_sigma = linearised.a_eval + beta * linearised.sigma1_eval + gamma;
    const Fr right_sigma = linearised.b_eval + beta * linearised.sigma2_eval + gamma;
    const Fr output_sigma =
        linearised.c_eval + beta * setup.proving_key.sigma3.eval(zeta) + gamma;
    const Fr permutation_eval =
        linearised.z_eval * left_identity * right_identity * output_identity -
        linearised.z_omega_eval * left_sigma * right_sigma * output_sigma;
    const Fr boundary_eval = linearised.l1_eval * (linearised.z_eval - Fr(1));

    EXPECT_EQ(gate_eval, quotient.n_gate.eval(zeta));
    EXPECT_EQ(permutation_eval, quotient.n_permutation.eval(zeta));
    EXPECT_EQ(boundary_eval, quotient.n_boundary.eval(zeta));

    const Fr lookup_compression_squared = lookup_compression * lookup_compression;
    const Fr lookup_query_eval =
        setup.proving_key.qK.eval(zeta) *
        (linearised.a_eval + lookup_compression * linearised.b_eval +
         lookup_compression_squared * linearised.c_eval - linearised.f_eval);
    const Fr lookup_numerator_factor =
        (Fr(1) + delta) * (epsilon + linearised.f_eval) *
        (epsilon * (Fr(1) + delta) + linearised.lookup_table_eval +
         delta * linearised.lookup_table_omega_eval);
    const Fr lookup_denominator_factor =
        (linearised.h1_eval + delta * linearised.h2_eval + epsilon * (Fr(1) + delta)) *
        (epsilon * (Fr(1) + delta) + linearised.h2_eval + delta * linearised.h1_omega_eval);
    const Fr lookup_grand_product_eval =
        linearised.z_lookup_eval * lookup_numerator_factor -
        linearised.z_lookup_omega_eval * lookup_denominator_factor;
    const Fr lookup_boundary_eval = linearised.l1_eval * (linearised.z_lookup_eval - Fr(1));

    EXPECT_EQ(lookup_query_eval, quotient.n_lookup_query.eval(zeta));
    EXPECT_EQ(lookup_grand_product_eval, quotient.n_lookup_permutation.eval(zeta));
    EXPECT_EQ(lookup_boundary_eval, quotient.n_lookup_boundary.eval(zeta));

    const Fr zeta_n = pow_fr(zeta, setup.proving_key.n);
    const Fr scaled_chunks_eval =
        quotient.t_lo.eval(zeta) + zeta_n * quotient.t_mid.eval(zeta) +
        zeta_n * zeta_n * quotient.t_hi.eval(zeta);
    EXPECT_EQ(scaled_chunks_eval, quotient.quotient.eval(zeta));

    const Fr expected = quotient.numerator.eval(zeta) - linearised.z_h_eval * quotient.quotient.eval(zeta);
    EXPECT_EQ(linearised.r_eval, expected);
    EXPECT_TRUE(expected.isZero());
}

TEST_F(PlonkLinearisationTest, LinearisationDetectsTamperedPublicInput)
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(256, 256));
    plonkWitness witness;
    witness.set(x, Fr(3));
    witness.set(y, Fr(4));
    witness.set(product, Fr(12));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);

    const Fr beta(7);
    const Fr gamma(11);
    const Fr lookup_compression(17);
    const Fr delta(19);
    const Fr epsilon(23);
    const Fr alpha(13);
    const Fr zeta(5);
    const plonkLookupPolynomials lookup =
        build_plonk_lookup_polynomials(setup.proving_key, wires, lookup_compression);
    const std::vector<Fr> z_evals =
        build_plonk_grand_product_evaluations(setup.proving_key, wires, beta, gamma);
    const std::vector<Fr> z_lookup_evals =
        build_plonk_lookup_grand_product_evaluations(lookup, setup.proving_key.omega, delta, epsilon);
    const plonkQuotientPolynomials quotient =
        build_plonk_quotient_polynomials(setup.proving_key,
                                         wires,
                                         {Fr(13)},
                                         z_evals,
                                         lookup,
                                         z_lookup_evals,
                                         lookup_compression,
                                         beta,
                                         gamma,
                                         delta,
                                         epsilon,
                                         alpha);
    ASSERT_FALSE(quotient.divisible);

    const plonkLinearisationPolynomial linearised =
        build_plonk_linearisation_polynomial(setup.proving_key,
                                             quotient,
                                             lookup_compression,
                                             beta,
                                             gamma,
                                             delta,
                                             epsilon,
                                             alpha,
                                             zeta);

    const Fr expected = quotient.numerator.eval(zeta) - linearised.z_h_eval * quotient.quotient.eval(zeta);
    EXPECT_EQ(linearised.r_eval, expected);
    EXPECT_FALSE(expected.isZero());
}

TEST_F(PlonkLinearisationTest, LinearisationEvaluatesLookupIdentity)
{
    plonkCircuitBuilder builder;
    const plonkVariable left = builder.add_private_variable();
    const plonkVariable right = builder.add_private_variable();
    const plonkVariable output = builder.add_private_variable();
    builder.add_lookup_table_row(Fr(1), Fr(2), Fr(3));
    builder.add_lookup_table_row(Fr(4), Fr(5), Fr(9));
    builder.add_lookup_table_row(Fr(7), Fr(8), Fr(15));
    builder.add_lookup(left, right, output);
    builder.expose_public(output);

    const plonkSetup setup = preprocess_plonk(builder.build(), testing_kzg_public_parameters(256, 256));
    plonkWitness witness;
    witness.set(left, Fr(4));
    witness.set(right, Fr(5));
    witness.set(output, Fr(9));
    const plonkWireAssignments wires = build_plonk_wire_assignments(setup.proving_key.circuit, witness);

    const Fr beta(7);
    const Fr gamma(11);
    const Fr lookup_compression(17);
    const Fr delta(19);
    const Fr epsilon(23);
    const Fr alpha(13);
    const Fr zeta(5);
    const plonkLookupPolynomials lookup =
        build_plonk_lookup_polynomials(setup.proving_key, wires, lookup_compression);
    const std::vector<Fr> z_evals =
        build_plonk_grand_product_evaluations(setup.proving_key, wires, beta, gamma);
    const std::vector<Fr> z_lookup_evals =
        build_plonk_lookup_grand_product_evaluations(lookup, setup.proving_key.omega, delta, epsilon);
    const plonkQuotientPolynomials quotient =
        build_plonk_quotient_polynomials(setup.proving_key,
                                         wires,
                                         {Fr(9)},
                                         z_evals,
                                         lookup,
                                         z_lookup_evals,
                                         lookup_compression,
                                         beta,
                                         gamma,
                                         delta,
                                         epsilon,
                                         alpha);
    ASSERT_TRUE(quotient.divisible);

    const plonkLinearisationPolynomial linearised =
        build_plonk_linearisation_polynomial(setup.proving_key,
                                             quotient,
                                             lookup_compression,
                                             beta,
                                             gamma,
                                             delta,
                                             epsilon,
                                             alpha,
                                             zeta);

    const Fr expected = quotient.numerator.eval(zeta) - linearised.z_h_eval * quotient.quotient.eval(zeta);
    EXPECT_EQ(linearised.r_eval, expected);
    EXPECT_TRUE(expected.isZero());
}
