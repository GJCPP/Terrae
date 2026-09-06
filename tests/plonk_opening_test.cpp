#include "bls12_381_wrapper.hpp"
#include "plonk/opening.hpp"
#include "plonk/permutation.hpp"
#include "plonk/verifier.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

namespace {

plonkPolynomialCommitment test_commit(const kzgPublicParameters& params, const poly& polynomial)
{
    return plonkPolynomialCommitment{commit_g1_with_public_parameters(params, polynomial)};
}

poly constant_poly(const Fr& value)
{
    poly out;
    out.add_term(0, value);
    return out;
}

poly linear_divisor(const Fr& point)
{
    return identity_poly() - point;
}

plonkProof make_proof_shell(const kzgPublicParameters& params,
                            const plonkQuotientPolynomials& quotient,
                            const plonkLinearisationPolynomial& linearised,
                            const plonkOpeningPolynomials& opening)
{
    plonkProof proof;
    proof.a_commitment = test_commit(params, quotient.a);
    proof.b_commitment = test_commit(params, quotient.b);
    proof.c_commitment = test_commit(params, quotient.c);
    proof.f_commitment = test_commit(params, quotient.f);
    proof.h1_commitment = test_commit(params, quotient.h1);
    proof.h2_commitment = test_commit(params, quotient.h2);
    proof.z_commitment = test_commit(params, quotient.z);
    proof.z_lookup_commitment = test_commit(params, quotient.z_lookup);
    proof.t_lo_commitment = test_commit(params, quotient.t_lo);
    proof.t_mid_commitment = test_commit(params, quotient.t_mid);
    proof.t_hi_commitment = test_commit(params, quotient.t_hi);
    proof.w_zeta_commitment = test_commit(params, opening.w_zeta);
    proof.w_zeta_omega_commitment = test_commit(params, opening.w_zeta_omega);
    proof.a_eval = linearised.a_eval;
    proof.b_eval = linearised.b_eval;
    proof.c_eval = linearised.c_eval;
    proof.sigma1_eval = linearised.sigma1_eval;
    proof.sigma2_eval = linearised.sigma2_eval;
    proof.f_eval = linearised.f_eval;
    proof.lookup_table_eval = linearised.lookup_table_eval;
    proof.lookup_table_omega_eval = linearised.lookup_table_omega_eval;
    proof.z_omega_eval = linearised.z_omega_eval;
    proof.z_lookup_omega_eval = linearised.z_lookup_omega_eval;
    proof.h1_omega_eval = linearised.h1_omega_eval;
    proof.h2_eval = linearised.h2_eval;
    proof.t_eval = linearised.t_eval;
    proof.r_eval = linearised.r_eval;
    proof.public_input_eval = linearised.public_input_eval;
    return proof;
}

struct OpeningFixture {
    kzgPublicParameters params;
    plonkSetup setup;
    plonkQuotientPolynomials quotient;
    plonkLinearisationPolynomial linearised;
    plonkOpeningPolynomials opening;
    plonkProof proof;
    Fr lookup_compression = Fr(17);
    Fr beta = Fr(7);
    Fr gamma = Fr(11);
    Fr delta = Fr(19);
    Fr epsilon = Fr(23);
    Fr alpha = Fr(13);
    Fr zeta = Fr(5);
    Fr v = Fr(17);
    Fr u = Fr(19);
};

OpeningFixture build_fixture()
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    OpeningFixture fixture;
    fixture.params = testing_kzg_public_parameters(256, 256);
    fixture.setup = preprocess_plonk(builder.build(), fixture.params);

    plonkWitness witness;
    witness.set(x, Fr(3));
    witness.set(y, Fr(4));
    witness.set(product, Fr(12));
    const plonkWireAssignments wires =
        build_plonk_wire_assignments(fixture.setup.proving_key.circuit, witness);
    const plonkLookupPolynomials lookup =
        build_plonk_lookup_polynomials(fixture.setup.proving_key, wires, fixture.lookup_compression);
    const std::vector<Fr> z_evals =
        build_plonk_grand_product_evaluations(fixture.setup.proving_key, wires, fixture.beta, fixture.gamma);
    const std::vector<Fr> z_lookup_evals =
        build_plonk_lookup_grand_product_evaluations(lookup,
                                                     fixture.setup.proving_key.omega,
                                                     fixture.delta,
                                                     fixture.epsilon);
    fixture.quotient = build_plonk_quotient_polynomials(fixture.setup.proving_key,
                                                        wires,
                                                        {Fr(12)},
                                                        z_evals,
                                                        lookup,
                                                        z_lookup_evals,
                                                        fixture.lookup_compression,
                                                        fixture.beta,
                                                        fixture.gamma,
                                                        fixture.delta,
                                                        fixture.epsilon,
                                                        fixture.alpha);
    fixture.linearised = build_plonk_linearisation_polynomial(fixture.setup.proving_key,
                                                              fixture.quotient,
                                                              fixture.lookup_compression,
                                                              fixture.beta,
                                                              fixture.gamma,
                                                              fixture.delta,
                                                              fixture.epsilon,
                                                              fixture.alpha,
                                                              fixture.zeta);
    fixture.opening = build_plonk_opening_polynomials(fixture.setup.proving_key,
                                                      fixture.quotient,
                                                      fixture.linearised,
                                                      fixture.v,
                                                      fixture.zeta);
    fixture.proof = make_proof_shell(fixture.params, fixture.quotient, fixture.linearised, fixture.opening);
    return fixture;
}

} // namespace

class PlonkOpeningTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkOpeningTest, OpeningWitnessesReconstructOpenedPolynomials)
{
    const OpeningFixture fixture = build_fixture();

    ASSERT_TRUE(fixture.opening.zeta_divisible);
    ASSERT_TRUE(fixture.opening.zeta_omega_divisible);
    EXPECT_TRUE((fixture.opening.w_zeta * linear_divisor(fixture.zeta) +
                 constant_poly(fixture.opening.batched_zeta_eval) -
                 fixture.opening.batched_zeta)
                    .is_zero());
    EXPECT_TRUE((fixture.opening.w_zeta_omega *
                     linear_divisor(fixture.zeta * fixture.setup.proving_key.omega) +
                 constant_poly(fixture.linearised.z_omega_eval) - fixture.quotient.z)
                    .is_zero());
}

TEST_F(PlonkOpeningTest, VerifierBatchedCommitmentsMatchExplicitPolynomials)
{
    const OpeningFixture fixture = build_fixture();

    EXPECT_EQ(evaluate_plonk_batched_zeta_opening(fixture.proof, fixture.v),
              fixture.opening.batched_zeta_eval);

    const Group1 batched_zeta_commitment = build_plonk_batched_zeta_commitment(fixture.setup.verification_key,
                                                                               {Fr(12)},
                                                                               fixture.proof,
                                                                               fixture.lookup_compression,
                                                                               fixture.beta,
                                                                               fixture.gamma,
                                                                               fixture.delta,
                                                                               fixture.epsilon,
                                                                               fixture.alpha,
                                                                               fixture.zeta,
                                                                               fixture.v);
    EXPECT_EQ(batched_zeta_commitment, commit_g1_with_public_parameters(fixture.params, fixture.opening.batched_zeta));

    const Group1 f_commitment = build_plonk_pairing_f_commitment(fixture.setup.verification_key,
                                                                 {Fr(12)},
                                                                 fixture.proof,
                                                                 fixture.lookup_compression,
                                                                 fixture.beta,
                                                                 fixture.gamma,
                                                                 fixture.delta,
                                                                 fixture.epsilon,
                                                                 fixture.alpha,
                                                                 fixture.zeta,
                                                                 fixture.v,
                                                                 fixture.u);
    EXPECT_EQ(f_commitment,
              commit_g1_with_public_parameters(fixture.params,
                                               fixture.opening.batched_zeta +
                                                   fixture.u *
                                                       (fixture.quotient.z +
                                                        fixture.quotient.lookup_table * fixture.v +
                                                        fixture.quotient.z_lookup * (fixture.v * fixture.v) +
                                                        fixture.quotient.h1 *
                                                            (fixture.v * fixture.v * fixture.v))));

    const Group1 e_commitment = build_plonk_pairing_e_commitment(fixture.proof, fixture.v, fixture.u);
    EXPECT_EQ(e_commitment,
              (fixture.opening.batched_zeta_eval +
               fixture.u *
                   (fixture.linearised.z_omega_eval +
                    fixture.v * fixture.linearised.lookup_table_omega_eval +
                    (fixture.v * fixture.v) * fixture.linearised.z_lookup_omega_eval +
                    (fixture.v * fixture.v * fixture.v) * fixture.linearised.h1_omega_eval)) *
                  Group1::one());
}

TEST_F(PlonkOpeningTest, PairingCheckAcceptsValidOpeningsAndRejectsTamperedEvaluation)
{
    const OpeningFixture fixture = build_fixture();

    EXPECT_TRUE(verify_plonk_batched_opening_pairing(fixture.params,
                                                     fixture.setup.verification_key,
                                                     {Fr(12)},
                                                     fixture.proof,
                                                     fixture.lookup_compression,
                                                     fixture.beta,
                                                     fixture.gamma,
                                                     fixture.delta,
                                                     fixture.epsilon,
                                                     fixture.alpha,
                                                     fixture.zeta,
                                                     fixture.v,
                                                     fixture.u));

    plonkProof tampered = fixture.proof;
    tampered.a_eval += Fr(1);
    EXPECT_FALSE(verify_plonk_batched_opening_pairing(fixture.params,
                                                      fixture.setup.verification_key,
                                                      {Fr(12)},
                                                      tampered,
                                                      fixture.lookup_compression,
                                                      fixture.beta,
                                                      fixture.gamma,
                                                      fixture.delta,
                                                      fixture.epsilon,
                                                      fixture.alpha,
                                                      fixture.zeta,
                                                      fixture.v,
                                                      fixture.u));
}
