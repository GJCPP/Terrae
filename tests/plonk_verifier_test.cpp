#include "bls12_381_wrapper.hpp"
#include "plonk/linearisation.hpp"
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

plonkProof make_linearisation_proof_shell(const kzgPublicParameters& params,
                                          const plonkQuotientPolynomials& quotient,
                                          const plonkLinearisationPolynomial& linearised)
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
    return proof;
}

struct LinearisationFixture {
    kzgPublicParameters params;
    plonkSetup setup;
    plonkQuotientPolynomials quotient;
    plonkLinearisationPolynomial linearised;
    plonkProof proof;
    Fr lookup_compression = Fr(17);
    Fr beta = Fr(7);
    Fr gamma = Fr(11);
    Fr delta = Fr(19);
    Fr epsilon = Fr(23);
    Fr alpha = Fr(13);
    Fr zeta = Fr(5);
};

LinearisationFixture build_fixture()
{
    plonkCircuitBuilder builder;
    const plonkVariable x = builder.add_private_variable();
    const plonkVariable y = builder.add_private_variable();
    const plonkVariable product = builder.add_multiplication(x, y);
    builder.expose_public(product);

    LinearisationFixture fixture;
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
    fixture.proof = make_linearisation_proof_shell(fixture.params, fixture.quotient, fixture.linearised);
    return fixture;
}

} // namespace

class PlonkVerifierTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(PlonkVerifierTest, LinearisationCommitmentMatchesExplicitPolynomial)
{
    const LinearisationFixture fixture = build_fixture();

    const Group1 verifier_commitment = build_plonk_linearisation_commitment(fixture.setup.verification_key,
                                                                            {Fr(12)},
                                                                            fixture.proof,
                                                                            fixture.lookup_compression,
                                                                            fixture.beta,
                                                                            fixture.gamma,
                                                                            fixture.delta,
                                                                            fixture.epsilon,
                                                                            fixture.alpha,
                                                                            fixture.zeta);
    const Group1 direct_commitment = commit_g1_with_public_parameters(fixture.params, fixture.linearised.r);
    EXPECT_EQ(verifier_commitment, direct_commitment);
}

TEST_F(PlonkVerifierTest, LinearisationCommitmentBindsPublicInput)
{
    const LinearisationFixture fixture = build_fixture();

    const Group1 valid_commitment = build_plonk_linearisation_commitment(fixture.setup.verification_key,
                                                                         {Fr(12)},
                                                                         fixture.proof,
                                                                         fixture.lookup_compression,
                                                                         fixture.beta,
                                                                         fixture.gamma,
                                                                         fixture.delta,
                                                                         fixture.epsilon,
                                                                         fixture.alpha,
                                                                         fixture.zeta);
    const Group1 tampered_commitment = build_plonk_linearisation_commitment(fixture.setup.verification_key,
                                                                            {Fr(13)},
                                                                            fixture.proof,
                                                                            fixture.lookup_compression,
                                                                            fixture.beta,
                                                                            fixture.gamma,
                                                                            fixture.delta,
                                                                            fixture.epsilon,
                                                                            fixture.alpha,
                                                                            fixture.zeta);
    EXPECT_NE(tampered_commitment, valid_commitment);
}
