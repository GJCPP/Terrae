#include "plonk/prover.hpp"

#include "plonk/linearisation.hpp"
#include "plonk/lookup.hpp"
#include "plonk/opening.hpp"
#include "plonk/permutation.hpp"
#include "plonk/proof_io.hpp"
#include "plonk/quotient.hpp"
#include "plonk/transcript.hpp"

#include <stdexcept>

namespace {

plonkPolynomialCommitment commit_poly(const kzgPublicParameters& public_parameters, const poly& polynomial)
{
    return plonkPolynomialCommitment{commit_g1_with_public_parameters(public_parameters, polynomial)};
}

} // namespace

static plonkProof prove_plonk_impl(const plonkProvingKey& proving_key,
                                   const plonkWitness& witness,
                                   const std::vector<Fr>& public_inputs,
                                   plonkOpeningPolynomials* deferred_opening)
{
    if (proving_key.public_parameters.g1_powers.empty()) {
        throw std::runtime_error("prove_plonk: proving key does not carry KZG public parameters");
    }

    if (deferred_opening == nullptr) {
    }

    const plonkWireAssignments wires = build_plonk_wire_assignments(proving_key.circuit, witness);
    plonkQuotientPolynomials quotient;
    quotient.a = poly_from_evals(wires.a_evals);
    quotient.b = poly_from_evals(wires.b_evals);
    quotient.c = poly_from_evals(wires.c_evals);

    plonkProof proof;
    proof.a_commitment = commit_poly(proving_key.public_parameters, quotient.a);
    proof.b_commitment = commit_poly(proving_key.public_parameters, quotient.b);
    proof.c_commitment = commit_poly(proving_key.public_parameters, quotient.c);

    ProofTranscript transcript =
        make_plonk_transcript(derive_plonk_verification_key(proving_key), public_inputs);
    absorb_plonk_witness_commitments(transcript, proof);
    const Fr lookup_compression_challenge = sample_plonk_lookup_compression_challenge(transcript);

    const plonkLookupPolynomials lookup =
        build_plonk_lookup_polynomials(proving_key, wires, lookup_compression_challenge);
    proof.f_commitment = commit_poly(proving_key.public_parameters, lookup.f);
    proof.h1_commitment = commit_poly(proving_key.public_parameters, lookup.h1);
    proof.h2_commitment = commit_poly(proving_key.public_parameters, lookup.h2);
    absorb_plonk_lookup_commitments(transcript, proof);
    const auto [beta, gamma] = sample_plonk_permutation_challenges(transcript);
    const auto [delta, epsilon] = sample_plonk_lookup_grand_product_challenges(transcript);

    const std::vector<Fr> z_evals = build_plonk_grand_product_evaluations(proving_key, wires, beta, gamma);
    quotient.z = poly_from_evals(z_evals);
    proof.z_commitment = commit_poly(proving_key.public_parameters, quotient.z);

    const std::vector<Fr> z_lookup_evals =
        build_plonk_lookup_grand_product_evaluations(lookup, proving_key.omega, delta, epsilon);
    quotient.z_lookup = poly_from_evals(z_lookup_evals);
    proof.z_lookup_commitment = commit_poly(proving_key.public_parameters, quotient.z_lookup);

    absorb_plonk_permutation_commitments(transcript, proof);
    const Fr alpha = sample_plonk_alpha(transcript);

    quotient = build_plonk_quotient_polynomials(proving_key,
                                                wires,
                                                public_inputs,
                                                z_evals,
                                                lookup,
                                                z_lookup_evals,
                                                lookup_compression_challenge,
                                                beta,
                                                gamma,
                                                delta,
                                                epsilon,
                                                alpha);
    if (!quotient.divisible) {
        throw std::runtime_error("prove_plonk: quotient numerator is not divisible by the subgroup vanishing polynomial");
    }
    proof.t_lo_commitment = commit_poly(proving_key.public_parameters, quotient.t_lo);
    proof.t_mid_commitment = commit_poly(proving_key.public_parameters, quotient.t_mid);
    proof.t_hi_commitment = commit_poly(proving_key.public_parameters, quotient.t_hi);

    absorb_plonk_quotient_commitments(transcript, proof);
    const Fr zeta = sample_plonk_zeta(transcript, proving_key.n);

    const plonkLinearisationPolynomial linearised =
        build_plonk_linearisation_polynomial(proving_key,
                                             quotient,
                                             lookup_compression_challenge,
                                             beta,
                                             gamma,
                                             delta,
                                             epsilon,
                                             alpha,
                                             zeta);
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
    proof.public_input_eval = linearised.public_input_eval;
    proof.t_eval = linearised.t_eval;
    proof.r_eval = linearised.r_eval;

    absorb_plonk_evaluations(transcript, proof);
    const auto [v, u] = sample_plonk_opening_challenges(transcript);
    (void)u;

    const plonkOpeningPolynomials opening =
        build_plonk_opening_polynomials(proving_key, quotient, linearised, v, zeta);
    if (!opening.zeta_divisible || !opening.zeta_omega_divisible) {
        throw std::runtime_error("prove_plonk: opening witness division failed");
    }
    if (deferred_opening != nullptr) {
        *deferred_opening = opening;
        return proof;
    }
    proof.w_zeta_commitment = commit_poly(proving_key.public_parameters, opening.w_zeta);
    proof.w_zeta_omega_commitment = commit_poly(proving_key.public_parameters, opening.w_zeta_omega);
    absorb_plonk_opening_commitments(transcript, proof);
    return proof;
}

plonkProof prove_plonk(const plonkProvingKey& proving_key,
                       const plonkWitness& witness,
                       const std::vector<Fr>& public_inputs)
{
    return prove_plonk_impl(proving_key, witness, public_inputs, nullptr);
}

plonkDeferredOpeningProof prove_plonk_deferred_opening(
    const plonkProvingKey& proving_key,
    const plonkWitness& witness,
    const std::vector<Fr>& public_inputs)
{
    plonkDeferredOpeningProof out;
    out.proof = prove_plonk_impl(proving_key, witness, public_inputs, &out.openings);
    return out;
}

SerializedProof prove_plonk_serialized(const plonkProvingKey& proving_key,
                                       const plonkWitness& witness,
                                       const std::vector<Fr>& public_inputs)
{
    return serialize_plonk_proof(prove_plonk(proving_key, witness, public_inputs));
}
