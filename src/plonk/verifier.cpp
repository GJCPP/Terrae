#include "plonk/verifier.hpp"

#include "algebra_utils.hpp"
#include "plonk/quotient.hpp"
#include "plonk/proof_io.hpp"
#include "plonk/transcript.hpp"

#include <stdexcept>
#include <string>

namespace {

Fr fr_from_size(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

Fr shifted_value(const Fr& value, const Fr& beta, const Fr& label, const Fr& gamma)
{
    Fr out;
    Fr::mul(out, beta, label);
    Fr::add(out, out, value);
    Fr::add(out, out, gamma);
    return out;
}

Group1 commitment_value(const plonkPolynomialCommitment& commitment)
{
    return commitment.value;
}

plonkPolynomialCommitment commit_poly(const kzgPublicParameters& public_parameters, const poly& polynomial)
{
    return plonkPolynomialCommitment{commit_g1_with_public_parameters(public_parameters, polynomial)};
}

std::vector<Fr> build_public_input_evaluations(const std::vector<size_t>& public_input_rows,
                                               size_t domain_size,
                                               const std::vector<Fr>& public_inputs)
{
    if (public_inputs.size() != public_input_rows.size()) {
        throw std::runtime_error("build_public_input_evaluations: public input count mismatch");
    }
    std::vector<Fr> evals(domain_size, Fr(0));
    for (size_t i = 0; i < public_inputs.size(); ++i) {
        if (public_input_rows[i] >= domain_size) {
            throw std::runtime_error("build_public_input_evaluations: public input row out of range");
        }
        Fr neg;
        Fr::neg(neg, public_inputs[i]);
        Fr::add(evals[public_input_rows[i]], evals[public_input_rows[i]], neg);
    }
    return evals;
}

Group1 compressed_lookup_table_commitment(const plonkVerificationKey& verification_key,
                                         const Fr& lookup_compression_challenge)
{
    const Fr lookup_compression_squared =
        lookup_compression_challenge * lookup_compression_challenge;
    return commitment_value(verification_key.table_1_commitment) +
           lookup_compression_challenge * commitment_value(verification_key.table_2_commitment) +
           lookup_compression_squared * commitment_value(verification_key.table_3_commitment);
}

Fr subgroup_lagrange_eval(size_t domain_size, const Fr& omega, size_t row, const Fr& point)
{
    if (domain_size == 0 || row >= domain_size) {
        throw std::runtime_error("subgroup_lagrange_eval: invalid row");
    }

    const Fr domain_point = pow_fr(omega, row);
    if (point == domain_point) {
        return Fr(1);
    }

    const Fr z_h_eval = pow_fr(point, domain_size) - Fr(1);
    const Fr denominator = fr_from_size(domain_size) * (point - domain_point);
    if (denominator.isZero()) {
        throw std::runtime_error("subgroup_lagrange_eval: zero denominator");
    }
    Fr denominator_inv;
    Fr::inv(denominator_inv, denominator);
    return z_h_eval * domain_point * denominator_inv;
}

Group1 quotient_chunk_commitment(const plonkProof& proof, size_t domain_size, const Fr& zeta)
{
    const Fr zeta_n = pow_fr(zeta, domain_size);
    const Fr zeta_2n = zeta_n * zeta_n;
    return commitment_value(proof.t_lo_commitment) + zeta_n * commitment_value(proof.t_mid_commitment) +
           zeta_2n * commitment_value(proof.t_hi_commitment);
}

std::vector<Fr> powers_of_v(const Fr& v, size_t count)
{
    std::vector<Fr> powers(count, Fr(1));
    for (size_t i = 1; i < count; ++i) {
        powers[i] = powers[i - 1] * v;
    }
    return powers;
}

} // namespace

plonkPreprocessedVerifierInput preprocess_plonk_verifier_input(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs)
{
    if (public_inputs.size() != verification_key.public_input_rows.size()) {
        throw std::runtime_error("preprocess_plonk_verifier_input: public input count mismatch");
    }
    plonkPreprocessedVerifierInput out;
    out.verification_key = &verification_key;
    out.public_input_count = public_inputs.size();
    out.transcript_prefix = make_plonk_transcript(verification_key, public_inputs);
    const std::vector<Fr> public_input_evals =
        build_public_input_evaluations(verification_key.public_input_rows,
                                       verification_key.n,
                                       public_inputs);
    out.public_input_commitment =
        commit_poly(verification_key.public_parameters, poly_from_evals(public_input_evals));
    return out;
}

Group1 build_plonk_linearisation_commitment_from_public_input_eval(const plonkVerificationKey& verification_key,
                                                                   const Fr& public_input_eval,
                                                                   const plonkProof& proof,
                                                                   const Fr& lookup_compression_challenge,
                                                                   const Fr& beta,
                                                                   const Fr& gamma,
                                                                   const Fr& delta,
                                                                   const Fr& epsilon,
                                                                   const Fr& alpha,
                                                                   const Fr& zeta);

Fr evaluate_plonk_public_input_at_point(const std::vector<size_t>& public_input_rows,
                                        size_t domain_size,
                                        const Fr& omega,
                                        const std::vector<Fr>& public_inputs,
                                        const Fr& point)
{
    if (public_inputs.size() != public_input_rows.size()) {
        throw std::runtime_error("evaluate_plonk_public_input_at_point: public input count mismatch");
    }

    Fr out(0);
    for (size_t i = 0; i < public_inputs.size(); ++i) {
        const Fr basis_eval = subgroup_lagrange_eval(domain_size, omega, public_input_rows[i], point);
        out -= public_inputs[i] * basis_eval;
    }
    return out;
}

Group1 build_plonk_linearisation_commitment(const plonkVerificationKey& verification_key,
                                            const std::vector<Fr>& public_inputs,
                                            const plonkProof& proof,
                                            const Fr& lookup_compression_challenge,
                                            const Fr& beta,
                                            const Fr& gamma,
                                            const Fr& delta,
                                            const Fr& epsilon,
                                            const Fr& alpha,
                                            const Fr& zeta)
{
    const Fr pi_eval = evaluate_plonk_public_input_at_point(verification_key.public_input_rows,
                                                            verification_key.n,
                                                            verification_key.omega,
                                                            public_inputs,
                                                            zeta);
    return build_plonk_linearisation_commitment_from_public_input_eval(verification_key,
                                                                       pi_eval,
                                                                       proof,
                                                                       lookup_compression_challenge,
                                                                       beta,
                                                                       gamma,
                                                                       delta,
                                                                       epsilon,
                                                                       alpha,
                                                                       zeta);
}

Group1 build_plonk_linearisation_commitment_from_public_input_eval(const plonkVerificationKey& verification_key,
                                                                   const Fr& public_input_eval,
                                                                   const plonkProof& proof,
                                                                   const Fr& lookup_compression_challenge,
                                                                   const Fr& beta,
                                                                   const Fr& gamma,
                                                                   const Fr& delta,
                                                                   const Fr& epsilon,
                                                                   const Fr& alpha,
                                                                   const Fr& zeta)
{
    if (verification_key.n == 0) {
        throw std::runtime_error("build_plonk_linearisation_commitment: empty domain");
    }

    const Fr z_h_eval = pow_fr(zeta, verification_key.n) - Fr(1);
    const Fr l1_eval =
        subgroup_lagrange_eval(verification_key.n, verification_key.omega, 0, zeta);

    const Fr left_identity = shifted_value(proof.a_eval, beta, zeta, gamma);
    const Fr right_identity = shifted_value(proof.b_eval, beta, verification_key.k1 * zeta, gamma);
    const Fr output_identity = shifted_value(proof.c_eval, beta, verification_key.k2 * zeta, gamma);
    const Fr left_sigma = shifted_value(proof.a_eval, beta, proof.sigma1_eval, gamma);
    const Fr right_sigma = shifted_value(proof.b_eval, beta, proof.sigma2_eval, gamma);

    const Fr identity_product = left_identity * right_identity * output_identity;
    const Fr sigma_prefix = left_sigma * right_sigma * proof.z_omega_eval;

    const Group1 gate_commitment =
        (proof.a_eval * proof.b_eval) * commitment_value(verification_key.qM_commitment) +
        proof.a_eval * commitment_value(verification_key.qL_commitment) +
        proof.b_eval * commitment_value(verification_key.qR_commitment) +
        proof.c_eval * commitment_value(verification_key.qO_commitment) +
        commitment_value(verification_key.qC_commitment) + public_input_eval * Group1::one();

    const Group1 sigma3_factor_commitment =
        beta * commitment_value(verification_key.sigma3_commitment) + (proof.c_eval + gamma) * Group1::one();
    const Group1 permutation_commitment =
        identity_product * commitment_value(proof.z_commitment) - sigma_prefix * sigma3_factor_commitment;

    const Group1 boundary_commitment =
        l1_eval * (commitment_value(proof.z_commitment) - Group1::one());

    const Fr lookup_compression_squared =
        lookup_compression_challenge * lookup_compression_challenge;
    const Group1 lookup_query_commitment =
        (proof.a_eval + lookup_compression_challenge * proof.b_eval +
         lookup_compression_squared * proof.c_eval - proof.f_eval) *
        commitment_value(verification_key.qK_commitment);

    const Fr lookup_numerator_factor =
        (Fr(1) + delta) * (epsilon + proof.f_eval) *
        (epsilon * (Fr(1) + delta) + proof.lookup_table_eval +
         delta * proof.lookup_table_omega_eval);
    const Fr lookup_denominator_factor =
        epsilon * (Fr(1) + delta) + proof.h2_eval + delta * proof.h1_omega_eval;
    const Group1 lookup_denominator_commitment =
        lookup_denominator_factor *
        (commitment_value(proof.h1_commitment) +
         delta * commitment_value(proof.h2_commitment) +
         epsilon * (Fr(1) + delta) * Group1::one());
    const Group1 lookup_grand_product_commitment =
        lookup_numerator_factor * commitment_value(proof.z_lookup_commitment) -
        proof.z_lookup_omega_eval * lookup_denominator_commitment;

    const Group1 lookup_boundary_commitment =
        l1_eval * (commitment_value(proof.z_lookup_commitment) - Group1::one());

    return gate_commitment + alpha * permutation_commitment +
           (alpha * alpha) * boundary_commitment +
           (alpha * alpha * alpha) * lookup_query_commitment +
           (alpha * alpha * alpha * alpha) * lookup_grand_product_commitment +
           (alpha * alpha * alpha * alpha * alpha) * lookup_boundary_commitment -
           z_h_eval * quotient_chunk_commitment(proof, verification_key.n, zeta);
}

Group1 build_plonk_batched_zeta_commitment(const plonkVerificationKey& verification_key,
                                           const std::vector<Fr>& public_inputs,
                                           const plonkProof& proof,
                                           const Fr& lookup_compression_challenge,
                                           const Fr& beta,
                                           const Fr& gamma,
                                           const Fr& delta,
                                           const Fr& epsilon,
                                           const Fr& alpha,
                                           const Fr& zeta,
                                           const Fr& v)
{
    const std::vector<Fr> powers = powers_of_v(v, 11);
    const Group1 r_commitment = build_plonk_linearisation_commitment(verification_key,
                                                                     public_inputs,
                                                                     proof,
                                                                     lookup_compression_challenge,
                                                                     beta,
                                                                     gamma,
                                                                     delta,
                                                                     epsilon,
                                                                     alpha,
                                                                     zeta);
    return quotient_chunk_commitment(proof, verification_key.n, zeta) +
           powers[1] * r_commitment + powers[2] * commitment_value(proof.a_commitment) +
           powers[3] * commitment_value(proof.b_commitment) +
           powers[4] * commitment_value(proof.c_commitment) +
           powers[5] * commitment_value(verification_key.sigma1_commitment) +
           powers[6] * commitment_value(verification_key.sigma2_commitment) +
           powers[7] * commitment_value(proof.f_commitment) +
           powers[8] * commitment_value(proof.h2_commitment) +
           powers[9] * compressed_lookup_table_commitment(verification_key, lookup_compression_challenge) +
           powers[10] * commitment_value(
               commit_poly(verification_key.public_parameters,
                           poly_from_evals(build_public_input_evaluations(verification_key.public_input_rows,
                                                                          verification_key.n,
                                                                          public_inputs))));
}

Group1 build_plonk_batched_zeta_commitment(const plonkVerificationKey& verification_key,
                                           const plonkPolynomialCommitment& public_input_commitment,
                                           const plonkProof& proof,
                                           const Fr& lookup_compression_challenge,
                                           const Fr& beta,
                                           const Fr& gamma,
                                           const Fr& delta,
                                           const Fr& epsilon,
                                           const Fr& alpha,
                                           const Fr& zeta,
                                           const Fr& v)
{
    const std::vector<Fr> powers = powers_of_v(v, 11);
    const Group1 r_commitment =
        build_plonk_linearisation_commitment_from_public_input_eval(verification_key,
                                                                    proof.public_input_eval,
                                                                    proof,
                                                                    lookup_compression_challenge,
                                                                    beta,
                                                                    gamma,
                                                                    delta,
                                                                    epsilon,
                                                                    alpha,
                                                                    zeta);
    return quotient_chunk_commitment(proof, verification_key.n, zeta) +
           powers[1] * r_commitment + powers[2] * commitment_value(proof.a_commitment) +
           powers[3] * commitment_value(proof.b_commitment) +
           powers[4] * commitment_value(proof.c_commitment) +
           powers[5] * commitment_value(verification_key.sigma1_commitment) +
           powers[6] * commitment_value(verification_key.sigma2_commitment) +
           powers[7] * commitment_value(proof.f_commitment) +
           powers[8] * commitment_value(proof.h2_commitment) +
           powers[9] * compressed_lookup_table_commitment(verification_key, lookup_compression_challenge) +
           powers[10] * commitment_value(public_input_commitment);
}

Group1 build_plonk_pairing_f_commitment(const plonkVerificationKey& verification_key,
                                        const std::vector<Fr>& public_inputs,
                                        const plonkProof& proof,
                                        const Fr& lookup_compression_challenge,
                                        const Fr& beta,
                                        const Fr& gamma,
                                        const Fr& delta,
                                        const Fr& epsilon,
                                        const Fr& alpha,
                                        const Fr& zeta,
                                        const Fr& v,
                                        const Fr& u)
{
    return build_plonk_batched_zeta_commitment(verification_key,
                                               public_inputs,
                                               proof,
                                               lookup_compression_challenge,
                                               beta,
                                               gamma,
                                               delta,
                                               epsilon,
                                               alpha,
                                               zeta,
                                               v) +
           u * (commitment_value(proof.z_commitment) +
                v * compressed_lookup_table_commitment(verification_key, lookup_compression_challenge) +
                (v * v) * commitment_value(proof.z_lookup_commitment) +
                (v * v * v) * commitment_value(proof.h1_commitment));
}

Group1 build_plonk_pairing_f_commitment(const plonkVerificationKey& verification_key,
                                        const plonkPolynomialCommitment& public_input_commitment,
                                        const plonkProof& proof,
                                        const Fr& lookup_compression_challenge,
                                        const Fr& beta,
                                        const Fr& gamma,
                                        const Fr& delta,
                                        const Fr& epsilon,
                                        const Fr& alpha,
                                        const Fr& zeta,
                                        const Fr& v,
                                        const Fr& u)
{
    return build_plonk_batched_zeta_commitment(verification_key,
                                               public_input_commitment,
                                               proof,
                                               lookup_compression_challenge,
                                               beta,
                                               gamma,
                                               delta,
                                               epsilon,
                                               alpha,
                                               zeta,
                                               v) +
           u * (commitment_value(proof.z_commitment) +
                v * compressed_lookup_table_commitment(verification_key, lookup_compression_challenge) +
                (v * v) * commitment_value(proof.z_lookup_commitment) +
                (v * v * v) * commitment_value(proof.h1_commitment));
}

Group1 build_plonk_pairing_e_commitment(const plonkProof& proof, const Fr& v, const Fr& u)
{
    const Fr batched_zeta_eval = evaluate_plonk_batched_zeta_opening(proof, v);
    return (batched_zeta_eval +
            u * (proof.z_omega_eval + v * proof.lookup_table_omega_eval +
                 (v * v) * proof.z_lookup_omega_eval + (v * v * v) * proof.h1_omega_eval)) *
           Group1::one();
}

bool verify_plonk_batched_opening_pairing(const kzgPublicParameters& public_parameters,
                                          const plonkVerificationKey& verification_key,
                                          const std::vector<Fr>& public_inputs,
                                          const plonkProof& proof,
                                          const Fr& lookup_compression_challenge,
                                          const Fr& beta,
                                          const Fr& gamma,
                                          const Fr& delta,
                                          const Fr& epsilon,
                                          const Fr& alpha,
                                          const Fr& zeta,
                                          const Fr& v,
                                          const Fr& u)
{
    if (public_parameters.g2_powers.size() < 2) {
        throw std::runtime_error("verify_plonk_batched_opening_pairing: missing x G2 SRS power");
    }

    const Group1 w_zeta = commitment_value(proof.w_zeta_commitment);
    const Group1 w_zeta_omega = commitment_value(proof.w_zeta_omega_commitment);
    const Group1 f_commitment = build_plonk_pairing_f_commitment(verification_key,
                                                                 public_inputs,
                                                                 proof,
                                                                 lookup_compression_challenge,
                                                                 beta,
                                                                 gamma,
                                                                 delta,
                                                                 epsilon,
                                                                 alpha,
                                                                 zeta,
                                                                 v,
                                                                 u);
    const Group1 e_commitment = build_plonk_pairing_e_commitment(proof, v, u);

    const Group1 left_g1 = w_zeta + u * w_zeta_omega;
    const Group1 right_g1 = zeta * w_zeta + (u * zeta * verification_key.omega) * w_zeta_omega +
                            f_commitment - e_commitment;
    proof_cost::add("pairing_count", 2);
    return pairing(left_g1, public_parameters.g2_powers[1]) == pairing(right_g1, Group2::one());
}

bool verify_plonk_batched_opening_pairing(const kzgPublicParameters& public_parameters,
                                          const plonkVerificationKey& verification_key,
                                          const plonkPolynomialCommitment& public_input_commitment,
                                          const plonkProof& proof,
                                          const Fr& lookup_compression_challenge,
                                          const Fr& beta,
                                          const Fr& gamma,
                                          const Fr& delta,
                                          const Fr& epsilon,
                                          const Fr& alpha,
                                          const Fr& zeta,
                                          const Fr& v,
                                          const Fr& u)
{
    if (public_parameters.g2_powers.size() < 2) {
        throw std::runtime_error("verify_plonk_batched_opening_pairing: missing x G2 SRS power");
    }

    const Group1 w_zeta = commitment_value(proof.w_zeta_commitment);
    const Group1 w_zeta_omega = commitment_value(proof.w_zeta_omega_commitment);
    const Group1 f_commitment = build_plonk_pairing_f_commitment(verification_key,
                                                                 public_input_commitment,
                                                                 proof,
                                                                 lookup_compression_challenge,
                                                                 beta,
                                                                 gamma,
                                                                 delta,
                                                                 epsilon,
                                                                 alpha,
                                                                 zeta,
                                                                 v,
                                                                 u);
    const Group1 e_commitment = build_plonk_pairing_e_commitment(proof, v, u);

    const Group1 left_g1 = w_zeta + u * w_zeta_omega;
    const Group1 right_g1 = zeta * w_zeta + (u * zeta * verification_key.omega) * w_zeta_omega +
                            f_commitment - e_commitment;
    proof_cost::add("pairing_count", 2);
    return pairing(left_g1, public_parameters.g2_powers[1]) == pairing(right_g1, Group2::one());
}

bool build_plonk_deferred_opening_claims(const plonkPreprocessedVerifierInput& preprocessed,
                                         const plonkProof& proof,
                                         plonkDeferredOpeningClaims& claims)
{
    try {
        if (preprocessed.verification_key == nullptr) {
            return false;
        }
        const plonkVerificationKey& verification_key = *preprocessed.verification_key;
        if (verification_key.public_parameters.g2_powers.size() < 2 ||
            verification_key.n == 0 ||
            preprocessed.public_input_count != verification_key.public_input_rows.size()) {
            return false;
        }

        ProofTranscript transcript = preprocessed.transcript_prefix;
        absorb_plonk_witness_commitments(transcript, proof);
        const Fr lookup_compression_challenge =
            sample_plonk_lookup_compression_challenge(transcript);
        absorb_plonk_lookup_commitments(transcript, proof);
        const auto [beta, gamma] = sample_plonk_permutation_challenges(transcript);
        const auto [delta, epsilon] =
            sample_plonk_lookup_grand_product_challenges(transcript);
        absorb_plonk_permutation_commitments(transcript, proof);
        const Fr alpha = sample_plonk_alpha(transcript);
        absorb_plonk_quotient_commitments(transcript, proof);
        const Fr zeta = sample_plonk_zeta(transcript, verification_key.n);
        absorb_plonk_evaluations(transcript, proof);
        const auto [v, u] = sample_plonk_opening_challenges(transcript);
        (void)u;

        if (!proof.r_eval.isZero()) {
            return false;
        }

        claims.zeta_commitment =
            build_plonk_batched_zeta_commitment(verification_key,
                                                preprocessed.public_input_commitment,
                                                proof,
                                                lookup_compression_challenge,
                                                beta,
                                                gamma,
                                                delta,
                                                epsilon,
                                                alpha,
                                                zeta,
                                                v);
        claims.zeta_omega_commitment =
            commitment_value(proof.z_commitment) +
            v * compressed_lookup_table_commitment(verification_key,
                                                   lookup_compression_challenge) +
            (v * v) * commitment_value(proof.z_lookup_commitment) +
            (v * v * v) * commitment_value(proof.h1_commitment);
        claims.zeta = zeta;
        claims.zeta_omega = zeta * verification_key.omega;
        claims.zeta_value = evaluate_plonk_batched_zeta_opening(proof, v);
        claims.zeta_omega_value =
            proof.z_omega_eval + v * proof.lookup_table_omega_eval +
            (v * v) * proof.z_lookup_omega_eval +
            (v * v * v) * proof.h1_omega_eval;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool verify_plonk(const plonkVerificationKey& verification_key,
                  const std::vector<Fr>& public_inputs,
                  const plonkProof& proof)
{
    try {
        return verify_plonk_preprocessed(preprocess_plonk_verifier_input(verification_key, public_inputs), proof);
    } catch (const std::exception&) {
        return false;
    }
}

bool verify_plonk_preprocessed(const plonkPreprocessedVerifierInput& preprocessed,
                               const plonkProof& proof)
{
    try {
        if (preprocessed.verification_key == nullptr) {
            return false;
        }
        const plonkVerificationKey& verification_key = *preprocessed.verification_key;
        if (verification_key.public_parameters.g2_powers.size() < 2 || verification_key.n == 0) {
            return false;
        }
        if (preprocessed.public_input_count != verification_key.public_input_rows.size()) {
            return false;
        }

        ProofTranscript transcript = preprocessed.transcript_prefix;
        absorb_plonk_witness_commitments(transcript, proof);
        const Fr lookup_compression_challenge = sample_plonk_lookup_compression_challenge(transcript);
        absorb_plonk_lookup_commitments(transcript, proof);
        const auto [beta, gamma] = sample_plonk_permutation_challenges(transcript);
        const auto [delta, epsilon] = sample_plonk_lookup_grand_product_challenges(transcript);

        absorb_plonk_permutation_commitments(transcript, proof);
        const Fr alpha = sample_plonk_alpha(transcript);

        absorb_plonk_quotient_commitments(transcript, proof);
        const Fr zeta = sample_plonk_zeta(transcript, verification_key.n);

        absorb_plonk_evaluations(transcript, proof);
        const auto [v, u] = sample_plonk_opening_challenges(transcript);

        if (!proof.r_eval.isZero()) {
            return false;
        }
        if (!verify_plonk_batched_opening_pairing(verification_key.public_parameters,
                                                  verification_key,
                                                  preprocessed.public_input_commitment,
                                                  proof,
                                                  lookup_compression_challenge,
                                                  beta,
                                                  gamma,
                                                  delta,
                                                  epsilon,
                                                  alpha,
                                                  zeta,
                                                  v,
                                                  u)) {
            return false;
        }

        absorb_plonk_opening_commitments(transcript, proof);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool verify_plonk_serialized(const plonkVerificationKey& verification_key,
                             const std::vector<Fr>& public_inputs,
                             const SerializedProof& serialized)
{
    try {
        return verify_plonk(verification_key, public_inputs, deserialize_plonk_proof(serialized));
    } catch (const std::exception&) {
        return false;
    }
}
