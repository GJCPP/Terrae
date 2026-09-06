#pragma once

#include "plonk/opening.hpp"
#include "plonk/proof_io.hpp"
#include "plonk/transcript.hpp"

#include <vector>

struct plonkPreprocessedVerifierInput {
    // Verifier preprocessing is tied to the lifetime of the source
    // verification key. Borrow it instead of copying its complete SRS.
    const plonkVerificationKey* verification_key = nullptr;
    size_t public_input_count = 0;
    ProofTranscript transcript_prefix;
    plonkPolynomialCommitment public_input_commitment;
};

struct plonkDeferredOpeningClaims {
    Group1 zeta_commitment;
    Group1 zeta_omega_commitment;
    Fr zeta = Fr(0);
    Fr zeta_omega = Fr(0);
    Fr zeta_value = Fr(0);
    Fr zeta_omega_value = Fr(0);
};

bool build_plonk_deferred_opening_claims(const plonkPreprocessedVerifierInput& preprocessed,
                                         const plonkProof& proof,
                                         plonkDeferredOpeningClaims& claims);

plonkPreprocessedVerifierInput preprocess_plonk_verifier_input(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs);

Fr evaluate_plonk_public_input_at_point(const std::vector<size_t>& public_input_rows,
                                        size_t domain_size,
                                        const Fr& omega,
                                        const std::vector<Fr>& public_inputs,
                                        const Fr& point);

Group1 build_plonk_linearisation_commitment(const plonkVerificationKey& verification_key,
                                            const std::vector<Fr>& public_inputs,
                                            const plonkProof& proof,
                                            const Fr& lookup_compression_challenge,
                                            const Fr& beta,
                                            const Fr& gamma,
                                            const Fr& delta,
                                            const Fr& epsilon,
                                            const Fr& alpha,
                                            const Fr& zeta);

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
                                           const Fr& v);

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
                                           const Fr& v);

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
                                        const Fr& u);

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
                                        const Fr& u);

Group1 build_plonk_pairing_e_commitment(const plonkProof& proof, const Fr& v, const Fr& u);

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
                                          const Fr& u);

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
                                          const Fr& u);

bool verify_plonk(const plonkVerificationKey& verification_key,
                  const std::vector<Fr>& public_inputs,
                  const plonkProof& proof);

bool verify_plonk_preprocessed(const plonkPreprocessedVerifierInput& preprocessed,
                               const plonkProof& proof);

bool verify_plonk_serialized(const plonkVerificationKey& verification_key,
                             const std::vector<Fr>& public_inputs,
                             const SerializedProof& serialized);
