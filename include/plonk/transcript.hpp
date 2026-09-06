#pragma once

#include "plonk/proof.hpp"
#include "transcript.hpp"

#include <utility>
#include <vector>

struct plonkTranscriptChallenges {
    Fr lookup_compression = Fr(0);
    Fr beta = Fr(0);
    Fr gamma = Fr(0);
    Fr delta = Fr(0);
    Fr epsilon = Fr(0);
    Fr alpha = Fr(0);
    Fr zeta = Fr(0);
    Fr v = Fr(0);
    Fr u = Fr(0);
};

void absorb_plonk_verification_key(ProofTranscript& transcript,
                                   const plonkVerificationKey& verification_key);
ProofTranscript make_plonk_transcript(const plonkVerificationKey& verification_key,
                                      const std::vector<Fr>& public_inputs);

void absorb_plonk_witness_commitments(ProofTranscript& transcript, const plonkProof& proof);
void absorb_plonk_lookup_commitments(ProofTranscript& transcript, const plonkProof& proof);
void absorb_plonk_permutation_commitments(ProofTranscript& transcript, const plonkProof& proof);
void absorb_plonk_quotient_commitments(ProofTranscript& transcript, const plonkProof& proof);
void absorb_plonk_evaluations(ProofTranscript& transcript, const plonkProof& proof);
void absorb_plonk_opening_commitments(ProofTranscript& transcript, const plonkProof& proof);

Fr sample_plonk_lookup_compression_challenge(ProofTranscript& transcript);
std::pair<Fr, Fr> sample_plonk_permutation_challenges(ProofTranscript& transcript);
std::pair<Fr, Fr> sample_plonk_lookup_grand_product_challenges(ProofTranscript& transcript);
Fr sample_plonk_alpha(ProofTranscript& transcript);
Fr sample_plonk_zeta(ProofTranscript& transcript, size_t domain_size);
std::pair<Fr, Fr> sample_plonk_opening_challenges(ProofTranscript& transcript);
