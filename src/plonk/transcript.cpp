#include "plonk/transcript.hpp"

#include "algebra_utils.hpp"

#include <string>

namespace {

void absorb_fr(ProofTranscript& transcript, const char* label, const Fr& value)
{
    transcript.absorb_string(label, transcript_representation(value));
}

std::array<uint8_t, 32> public_input_rows_digest(const std::vector<size_t>& rows)
{
    ProofTranscript digest("PLONK public input rows");
    digest.absorb_size("plonk.public_input_rows.count", rows.size());
    for (const size_t row : rows) {
        digest.absorb_size("plonk.public_input_rows.row", row);
    }
    return digest.squeeze_hash("plonk.public_input_rows.digest");
}

std::array<uint8_t, 32> public_inputs_digest(const std::vector<Fr>& public_inputs)
{
    ProofTranscript digest("PLONK public inputs");
    digest.absorb_size("plonk.public_inputs.count", public_inputs.size());
    for (const Fr& value : public_inputs) {
        absorb_fr(digest, "plonk.public_input", value);
    }
    return digest.squeeze_hash("plonk.public_inputs.digest");
}

void absorb_commitment(ProofTranscript& transcript, const char* label, const plonkPolynomialCommitment& commitment)
{
    transcript.absorb_string(label, transcript_representation(commitment.value));
}

} // namespace

ProofTranscript make_plonk_transcript(const plonkVerificationKey& verification_key,
                                      const std::vector<Fr>& public_inputs)
{
    ProofTranscript transcript("PLONK transcript");
    absorb_plonk_verification_key(transcript, verification_key);

    transcript.absorb_size("plonk.public_inputs.count", public_inputs.size());
    const auto digest = public_inputs_digest(public_inputs);
    transcript.absorb_bytes("plonk.public_inputs.digest", digest.data(), digest.size());
    return transcript;
}

void absorb_plonk_verification_key(ProofTranscript& transcript,
                                   const plonkVerificationKey& verification_key)
{
    transcript.absorb_size("plonk.domain.n", verification_key.n);
    absorb_fr(transcript, "plonk.domain.omega", verification_key.omega);
    absorb_fr(transcript, "plonk.domain.k1", verification_key.k1);
    absorb_fr(transcript, "plonk.domain.k2", verification_key.k2);

    transcript.absorb_size("plonk.public_input_rows.count", verification_key.public_input_rows.size());
    const auto rows_digest = public_input_rows_digest(verification_key.public_input_rows);
    transcript.absorb_bytes("plonk.public_input_rows.digest", rows_digest.data(), rows_digest.size());

    absorb_commitment(transcript, "plonk.vk.selector.qM", verification_key.qM_commitment);
    absorb_commitment(transcript, "plonk.vk.selector.qL", verification_key.qL_commitment);
    absorb_commitment(transcript, "plonk.vk.selector.qR", verification_key.qR_commitment);
    absorb_commitment(transcript, "plonk.vk.selector.qO", verification_key.qO_commitment);
    absorb_commitment(transcript, "plonk.vk.selector.qC", verification_key.qC_commitment);
    absorb_commitment(transcript, "plonk.vk.selector.qK", verification_key.qK_commitment);
    absorb_commitment(transcript, "plonk.vk.sigma.1", verification_key.sigma1_commitment);
    absorb_commitment(transcript, "plonk.vk.sigma.2", verification_key.sigma2_commitment);
    absorb_commitment(transcript, "plonk.vk.sigma.3", verification_key.sigma3_commitment);
    absorb_commitment(transcript, "plonk.vk.table.1", verification_key.table_1_commitment);
    absorb_commitment(transcript, "plonk.vk.table.2", verification_key.table_2_commitment);
    absorb_commitment(transcript, "plonk.vk.table.3", verification_key.table_3_commitment);
}

void absorb_plonk_witness_commitments(ProofTranscript& transcript, const plonkProof& proof)
{
    absorb_commitment(transcript, "plonk.commitment.a", proof.a_commitment);
    absorb_commitment(transcript, "plonk.commitment.b", proof.b_commitment);
    absorb_commitment(transcript, "plonk.commitment.c", proof.c_commitment);
}

void absorb_plonk_lookup_commitments(ProofTranscript& transcript, const plonkProof& proof)
{
    absorb_commitment(transcript, "plonk.commitment.f", proof.f_commitment);
    absorb_commitment(transcript, "plonk.commitment.h1", proof.h1_commitment);
    absorb_commitment(transcript, "plonk.commitment.h2", proof.h2_commitment);
}

void absorb_plonk_permutation_commitments(ProofTranscript& transcript, const plonkProof& proof)
{
    absorb_commitment(transcript, "plonk.commitment.z", proof.z_commitment);
    absorb_commitment(transcript, "plonk.commitment.z_lookup", proof.z_lookup_commitment);
}

void absorb_plonk_quotient_commitments(ProofTranscript& transcript, const plonkProof& proof)
{
    absorb_commitment(transcript, "plonk.commitment.t_lo", proof.t_lo_commitment);
    absorb_commitment(transcript, "plonk.commitment.t_mid", proof.t_mid_commitment);
    absorb_commitment(transcript, "plonk.commitment.t_hi", proof.t_hi_commitment);
}

void absorb_plonk_evaluations(ProofTranscript& transcript, const plonkProof& proof)
{
    absorb_fr(transcript, "plonk.evaluation.a", proof.a_eval);
    absorb_fr(transcript, "plonk.evaluation.b", proof.b_eval);
    absorb_fr(transcript, "plonk.evaluation.c", proof.c_eval);
    absorb_fr(transcript, "plonk.evaluation.sigma1", proof.sigma1_eval);
    absorb_fr(transcript, "plonk.evaluation.sigma2", proof.sigma2_eval);
    absorb_fr(transcript, "plonk.evaluation.f", proof.f_eval);
    absorb_fr(transcript, "plonk.evaluation.lookup_table", proof.lookup_table_eval);
    absorb_fr(transcript, "plonk.evaluation.lookup_table_omega", proof.lookup_table_omega_eval);
    absorb_fr(transcript, "plonk.evaluation.z_omega", proof.z_omega_eval);
    absorb_fr(transcript, "plonk.evaluation.z_lookup_omega", proof.z_lookup_omega_eval);
    absorb_fr(transcript, "plonk.evaluation.h1_omega", proof.h1_omega_eval);
    absorb_fr(transcript, "plonk.evaluation.h2", proof.h2_eval);
    absorb_fr(transcript, "plonk.evaluation.public_input", proof.public_input_eval);
    absorb_fr(transcript, "plonk.evaluation.t", proof.t_eval);
    absorb_fr(transcript, "plonk.evaluation.r", proof.r_eval);
}

void absorb_plonk_opening_commitments(ProofTranscript& transcript, const plonkProof& proof)
{
    absorb_commitment(transcript, "plonk.commitment.w_zeta", proof.w_zeta_commitment);
    absorb_commitment(transcript, "plonk.commitment.w_zeta_omega", proof.w_zeta_omega_commitment);
}

std::pair<Fr, Fr> sample_plonk_permutation_challenges(ProofTranscript& transcript)
{
    return {transcript.squeeze_nonzero_fr("plonk.beta"), transcript.squeeze_nonzero_fr("plonk.gamma")};
}

Fr sample_plonk_lookup_compression_challenge(ProofTranscript& transcript)
{
    return transcript.squeeze_nonzero_fr("plonk.lookup.compression");
}

std::pair<Fr, Fr> sample_plonk_lookup_grand_product_challenges(ProofTranscript& transcript)
{
    return {transcript.squeeze_nonzero_fr("plonk.lookup.delta"),
            transcript.squeeze_nonzero_fr("plonk.lookup.epsilon")};
}

Fr sample_plonk_alpha(ProofTranscript& transcript)
{
    return transcript.squeeze_nonzero_fr("plonk.alpha");
}

Fr sample_plonk_zeta(ProofTranscript& transcript, size_t domain_size)
{
    for (size_t attempt = 0;; ++attempt) {
        std::string label = "plonk.zeta";
        if (attempt != 0) {
            label += ".domain_retry.";
            label += std::to_string(attempt);
        }
        const Fr zeta = transcript.squeeze_nonzero_fr(label);
        if (!((pow_fr(zeta, domain_size) - Fr(1)).isZero())) {
            return zeta;
        }
    }
}

std::pair<Fr, Fr> sample_plonk_opening_challenges(ProofTranscript& transcript)
{
    return {transcript.squeeze_nonzero_fr("plonk.v"), transcript.squeeze_nonzero_fr("plonk.u")};
}
