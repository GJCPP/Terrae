#pragma once

#include "plonk/preprocess.hpp"
#include "plonk/proof_io.hpp"
#include "plonk/witness.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

struct plonkProofBundle {
    SerializedProof proof;
    size_t public_input_count = 0;
    std::array<uint8_t, 32> verification_key_digest{};

    friend bool operator==(const plonkProofBundle& lhs, const plonkProofBundle& rhs)
    {
        return lhs.proof.bytes == rhs.proof.bytes &&
               lhs.public_input_count == rhs.public_input_count &&
               lhs.verification_key_digest == rhs.verification_key_digest;
    }
};

std::array<uint8_t, 32> plonk_verification_key_digest(const plonkVerificationKey& verification_key);

plonkProofBundle make_plonk_proof_bundle(const plonkVerificationKey& verification_key,
                                         const std::vector<Fr>& public_inputs,
                                         const plonkProof& proof);

SerializedProof serialize_plonk_proof_bundle(const plonkProofBundle& bundle);
plonkProofBundle deserialize_plonk_proof_bundle(const SerializedProof& serialized);

plonkProofBundle prove_plonk_bundle(const plonkProvingKey& proving_key,
                                    const plonkWitness& witness,
                                    const std::vector<Fr>& public_inputs);

bool verify_plonk_bundle(const plonkVerificationKey& verification_key,
                         const std::vector<Fr>& public_inputs,
                         const plonkProofBundle& bundle);

bool verify_plonk_serialized_bundle(const plonkVerificationKey& verification_key,
                                    const std::vector<Fr>& public_inputs,
                                    const SerializedProof& serialized);
