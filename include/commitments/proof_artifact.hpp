#pragma once

#include <cstdint>
#include <vector>

#include "proof_session.hpp"

class ProofArtifact {
public:
    ProofArtifact() = default;

    explicit ProofArtifact(SerializedProof proof)
        : proof(std::move(proof))
    {
    }

    SerializedProof proof;
};

class VerificationResult {
public:
    VerificationResult() = default;

    VerificationResult(bool accepted, bool proof_fully_consumed, std::vector<uint8_t> transcript_state)
        : accepted(accepted)
        , proof_fully_consumed(proof_fully_consumed)
        , transcript_state(std::move(transcript_state))
    {
    }

    bool accepted = false;
    bool proof_fully_consumed = false;
    std::vector<uint8_t> transcript_state;
};

inline ProofArtifact make_proof_artifact(SerializedProof proof)
{
    return ProofArtifact(std::move(proof));
}
