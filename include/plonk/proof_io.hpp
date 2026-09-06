#pragma once

#include "plonk/proof.hpp"
#include "proof_session.hpp"

SerializedProof serialize_plonk_proof(const plonkProof& proof);
plonkProof deserialize_plonk_proof(const SerializedProof& serialized);
