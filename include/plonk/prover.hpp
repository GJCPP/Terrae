#pragma once

#include "plonk/opening.hpp"
#include "plonk/proof_io.hpp"
#include "plonk/proof.hpp"
#include "plonk/witness.hpp"

#include <vector>

struct plonkDeferredOpeningProof {
    plonkProof proof;
    plonkOpeningPolynomials openings;
};

plonkProof prove_plonk(const plonkProvingKey& proving_key,
                       const plonkWitness& witness,
                       const std::vector<Fr>& public_inputs);

plonkDeferredOpeningProof prove_plonk_deferred_opening(
    const plonkProvingKey& proving_key,
    const plonkWitness& witness,
    const std::vector<Fr>& public_inputs);

SerializedProof prove_plonk_serialized(const plonkProvingKey& proving_key,
                                       const plonkWitness& witness,
                                       const std::vector<Fr>& public_inputs);
