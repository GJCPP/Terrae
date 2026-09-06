#pragma once

#include "plonk/linearisation.hpp"
#include "plonk/proof.hpp"

struct plonkOpeningPolynomials {
    poly batched_zeta;
    poly batched_zeta_omega;
    poly w_zeta;
    poly w_zeta_omega;
    poly zeta_remainder;
    poly zeta_omega_remainder;
    Fr batched_zeta_eval = Fr(0);
    Fr batched_zeta_omega_eval = Fr(0);
    Fr zeta = Fr(0);
    Fr zeta_omega = Fr(0);
    bool zeta_divisible = false;
    bool zeta_omega_divisible = false;
};

Fr evaluate_plonk_batched_zeta_opening(const plonkProof& proof, const Fr& v);

plonkOpeningPolynomials build_plonk_opening_polynomials(const plonkProvingKey& proving_key,
                                                        const plonkQuotientPolynomials& quotient,
                                                        const plonkLinearisationPolynomial& linearised,
                                                        const Fr& v,
                                                        const Fr& zeta);
