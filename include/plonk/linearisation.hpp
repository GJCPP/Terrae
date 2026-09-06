#pragma once

#include "plonk/quotient.hpp"

struct plonkLinearisationPolynomial {
    poly r;
    Fr a_eval = Fr(0);
    Fr b_eval = Fr(0);
    Fr c_eval = Fr(0);
    Fr sigma1_eval = Fr(0);
    Fr sigma2_eval = Fr(0);
    Fr f_eval = Fr(0);
    Fr lookup_table_eval = Fr(0);
    Fr lookup_table_omega_eval = Fr(0);
    Fr z_eval = Fr(0);
    Fr z_omega_eval = Fr(0);
    Fr z_lookup_eval = Fr(0);
    Fr z_lookup_omega_eval = Fr(0);
    Fr h1_eval = Fr(0);
    Fr h1_omega_eval = Fr(0);
    Fr h2_eval = Fr(0);
    Fr public_input_eval = Fr(0);
    Fr z_h_eval = Fr(0);
    Fr l1_eval = Fr(0);
    Fr t_eval = Fr(0);
    Fr r_eval = Fr(0);
};

plonkLinearisationPolynomial build_plonk_linearisation_polynomial(const plonkProvingKey& proving_key,
                                                                  const plonkQuotientPolynomials& quotient,
                                                                  const Fr& lookup_compression_challenge,
                                                                  const Fr& beta,
                                                                  const Fr& gamma,
                                                                  const Fr& delta,
                                                                  const Fr& epsilon,
                                                                  const Fr& alpha,
                                                                  const Fr& zeta);
