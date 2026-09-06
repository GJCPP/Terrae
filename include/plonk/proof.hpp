#pragma once

#include "plonk/preprocess.hpp"

struct plonkProof {
    plonkPolynomialCommitment a_commitment;
    plonkPolynomialCommitment b_commitment;
    plonkPolynomialCommitment c_commitment;
    plonkPolynomialCommitment f_commitment;
    plonkPolynomialCommitment h1_commitment;
    plonkPolynomialCommitment h2_commitment;
    plonkPolynomialCommitment z_commitment;
    plonkPolynomialCommitment z_lookup_commitment;
    plonkPolynomialCommitment t_lo_commitment;
    plonkPolynomialCommitment t_mid_commitment;
    plonkPolynomialCommitment t_hi_commitment;
    plonkPolynomialCommitment w_zeta_commitment;
    plonkPolynomialCommitment w_zeta_omega_commitment;

    Fr a_eval = Fr(0);
    Fr b_eval = Fr(0);
    Fr c_eval = Fr(0);
    Fr sigma1_eval = Fr(0);
    Fr sigma2_eval = Fr(0);
    Fr f_eval = Fr(0);
    Fr lookup_table_eval = Fr(0);
    Fr lookup_table_omega_eval = Fr(0);
    Fr z_omega_eval = Fr(0);
    Fr z_lookup_omega_eval = Fr(0);
    Fr h1_omega_eval = Fr(0);
    Fr h2_eval = Fr(0);
    Fr public_input_eval = Fr(0);
    Fr t_eval = Fr(0);
    Fr r_eval = Fr(0);

    friend bool operator==(const plonkProof& lhs, const plonkProof& rhs)
    {
        return lhs.a_commitment == rhs.a_commitment &&
               lhs.b_commitment == rhs.b_commitment &&
               lhs.c_commitment == rhs.c_commitment &&
               lhs.f_commitment == rhs.f_commitment &&
               lhs.h1_commitment == rhs.h1_commitment &&
               lhs.h2_commitment == rhs.h2_commitment &&
               lhs.z_commitment == rhs.z_commitment &&
               lhs.z_lookup_commitment == rhs.z_lookup_commitment &&
               lhs.t_lo_commitment == rhs.t_lo_commitment &&
               lhs.t_mid_commitment == rhs.t_mid_commitment &&
               lhs.t_hi_commitment == rhs.t_hi_commitment &&
               lhs.w_zeta_commitment == rhs.w_zeta_commitment &&
               lhs.w_zeta_omega_commitment == rhs.w_zeta_omega_commitment &&
               lhs.a_eval == rhs.a_eval &&
               lhs.b_eval == rhs.b_eval &&
               lhs.c_eval == rhs.c_eval &&
               lhs.sigma1_eval == rhs.sigma1_eval &&
               lhs.sigma2_eval == rhs.sigma2_eval &&
               lhs.f_eval == rhs.f_eval &&
               lhs.lookup_table_eval == rhs.lookup_table_eval &&
               lhs.lookup_table_omega_eval == rhs.lookup_table_omega_eval &&
               lhs.z_omega_eval == rhs.z_omega_eval &&
               lhs.z_lookup_omega_eval == rhs.z_lookup_omega_eval &&
               lhs.h1_omega_eval == rhs.h1_omega_eval &&
               lhs.h2_eval == rhs.h2_eval &&
               lhs.public_input_eval == rhs.public_input_eval &&
               lhs.t_eval == rhs.t_eval &&
               lhs.r_eval == rhs.r_eval;
    }
};
