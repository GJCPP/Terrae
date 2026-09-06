#include "plonk/opening.hpp"

#include "algebra_utils.hpp"

#include <stdexcept>

namespace {

poly linear_divisor(const Fr& point)
{
    return identity_poly() - point;
}

poly constant_poly(const Fr& value)
{
    poly out;
    out.add_term(0, value);
    return out;
}

std::vector<Fr> powers_of_v(const Fr& v, size_t count)
{
    std::vector<Fr> powers(count, Fr(1));
    for (size_t i = 1; i < count; ++i) {
        powers[i] = powers[i - 1] * v;
    }
    return powers;
}

} // namespace

Fr evaluate_plonk_batched_zeta_opening(const plonkProof& proof, const Fr& v)
{
    const std::vector<Fr> powers = powers_of_v(v, 11);
    return proof.t_eval + powers[1] * proof.r_eval + powers[2] * proof.a_eval +
           powers[3] * proof.b_eval + powers[4] * proof.c_eval +
           powers[5] * proof.sigma1_eval + powers[6] * proof.sigma2_eval +
           powers[7] * proof.f_eval + powers[8] * proof.h2_eval +
           powers[9] * proof.lookup_table_eval + powers[10] * proof.public_input_eval;
}

plonkOpeningPolynomials build_plonk_opening_polynomials(const plonkProvingKey& proving_key,
                                                        const plonkQuotientPolynomials& quotient,
                                                        const plonkLinearisationPolynomial& linearised,
                                                        const Fr& v,
                                                        const Fr& zeta)
{
    if (proving_key.n == 0) {
        throw std::runtime_error("build_plonk_opening_polynomials: empty domain");
    }

    plonkOpeningPolynomials out;
    const std::vector<Fr> powers = powers_of_v(v, 11);
    const Fr zeta_n = pow_fr(zeta, proving_key.n);
    const Fr zeta_2n = zeta_n * zeta_n;
    const poly quotient_at_zeta_chunks = quotient.t_lo + quotient.t_mid * zeta_n + quotient.t_hi * zeta_2n;
    out.batched_zeta = quotient_at_zeta_chunks + powers[1] * linearised.r + powers[2] * quotient.a +
                       powers[3] * quotient.b + powers[4] * quotient.c + powers[5] * proving_key.sigma1 +
                       powers[6] * proving_key.sigma2 + powers[7] * quotient.f +
                       powers[8] * quotient.h2 + powers[9] * quotient.lookup_table +
                       powers[10] * quotient.public_input;
    out.batched_zeta_eval = linearised.t_eval + powers[1] * linearised.r_eval +
                            powers[2] * linearised.a_eval + powers[3] * linearised.b_eval +
                            powers[4] * linearised.c_eval + powers[5] * linearised.sigma1_eval +
                            powers[6] * linearised.sigma2_eval + powers[7] * linearised.f_eval +
                            powers[8] * linearised.h2_eval + powers[9] * linearised.lookup_table_eval +
                            powers[10] * linearised.public_input_eval;

    const auto [w_zeta, zeta_remainder] =
        div(out.batched_zeta - constant_poly(out.batched_zeta_eval), linear_divisor(zeta));
    out.w_zeta = w_zeta;
    out.zeta_remainder = zeta_remainder;
    out.zeta_divisible = out.zeta_remainder.is_zero();

    out.zeta = zeta;
    out.zeta_omega = zeta * proving_key.omega;
    out.batched_zeta_omega =
        quotient.z + quotient.lookup_table * v + quotient.z_lookup * (v * v) +
        quotient.h1 * (v * v * v);
    out.batched_zeta_omega_eval =
        linearised.z_omega_eval + v * linearised.lookup_table_omega_eval +
        (v * v) * linearised.z_lookup_omega_eval + (v * v * v) * linearised.h1_omega_eval;
    const auto [w_zeta_omega, zeta_omega_remainder] =
        div(out.batched_zeta_omega - constant_poly(out.batched_zeta_omega_eval),
            linear_divisor(out.zeta_omega));
    out.w_zeta_omega = w_zeta_omega;
    out.zeta_omega_remainder = zeta_omega_remainder;
    out.zeta_omega_divisible = out.zeta_omega_remainder.is_zero();
    return out;
}
