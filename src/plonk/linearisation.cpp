#include "plonk/linearisation.hpp"

#include "algebra_utils.hpp"
#include "lagrange_utils.hpp"

namespace {

Fr shifted_value(const Fr& value, const Fr& beta, const Fr& label, const Fr& gamma)
{
    Fr out;
    Fr::mul(out, beta, label);
    Fr::add(out, out, value);
    Fr::add(out, out, gamma);
    return out;
}

poly scaled_t_chunks(const plonkQuotientPolynomials& quotient,
                     size_t domain_size,
                     const Fr& zeta)
{
    const Fr zeta_n = pow_fr(zeta, domain_size);
    const Fr zeta_2n = zeta_n * zeta_n;
    return quotient.t_lo + quotient.t_mid * zeta_n + quotient.t_hi * zeta_2n;
}

} // namespace

plonkLinearisationPolynomial build_plonk_linearisation_polynomial(const plonkProvingKey& proving_key,
                                                                  const plonkQuotientPolynomials& quotient,
                                                                  const Fr& lookup_compression_challenge,
                                                                  const Fr& beta,
                                                                  const Fr& gamma,
                                                                  const Fr& delta,
                                                                  const Fr& epsilon,
                                                                  const Fr& alpha,
                                                                  const Fr& zeta)
{
    plonkLinearisationPolynomial out;
    out.a_eval = quotient.a.eval(zeta);
    out.b_eval = quotient.b.eval(zeta);
    out.c_eval = quotient.c.eval(zeta);
    out.sigma1_eval = proving_key.sigma1.eval(zeta);
    out.sigma2_eval = proving_key.sigma2.eval(zeta);
    out.f_eval = quotient.f.eval(zeta);
    out.lookup_table_eval = quotient.lookup_table.eval(zeta);
    out.lookup_table_omega_eval = quotient.lookup_table.eval(zeta * proving_key.omega);
    out.z_eval = quotient.z.eval(zeta);
    out.z_omega_eval = quotient.z.eval(zeta * proving_key.omega);
    out.z_lookup_eval = quotient.z_lookup.eval(zeta);
    out.z_lookup_omega_eval = quotient.z_lookup.eval(zeta * proving_key.omega);
    out.h1_eval = quotient.h1.eval(zeta);
    out.h1_omega_eval = quotient.h1.eval(zeta * proving_key.omega);
    out.h2_eval = quotient.h2.eval(zeta);
    out.public_input_eval = quotient.public_input.eval(zeta);
    out.z_h_eval = vanishing_poly(proving_key.n).eval(zeta);
    out.l1_eval = subgroup_lagrange_polynomial(proving_key.n, 0).eval(zeta);
    out.t_eval = quotient.quotient.eval(zeta);

    const Fr left_identity = shifted_value(out.a_eval, beta, zeta, gamma);
    const Fr right_identity = shifted_value(out.b_eval, beta, proving_key.k1 * zeta, gamma);
    const Fr output_identity = shifted_value(out.c_eval, beta, proving_key.k2 * zeta, gamma);
    const Fr left_sigma = shifted_value(out.a_eval, beta, out.sigma1_eval, gamma);
    const Fr right_sigma = shifted_value(out.b_eval, beta, out.sigma2_eval, gamma);

    Fr identity_product;
    Fr::mul(identity_product, left_identity, right_identity);
    Fr::mul(identity_product, identity_product, output_identity);

    Fr sigma_prefix;
    Fr::mul(sigma_prefix, left_sigma, right_sigma);
    Fr::mul(sigma_prefix, sigma_prefix, out.z_omega_eval);

    const poly gate_linearisation =
        proving_key.qM * (out.a_eval * out.b_eval) + proving_key.qL * out.a_eval +
        proving_key.qR * out.b_eval + proving_key.qO * out.c_eval + proving_key.qC +
        out.public_input_eval;

    const poly sigma3_factor = proving_key.sigma3 * beta + (out.c_eval + gamma);
    const poly permutation_linearisation =
        quotient.z * identity_product - sigma3_factor * sigma_prefix;

    const poly boundary_linearisation =
        (quotient.z - Fr(1)) * out.l1_eval;

    const Fr lookup_compression_squared =
        lookup_compression_challenge * lookup_compression_challenge;
    const poly lookup_query_linearisation =
        proving_key.qK * (out.a_eval + lookup_compression_challenge * out.b_eval +
                          lookup_compression_squared * out.c_eval - out.f_eval);

    const Fr lookup_numerator_factor = (Fr(1) + delta) * (epsilon + out.f_eval) *
                                       (epsilon * (Fr(1) + delta) + out.lookup_table_eval +
                                        delta * out.lookup_table_omega_eval);
    const poly lookup_denominator_linearisation =
        (quotient.h1 + delta * quotient.h2 + epsilon * (Fr(1) + delta)) *
        (epsilon * (Fr(1) + delta) + out.h2_eval + delta * out.h1_omega_eval);
    const poly lookup_grand_product_linearisation =
        quotient.z_lookup * lookup_numerator_factor -
        lookup_denominator_linearisation * out.z_lookup_omega_eval;

    const poly lookup_boundary_linearisation =
        (quotient.z_lookup - Fr(1)) * out.l1_eval;

    out.r = gate_linearisation + permutation_linearisation * alpha +
            boundary_linearisation * (alpha * alpha) +
            lookup_query_linearisation * (alpha * alpha * alpha) +
            lookup_grand_product_linearisation * (alpha * alpha * alpha * alpha) +
            lookup_boundary_linearisation * (alpha * alpha * alpha * alpha * alpha) -
            scaled_t_chunks(quotient, proving_key.n, zeta) * out.z_h_eval;
    out.r_eval = out.r.eval(zeta);
    return out;
}
