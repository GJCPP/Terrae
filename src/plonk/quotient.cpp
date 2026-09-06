#include "plonk/quotient.hpp"

#include "lagrange_utils.hpp"

#include <stdexcept>
#include <string>

namespace {

void validate_quotient_shape(const plonkProvingKey& proving_key,
                             const plonkWireAssignments& wires,
                             const std::vector<Fr>& z_evals,
                             const plonkLookupPolynomials& lookup,
                             const std::vector<Fr>& z_lookup_evals,
                             const char* context)
{
    if (proving_key.n == 0 || wires.a_evals.size() != proving_key.n || wires.b_evals.size() != proving_key.n ||
        wires.c_evals.size() != proving_key.n || z_evals.size() != proving_key.n ||
        lookup.f_evals.size() != proving_key.n || lookup.table_evals.size() != proving_key.n ||
        lookup.h1_evals.size() != proving_key.n || lookup.h2_evals.size() != proving_key.n ||
        z_lookup_evals.size() != proving_key.n) {
        throw std::runtime_error(std::string(context) + ": invalid domain shape");
    }
}

poly linear_term(const Fr& slope, const Fr& intercept)
{
    return identity_poly() * slope + intercept;
}

poly split_quotient_chunk(const poly& quotient, size_t start, size_t count)
{
    const std::vector<Fr>& coeffs = quotient.coefficients();
    std::vector<Fr> out(count, Fr(0));
    for (size_t i = 0; i < count && start + i < coeffs.size(); ++i) {
        out[i] = coeffs[start + i];
    }
    return poly_from_coeff(out);
}

} // namespace

std::vector<Fr> build_plonk_public_input_evaluations(const plonkCircuit& circuit,
                                                     size_t domain_size,
                                                     const std::vector<Fr>& public_inputs)
{
    if (public_inputs.size() != circuit.public_input_rows.size()) {
        throw std::runtime_error("build_plonk_public_input_evaluations: public input count mismatch");
    }
    std::vector<Fr> pi(domain_size, Fr(0));
    for (size_t i = 0; i < public_inputs.size(); ++i) {
        const size_t row = circuit.public_input_rows[i];
        if (row >= domain_size) {
            throw std::runtime_error("build_plonk_public_input_evaluations: public input row out of range");
        }
        Fr neg;
        Fr::neg(neg, public_inputs[i]);
        Fr::add(pi[row], pi[row], neg);
    }
    return pi;
}

std::vector<Fr> evaluate_plonk_gate_constraints(const plonkProvingKey& proving_key,
                                                const plonkWireAssignments& wires,
                                                const std::vector<Fr>& public_inputs)
{
    if (wires.a_evals.size() != proving_key.n || wires.b_evals.size() != proving_key.n ||
        wires.c_evals.size() != proving_key.n) {
        throw std::runtime_error("evaluate_plonk_gate_constraints: invalid wire assignment shape");
    }
    const std::vector<Fr> pi =
        build_plonk_public_input_evaluations(proving_key.circuit, proving_key.n, public_inputs);
    std::vector<Fr> residuals(proving_key.n, Fr(0));
    for (size_t row = 0; row < proving_key.n; ++row) {
        Fr acc = proving_key.qC_evals[row];
        Fr term;
        Fr::mul(term, proving_key.qM_evals[row], wires.a_evals[row]);
        Fr::mul(term, term, wires.b_evals[row]);
        Fr::add(acc, acc, term);
        Fr::mul(term, proving_key.qL_evals[row], wires.a_evals[row]);
        Fr::add(acc, acc, term);
        Fr::mul(term, proving_key.qR_evals[row], wires.b_evals[row]);
        Fr::add(acc, acc, term);
        Fr::mul(term, proving_key.qO_evals[row], wires.c_evals[row]);
        Fr::add(acc, acc, term);
        Fr::add(acc, acc, pi[row]);
        residuals[row] = acc;
    }
    return residuals;
}

plonkGateConstraintCheck check_plonk_gate_constraints(const plonkProvingKey& proving_key,
                                                      const plonkWireAssignments& wires,
                                                      const std::vector<Fr>& public_inputs)
{
    plonkGateConstraintCheck check;
    check.public_input_evals =
        build_plonk_public_input_evaluations(proving_key.circuit, proving_key.n, public_inputs);
    check.residuals = evaluate_plonk_gate_constraints(proving_key, wires, public_inputs);
    check.satisfied = true;
    for (const Fr& residual : check.residuals) {
        if (!residual.isZero()) {
            check.satisfied = false;
            break;
        }
    }
    return check;
}

plonkQuotientPolynomials build_plonk_quotient_polynomials(const plonkProvingKey& proving_key,
                                                          const plonkWireAssignments& wires,
                                                          const std::vector<Fr>& public_inputs,
                                                          const std::vector<Fr>& z_evals,
                                                          const plonkLookupPolynomials& lookup,
                                                          const std::vector<Fr>& z_lookup_evals,
                                                          const Fr& lookup_compression_challenge,
                                                          const Fr& beta,
                                                          const Fr& gamma,
                                                          const Fr& delta,
                                                          const Fr& epsilon,
                                                          const Fr& alpha)
{
    validate_quotient_shape(proving_key, wires, z_evals, lookup, z_lookup_evals, "build_plonk_quotient_polynomials");

    plonkQuotientPolynomials out;
    out.public_input_evals =
        build_plonk_public_input_evaluations(proving_key.circuit, proving_key.n, public_inputs);
    out.public_input = poly_from_evals(out.public_input_evals);
    out.a = poly_from_evals(wires.a_evals);
    out.b = poly_from_evals(wires.b_evals);
    out.c = poly_from_evals(wires.c_evals);
    out.z = poly_from_evals(z_evals);
    out.f = lookup.f;
    out.lookup_table = lookup.table;
    out.h1 = lookup.h1;
    out.h2 = lookup.h2;
    out.z_lookup = poly_from_evals(z_lookup_evals);

    out.n_gate = proving_key.qM * out.a * out.b + proving_key.qL * out.a + proving_key.qR * out.b +
                 proving_key.qO * out.c + proving_key.qC + out.public_input;

    const poly left_identity = linear_term(beta, gamma);
    const poly right_identity = linear_term(beta * proving_key.k1, gamma);
    const poly output_identity = linear_term(beta * proving_key.k2, gamma);
    const poly left_sigma = out.a + proving_key.sigma1 * beta + gamma;
    const poly right_sigma = out.b + proving_key.sigma2 * beta + gamma;
    const poly output_sigma = out.c + proving_key.sigma3 * beta + gamma;
    const poly z_omega = out.z.scale_argument(proving_key.omega);

    out.n_permutation = out.z * (out.a + left_identity) * (out.b + right_identity) * (out.c + output_identity) -
                        z_omega * left_sigma * right_sigma * output_sigma;

    const poly l1 = subgroup_lagrange_polynomial(proving_key.n, 0);
    out.n_boundary = l1 * (out.z - Fr(1));

    const poly lookup_table_omega = out.lookup_table.scale_argument(proving_key.omega);
    const poly h1_omega = out.h1.scale_argument(proving_key.omega);
    const poly z_lookup_omega = out.z_lookup.scale_argument(proving_key.omega);
    const Fr lookup_compression_squared =
        lookup_compression_challenge * lookup_compression_challenge;

    out.n_lookup_query = proving_key.qK *
                         (out.a + out.b * lookup_compression_challenge +
                          out.c * lookup_compression_squared - out.f);

    out.n_lookup_permutation =
        out.z_lookup * ((Fr(1) + delta) * (out.f + epsilon) *
                        (lookup_table_omega * delta + out.lookup_table + epsilon * (Fr(1) + delta))) -
        z_lookup_omega * ((out.h1 + delta * out.h2 + epsilon * (Fr(1) + delta)) *
                          (out.h2 + delta * h1_omega + epsilon * (Fr(1) + delta)));

    out.n_lookup_boundary = l1 * (out.z_lookup - Fr(1));

    out.numerator = out.n_gate + out.n_permutation * alpha + out.n_boundary * (alpha * alpha) +
                    out.n_lookup_query * (alpha * alpha * alpha) +
                    out.n_lookup_permutation * (alpha * alpha * alpha * alpha) +
                    out.n_lookup_boundary * (alpha * alpha * alpha * alpha * alpha);

    const auto [quotient, remainder] = div(out.numerator, vanishing_poly(proving_key.n));
    out.quotient = quotient;
    out.remainder = remainder;
    out.divisible = out.remainder.is_zero();

    const std::vector<Fr>& coeffs = out.quotient.coefficients();
    if (coeffs.size() > 3 * proving_key.n) {
        throw std::runtime_error("build_plonk_quotient_polynomials: quotient does not fit into three n-sized chunks");
    }
    out.t_lo = split_quotient_chunk(out.quotient, 0, proving_key.n);
    out.t_mid = split_quotient_chunk(out.quotient, proving_key.n, proving_key.n);
    out.t_hi = split_quotient_chunk(out.quotient, 2 * proving_key.n, proving_key.n);

    return out;
}
