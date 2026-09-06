#include "plonk/permutation.hpp"

#include <stdexcept>
#include <string>

namespace {

Fr shifted_wire_value(const Fr& value, const Fr& beta, const Fr& label, const Fr& gamma)
{
    Fr out;
    Fr::mul(out, beta, label);
    Fr::add(out, out, value);
    Fr::add(out, out, gamma);
    return out;
}

Fr product3(const Fr& a, const Fr& b, const Fr& c)
{
    Fr out;
    Fr::mul(out, a, b);
    Fr::mul(out, out, c);
    return out;
}

Fr identity_product(const plonkProvingKey& proving_key,
                    const plonkWireAssignments& wires,
                    size_t row,
                    const Fr& beta,
                    const Fr& gamma)
{
    const Fr left_label = plonk_wire_label(proving_key.n,
                                           proving_key.omega,
                                           proving_key.k1,
                                           proving_key.k2,
                                           {row, plonkWireColumn::Left});
    const Fr right_label = plonk_wire_label(proving_key.n,
                                            proving_key.omega,
                                            proving_key.k1,
                                            proving_key.k2,
                                            {row, plonkWireColumn::Right});
    const Fr output_label = plonk_wire_label(proving_key.n,
                                             proving_key.omega,
                                             proving_key.k1,
                                             proving_key.k2,
                                             {row, plonkWireColumn::Output});
    return product3(shifted_wire_value(wires.a_evals[row], beta, left_label, gamma),
                    shifted_wire_value(wires.b_evals[row], beta, right_label, gamma),
                    shifted_wire_value(wires.c_evals[row], beta, output_label, gamma));
}

Fr sigma_product(const plonkProvingKey& proving_key,
                 const plonkWireAssignments& wires,
                 size_t row,
                 const Fr& beta,
                 const Fr& gamma)
{
    return product3(shifted_wire_value(wires.a_evals[row], beta, proving_key.sigma1_evals[row], gamma),
                    shifted_wire_value(wires.b_evals[row], beta, proving_key.sigma2_evals[row], gamma),
                    shifted_wire_value(wires.c_evals[row], beta, proving_key.sigma3_evals[row], gamma));
}

void validate_shape(const plonkProvingKey& proving_key, const plonkWireAssignments& wires, const char* context)
{
    if (proving_key.n == 0 || wires.a_evals.size() != proving_key.n || wires.b_evals.size() != proving_key.n ||
        wires.c_evals.size() != proving_key.n || proving_key.sigma1_evals.size() != proving_key.n ||
        proving_key.sigma2_evals.size() != proving_key.n || proving_key.sigma3_evals.size() != proving_key.n) {
        throw std::runtime_error(std::string(context) + ": invalid domain shape");
    }
}

} // namespace

std::vector<Fr> build_plonk_grand_product_evaluations(const plonkProvingKey& proving_key,
                                                      const plonkWireAssignments& wires,
                                                      const Fr& beta,
                                                      const Fr& gamma)
{
    validate_shape(proving_key, wires, "build_plonk_grand_product_evaluations");
    std::vector<Fr> z_evals(proving_key.n, Fr(1));
    for (size_t row = 0; row + 1 < proving_key.n; ++row) {
        const Fr numerator = identity_product(proving_key, wires, row, beta, gamma);
        const Fr denominator = sigma_product(proving_key, wires, row, beta, gamma);
        if (denominator.isZero()) {
            throw std::runtime_error("build_plonk_grand_product_evaluations: zero permutation denominator");
        }
        Fr ratio;
        Fr::div(ratio, numerator, denominator);
        Fr::mul(z_evals[row + 1], z_evals[row], ratio);
    }
    return z_evals;
}

std::vector<Fr> evaluate_plonk_permutation_residuals(const plonkProvingKey& proving_key,
                                                     const plonkWireAssignments& wires,
                                                     const std::vector<Fr>& z_evals,
                                                     const Fr& beta,
                                                     const Fr& gamma)
{
    validate_shape(proving_key, wires, "evaluate_plonk_permutation_residuals");
    if (z_evals.size() != proving_key.n) {
        throw std::runtime_error("evaluate_plonk_permutation_residuals: invalid z shape");
    }
    std::vector<Fr> residuals(proving_key.n, Fr(0));
    for (size_t row = 0; row < proving_key.n; ++row) {
        Fr lhs;
        Fr::mul(lhs, z_evals[(row + 1) % proving_key.n], sigma_product(proving_key, wires, row, beta, gamma));
        Fr rhs;
        Fr::mul(rhs, z_evals[row], identity_product(proving_key, wires, row, beta, gamma));
        Fr::sub(residuals[row], lhs, rhs);
    }
    return residuals;
}

plonkPermutationCheck check_plonk_permutation_argument(const plonkProvingKey& proving_key,
                                                       const plonkWireAssignments& wires,
                                                       const Fr& beta,
                                                       const Fr& gamma)
{
    plonkPermutationCheck check;
    check.z_evals = build_plonk_grand_product_evaluations(proving_key, wires, beta, gamma);
    check.residuals = evaluate_plonk_permutation_residuals(proving_key, wires, check.z_evals, beta, gamma);
    check.z = poly_from_evals(check.z_evals);
    check.satisfied = true;
    for (const Fr& residual : check.residuals) {
        if (!residual.isZero()) {
            check.satisfied = false;
            break;
        }
    }
    return check;
}
