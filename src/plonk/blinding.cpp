#include "plonk/blinding.hpp"

#include <stdexcept>

poly blind_plonk_polynomial_on_subgroup(const poly& base,
                                        size_t subgroup_size,
                                        const std::vector<Fr>& blinding_coefficients)
{
    if (!isPowerOfTwo(subgroup_size)) {
        throw std::runtime_error("blind_plonk_polynomial_on_subgroup: subgroup_size must be a power of two");
    }
    if (blinding_coefficients.empty()) {
        return base;
    }
    return base + vanishing_poly(subgroup_size) * poly_from_coeff(blinding_coefficients);
}
