#pragma once

#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"

#include <vector>

poly blind_plonk_polynomial_on_subgroup(const poly& base,
                                        size_t subgroup_size,
                                        const std::vector<Fr>& blinding_coefficients);
