#pragma once

#include "plonk/preprocess.hpp"
#include "plonk/witness.hpp"
#include "polynomial.hpp"

#include <vector>

struct plonkPermutationCheck {
    std::vector<Fr> z_evals;
    std::vector<Fr> residuals;
    poly z;
    bool satisfied = false;
};

std::vector<Fr> build_plonk_grand_product_evaluations(const plonkProvingKey& proving_key,
                                                      const plonkWireAssignments& wires,
                                                      const Fr& beta,
                                                      const Fr& gamma);

std::vector<Fr> evaluate_plonk_permutation_residuals(const plonkProvingKey& proving_key,
                                                     const plonkWireAssignments& wires,
                                                     const std::vector<Fr>& z_evals,
                                                     const Fr& beta,
                                                     const Fr& gamma);

plonkPermutationCheck check_plonk_permutation_argument(const plonkProvingKey& proving_key,
                                                       const plonkWireAssignments& wires,
                                                       const Fr& beta,
                                                       const Fr& gamma);
