#pragma once

#include "plonk/preprocess.hpp"
#include "plonk/witness.hpp"

#include <vector>

struct plonkLookupPolynomials {
    std::vector<Fr> f_evals;
    std::vector<Fr> table_evals;
    std::vector<Fr> h1_evals;
    std::vector<Fr> h2_evals;

    poly f;
    poly table;
    poly h1;
    poly h2;
};

Fr compress_plonk_lookup_row(const Fr& left,
                             const Fr& right,
                             const Fr& output,
                             const Fr& compression_challenge);

plonkLookupPolynomials build_plonk_lookup_polynomials(const plonkProvingKey& proving_key,
                                                      const plonkWireAssignments& wires,
                                                      const Fr& compression_challenge);

std::vector<Fr> build_plonk_lookup_grand_product_evaluations(const plonkLookupPolynomials& lookup,
                                                             const Fr& omega,
                                                             const Fr& delta,
                                                             const Fr& epsilon);
