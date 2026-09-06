#pragma once

#include "plonk/lookup.hpp"
#include "plonk/preprocess.hpp"
#include "plonk/witness.hpp"

#include <vector>

struct plonkGateConstraintCheck {
    std::vector<Fr> public_input_evals;
    std::vector<Fr> residuals;
    bool satisfied = false;
};

struct plonkQuotientPolynomials {
    std::vector<Fr> public_input_evals;
    poly public_input;
    poly a;
    poly b;
    poly c;
    poly z;
    poly f;
    poly lookup_table;
    poly h1;
    poly h2;
    poly z_lookup;
    poly n_gate;
    poly n_permutation;
    poly n_boundary;
    poly n_lookup_query;
    poly n_lookup_permutation;
    poly n_lookup_boundary;
    poly numerator;
    poly quotient;
    poly remainder;
    poly t_lo;
    poly t_mid;
    poly t_hi;
    bool divisible = false;
};

std::vector<Fr> build_plonk_public_input_evaluations(const plonkCircuit& circuit,
                                                     size_t domain_size,
                                                     const std::vector<Fr>& public_inputs);

std::vector<Fr> evaluate_plonk_gate_constraints(const plonkProvingKey& proving_key,
                                                const plonkWireAssignments& wires,
                                                const std::vector<Fr>& public_inputs);

plonkGateConstraintCheck check_plonk_gate_constraints(const plonkProvingKey& proving_key,
                                                      const plonkWireAssignments& wires,
                                                      const std::vector<Fr>& public_inputs);

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
                                                          const Fr& alpha);
