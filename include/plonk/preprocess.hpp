#pragma once

#include "kzg.hpp"
#include "plonk/circuit.hpp"
#include "polynomial.hpp"

#include <vector>

struct plonkPolynomialCommitment {
    Group1 value;

    friend bool operator==(const plonkPolynomialCommitment& lhs, const plonkPolynomialCommitment& rhs)
    {
        return lhs.value == rhs.value;
    }
};

struct plonkProvingKey {
    plonkCircuit circuit;
    kzgPublicParameters public_parameters;
    size_t n = 0;
    Fr omega = Fr(0);
    Fr k1 = Fr(0);
    Fr k2 = Fr(0);

    std::vector<Fr> qM_evals;
    std::vector<Fr> qL_evals;
    std::vector<Fr> qR_evals;
    std::vector<Fr> qO_evals;
    std::vector<Fr> qC_evals;
    std::vector<Fr> qK_evals;
    std::vector<Fr> table_1_evals;
    std::vector<Fr> table_2_evals;
    std::vector<Fr> table_3_evals;
    std::vector<Fr> sigma1_evals;
    std::vector<Fr> sigma2_evals;
    std::vector<Fr> sigma3_evals;

    poly qM;
    poly qL;
    poly qR;
    poly qO;
    poly qC;
    poly qK;
    poly table_1;
    poly table_2;
    poly table_3;
    poly sigma1;
    poly sigma2;
    poly sigma3;

    plonkPolynomialCommitment qM_commitment;
    plonkPolynomialCommitment qL_commitment;
    plonkPolynomialCommitment qR_commitment;
    plonkPolynomialCommitment qO_commitment;
    plonkPolynomialCommitment qC_commitment;
    plonkPolynomialCommitment qK_commitment;
    plonkPolynomialCommitment table_1_commitment;
    plonkPolynomialCommitment table_2_commitment;
    plonkPolynomialCommitment table_3_commitment;
    plonkPolynomialCommitment sigma1_commitment;
    plonkPolynomialCommitment sigma2_commitment;
    plonkPolynomialCommitment sigma3_commitment;
};

struct plonkVerificationKey {
    kzgPublicParameters public_parameters;
    size_t n = 0;
    Fr omega = Fr(0);
    Fr k1 = Fr(0);
    Fr k2 = Fr(0);
    std::vector<size_t> public_input_rows;

    plonkPolynomialCommitment qM_commitment;
    plonkPolynomialCommitment qL_commitment;
    plonkPolynomialCommitment qR_commitment;
    plonkPolynomialCommitment qO_commitment;
    plonkPolynomialCommitment qC_commitment;
    plonkPolynomialCommitment qK_commitment;
    plonkPolynomialCommitment table_1_commitment;
    plonkPolynomialCommitment table_2_commitment;
    plonkPolynomialCommitment table_3_commitment;
    plonkPolynomialCommitment sigma1_commitment;
    plonkPolynomialCommitment sigma2_commitment;
    plonkPolynomialCommitment sigma3_commitment;
};

struct plonkSetup {
    plonkProvingKey proving_key;
    plonkVerificationKey verification_key;
};

Fr plonk_wire_label(size_t n, const Fr& omega, const Fr& k1, const Fr& k2, const plonkWireCell& cell);
plonkVerificationKey derive_plonk_verification_key(const plonkProvingKey& proving_key);
plonkSetup preprocess_plonk(const plonkCircuit& circuit, const kzgPublicParameters& public_parameters);
