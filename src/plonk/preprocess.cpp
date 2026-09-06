#include "plonk/preprocess.hpp"

#include "algebra_utils.hpp"

#include <set>
#include <stdexcept>
#include <string>

namespace {

Fr domain_label(const Fr& omega, size_t row)
{
    return pow_fr(omega, row);
}

bool cosets_are_disjoint(size_t n, const Fr& omega, const Fr& k1, const Fr& k2)
{
    std::set<Fr> labels;
    for (size_t row = 0; row < n; ++row) {
        const Fr x = domain_label(omega, row);
        if (!labels.insert(x).second) {
            return false;
        }
        if (!labels.insert(k1 * x).second) {
            return false;
        }
        if (!labels.insert(k2 * x).second) {
            return false;
        }
    }
    return true;
}

std::pair<Fr, Fr> choose_coset_generators(size_t n, const Fr& omega)
{
    for (int first = 2; first < 512; ++first) {
        for (int second = first + 1; second < 512; ++second) {
            const Fr k1 = fr_from_int(first);
            const Fr k2 = fr_from_int(second);
            if (cosets_are_disjoint(n, omega, k1, k2)) {
                return {k1, k2};
            }
        }
    }
    throw std::runtime_error("choose_coset_generators: failed to find disjoint cosets");
}

plonkPolynomialCommitment commit_poly(const kzgPublicParameters& public_parameters, const poly& polynomial)
{
    return plonkPolynomialCommitment{commit_g1_with_public_parameters(public_parameters, polynomial)};
}

void add_cell(std::vector<std::vector<plonkWireCell>>& groups,
              const plonkCircuit& circuit,
              plonkVariable variable,
              const plonkWireCell& cell)
{
    const size_t root = circuit.canonical_variable(variable);
    if (root >= groups.size()) {
        throw std::runtime_error("preprocess_plonk: canonical variable root out of range");
    }
    groups[root].push_back(cell);
}

void set_sigma_eval(std::vector<Fr>& sigma1,
                    std::vector<Fr>& sigma2,
                    std::vector<Fr>& sigma3,
                    const plonkWireCell& cell,
                    const Fr& value)
{
    switch (cell.column) {
    case plonkWireColumn::Left:
        sigma1[cell.row] = value;
        return;
    case plonkWireColumn::Right:
        sigma2[cell.row] = value;
        return;
    case plonkWireColumn::Output:
        sigma3[cell.row] = value;
        return;
    }
    throw std::runtime_error("set_sigma_eval: unknown wire column");
}

} // namespace

Fr plonk_wire_label(size_t n, const Fr& omega, const Fr& k1, const Fr& k2, const plonkWireCell& cell)
{
    if (cell.row >= n) {
        throw std::runtime_error("plonk_wire_label: row out of range");
    }
    const Fr x = domain_label(omega, cell.row);
    switch (cell.column) {
    case plonkWireColumn::Left:
        return x;
    case plonkWireColumn::Right:
        return k1 * x;
    case plonkWireColumn::Output:
        return k2 * x;
    }
    throw std::runtime_error("plonk_wire_label: unknown wire column");
}

plonkVerificationKey derive_plonk_verification_key(const plonkProvingKey& proving_key)
{
    plonkVerificationKey verification_key;
    verification_key.public_parameters = proving_key.public_parameters;
    verification_key.n = proving_key.n;
    verification_key.omega = proving_key.omega;
    verification_key.k1 = proving_key.k1;
    verification_key.k2 = proving_key.k2;
    verification_key.public_input_rows = proving_key.circuit.public_input_rows;
    verification_key.qM_commitment = proving_key.qM_commitment;
    verification_key.qL_commitment = proving_key.qL_commitment;
    verification_key.qR_commitment = proving_key.qR_commitment;
    verification_key.qO_commitment = proving_key.qO_commitment;
    verification_key.qC_commitment = proving_key.qC_commitment;
    verification_key.qK_commitment = proving_key.qK_commitment;
    verification_key.table_1_commitment = proving_key.table_1_commitment;
    verification_key.table_2_commitment = proving_key.table_2_commitment;
    verification_key.table_3_commitment = proving_key.table_3_commitment;
    verification_key.sigma1_commitment = proving_key.sigma1_commitment;
    verification_key.sigma2_commitment = proving_key.sigma2_commitment;
    verification_key.sigma3_commitment = proving_key.sigma3_commitment;
    return verification_key;
}

plonkSetup preprocess_plonk(const plonkCircuit& circuit, const kzgPublicParameters& public_parameters)
{
    plonkSetup setup;
    plonkProvingKey& pk = setup.proving_key;
    pk.circuit = circuit;
    pk.public_parameters = public_parameters;
    pk.n = circuit.domain_size();
    pk.omega = get_root_of_unity(pk.n);
    const auto [k1, k2] = choose_coset_generators(pk.n, pk.omega);
    pk.k1 = k1;
    pk.k2 = k2;

    pk.qM_evals.assign(pk.n, Fr(0));
    pk.qL_evals.assign(pk.n, Fr(0));
    pk.qR_evals.assign(pk.n, Fr(0));
    pk.qO_evals.assign(pk.n, Fr(0));
    pk.qC_evals.assign(pk.n, Fr(0));
    pk.qK_evals.assign(pk.n, Fr(0));
    pk.table_1_evals.assign(pk.n, Fr(0));
    pk.table_2_evals.assign(pk.n, Fr(0));
    pk.table_3_evals.assign(pk.n, Fr(0));

    for (size_t row = 0; row < circuit.gates.size(); ++row) {
        const plonkGate& gate = circuit.gates[row];
        pk.qM_evals[row] = gate.qM;
        pk.qL_evals[row] = gate.qL;
        pk.qR_evals[row] = gate.qR;
        pk.qO_evals[row] = gate.qO;
        pk.qC_evals[row] = gate.qC;
        pk.qK_evals[row] = gate.qK;
    }

    for (size_t row = 0; row < circuit.lookup_table_rows.size(); ++row) {
        const plonkLookupTableRow& table_row = circuit.lookup_table_rows[row];
        pk.table_1_evals[row] = table_row.left;
        pk.table_2_evals[row] = table_row.right;
        pk.table_3_evals[row] = table_row.output;
    }

    pk.qM = poly_from_evals(pk.qM_evals);
    pk.qL = poly_from_evals(pk.qL_evals);
    pk.qR = poly_from_evals(pk.qR_evals);
    pk.qO = poly_from_evals(pk.qO_evals);
    pk.qC = poly_from_evals(pk.qC_evals);
    pk.qK = poly_from_evals(pk.qK_evals);
    pk.table_1 = poly_from_evals(pk.table_1_evals);
    pk.table_2 = poly_from_evals(pk.table_2_evals);
    pk.table_3 = poly_from_evals(pk.table_3_evals);

    pk.sigma1_evals.resize(pk.n);
    pk.sigma2_evals.resize(pk.n);
    pk.sigma3_evals.resize(pk.n);
    for (size_t row = 0; row < pk.n; ++row) {
        pk.sigma1_evals[row] = plonk_wire_label(pk.n, pk.omega, pk.k1, pk.k2, {row, plonkWireColumn::Left});
        pk.sigma2_evals[row] = plonk_wire_label(pk.n, pk.omega, pk.k1, pk.k2, {row, plonkWireColumn::Right});
        pk.sigma3_evals[row] = plonk_wire_label(pk.n, pk.omega, pk.k1, pk.k2, {row, plonkWireColumn::Output});
    }

    std::vector<std::vector<plonkWireCell>> groups(circuit.variable_count);
    for (size_t row = 0; row < circuit.gates.size(); ++row) {
        const plonkGate& gate = circuit.gates[row];
        add_cell(groups, circuit, gate.left, {row, plonkWireColumn::Left});
        add_cell(groups, circuit, gate.right, {row, plonkWireColumn::Right});
        add_cell(groups, circuit, gate.output, {row, plonkWireColumn::Output});
    }

    for (const auto& group : groups) {
        if (group.empty()) {
            continue;
        }
        for (size_t i = 0; i < group.size(); ++i) {
            const plonkWireCell& current = group[i];
            const plonkWireCell& next = group[(i + 1) % group.size()];
            set_sigma_eval(pk.sigma1_evals,
                           pk.sigma2_evals,
                           pk.sigma3_evals,
                           current,
                           plonk_wire_label(pk.n, pk.omega, pk.k1, pk.k2, next));
        }
    }

    pk.sigma1 = poly_from_evals(pk.sigma1_evals);
    pk.sigma2 = poly_from_evals(pk.sigma2_evals);
    pk.sigma3 = poly_from_evals(pk.sigma3_evals);

    pk.qM_commitment = commit_poly(public_parameters, pk.qM);
    pk.qL_commitment = commit_poly(public_parameters, pk.qL);
    pk.qR_commitment = commit_poly(public_parameters, pk.qR);
    pk.qO_commitment = commit_poly(public_parameters, pk.qO);
    pk.qC_commitment = commit_poly(public_parameters, pk.qC);
    pk.qK_commitment = commit_poly(public_parameters, pk.qK);
    pk.table_1_commitment = commit_poly(public_parameters, pk.table_1);
    pk.table_2_commitment = commit_poly(public_parameters, pk.table_2);
    pk.table_3_commitment = commit_poly(public_parameters, pk.table_3);
    pk.sigma1_commitment = commit_poly(public_parameters, pk.sigma1);
    pk.sigma2_commitment = commit_poly(public_parameters, pk.sigma2);
    pk.sigma3_commitment = commit_poly(public_parameters, pk.sigma3);

    setup.verification_key = derive_plonk_verification_key(pk);
    return setup;
}
