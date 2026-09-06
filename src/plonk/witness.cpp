#include "plonk/witness.hpp"

#include <stdexcept>

void plonkWitness::set(plonkVariable variable, const Fr& value)
{
    if (values_.size() <= variable.id) {
        values_.resize(variable.id + 1, Fr(0));
        assigned_.resize(variable.id + 1, false);
    }
    values_[variable.id] = value;
    assigned_[variable.id] = true;
}

bool plonkWitness::has(plonkVariable variable) const
{
    return variable.id < assigned_.size() && assigned_[variable.id];
}

Fr plonkWitness::value(plonkVariable variable) const
{
    if (!has(variable)) {
        throw std::runtime_error("plonkWitness::value: variable is not assigned");
    }
    return values_[variable.id];
}

plonkWireAssignments build_plonk_wire_assignments(const plonkCircuit& circuit, const plonkWitness& witness)
{
    const size_t n = circuit.domain_size();
    plonkWireAssignments assignments;
    assignments.a_evals.assign(n, Fr(0));
    assignments.b_evals.assign(n, Fr(0));
    assignments.c_evals.assign(n, Fr(0));

    for (size_t row = 0; row < circuit.gates.size(); ++row) {
        const plonkGate& gate = circuit.gates[row];
        circuit.validate_variable(gate.left, "build_plonk_wire_assignments");
        circuit.validate_variable(gate.right, "build_plonk_wire_assignments");
        circuit.validate_variable(gate.output, "build_plonk_wire_assignments");
        assignments.a_evals[row] = witness.value(gate.left);
        assignments.b_evals[row] = witness.value(gate.right);
        assignments.c_evals[row] = witness.value(gate.output);
    }
    return assignments;
}
