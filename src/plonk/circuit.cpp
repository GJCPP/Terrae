#include "plonk/circuit.hpp"

#include <algorithm>
#include <string>

size_t plonkCircuit::domain_size() const
{
    return nextPow2(std::max({gates.size(), lookup_table_rows.size(), size_t(1)}));
}

void plonkCircuit::validate_variable(plonkVariable variable, const char* context) const
{
    if (variable.id >= variable_count) {
        throw std::runtime_error(std::string(context) + ": variable id out of range");
    }
}

size_t plonkCircuit::canonical_variable(plonkVariable variable) const
{
    validate_variable(variable, "plonkCircuit::canonical_variable");
    if (variable_roots.size() != variable_count) {
        throw std::runtime_error("plonkCircuit::canonical_variable: missing variable roots");
    }
    return variable_roots[variable.id];
}

plonkVariable plonkCircuitBuilder::make_variable()
{
    const size_t id = parent_.size();
    parent_.push_back(id);
    return plonkVariable{id};
}

void plonkCircuitBuilder::validate_variable(plonkVariable variable, const char* context) const
{
    if (variable.id >= parent_.size()) {
        throw std::runtime_error(std::string(context) + ": variable id out of range");
    }
}

size_t plonkCircuitBuilder::find_root(size_t id) const
{
    if (id >= parent_.size()) {
        throw std::runtime_error("plonkCircuitBuilder::find_root: variable id out of range");
    }
    while (parent_[id] != id) {
        id = parent_[id];
    }
    return id;
}

size_t plonkCircuitBuilder::find_root_mut(size_t id)
{
    if (id >= parent_.size()) {
        throw std::runtime_error("plonkCircuitBuilder::find_root_mut: variable id out of range");
    }
    if (parent_[id] == id) {
        return id;
    }
    parent_[id] = find_root_mut(parent_[id]);
    return parent_[id];
}

plonkVariable plonkCircuitBuilder::add_private_variable()
{
    return make_variable();
}

plonkVariable plonkCircuitBuilder::add_constant(const Fr& value)
{
    plonkVariable variable = make_variable();
    Fr neg_value;
    Fr::neg(neg_value, value);
    add_gate(variable, variable, variable, Fr(0), Fr(0), Fr(0), Fr(1), neg_value);
    return variable;
}

plonkVariable plonkCircuitBuilder::add_public_input()
{
    plonkVariable variable = make_variable();
    expose_public(variable);
    return variable;
}

plonkVariable plonkCircuitBuilder::add_public_input(const Fr& expected_value_placeholder)
{
    (void)expected_value_placeholder;
    return add_public_input();
}

void plonkCircuitBuilder::expose_public(plonkVariable variable)
{
    validate_variable(variable, "plonkCircuitBuilder::expose_public");
    const size_t row = gates_.size();
    add_gate(variable, variable, variable, Fr(0), Fr(0), Fr(0), Fr(1), Fr(0));
    public_input_rows_.push_back(row);
    public_input_variables_.push_back(variable);
}

plonkVariable plonkCircuitBuilder::add_addition(plonkVariable left, plonkVariable right)
{
    plonkVariable output = make_variable();
    add_gate(left, right, output, Fr(0), Fr(1), Fr(1), -Fr(1), Fr(0));
    return output;
}

plonkVariable plonkCircuitBuilder::add_multiplication(plonkVariable left, plonkVariable right)
{
    plonkVariable output = make_variable();
    add_gate(left, right, output, Fr(1), Fr(0), Fr(0), -Fr(1), Fr(0));
    return output;
}

void plonkCircuitBuilder::add_lookup(plonkVariable left, plonkVariable right, plonkVariable output)
{
    add_gate(left, right, output, Fr(0), Fr(0), Fr(0), Fr(0), Fr(0), Fr(1));
}

void plonkCircuitBuilder::add_lookup_table_row(const Fr& left, const Fr& right, const Fr& output)
{
    lookup_table_rows_.push_back(plonkLookupTableRow{left, right, output});
}

void plonkCircuitBuilder::add_gate(plonkVariable left,
                                   plonkVariable right,
                                   plonkVariable output,
                                   const Fr& qM,
                                   const Fr& qL,
                                   const Fr& qR,
                                   const Fr& qO,
                                   const Fr& qC,
                                   const Fr& qK)
{
    validate_variable(left, "plonkCircuitBuilder::add_gate");
    validate_variable(right, "plonkCircuitBuilder::add_gate");
    validate_variable(output, "plonkCircuitBuilder::add_gate");
    gates_.push_back(plonkGate{left, right, output, qM, qL, qR, qO, qC, qK});
}

void plonkCircuitBuilder::constrain_equal(plonkVariable left, plonkVariable right)
{
    validate_variable(left, "plonkCircuitBuilder::constrain_equal");
    validate_variable(right, "plonkCircuitBuilder::constrain_equal");
    const size_t left_root = find_root_mut(left.id);
    const size_t right_root = find_root_mut(right.id);
    if (left_root != right_root) {
        parent_[right_root] = left_root;
    }
}

plonkCircuit plonkCircuitBuilder::build() const
{
    plonkCircuit circuit;
    circuit.variable_count = parent_.size();
    circuit.gates = gates_;
    circuit.lookup_table_rows = lookup_table_rows_;
    circuit.public_input_rows = public_input_rows_;
    circuit.public_input_variables = public_input_variables_;
    circuit.variable_roots.resize(parent_.size());
    for (size_t i = 0; i < parent_.size(); ++i) {
        circuit.variable_roots[i] = find_root(i);
    }
    return circuit;
}

Fr evaluate_plonk_gate(const plonkGate& gate, const Fr& left, const Fr& right, const Fr& output)
{
    Fr acc = gate.qC;
    Fr term;
    Fr::mul(term, gate.qM, left);
    Fr::mul(term, term, right);
    Fr::add(acc, acc, term);
    Fr::mul(term, gate.qL, left);
    Fr::add(acc, acc, term);
    Fr::mul(term, gate.qR, right);
    Fr::add(acc, acc, term);
    Fr::mul(term, gate.qO, output);
    Fr::add(acc, acc, term);
    return acc;
}
