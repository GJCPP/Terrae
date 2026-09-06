#pragma once

#include "bls12_381_wrapper.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace mcl::bn;

enum class plonkWireColumn {
    Left,
    Right,
    Output,
};

struct plonkVariable {
    size_t id = 0;

    friend bool operator==(const plonkVariable& lhs, const plonkVariable& rhs)
    {
        return lhs.id == rhs.id;
    }
};

struct plonkWireCell {
    size_t row = 0;
    plonkWireColumn column = plonkWireColumn::Left;
};

struct plonkGate {
    plonkVariable left;
    plonkVariable right;
    plonkVariable output;
    Fr qM = Fr(0);
    Fr qL = Fr(0);
    Fr qR = Fr(0);
    Fr qO = Fr(0);
    Fr qC = Fr(0);
    Fr qK = Fr(0);
};

struct plonkLookupTableRow {
    Fr left = Fr(0);
    Fr right = Fr(0);
    Fr output = Fr(0);
};

class plonkCircuit {
public:
    size_t variable_count = 0;
    std::vector<plonkGate> gates;
    std::vector<plonkLookupTableRow> lookup_table_rows;
    std::vector<size_t> public_input_rows;
    std::vector<plonkVariable> public_input_variables;
    std::vector<size_t> variable_roots;

    size_t domain_size() const;
    size_t canonical_variable(plonkVariable variable) const;
    void validate_variable(plonkVariable variable, const char* context) const;
};

class plonkCircuitBuilder {
public:
    plonkVariable add_private_variable();
    plonkVariable add_constant(const Fr& value);
    plonkVariable add_public_input();
    plonkVariable add_public_input(const Fr& expected_value_placeholder);
    void expose_public(plonkVariable variable);

    plonkVariable add_addition(plonkVariable left, plonkVariable right);
    plonkVariable add_multiplication(plonkVariable left, plonkVariable right);
    void add_lookup(plonkVariable left, plonkVariable right, plonkVariable output);
    void add_lookup_table_row(const Fr& left, const Fr& right, const Fr& output);
    void add_gate(plonkVariable left,
                  plonkVariable right,
                  plonkVariable output,
                  const Fr& qM,
                  const Fr& qL,
                  const Fr& qR,
                  const Fr& qO,
                  const Fr& qC,
                  const Fr& qK = Fr(0));
    void constrain_equal(plonkVariable left, plonkVariable right);

    plonkCircuit build() const;

private:
    std::vector<plonkGate> gates_;
    std::vector<plonkLookupTableRow> lookup_table_rows_;
    std::vector<size_t> public_input_rows_;
    std::vector<plonkVariable> public_input_variables_;
    std::vector<size_t> parent_;

    plonkVariable make_variable();
    void validate_variable(plonkVariable variable, const char* context) const;
    size_t find_root(size_t id) const;
    size_t find_root_mut(size_t id);
};

Fr evaluate_plonk_gate(const plonkGate& gate, const Fr& left, const Fr& right, const Fr& output);
