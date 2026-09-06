#pragma once

#include "plonk/circuit.hpp"

#include <vector>

class plonkWitness {
public:
    void set(plonkVariable variable, const Fr& value);
    bool has(plonkVariable variable) const;
    Fr value(plonkVariable variable) const;
    size_t size() const noexcept { return values_.size(); }

private:
    std::vector<Fr> values_;
    std::vector<bool> assigned_;
};

struct plonkWireAssignments {
    std::vector<Fr> a_evals;
    std::vector<Fr> b_evals;
    std::vector<Fr> c_evals;
};

plonkWireAssignments build_plonk_wire_assignments(const plonkCircuit& circuit, const plonkWitness& witness);
