#pragma once

#include "zkboost/compiler/plonk_compiler.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

class zkboostCircuitVariable {
public:
    size_t index = 0;
};

class zkboostCircuitLinearConstraint {
public:
    zkboostCircuitVariable left;
    zkboostCircuitVariable right;
    zkboostCircuitVariable output;
    Fr left_scale = Fr(0);
    Fr right_scale = Fr(0);
    Fr output_scale = Fr(0);
    Fr constant_term = Fr(0);
};

class zkboostCircuitEqualityConstraint {
public:
    zkboostCircuitVariable left;
    zkboostCircuitVariable right;
};

class zkboostCircuitMulGate {
public:
    zkboostCircuitVariable left;
    zkboostCircuitVariable right;
    zkboostCircuitVariable output;
};

class zkboostCircuitBooleanGate {
public:
    zkboostCircuitVariable value;
};

class zkboostCircuitRangeCheck {
public:
    zkboostCircuitVariable value;
    std::int64_t min_value = 0;
    std::int64_t max_value = 0;
};

enum class zkboostCircuitComparisonMode {
    LessOrEqual,
    LessThan,
};

class zkboostCircuitComparison {
public:
    zkboostCircuitVariable left;
    zkboostCircuitVariable right;
    zkboostCircuitVariable output;
    std::int64_t min_value = 0;
    std::int64_t max_value = 0;
    zkboostCircuitComparisonMode mode = zkboostCircuitComparisonMode::LessOrEqual;
};

class zkboostCircuitLookup {
public:
    zkboostCircuitVariable left;
    zkboostCircuitVariable right;
    zkboostCircuitVariable output;
};

class zkboostCircuitLookupTableRow {
public:
    Fr left = Fr(0);
    Fr right = Fr(0);
    Fr output = Fr(0);
};

class zkboostCircuitRamReadWriteStep {
public:
    zkboostCircuitVariable address;
    zkboostCircuitVariable same_address_as_previous;
    std::vector<zkboostCircuitVariable> deltas;
    std::vector<zkboostCircuitVariable> running_values;
};

class zkboostCircuitRamTrace {
public:
    std::vector<zkboostCircuitRamReadWriteStep> steps;
    std::int64_t min_address = 0;
    std::int64_t max_address = 0;
    bool addresses_are_preconstrained = false;
};

class zkboostCircuitValue {
public:
    Fr value = Fr(0);
    bool is_constant = false;
};

class zkboostAssignedCircuitValue {
public:
    zkboostCircuitVariable variable;
    Fr value = Fr(0);
};

class zkboostGenericCircuit {
public:
    std::vector<zkboostCircuitValue> values;
    std::vector<size_t> public_input_variables;
    std::vector<zkboostCircuitLinearConstraint> linear_constraints;
    std::vector<zkboostCircuitEqualityConstraint> equality_constraints;
    std::vector<zkboostCircuitMulGate> mul_gates;
    std::vector<zkboostCircuitBooleanGate> boolean_gates;
    std::vector<zkboostCircuitRangeCheck> range_checks;
    std::vector<zkboostCircuitComparison> comparisons;
    std::vector<zkboostCircuitLookup> lookups;
    std::vector<zkboostCircuitLookupTableRow> lookup_table_rows;
    std::vector<zkboostCircuitRamTrace> ram_traces;
};

zkboostGenericCircuit concatenate_zkboost_generic_circuits(
    const std::vector<zkboostGenericCircuit>& circuits);

class zkboostGenericCircuitBuilder {
public:
    zkboostCircuitVariable add_witness(const Fr& value);
    zkboostCircuitVariable add_witness(std::int64_t value);

    zkboostCircuitVariable add_constant(const Fr& value);
    zkboostCircuitVariable add_constant(std::int64_t value);

    zkboostCircuitVariable add_public_input(const Fr& value);
    zkboostCircuitVariable add_public_input(std::int64_t value);
    void expose_public(const zkboostCircuitVariable& variable);

    void add_linear_constraint(const zkboostCircuitVariable& left,
                               const Fr& left_scale,
                               const zkboostCircuitVariable& right,
                               const Fr& right_scale,
                               const zkboostCircuitVariable& output,
                               const Fr& output_scale,
                               const Fr& constant_term);
    void constrain_equal(const zkboostCircuitVariable& left,
                         const zkboostCircuitVariable& right);

    zkboostCircuitVariable add_mul_gate(const zkboostCircuitVariable& left,
                                        const zkboostCircuitVariable& right);
    void add_mul_gate(const zkboostCircuitVariable& left,
                      const zkboostCircuitVariable& right,
                      const zkboostCircuitVariable& output);

    void add_boolean(const zkboostCircuitVariable& value);
    void add_range_check(const zkboostCircuitVariable& value,
                         std::int64_t min_value,
                         std::int64_t max_value);
    zkboostCircuitVariable add_comparison(const zkboostCircuitVariable& left,
                                          const zkboostCircuitVariable& right,
                                          std::int64_t min_value,
                                          std::int64_t max_value,
                                          zkboostCircuitComparisonMode mode =
                                              zkboostCircuitComparisonMode::LessOrEqual);

    void add_lookup(const zkboostCircuitVariable& left,
                    const zkboostCircuitVariable& right,
                    const zkboostCircuitVariable& output);
    void add_lookup_table_row(const Fr& left, const Fr& right, const Fr& output);
    void add_lookup_table_row(std::int64_t left, std::int64_t right, std::int64_t output);
    void add_ram_trace(const zkboostCircuitRamTrace& trace);

    const Fr& value_of(const zkboostCircuitVariable& variable) const;

    zkboostGenericCircuit build() const;

private:
    std::vector<zkboostCircuitValue> values_;
    std::vector<size_t> public_input_variables_;
    std::vector<zkboostCircuitLinearConstraint> linear_constraints_;
    std::vector<zkboostCircuitEqualityConstraint> equality_constraints_;
    std::vector<zkboostCircuitMulGate> mul_gates_;
    std::vector<zkboostCircuitBooleanGate> boolean_gates_;
    std::vector<zkboostCircuitRangeCheck> range_checks_;
    std::vector<zkboostCircuitComparison> comparisons_;
    std::vector<zkboostCircuitLookup> lookups_;
    std::vector<zkboostCircuitLookupTableRow> lookup_table_rows_;
    std::vector<zkboostCircuitRamTrace> ram_traces_;
};

zkboostCompiledCircuitStatement compile_zkboost_generic_circuit_statement(
    const std::string& name,
    const zkboostGenericCircuit& circuit,
    const kzgPublicParameters& public_parameters,
    const zkboostNamedPublicSpans& named_public_spans = {});

size_t required_zkboost_generic_circuit_plonk_srs_size(
    const zkboostGenericCircuit& circuit);

class zkboostGenericCircuitCompilerBase {
public:
    zkboostGenericCircuit build_circuit() const;
    const std::vector<Fr>& public_inputs() const noexcept;

protected:
    zkboostAssignedCircuitValue constant(std::int64_t value);
    zkboostAssignedCircuitValue private_value(std::int64_t value);
    zkboostAssignedCircuitValue public_value(std::int64_t value);

    zkboostAssignedCircuitValue multiply(const zkboostAssignedCircuitValue& left,
                                         const zkboostAssignedCircuitValue& right);
    zkboostAssignedCircuitValue affine(const zkboostAssignedCircuitValue& left,
                                       const Fr& left_scale,
                                       const Fr& constant_term);
    zkboostAssignedCircuitValue add_scaled(const zkboostAssignedCircuitValue& left,
                                           const Fr& left_scale,
                                           const zkboostAssignedCircuitValue& right,
                                           const Fr& right_scale,
                                           const Fr& constant_term);
    zkboostAssignedCircuitValue weighted_sum(const std::vector<zkboostAssignedCircuitValue>& indicators,
                                             const std::vector<std::int64_t>& weights);
    zkboostAssignedCircuitValue one_minus(const zkboostAssignedCircuitValue& bit);

    // ZKBoost-style grouped decomposition. Each limb is checked against a
    // reusable 2^limb_bits lookup table instead of being decomposed into bits.
    void constrain_nonnegative_with_limbs(
        const zkboostAssignedCircuitValue& value,
        std::int64_t witness_value,
        size_t bit_width,
        size_t limb_bits = 8);
    void constrain_nonnegative_at_most_with_limbs(
        const zkboostAssignedCircuitValue& value,
        std::int64_t witness_value,
        std::int64_t max_value,
        size_t limb_bits = 8);

    void constrain_boolean(const zkboostAssignedCircuitValue& bit);
    void constrain_equal(const zkboostAssignedCircuitValue& left,
                         const zkboostAssignedCircuitValue& right);

    void add_lookup(const zkboostAssignedCircuitValue& left,
                    const zkboostAssignedCircuitValue& right,
                    const zkboostAssignedCircuitValue& output);
    void add_lookup_table_row(std::int64_t left, std::int64_t right, std::int64_t output);
    void add_lookup_table_row(const Fr& left, const Fr& right, const Fr& output);

    zkboostCircuitVariable comparison(const zkboostAssignedCircuitValue& left,
                                      const zkboostAssignedCircuitValue& right,
                                      std::int64_t min_value,
                                      std::int64_t max_value,
                                      zkboostCircuitComparisonMode mode =
                                          zkboostCircuitComparisonMode::LessOrEqual);

    zkboostGenericCircuitBuilder builder_;

private:
    std::vector<Fr> public_inputs_;
    std::map<std::int64_t, zkboostAssignedCircuitValue> constant_cache_;
    std::set<size_t> grouped_range_table_widths_;
};
