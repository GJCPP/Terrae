#include "zkboost/compiler/generic_circuit.hpp"

#include <algorithm>
#include <string>
#include <stdexcept>
#include <unordered_set>

namespace {

constexpr std::int64_t kTaggedRamRangeLookupMaxUpperBound = 4095;

zkboostCircuitVariable make_variable(size_t index)
{
    zkboostCircuitVariable out;
    out.index = index;
    return out;
}

bool comparison_holds(const Fr& left, const Fr& right, zkboostCircuitComparisonMode mode)
{
    const std::int64_t left_i64 = std::stoll(left.getStr(10));
    const std::int64_t right_i64 = std::stoll(right.getStr(10));
    if (mode == zkboostCircuitComparisonMode::LessOrEqual) {
        return left_i64 <= right_i64;
    }
    return left_i64 < right_i64;
}

std::int64_t fr_to_i64(const Fr& value)
{
    return std::stoll(value.getStr(10));
}

std::size_t nonnegative_bit_width(std::uint64_t max_value)
{
    std::size_t bits = 0;
    do {
        ++bits;
        max_value >>= 1U;
    } while (max_value != 0);
    return bits;
}

std::int64_t nonnegative_full_range_upper_bound(std::int64_t max_value)
{
    if (max_value <= 0) {
        return 0;
    }
    const std::size_t bit_width = nonnegative_bit_width(static_cast<std::uint64_t>(max_value));
    return (std::int64_t(1) << bit_width) - 1;
}

Fr tagged_ram_range_lookup_tag()
{
    Fr tag;
    tag.setStr("1461501637330902918203684832716283019655932542983", 10);
    return tag;
}

plonkVariable add_private_variable(plonkCircuitBuilder& builder,
                                   plonkWitness& witness,
                                   const Fr& value)
{
    const plonkVariable variable = builder.add_private_variable();
    witness.set(variable, value);
    return variable;
}

void constrain_boolean_variable(plonkCircuitBuilder& builder, const plonkVariable& bit)
{
    builder.add_gate(bit, bit, bit, Fr(1), -Fr(1), Fr(0), Fr(0), Fr(0));
}

void constrain_equal_variables(plonkCircuitBuilder& builder,
                               const plonkVariable& left,
                               const plonkVariable& right,
                               const plonkVariable& zero)
{
    builder.add_gate(left, right, zero, Fr(0), Fr(1), -Fr(1), Fr(0), Fr(0));
}

plonkVariable add_one_minus_variable(plonkCircuitBuilder& builder,
                                     plonkWitness& witness,
                                     const plonkVariable& one,
                                     const plonkVariable& bit,
                                     const Fr& bit_value)
{
    const Fr out_value = Fr(1) - bit_value;
    const plonkVariable out = add_private_variable(builder, witness, out_value);
    builder.add_gate(bit, one, out, Fr(0), Fr(-1), Fr(0), Fr(-1), Fr(1));
    return out;
}

plonkVariable add_product_variable(plonkCircuitBuilder& builder,
                                   plonkWitness& witness,
                                   const plonkVariable& left,
                                   const plonkVariable& right,
                                   const Fr& product_value)
{
    const plonkVariable out = add_private_variable(builder, witness, product_value);
    builder.add_gate(left, right, out, Fr(1), Fr(0), Fr(0), -Fr(1), Fr(0));
    return out;
}

void add_bounded_nonnegative_integer_constraints(plonkCircuitBuilder& builder,
                                                 plonkWitness& witness,
                                                 const plonkVariable& zero,
                                                 const plonkVariable& one,
                                                 const plonkVariable& value_variable,
                                                 const Fr& value,
                                                 std::uint64_t max_value)
{
    const std::size_t bit_width = nonnegative_bit_width(max_value);
    const std::int64_t value_i64 = fr_to_i64(value);
    const std::uint64_t witness_value = value_i64 >= 0 ? static_cast<std::uint64_t>(value_i64) : 0;

    std::vector<plonkVariable> bits;
    std::vector<Fr> bit_values;
    bits.reserve(bit_width);
    bit_values.reserve(bit_width);

    Fr reconstruction_value = Fr(0);
    plonkVariable reconstruction = zero;
    for (std::size_t bit = 0; bit < bit_width; ++bit) {
        const bool bit_is_set = ((witness_value >> bit) & 1U) != 0;
        const Fr bit_value = bit_is_set ? Fr(1) : Fr(0);
        const plonkVariable bit_variable = add_private_variable(builder, witness, bit_value);
        constrain_boolean_variable(builder, bit_variable);

        const Fr next_reconstruction_value =
            reconstruction_value + (bit_is_set ? zkboost_int64_to_fr(std::int64_t(1) << bit) : Fr(0));
        const plonkVariable next_reconstruction =
            add_private_variable(builder, witness, next_reconstruction_value);
        builder.add_gate(reconstruction,
                         bit_variable,
                         next_reconstruction,
                         Fr(0),
                         Fr(1),
                         zkboost_int64_to_fr(std::int64_t(1) << bit),
                         -Fr(1),
                         Fr(0));

        bits.push_back(bit_variable);
        bit_values.push_back(bit_value);
        reconstruction = next_reconstruction;
        reconstruction_value = next_reconstruction_value;
    }
    constrain_equal_variables(builder, value_variable, reconstruction, zero);

    const std::uint64_t full_range_max = (std::uint64_t(1) << bit_width) - 1U;
    if (max_value == full_range_max) {
        return;
    }

    plonkVariable prefix_equal = one;
    Fr prefix_equal_value = Fr(1);
    for (std::size_t reversed = 0; reversed < bit_width; ++reversed) {
        const std::size_t bit = bit_width - 1 - reversed;
        const bool bound_bit_is_set = ((max_value >> bit) & 1U) != 0;
        if (!bound_bit_is_set) {
            const Fr forced_zero_value = prefix_equal_value * bit_values[bit];
            const plonkVariable forced_zero =
                add_product_variable(builder, witness, prefix_equal, bits[bit], forced_zero_value);
            constrain_equal_variables(builder, forced_zero, zero, zero);
        }

        const plonkVariable equality_bit =
            bound_bit_is_set ? bits[bit] :
                               add_one_minus_variable(builder, witness, one, bits[bit], bit_values[bit]);
        const Fr equality_bit_value = bound_bit_is_set ? bit_values[bit] : (Fr(1) - bit_values[bit]);
        prefix_equal =
            add_product_variable(builder,
                                 witness,
                                 prefix_equal,
                                 equality_bit,
                                 prefix_equal_value * equality_bit_value);
        prefix_equal_value *= equality_bit_value;
    }
}

void add_bounded_integer_constraints(plonkCircuitBuilder& builder,
                                     plonkWitness& witness,
                                     const plonkVariable& zero,
                                     const plonkVariable& one,
                                     const plonkVariable& value_variable,
                                     const Fr& value,
                                     std::int64_t min_value,
                                     std::int64_t max_value)
{
    if (min_value > max_value) {
        throw std::runtime_error("compile_zkboost_generic_circuit_statement: invalid bounded integer interval");
    }

    const std::int64_t width = max_value - min_value;
    if (min_value == 0) {
        add_bounded_nonnegative_integer_constraints(
            builder,
            witness,
            zero,
            one,
            value_variable,
            value,
            static_cast<std::uint64_t>(width));
        return;
    }

    const Fr shifted_value = value - zkboost_int64_to_fr(min_value);
    const plonkVariable shifted_variable = add_private_variable(builder, witness, shifted_value);
    builder.add_gate(value_variable,
                     shifted_variable,
                     zero,
                     Fr(0),
                     Fr(1),
                     -Fr(1),
                     Fr(0),
                     -zkboost_int64_to_fr(min_value));
    add_bounded_nonnegative_integer_constraints(builder,
                                                witness,
                                                zero,
                                                one,
                                                shifted_variable,
                                                shifted_value,
                                                static_cast<std::uint64_t>(width));
}

} // namespace

zkboostCircuitVariable zkboostGenericCircuitBuilder::add_witness(const Fr& value)
{
    values_.push_back({value, false});
    return make_variable(values_.size() - 1);
}

zkboostCircuitVariable zkboostGenericCircuitBuilder::add_witness(std::int64_t value)
{
    return add_witness(zkboost_int64_to_fr(value));
}

zkboostCircuitVariable zkboostGenericCircuitBuilder::add_constant(const Fr& value)
{
    values_.push_back({value, true});
    return make_variable(values_.size() - 1);
}

zkboostCircuitVariable zkboostGenericCircuitBuilder::add_constant(std::int64_t value)
{
    return add_constant(zkboost_int64_to_fr(value));
}

zkboostCircuitVariable zkboostGenericCircuitBuilder::add_public_input(const Fr& value)
{
    const zkboostCircuitVariable variable = add_witness(value);
    expose_public(variable);
    return variable;
}

zkboostCircuitVariable zkboostGenericCircuitBuilder::add_public_input(std::int64_t value)
{
    return add_public_input(zkboost_int64_to_fr(value));
}

void zkboostGenericCircuitBuilder::expose_public(const zkboostCircuitVariable& variable)
{
    zkboost_require(variable.index < values_.size(),
                           "zkboost_generic_circuit",
                           "public variable is out of range");
    if (values_[variable.index].is_constant) {
        throw std::runtime_error("zkboost_generic_circuit: constant variable cannot be exposed as public");
    }

    const bool already_public =
        std::find(public_input_variables_.begin(), public_input_variables_.end(), variable.index) !=
        public_input_variables_.end();
    if (!already_public) {
        public_input_variables_.push_back(variable.index);
    }
}

void zkboostGenericCircuitBuilder::add_linear_constraint(const zkboostCircuitVariable& left,
                                                         const Fr& left_scale,
                                                         const zkboostCircuitVariable& right,
                                                         const Fr& right_scale,
                                                         const zkboostCircuitVariable& output,
                                                         const Fr& output_scale,
                                                         const Fr& constant_term)
{
    linear_constraints_.push_back(
        {left, right, output, left_scale, right_scale, output_scale, constant_term});
}

void zkboostGenericCircuitBuilder::constrain_equal(const zkboostCircuitVariable& left,
                                                   const zkboostCircuitVariable& right)
{
    equality_constraints_.push_back({left, right});
}

zkboostCircuitVariable zkboostGenericCircuitBuilder::add_mul_gate(const zkboostCircuitVariable& left,
                                                                  const zkboostCircuitVariable& right)
{
    const zkboostCircuitVariable output = add_witness(value_of(left) * value_of(right));
    add_mul_gate(left, right, output);
    return output;
}

void zkboostGenericCircuitBuilder::add_mul_gate(const zkboostCircuitVariable& left,
                                                const zkboostCircuitVariable& right,
                                                const zkboostCircuitVariable& output)
{
    mul_gates_.push_back({left, right, output});
}

void zkboostGenericCircuitBuilder::add_boolean(const zkboostCircuitVariable& value)
{
    boolean_gates_.push_back({value});
}

void zkboostGenericCircuitBuilder::add_range_check(const zkboostCircuitVariable& value,
                                                   std::int64_t min_value,
                                                   std::int64_t max_value)
{
    if (min_value > max_value) {
        throw std::runtime_error("zkboost_generic_circuit: invalid range-check interval");
    }
    range_checks_.push_back({value, min_value, max_value});
}

zkboostCircuitVariable zkboostGenericCircuitBuilder::add_comparison(
    const zkboostCircuitVariable& left,
    const zkboostCircuitVariable& right,
    std::int64_t min_value,
    std::int64_t max_value,
    zkboostCircuitComparisonMode mode)
{
    if (min_value > max_value) {
        throw std::runtime_error("zkboost_generic_circuit: invalid comparison interval");
    }

    const Fr bit_value = comparison_holds(value_of(left), value_of(right), mode) ? Fr(1) : Fr(0);
    const zkboostCircuitVariable output = add_witness(bit_value);
    comparisons_.push_back({left, right, output, min_value, max_value, mode});
    add_boolean(output);
    return output;
}

void zkboostGenericCircuitBuilder::add_lookup(const zkboostCircuitVariable& left,
                                              const zkboostCircuitVariable& right,
                                              const zkboostCircuitVariable& output)
{
    lookups_.push_back({left, right, output});
}

void zkboostGenericCircuitBuilder::add_lookup_table_row(const Fr& left,
                                                        const Fr& right,
                                                        const Fr& output)
{
    lookup_table_rows_.push_back({left, right, output});
}

void zkboostGenericCircuitBuilder::add_lookup_table_row(std::int64_t left,
                                                        std::int64_t right,
                                                        std::int64_t output)
{
    add_lookup_table_row(zkboost_int64_to_fr(left),
                         zkboost_int64_to_fr(right),
                         zkboost_int64_to_fr(output));
}

void zkboostGenericCircuitBuilder::add_ram_trace(const zkboostCircuitRamTrace& trace)
{
    if (trace.min_address > trace.max_address) {
        throw std::runtime_error("zkboost_generic_circuit: invalid RAM address interval");
    }
    ram_traces_.push_back(trace);
}

const Fr& zkboostGenericCircuitBuilder::value_of(const zkboostCircuitVariable& variable) const
{
    if (variable.index >= values_.size()) {
        throw std::runtime_error("zkboost_generic_circuit: variable is out of range");
    }
    return values_[variable.index].value;
}

zkboostGenericCircuit zkboostGenericCircuitBuilder::build() const
{
    zkboostGenericCircuit out;
    out.values = values_;
    out.public_input_variables = public_input_variables_;
    out.linear_constraints = linear_constraints_;
    out.equality_constraints = equality_constraints_;
    out.mul_gates = mul_gates_;
    out.boolean_gates = boolean_gates_;
    out.range_checks = range_checks_;
    out.comparisons = comparisons_;
    out.lookups = lookups_;
    out.lookup_table_rows = lookup_table_rows_;
    out.ram_traces = ram_traces_;
    return out;
}

zkboostGenericCircuit concatenate_zkboost_generic_circuits(
    const std::vector<zkboostGenericCircuit>& circuits)
{
    zkboostGenericCircuit out;
    const auto shifted = [](zkboostCircuitVariable variable, size_t offset) {
        variable.index += offset;
        return variable;
    };
    const auto append_lookup_row_once = [&](const zkboostCircuitLookupTableRow& row) {
        const auto duplicate = std::find_if(
            out.lookup_table_rows.begin(),
            out.lookup_table_rows.end(),
            [&](const zkboostCircuitLookupTableRow& existing) {
                return existing.left == row.left &&
                       existing.right == row.right &&
                       existing.output == row.output;
            });
        if (duplicate == out.lookup_table_rows.end()) {
            out.lookup_table_rows.push_back(row);
        }
    };

    for (const zkboostGenericCircuit& circuit : circuits) {
        const size_t offset = out.values.size();
        out.values.insert(out.values.end(), circuit.values.begin(), circuit.values.end());
        for (size_t variable : circuit.public_input_variables) {
            out.public_input_variables.push_back(variable + offset);
        }
        for (zkboostCircuitLinearConstraint constraint : circuit.linear_constraints) {
            constraint.left = shifted(constraint.left, offset);
            constraint.right = shifted(constraint.right, offset);
            constraint.output = shifted(constraint.output, offset);
            out.linear_constraints.push_back(std::move(constraint));
        }
        for (zkboostCircuitEqualityConstraint constraint : circuit.equality_constraints) {
            constraint.left = shifted(constraint.left, offset);
            constraint.right = shifted(constraint.right, offset);
            out.equality_constraints.push_back(std::move(constraint));
        }
        for (zkboostCircuitMulGate gate : circuit.mul_gates) {
            gate.left = shifted(gate.left, offset);
            gate.right = shifted(gate.right, offset);
            gate.output = shifted(gate.output, offset);
            out.mul_gates.push_back(std::move(gate));
        }
        for (zkboostCircuitBooleanGate gate : circuit.boolean_gates) {
            gate.value = shifted(gate.value, offset);
            out.boolean_gates.push_back(std::move(gate));
        }
        for (zkboostCircuitRangeCheck range_check : circuit.range_checks) {
            range_check.value = shifted(range_check.value, offset);
            out.range_checks.push_back(std::move(range_check));
        }
        for (zkboostCircuitComparison comparison : circuit.comparisons) {
            comparison.left = shifted(comparison.left, offset);
            comparison.right = shifted(comparison.right, offset);
            comparison.output = shifted(comparison.output, offset);
            out.comparisons.push_back(std::move(comparison));
        }
        for (zkboostCircuitLookup lookup : circuit.lookups) {
            lookup.left = shifted(lookup.left, offset);
            lookup.right = shifted(lookup.right, offset);
            lookup.output = shifted(lookup.output, offset);
            out.lookups.push_back(std::move(lookup));
        }
        for (const zkboostCircuitLookupTableRow& row : circuit.lookup_table_rows) {
            append_lookup_row_once(row);
        }
        for (zkboostCircuitRamTrace trace : circuit.ram_traces) {
            for (zkboostCircuitRamReadWriteStep& step : trace.steps) {
                step.address = shifted(step.address, offset);
                step.same_address_as_previous =
                    shifted(step.same_address_as_previous, offset);
                for (zkboostCircuitVariable& delta : step.deltas) {
                    delta = shifted(delta, offset);
                }
                for (zkboostCircuitVariable& running : step.running_values) {
                    running = shifted(running, offset);
                }
            }
            out.ram_traces.push_back(std::move(trace));
        }
    }
    return out;
}

zkboostCompiledCircuitStatement compile_zkboost_generic_circuit_statement(
    const std::string& name,
    const zkboostGenericCircuit& circuit,
    const kzgPublicParameters& public_parameters,
    const zkboostNamedPublicSpans& named_public_spans)
{
    proof_cost::PrefixScope component_scope("component_" + name);
    plonkCircuitBuilder builder;
    plonkWitness witness;
    std::vector<Fr> public_inputs;
    std::vector<plonkVariable> variable_map(circuit.values.size());

    std::unordered_set<size_t> public_variable_set(circuit.public_input_variables.begin(),
                                                   circuit.public_input_variables.end());
    size_t public_input_offset = 0;
    for (size_t variable_index : circuit.public_input_variables) {
        if (variable_index >= circuit.values.size()) {
            throw std::runtime_error("compile_zkboost_generic_circuit_statement: public variable out of range");
        }
        if (circuit.values[variable_index].is_constant) {
            throw std::runtime_error("compile_zkboost_generic_circuit_statement: public variable cannot be constant");
        }
        ++public_input_offset;
    }

    (void)public_input_offset;
    for (size_t i = 0; i < circuit.values.size(); ++i) {
        const zkboostCircuitValue& value = circuit.values[i];
        if (public_variable_set.count(i) != 0) {
            variable_map[i] = builder.add_public_input();
            public_inputs.push_back(value.value);
        } else if (value.is_constant) {
            variable_map[i] = builder.add_constant(value.value);
        } else {
            variable_map[i] = builder.add_private_variable();
        }
        witness.set(variable_map[i], value.value);
    }

    for (const zkboostCircuitLookupTableRow& row : circuit.lookup_table_rows) {
        builder.add_lookup_table_row(row.left, row.right, row.output);
    }

    const plonkVariable zero = builder.add_constant(Fr(0));
    witness.set(zero, Fr(0));
    const plonkVariable one = builder.add_constant(Fr(1));
    witness.set(one, Fr(1));

    std::vector<size_t> variable_use_counts(circuit.values.size(), 0);
    for (size_t i = 0; i < circuit.values.size(); ++i) {
        if (circuit.values[i].is_constant || public_variable_set.count(i) != 0) {
            ++variable_use_counts[i];
        }
    }
    const auto mark_variable_use = [&](const zkboostCircuitVariable& variable) {
        if (variable.index >= variable_use_counts.size()) {
            throw std::runtime_error("compile_zkboost_generic_circuit_statement: variable out of range");
        }
        ++variable_use_counts[variable.index];
    };
    for (const zkboostCircuitLinearConstraint& constraint : circuit.linear_constraints) {
        mark_variable_use(constraint.left);
        mark_variable_use(constraint.right);
        mark_variable_use(constraint.output);
    }
    for (const zkboostCircuitMulGate& gate : circuit.mul_gates) {
        mark_variable_use(gate.left);
        mark_variable_use(gate.right);
        mark_variable_use(gate.output);
    }
    for (const zkboostCircuitBooleanGate& gate : circuit.boolean_gates) {
        mark_variable_use(gate.value);
    }
    for (const zkboostCircuitRangeCheck& range_check : circuit.range_checks) {
        mark_variable_use(range_check.value);
    }
    for (const zkboostCircuitComparison& comparison : circuit.comparisons) {
        mark_variable_use(comparison.left);
        mark_variable_use(comparison.right);
        mark_variable_use(comparison.output);
    }
    for (const zkboostCircuitLookup& lookup : circuit.lookups) {
        mark_variable_use(lookup.left);
        mark_variable_use(lookup.right);
        mark_variable_use(lookup.output);
    }
    for (const zkboostCircuitRamTrace& trace : circuit.ram_traces) {
        for (const zkboostCircuitRamReadWriteStep& step : trace.steps) {
            mark_variable_use(step.address);
            mark_variable_use(step.same_address_as_previous);
            for (const zkboostCircuitVariable& delta : step.deltas) {
                mark_variable_use(delta);
            }
            for (const zkboostCircuitVariable& running : step.running_values) {
                mark_variable_use(running);
            }
        }
    }

    for (const zkboostCircuitEqualityConstraint& equality : circuit.equality_constraints) {
        const plonkVariable left = variable_map.at(equality.left.index);
        const plonkVariable right = variable_map.at(equality.right.index);
        if (variable_use_counts.at(equality.left.index) == 0) {
            builder.add_gate(left, left, left, Fr(0), Fr(0), Fr(0), Fr(0), Fr(0));
            ++variable_use_counts[equality.left.index];
        }
        if (variable_use_counts.at(equality.right.index) == 0) {
            builder.add_gate(right, right, right, Fr(0), Fr(0), Fr(0), Fr(0), Fr(0));
            ++variable_use_counts[equality.right.index];
        }
        builder.constrain_equal(left, right);
    }

    for (const zkboostCircuitLinearConstraint& constraint : circuit.linear_constraints) {
        builder.add_gate(variable_map.at(constraint.left.index),
                         variable_map.at(constraint.right.index),
                         variable_map.at(constraint.output.index),
                         Fr(0),
                         constraint.left_scale,
                         constraint.right_scale,
                         constraint.output_scale,
                         constraint.constant_term);
    }

    for (const zkboostCircuitMulGate& gate : circuit.mul_gates) {
        builder.add_gate(variable_map.at(gate.left.index),
                         variable_map.at(gate.right.index),
                         variable_map.at(gate.output.index),
                         Fr(1),
                         Fr(0),
                         Fr(0),
                         -Fr(1),
                         Fr(0));
    }

    for (const zkboostCircuitBooleanGate& gate : circuit.boolean_gates) {
        builder.add_gate(variable_map.at(gate.value.index),
                         variable_map.at(gate.value.index),
                         variable_map.at(gate.value.index),
                         Fr(1),
                         -Fr(1),
                         Fr(0),
                         Fr(0),
                         Fr(0));
    }

    for (const zkboostCircuitRangeCheck& range_check : circuit.range_checks) {
        for (std::int64_t value = range_check.min_value; value <= range_check.max_value; ++value) {
            builder.add_lookup_table_row(zkboost_int64_to_fr(value), Fr(0), Fr(0));
        }
        builder.add_lookup(variable_map.at(range_check.value.index), zero, zero);
    }

    for (const zkboostCircuitComparison& comparison : circuit.comparisons) {
        for (std::int64_t left = comparison.min_value; left <= comparison.max_value; ++left) {
            for (std::int64_t right = comparison.min_value; right <= comparison.max_value; ++right) {
                const bool bit = (comparison.mode == zkboostCircuitComparisonMode::LessOrEqual) ?
                                     (left <= right) :
                                     (left < right);
                builder.add_lookup_table_row(zkboost_int64_to_fr(left),
                                             zkboost_int64_to_fr(right),
                                             bit ? Fr(1) : Fr(0));
            }
        }
        builder.add_lookup(variable_map.at(comparison.left.index),
                           variable_map.at(comparison.right.index),
                           variable_map.at(comparison.output.index));
    }

    for (const zkboostCircuitLookup& lookup : circuit.lookups) {
        builder.add_lookup(variable_map.at(lookup.left.index),
                           variable_map.at(lookup.right.index),
                           variable_map.at(lookup.output.index));
    }

    for (const zkboostCircuitRamTrace& trace : circuit.ram_traces) {
        const std::int64_t max_address_difference = trace.max_address - trace.min_address;
        const std::int64_t address_difference_range =
            nonnegative_full_range_upper_bound(max_address_difference);
        const bool use_tagged_difference_range_lookup =
            address_difference_range <= kTaggedRamRangeLookupMaxUpperBound;
        plonkVariable range_lookup_tag;
        if (use_tagged_difference_range_lookup) {
            const Fr tag = tagged_ram_range_lookup_tag();
            range_lookup_tag = builder.add_constant(tag);
            witness.set(range_lookup_tag, tag);
            for (std::int64_t value = 0; value <= address_difference_range; ++value) {
                builder.add_lookup_table_row(zkboost_int64_to_fr(value), tag, Fr(0));
            }
        }

        for (size_t step_index = 0; step_index < trace.steps.size(); ++step_index) {
            const zkboostCircuitRamReadWriteStep& step = trace.steps[step_index];
            if (step.deltas.size() != step.running_values.size()) {
                throw std::runtime_error(
                    "compile_zkboost_generic_circuit_statement: RAM delta/running size mismatch");
            }

            const plonkVariable address = variable_map.at(step.address.index);
            const plonkVariable same = variable_map.at(step.same_address_as_previous.index);
            if (!trace.addresses_are_preconstrained) {
                add_bounded_integer_constraints(builder,
                                                witness,
                                                zero,
                                                one,
                                                address,
                                                circuit.values.at(step.address.index).value,
                                                trace.min_address,
                                                trace.max_address);
            }
            builder.add_gate(same, same, same, Fr(1), -Fr(1), Fr(0), Fr(0), Fr(0));

            if (step_index == 0) {
                builder.add_gate(same, same, same, Fr(0), Fr(1), Fr(0), Fr(0), Fr(0));
                for (size_t lane = 0; lane < step.deltas.size(); ++lane) {
                    builder.add_gate(variable_map.at(step.running_values[lane].index),
                                     variable_map.at(step.deltas[lane].index),
                                     variable_map.at(step.running_values[lane].index),
                                     Fr(0),
                                     Fr(1),
                                     -Fr(1),
                                     Fr(0),
                                     Fr(0));
                }
                continue;
            }

            const zkboostCircuitRamReadWriteStep& previous = trace.steps[step_index - 1];
            const plonkVariable previous_address = variable_map.at(previous.address.index);

            const Fr address_difference =
                circuit.values.at(step.address.index).value -
                circuit.values.at(previous.address.index).value;
            const plonkVariable address_difference_variable = builder.add_private_variable();
            witness.set(address_difference_variable, address_difference);
            builder.add_gate(address,
                             previous_address,
                             address_difference_variable,
                             Fr(0),
                             Fr(1),
                             -Fr(1),
                             -Fr(1),
                             Fr(0));

            const Fr same_times_difference =
                circuit.values.at(step.same_address_as_previous.index).value * address_difference;
            const plonkVariable same_times_difference_variable = builder.add_private_variable();
            witness.set(same_times_difference_variable, same_times_difference);
            builder.add_gate(same,
                             address_difference_variable,
                             same_times_difference_variable,
                             Fr(1),
                             Fr(0),
                             Fr(0),
                             -Fr(1),
                             Fr(0));
            builder.add_gate(same_times_difference_variable,
                             same_times_difference_variable,
                             same_times_difference_variable,
                             Fr(0),
                             Fr(1),
                             Fr(0),
                             Fr(0),
                             Fr(0));

            const Fr new_segment_value = Fr(1) - circuit.values.at(step.same_address_as_previous.index).value;
            const plonkVariable new_segment_variable = builder.add_private_variable();
            witness.set(new_segment_variable, new_segment_value);
            builder.add_gate(same,
                             same,
                             new_segment_variable,
                             Fr(0),
                             Fr(-1),
                             Fr(0),
                             Fr(-1),
                             Fr(1));

            if (use_tagged_difference_range_lookup) {
                builder.add_lookup(address_difference_variable, range_lookup_tag, zero);
            } else {
                add_bounded_integer_constraints(builder,
                                                witness,
                                                zero,
                                                one,
                                                address_difference_variable,
                                                address_difference,
                                                0,
                                                address_difference_range);
            }

            // Non-zero address differences mark a fresh segment without needing
            // a quadratic `(previous, current)` comparison table.
            Fr address_difference_inverse = Fr(0);
            if (!address_difference.isZero()) {
                Fr::inv(address_difference_inverse, address_difference);
            }
            const plonkVariable address_difference_inverse_variable =
                builder.add_private_variable();
            witness.set(address_difference_inverse_variable, address_difference_inverse);
            builder.add_gate(address_difference_variable,
                             address_difference_inverse_variable,
                             new_segment_variable,
                             Fr(1),
                             Fr(0),
                             Fr(0),
                             -Fr(1),
                             Fr(0));

            for (size_t lane = 0; lane < step.deltas.size(); ++lane) {
                const plonkVariable previous_running =
                    variable_map.at(previous.running_values[lane].index);
                const plonkVariable delta = variable_map.at(step.deltas[lane].index);
                const plonkVariable running = variable_map.at(step.running_values[lane].index);

                const Fr carried_value =
                    circuit.values.at(step.same_address_as_previous.index).value *
                    circuit.values.at(previous.running_values[lane].index).value;
                const plonkVariable carried_variable = builder.add_private_variable();
                witness.set(carried_variable, carried_value);
                builder.add_gate(same,
                                 previous_running,
                                 carried_variable,
                                 Fr(1),
                                 Fr(0),
                                 Fr(0),
                                 -Fr(1),
                                 Fr(0));

                builder.add_gate(running,
                                 delta,
                                 carried_variable,
                                 Fr(0),
                                 Fr(1),
                                 -Fr(1),
                                 -Fr(1),
                                 Fr(0));
            }
        }
    }

    const plonkCircuit compiled_circuit = builder.build();
    proof_cost::add("zkboost_plonk_active_gate_row_count", compiled_circuit.gates.size());

    return compile_zkboost_compiled_circuit_statement(
        name,
        compiled_circuit,
        witness,
        public_inputs,
        public_parameters,
        named_public_spans);
}

size_t required_zkboost_generic_circuit_plonk_srs_size(
    const zkboostGenericCircuit& circuit)
{
    std::unordered_set<size_t> public_variable_set(circuit.public_input_variables.begin(),
                                                   circuit.public_input_variables.end());
    std::vector<size_t> variable_use_counts(circuit.values.size(), 0);
    for (size_t i = 0; i < circuit.values.size(); ++i) {
        if (circuit.values[i].is_constant || public_variable_set.count(i) != 0) {
            ++variable_use_counts[i];
        }
    }
    const auto mark_variable_use = [&](const zkboostCircuitVariable& variable) {
        if (variable.index >= variable_use_counts.size()) {
            throw std::runtime_error("required_zkboost_generic_circuit_plonk_srs_size: variable out of range");
        }
        ++variable_use_counts[variable.index];
    };

    for (const zkboostCircuitLinearConstraint& constraint : circuit.linear_constraints) {
        mark_variable_use(constraint.left);
        mark_variable_use(constraint.right);
        mark_variable_use(constraint.output);
    }
    for (const zkboostCircuitMulGate& gate : circuit.mul_gates) {
        mark_variable_use(gate.left);
        mark_variable_use(gate.right);
        mark_variable_use(gate.output);
    }
    for (const zkboostCircuitBooleanGate& gate : circuit.boolean_gates) {
        mark_variable_use(gate.value);
    }
    for (const zkboostCircuitRangeCheck& range_check : circuit.range_checks) {
        mark_variable_use(range_check.value);
    }
    for (const zkboostCircuitComparison& comparison : circuit.comparisons) {
        mark_variable_use(comparison.left);
        mark_variable_use(comparison.right);
        mark_variable_use(comparison.output);
    }
    for (const zkboostCircuitLookup& lookup : circuit.lookups) {
        mark_variable_use(lookup.left);
        mark_variable_use(lookup.right);
        mark_variable_use(lookup.output);
    }
    for (const zkboostCircuitRamTrace& trace : circuit.ram_traces) {
        for (const zkboostCircuitRamReadWriteStep& step : trace.steps) {
            mark_variable_use(step.address);
            mark_variable_use(step.same_address_as_previous);
            for (const zkboostCircuitVariable& delta : step.deltas) {
                mark_variable_use(delta);
            }
            for (const zkboostCircuitVariable& running : step.running_values) {
                mark_variable_use(running);
            }
        }
    }

    size_t gate_count = circuit.public_input_variables.size() + 2;
    for (const zkboostCircuitValue& value : circuit.values) {
        if (value.is_constant) {
            ++gate_count;
        }
    }
    for (const zkboostCircuitEqualityConstraint& equality : circuit.equality_constraints) {
        if (equality.left.index >= variable_use_counts.size() ||
            equality.right.index >= variable_use_counts.size()) {
            throw std::runtime_error("required_zkboost_generic_circuit_plonk_srs_size: equality variable out of range");
        }
        if (variable_use_counts.at(equality.left.index) == 0) {
            ++gate_count;
            ++variable_use_counts[equality.left.index];
        }
        if (variable_use_counts.at(equality.right.index) == 0) {
            ++gate_count;
            ++variable_use_counts[equality.right.index];
        }
    }

    gate_count += circuit.linear_constraints.size();
    gate_count += circuit.mul_gates.size();
    gate_count += circuit.boolean_gates.size();
    gate_count += circuit.range_checks.size();
    gate_count += circuit.comparisons.size();
    gate_count += circuit.lookups.size();

    size_t lookup_row_count = circuit.lookup_table_rows.size();
    for (const zkboostCircuitRangeCheck& range_check : circuit.range_checks) {
        if (range_check.min_value > range_check.max_value) {
            throw std::runtime_error("required_zkboost_generic_circuit_plonk_srs_size: invalid range check interval");
        }
        lookup_row_count += static_cast<size_t>(range_check.max_value - range_check.min_value + 1);
    }
    for (const zkboostCircuitComparison& comparison : circuit.comparisons) {
        if (comparison.min_value > comparison.max_value) {
            throw std::runtime_error("required_zkboost_generic_circuit_plonk_srs_size: invalid comparison interval");
        }
        const size_t width = static_cast<size_t>(comparison.max_value - comparison.min_value + 1);
        lookup_row_count += width * width;
    }
    if (!circuit.ram_traces.empty()) {
        throw std::runtime_error("required_zkboost_generic_circuit_plonk_srs_size: RAM traces are not supported");
    }

    return nextPow2(std::max({gate_count, lookup_row_count, size_t(1)}));
}

zkboostGenericCircuit zkboostGenericCircuitCompilerBase::build_circuit() const
{
    return builder_.build();
}

const std::vector<Fr>& zkboostGenericCircuitCompilerBase::public_inputs() const noexcept
{
    return public_inputs_;
}

zkboostAssignedCircuitValue zkboostGenericCircuitCompilerBase::constant(std::int64_t value)
{
    const auto it = constant_cache_.find(value);
    if (it != constant_cache_.end()) {
        return it->second;
    }

    const Fr field_value = zkboost_int64_to_fr(value);
    const zkboostAssignedCircuitValue out{builder_.add_constant(field_value), field_value};
    constant_cache_.emplace(value, out);
    return out;
}

zkboostAssignedCircuitValue zkboostGenericCircuitCompilerBase::private_value(std::int64_t value)
{
    const Fr field_value = zkboost_int64_to_fr(value);
    return {builder_.add_witness(field_value), field_value};
}

zkboostAssignedCircuitValue zkboostGenericCircuitCompilerBase::public_value(std::int64_t value)
{
    const Fr field_value = zkboost_int64_to_fr(value);
    const zkboostCircuitVariable variable = builder_.add_public_input(field_value);
    public_inputs_.push_back(field_value);
    return {variable, field_value};
}

zkboostAssignedCircuitValue zkboostGenericCircuitCompilerBase::multiply(
    const zkboostAssignedCircuitValue& left,
    const zkboostAssignedCircuitValue& right)
{
    const Fr out_value = left.value * right.value;
    const zkboostCircuitVariable output = builder_.add_witness(out_value);
    builder_.add_mul_gate(left.variable, right.variable, output);
    return {output, out_value};
}

zkboostAssignedCircuitValue zkboostGenericCircuitCompilerBase::affine(
    const zkboostAssignedCircuitValue& left,
    const Fr& left_scale,
    const Fr& constant_term)
{
    const Fr out_value = (left.value * left_scale) + constant_term;
    const zkboostCircuitVariable output = builder_.add_witness(out_value);
    builder_.add_linear_constraint(left.variable, left_scale, output, -Fr(1), output, Fr(0), constant_term);
    return {output, out_value};
}

zkboostAssignedCircuitValue zkboostGenericCircuitCompilerBase::add_scaled(
    const zkboostAssignedCircuitValue& left,
    const Fr& left_scale,
    const zkboostAssignedCircuitValue& right,
    const Fr& right_scale,
    const Fr& constant_term)
{
    const Fr out_value = (left.value * left_scale) + (right.value * right_scale) + constant_term;
    const zkboostCircuitVariable output = builder_.add_witness(out_value);
    builder_.add_linear_constraint(
        left.variable, left_scale, right.variable, right_scale, output, -Fr(1), constant_term);
    return {output, out_value};
}

zkboostAssignedCircuitValue zkboostGenericCircuitCompilerBase::weighted_sum(
    const std::vector<zkboostAssignedCircuitValue>& indicators,
    const std::vector<std::int64_t>& weights)
{
    zkboost_require(indicators.size() == weights.size(),
                           "zkboost_generic_circuit_compiler",
                           "weighted sum size mismatch");
    if (indicators.empty()) {
        return constant(0);
    }

    zkboostAssignedCircuitValue acc =
        affine(indicators.front(), zkboost_int64_to_fr(weights.front()), Fr(0));
    for (size_t i = 1; i < indicators.size(); ++i) {
        acc = add_scaled(acc,
                         Fr(1),
                         indicators[i],
                         zkboost_int64_to_fr(weights[i]),
                         Fr(0));
    }
    return acc;
}

zkboostAssignedCircuitValue zkboostGenericCircuitCompilerBase::one_minus(
    const zkboostAssignedCircuitValue& bit)
{
    return affine(bit, Fr(-1), Fr(1));
}

void zkboostGenericCircuitCompilerBase::constrain_nonnegative_with_limbs(
    const zkboostAssignedCircuitValue& value,
    std::int64_t witness_value,
    size_t bit_width,
    size_t limb_bits)
{
    zkboost_require(witness_value >= 0,
                    "zkboost_grouped_range",
                    "witness must be non-negative");
    zkboost_require(bit_width > 0 && bit_width <= 63,
                    "zkboost_grouped_range",
                    "bit width must be in [1, 63]");
    zkboost_require(limb_bits > 0 && limb_bits <= 16,
                    "zkboost_grouped_range",
                    "limb width must be in [1, 16]");

    const std::uint64_t witness = static_cast<std::uint64_t>(witness_value);
    if (bit_width < 63) {
        zkboost_require(witness < (std::uint64_t(1) << bit_width),
                        "zkboost_grouped_range",
                        "witness exceeds configured bit width");
    }

    std::vector<zkboostAssignedCircuitValue> limbs;
    std::vector<std::int64_t> weights;
    for (size_t offset = 0; offset < bit_width; offset += limb_bits) {
        const size_t width = std::min(limb_bits, bit_width - offset);
        const std::uint64_t mask = (std::uint64_t(1) << width) - 1;
        const std::int64_t limb = static_cast<std::int64_t>((witness >> offset) & mask);

        const std::int64_t tag = -4000000000LL - static_cast<std::int64_t>(width);
        if (grouped_range_table_widths_.insert(width).second) {
            const std::uint64_t table_size = std::uint64_t(1) << width;
            for (std::uint64_t entry = 0; entry < table_size; ++entry) {
                add_lookup_table_row(static_cast<std::int64_t>(entry), tag, 0);
            }
        }

        const zkboostAssignedCircuitValue limb_value = private_value(limb);
        add_lookup(limb_value, constant(tag), constant(0));
        limbs.push_back(limb_value);
        weights.push_back(std::int64_t(1) << offset);
    }
    constrain_equal(value, weighted_sum(limbs, weights));
}

void zkboostGenericCircuitCompilerBase::constrain_nonnegative_at_most_with_limbs(
    const zkboostAssignedCircuitValue& value,
    std::int64_t witness_value,
    std::int64_t max_value,
    size_t limb_bits)
{
    zkboost_require(max_value >= 0 && witness_value >= 0 && witness_value <= max_value,
                    "zkboost_grouped_range",
                    "bounded witness is outside [0, max_value]");
    const size_t bit_width = nonnegative_bit_width(static_cast<std::uint64_t>(max_value));
    constrain_nonnegative_with_limbs(value, witness_value, bit_width, limb_bits);
    const zkboostAssignedCircuitValue slack = affine(value, Fr(-1), Fr(max_value));
    constrain_nonnegative_with_limbs(slack, max_value - witness_value, bit_width, limb_bits);
}

void zkboostGenericCircuitCompilerBase::constrain_boolean(const zkboostAssignedCircuitValue& bit)
{
    builder_.add_boolean(bit.variable);
}

void zkboostGenericCircuitCompilerBase::constrain_equal(const zkboostAssignedCircuitValue& left,
                                                        const zkboostAssignedCircuitValue& right)
{
    builder_.constrain_equal(left.variable, right.variable);
}

void zkboostGenericCircuitCompilerBase::add_lookup(const zkboostAssignedCircuitValue& left,
                                                   const zkboostAssignedCircuitValue& right,
                                                   const zkboostAssignedCircuitValue& output)
{
    builder_.add_lookup(left.variable, right.variable, output.variable);
}

void zkboostGenericCircuitCompilerBase::add_lookup_table_row(std::int64_t left,
                                                             std::int64_t right,
                                                             std::int64_t output)
{
    add_lookup_table_row(zkboost_int64_to_fr(left),
                         zkboost_int64_to_fr(right),
                         zkboost_int64_to_fr(output));
}

void zkboostGenericCircuitCompilerBase::add_lookup_table_row(const Fr& left,
                                                             const Fr& right,
                                                             const Fr& output)
{
    builder_.add_lookup_table_row(left, right, output);
}

zkboostCircuitVariable zkboostGenericCircuitCompilerBase::comparison(
    const zkboostAssignedCircuitValue& left,
    const zkboostAssignedCircuitValue& right,
    std::int64_t min_value,
    std::int64_t max_value,
    zkboostCircuitComparisonMode mode)
{
    return builder_.add_comparison(left.variable, right.variable, min_value, max_value, mode);
}
