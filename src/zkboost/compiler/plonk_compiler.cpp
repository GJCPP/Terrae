#include "zkboost/compiler/plonk_compiler.hpp"

#include <stdexcept>

Fr zkboost_int64_to_fr(std::int64_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

void zkboost_require(bool condition, const std::string& context, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(context + ": " + message);
    }
}

namespace {

std::vector<Fr> slice_public_inputs_or_throw(const std::vector<Fr>& public_inputs,
                                             const zkboostNamedPublicSpan& span,
                                             const std::string& circuit_name,
                                             const std::string& value_id)
{
    if (span.offset > public_inputs.size() || span.count > public_inputs.size() - span.offset) {
        throw std::runtime_error("zkboost_compiled_circuit(" + circuit_name +
                                 "): named public value '" + value_id + "' is out of range");
    }
    return std::vector<Fr>(public_inputs.begin() + static_cast<std::ptrdiff_t>(span.offset),
                           public_inputs.begin() + static_cast<std::ptrdiff_t>(span.offset + span.count));
}

zkboostNamedPublicSpan named_public_span_or_throw(const zkboostNamedPublicSpans& named_public_spans,
                                                  const std::string& circuit_name,
                                                  const std::string& value_id)
{
    const auto it = named_public_spans.find(value_id);
    if (it == named_public_spans.end()) {
        throw std::runtime_error("zkboost_compiled_circuit(" + circuit_name +
                                 "): missing named public value '" + value_id + "'");
    }
    return it->second;
}

} // namespace

zkboostCompiledCircuitStatement make_zkboost_compiled_circuit_statement(
    const std::string& name,
    const plonkSetup& setup,
    const plonkWitness& witness,
    const std::vector<Fr>& public_inputs,
    const zkboostNamedPublicSpans& named_public_spans)
{
    zkboost_require(!name.empty(),
                    "zkboost_compiled_circuit",
                    "statement name must not be empty");

    zkboostCompiledCircuitStatement out;
    out.name = name;
    out.setup = setup;
    out.witness = witness;
    out.public_inputs = public_inputs;
    out.named_public_spans = named_public_spans;
    return out;
}

zkboostCompiledCircuitStatement compile_zkboost_compiled_circuit_statement(
    const std::string& name,
    const plonkCircuit& circuit,
    const plonkWitness& witness,
    const std::vector<Fr>& public_inputs,
    const kzgPublicParameters& public_parameters,
    const zkboostNamedPublicSpans& named_public_spans)
{
    return make_zkboost_compiled_circuit_statement(
        name,
        preprocess_plonk(circuit, public_parameters),
        witness,
        public_inputs,
        named_public_spans);
}

zkboostCompiledCircuitProof prove_zkboost_compiled_circuit(
    const zkboostCompiledCircuitStatement& statement)
{
    zkboostCompiledCircuitProof out;
    out.name = statement.name;
    out.bundle = prove_plonk_bundle(statement.setup.proving_key,
                                    statement.witness,
                                    statement.public_inputs);
    return out;
}

zkboostCompiledCircuitVerifierMetadata prepare_zkboost_compiled_circuit_verifier_metadata(
    const zkboostCompiledCircuitStatement& statement)
{
    zkboostCompiledCircuitVerifierMetadata out;
    out.name = statement.name;
    out.verification_key = statement.setup.verification_key;
    out.public_inputs = statement.public_inputs;
    out.named_public_spans = statement.named_public_spans;
    return out;
}

bool verify_zkboost_compiled_circuit(
    const zkboostCompiledCircuitVerifierMetadata& verifier_metadata,
    const zkboostCompiledCircuitProof& proof)
{
    if (verifier_metadata.name != proof.name) {
        return false;
    }
    return verify_plonk_bundle(verifier_metadata.verification_key,
                               verifier_metadata.public_inputs,
                               proof.bundle);
}

std::vector<Fr> zkboost_compiled_circuit_named_public_values(
    const zkboostCompiledCircuitVerifierMetadata& metadata,
    const std::string& value_id)
{
    return slice_public_inputs_or_throw(metadata.public_inputs,
                                        named_public_span_or_throw(metadata.named_public_spans,
                                                                   metadata.name,
                                                                   value_id),
                                        metadata.name,
                                        value_id);
}

bool link_zkboost_compiled_circuit_public_values(
    const zkboostCompiledCircuitVerifierMetadata& previous,
    const std::string& previous_value_id,
    const zkboostCompiledCircuitVerifierMetadata& next,
    const std::string& next_value_id)
{
    return zkboost_compiled_circuit_named_public_values(previous, previous_value_id) ==
           zkboost_compiled_circuit_named_public_values(next, next_value_id);
}

plonkCircuit zkboostPlonkCompilerBase::build_circuit() const
{
    return builder_.build();
}

const plonkWitness& zkboostPlonkCompilerBase::witness() const noexcept
{
    return witness_;
}

const std::vector<Fr>& zkboostPlonkCompilerBase::public_inputs() const noexcept
{
    return public_inputs_;
}

zkboostAssignedPlonkValue zkboostPlonkCompilerBase::constant(std::int64_t value)
{
    const auto it = constant_cache_.find(value);
    if (it != constant_cache_.end()) {
        return it->second;
    }

    const Fr field_value = zkboost_int64_to_fr(value);
    const plonkVariable variable = builder_.add_constant(field_value);
    witness_.set(variable, field_value);

    const zkboostAssignedPlonkValue out{variable, field_value};
    constant_cache_.emplace(value, out);
    return out;
}

zkboostAssignedPlonkValue zkboostPlonkCompilerBase::private_value(std::int64_t value)
{
    const Fr field_value = zkboost_int64_to_fr(value);
    const plonkVariable variable = builder_.add_private_variable();
    witness_.set(variable, field_value);
    return {variable, field_value};
}

zkboostAssignedPlonkValue zkboostPlonkCompilerBase::public_value(std::int64_t value)
{
    const Fr field_value = zkboost_int64_to_fr(value);
    const plonkVariable variable = builder_.add_public_input();
    witness_.set(variable, field_value);
    public_inputs_.push_back(field_value);
    return {variable, field_value};
}

zkboostAssignedPlonkValue zkboostPlonkCompilerBase::multiply(
    const zkboostAssignedPlonkValue& left,
    const zkboostAssignedPlonkValue& right)
{
    const Fr out_value = left.value * right.value;
    const plonkVariable output = builder_.add_multiplication(left.variable, right.variable);
    witness_.set(output, out_value);
    return {output, out_value};
}

zkboostAssignedPlonkValue zkboostPlonkCompilerBase::affine(
    const zkboostAssignedPlonkValue& left,
    const Fr& left_scale,
    const Fr& constant_term)
{
    const Fr out_value = (left.value * left_scale) + constant_term;
    const plonkVariable output = builder_.add_private_variable();
    witness_.set(output, out_value);
    builder_.add_gate(left.variable,
                      left.variable,
                      output,
                      Fr(0),
                      left_scale,
                      Fr(0),
                      -Fr(1),
                      constant_term);
    return {output, out_value};
}

zkboostAssignedPlonkValue zkboostPlonkCompilerBase::add_scaled(
    const zkboostAssignedPlonkValue& left,
    const Fr& left_scale,
    const zkboostAssignedPlonkValue& right,
    const Fr& right_scale,
    const Fr& constant_term)
{
    const Fr out_value = (left.value * left_scale) + (right.value * right_scale) + constant_term;
    const plonkVariable output = builder_.add_private_variable();
    witness_.set(output, out_value);
    builder_.add_gate(left.variable,
                      right.variable,
                      output,
                      Fr(0),
                      left_scale,
                      right_scale,
                      -Fr(1),
                      constant_term);
    return {output, out_value};
}

zkboostAssignedPlonkValue zkboostPlonkCompilerBase::weighted_sum(
    const std::vector<zkboostAssignedPlonkValue>& indicators,
    const std::vector<std::int64_t>& weights)
{
    zkboost_require(indicators.size() == weights.size(),
                    "zkboost_plonk_compiler",
                    "weighted sum size mismatch");
    if (indicators.empty()) {
        return constant(0);
    }

    zkboostAssignedPlonkValue acc =
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

zkboostAssignedPlonkValue zkboostPlonkCompilerBase::one_minus(
    const zkboostAssignedPlonkValue& bit)
{
    return affine(bit, Fr(-1), Fr(1));
}

void zkboostPlonkCompilerBase::constrain_boolean(const zkboostAssignedPlonkValue& bit)
{
    builder_.add_gate(bit.variable, bit.variable, bit.variable, Fr(1), -Fr(1), Fr(0), Fr(0), Fr(0));
}

void zkboostPlonkCompilerBase::constrain_equal(const zkboostAssignedPlonkValue& left,
                                               const zkboostAssignedPlonkValue& right)
{
    builder_.constrain_equal(left.variable, right.variable);
}

void zkboostPlonkCompilerBase::add_lookup(const zkboostAssignedPlonkValue& left,
                                          const zkboostAssignedPlonkValue& right,
                                          const zkboostAssignedPlonkValue& output)
{
    builder_.add_lookup(left.variable, right.variable, output.variable);
}

void zkboostPlonkCompilerBase::add_lookup_table_row(std::int64_t left,
                                                    std::int64_t right,
                                                    std::int64_t output)
{
    add_lookup_table_row(zkboost_int64_to_fr(left),
                         zkboost_int64_to_fr(right),
                         zkboost_int64_to_fr(output));
}

void zkboostPlonkCompilerBase::add_lookup_table_row(const Fr& left,
                                                    const Fr& right,
                                                    const Fr& output)
{
    builder_.add_lookup_table_row(left, right, output);
}
