#pragma once

#include "plonk/plonk.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct zkboostAssignedPlonkValue {
    plonkVariable variable;
    Fr value = Fr(0);
};

Fr zkboost_int64_to_fr(std::int64_t value);

void zkboost_require(bool condition, const std::string& context, const std::string& message);

class zkboostNamedPublicSpan {
public:
    size_t offset = 0;
    size_t count = 0;
};

using zkboostNamedPublicSpans = std::map<std::string, zkboostNamedPublicSpan>;

class zkboostCompiledCircuitStatement {
public:
    std::string name;
    plonkSetup setup;
    plonkWitness witness;
    std::vector<Fr> public_inputs;
    zkboostNamedPublicSpans named_public_spans;
};

class zkboostCompiledCircuitProof {
public:
    std::string name;
    plonkProofBundle bundle;
};

class zkboostCompiledCircuitVerifierMetadata {
public:
    std::string name;
    plonkVerificationKey verification_key;
    std::vector<Fr> public_inputs;
    zkboostNamedPublicSpans named_public_spans;
};

zkboostCompiledCircuitStatement make_zkboost_compiled_circuit_statement(
    const std::string& name,
    const plonkSetup& setup,
    const plonkWitness& witness,
    const std::vector<Fr>& public_inputs,
    const zkboostNamedPublicSpans& named_public_spans = {});

zkboostCompiledCircuitStatement compile_zkboost_compiled_circuit_statement(
    const std::string& name,
    const plonkCircuit& circuit,
    const plonkWitness& witness,
    const std::vector<Fr>& public_inputs,
    const kzgPublicParameters& public_parameters,
    const zkboostNamedPublicSpans& named_public_spans = {});

zkboostCompiledCircuitProof prove_zkboost_compiled_circuit(
    const zkboostCompiledCircuitStatement& statement);

zkboostCompiledCircuitVerifierMetadata prepare_zkboost_compiled_circuit_verifier_metadata(
    const zkboostCompiledCircuitStatement& statement);

bool verify_zkboost_compiled_circuit(const zkboostCompiledCircuitVerifierMetadata& verifier_metadata,
                                     const zkboostCompiledCircuitProof& proof);

std::vector<Fr> zkboost_compiled_circuit_named_public_values(
    const zkboostCompiledCircuitVerifierMetadata& metadata,
    const std::string& value_id);

bool link_zkboost_compiled_circuit_public_values(
    const zkboostCompiledCircuitVerifierMetadata& previous,
    const std::string& previous_value_id,
    const zkboostCompiledCircuitVerifierMetadata& next,
    const std::string& next_value_id);

class zkboostPlonkCompilerBase {
public:
    plonkCircuit build_circuit() const;
    const plonkWitness& witness() const noexcept;
    const std::vector<Fr>& public_inputs() const noexcept;

protected:
    zkboostAssignedPlonkValue constant(std::int64_t value);
    zkboostAssignedPlonkValue private_value(std::int64_t value);
    zkboostAssignedPlonkValue public_value(std::int64_t value);

    zkboostAssignedPlonkValue multiply(const zkboostAssignedPlonkValue& left,
                                       const zkboostAssignedPlonkValue& right);
    zkboostAssignedPlonkValue affine(const zkboostAssignedPlonkValue& left,
                                     const Fr& left_scale,
                                     const Fr& constant_term);
    zkboostAssignedPlonkValue add_scaled(const zkboostAssignedPlonkValue& left,
                                         const Fr& left_scale,
                                         const zkboostAssignedPlonkValue& right,
                                         const Fr& right_scale,
                                         const Fr& constant_term);
    zkboostAssignedPlonkValue weighted_sum(const std::vector<zkboostAssignedPlonkValue>& indicators,
                                           const std::vector<std::int64_t>& weights);
    zkboostAssignedPlonkValue one_minus(const zkboostAssignedPlonkValue& bit);

    void constrain_boolean(const zkboostAssignedPlonkValue& bit);
    void constrain_equal(const zkboostAssignedPlonkValue& left,
                         const zkboostAssignedPlonkValue& right);

    void add_lookup(const zkboostAssignedPlonkValue& left,
                    const zkboostAssignedPlonkValue& right,
                    const zkboostAssignedPlonkValue& output);
    void add_lookup_table_row(std::int64_t left, std::int64_t right, std::int64_t output);
    void add_lookup_table_row(const Fr& left, const Fr& right, const Fr& output);

    plonkCircuitBuilder builder_;
    plonkWitness witness_;

private:
    std::vector<Fr> public_inputs_;
    std::map<std::int64_t, zkboostAssignedPlonkValue> constant_cache_;
};
