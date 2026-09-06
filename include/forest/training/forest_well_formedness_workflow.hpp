#pragma once

#include <memory>
#include <string>

#include "forest_training_export.hpp"
#include "proof_runtime.hpp"
#include "training_runtime_utils.hpp"

class forestWellFormednessInputs {
public:
    forestCommittedNodeTable committed_forest;
    poly left_lower;
    poly left_upper;
    poly right_lower;
    poly right_upper;
    size_t full_upper_bound = 0;
    size_t structure_bit_bound = 0;
    // One-based maximum node depth. The committed root has depth 1 and every
    // active committed node must have depth at most max_depth.
    size_t max_depth = 0;
};

class forestWellFormednessPublicInputs {
public:
    forestCommittedNodePublicCommitments committed_forest;
    size_t full_upper_bound = 0;
    size_t structure_bit_bound = 0;
    size_t max_depth = 0;
};

class forestWellFormednessVerifierPreprocessing {
public:
    std::unique_ptr<ProofRuntime> runtime;
    std::unique_ptr<ProofProgram> program;
};

forestWellFormednessInputs forest_well_formedness_inputs_from_training_inputs(
    const forestTrainingProofInputs& inputs);

forestWellFormednessPublicInputs forest_well_formedness_public_inputs_from_training_public_inputs(
    const forestTrainingProofPublicInputs& inputs);

forestWellFormednessPublicInputs forest_well_formedness_public_inputs_from_inputs(
    const kzgPublicParameters& public_params,
    const forestWellFormednessInputs& inputs,
    const std::string& forest_key_prefix = "committed_forest");

forestWellFormednessPublicInputs forest_well_formedness_public_inputs_from_inputs(
    const forestCommittedNodePublicCommitments& committed_forest,
    const forestWellFormednessInputs& inputs);

kzgSrsRequirement required_forest_well_formedness_srs(const forestWellFormednessInputs& inputs);

kzgSrsRequirement required_forest_well_formedness_srs(const forestWellFormednessPublicInputs& inputs);

bool add_forest_well_formedness_workflow_to_program(
    ProofRuntime& runtime,
    ProofProgram& program,
    const forestWellFormednessInputs& inputs,
    const std::string& label_prefix = "forest_well_formedness",
    const forestCommittedNodePublicCommitments* committed_forest_public_inputs = nullptr);

bool add_forest_well_formedness_verifier_workflow_to_program(
    ProofRuntime& runtime,
    ProofProgram& program,
    const forestWellFormednessPublicInputs& inputs,
    const std::string& label_prefix = "forest_well_formedness");

bool run_forest_well_formedness_workflow(
    ProofRuntime& runtime,
    const forestWellFormednessInputs& inputs,
    const std::string& label_prefix = "forest_well_formedness",
    const forestCommittedNodePublicCommitments* committed_forest_public_inputs = nullptr);

bool run_forest_well_formedness_verifier_workflow(
    ProofRuntime& runtime,
    const forestWellFormednessPublicInputs& inputs,
    const std::string& label_prefix = "forest_well_formedness");

ProofArtifact prove_forest_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const forestWellFormednessInputs& inputs,
    const std::string& label_prefix = "forest_well_formedness");

ProofArtifact prove_forest_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const forestWellFormednessInputs& inputs,
    const forestCommittedNodePublicCommitments& committed_forest_public_inputs,
    const std::string& label_prefix = "forest_well_formedness");

VerificationResult verify_forest_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestWellFormednessInputs& inputs,
    const std::string& label_prefix = "forest_well_formedness");

VerificationResult verify_forest_well_formedness_with_runtime(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestWellFormednessPublicInputs& inputs,
    const std::string& label_prefix = "forest_well_formedness");

forestWellFormednessVerifierPreprocessing prepare_forest_well_formedness_verifier_preprocessing(
    const kzgPublicParameters& public_params,
    const ProofArtifact& artifact,
    const forestWellFormednessPublicInputs& inputs,
    const std::string& label_prefix = "forest_well_formedness");

VerificationResult verify_forest_well_formedness_preprocessed(
    forestWellFormednessVerifierPreprocessing& preprocessing);
