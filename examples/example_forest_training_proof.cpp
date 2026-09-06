#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_training_demo_data.hpp"
#include "forest_training_workflow.hpp"
#include "test_kzg_srs.hpp"

#include <iostream>

using namespace mcl::bn;

int main()
{
    init_groups();

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace trace = build_demo_training_trace();
    const forestTrainingProofInputs inputs = assemble_forest_training_inputs_from_trace(trace);
    const forestTrainingProofPublicInputs public_inputs =
        forest_training_proof_public_inputs_from_inputs(public_params, inputs);

    const ProofArtifact artifact =
        prove_forest_training_proof_with_runtime(public_params, inputs, "example.forest_training");
    const VerificationResult result =
        verify_forest_training_proof_with_runtime(
            public_params, artifact, public_inputs, "example.forest_training");

    if (!result.accepted || !result.proof_fully_consumed) {
        std::cerr << "forest training proof verification failed" << std::endl;
        return 1;
    }

    std::cout << "forest training proof succeeded" << std::endl;
    return 0;
}
