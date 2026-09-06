#include "bls12_381_wrapper.hpp"
#include "forest_training_demo_data.hpp"
#include "forest_training_then_inference_workflow.hpp"
#include "test_kzg_srs.hpp"

#include <iostream>

using namespace mcl::bn;

int main()
{
    init_groups();

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace trace = build_demo_training_trace();
    const forestSemanticInferenceQueryBatch inference_trace = build_demo_inference_query_trace();
    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(trace, inference_trace);

    const forestTrainingThenInferencePreparedRun prepared =
        prove_and_prepare_training_then_inference_with_runtime(
            public_params, inputs, "example.training_then_inference");
    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, prepared.verifier_bundle, "example.training_then_inference");

    if (!result.accepted || !result.proof_fully_consumed) {
        std::cerr << "combined training-then-inference verification failed" << std::endl;
        return 1;
    }

    std::cout << "training proof succeeded and exported inference proof succeeded" << std::endl;
    return 0;
}
