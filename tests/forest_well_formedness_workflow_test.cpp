#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_well_formedness_workflow.hpp"
#include "test_kzg_srs.hpp"
#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

namespace {

forestTrainingProofInputs valid_training_inputs()
{
    return build_demo_training_inputs();
}

forestWellFormednessInputs valid_well_formedness_inputs()
{
    return forest_well_formedness_inputs_from_training_inputs(valid_training_inputs());
}

} // namespace

class ForestWellFormednessWorkflowTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestWellFormednessWorkflowTest, HelperSupportsProveAndVerify)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    const forestWellFormednessPublicInputs public_inputs =
        forest_well_formedness_public_inputs_from_inputs(public_params, inputs);

    const ProofArtifact artifact =
        prove_forest_well_formedness_with_runtime(public_params, inputs, "workflow.well_formedness");
    const VerificationResult result =
        verify_forest_well_formedness_with_runtime(public_params, artifact, public_inputs, "workflow.well_formedness");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestWellFormednessWorkflowTest, HelperReusesPrecomputedCommittedForestPublicCommitments)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    const forestCommittedNodePublicCommitments committed_public =
        commit_committed_forest_public_commitments(public_params, inputs.committed_forest, "workflow.precomputed_forest");
    const forestWellFormednessPublicInputs public_inputs =
        forest_well_formedness_public_inputs_from_inputs(committed_public, inputs);

    const ProofArtifact artifact = prove_forest_well_formedness_with_runtime(
        public_params, inputs, committed_public, "workflow.well_formedness.precomputed");
    const VerificationResult result = verify_forest_well_formedness_with_runtime(
        public_params, artifact, public_inputs, "workflow.well_formedness.precomputed");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestWellFormednessWorkflowTest, PreparedVerifierAcceptsValidProof)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    const forestCommittedNodePublicCommitments committed_public =
        commit_committed_forest_public_commitments(public_params, inputs.committed_forest, "workflow.prepared_forest");
    const forestWellFormednessPublicInputs public_inputs =
        forest_well_formedness_public_inputs_from_inputs(committed_public, inputs);

    const ProofArtifact artifact = prove_forest_well_formedness_with_runtime(
        public_params, inputs, committed_public, "workflow.well_formedness.prepared");
    forestWellFormednessVerifierPreprocessing verifier =
        prepare_forest_well_formedness_verifier_preprocessing(
            public_params, artifact, public_inputs, "workflow.well_formedness.prepared");
    const VerificationResult result =
        verify_forest_well_formedness_preprocessed(verifier);

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestWellFormednessWorkflowTest, VerifierRejectsTamperedCommittedDepth)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    forestWellFormednessPublicInputs public_inputs =
        forest_well_formedness_public_inputs_from_inputs(public_params, inputs);
    const ProofArtifact artifact =
        prove_forest_well_formedness_with_runtime(public_params, inputs, "workflow.well_formedness.tamper");

    public_inputs.committed_forest.node_depths += Group1::one();
    const VerificationResult result =
        verify_forest_well_formedness_with_runtime(public_params, artifact, public_inputs, "workflow.well_formedness.tamper");

    EXPECT_FALSE(result.accepted && result.proof_fully_consumed);
}

TEST_F(ForestWellFormednessWorkflowTest, WrongChildBoxWitnessMakesWorkflowFail)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    inputs.left_lower = inputs.right_upper;

    ProofRuntime prover = ProofRuntime::prover(public_params);
    EXPECT_FALSE(run_forest_well_formedness_workflow(prover, inputs, "workflow.bad_well_formedness"));
}

TEST_F(ForestWellFormednessWorkflowTest, MaximumDepthBelowCommittedDepthMakesWorkflowFail)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    inputs.max_depth = 1;

    ProofRuntime prover = ProofRuntime::prover(public_params);
    EXPECT_FALSE(run_forest_well_formedness_workflow(
        prover, inputs, "workflow.depth_exceeds_maximum"));
}

TEST_F(ForestWellFormednessWorkflowTest, NonZeroInternalNodeScoreMakesWorkflowFail)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    inputs.committed_forest.node_leaf_values =
        example_demo_poly_from_domain_ints({7, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(public_params);
    EXPECT_FALSE(run_forest_well_formedness_workflow(
        prover, inputs, "workflow.non_zero_internal_score"));
}

TEST_F(ForestWellFormednessWorkflowTest, NonBooleanSplitMaskMakesWorkflowFail)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    inputs.committed_forest.split_node_mask =
        example_demo_poly_from_domain_ints({1, 0, 0, -1});
    inputs.committed_forest.node_leaf_mask =
        example_demo_poly_from_domain_ints({0, 1, 1, 1});

    ProofRuntime prover = ProofRuntime::prover(public_params);
    EXPECT_FALSE(run_forest_well_formedness_workflow(
        prover, inputs, "workflow.non_boolean_split_mask"));
}

TEST_F(ForestWellFormednessWorkflowTest, VerifierRejectsTamperedMaximumDepth)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestWellFormednessInputs inputs = valid_well_formedness_inputs();
    forestWellFormednessPublicInputs public_inputs =
        forest_well_formedness_public_inputs_from_inputs(public_params, inputs);
    const ProofArtifact artifact = prove_forest_well_formedness_with_runtime(
        public_params, inputs, "workflow.well_formedness.max_depth_tamper");

    public_inputs.max_depth = 1;
    const VerificationResult result = verify_forest_well_formedness_with_runtime(
        public_params, artifact, public_inputs, "workflow.well_formedness.max_depth_tamper");

    EXPECT_FALSE(result.accepted && result.proof_fully_consumed);
}
