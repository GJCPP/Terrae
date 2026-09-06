#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_training_workflow.hpp"
#include "test_kzg_srs.hpp"
#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mcl::bn;

namespace {

poly poly_from_domain_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

forestTrainingProofInputs valid_inputs()
{
    return build_demo_training_inputs();
}

forestTrainingProofInputs inputs_with_leaf_candidate_at_threshold(size_t node)
{
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    const size_t column_count = trace.feature_count * trace.bin_count;
    const size_t candidate_column = 1;

    // Keep the sample routed to the same leaf while giving that leaf values on
    // both sides of candidate_column. Reassembling the trace keeps histogram,
    // split-statistic, gain, and selector witnesses mutually consistent.
    trace.training_inputs.back() = 2;
    trace.candidate_mask[node * column_count + candidate_column] = 1;
    trace.selected_candidate_indices[node] = candidate_column;

    forestTrainingProofInputs inputs = assemble_forest_training_inputs_from_trace(trace);
    const std::vector<Fr> selected_gains =
        inputs.selected_gains.evaluations_on_domain(inputs.node_count);
    inputs.split_gain_threshold = selected_gains[node];
    return inputs;
}

} // namespace

class ForestTrainingWorkflowTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTrainingWorkflowTest, HelperSupportsProveAndVerify)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);
    const forestTrainingProofInputs inputs = valid_inputs();
    const forestTrainingProofPublicInputs public_inputs =
        forest_training_proof_public_inputs_from_inputs(public_params, inputs);

    const ProofArtifact artifact =
        prove_forest_training_proof_with_runtime(public_params, inputs, "workflow.training");
    const VerificationResult result =
        verify_forest_training_proof_with_runtime(public_params, artifact, public_inputs, "workflow.training");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestTrainingWorkflowTest, HelperReusesPrecomputedCommittedForestPublicCommitments)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);
    const forestTrainingProofInputs inputs = valid_inputs();
    const forestCommittedNodeTable committed_forest = forest_committed_node_table_from_training_inputs(inputs);
    const forestCommittedNodePublicCommitments committed_public =
        commit_committed_forest_public_commitments(public_params, committed_forest, "workflow.precomputed_forest");
    const forestTrainingProofPublicInputs public_inputs =
        forest_training_proof_public_inputs_from_inputs(committed_public, inputs);

    const ProofArtifact artifact = prove_forest_training_proof_with_runtime(
        public_params, inputs, committed_public, "workflow.training.precomputed");
    const VerificationResult result = verify_forest_training_proof_with_runtime(
        public_params, artifact, public_inputs, "workflow.training.precomputed");

    EXPECT_TRUE(result.accepted);
    EXPECT_TRUE(result.proof_fully_consumed);
}

TEST_F(ForestTrainingWorkflowTest, VerifierRejectsTamperedCommittedDepth)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);
    const forestTrainingProofInputs inputs = valid_inputs();
    forestTrainingProofPublicInputs public_inputs =
        forest_training_proof_public_inputs_from_inputs(public_params, inputs);
    const ProofArtifact artifact =
        prove_forest_training_proof_with_runtime(public_params, inputs, "workflow.training.depth_tamper");

    public_inputs.committed_forest.node_depths += Group1::one();
    const VerificationResult result = verify_forest_training_proof_with_runtime(
        public_params, artifact, public_inputs, "workflow.training.depth_tamper");

    EXPECT_FALSE(result.accepted && result.proof_fully_consumed);
}

TEST_F(ForestTrainingWorkflowTest, WrongResidualChannelMakesWorkflowFail)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);
    forestTrainingProofInputs inputs = valid_inputs();
    inputs.channels[1].query_values = poly_from_domain_ints({1, 0, 0, 0});

    ProofRuntime prover = ProofRuntime::prover(public_params);
    EXPECT_FALSE(run_forest_training_proof_workflow(prover, inputs, "workflow.bad_training"));
}

TEST_F(ForestTrainingWorkflowTest, InternalSplitBelowThresholdMakesWorkflowFail)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);
    forestTrainingProofInputs inputs = valid_inputs();
    inputs.split_gain_threshold = fr_from_int(100);

    ProofRuntime prover = ProofRuntime::prover(public_params);
    EXPECT_FALSE(run_forest_training_proof_workflow(prover, inputs, "workflow.split_below_threshold"));
}

TEST_F(ForestTrainingWorkflowTest, LeafCandidateAtThresholdMakesWorkflowFail)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(8192, 8192);
    const forestTrainingProofInputs inputs = inputs_with_leaf_candidate_at_threshold(2);

    ProofRuntime prover = ProofRuntime::prover(public_params);
    EXPECT_FALSE(run_forest_training_proof_workflow(prover, inputs, "workflow.leaf_candidate_at_threshold"));
}

TEST_F(ForestTrainingWorkflowTest, SrsRequirementTracksDegreeShiftAndOpeningFloorSeparately)
{
    kzgSrsRequirement requirement;
    update_kzg_srs_requirement(requirement, 32);
    update_kzg_srs_requirement(requirement, 8);
    require_kzg_distinct_opening_points(requirement, 1);

    EXPECT_EQ(requirement.deg1_plus_1, 32u);
    EXPECT_EQ(requirement.deg2_plus_1, 25u);
}

TEST_F(ForestTrainingWorkflowTest, ConstantDegreeClaimCanForceDeg2BackToDeg1)
{
    kzgSrsRequirement requirement;
    update_kzg_srs_requirement(requirement, 32);
    update_kzg_srs_requirement(requirement, 8);
    require_kzg_constant_degree_claim(requirement);

    EXPECT_EQ(requirement.deg1_plus_1, 32u);
    EXPECT_EQ(requirement.deg2_plus_1, 32u);
}

TEST_F(ForestTrainingWorkflowTest, SrsRequirementUsesProtocolAwareRangeLookupEstimate)
{
    forestTrainingProofInputs inputs = valid_inputs();
    inputs.iteration_count = 1;
    inputs.class_count = 8;
    inputs.tree_count = 8;
    inputs.node_count = 256;
    inputs.feature_count = 16;
    inputs.bin_count = 8;
    inputs.query_count = 512;

    const kzgSrsRequirement requirement = required_forest_training_proof_srs(inputs);

    EXPECT_GE(requirement.deg1_plus_1, 1048576u);
    EXPECT_GE(requirement.deg2_plus_1, 1048576u);
    EXPECT_LT(requirement.deg1_plus_1, 16777216u);
    EXPECT_LT(requirement.deg2_plus_1, 16777216u);
}
