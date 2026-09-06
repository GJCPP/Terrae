#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/validate_forest_fetch.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedValidateForestTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

namespace {

forestTrainingSemanticTrace two_class_training_trace()
{
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    trace.tree_count = 2;
    trace.node_count = 8;
    trace.iteration_count = 1;
    trace.class_count = 2;
    trace.query_count = 4;
    trace.feature_count = 1;
    trace.bin_count = 8;

    trace.root_indices = {0, 4};
    trace.split_node_mask = {1, 0, 0, 0, 1, 0, 0, 0};
    trace.left_child_indices = {1, 1, 2, 3, 5, 5, 6, 7};
    trace.right_child_indices = {2, 1, 2, 3, 6, 5, 6, 7};
    trace.node_tree_ids = {1, 1, 1, 0, 2, 2, 2, 0};
    trace.feature_selector = {1, 0, 0, 0, 1, 0, 0, 0};
    trace.thresholds = {2, 0, 0, 0, 2, 0, 0, 0};
    trace.node_leaf_mask = {0, 1, 1, 0, 0, 1, 1, 0};
    trace.node_leaf_values = {0, 5, -5, 0, 0, -5, 5, 0};

    trace.training_inputs = {0, 1, 2, 3};
    trace.base_labels = {
        1, 1, 0, 0,
        0, 0, 1, 1,
    };
    trace.leaf_indices = {
        1, 1, 2, 2,
        5, 5, 6, 6,
    };
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);
    trace.selected_candidate_indices.assign(trace.node_count, 0);
    trace.selected_candidate_indices[0] = 1;
    trace.selected_candidate_indices[4] = 1;
    return trace;
}

forestSemanticInferenceQueryBatch two_class_inference_trace()
{
    forestSemanticInferenceQueryBatch trace;
    trace.query_count = 8;
    trace.feature_count = 1;
    trace.query_tree_ids = {1, 1, 1, 1, 2, 2, 2, 2};
    trace.lifted_inputs = {
        0, 1, 2, 3,
        0, 1, 2, 3,
    };
    return trace;
}

} // namespace

TEST_F(ZkboostSharedValidateForestTest, BuildsValidateForestViewFromSharedTrace)
{
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(two_class_training_trace(), two_class_inference_trace());

    EXPECT_EQ(view.summary.tree_count, 2u);
    EXPECT_EQ(view.summary.sample_count, 4u);
    EXPECT_EQ(view.summary.feature_count, 1u);
    EXPECT_EQ(view.summary.max_depth, 1u);
    EXPECT_EQ(view.summary.checked_paths, 8u);
    EXPECT_EQ(view.summary.split_node_count, 2u);
    EXPECT_EQ(view.summary.leaf_node_count, 4u);
    EXPECT_EQ(view.routing.summary.checked_paths, 8u);
    EXPECT_EQ(view.fetch.summary.split_node_count, 2u);
    EXPECT_EQ(view.fetch.public_split_masks, std::vector<std::int64_t>({1, 0, 0, 0, 1, 0, 0, 0}));
    EXPECT_EQ(view.fetch.public_split_features, std::vector<std::int64_t>({0, 0, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(view.fetch.public_split_thresholds, std::vector<std::int64_t>({2, 0, 0, 0, 2, 0, 0, 0}));
    ASSERT_EQ(view.fetch.split_field_reads.size(), 2u);
    EXPECT_EQ(view.fetch.split_field_reads[0].node, 0u);
    EXPECT_EQ(view.fetch.split_field_reads[0].tree, 0u);
    EXPECT_EQ(view.fetch.split_field_reads[0].internal_node_index, 0u);
    EXPECT_EQ(view.fetch.split_field_reads[0].feature, 0);
    EXPECT_EQ(view.fetch.split_field_reads[0].threshold, 2);
    ASSERT_EQ(view.routing.path_claims.size(), 8u);
    EXPECT_EQ(view.routing.path_claims[0].trace_leaf_index, 1);
    EXPECT_EQ(view.routing.path_claims[0].logical_leaf_id, 1);
    EXPECT_EQ(view.routing.path_claims[0].path_features, std::vector<std::int64_t>({0}));
    EXPECT_EQ(view.routing.path_claims[0].path_thresholds, std::vector<std::int64_t>({2}));
    EXPECT_EQ(view.routing.path_claims[0].permuted_inputs, std::vector<std::int64_t>({0}));
    EXPECT_EQ(view.routing.path_claims[0].path_branch_bits, std::vector<std::int64_t>({0}));
    EXPECT_EQ(view.routing.path_claims[3].logical_leaf_id, 2);
    EXPECT_EQ(view.routing.path_claims[4].tree_score, -5);
    EXPECT_EQ(view.routing.path_claims[7].tree_score, 5);
    EXPECT_EQ(view.routing.public_query_inputs, two_class_inference_trace().lifted_inputs);
    ASSERT_EQ(view.fetch.path_reads.size(), 8u);
    EXPECT_EQ(view.fetch.path_reads[0].query, 0u);
    EXPECT_EQ(view.fetch.path_reads[0].logical_leaf_id, 1);
    EXPECT_EQ(view.fetch.path_reads[0].internal_node_index, 0u);
    EXPECT_EQ(view.fetch.path_reads[0].feature, 0);
    EXPECT_EQ(view.fetch.path_reads[7].threshold, 2);
    ASSERT_EQ(view.fetch.leaf_value_reads.size(), 8u);
    EXPECT_EQ(view.fetch.leaf_value_reads[0].logical_leaf_id, 1);
    EXPECT_EQ(view.fetch.leaf_value_reads[7].tree_score, 5);
}

TEST_F(ZkboostSharedValidateForestTest, BuildsRoutingAndFetchViewsSeparately)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();

    const zkboostSharedValidateForestRoutingView routing =
        build_zkboost_shared_validate_forest_routing_view(training, inference);
    const zkboostSharedValidateForestFetchView fetch =
        build_zkboost_shared_validate_forest_fetch_view(training, inference);

    EXPECT_EQ(routing.summary.checked_paths, 8u);
    EXPECT_EQ(routing.trace_leaf_indices, training.leaf_indices);
    EXPECT_EQ(routing.logical_leaf_ids, std::vector<std::int64_t>({1, 1, 2, 2, 1, 1, 2, 2}));
    EXPECT_EQ(routing.public_query_inputs, inference.lifted_inputs);
    ASSERT_EQ(routing.path_claims.size(), 8u);
    EXPECT_EQ(routing.path_claims[0].path_features, std::vector<std::int64_t>({0}));
    EXPECT_EQ(routing.path_claims[0].permuted_inputs, std::vector<std::int64_t>({0}));
    EXPECT_EQ(routing.path_claims[4].trace_leaf_index, 5);

    EXPECT_EQ(fetch.summary.node_count, training.node_count);
    ASSERT_EQ(fetch.path_reads.size(), routing.summary.checked_paths);
    ASSERT_EQ(fetch.leaf_value_reads.size(), routing.summary.checked_paths);
    EXPECT_EQ(fetch.path_reads[0].logical_leaf_id, 1);
    EXPECT_EQ(fetch.path_reads[0].threshold, 2);
    EXPECT_EQ(fetch.leaf_value_reads[4].logical_leaf_id, 1);
}

TEST_F(ZkboostSharedValidateForestTest, RoutesEarlyLeafThroughInputIndependentDummyPath)
{
    forestTrainingSemanticTrace training = two_class_training_trace();
    training.split_node_mask[4] = 0;
    training.node_leaf_mask[4] = 1;
    training.node_leaf_values[4] = 9;
    for (size_t query = 4; query < 8; ++query) {
        training.leaf_indices[query] = 4;
    }

    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, two_class_inference_trace());

    EXPECT_EQ(view.routing.logical_leaf_ids, std::vector<std::int64_t>({1, 1, 2, 2, 2, 2, 2, 2}));
    EXPECT_EQ(view.routing.tree_scores, std::vector<std::int64_t>({5, 5, -5, -5, 9, 9, 9, 9}));
    EXPECT_EQ(view.routing.path_claims[4].path_thresholds,
              std::vector<std::int64_t>({0}));
    EXPECT_EQ(view.routing.path_claims[4].path_branch_bits,
              std::vector<std::int64_t>({1}));
}

TEST_F(ZkboostSharedValidateForestTest, RejectsBadLeafIndexClaim)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.leaf_indices[0] = 2;

    EXPECT_THROW(
        (void)validate_zkboost_shared_forest(trace, two_class_inference_trace()),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateForestTest, RejectsThresholdThatChangesRouting)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.thresholds[0] = 0;

    EXPECT_THROW(
        (void)validate_zkboost_shared_forest(trace, two_class_inference_trace()),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateForestTest, RejectsChildPointerIntoWrongTree)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.right_child_indices[0] = 6;

    EXPECT_THROW(
        (void)validate_zkboost_shared_forest(trace, two_class_inference_trace()),
        std::runtime_error);
}

TEST_F(ZkboostSharedValidateForestTest, LeafValueChangeChangesTreeScoreClaims)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.node_leaf_values[6] = -5;

    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(trace, two_class_inference_trace());
    EXPECT_EQ(view.routing.tree_scores, std::vector<std::int64_t>({5, 5, -5, -5, -5, -5, -5, -5}));
    EXPECT_EQ(view.routing.logical_leaf_ids, std::vector<std::int64_t>({1, 1, 2, 2, 1, 1, 2, 2}));
}

TEST_F(ZkboostSharedValidateForestTest, ProvesAndVerifiesValidateForestCircuit)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance(training, inference, public_params);

    EXPECT_EQ(instance.public_inputs, encode_zkboost_shared_validate_forest_public_inputs(view));
    EXPECT_EQ(instance.summary.checked_paths, 8u);
    EXPECT_EQ(instance.committed_table.comparison_row_count, 64u);
    // The fixed table also contains one tagged input row per query/feature.
    EXPECT_EQ(instance.committed_table.total_lookup_row_count, 72u);
    // Path feature selectors are now checked by lookup rather than public one-hot vectors.
    EXPECT_EQ(instance.public_inputs.size(), 96u);
    EXPECT_TRUE(verify_zkboost_shared_validate_forest_committed_table(
        instance.setup.verification_key,
        instance.committed_table));

    const plonkProof proof = prove_zkboost_shared_validate_forest_plonk(instance);

    EXPECT_TRUE(verify_zkboost_shared_validate_forest_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateForestTest, PreprocessedValidateForestCircuitHasNoOnlinePublicInputs)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance_from_view(
            view,
            training.bin_count,
            public_params,
            zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);

    EXPECT_TRUE(instance.public_inputs.empty());
    EXPECT_EQ(instance.linked_input_digest, digest_zkboost_shared_validate_forest_public_inputs(view));
    EXPECT_EQ(instance.public_input_mode,
              zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);

    const plonkProof proof = prove_zkboost_shared_validate_forest_plonk(instance);
    EXPECT_TRUE(verify_zkboost_shared_validate_forest_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateForestTest, PreprocessedFetchVerifiesAndRejectsTamperedProof)
{
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedValidateForestFetchInstance instance =
        build_zkboost_shared_validate_forest_fetch_instance_from_view(
            view,
            shared_tree_ensemble_from_training_trace(training),
            public_params);

    const SerializedProof proof = prove_zkboost_shared_validate_forest_fetch(instance);
    const zkboostSharedValidateForestFetchVerifierBundle verifier_bundle =
        prepare_zkboost_shared_validate_forest_fetch_verifier_bundle(instance);

    EXPECT_TRUE(verify_zkboost_shared_validate_forest_fetch(instance, proof));
    EXPECT_TRUE(verify_zkboost_shared_validate_forest_fetch_preprocessed(verifier_bundle, proof));

    SerializedProof tampered = proof;
    ASSERT_FALSE(tampered.bytes.empty());
    tampered.bytes.back() ^= 0x01u;
    EXPECT_FALSE(verify_zkboost_shared_validate_forest_fetch_preprocessed(verifier_bundle, tampered));
}

TEST_F(ZkboostSharedValidateForestTest, PathWitnessAffectsPublicInputEncoding)
{
    zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(two_class_training_trace(), two_class_inference_trace());
    const std::vector<Fr> base_public_inputs =
        encode_zkboost_shared_validate_forest_public_inputs(view);

    ASSERT_FALSE(view.routing.path_claims.empty());
    ASSERT_FALSE(view.routing.path_claims[0].path_branch_bits.empty());
    ASSERT_FALSE(view.routing.path_claims[0].permuted_inputs.empty());
    view.routing.path_claims[0].path_branch_bits[0] = 1;
    view.routing.path_claims[0].permuted_inputs[0] += 1;
    view.routing.path_claims[0].logical_leaf_id = 2;

    EXPECT_NE(encode_zkboost_shared_validate_forest_public_inputs(view), base_public_inputs);
}

TEST_F(ZkboostSharedValidateForestTest, VerifyRejectsTamperedCommittedLookupTable)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const plonkProof proof = prove_zkboost_shared_validate_forest_plonk(instance);

    instance.committed_table.left_commitment.value = Group1::zero();

    EXPECT_FALSE(verify_zkboost_shared_validate_forest_committed_table(
        instance.setup.verification_key,
        instance.committed_table));
    EXPECT_FALSE(verify_zkboost_shared_validate_forest_plonk(instance, proof));
}

TEST_F(ZkboostSharedValidateForestTest, VerifyRejectsTamperedLeafPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const plonkProof proof = prove_zkboost_shared_validate_forest_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    ASSERT_FALSE(bad_public_inputs.empty());
    bad_public_inputs[0] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_validate_forest_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateForestTest, VerifyRejectsTamperedTreeScorePublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const plonkProof proof = prove_zkboost_shared_validate_forest_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    const size_t tree_score_offset = two_class_training_trace().leaf_indices.size();
    ASSERT_LT(tree_score_offset, bad_public_inputs.size());
    bad_public_inputs[tree_score_offset] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_validate_forest_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateForestTest, VerifyRejectsTamperedPublicSplitField)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const plonkProof proof = prove_zkboost_shared_validate_forest_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    const size_t split_mask_offset = instance.summary.checked_paths * 2;
    ASSERT_LT(split_mask_offset, bad_public_inputs.size());
    bad_public_inputs[split_mask_offset] = Fr(0);

    EXPECT_FALSE(verify_zkboost_shared_validate_forest_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateForestTest, VerifyRejectsTamperedPathPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance(
            two_class_training_trace(),
            two_class_inference_trace(),
            public_params);
    const plonkProof proof = prove_zkboost_shared_validate_forest_plonk(instance);

    std::vector<Fr> bad_public_inputs = instance.public_inputs;
    const size_t q = instance.summary.checked_paths;
    const size_t n = instance.summary.node_count;
    const size_t f = instance.summary.feature_count;
    const size_t d = instance.summary.max_depth;
    const size_t branch_offset = (2 * q) + (3 * n) + (q * f) + q + (3 * q * d);
    ASSERT_LT(branch_offset, bad_public_inputs.size());
    bad_public_inputs[branch_offset] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_validate_forest_plonk(
        instance.setup.verification_key,
        bad_public_inputs,
        proof));
}

TEST_F(ZkboostSharedValidateForestTest, LinkRejectsFetchPathOpeningMismatch)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);

    const zkboostSharedValidateForestPlonkInstance forest_instance =
        build_zkboost_shared_validate_forest_plonk_instance_from_view(
            view,
            training.bin_count,
            public_params);
    zkboostSharedValidateForestFetchInstance fetch_instance =
        build_zkboost_shared_validate_forest_fetch_instance_from_view(
            view,
            shared_tree_ensemble_from_training_trace(training),
            public_params);

    EXPECT_TRUE(link_zkboost_shared_validate_forest_to_fetch(forest_instance, fetch_instance));

    ASSERT_FALSE(fetch_instance.view.fetch.path_reads.empty());
    fetch_instance.view.fetch.path_reads[0].threshold += 1;

    EXPECT_FALSE(link_zkboost_shared_validate_forest_to_fetch(forest_instance, fetch_instance));
}

TEST_F(ZkboostSharedValidateForestTest, LinkRejectsFetchLeafOpeningMismatch)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);

    const zkboostSharedValidateForestPlonkInstance forest_instance =
        build_zkboost_shared_validate_forest_plonk_instance_from_view(
            view,
            training.bin_count,
            public_params);
    zkboostSharedValidateForestFetchInstance fetch_instance =
        build_zkboost_shared_validate_forest_fetch_instance_from_view(
            view,
            shared_tree_ensemble_from_training_trace(training),
            public_params);

    EXPECT_TRUE(link_zkboost_shared_validate_forest_to_fetch(forest_instance, fetch_instance));

    ASSERT_FALSE(fetch_instance.view.fetch.leaf_value_reads.empty());
    fetch_instance.view.fetch.leaf_value_reads[0].tree_score += 1;

    EXPECT_FALSE(link_zkboost_shared_validate_forest_to_fetch(forest_instance, fetch_instance));
}

TEST_F(ZkboostSharedValidateForestTest, LinkRejectsFetchInternalNodeMismatch)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const forestTrainingSemanticTrace training = two_class_training_trace();
    const forestSemanticInferenceQueryBatch inference = two_class_inference_trace();
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);

    const zkboostSharedValidateForestPlonkInstance forest_instance =
        build_zkboost_shared_validate_forest_plonk_instance_from_view(
            view,
            training.bin_count,
            public_params);
    zkboostSharedValidateForestFetchInstance fetch_instance =
        build_zkboost_shared_validate_forest_fetch_instance_from_view(
            view,
            shared_tree_ensemble_from_training_trace(training),
            public_params);

    EXPECT_TRUE(link_zkboost_shared_validate_forest_to_fetch(forest_instance, fetch_instance));

    ASSERT_FALSE(fetch_instance.view.fetch.path_reads.empty());
    fetch_instance.view.fetch.path_reads[0].internal_node_index += 1;

    EXPECT_FALSE(link_zkboost_shared_validate_forest_to_fetch(forest_instance, fetch_instance));
}

TEST_F(ZkboostSharedValidateForestTest, ProverRejectsWrongPathTransitionWitness)
{
    zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(two_class_training_trace(), two_class_inference_trace());
    ASSERT_FALSE(view.routing.path_claims.empty());
    ASSERT_FALSE(view.routing.path_claims[0].path_branch_bits.empty());
    view.routing.path_claims[0].path_branch_bits[0] = 1;

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance_from_view(
            view,
            two_class_training_trace().bin_count,
            public_params);

    EXPECT_THROW((void)prove_zkboost_shared_validate_forest_plonk(instance), std::runtime_error);
}

TEST_F(ZkboostSharedValidateForestTest, ProverRejectsWrongPermutedInputWitness)
{
    zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(two_class_training_trace(), two_class_inference_trace());
    ASSERT_FALSE(view.routing.path_claims.empty());
    ASSERT_FALSE(view.routing.path_claims[0].permuted_inputs.empty());
    view.routing.path_claims[0].permuted_inputs[0] += 1;

    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    const zkboostSharedValidateForestPlonkInstance instance =
        build_zkboost_shared_validate_forest_plonk_instance_from_view(
            view,
            two_class_training_trace().bin_count,
            public_params);

    EXPECT_THROW((void)prove_zkboost_shared_validate_forest_plonk(instance), std::runtime_error);
}

TEST_F(ZkboostSharedValidateForestTest, BuildRejectsMalformedForestBeforeCircuit)
{
    forestTrainingSemanticTrace trace = two_class_training_trace();
    trace.right_child_indices[0] = 6;
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);

    EXPECT_THROW(
        (void)build_zkboost_shared_validate_forest_plonk_instance(
            trace,
            two_class_inference_trace(),
            public_params),
        std::runtime_error);
}
