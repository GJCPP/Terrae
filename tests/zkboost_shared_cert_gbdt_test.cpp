#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/cert_gbdt.hpp"

#include "../examples/forest_training_demo_data.hpp"

#include <gtest/gtest.h>

class ZkboostSharedCertGBDTTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

namespace {

forestTrainingSemanticTrace two_class_shared_trace()
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
    trace.node_leaf_values = {0, 1, -2, 0, 0, -2, 1, 0};

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
    for (size_t bin = 0; bin < 3; ++bin) {
        trace.candidate_mask[bin] = 1;
        trace.candidate_mask[4 * trace.feature_count * trace.bin_count + bin] = 1;
    }
    trace.candidate_mask[1 * trace.bin_count + 0] = 1;
    trace.candidate_mask[2 * trace.bin_count + 2] = 1;
    trace.candidate_mask[5 * trace.bin_count + 0] = 1;
    trace.candidate_mask[6 * trace.bin_count + 2] = 1;
    trace.selected_candidate_indices.assign(trace.node_count, 0);
    trace.selected_candidate_indices[0] = 1;
    trace.selected_candidate_indices[1] = 0;
    trace.selected_candidate_indices[2] = 2;
    trace.selected_candidate_indices[4] = 1;
    trace.selected_candidate_indices[5] = 0;
    trace.selected_candidate_indices[6] = 2;
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

TEST_F(ZkboostSharedCertGBDTTest, ProvesAndVerifiesSharedTrainingWorkflow)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const zkboostSharedCertGBDTInstance instance =
        build_zkboost_shared_cert_gbdt_instance(
            two_class_shared_trace(),
            two_class_inference_trace(),
            public_params);

    EXPECT_EQ(instance.summary.tree_count, 2u);
    EXPECT_EQ(instance.summary.candidate_count, 64u);
    EXPECT_TRUE(verify_zkboost_shared_cert_gbdt_links(instance));
    EXPECT_TRUE(check_zkboost_shared_cert_gbdt_links(instance).all_ok());

    const zkboostSharedCertGBDTProof proof =
        prove_zkboost_shared_cert_gbdt(instance);

    EXPECT_TRUE(verify_zkboost_shared_cert_gbdt(instance, proof));
    ASSERT_FALSE(proof.cross_session_opening_batch.bytes.empty());
    ASSERT_FALSE(proof.forest_fetch.bytes.empty());
    EXPECT_GT(serialized_zkboost_shared_cert_gbdt_proof_bytes(proof), 0u);

    zkboostSharedCertGBDTProof tampered = proof;
    ASSERT_FALSE(tampered.forest_fetch.bytes.empty());
    tampered.forest_fetch.bytes.back() ^= 0x01u;
    EXPECT_FALSE(verify_zkboost_shared_cert_gbdt(instance, tampered));
}

TEST_F(ZkboostSharedCertGBDTTest, ProvesAndVerifiesFirstOrderOneBasedDepthWorkflow)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    const forestTrainingSemanticTrace training = build_demo_first_order_training_trace();
    ASSERT_EQ(training.max_depth, 2u);

    const zkboostSharedCertGBDTInstance instance =
        build_zkboost_shared_cert_gbdt_instance(
            training,
            build_demo_inference_query_trace(),
            public_params);

    EXPECT_EQ(instance.forest.summary.max_depth, 1u);
    EXPECT_TRUE(verify_zkboost_shared_cert_gbdt_links(instance));

    const zkboostSharedCertGBDTProof proof =
        prove_zkboost_shared_cert_gbdt(instance);
    EXPECT_TRUE(verify_zkboost_shared_cert_gbdt(instance, proof));
}

TEST_F(ZkboostSharedCertGBDTTest, LinksRejectTamperedBoostingTreeScore)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    zkboostSharedCertGBDTInstance instance =
        build_zkboost_shared_cert_gbdt_instance(
            two_class_shared_trace(),
            two_class_inference_trace(),
            public_params);
    ASSERT_FALSE(instance.boosting_state.public_inputs.empty());
    instance.boosting_state.public_inputs[0] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_cert_gbdt_links(instance));
}

TEST_F(ZkboostSharedCertGBDTTest, LinksRejectTamperedSplitHistogramInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    zkboostSharedCertGBDTInstance instance =
        build_zkboost_shared_cert_gbdt_instance(
            two_class_shared_trace(),
            two_class_inference_trace(),
            public_params);
    ASSERT_FALSE(instance.splits.public_inputs.empty());
    instance.splits.public_inputs[0] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_cert_gbdt_links(instance));
}

TEST_F(ZkboostSharedCertGBDTTest, LinksRejectTamperedForestToSplitField)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    zkboostSharedCertGBDTInstance instance =
        build_zkboost_shared_cert_gbdt_instance(
            two_class_shared_trace(),
            two_class_inference_trace(),
            public_params);
    const size_t selected_bin_offset = instance.splits.summary.candidate_count * 16 +
                                       instance.splits.summary.node_count * 3;
    ASSERT_LT(selected_bin_offset, instance.splits.public_inputs.size());
    instance.splits.public_inputs[selected_bin_offset] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_cert_gbdt_links(instance));
}

TEST_F(ZkboostSharedCertGBDTTest, VerifyRejectsTamperedSplitNodeDepthPublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    zkboostSharedCertGBDTInstance instance =
        build_zkboost_shared_cert_gbdt_instance(
            two_class_shared_trace(),
            two_class_inference_trace(),
            public_params);
    const zkboostSharedCertGBDTProof proof =
        prove_zkboost_shared_cert_gbdt(instance);
    const size_t node_depth_offset = instance.splits.summary.candidate_count * 16 +
                                     instance.splits.summary.node_count * 7;
    ASSERT_LT(node_depth_offset, instance.splits.public_inputs.size());
    instance.splits.public_inputs[node_depth_offset] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_cert_gbdt(instance, proof));
}

TEST_F(ZkboostSharedCertGBDTTest, VerifyRejectsTamperedLeafValuePublicInput)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    zkboostSharedCertGBDTInstance instance =
        build_zkboost_shared_cert_gbdt_instance(
            two_class_shared_trace(),
            two_class_inference_trace(),
            public_params);
    const zkboostSharedCertGBDTProof proof =
        prove_zkboost_shared_cert_gbdt(instance);

    const size_t decoded_leaf_offset = 3 * instance.leaf_values.summary.node_count;
    ASSERT_LT(decoded_leaf_offset + 1, instance.leaf_values.public_inputs.size());
    instance.leaf_values.public_inputs[decoded_leaf_offset + 1] += Fr(1);

    EXPECT_FALSE(verify_zkboost_shared_cert_gbdt(instance, proof));
}

TEST_F(ZkboostSharedCertGBDTTest, VerifyRejectsTamperedForestFetchCommitment)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(16384, 16384);
    zkboostSharedCertGBDTInstance instance =
        build_zkboost_shared_cert_gbdt_instance(
            two_class_shared_trace(),
            two_class_inference_trace(),
            public_params);
    const zkboostSharedCertGBDTProof proof =
        prove_zkboost_shared_cert_gbdt(instance);

    instance.forest_fetch.commitments.com_t = Group1::zero();

    EXPECT_FALSE(verify_zkboost_shared_cert_gbdt(instance, proof));
}
