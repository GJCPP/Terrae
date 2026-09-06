#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/prediction_plonk.hpp"

#include <gtest/gtest.h>

class ZkboostSharedPredictionPlonkTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

namespace {

forestTrainingSemanticTrace seven_class_training_metadata()
{
    forestTrainingSemanticTrace training;
    training.iteration_count = 2;
    training.class_count = 7;
    training.tree_count = training.iteration_count * training.class_count;
    training.score_encoding = fixedPointEncoding::signed_offset_binary(8, 0);
    return training;
}

zkboostSharedPredictionView seven_class_prediction_view()
{
    zkboostSharedPredictionView view;
    view.summary.iteration_count = 2;
    view.summary.class_count = 7;
    view.summary.sample_count = 2;
    view.summary.tree_score_count = 28;
    view.summary.score_bit_bound = 8;

    const std::vector<std::int64_t> final_scores = {
        5, -5,
        -1, -2,
        5, 1,
        -3, 7,
        0, 7,
        2, 0,
        4, -1,
    };
    view.tree_scores = final_scores;
    view.tree_scores.insert(view.tree_scores.end(), 14, 0);
    view.class_scores = final_scores;
    view.predicted_scores = {5, 7};
    view.predicted_labels = {0, 3};
    view.class_selector = {
        1, 0,
        0, 0,
        0, 0,
        0, 1,
        0, 0,
        0, 0,
        0, 0,
    };
    return view;
}

kzgPublicParameters prediction_public_parameters(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedPredictionView& view,
    zkboostSharedValidateForestPublicInputMode mode =
        zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs)
{
    const size_t required =
        required_zkboost_shared_prediction_plonk_srs_size_from_view(
            training,
            view,
            mode);
    return testing_kzg_public_parameters(required, required);
}

} // namespace

TEST_F(ZkboostSharedPredictionPlonkTest, ProvesSevenClassAggregationAndTieBrokenArgmax)
{
    const forestTrainingSemanticTrace training = seven_class_training_metadata();
    const zkboostSharedPredictionView view = seven_class_prediction_view();
    const kzgPublicParameters public_params =
        prediction_public_parameters(training, view);
    const zkboostSharedPredictionPlonkInstance instance =
        build_zkboost_shared_prediction_plonk_instance_from_view(
            training,
            view,
            public_params);

    EXPECT_EQ(instance.public_inputs,
              encode_zkboost_shared_prediction_public_inputs(view));
    const plonkProof proof = prove_zkboost_shared_prediction_plonk(instance);
    EXPECT_TRUE(verify_zkboost_shared_prediction_plonk(instance, proof));
}

TEST_F(ZkboostSharedPredictionPlonkTest, PreprocessedConstantsProveAndVerify)
{
    const forestTrainingSemanticTrace training = seven_class_training_metadata();
    const zkboostSharedPredictionView view = seven_class_prediction_view();
    const auto mode =
        zkboostSharedValidateForestPublicInputMode::PreprocessedConstants;
    const kzgPublicParameters public_params =
        prediction_public_parameters(training, view, mode);
    const zkboostSharedPredictionPlonkInstance instance =
        build_zkboost_shared_prediction_plonk_instance_from_view(
            training,
            view,
            public_params,
            mode);

    EXPECT_TRUE(instance.public_inputs.empty());
    const plonkProof proof = prove_zkboost_shared_prediction_plonk(instance);
    EXPECT_TRUE(verify_zkboost_shared_prediction_plonk(instance, proof));
}

TEST_F(ZkboostSharedPredictionPlonkTest, RejectsTamperedAggregationPublicInput)
{
    const forestTrainingSemanticTrace training = seven_class_training_metadata();
    const zkboostSharedPredictionView view = seven_class_prediction_view();
    const kzgPublicParameters public_params =
        prediction_public_parameters(training, view);
    zkboostSharedPredictionPlonkInstance instance =
        build_zkboost_shared_prediction_plonk_instance_from_view(
            training,
            view,
            public_params);
    const plonkProof proof = prove_zkboost_shared_prediction_plonk(instance);

    const size_t class_score_offset = view.tree_scores.size();
    instance.public_inputs.at(class_score_offset) += Fr(1);
    EXPECT_FALSE(verify_zkboost_shared_prediction_plonk(instance, proof));
}

TEST_F(ZkboostSharedPredictionPlonkTest, RejectsTamperedTreeAndPredictedScorePublicInputs)
{
    const forestTrainingSemanticTrace training = seven_class_training_metadata();
    const zkboostSharedPredictionView view = seven_class_prediction_view();
    const kzgPublicParameters public_params =
        prediction_public_parameters(training, view);
    const zkboostSharedPredictionPlonkInstance original =
        build_zkboost_shared_prediction_plonk_instance_from_view(
            training,
            view,
            public_params);
    const plonkProof proof = prove_zkboost_shared_prediction_plonk(original);

    zkboostSharedPredictionPlonkInstance tree_score_tampered = original;
    tree_score_tampered.public_inputs.at(0) += Fr(1);
    EXPECT_FALSE(
        verify_zkboost_shared_prediction_plonk(tree_score_tampered, proof));

    zkboostSharedPredictionPlonkInstance predicted_score_tampered = original;
    const size_t predicted_score_offset =
        view.tree_scores.size() + view.class_scores.size();
    predicted_score_tampered.public_inputs.at(predicted_score_offset) += Fr(1);
    EXPECT_FALSE(
        verify_zkboost_shared_prediction_plonk(predicted_score_tampered, proof));
}

TEST_F(ZkboostSharedPredictionPlonkTest, RejectsTamperedSelectorAndLabelPublicInputs)
{
    const forestTrainingSemanticTrace training = seven_class_training_metadata();
    const zkboostSharedPredictionView view = seven_class_prediction_view();
    const kzgPublicParameters public_params =
        prediction_public_parameters(training, view);
    const zkboostSharedPredictionPlonkInstance original =
        build_zkboost_shared_prediction_plonk_instance_from_view(
            training,
            view,
            public_params);
    const plonkProof proof = prove_zkboost_shared_prediction_plonk(original);

    zkboostSharedPredictionPlonkInstance selector_tampered = original;
    const size_t selector_offset =
        view.tree_scores.size() +
        view.class_scores.size() +
        view.predicted_scores.size();
    selector_tampered.public_inputs.at(selector_offset) = Fr(0);
    EXPECT_FALSE(
        verify_zkboost_shared_prediction_plonk(selector_tampered, proof));

    zkboostSharedPredictionPlonkInstance label_tampered = original;
    const size_t label_offset = selector_offset + view.class_selector.size();
    label_tampered.public_inputs.at(label_offset) += Fr(1);
    EXPECT_FALSE(
        verify_zkboost_shared_prediction_plonk(label_tampered, proof));
}

TEST_F(ZkboostSharedPredictionPlonkTest, RejectsTreeScoreOutsideConfiguredRange)
{
    const forestTrainingSemanticTrace training = seven_class_training_metadata();
    zkboostSharedPredictionView view = seven_class_prediction_view();
    view.tree_scores.at(0) = 128;
    const kzgPublicParameters public_params =
        testing_kzg_public_parameters(4096, 4096);

    EXPECT_THROW(
        build_zkboost_shared_prediction_plonk_instance_from_view(
            training,
            view,
            public_params),
        std::runtime_error);
}

TEST_F(ZkboostSharedPredictionPlonkTest, RejectsClassScoreOutsideConfiguredRange)
{
    const forestTrainingSemanticTrace training = seven_class_training_metadata();
    zkboostSharedPredictionView view = seven_class_prediction_view();
    view.class_scores.at(0) = 128;
    const kzgPublicParameters public_params =
        testing_kzg_public_parameters(4096, 4096);

    EXPECT_THROW(
        build_zkboost_shared_prediction_plonk_instance_from_view(
            training,
            view,
            public_params),
        std::runtime_error);
}
