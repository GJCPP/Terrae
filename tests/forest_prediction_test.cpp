#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_prediction.hpp"
#include "proof_runtime.hpp"
#include "test_kzg_srs.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mcl::bn;

namespace {

poly matrix_poly_from_rows(const std::vector<std::vector<int>>& rows)
{
    std::vector<Fr> evals;
    for (const auto& row : rows) {
        for (int value : row) {
            evals.push_back(fr_from_int(value));
        }
    }
    return poly_from_evals(evals);
}

} // namespace

class ForestPredictionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestPredictionTest, BuilderTracksAggregationMetadata)
{
    const forestPredictionPublicTable table = build_forest_prediction_public_table(2, 2, 3);
    EXPECT_EQ(table.iteration_count, 2u);
    EXPECT_EQ(table.class_count, 2u);
    EXPECT_EQ(table.score_bit_bound, 3u);
}

TEST_F(ForestPredictionTest, PublicArtifactSupportsProveAndVerify)
{
    const forestPredictionPublicTable table = build_forest_prediction_public_table(2, 2, 3);
    const poly query_scores = matrix_poly_from_rows({{1}, {2}, {3}, {1}, {2}, {1}, {1}, {4}});
    const poly class_scores = matrix_poly_from_rows({{3, 3}, {4, 5}});
    const poly predicted_scores = matrix_poly_from_rows({{4, 5}});
    const poly class_selector = matrix_poly_from_rows({{0, 0}, {1, 1}});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_proof_program(prover, [&](ProofProgram& program) {
        auto& context = program.create_forest_prediction_context(table);
        context.add_check(WitnessCommitmentInput::prover("forest_prediction.scores", query_scores, 7),
                          WitnessCommitmentInput::prover("forest_prediction.class_scores", class_scores, 3),
                          WitnessCommitmentInput::prover("forest_prediction.predicted_scores", predicted_scores, 1),
                          WitnessCommitmentInput::prover("forest_prediction.class_selector", class_selector, 3),
                          8);
    }));

    const SerializedProof proof = prover.take_proof();
    ProofRuntime verifier = ProofRuntime::verifier(prover.public_parameters(), proof);
    EXPECT_TRUE(run_proof_program(verifier, [&](ProofProgram& program) {
        auto& context = program.create_forest_prediction_context(table.verifier_view());
        context.add_check(WitnessCommitmentInput::verifier("forest_prediction.scores", 7),
                          WitnessCommitmentInput::verifier("forest_prediction.class_scores", 3),
                          WitnessCommitmentInput::verifier("forest_prediction.predicted_scores", 1),
                          WitnessCommitmentInput::verifier("forest_prediction.class_selector", 3),
                          8);
    }));
    EXPECT_TRUE(verifier.proof_fully_consumed());
}

TEST_F(ForestPredictionTest, ScoreAggregationAndArgmaxSupportPredictionLayout)
{
    const poly query_scores = matrix_poly_from_rows({{1}, {2}, {3}, {1}, {2}, {1}, {1}, {4}});
    const poly class_scores = matrix_poly_from_rows({{3, 3}, {4, 5}});
    const poly predicted_scores = matrix_poly_from_rows({{4, 5}});
    const poly class_selector = matrix_poly_from_rows({{0, 0}, {1, 1}});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_proof_program(prover, [&](ProofProgram& program) {
        const virCommitment scores_commitment =
            program.bind_virtual(WitnessCommitmentInput::prover("forest_prediction.only_scores", query_scores, 7));
        const virCommitment class_scores_commitment =
            program.bind_virtual(WitnessCommitmentInput::prover("forest_prediction.only_class_scores", class_scores, 3));
        const virCommitment predicted_scores_commitment = program.bind_virtual(
            WitnessCommitmentInput::prover("forest_prediction.only_predicted_scores", predicted_scores, 1));
        const virCommitment class_selector_commitment =
            program.bind_virtual(WitnessCommitmentInput::prover("forest_prediction.only_class_selector", class_selector, 3));

        program.column_sum().add_check(scores_commitment, class_scores_commitment, 2, 4);
        program.cube_middle_argmax().add_check(
            class_scores_commitment, predicted_scores_commitment, class_selector_commitment, 1, 2, 2, 3);
    }));

    const SerializedProof proof = prover.take_proof();
    ProofRuntime verifier = ProofRuntime::verifier(prover.public_parameters(), proof);
    EXPECT_TRUE(run_proof_program(verifier, [&](ProofProgram& program) {
        const virCommitment scores_commitment =
            program.bind_virtual(WitnessCommitmentInput::verifier("forest_prediction.only_scores", 7));
        const virCommitment class_scores_commitment =
            program.bind_virtual(WitnessCommitmentInput::verifier("forest_prediction.only_class_scores", 3));
        const virCommitment predicted_scores_commitment =
            program.bind_virtual(WitnessCommitmentInput::verifier("forest_prediction.only_predicted_scores", 1));
        const virCommitment class_selector_commitment =
            program.bind_virtual(WitnessCommitmentInput::verifier("forest_prediction.only_class_selector", 3));

        program.column_sum().add_check(scores_commitment, class_scores_commitment, 2, 4);
        program.cube_middle_argmax().add_check(
            class_scores_commitment, predicted_scores_commitment, class_selector_commitment, 1, 2, 2, 3);
    }));
    EXPECT_TRUE(verifier.proof_fully_consumed());
}
