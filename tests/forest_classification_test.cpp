#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_classification.hpp"
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

class ForestClassificationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestClassificationTest, BuilderTracksInferenceAndPredictionMetadata)
{
    const forestClassificationPublicTable table = build_forest_classification_public_table(
        testing_kzg_public_parameters(4096, 4096),
        "forest_classification.schema",
        matrix_poly_from_rows({{1}, {1}, {2}, {2}, {3}, {3}, {4}, {4}}),
        matrix_poly_from_rows(
            {{10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}}),
        matrix_poly_from_rows(
            {{12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}}),
        matrix_poly_from_rows({{1}, {2}, {3}, {1}, {2}, {1}, {1}, {4}}),
        8,
        2,
        6,
        2,
        2,
        3,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}}),
        15);

    EXPECT_EQ(table.inference_table.row_fetch_table.row_count, 8u);
    EXPECT_EQ(table.inference_table.row_fetch_table.feature_count, 2u);
    EXPECT_EQ(table.inference_table.input_bit_bound, 6u);
    EXPECT_EQ(table.legality_table.tree_count, 4u);
    EXPECT_EQ(table.legality_table.box_consistency_table.row_count, 8u);
    EXPECT_EQ(table.legality_table.box_consistency_table.feature_count, 2u);
    EXPECT_TRUE(table.legality_table.box_consistency_table.includes_leaf_index);
    EXPECT_EQ(table.prediction_table.iteration_count, 2u);
    EXPECT_EQ(table.prediction_table.class_count, 2u);
    EXPECT_EQ(table.prediction_table.score_bit_bound, 3u);
}

TEST_F(ForestClassificationTest, PublicArtifactSupportsProveAndVerify)
{
    const forestClassificationPublicTable table = build_forest_classification_public_table(
        testing_kzg_public_parameters(4096, 4096),
        "forest_classification.table",
        matrix_poly_from_rows({{1}, {1}, {2}, {2}, {3}, {3}, {4}, {4}}),
        matrix_poly_from_rows(
            {{10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}}),
        matrix_poly_from_rows(
            {{12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}}),
        matrix_poly_from_rows({{1}, {2}, {3}, {1}, {2}, {1}, {1}, {4}}),
        8,
        2,
        6,
        2,
        2,
        3,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}}),
        15);

    const poly query_tree = matrix_poly_from_rows({{1}, {1}, {2}, {2}, {3}, {3}, {4}, {4}});
    const poly lifted_inputs = matrix_poly_from_rows(
        {{10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}});
    const poly query_lower = matrix_poly_from_rows(
        {{10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}, {10, 11}, {20, 21}});
    const poly query_upper = matrix_poly_from_rows(
        {{12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}, {12, 13}, {22, 23}});
    const poly query_scores = matrix_poly_from_rows({{1}, {2}, {3}, {1}, {2}, {1}, {1}, {4}});
    const poly query_leaf = matrix_poly_from_rows({{7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}});
    const poly class_scores = matrix_poly_from_rows({{3, 3}, {4, 5}});
    const poly predicted_scores = matrix_poly_from_rows({{4, 5}});
    const poly class_selector = matrix_poly_from_rows({{0, 0}, {1, 1}});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_proof_program(prover, [&](ProofProgram& program) {
        auto& context = program.create_forest_classification_context(table);
        context.add_check(WitnessCommitmentInput::prover("forest_class.query_tree", query_tree, 7),
                          WitnessCommitmentInput::prover("forest_class.inputs", lifted_inputs, 15),
                          WitnessCommitmentInput::prover("forest_class.query_lower", query_lower, 15),
                          WitnessCommitmentInput::prover("forest_class.query_upper", query_upper, 15),
                          WitnessCommitmentInput::prover("forest_class.query_score", query_scores, 7),
                          WitnessCommitmentInput::prover("forest_class.query_leaf", query_leaf, 7),
                          WitnessCommitmentInput::prover("forest_class.class_scores", class_scores, 3),
                          WitnessCommitmentInput::prover(
                              "forest_class.predicted_scores", predicted_scores, 1),
                          WitnessCommitmentInput::prover("forest_class.class_selector", class_selector, 3),
                          8);
    }));

    const SerializedProof proof = prover.take_proof();
    ProofRuntime verifier = ProofRuntime::verifier(prover.public_parameters(), proof);
    EXPECT_TRUE(run_proof_program(verifier, [&](ProofProgram& program) {
        auto& context = program.create_forest_classification_context(table.verifier_view());
        context.add_check(WitnessCommitmentInput::verifier("forest_class.query_tree", 7),
                          WitnessCommitmentInput::verifier("forest_class.inputs", 15),
                          WitnessCommitmentInput::verifier("forest_class.query_lower", 15),
                          WitnessCommitmentInput::verifier("forest_class.query_upper", 15),
                          WitnessCommitmentInput::verifier("forest_class.query_score", 7),
                          WitnessCommitmentInput::verifier("forest_class.query_leaf", 7),
                          WitnessCommitmentInput::verifier("forest_class.class_scores", 3),
                          WitnessCommitmentInput::verifier("forest_class.predicted_scores", 1),
                          WitnessCommitmentInput::verifier("forest_class.class_selector", 3),
                          8);
    }));
    EXPECT_TRUE(verifier.proof_fully_consumed());
}
