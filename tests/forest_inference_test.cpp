#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_inference.hpp"
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

class ForestInferenceTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestInferenceTest, BuilderTracksRowFetchAndRangeMetadata)
{
    const forestInferencePublicTable table = build_forest_inference_public_table(
        testing_kzg_public_parameters(4096, 4096),
        "forest_inference.schema",
        matrix_poly_from_rows({{0}, {1}, {2}, {3}}),
        matrix_poly_from_rows({{10, 11}, {20, 21}, {30, 31}, {40, 41}}),
        matrix_poly_from_rows({{12, 13}, {22, 23}, {32, 33}, {42, 43}}),
        matrix_poly_from_rows({{100}, {200}, {300}, {400}}),
        4,
        2,
        6,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}}),
        15);

    EXPECT_EQ(table.row_fetch_table.row_count, 4u);
    EXPECT_EQ(table.row_fetch_table.feature_count, 2u);
    EXPECT_EQ(table.input_bit_bound, 6u);
    EXPECT_TRUE(table.row_fetch_table.includes_leaf_index);
}

TEST_F(ForestInferenceTest, PublicArtifactSupportsProveAndVerify)
{
    const forestInferencePublicTable table = build_forest_inference_public_table(
        testing_kzg_public_parameters(4096, 4096),
        "forest_inference.table",
        matrix_poly_from_rows({{0}, {1}, {2}, {3}}),
        matrix_poly_from_rows({{10, 11}, {20, 21}, {30, 31}, {40, 41}}),
        matrix_poly_from_rows({{12, 13}, {22, 23}, {32, 33}, {42, 43}}),
        matrix_poly_from_rows({{100}, {200}, {300}, {400}}),
        4,
        2,
        6,
        matrix_poly_from_rows({{7}, {8}, {9}, {10}}),
        15);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_proof_program(prover, [&](ProofProgram& program) {
        auto& context = program.create_forest_inference_context(table);
        context.add_check(WitnessCommitmentInput::prover("forest_infer.query_tree", matrix_poly_from_rows({{1}, {3}}), 1),
                          WitnessCommitmentInput::prover(
                              "forest_infer.inputs", matrix_poly_from_rows({{20, 21}, {40, 41}}), 3),
                          WitnessCommitmentInput::prover(
                              "forest_infer.query_lower", matrix_poly_from_rows({{20, 21}, {40, 41}}), 3),
                          WitnessCommitmentInput::prover(
                              "forest_infer.query_upper", matrix_poly_from_rows({{22, 23}, {42, 43}}), 3),
                          WitnessCommitmentInput::prover(
                              "forest_infer.query_score", matrix_poly_from_rows({{200}, {400}}), 1),
                          WitnessCommitmentInput::prover(
                              "forest_infer.query_leaf", matrix_poly_from_rows({{8}, {10}}), 1),
                          2);
    }));

    const SerializedProof proof = prover.take_proof();
    ProofRuntime verifier = ProofRuntime::verifier(prover.public_parameters(), proof);
    EXPECT_TRUE(run_proof_program(verifier, [&](ProofProgram& program) {
        auto& context = program.create_forest_inference_context(table.verifier_view());
        context.add_check(WitnessCommitmentInput::verifier("forest_infer.query_tree", 1),
                          WitnessCommitmentInput::verifier("forest_infer.inputs", 3),
                          WitnessCommitmentInput::verifier("forest_infer.query_lower", 3),
                          WitnessCommitmentInput::verifier("forest_infer.query_upper", 3),
                          WitnessCommitmentInput::verifier("forest_infer.query_score", 1),
                          WitnessCommitmentInput::verifier("forest_infer.query_leaf", 1),
                          2);
    }));
    EXPECT_TRUE(verifier.proof_fully_consumed());
}
