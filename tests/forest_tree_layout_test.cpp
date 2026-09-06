#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_tree_layout.hpp"
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

class ForestTreeLayoutTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ForestTreeLayoutTest, BuilderTracksIterationAndClassCounts)
{
    const forestTreeLayoutPublicTable table =
        build_forest_tree_layout_public_table("forest_tree_layout.schema", 2, 2);
    EXPECT_EQ(table.iteration_count, 2u);
    EXPECT_EQ(table.class_count, 2u);
    EXPECT_EQ(table.tree_count(), 4u);
}

TEST_F(ForestTreeLayoutTest, LayoutExpressionMatchesSequentialTreeIds)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    const virCommitment layout = forest_tree_layout_expression(4);
    EXPECT_EQ(kzg.evaluations(layout, 4, kzgProtocol::EvaluationMode::PreferEval),
              (std::vector<Fr>{Fr(1), Fr(2), Fr(3), Fr(4)}));
    EXPECT_EQ(forest_tree_layout_polynomial(4).evaluations_on_domain(4),
              (std::vector<Fr>{Fr(1), Fr(2), Fr(3), Fr(4)}));
}

TEST_F(ForestTreeLayoutTest, PublicArtifactSupportsProveAndVerify)
{
    const forestTreeLayoutPublicTable table =
        build_forest_tree_layout_public_table("forest_tree_layout.table", 2, 2);
    const poly query_tree = matrix_poly_from_rows({{1}, {1}, {2}, {2}, {3}, {3}, {4}, {4}});

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_proof_program(prover, [&](ProofProgram& program) {
        auto& context = program.create_forest_tree_layout_context(table);
        context.add_check(WitnessCommitmentInput::prover("forest_tree_layout.query_tree", query_tree, 7), 8);
    }));

    const SerializedProof proof = prover.take_proof();
    ProofRuntime verifier = ProofRuntime::verifier(prover.public_parameters(), proof);
    EXPECT_TRUE(run_proof_program(verifier, [&](ProofProgram& program) {
        auto& context = program.create_forest_tree_layout_context(table.verifier_view());
        context.add_check(WitnessCommitmentInput::verifier("forest_tree_layout.query_tree", 7), 8);
    }));
    EXPECT_TRUE(verifier.proof_fully_consumed());
}
