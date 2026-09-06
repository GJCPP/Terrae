#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "cq_concatenated_table_lookup.hpp"
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

class CqConcatenatedTableLookupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(CqConcatenatedTableLookupTest, BuilderProducesExpectedMergedShape)
{
    const std::vector<size_t> column_counts = {1, 2, 1};
    const size_t merged_width = cq_concatenated_table_lookup_merged_width(column_counts);
    EXPECT_EQ(merged_width, 4u);

    const std::vector<size_t> offsets = cq_concatenated_table_lookup_offsets(merged_width, column_counts);
    EXPECT_EQ(offsets.size(), 3u);
    EXPECT_EQ(offsets[1], 0u);
    EXPECT_NE(offsets[0], offsets[2]);
}

TEST_F(CqConcatenatedTableLookupTest, PublicArtifactSupportsProveAndVerify)
{
    const std::vector<poly> tables = {
        matrix_poly_from_rows({{1}, {2}, {3}, {4}}),
        matrix_poly_from_rows({{10, 11}, {20, 21}, {30, 31}, {40, 41}}),
        matrix_poly_from_rows({{100}, {200}, {300}, {400}}),
    };
    const std::vector<size_t> column_counts = {1, 2, 1};
    const cqConcatenatedTableLookupPublicTable public_table = build_cq_concatenated_table_lookup_public_table(
        testing_kzg_public_parameters(4096, 4096), "cq_concat.builder_table", tables, 4, column_counts, 15);

    ProofRuntime prover = ProofRuntime::prover(testing_kzg_public_parameters(4096, 4096));
    EXPECT_TRUE(run_proof_program(prover, [&](ProofProgram& program) {
        auto& context = program.create_cq_concatenated_table_lookup_context(public_table);
        context.add_check({WitnessCommitmentInput::prover("cq_concat.builder_query_tree", matrix_poly_from_rows({{2}, {4}}), 1),
                           WitnessCommitmentInput::prover(
                               "cq_concat.builder_query_box", matrix_poly_from_rows({{20, 21}, {40, 41}}), 3),
                           WitnessCommitmentInput::prover(
                               "cq_concat.builder_query_score", matrix_poly_from_rows({{200}, {400}}), 1)},
                          2);
    }));

    const SerializedProof proof = prover.take_proof();
    ProofRuntime verifier = ProofRuntime::verifier(prover.public_parameters(), proof);
    EXPECT_TRUE(run_proof_program(verifier, [&](ProofProgram& program) {
        auto& context = program.create_cq_concatenated_table_lookup_context(public_table.verifier_view());
        context.add_check({WitnessCommitmentInput::verifier("cq_concat.builder_query_tree", 1),
                           WitnessCommitmentInput::verifier("cq_concat.builder_query_box", 3),
                           WitnessCommitmentInput::verifier("cq_concat.builder_query_score", 1)},
                          2);
    }));
    EXPECT_TRUE(verifier.proof_fully_consumed());
}
