#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "cube_sum.hpp"
#include "kzg.hpp"
#include "lagrange_utils.hpp"
#include "lookup.hpp"
#include "matrix_utils.hpp"
#include "polynomial.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <vector>

using namespace mcl::bn;

namespace {

poly cube_poly_from_slices(const std::vector<std::vector<std::vector<int>>>& cube)
{
    std::vector<Fr> evals;
    for (const auto& outer : cube) {
        for (const auto& middle : outer) {
            for (int value : middle) {
                evals.push_back(fr_from_int(value));
            }
        }
    }
    return poly_from_evals(evals);
}

std::vector<Fr> evaluations(const poly& p, size_t domain_size)
{
    return p.evaluations_on_domain(domain_size);
}

poly random_cube_poly(std::mt19937& rng,
                      size_t outer_count,
                      size_t middle_count,
                      size_t inner_count,
                      int coeff_abs_max)
{
    std::uniform_int_distribution<int> coeff_dist(-coeff_abs_max, coeff_abs_max);
    std::vector<Fr> evals(outer_count * middle_count * inner_count, Fr(0));
    for (Fr& value : evals) {
        value = fr_from_int(coeff_dist(rng));
    }
    return poly_from_evals(evals);
}

poly perturb_evaluation(const poly& p, size_t domain_size, size_t index, const Fr& delta)
{
    std::vector<Fr> evals = p.evaluations_on_domain(domain_size);
    Fr::add(evals[index], evals[index], delta);
    return poly_from_evals(evals);
}

poly cube_boundary_selector_poly(size_t outer_count,
                                 size_t middle_count,
                                 size_t inner_count,
                                 bool select_block_starts)
{
    std::vector<Fr> evals(outer_count * middle_count * inner_count, Fr(0));
    const size_t row_offset = select_block_starts ? 0 : middle_count - 1;
    for (size_t outer = 0; outer < outer_count; ++outer) {
        const size_t row = outer * middle_count + row_offset;
        for (size_t inner = 0; inner < inner_count; ++inner) {
            evals[row * inner_count + inner] = Fr(1);
        }
    }
    return poly_from_evals(evals);
}

virCommitment cube_boundary_transition_virtual(size_t outer_count, size_t middle_count, size_t inner_count)
{
    const size_t domain_size = outer_count * middle_count * inner_count;
    const virCommitment block_start_first_column =
        x_pow(subgroup_zero_lagrange_expression(middle_count * inner_count), outer_count);
    const Fr next_row_shift = pow_fr(inverse_root_of_unity(domain_size), inner_count);
    return block_start_first_column - x_scale(block_start_first_column, next_row_shift);
}

void add_cube_lookup_row_index_constraints_for_test(zeroProtocol& zero,
                                                    const virCommitment& row_index,
                                                    size_t outer_count,
                                                    size_t middle_count,
                                                    size_t inner_count)
{
    const size_t table_row_count = outer_count * middle_count;
    const size_t query_domain_size = outer_count * inner_count;
    const Fr previous_row_shift = inverse_root_of_unity(outer_count);
    const Fr row_ratio = subgroup_label(table_row_count, middle_count);

    virCommitment row_transition_zero =
        row_index - scale_by_constant(x_scale(row_index, previous_row_shift), row_ratio);
    zero.add_check(row_transition_zero, query_domain_size);
}

} // namespace

class CubeSumProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(CubeSumProtocolTest, HelperPolynomialsMatchExpectedBlockPrefixesAndSums)
{
    const poly cube = cube_poly_from_slices({
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    });

    const poly prefix = cube_middle_prefix_sum_polynomial(cube, 2, 2, 2);
    const poly sum = cube_middle_sum_polynomial(cube, 2, 2, 2);

    const std::vector<Fr> expected_prefix = {
        fr_from_int(1),
        fr_from_int(2),
        fr_from_int(4),
        fr_from_int(6),
        fr_from_int(5),
        fr_from_int(6),
        fr_from_int(12),
        fr_from_int(14),
    };
    const std::vector<Fr> expected_sum = {
        fr_from_int(4),
        fr_from_int(6),
        fr_from_int(12),
        fr_from_int(14),
    };

    EXPECT_EQ(evaluations(prefix, 8), expected_prefix);
    EXPECT_EQ(evaluations(sum, 4), expected_sum);
}

TEST_F(CubeSumProtocolTest, RejectsNonPowerOfTwoShapes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    cubeSumProtocol protocol(table_lookup);
    protocol.setup(kzg, zero);

    const virCommitment cube(fr_from_int(0));
    EXPECT_THROW(protocol.add_check(cube, cube, 3, 2, 2), std::runtime_error);
    EXPECT_THROW(protocol.add_check(cube, cube, 2, 3, 2), std::runtime_error);
    EXPECT_THROW(protocol.add_check(cube, cube, 2, 2, 3), std::runtime_error);
}

TEST_F(CubeSumProtocolTest, ValidCubeSumProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    const poly cube = cube_poly_from_slices({
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    });
    const poly cube_sum = cube_middle_sum_polynomial(cube, 2, 2, 2);

    cubeSumProtocol protocol(table_lookup);
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(cube)), virCommitment(kzg.commit(cube_sum)), 2, 2, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSumProtocolTest, ProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(2048, 2048));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_reduced_lookup;
    prover_reduced_lookup.setup(prover_kzg, prover_zero, prover_sum);

    tableLookupProtocol prover_table_lookup(prover_reduced_lookup);
    prover_table_lookup.setup(prover_kzg, prover_zero, prover_sum);

    cubeSumProtocol prover_protocol(prover_table_lookup);
    prover_protocol.setup(prover_kzg, prover_zero);

    const poly cube = cube_poly_from_slices({
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    });
    const poly cube_sum = cube_middle_sum_polynomial(cube, 2, 2, 2);
    const kzgCommitment prover_cube_commitment = prover_kzg.commit(cube);
    const kzgCommitment prover_sum_commitment = prover_kzg.commit(cube_sum);

    prover_protocol.add_check(virCommitment(prover_cube_commitment), virCommitment(prover_sum_commitment), 2, 2, 2);

    EXPECT_TRUE(prover_protocol.finalize());
    EXPECT_TRUE(prover_table_lookup.finalize());
    EXPECT_TRUE(prover_reduced_lookup.finalize());
    EXPECT_TRUE(prover_sum.finalize());
    EXPECT_TRUE(prover_zero.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    zeroProtocol verifier_zero;
    verifier_zero.setup(verifier_kzg);

    sumProtocol verifier_sum;
    verifier_sum.setup(verifier_kzg, verifier_zero);

    lookupProtocol verifier_reduced_lookup;
    verifier_reduced_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    tableLookupProtocol verifier_table_lookup(verifier_reduced_lookup);
    verifier_table_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    cubeSumProtocol verifier_protocol(verifier_table_lookup);
    verifier_protocol.setup(verifier_kzg, verifier_zero);

    const kzgCommitment verifier_cube_commitment = verifier_kzg.adopt_commitment(prover_cube_commitment.com);
    const kzgCommitment verifier_sum_commitment = verifier_kzg.adopt_commitment(prover_sum_commitment.com);
    verifier_protocol.add_check(
        virCommitment(verifier_cube_commitment), virCommitment(verifier_sum_commitment), 2, 2, 2);

    EXPECT_TRUE(verifier_protocol.finalize());
    EXPECT_TRUE(verifier_table_lookup.finalize());
    EXPECT_TRUE(verifier_reduced_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(CubeSumProtocolTest, BoundarySelectorConstraintsAcceptCorrectSelector)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    const size_t outer_count = 2;
    const size_t middle_count = 2;
    const size_t inner_count = 2;
    const size_t domain_size = outer_count * middle_count * inner_count;

    virCommitment boundary(kzg.commit(cube_boundary_selector_poly(outer_count, middle_count, inner_count, true)));
    virCommitment step_zero =
        boundary - x_scale(boundary, inverse_root_of_unity(domain_size)) -
        cube_boundary_transition_virtual(outer_count, middle_count, inner_count);
    zero.add_check(step_zero, domain_size);

    virCommitment anchor_zero = boundary - virCommitment(Fr(1));
    zero.add_check(anchor_zero, 1);

    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSumProtocolTest, BoundarySelectorConstraintsRejectWrongSelector)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    const size_t outer_count = 2;
    const size_t middle_count = 2;
    const size_t inner_count = 2;
    const size_t domain_size = outer_count * middle_count * inner_count;

    virCommitment boundary(kzg.commit(cube_boundary_selector_poly(outer_count, middle_count, inner_count, false)));
    virCommitment step_zero =
        boundary - x_scale(boundary, inverse_root_of_unity(domain_size)) -
        cube_boundary_transition_virtual(outer_count, middle_count, inner_count);
    zero.add_check(step_zero, domain_size);

    virCommitment anchor_zero = boundary - virCommitment(Fr(1));
    zero.add_check(anchor_zero, 1);

    EXPECT_FALSE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSumProtocolTest, LookupRowIndexConstraintAcceptsExpectedPositions)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    const size_t outer_count = 2;
    const size_t middle_count = 2;
    const size_t inner_count = 2;
    const size_t table_row_count = outer_count * middle_count;
    const poly zero_cube = poly_from_evals(std::vector<Fr>(outer_count * middle_count * inner_count, Fr(0)));
    const poly zero_sum = cube_middle_sum_polynomial(zero_cube, outer_count, middle_count, inner_count);
    const poly zero_prefix = cube_middle_prefix_sum_polynomial(zero_cube, outer_count, middle_count, inner_count);
    const tableLookupWitness witness = tableLookupWitness::from_positions(table_row_count, {1, 3});

    tableLookupCheckRecord lookup_record;
    table_lookup.add_check(virCommitment(kzg.commit(zero_sum)),
                           virCommitment(kzg.commit(zero_prefix)),
                           outer_count,
                           table_row_count,
                           inner_count,
                           witness,
                           lookup_record);

    add_cube_lookup_row_index_constraints_for_test(
        zero, virCommitment(lookup_record.row_index_commitment), outer_count, middle_count, inner_count);
    const Fr first_row_label = subgroup_label(table_row_count, middle_count - 1);
    EXPECT_EQ(kzg.open(lookup_record.row_index_commitment, Fr(1)), first_row_label);

    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSumProtocolTest, LookupRowIndexConstraintRejectsWrongPositions)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    const size_t outer_count = 2;
    const size_t middle_count = 2;
    const size_t inner_count = 2;
    const size_t table_row_count = outer_count * middle_count;
    const poly zero_cube = poly_from_evals(std::vector<Fr>(outer_count * middle_count * inner_count, Fr(0)));
    const poly zero_sum = cube_middle_sum_polynomial(zero_cube, outer_count, middle_count, inner_count);
    const poly zero_prefix = cube_middle_prefix_sum_polynomial(zero_cube, outer_count, middle_count, inner_count);
    const tableLookupWitness wrong_witness = tableLookupWitness::from_positions(table_row_count, {0, 2});

    tableLookupCheckRecord lookup_record;
    table_lookup.add_check(virCommitment(kzg.commit(zero_sum)),
                           virCommitment(kzg.commit(zero_prefix)),
                           outer_count,
                           table_row_count,
                           inner_count,
                           wrong_witness,
                           lookup_record);

    add_cube_lookup_row_index_constraints_for_test(
        zero, virCommitment(lookup_record.row_index_commitment), outer_count, middle_count, inner_count);
    const Fr first_row_label = subgroup_label(table_row_count, middle_count - 1);
    EXPECT_NE(kzg.open(lookup_record.row_index_commitment, Fr(1)), first_row_label);

    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSumProtocolTest, InvalidCubeSumProofFailsLookup)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    const poly cube = cube_poly_from_slices({
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    });
    const poly wrong_sum = poly_from_evals({
        fr_from_int(4),
        fr_from_int(7),
        fr_from_int(12),
        fr_from_int(14),
    });

    cubeSumProtocol protocol(table_lookup);
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(cube)), virCommitment(kzg.commit(wrong_sum)), 2, 2, 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSumProtocolTest, SupportsBatchedChecks)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    const poly cube_a = cube_poly_from_slices({
        {
            {2, 3},
            {5, 7},
        },
        {
            {11, 13},
            {17, 19},
        },
    });
    const poly cube_b = cube_poly_from_slices({
        {
            {23, 29, 31, 37},
            {41, 43, 47, 53},
        },
        {
            {59, 61, 67, 71},
            {73, 79, 83, 89},
        },
    });

    cubeSumProtocol protocol(table_lookup);
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(cube_a)),
                       virCommitment(kzg.commit(cube_middle_sum_polynomial(cube_a, 2, 2, 2))),
                       2,
                       2,
                       2);
    protocol.add_check(virCommitment(kzg.commit(cube_b)),
                       virCommitment(kzg.commit(cube_middle_sum_polynomial(cube_b, 2, 2, 4))),
                       2,
                       2,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(CubeSumProtocolTest, RandomizedValidProofsPass)
{
    std::mt19937 rng(20260402);
    const std::vector<std::tuple<size_t, size_t, size_t>> shapes = {
        {2, 2, 2},
        {2, 2, 4},
        {2, 4, 2},
        {4, 2, 2},
        {4, 2, 4},
    };
    std::uniform_int_distribution<int> shape_dist(0, static_cast<int>(shapes.size()) - 1);
    std::uniform_int_distribution<int> checks_dist(2, 4);

    for (int round = 0; round < 12; ++round) {
        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(4096, 4096));

        zeroProtocol zero;
        zero.setup(kzg);

        sumProtocol sum;
        sum.setup(kzg, zero);

        lookupProtocol reduced_lookup;
        reduced_lookup.setup(kzg, zero, sum);

        tableLookupProtocol table_lookup(reduced_lookup);
        table_lookup.setup(kzg, zero, sum);

        cubeSumProtocol protocol(table_lookup);
        protocol.setup(kzg, zero);

        const int checks = checks_dist(rng);
        for (int i = 0; i < checks; ++i) {
            const auto [outer_count, middle_count, inner_count] = shapes[static_cast<size_t>(shape_dist(rng))];
            const poly cube = random_cube_poly(rng, outer_count, middle_count, inner_count, 20);
            const poly cube_sum = cube_middle_sum_polynomial(cube, outer_count, middle_count, inner_count);
            protocol.add_check(virCommitment(kzg.commit(cube)),
                               virCommitment(kzg.commit(cube_sum)),
                               outer_count,
                               middle_count,
                               inner_count);
        }

        EXPECT_TRUE(protocol.finalize()) << "round=" << round;
        EXPECT_TRUE(table_lookup.finalize()) << "round=" << round;
        EXPECT_TRUE(reduced_lookup.finalize()) << "round=" << round;
        EXPECT_TRUE(sum.finalize()) << "round=" << round;
        EXPECT_TRUE(zero.finalize()) << "round=" << round;
        EXPECT_TRUE(kzg.finalize()) << "round=" << round;
    }
}

TEST_F(CubeSumProtocolTest, RandomizedInvalidClaimsFailLookup)
{
    std::mt19937 rng(20260403);
    const std::vector<std::tuple<size_t, size_t, size_t>> shapes = {
        {2, 2, 2},
        {2, 2, 4},
        {2, 4, 2},
        {4, 2, 2},
    };
    std::uniform_int_distribution<int> shape_dist(0, static_cast<int>(shapes.size()) - 1);
    std::uniform_int_distribution<int> delta_dist(1, 9);

    for (int round = 0; round < 12; ++round) {
        const auto [outer_count, middle_count, inner_count] = shapes[static_cast<size_t>(shape_dist(rng))];

        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(4096, 4096));

        zeroProtocol zero;
        zero.setup(kzg);

        sumProtocol sum;
        sum.setup(kzg, zero);

        lookupProtocol reduced_lookup;
        reduced_lookup.setup(kzg, zero, sum);

        tableLookupProtocol table_lookup(reduced_lookup);
        table_lookup.setup(kzg, zero, sum);

        cubeSumProtocol protocol(table_lookup);
        protocol.setup(kzg, zero);

        const poly cube = random_cube_poly(rng, outer_count, middle_count, inner_count, 18);
        std::uniform_int_distribution<size_t> index_dist(0, outer_count * inner_count - 1);
        const poly wrong_sum = perturb_evaluation(cube_middle_sum_polynomial(cube, outer_count, middle_count, inner_count),
                                                  outer_count * inner_count,
                                                  index_dist(rng),
                                                  fr_from_int(delta_dist(rng)));

        protocol.add_check(virCommitment(kzg.commit(cube)),
                           virCommitment(kzg.commit(wrong_sum)),
                           outer_count,
                           middle_count,
                           inner_count);

        EXPECT_TRUE(protocol.finalize()) << "round=" << round;
        EXPECT_TRUE(table_lookup.finalize()) << "round=" << round;
        EXPECT_FALSE(reduced_lookup.finalize()) << "round=" << round;
        EXPECT_TRUE(sum.finalize()) << "round=" << round;
        EXPECT_TRUE(zero.finalize()) << "round=" << round;
        EXPECT_TRUE(kzg.finalize()) << "round=" << round;
    }
}
