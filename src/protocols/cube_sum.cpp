#include "cube_sum.hpp"

#include "matrix_utils.hpp"
#include "terrae_env.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool cube_sum_failure_debug_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_DEBUG_CUBE_SUM_FAILURE");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

// Validate that every cube dimension is compatible with the subgroup-based protocols.
void validate_cube_shape(size_t outer_count, size_t middle_count, size_t inner_count, const char* context)
{
    if (!isPowerOfTwo(outer_count) || !isPowerOfTwo(middle_count) || !isPowerOfTwo(inner_count) ||
        outer_count == 0 || middle_count == 0 || inner_count == 0) {
        throw std::runtime_error(std::string(context) +
                                 ": outer_count, middle_count, and inner_count must be non-zero powers of two");
    }
}

// Evaluation domain size of the flattened cube.
size_t cube_domain_size(size_t outer_count, size_t middle_count, size_t inner_count)
{
    return outer_count * middle_count * inner_count;
}

// Materialize the cube on its natural flattened domain.
std::vector<Fr> cube_evaluations(const poly& cube,
                                 size_t outer_count,
                                 size_t middle_count,
                                 size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cube_evaluations");
    return cube.evaluations_on_domain(cube_domain_size(outer_count, middle_count, inner_count));
}

// Start rows of each middle-dimension block after flattening the cube into a matrix.
// Example: outer_count = 2, middle_count = 3 -> [0, 3].
std::vector<size_t> block_start_rows(size_t outer_count, size_t middle_count)
{
    std::vector<size_t> rows;
    rows.reserve(outer_count);
    for (size_t outer = 0; outer < outer_count; ++outer) {
        rows.push_back(outer * middle_count);
    }
    return rows;
}

// End rows of each middle-dimension block after flattening the cube into a matrix.
// Example: outer_count = 2, middle_count = 3 -> [2, 5].
std::vector<size_t> block_end_rows(size_t outer_count, size_t middle_count)
{
    std::vector<size_t> rows;
    rows.reserve(outer_count);
    for (size_t outer = 0; outer < outer_count; ++outer) {
        rows.push_back(outer * middle_count + middle_count - 1);
    }
    return rows;
}

// Virtual transition polynomial for the block-start selector.
//
// If boundary_selector marks the first row of each size-(middle_count * inner_count)
// block, then boundary_selector(X) - boundary_selector(X w^{-1}) is non-zero only at
// the first column of the block boundary. This helper expresses that sparse pattern as
// a shifted difference of one Lagrange basis polynomial.
//
// Example for outer_count = 2, middle_count = 3, inner_count = 2:
// the flattened matrix has 6 rows, and the selector marks rows 0 and 3.
// The transition is +1 at (0, 0) and (3, 0), and -1 one row above each of them
// in the cyclic order.
virCommitment cube_boundary_selector_transition_expression(size_t outer_count,
                                                           size_t middle_count,
                                                           size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cube_boundary_selector_transition_expression");

    const size_t domain_size = cube_domain_size(outer_count, middle_count, inner_count);
    const size_t block_size = middle_count * inner_count;
    const virCommitment block_start_first_column =
        x_pow(subgroup_zero_lagrange_expression(block_size), outer_count);
    const Fr next_row_shift = pow_fr(inverse_root_of_unity(domain_size), inner_count);
    return block_start_first_column - x_scale(block_start_first_column, next_row_shift);
}

// Prove that boundary_selector really marks the first row of every middle block.
//
// The verifier does not rebuild the dense selector polynomial. Instead it checks
// a local transition relation plus one anchor value:
//   boundary_selector(X) - boundary_selector(X w^{-1}) = transition(X)
// and boundary_selector(1) = 1.
void add_cube_boundary_selector_constraints(zeroProtocol& zero,
                                            const virCommitment& boundary_selector,
                                            size_t outer_count,
                                            size_t middle_count,
                                            size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cubeSumProtocol");

    const size_t domain_size = cube_domain_size(outer_count, middle_count, inner_count);
    if (middle_count == 1) {
        virCommitment all_one_zero = boundary_selector - virCommitment(Fr(1));
        zero.add_check(all_one_zero, domain_size);
        return;
    }

    const virCommitment transition =
        cube_boundary_selector_transition_expression(outer_count, middle_count, inner_count);
    virCommitment step_zero =
        boundary_selector - x_scale(boundary_selector, inverse_root_of_unity(domain_size)) - transition;
    zero.add_check(step_zero, domain_size);

    virCommitment anchor_zero = boundary_selector - virCommitment(Fr(1));
    zero.add_check(anchor_zero, 1);
}

// Prove that prefix_sum is the segmented prefix sum of cube along the middle dimension.
//
// After flattening cube into an (outer_count * middle_count) x inner_count matrix,
// each length-middle_count row block is accumulated independently.
//
// Example for outer_count = 2, middle_count = 3, inner_count = 1 and flattened values
// [a, b, c, d, e, f]:
//   prefix_sum should be [a, a + b, a + b + c, d, d + e, d + e + f].
// The boundary selector resets the recurrence at the start of each block.
void add_cube_middle_prefix_constraints(zeroProtocol& zero,
                                        const virCommitment& cube,
                                        const virCommitment& prefix_sum,
                                        const virCommitment& boundary_selector,
                                        size_t outer_count,
                                        size_t middle_count,
                                        size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cubeSumProtocol");

    const size_t row_count = outer_count * middle_count;
    const size_t domain_size = row_count * inner_count;
    const Fr previous_row_shift = inverse_root_of_unity(row_count);
    const virCommitment non_boundary_selector = virCommitment(Fr(1)) - boundary_selector;

    virCommitment boundary_zero = (prefix_sum - cube) * boundary_selector;
    virCommitment step_zero = (prefix_sum - x_scale(prefix_sum, previous_row_shift) - cube) * non_boundary_selector;

    zero.add_check(boundary_zero, domain_size);
    zero.add_check(step_zero, domain_size);
}

// Prove that the lookup row-index witness points to the last row of each block.
//
// table_lookup already proves that row_index is constant inside each query row.
// Here we only need to constrain how those row labels evolve from one query row
// to the next:
//   row_index[o] = row_index[o - 1] * subgroup_label(row_count, middle_count).
//
// Example for outer_count = 4 and middle_count = 2:
// the selected rows are 1, 3, 5, 7, so the row labels form a geometric progression
// with ratio w_8^2.
void add_cube_lookup_row_index_constraints(zeroProtocol& zero,
                                           const virCommitment& row_index,
                                           size_t outer_count,
                                           size_t middle_count,
                                           size_t inner_count)
{
    validate_cube_shape(outer_count, middle_count, inner_count, "cubeSumProtocol");

    const size_t table_row_count = outer_count * middle_count;
    const size_t query_domain_size = outer_count * inner_count;
    const Fr previous_row_shift = inverse_root_of_unity(outer_count);
    const Fr row_ratio = subgroup_label(table_row_count, middle_count);

    virCommitment row_transition_zero =
        row_index - scale_by_constant(x_scale(row_index, previous_row_shift), row_ratio);
    zero.add_check(row_transition_zero, query_domain_size);
}

} // namespace

// Record one cube-sum check for later verifier-side bookkeeping.
void cubeSumVerifier::add_check(const virCommitment& cube,
                                const virCommitment& sum,
                                const cubeSumCheckRecord& record)
{
    cubes.push_back(cube);
    sums.push_back(sum);
    checks.push_back(record);
}

// Record a single explicit KZG opening check used for anchor constraints.
void cubeSumVerifier::add_direct_opening_check(Fr opened_value, Fr expected_value)
{
    direct_opening_checks.emplace_back(opened_value, expected_value);
}

// Final verifier-side checks that are cheaper to express as direct openings
// than as zero checks.
bool cubeSumVerifier::finalize() const
{
    for (const auto& [opened_value, expected_value] : direct_opening_checks) {
        if (opened_value != expected_value) {
            return false;
        }
    }
    return true;
}

// Generate the two witnesses used by cubeSumProtocol:
// 1. the segmented prefix-sum witness; and
// 2. the block-start selector witness.
cubeSumCheckRecord cubeSumProver::add_check(const virCommitment& cube,
                                            size_t outer_count,
                                            size_t middle_count,
                                            size_t inner_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("cubeSumProver not setup");
    }

    cubeSumCheckRecord record;
    record.outer_count = outer_count;
    record.middle_count = middle_count;
    record.inner_count = inner_count;
    const poly prefix_poly =
        cube_middle_prefix_sum_polynomial(*params->kzg->materialize(cube), outer_count, middle_count, inner_count);
    record.prefix_commitment = params->kzg->commit(prefix_poly, cube_domain_size(outer_count, middle_count, inner_count) - 1);
    record.boundary_selector_commitment = matrix_row_subset_selector_commitment(
        *params->kzg,
        outer_count * middle_count,
        inner_count,
        block_start_rows(outer_count, middle_count));
    checks.push_back(record);
    return record;
}

// Explicit helper that computes the claimed middle-dimension sums from cube.
poly cube_middle_sum_polynomial(const poly& cube,
                                size_t outer_count,
                                size_t middle_count,
                                size_t inner_count)
{
    const std::vector<Fr> cube_evals = cube_evaluations(cube, outer_count, middle_count, inner_count);
    std::vector<Fr> sum(outer_count * inner_count, Fr(0));

    for (size_t outer = 0; outer < outer_count; ++outer) {
        for (size_t middle = 0; middle < middle_count; ++middle) {
            const size_t cube_row = outer * middle_count + middle;
            for (size_t inner = 0; inner < inner_count; ++inner) {
                Fr::add(sum[matrix_index(outer, inner, inner_count)],
                        sum[matrix_index(outer, inner, inner_count)],
                        cube_evals[matrix_index(cube_row, inner, inner_count)]);
            }
        }
    }

    return poly_from_evals(sum);
}

// Explicit helper that computes the segmented prefix-sum witness used in the protocol.
poly cube_middle_prefix_sum_polynomial(const poly& cube,
                                       size_t outer_count,
                                       size_t middle_count,
                                       size_t inner_count)
{
    const std::vector<Fr> cube_evals = cube_evaluations(cube, outer_count, middle_count, inner_count);
    std::vector<Fr> prefix(cube_domain_size(outer_count, middle_count, inner_count), Fr(0));

    for (size_t outer = 0; outer < outer_count; ++outer) {
        for (size_t inner = 0; inner < inner_count; ++inner) {
            Fr running = Fr(0);
            for (size_t middle = 0; middle < middle_count; ++middle) {
                const size_t cube_row = outer * middle_count + middle;
                const size_t index = matrix_index(cube_row, inner, inner_count);
                Fr::add(running, running, cube_evals[index]);
                prefix[index] = running;
            }
        }
    }

    return poly_from_evals(prefix);
}

// Register one proof that sum[i][k] = sum_j cube[i][j][k].
//
// The proof decomposes into three pieces:
// 1. prove the block-start selector is correct;
// 2. prove prefix_sum is the segmented prefix sum of cube; and
// 3. prove sum is copied from the last row of each block, with the copied
//    positions themselves constrained by row-index relations.
void cubeSumProtocol::add_check(const virCommitment& cube,
                                const virCommitment& sum,
                                size_t outer_count,
                                size_t middle_count,
                                size_t inner_count)
{
    if (!is_setup) {
        throw std::runtime_error("cubeSumProtocol not setup");
    }
    if (params.kzg == nullptr || params.zero_check == nullptr || params.table_lookup == nullptr) {
        throw std::runtime_error("cubeSumProtocol dependencies not setup");
    }

    validate_cube_shape(outer_count, middle_count, inner_count, "cubeSumProtocol");

    const size_t row_count = outer_count * middle_count;
    cubeSumCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.outer_count = outer_count;
        record.middle_count = middle_count;
        record.inner_count = inner_count;
        const Group1 prefix_group = proof_session->transport_group1({});
        record.prefix_commitment = params.kzg->adopt_commitment(prefix_group);
        params.kzg->prove_degree(
            record.prefix_commitment, cube_domain_size(outer_count, middle_count, inner_count) - 1);
        record.boundary_selector_commitment = matrix_row_subset_selector_commitment(
            *params.kzg,
            row_count,
            inner_count,
            block_start_rows(outer_count, middle_count));
    } else {
        record = prover.add_check(cube, outer_count, middle_count, inner_count);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 prefix_group = proof_session->transport_group1([&] { return record.prefix_commitment.com; });
            if (prefix_group != record.prefix_commitment.com) {
                throw std::runtime_error("cubeSumProtocol: transported prefix commitment mismatch");
            }
        }
    }
    verifier.add_check(cube, sum, record);
    const virCommitment prefix_sum(record.prefix_commitment);
    const virCommitment boundary_selector(record.boundary_selector_commitment);

    add_cube_boundary_selector_constraints(
        *params.zero_check, boundary_selector, outer_count, middle_count, inner_count);
    add_cube_middle_prefix_constraints(
        *params.zero_check, cube, prefix_sum, boundary_selector, outer_count, middle_count, inner_count);

    const tableLookupWitness witness = tableLookupWitness::from_positions(
        row_count, block_end_rows(outer_count, middle_count));
    tableLookupCheckRecord lookup_record;
    if (cube_sum_failure_debug_enabled()) {
        const std::vector<Fr> sum_evals =
            params.kzg->evaluations(sum, outer_count * inner_count, kzgProtocol::EvaluationMode::PreferEval);
        const std::vector<Fr> prefix_evals =
            params.kzg->evaluations(prefix_sum, row_count * inner_count, kzgProtocol::EvaluationMode::PreferEval);
        size_t mismatch_count = 0;
        std::vector<std::tuple<size_t, size_t, Fr, Fr>> first_mismatches;
        for (size_t outer = 0; outer < outer_count; ++outer) {
            const size_t prefix_row = outer * middle_count + (middle_count - 1);
            for (size_t inner = 0; inner < inner_count; ++inner) {
                const size_t sum_index = matrix_index(outer, inner, inner_count);
                const size_t prefix_index = matrix_index(prefix_row, inner, inner_count);
                if (sum_evals[sum_index] == prefix_evals[prefix_index]) {
                    continue;
                }
                ++mismatch_count;
                if (first_mismatches.size() < 8) {
                    first_mismatches.emplace_back(outer, inner, sum_evals[sum_index], prefix_evals[prefix_index]);
                }
            }
        }
        if (mismatch_count != 0) {
            std::cerr << "cube_sum_debug lookup_mismatch"
                      << " check_index=" << params.check_count
                      << " outer_count=" << outer_count
                      << " middle_count=" << middle_count
                      << " inner_count=" << inner_count
                      << " mismatch_count=" << mismatch_count;
            for (const auto& [outer, inner, sum_value, prefix_value] : first_mismatches) {
                std::cerr << " sample=" << outer << "," << inner
                          << ":sum=" << sum_value.getStr(10)
                          << ",prefix=" << prefix_value.getStr(10);
            }
            std::cerr << "\n";
        }
    }
    params.table_lookup->add_check(sum, prefix_sum, outer_count, row_count, inner_count, witness, lookup_record);
    add_cube_lookup_row_index_constraints(*params.zero_check,
                                          virCommitment(lookup_record.row_index_commitment),
                                          outer_count,
                                          middle_count,
                                          inner_count);
    const Fr first_row_label = subgroup_label(row_count, middle_count - 1);
    const Fr opened_row_index = params.kzg->open(lookup_record.row_index_commitment, Fr(1), kzgOpeningRootRole::FixedEvaluation);
    verifier.add_direct_opening_check(opened_row_index, first_row_label);
    ++params.check_count;
}

bool cubeSumProtocol::finalize()
{
    begin_finalize("cubeSumProtocol");
    return verifier.finalize();
}
