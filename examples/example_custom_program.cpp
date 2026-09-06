#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "fixed_point.hpp"
#include "matrix_sum.hpp"
#include "proof_runtime.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <iostream>
#include <optional>
#include <vector>

using namespace mcl::bn;

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

poly poly_from_domain_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

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

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

poly signed_exp_output_poly(const std::vector<std::int64_t>& scaled_values, const fixedPointEncoding& encoding)
{
    std::vector<Fr> evals;
    evals.reserve(scaled_values.size());
    for (std::int64_t scaled_value : scaled_values) {
        const std::string repr = quantized_exp_output(scaled_value, encoding.fractional_bits).convert_to<std::string>();
        Fr field_value;
        field_value.setStr(repr, 10);
        if (field_value.getStr(10) != repr) {
            throw std::runtime_error("example_custom_program: signed exp output does not fit canonically in Fr");
        }
        evals.push_back(field_value);
    }
    return poly_from_evals(evals);
}

poly shifted_scores_poly(const poly& encoded_scores,
                         const poly& encoded_shifts,
                         const fixedPointEncoding& encoding,
                         size_t outer_count,
                         size_t middle_count,
                         size_t inner_count)
{
    const std::vector<Fr> score_evals = encoded_scores.evaluations_on_domain(outer_count * middle_count * inner_count);
    const poly lifted_shifts = cube_middle_replication_polynomial(encoded_shifts, outer_count, middle_count, inner_count);
    const std::vector<Fr> lifted_shift_evals =
        lifted_shifts.evaluations_on_domain(outer_count * middle_count * inner_count);

    std::vector<Fr> shifted_evals;
    shifted_evals.reserve(score_evals.size());
    const Fr offset = fixed_point_offset_fr(encoding);
    for (size_t i = 0; i < score_evals.size(); ++i) {
        shifted_evals.push_back(score_evals[i] - lifted_shift_evals[i] + offset);
    }
    return poly_from_evals(shifted_evals);
}

poly shifted_cube_softmax_output_poly(const poly& shifted_scores,
                                      const fixedPointEncoding& encoding,
                                      size_t outer_count,
                                      size_t middle_count,
                                      size_t inner_count)
{
    const std::vector<Fr> score_evals = shifted_scores.evaluations_on_domain(outer_count * middle_count * inner_count);
    std::vector<Fr> exp_evals;
    exp_evals.reserve(score_evals.size());
    for (const Fr& encoded_score : score_evals) {
        const size_t encoded = fixed_point_size_t_from_fr(encoded_score, "example_custom_program");
        const std::int64_t scaled_score = decode_fixed_point_scaled_integer(encoded, encoding);
        const std::string repr = quantized_exp_output(scaled_score, encoding.fractional_bits).convert_to<std::string>();
        Fr exp_value;
        exp_value.setStr(repr, 10);
        if (exp_value.getStr(10) != repr) {
            throw std::runtime_error("example_custom_program: shifted softmax exp value does not fit canonically in Fr");
        }
        exp_evals.push_back(exp_value);
    }

    const poly exp_poly = poly_from_evals(exp_evals);
    const poly denominator = cube_middle_sum_polynomial(exp_poly, outer_count, middle_count, inner_count);
    const poly lifted_denominator =
        cube_middle_replication_polynomial(denominator, outer_count, middle_count, inner_count);
    const std::vector<Fr> lifted_denominator_evals =
        lifted_denominator.evaluations_on_domain(outer_count * middle_count * inner_count);

    std::vector<Fr> softmax_evals(exp_evals.size(), Fr(0));
    for (size_t i = 0; i < exp_evals.size(); ++i) {
        Fr inverse;
        Fr::inv(inverse, lifted_denominator_evals[i]);
        softmax_evals[i] = exp_evals[i] * inverse;
    }
    return poly_from_evals(softmax_evals);
}

struct PublicCommitmentBundle {
    Group1 row_sum;
    Group1 column_sum;
    Group1 lookup_table;
    Group1 allowed_rows;
};

struct ProgramInputs {
    WitnessCommitmentInput matrix;
    WitnessCommitmentInput signed_values;
    WitnessCommitmentInput signed_exp_values;
    WitnessCommitmentInput rational_numerator;
    WitnessCommitmentInput rational_denominator;
    WitnessCommitmentInput rational_quotient;
    WitnessCommitmentInput rational_mask;
    WitnessCommitmentInput fixed_division_numerator;
    WitnessCommitmentInput fixed_division_denominator;
    WitnessCommitmentInput fixed_division_denominator_abs;
    WitnessCommitmentInput fixed_division_denominator_is_negative;
    WitnessCommitmentInput fixed_division_quotient;
    WitnessCommitmentInput fixed_division_remainder_abs;
    WitnessCommitmentInput fixed_division_mask;
    WitnessCommitmentInput shifted_softmax_scores;
    WitnessCommitmentInput shifted_softmax_shifted_scores;
    WitnessCommitmentInput shifted_softmax_shifts;
    WitnessCommitmentInput shifted_softmax_shift_selector;
    WitnessCommitmentInput shifted_softmax_output;
    WitnessCommitmentInput lookup_query;
    WitnessCommitmentInput selected_rows;
    PublicCommitmentInput row_sum;
    PublicCommitmentInput column_sum;
    PublicCommitmentInput lookup_table;
    PublicCommitmentInput allowed_rows;
};

PublicCommitmentBundle compute_public_commitments(const kzgPublicParameters& public_params,
                                                  const poly& row_sum_claim,
                                                  const poly& column_sum_claim,
                                                  const poly& lookup_table_poly,
                                                  const poly& allowed_rows_poly)
{
    kzgProtocol kzg;
    kzg.setup(public_params);

    PublicCommitmentBundle bundle;
    bundle.row_sum = kzg.commit_public("demo.matrix.row_sum", row_sum_claim, 1).com;
    bundle.column_sum = kzg.commit_public("demo.matrix.column_sum", column_sum_claim, 3).com;
    bundle.lookup_table = kzg.commit_public("demo.lookup.table", lookup_table_poly, 7).com;
    bundle.allowed_rows = kzg.commit_public("demo.allowed_rows", allowed_rows_poly, 7).com;
    return bundle;
}

bool run_custom_program(ProofRuntime& runtime,
                        const ProgramInputs& inputs)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        const fixedPointEncoding signed_encoding = fixedPointEncoding::signed_offset_binary(4, 1);

        // One witness commitment can be shared across multiple protocols.
        const virCommitment matrix_commitment = program.bind_virtual(inputs.matrix);
        const virCommitment row_sum_commitment = program.bind_virtual(inputs.row_sum);
        const virCommitment column_sum_commitment = program.bind_virtual(inputs.column_sum);

        // Prove two different statements about the same matrix witness.
        program.row_sum().add_check(matrix_commitment, row_sum_commitment, 2, 4);
        program.column_sum().add_check(matrix_commitment, column_sum_commitment, 2, 4);

        // Signed values are kept in offset-binary encoded form and checked against
        // the metadata describing the signed fixed-point layout.
        const virCommitment signed_values_commitment = program.bind_virtual(inputs.signed_values);
        program.signed_range_check().add_check(signed_values_commitment, signed_encoding, 4);

        // The same encoded witness can also feed the signed exp table.
        // Example: encoded value 5 means scaled_value = -3 and real value = -1.5 here.
        program.exp_check().add_check(
            signed_values_commitment, program.bind_virtual(inputs.signed_exp_values), signed_encoding, 4);

        // General rational checks use the multiplicative identity
        //   numerator = denominator * quotient
        // and can be masked to tolerate disabled rows such as zero-denominator cases.
        program.rational_check().add_masked_check(program.bind_virtual(inputs.rational_numerator),
                                                  program.bind_virtual(inputs.rational_denominator),
                                                  program.bind_virtual(inputs.rational_quotient),
                                                  program.bind_virtual(inputs.rational_mask),
                                                  4);

        // Signed fixed-point division proves a scaled Euclidean division identity:
        //   decode(num) * 2^(quo_frac + den_frac - num_frac) = den * decode(quo) + rem
        // together with floor-style remainder bounds on active rows.
        program.fixed_point_division().add_masked_check(
            program.bind_virtual(inputs.fixed_division_numerator),
            program.bind_virtual(inputs.fixed_division_denominator),
            program.bind_virtual(inputs.fixed_division_denominator_abs),
            program.bind_virtual(inputs.fixed_division_denominator_is_negative),
            program.bind_virtual(inputs.fixed_division_quotient),
            program.bind_virtual(inputs.fixed_division_remainder_abs),
            program.bind_virtual(inputs.fixed_division_mask),
            fixedPointEncoding::signed_offset_binary(5, 0),
            fixedPointEncoding::signed_offset_binary(5, 1),
            fixedPointEncoding::signed_offset_binary(6, 2),
            3,
            4);

        // Stabilized softmax keeps the exp inputs in a small signed interval by proving:
        // 1. shifts are the middle-axis maxima of scores via a one-hot selector, and
        // 2. shifted_scores = scores - lift(shifts) + offset on encoded witnesses.
        program.cube_middle_softmax().add_row_max_shifted_check(
            program.bind_virtual(inputs.shifted_softmax_scores),
            program.bind_virtual(inputs.shifted_softmax_shifted_scores),
            program.bind_virtual(inputs.shifted_softmax_shifts),
            program.bind_virtual(inputs.shifted_softmax_shift_selector),
            program.bind_virtual(inputs.shifted_softmax_output),
            signed_encoding,
            2,
            2,
            2);

        // Ordinary lookup: query is private, table is only a public KZG commitment on the verifier side.
        program.lookup().add_check(
            program.bind_virtual(inputs.lookup_query),
            program.bind_virtual(inputs.lookup_table),
            4,
            8);

        // Matrix/table lookup: the verifier only receives the committed public table.
        program.table_lookup().add_check(
            program.bind_virtual(inputs.selected_rows),
            program.bind_virtual(inputs.allowed_rows),
            2,
            4,
            2);
    });
}

} // namespace

int main()
{
    init_groups();

    // Private witness: a 2x4 matrix.
    const poly matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });

    // Public claims derived from the matrix witness.
    const poly row_sum_claim = matrix_row_sum_polynomial(matrix, 2, 4);
    const poly column_sum_claim = matrix_column_sum_polynomial(matrix, 2, 4);

    // Signed fixed-point witness encoded with total_bits = 4 and fractional_bits = 1.
    // Scaled integers [-3, 0, 5, 7] correspond to actual values [-1.5, 0.0, 2.5, 3.5].
    const fixedPointEncoding signed_encoding = fixedPointEncoding::signed_offset_binary(4, 1);
    const std::vector<std::int64_t> signed_scaled_values = {-3, 0, 5, 7};
    const poly signed_values = fixed_point_encoded_poly_from_scaled_integers(signed_scaled_values, signed_encoding);
    const poly signed_exp_values = signed_exp_output_poly(signed_scaled_values, signed_encoding);

    // Masked rational example:
    // the third row is disabled, so denominator = 0 is allowed there.
    const poly rational_numerator = poly_from_domain_ints({6, 10, 999, 0});
    const poly rational_denominator = poly_from_domain_ints({2, 5, 0, 1});
    const poly rational_quotient = poly_from_domain_ints({3, 2, 17, 0});
    const poly rational_mask = poly_from_domain_ints({1, 1, 0, 1});

    // Signed fixed-point floor division example:
    // encoded numerators are signed integers and encoded denominators are signed fixed-point values,
    // encoded quotients have two fractional bits, and the prover also supplies
    // |denominator| plus a sign bit so the protocol can enforce floor semantics.
    const fixedPointEncoding fixed_division_numerator_encoding = fixedPointEncoding::signed_offset_binary(5, 0);
    const fixedPointEncoding fixed_division_denominator_encoding = fixedPointEncoding::signed_offset_binary(5, 1);
    const fixedPointEncoding fixed_division_quotient_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    const poly fixed_division_numerator =
        fixed_point_encoded_poly_from_scaled_integers({7, -7, -3, 0}, fixed_division_numerator_encoding);
    const poly fixed_division_denominator =
        fixed_point_encoded_poly_from_scaled_integers({-3, -5, 5, -1}, fixed_division_denominator_encoding);
    const poly fixed_division_denominator_abs = poly_from_domain_ints({3, 5, 5, 1});
    const poly fixed_division_denominator_is_negative = poly_from_domain_ints({1, 1, 0, 1});
    const poly fixed_division_quotient =
        fixed_point_encoded_poly_from_scaled_integers({-19, 17, -5, 0}, fixed_division_quotient_encoding);
    const poly fixed_division_remainder_abs = poly_from_domain_ints({1, 31, 1, 0});
    const poly fixed_division_mask = poly_from_domain_ints({1, 0, 1, 1});

    // Small shifted softmax example over a 2 x 2 x 2 cube:
    // shifts are chosen as per-(outer, inner) maxima so shifted scores land in [-6, 0].
    const poly shifted_softmax_scores =
        fixed_point_encoded_poly_from_scaled_integers({1, 4, 3, 2, 0, 5, 6, 1}, signed_encoding);
    const poly shifted_softmax_shifts =
        fixed_point_encoded_poly_from_scaled_integers({3, 4, 6, 5}, signed_encoding);
    const poly shifted_softmax_shift_selector = poly_from_evals({
        Fr(0), Fr(1),
        Fr(1), Fr(0),
        Fr(0), Fr(1),
        Fr(1), Fr(0),
    });
    const poly shifted_softmax_shifted_scores =
        shifted_scores_poly(shifted_softmax_scores, shifted_softmax_shifts, signed_encoding, 2, 2, 2);
    const poly shifted_softmax_output =
        shifted_cube_softmax_output_poly(shifted_softmax_shifted_scores, signed_encoding, 2, 2, 2);

    // Private lookup witness and its public table.
    const poly lookup_query = poly_from_domain_ints({5, 7, 7, 11});
    const poly lookup_table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});

    // Private matrix rows and a public table they must come from.
    const poly selected_rows = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });
    const poly allowed_rows = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });

    const kzgPublicParameters public_params = generate_kzg_public_parameters(4096, 4096);
    ProofRuntime prover = ProofRuntime::prover(public_params);
    const PublicCommitmentBundle public_commitments = compute_public_commitments(
        prover.public_parameters(), row_sum_claim, column_sum_claim, lookup_table, allowed_rows);

    const ProgramInputs prover_inputs{
        WitnessCommitmentInput::prover("demo.matrix", matrix, 7),
        WitnessCommitmentInput::prover("demo.signed_values", signed_values, 3),
        WitnessCommitmentInput::prover("demo.signed_exp_values", signed_exp_values, 7),
        WitnessCommitmentInput::prover("demo.rational.numerator", rational_numerator, 3),
        WitnessCommitmentInput::prover("demo.rational.denominator", rational_denominator, 3),
        WitnessCommitmentInput::prover("demo.rational.quotient", rational_quotient, 4),
        WitnessCommitmentInput::prover("demo.rational.mask", rational_mask, 3),
        WitnessCommitmentInput::prover("demo.fixed_division.numerator", fixed_division_numerator, 3),
        WitnessCommitmentInput::prover("demo.fixed_division.denominator", fixed_division_denominator, 3),
        WitnessCommitmentInput::prover("demo.fixed_division.denominator_abs", fixed_division_denominator_abs, 3),
        WitnessCommitmentInput::prover(
            "demo.fixed_division.denominator_is_negative", fixed_division_denominator_is_negative, 3),
        WitnessCommitmentInput::prover("demo.fixed_division.quotient", fixed_division_quotient, 3),
        WitnessCommitmentInput::prover("demo.fixed_division.remainder_abs", fixed_division_remainder_abs, 5),
        WitnessCommitmentInput::prover("demo.fixed_division.mask", fixed_division_mask, 3),
        WitnessCommitmentInput::prover("demo.shifted_softmax.scores", shifted_softmax_scores, 7),
        WitnessCommitmentInput::prover("demo.shifted_softmax.shifted_scores", shifted_softmax_shifted_scores, 7),
        WitnessCommitmentInput::prover("demo.shifted_softmax.shifts", shifted_softmax_shifts, 3),
        WitnessCommitmentInput::prover("demo.shifted_softmax.shift_selector", shifted_softmax_shift_selector, 7),
        WitnessCommitmentInput::prover("demo.shifted_softmax.output", shifted_softmax_output, 7),
        WitnessCommitmentInput::prover("demo.lookup.query", lookup_query, 3),
        WitnessCommitmentInput::prover("demo.selected_rows", selected_rows, 3),
        PublicCommitmentInput::prover("demo.matrix.row_sum", row_sum_claim, 1),
        PublicCommitmentInput::prover("demo.matrix.column_sum", column_sum_claim, 3),
        PublicCommitmentInput::prover("demo.lookup.table", lookup_table, 7),
        PublicCommitmentInput::prover("demo.allowed_rows", allowed_rows, 7),
    };

    const bool prover_ok = run_custom_program(prover, prover_inputs);

    const SerializedProof proof = prover.take_proof();
    ProofRuntime verifier = ProofRuntime::verifier(public_params, proof);
    const ProgramInputs verifier_inputs{
        WitnessCommitmentInput::verifier("demo.matrix", 7),
        WitnessCommitmentInput::verifier("demo.signed_values", 3),
        WitnessCommitmentInput::verifier("demo.signed_exp_values", 7),
        WitnessCommitmentInput::verifier("demo.rational.numerator", 3),
        WitnessCommitmentInput::verifier("demo.rational.denominator", 3),
        WitnessCommitmentInput::verifier("demo.rational.quotient", 4),
        WitnessCommitmentInput::verifier("demo.rational.mask", 3),
        WitnessCommitmentInput::verifier("demo.fixed_division.numerator", 3),
        WitnessCommitmentInput::verifier("demo.fixed_division.denominator", 3),
        WitnessCommitmentInput::verifier("demo.fixed_division.denominator_abs", 3),
        WitnessCommitmentInput::verifier("demo.fixed_division.denominator_is_negative", 3),
        WitnessCommitmentInput::verifier("demo.fixed_division.quotient", 3),
        WitnessCommitmentInput::verifier("demo.fixed_division.remainder_abs", 5),
        WitnessCommitmentInput::verifier("demo.fixed_division.mask", 3),
        WitnessCommitmentInput::verifier("demo.shifted_softmax.scores", 7),
        WitnessCommitmentInput::verifier("demo.shifted_softmax.shifted_scores", 7),
        WitnessCommitmentInput::verifier("demo.shifted_softmax.shifts", 3),
        WitnessCommitmentInput::verifier("demo.shifted_softmax.shift_selector", 7),
        WitnessCommitmentInput::verifier("demo.shifted_softmax.output", 7),
        WitnessCommitmentInput::verifier("demo.lookup.query", 3),
        WitnessCommitmentInput::verifier("demo.selected_rows", 3),
        PublicCommitmentInput::verifier("demo.matrix.row_sum", public_commitments.row_sum, 1),
        PublicCommitmentInput::verifier("demo.matrix.column_sum", public_commitments.column_sum, 3),
        PublicCommitmentInput::verifier("demo.lookup.table", public_commitments.lookup_table, 7),
        PublicCommitmentInput::verifier("demo.allowed_rows", public_commitments.allowed_rows, 7),
    };

    const bool verifier_ok = run_custom_program(verifier, verifier_inputs);

    std::cout << "Custom proof program example\n";
    std::cout << "  prover result: " << (prover_ok ? "success" : "failure") << "\n";
    std::cout << "  verifier result: " << (verifier_ok ? "success" : "failure") << "\n";
    std::cout << "  proof size: " << proof.bytes.size() << " bytes\n";
    std::cout << "  verifier consumed proof: " << (verifier.proof_fully_consumed() ? "yes" : "no") << "\n";
    std::cout << "  transcript match: "
              << (prover.transcript_state() == verifier.transcript_state() ? "yes" : "no") << "\n";

    return (prover_ok && verifier_ok && verifier.proof_fully_consumed() &&
            prover.transcript_state() == verifier.transcript_state())
               ? 0
               : 1;
}
