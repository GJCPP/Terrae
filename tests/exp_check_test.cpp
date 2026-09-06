#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "exp_check.hpp"
#include "fixed_point.hpp"
#include "kzg.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "test_kzg_srs.hpp"
#include "zero_check.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace mcl::bn;

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

poly poly_from_size_t_values(const std::vector<size_t>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (size_t value : values) {
        Fr field_value;
        field_value.setStr(std::to_string(value), 10);
        evals.push_back(field_value);
    }
    return poly_from_evals(evals);
}

poly poly_from_cpp_int_values(const std::vector<cpp_int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (const cpp_int& value : values) {
        Fr field_value;
        const std::string repr = value.convert_to<std::string>();
        field_value.setStr(repr, 10);
        EXPECT_EQ(field_value.getStr(10), repr);
        evals.push_back(field_value);
    }
    return poly_from_evals(evals);
}

size_t quantized_exp_output(size_t input, size_t quantization_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(quantization_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>().convert_to<size_t>();
}

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

std::vector<size_t> quantized_exp_outputs(const std::vector<size_t>& inputs, size_t quantization_bits)
{
    std::vector<size_t> outputs;
    outputs.reserve(inputs.size());
    for (size_t input : inputs) {
        outputs.push_back(quantized_exp_output(input, quantization_bits));
    }
    return outputs;
}

std::vector<cpp_int> quantized_exp_outputs(const std::vector<std::int64_t>& scaled_inputs,
                                           const fixedPointEncoding& encoding)
{
    std::vector<cpp_int> outputs;
    outputs.reserve(scaled_inputs.size());
    for (std::int64_t scaled_input : scaled_inputs) {
        outputs.push_back(quantized_exp_output(scaled_input, encoding.fractional_bits));
    }
    return outputs;
}

class TestableTableLookupProtocol : public tableLookupProtocol {
public:
    using tableLookupProtocol::tableLookupProtocol;

    const tableLookupVerifier& verifier_ref() const { return verifier; }
};

} // namespace

class ExpCheckProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ExpCheckProtocolTest, AddCheckRejectsInvalidParameters)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    const virCommitment values(fr_from_int(0));
    EXPECT_THROW(exp_check.add_check(values, values, 3, 3), std::runtime_error);
    EXPECT_THROW(exp_check.add_check(values, values, Fr::getBitSize() - 2, 4), std::runtime_error);
    EXPECT_THROW(exp_check.add_check(values, values, fixedPointEncoding::signed_offset_binary(4, 4), 4),
                 std::runtime_error);
    fixedPointEncoding unsigned_like = fixedPointEncoding::signed_offset_binary(4, 1);
    unsigned_like.signed_values = false;
    EXPECT_THROW(exp_check.add_check(values, values, unsigned_like, 4), std::runtime_error);
}

TEST_F(ExpCheckProtocolTest, ValidChecksBatchAcrossSharedQuantization)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    TestableTableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    const std::vector<size_t> q3_input_a = {0, 1};
    const std::vector<size_t> q3_input_b = {2, 3, 4, 5};
    const std::vector<size_t> q2_input = {0, 1, 2, 3};

    exp_check.add_check(
        virCommitment(kzg.commit(poly_from_size_t_values(q3_input_a))),
        virCommitment(kzg.commit(poly_from_size_t_values(quantized_exp_outputs(q3_input_a, 3)))),
        3,
        2);
    exp_check.add_check(
        virCommitment(kzg.commit(poly_from_size_t_values(q3_input_b))),
        virCommitment(kzg.commit(poly_from_size_t_values(quantized_exp_outputs(q3_input_b, 3)))),
        3,
        4);
    exp_check.add_check(
        virCommitment(kzg.commit(poly_from_size_t_values(q2_input))),
        virCommitment(kzg.commit(poly_from_size_t_values(quantized_exp_outputs(q2_input, 2)))),
        2,
        4);

    EXPECT_TRUE(exp_check.finalize());
    ASSERT_EQ(table_lookup.verifier_ref().checks.size(), 2u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[0].query_row_count, 8u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[0].table_row_count, 8u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[0].column_count, 2u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[1].query_row_count, 4u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[1].table_row_count, 4u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[1].column_count, 2u);

    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ExpCheckProtocolTest, SignedChecksBatchAcrossSharedEncoding)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    TestableTableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const std::vector<std::int64_t> signed_input_a = {-4, -1};
    const std::vector<std::int64_t> signed_input_b = {0, 1, 3, 7};
    const std::vector<size_t> unsigned_input = {0, 1, 2, 3};

    exp_check.add_check(virCommitment(kzg.commit(fixed_point_encoded_poly_from_scaled_integers(signed_input_a, encoding))),
                        virCommitment(kzg.commit(poly_from_cpp_int_values(quantized_exp_outputs(signed_input_a, encoding)))),
                        encoding,
                        2);
    exp_check.add_check(virCommitment(kzg.commit(fixed_point_encoded_poly_from_scaled_integers(signed_input_b, encoding))),
                        virCommitment(kzg.commit(poly_from_cpp_int_values(quantized_exp_outputs(signed_input_b, encoding)))),
                        encoding,
                        4);
    exp_check.add_check(virCommitment(kzg.commit(poly_from_size_t_values(unsigned_input))),
                        virCommitment(kzg.commit(poly_from_size_t_values(quantized_exp_outputs(unsigned_input, 2)))),
                        2,
                        4);

    EXPECT_TRUE(exp_check.finalize());
    ASSERT_EQ(table_lookup.verifier_ref().checks.size(), 2u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[0].table_row_count, 16u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[1].table_row_count, 4u);

    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ExpCheckProtocolTest, NonPositiveSignedChecksSupportWideShiftedEncoding)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    TestableTableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(11, 0);
    const std::vector<std::int64_t> shifted_inputs = {-516, -32, -1, 0};
    exp_check.add_non_positive_check(
        virCommitment(kzg.commit(fixed_point_encoded_poly_from_scaled_integers(shifted_inputs, encoding))),
        virCommitment(kzg.commit(poly_from_cpp_int_values(quantized_exp_outputs(shifted_inputs, encoding)))),
        encoding,
        4);

    EXPECT_TRUE(exp_check.finalize());
    ASSERT_EQ(table_lookup.verifier_ref().checks.size(), 1u);
    EXPECT_EQ(table_lookup.verifier_ref().checks[0].table_row_count, size_t(1) << encoding.total_bits);

    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ExpCheckProtocolTest, IncorrectMappedOutputFailsLookup)
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

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    const std::vector<size_t> inputs = {0, 1, 2, 3};
    std::vector<size_t> outputs = quantized_exp_outputs(inputs, 3);
    outputs[2] += 1;

    exp_check.add_check(
        virCommitment(kzg.commit(poly_from_size_t_values(inputs))),
        virCommitment(kzg.commit(poly_from_size_t_values(outputs))),
        3,
        4);

    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ExpCheckProtocolTest, IncorrectSignedMappedOutputFailsLookup)
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

    expCheckProtocol exp_check;
    exp_check.setup(kzg, table_lookup);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const std::vector<std::int64_t> inputs = {-4, -1, 0, 3};
    std::vector<cpp_int> outputs = quantized_exp_outputs(inputs, encoding);
    outputs[2] += 1;

    exp_check.add_check(
        virCommitment(kzg.commit(fixed_point_encoded_poly_from_scaled_integers(inputs, encoding))),
        virCommitment(kzg.commit(poly_from_cpp_int_values(outputs))),
        encoding,
        4);

    EXPECT_TRUE(exp_check.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ExpCheckProtocolTest, ProveAndVerifyWithSerializedProof)
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

    expCheckProtocol prover_exp_check;
    prover_exp_check.setup(prover_kzg, prover_table_lookup);

    const std::vector<size_t> inputs = {0, 1, 2, 3};
    const std::vector<size_t> outputs = quantized_exp_outputs(inputs, 3);
    const kzgCommitment prover_input_commitment = prover_kzg.commit(poly_from_size_t_values(inputs));
    const kzgCommitment prover_output_commitment = prover_kzg.commit(poly_from_size_t_values(outputs));

    prover_exp_check.add_check(virCommitment(prover_input_commitment), virCommitment(prover_output_commitment), 3, 4);

    EXPECT_TRUE(prover_exp_check.finalize());
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

    expCheckProtocol verifier_exp_check;
    verifier_exp_check.setup(verifier_kzg, verifier_table_lookup);

    const kzgCommitment verifier_input_commitment = verifier_kzg.adopt_commitment(prover_input_commitment.com);
    const kzgCommitment verifier_output_commitment = verifier_kzg.adopt_commitment(prover_output_commitment.com);

    verifier_exp_check.add_check(
        virCommitment(verifier_input_commitment), virCommitment(verifier_output_commitment), 3, 4);

    EXPECT_TRUE(verifier_exp_check.finalize());
    EXPECT_TRUE(verifier_table_lookup.finalize());
    EXPECT_TRUE(verifier_reduced_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(ExpCheckProtocolTest, SignedProveAndVerifyWithSerializedProof)
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

    expCheckProtocol prover_exp_check;
    prover_exp_check.setup(prover_kzg, prover_table_lookup);

    const fixedPointEncoding encoding = fixedPointEncoding::signed_offset_binary(4, 2);
    const std::vector<std::int64_t> inputs = {-4, -1, 0, 3};
    const kzgCommitment prover_input_commitment =
        prover_kzg.commit(fixed_point_encoded_poly_from_scaled_integers(inputs, encoding));
    const kzgCommitment prover_output_commitment =
        prover_kzg.commit(poly_from_cpp_int_values(quantized_exp_outputs(inputs, encoding)));

    prover_exp_check.add_check(
        virCommitment(prover_input_commitment), virCommitment(prover_output_commitment), encoding, 4);

    EXPECT_TRUE(prover_exp_check.finalize());
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

    expCheckProtocol verifier_exp_check;
    verifier_exp_check.setup(verifier_kzg, verifier_table_lookup);

    const kzgCommitment verifier_input_commitment = verifier_kzg.adopt_commitment(prover_input_commitment.com);
    const kzgCommitment verifier_output_commitment = verifier_kzg.adopt_commitment(prover_output_commitment.com);

    verifier_exp_check.add_check(
        virCommitment(verifier_input_commitment), virCommitment(verifier_output_commitment), encoding, 4);

    EXPECT_TRUE(verifier_exp_check.finalize());
    EXPECT_TRUE(verifier_table_lookup.finalize());
    EXPECT_TRUE(verifier_reduced_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}
