#include "exp_check.hpp"

#include "algebra_utils.hpp"
#include "interleaved_batching.hpp"
#include "polynomial.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using boost::multiprecision::cpp_dec_float_100;
using boost::multiprecision::cpp_int;

size_t checked_power_of_two_row_count(size_t bit_bound, const char* context)
{
    if (bit_bound == 0) {
        throw std::runtime_error(std::string(context) + ": bit bound must be non-zero");
    }
    if (bit_bound >= std::numeric_limits<size_t>::digits - 1) {
        throw std::runtime_error(std::string(context) + ": bit bound is too large for size_t domain sizing");
    }
    return size_t(1) << bit_bound;
}

size_t checked_exp_table_row_count(size_t quantization_bits)
{
    const size_t max_safe_quantization_bits = Fr::getBitSize() - 3;
    if (quantization_bits > max_safe_quantization_bits) {
        throw std::runtime_error("expCheckProtocol: quantization bits exceed the safely representable integer range of Fr");
    }
    return checked_power_of_two_row_count(quantization_bits, "expCheckProtocol");
}

size_t checked_exp_table_domain_size(size_t row_count)
{
    if (row_count > std::numeric_limits<size_t>::max() / 2) {
        throw std::runtime_error("expCheckProtocol: table domain size overflow");
    }
    return row_count * 2;
}

std::string exp_table_key(size_t quantization_bits)
{
    return "exp_check.table." + std::to_string(quantization_bits);
}

std::string signed_exp_table_key(const fixedPointEncoding& input_encoding)
{
    return "exp_check.signed_table.offset_binary.total_bits." + std::to_string(input_encoding.total_bits) +
           ".fractional_bits." + std::to_string(input_encoding.fractional_bits);
}

std::string non_positive_signed_exp_table_key(const fixedPointEncoding& input_encoding)
{
    return "exp_check.non_positive_signed_table.offset_binary.total_bits." + std::to_string(input_encoding.total_bits) +
           ".fractional_bits." + std::to_string(input_encoding.fractional_bits);
}

Fr fr_from_size_t(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

Fr fr_from_cpp_int(const cpp_int& value)
{
    if (value < 0) {
        throw std::runtime_error("expCheckProtocol: expected a non-negative exp table value");
    }

    const std::string repr = value.convert_to<std::string>();
    Fr out;
    out.setStr(repr, 10);
    if (out.getStr(10) != repr) {
        throw std::runtime_error("expCheckProtocol: exp table value does not fit canonically in Fr");
    }
    return out;
}

cpp_int quantized_exp_output(size_t input, size_t quantization_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(quantization_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

cpp_int quantized_exp_output(std::int64_t scaled_input, size_t fractional_bits)
{
    const cpp_dec_float_100 scale = pow(cpp_dec_float_100(2), static_cast<unsigned>(fractional_bits));
    const cpp_dec_float_100 x = cpp_dec_float_100(scaled_input) / scale;
    const cpp_dec_float_100 y = floor(exp(x) * scale);
    return y.convert_to<cpp_int>();
}

poly exp_table_polynomial(size_t quantization_bits)
{
    const size_t row_count = checked_exp_table_row_count(quantization_bits);
    std::vector<Fr> evals(checked_exp_table_domain_size(row_count), Fr(0));
    for (size_t input = 0; input < row_count; ++input) {
        evals[2 * input] = fr_from_size_t(input);
        evals[2 * input + 1] = fr_from_cpp_int(quantized_exp_output(input, quantization_bits));
    }

    return poly_from_evals(evals);
}

poly signed_exp_table_polynomial(const fixedPointEncoding& input_encoding)
{
    validate_fixed_point_encoding(input_encoding, "expCheckProtocol");
    if (!input_encoding.signed_values) {
        throw std::runtime_error("expCheckProtocol: signed exp tables require a signed fixed-point encoding");
    }

    const size_t row_count = checked_power_of_two_row_count(input_encoding.total_bits, "expCheckProtocol");
    std::vector<Fr> evals(checked_exp_table_domain_size(row_count), Fr(0));
    for (size_t encoded_input = 0; encoded_input < row_count; ++encoded_input) {
        // The lookup table is keyed by the encoded witness value, but the exp output is
        // computed on the decoded scaled integer that the fixed-point metadata describes.
        evals[2 * encoded_input] = fr_from_size_t(encoded_input);
        const std::int64_t scaled_input = decode_fixed_point_scaled_integer(encoded_input, input_encoding);
        evals[2 * encoded_input + 1] =
            fr_from_cpp_int(quantized_exp_output(scaled_input, input_encoding.fractional_bits));
    }

    return poly_from_evals(evals);
}

poly non_positive_signed_exp_table_polynomial(const fixedPointEncoding& input_encoding)
{
    validate_fixed_point_encoding(input_encoding, "expCheckProtocol");
    if (!input_encoding.signed_values) {
        throw std::runtime_error("expCheckProtocol: non-positive signed exp tables require a signed fixed-point encoding");
    }

    const size_t row_count = checked_power_of_two_row_count(input_encoding.total_bits, "expCheckProtocol");
    std::vector<Fr> evals(checked_exp_table_domain_size(row_count), Fr(0));
    for (size_t encoded_input = 0; encoded_input < row_count; ++encoded_input) {
        evals[2 * encoded_input] = fr_from_size_t(encoded_input);
        const std::int64_t scaled_input = decode_fixed_point_scaled_integer(encoded_input, input_encoding);
        if (scaled_input <= 0) {
            evals[2 * encoded_input + 1] =
                fr_from_cpp_int(quantized_exp_output(scaled_input, input_encoding.fractional_bits));
        } else {
            // Stabilized softmax only queries non-positive shifted scores. Positive rows are
            // intentionally mapped to a dummy value so large exp outputs do not need to be
            // represented in Fr; callers must separately constrain the queried inputs.
            evals[2 * encoded_input + 1] = Fr(0);
        }
    }

    return poly_from_evals(evals);
}

std::string table_group_key(const expCheckRecord& record)
{
    if (record.uses_signed_input_encoding) {
        if (record.assumes_non_positive_signed_inputs) {
            return non_positive_signed_exp_table_key(record.input_encoding);
        }
        return signed_exp_table_key(record.input_encoding);
    }
    return exp_table_key(record.quantization_bits);
}

size_t table_row_count(const expCheckRecord& record)
{
    if (record.uses_signed_input_encoding) {
        return checked_power_of_two_row_count(record.input_encoding.total_bits, "expCheckProtocol");
    }
    return checked_exp_table_row_count(record.quantization_bits);
}

poly table_polynomial(const expCheckRecord& record)
{
    if (record.uses_signed_input_encoding) {
        if (record.assumes_non_positive_signed_inputs) {
            return non_positive_signed_exp_table_polynomial(record.input_encoding);
        }
        return signed_exp_table_polynomial(record.input_encoding);
    }
    return exp_table_polynomial(record.quantization_bits);
}

std::string exp_table_cache_key(const kzgProtocol& kzg, const expCheckRecord& record)
{
    return transcript_representation(kzg.g1_power(0)) + ":" + transcript_representation(kzg.g1_power(1)) + ":" +
           table_group_key(record);
}

void cache_exp_table_group(const kzgProtocol& kzg, const expCheckRecord& record, const Group1& commitment)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = exp_table_cache_key(kzg, record);
    const auto it = cache.find(key);
    if (it == cache.end()) {
        cache.emplace(key, commitment);
        return;
    }
    if (it->second != commitment) {
        throw std::runtime_error("expCheckProtocol: inconsistent cached exp-table commitment");
    }
}

Group1 preprocessed_exp_table_group(const kzgProtocol& kzg, const expCheckRecord& record)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = exp_table_cache_key(kzg, record);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const Group1 commitment = kzg.commit_g1_group(table_polynomial(record));
    cache.emplace(key, commitment);
    return commitment;
}

bool verifier_session(const std::shared_ptr<ProofSession>& proof_session)
{
    return proof_session && proof_session->mode() == ProofMode::Verify;
}

virCommitment exp_table_commitment(kzgProtocol& kzg, const expCheckRecord& record, size_t table_domain_size)
{
    const std::string key = table_group_key(record);
    const std::shared_ptr<ProofSession> proof_session = kzg.proof_session();
    if (verifier_session(proof_session)) {
        const kzgCommitment commitment = kzg.adopt_public_commitment(key, preprocessed_exp_table_group(kzg, record));
        if (!kzg.has_degree_bound(virCommitment(commitment), table_domain_size - 1)) {
            kzg.prove_degree(commitment, table_domain_size - 1);
        }
        return virCommitment(commitment);
    }

    const poly table_poly = table_polynomial(record);
    const Group1 table_group = preprocessed_exp_table_group(kzg, record);
    const kzgCommitment commitment =
        kzg.commit_public_with_group(key, table_poly, table_group, table_domain_size - 1);
    cache_exp_table_group(kzg, record, commitment.com);
    return virCommitment(commitment);
}

} // namespace

void preprocess_non_positive_exp_check_table(kzgProtocol& kzg, const fixedPointEncoding& input_encoding)
{
    validate_fixed_point_encoding(input_encoding, "preprocess_non_positive_exp_check_table");
    if (!input_encoding.signed_values) {
        throw std::runtime_error("preprocess_non_positive_exp_check_table: expected signed fixed-point encoding");
    }
    expCheckRecord record;
    record.quantization_bits = input_encoding.fractional_bits;
    record.uses_signed_input_encoding = true;
    record.assumes_non_positive_signed_inputs = true;
    record.input_encoding = input_encoding;
    (void)preprocessed_exp_table_group(kzg, record);
}

void expCheckProtocol::add_check(const virCommitment& input_values,
                                 const virCommitment& output_values,
                                 size_t quantization_bits,
                                 size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("expCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.table_lookup == nullptr) {
        throw std::runtime_error("expCheckProtocol subprotocols not setup");
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("expCheckProtocol: domain size must be a non-zero power of two");
    }
    checked_exp_table_row_count(quantization_bits);

    expCheckRecord record;
    record.quantization_bits = quantization_bits;
    record.domain_size = domain_size;
    input_values_.push_back(input_values);
    output_values_.push_back(output_values);
    checks_.push_back(record);
}

void expCheckProtocol::add_check(const virCommitment& input_values,
                                 const virCommitment& output_values,
                                 const fixedPointEncoding& input_encoding,
                                 size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("expCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.table_lookup == nullptr) {
        throw std::runtime_error("expCheckProtocol subprotocols not setup");
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("expCheckProtocol: domain size must be a non-zero power of two");
    }

    validate_fixed_point_encoding(input_encoding, "expCheckProtocol");
    if (!input_encoding.signed_values) {
        throw std::runtime_error("expCheckProtocol: fixed-point overload currently expects a signed encoding");
    }
    // The signed path still uses a power-of-two lookup table. The key change is that the
    // table rows are indexed by encoded offset-binary values instead of plain unsigned inputs.
    checked_power_of_two_row_count(input_encoding.total_bits, "expCheckProtocol");

    expCheckRecord record;
    record.quantization_bits = input_encoding.fractional_bits;
    record.domain_size = domain_size;
    record.uses_signed_input_encoding = true;
    record.input_encoding = input_encoding;
    input_values_.push_back(input_values);
    output_values_.push_back(output_values);
    checks_.push_back(record);
}

void expCheckProtocol::add_non_positive_check(const virCommitment& input_values,
                                              const virCommitment& output_values,
                                              const fixedPointEncoding& input_encoding,
                                              size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("expCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.table_lookup == nullptr) {
        throw std::runtime_error("expCheckProtocol subprotocols not setup");
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("expCheckProtocol: domain size must be a non-zero power of two");
    }

    validate_fixed_point_encoding(input_encoding, "expCheckProtocol");
    if (!input_encoding.signed_values) {
        throw std::runtime_error("expCheckProtocol: non-positive fixed-point overload expects a signed encoding");
    }
    checked_power_of_two_row_count(input_encoding.total_bits, "expCheckProtocol");

    expCheckRecord record;
    record.quantization_bits = input_encoding.fractional_bits;
    record.domain_size = domain_size;
    record.uses_signed_input_encoding = true;
    record.assumes_non_positive_signed_inputs = true;
    record.input_encoding = input_encoding;
    input_values_.push_back(input_values);
    output_values_.push_back(output_values);
    checks_.push_back(record);
}

bool expCheckProtocol::finalize()
{
    begin_finalize("expCheckProtocol");

    if (checks_.empty()) {
        return true;
    }

    std::vector<size_t> group_representatives;
    std::map<std::string, std::vector<size_t>> grouped_check_indices;

    for (size_t i = 0; i < checks_.size(); ++i) {
        const std::string key = table_group_key(checks_[i]);
        if (grouped_check_indices.find(key) == grouped_check_indices.end()) {
            group_representatives.push_back(i);
        }
        grouped_check_indices[key].push_back(i);
    }

    for (size_t first_group_index : group_representatives) {
        const expCheckRecord& representative = checks_[first_group_index];
        const std::string group_key = table_group_key(representative);
        const size_t row_count = table_row_count(representative);
        const size_t table_domain_size = checked_exp_table_domain_size(row_count);
        const virCommitment table_commitment =
            exp_table_commitment(*params_.kzg, representative, table_domain_size);

        const std::vector<size_t>& group_indices = grouped_check_indices.at(group_key);
        std::vector<InterleavedBatchNode> leaves;
        leaves.reserve(group_indices.size());
        for (size_t index : group_indices) {
            const expCheckRecord& check = checks_[index];
            leaves.push_back(InterleavedBatchNode{{input_values_[index], output_values_[index]}, check.domain_size});
        }

        const InterleavedBatchNode merged = merge_interleaved_nodes(leaves, "expCheckProtocol");
        const virCommitment query_commitment =
            interleave_commitments(merged.commitments[0], merged.commitments[1], merged.domain_size, "expCheckProtocol");
        params_.table_lookup->add_check(query_commitment, table_commitment, merged.domain_size, row_count, 2);
    }

    return true;
}
