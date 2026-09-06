#include "forest_training_then_inference_io.hpp"

#include "algebra_utils.hpp"
#include "fixed_point.hpp"
#include "matrix_utils.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kCaseFormatTag = "forest_training_then_inference_case_v1";
constexpr const char* kBundleFormatTag = "forest_training_then_inference_bundle_v3";

using ParsedFields = std::map<std::string, std::vector<std::string>>;

std::string bytes_to_hex(const std::vector<uint8_t>& bytes)
{
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (uint8_t byte : bytes) {
        out.push_back(kHex[(byte >> 4u) & 0x0fu]);
        out.push_back(kHex[byte & 0x0fu]);
    }
    return out;
}

uint8_t hex_nibble(char c)
{
    if (c >= '0' && c <= '9') {
        return static_cast<uint8_t>(c - '0');
    }
    if (c >= 'a' && c <= 'f') {
        return static_cast<uint8_t>(10 + c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return static_cast<uint8_t>(10 + c - 'A');
    }
    throw std::runtime_error("forest_training_then_inference_io: invalid hex digit");
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex)
{
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("forest_training_then_inference_io: hex string must have even length");
    }

    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<uint8_t>((hex_nibble(hex[i]) << 4u) | hex_nibble(hex[i + 1])));
    }
    return out;
}

Fr parse_fr_token(const std::string& token, const char* context)
{
    Fr out;
    try {
        out.setStr(token, 10);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(context) + ": invalid field element token");
    }
    return out;
}

size_t parse_size_t_token(const std::string& token, const char* context)
{
    size_t out = 0;
    try {
        out = static_cast<size_t>(std::stoull(token));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(context) + ": invalid size_t token");
    }
    if (std::to_string(out) != token) {
        throw std::runtime_error(std::string(context) + ": non-canonical size_t token");
    }
    return out;
}

const std::vector<std::string>& required_field(const ParsedFields& fields, const std::string& name)
{
    const auto it = fields.find(name);
    if (it == fields.end()) {
        throw std::runtime_error("forest_training_then_inference_io: missing field '" + name + "'");
    }
    return it->second;
}

size_t required_scalar_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 1) {
        throw std::runtime_error("forest_training_then_inference_io: scalar field '" + name + "' must contain exactly one token");
    }
    return parse_size_t_token(values[0], name.c_str());
}

size_t optional_scalar_field(const ParsedFields& fields, const std::string& name, size_t default_value)
{
    const auto it = fields.find(name);
    if (it == fields.end()) {
        return default_value;
    }
    if (it->second.size() != 1) {
        throw std::runtime_error(
            "forest_training_then_inference_io: scalar field '" + name + "' must contain exactly one token");
    }
    return parse_size_t_token(it->second[0], name.c_str());
}

Fr required_fr_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 1) {
        throw std::runtime_error("forest_training_then_inference_io: fr field '" + name + "' must contain exactly one token");
    }
    return parse_fr_token(values[0], name.c_str());
}

Fr optional_fr_field(const ParsedFields& fields, const std::string& name, const Fr& default_value)
{
    const auto it = fields.find(name);
    if (it == fields.end()) {
        return default_value;
    }
    if (it->second.size() != 1) {
        throw std::runtime_error("forest_training_then_inference_io: fr field '" + name + "' must contain exactly one token");
    }
    return parse_fr_token(it->second[0], name.c_str());
}

fixedPointEncoding required_encoding_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 4) {
        throw std::runtime_error("forest_training_then_inference_io: encoding field '" + name + "' must contain 4 tokens");
    }
    if (values[0] != "offset_binary") {
        throw std::runtime_error("forest_training_then_inference_io: unsupported encoding kind in '" + name + "'");
    }

    fixedPointEncoding encoding;
    encoding.kind = fixedPointEncodingKind::OffsetBinary;
    encoding.signed_values = (parse_size_t_token(values[1], name.c_str()) != 0);
    encoding.total_bits = parse_size_t_token(values[2], name.c_str());
    encoding.fractional_bits = parse_size_t_token(values[3], name.c_str());
    validate_fixed_point_encoding(encoding, name.c_str());
    return encoding;
}

poly required_poly_field(const ParsedFields& fields, const std::string& name, size_t expected_count)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.empty()) {
        throw std::runtime_error("forest_training_then_inference_io: poly field '" + name + "' is empty");
    }
    const size_t stored_count = parse_size_t_token(values[0], name.c_str());
    if (stored_count != expected_count) {
        throw std::runtime_error("forest_training_then_inference_io: poly field '" + name + "' has unexpected domain size");
    }
    if (values.size() != stored_count + 1) {
        throw std::runtime_error("forest_training_then_inference_io: poly field '" + name + "' has inconsistent token count");
    }

    std::vector<Fr> evals;
    evals.reserve(stored_count);
    for (size_t i = 0; i < stored_count; ++i) {
        evals.push_back(parse_fr_token(values[i + 1], name.c_str()));
    }
    return poly_from_evals(evals);
}

poly optional_poly_field(const ParsedFields& fields,
                         const std::string& name,
                         size_t expected_count,
                         const std::vector<Fr>& default_evals)
{
    const auto it = fields.find(name);
    if (it == fields.end()) {
        if (default_evals.size() != expected_count) {
            throw std::runtime_error("forest_training_then_inference_io: invalid default domain for '" + name + "'");
        }
        return poly_from_evals(default_evals);
    }
    return required_poly_field(fields, name, expected_count);
}

size_t infer_max_depth_from_shape(size_t tree_count, size_t node_count)
{
    if (tree_count == 0 || node_count == 0 || node_count % tree_count != 0) {
        return 0;
    }
    size_t nodes_per_tree = node_count / tree_count;
    size_t log_nodes = 0;
    while (nodes_per_tree > 1 && nodes_per_tree % 2 == 0) {
        nodes_per_tree /= 2;
        ++log_nodes;
    }
    if (nodes_per_tree != 1 || log_nodes == 0) {
        return 0;
    }
    return log_nodes;
}

ParsedFields parse_fields_from_file(const std::string& path, const char* expected_format_tag)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("forest_training_then_inference_io: failed to open input file: " + path);
    }

    ParsedFields fields;
    std::string line;
    size_t line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string kind;
        if (!(iss >> kind)) {
            continue;
        }

        if (kind == "format") {
            std::string tag;
            if (!(iss >> tag) || tag != expected_format_tag) {
                throw std::runtime_error("forest_training_then_inference_io: unexpected format tag in " + path);
            }
            fields["__format__"] = {tag};
            continue;
        }

        std::string name;
        if (!(iss >> name)) {
            throw std::runtime_error(
                "forest_training_then_inference_io: malformed line " + std::to_string(line_number) + " in " + path);
        }

        if (fields.count(name) != 0) {
            throw std::runtime_error("forest_training_then_inference_io: duplicate field '" + name + "'");
        }

        std::vector<std::string> values;
        std::string value;
        while (iss >> value) {
            values.push_back(value);
        }

        if (kind != "scalar" && kind != "fr" && kind != "encoding" && kind != "poly" && kind != "proof") {
            throw std::runtime_error("forest_training_then_inference_io: unknown field kind '" + kind + "'");
        }
        fields[name] = std::move(values);
    }

    if (fields.count("__format__") == 0) {
        throw std::runtime_error("forest_training_then_inference_io: missing format line in " + path);
    }
    return fields;
}

void write_scalar_field(std::ostream& out, const std::string& name, size_t value)
{
    out << "scalar " << name << " " << value << "\n";
}

void write_fr_field(std::ostream& out, const std::string& name, const Fr& value)
{
    out << "fr " << name << " " << value.getStr(10) << "\n";
}

void write_encoding_field(std::ostream& out, const std::string& name, const fixedPointEncoding& encoding)
{
    validate_fixed_point_encoding(encoding, name.c_str());
    out << "encoding " << name << " offset_binary " << (encoding.signed_values ? 1 : 0) << " " << encoding.total_bits
        << " " << encoding.fractional_bits << "\n";
}

void write_poly_field(std::ostream& out, const std::string& name, const poly& polynomial, size_t domain_size)
{
    const std::vector<Fr> evals = polynomial.evaluations_on_domain(domain_size);
    out << "poly " << name << " " << domain_size;
    for (const Fr& value : evals) {
        out << " " << value.getStr(10);
    }
    out << "\n";
}

void write_proof_field(std::ostream& out, const std::string& name, const ProofArtifact& artifact)
{
    out << "proof " << name << " " << bytes_to_hex(artifact.proof.bytes) << "\n";
}

ProofArtifact read_proof_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 1) {
        throw std::runtime_error("forest_training_then_inference_io: proof field '" + name + "' must contain exactly one token");
    }

    SerializedProof proof;
    proof.bytes = hex_to_bytes(values[0]);
    return make_proof_artifact(std::move(proof));
}

void validate_case_query_batch_shape(const forestInferenceQueryBatch& queries)
{
    if (queries.query_count == 0 || queries.feature_count == 0) {
        throw std::runtime_error("forest_training_then_inference_io: query_count and feature_count must be non-zero");
    }
    if (!isPowerOfTwo(queries.query_count) || !isPowerOfTwo(queries.feature_count)) {
        throw std::runtime_error("forest_training_then_inference_io: query_count and feature_count must be powers of two");
    }
    validate_matrix_shape(queries.query_count, 1, "forest_training_then_inference_io inference.query_tree_ids");
    validate_matrix_shape(
        queries.query_count, queries.feature_count, "forest_training_then_inference_io inference.lifted_inputs");
}

void validate_case_witness_matches_queries(const forestExportedInferenceWitnessBatch& witness,
                                           const forestInferenceQueryBatch& queries)
{
    validate_case_query_batch_shape(queries);
    if (witness.query_count != queries.query_count || witness.feature_count != queries.feature_count) {
        throw std::runtime_error(
            "forest_training_then_inference_io: explicit inference witness shape must match inference queries");
    }
    validate_matrix_shape(
        witness.query_count, witness.feature_count, "forest_training_then_inference_io inference_witness.lower_bounds");
    validate_matrix_shape(
        witness.query_count, witness.feature_count, "forest_training_then_inference_io inference_witness.upper_bounds");
    validate_matrix_shape(witness.query_count, 1, "forest_training_then_inference_io inference_witness.scores");
    validate_matrix_shape(witness.query_count, 1, "forest_training_then_inference_io inference_witness.leaf_indices");
    if (witness.query_tree_ids.evaluations_on_domain(witness.query_count) !=
            queries.query_tree_ids.evaluations_on_domain(queries.query_count) ||
        witness.lifted_inputs.evaluations_on_domain(witness.query_count * witness.feature_count) !=
            queries.lifted_inputs.evaluations_on_domain(queries.query_count * queries.feature_count)) {
        throw std::runtime_error(
            "forest_training_then_inference_io: explicit inference witness must reuse the same query_tree_ids and lifted_inputs");
    }
}

size_t count_exported_leaf_rows_from_training_inputs(const forestTrainingProofInputs& inputs)
{
    if (inputs.node_count == 0) {
        throw std::runtime_error("forest_training_then_inference_io: node_count must be non-zero");
    }

    const std::vector<Fr> leaf_mask_evals = inputs.node_leaf_mask.evaluations_on_domain(inputs.node_count);
    size_t leaf_row_count = 0;
    for (const Fr& value : leaf_mask_evals) {
        const size_t leaf_mask = canonical_size_t_from_fr(value, "forest_training_then_inference_io node_leaf_mask");
        if (leaf_mask > 1) {
            throw std::runtime_error("forest_training_then_inference_io: node_leaf_mask must be boolean");
        }
        leaf_row_count += leaf_mask;
    }

    if (leaf_row_count == 0) {
        throw std::runtime_error("forest_training_then_inference_io: committed forest exports no leaf rows");
    }
    if (!isPowerOfTwo(leaf_row_count)) {
        throw std::runtime_error("forest_training_then_inference_io: exported leaf row count must be a power of two");
    }
    return leaf_row_count;
}

void write_training_inputs_fields(std::ostream& out, const forestTrainingProofInputs& inputs)
{
    const size_t query_row_count = inputs.tree_count * inputs.query_count;
    const size_t candidate_width = inputs.feature_count * inputs.bin_count;
    const size_t boosting_domain_size = inputs.iteration_count * inputs.class_count * inputs.query_count;

    write_scalar_field(out, "training.tree_count", inputs.tree_count);
    write_scalar_field(out, "training.node_count", inputs.node_count);
    write_scalar_field(out, "training.feature_count", inputs.feature_count);
    write_scalar_field(out, "training.bin_count", inputs.bin_count);
    write_scalar_field(out, "training.iteration_count", inputs.iteration_count);
    write_scalar_field(out, "training.class_count", inputs.class_count);
    write_scalar_field(out, "training.query_count", inputs.query_count);
    write_scalar_field(out, "training.input_bit_bound", inputs.input_bit_bound);
    write_scalar_field(out, "training.full_upper_bound", inputs.full_upper_bound);
    write_scalar_field(out, "training.structure_bit_bound", inputs.structure_bit_bound);
    write_scalar_field(out, "training.hessian_bit_bound", inputs.hessian_bit_bound);
    write_scalar_field(out, "training.gain_denominator_bit_bound", inputs.gain_denominator_bit_bound);
    write_scalar_field(out, "training.leaf_denominator_bit_bound", inputs.leaf_denominator_bit_bound);
    write_scalar_field(out, "training.gain_gap_bit_bound", inputs.gain_gap_bit_bound);
    write_scalar_field(out, "training.loss_mode", forest_training_loss_code(inputs.loss_mode));
    write_scalar_field(out, "training.l2_regularization", inputs.l2_regularization);
    write_scalar_field(out, "training.learning_rate_numerator", inputs.learning_rate_numerator);
    write_scalar_field(out, "training.learning_rate_denominator", inputs.learning_rate_denominator);
    write_fr_field(out, "training.split_penalty", inputs.split_penalty);
    write_fr_field(out, "training.split_gain_threshold", inputs.split_gain_threshold);
    write_scalar_field(out, "training.max_depth", inputs.max_depth);

    write_encoding_field(out, "training.gradient_encoding", inputs.gradient_encoding);
    write_encoding_field(out, "training.gain_numerator_encoding", inputs.gain_numerator_encoding);
    write_encoding_field(out, "training.quantized_gain_encoding", inputs.quantized_gain_encoding);
    write_encoding_field(out, "training.leaf_value_encoding", inputs.leaf_value_encoding);
    write_encoding_field(out, "training.score_encoding", inputs.score_encoding);
    write_encoding_field(out, "training.softmax_encoding", inputs.softmax_encoding);

    write_poly_field(out, "training.root_indices", inputs.root_indices, inputs.tree_count);
    write_poly_field(out, "training.split_node_mask", inputs.split_node_mask, inputs.node_count);
    write_poly_field(out, "training.node_depths", inputs.node_depths, inputs.node_count);
    write_poly_field(out, "training.left_child_indices", inputs.left_child_indices, inputs.node_count);
    write_poly_field(out, "training.right_child_indices", inputs.right_child_indices, inputs.node_count);
    write_poly_field(out, "training.node_tree_ids", inputs.node_tree_ids, inputs.node_count);
    write_poly_field(out, "training.feature_selector", inputs.feature_selector, inputs.node_count * inputs.feature_count);
    write_poly_field(out, "training.lower_bounds", inputs.lower_bounds, inputs.node_count * inputs.feature_count);
    write_poly_field(out, "training.upper_bounds", inputs.upper_bounds, inputs.node_count * inputs.feature_count);
    write_poly_field(out, "training.left_lower", inputs.left_lower, inputs.node_count * inputs.feature_count);
    write_poly_field(out, "training.left_upper", inputs.left_upper, inputs.node_count * inputs.feature_count);
    write_poly_field(out, "training.right_lower", inputs.right_lower, inputs.node_count * inputs.feature_count);
    write_poly_field(out, "training.right_upper", inputs.right_upper, inputs.node_count * inputs.feature_count);
    write_poly_field(out, "training.thresholds", inputs.thresholds, inputs.node_count);

    write_poly_field(out, "training.query_tree_ids", inputs.query_tree_ids, query_row_count);
    write_poly_field(out, "training.lifted_inputs", inputs.lifted_inputs, query_row_count * inputs.feature_count);
    write_poly_field(out, "training.leaf_indices", inputs.leaf_indices, query_row_count);
    write_poly_field(out, "training.leaf_lower_bounds", inputs.leaf_lower_bounds, query_row_count * inputs.feature_count);
    write_poly_field(out, "training.leaf_upper_bounds", inputs.leaf_upper_bounds, query_row_count * inputs.feature_count);
    write_poly_field(out, "training.tree_scores", inputs.tree_scores, query_row_count);
    write_poly_field(out, "training.training_node_tree_ids", inputs.training_node_tree_ids, inputs.node_count);
    write_poly_field(
        out, "training.training_node_lower_bounds", inputs.training_node_lower_bounds, inputs.node_count * inputs.feature_count);
    write_poly_field(
        out, "training.training_node_upper_bounds", inputs.training_node_upper_bounds, inputs.node_count * inputs.feature_count);
    write_poly_field(out, "training.node_leaf_mask", inputs.node_leaf_mask, inputs.node_count);
    write_poly_field(out, "training.node_leaf_values", inputs.node_leaf_values, inputs.node_count);
    write_poly_field(out, "training.lifted_leaf_indices", inputs.lifted_leaf_indices, query_row_count * inputs.feature_count);
    write_poly_field(
        out, "training.flattened_bucket_indices", inputs.flattened_bucket_indices, query_row_count * inputs.feature_count);

    for (size_t channel = 0; channel < inputs.channels.size(); ++channel) {
        const std::string prefix = "training.channel" + std::to_string(channel);
        write_poly_field(out, prefix + ".query_values", inputs.channels[channel].query_values, query_row_count);
        write_poly_field(
            out, prefix + ".lifted_values", inputs.channels[channel].lifted_values, query_row_count * inputs.feature_count);
        write_poly_field(
            out, prefix + ".leaf_histograms", inputs.channels[channel].leaf_histograms, inputs.node_count * candidate_width);
        write_poly_field(
            out, prefix + ".full_histograms", inputs.channels[channel].full_histograms, inputs.node_count * candidate_width);
        write_poly_field(
            out, prefix + ".split_histograms", inputs.channels[channel].split_histograms, inputs.node_count * candidate_width);
        write_poly_field(
            out, prefix + ".left_statistics", inputs.channels[channel].left_statistics, inputs.node_count * candidate_width);
        write_poly_field(
            out, prefix + ".right_statistics", inputs.channels[channel].right_statistics, inputs.node_count * candidate_width);
    }

    write_poly_field(out, "training.left_gradients", inputs.left_gradients, inputs.node_count * candidate_width);
    write_poly_field(out, "training.right_gradients", inputs.right_gradients, inputs.node_count * candidate_width);
    write_poly_field(out, "training.left_hessians", inputs.left_hessians, inputs.node_count * candidate_width);
    write_poly_field(out, "training.right_hessians", inputs.right_hessians, inputs.node_count * candidate_width);
    write_poly_field(
        out, "training.left_gradient_squares", inputs.left_gradient_squares, inputs.node_count * candidate_width);
    write_poly_field(
        out, "training.right_gradient_squares", inputs.right_gradient_squares, inputs.node_count * candidate_width);
    write_poly_field(
        out, "training.parent_gradient_squares", inputs.parent_gradient_squares, inputs.node_count * candidate_width);
    write_poly_field(out, "training.gain_numerators", inputs.gain_numerators, inputs.node_count * candidate_width);
    write_poly_field(out, "training.gain_denominators", inputs.gain_denominators, inputs.node_count * candidate_width);
    write_poly_field(
        out, "training.encoded_gain_numerators", inputs.encoded_gain_numerators, inputs.node_count * candidate_width);
    write_poly_field(
        out, "training.encoded_quantized_gains", inputs.encoded_quantized_gains, inputs.node_count * candidate_width);
    write_poly_field(out, "training.gain_remainders", inputs.gain_remainders, inputs.node_count * candidate_width);

    write_poly_field(out, "training.quantized_gains", inputs.quantized_gains, inputs.node_count * candidate_width);
    write_poly_field(out, "training.tie_break_scores", inputs.tie_break_scores, inputs.node_count * candidate_width);
    write_poly_field(out, "training.candidate_mask", inputs.candidate_mask, inputs.node_count * candidate_width);
    write_poly_field(out, "training.active_node_mask", inputs.active_node_mask, inputs.node_count);
    write_poly_field(out, "training.depth_limit_mask", inputs.depth_limit_mask, inputs.node_count);
    write_poly_field(out, "training.selector", inputs.selector, inputs.node_count * candidate_width);
    write_poly_field(out, "training.selected_gains", inputs.selected_gains, inputs.node_count);
    write_poly_field(out, "training.selected_tie_break_scores", inputs.selected_tie_break_scores, inputs.node_count);
    write_poly_field(out, "training.selected_candidate_indices", inputs.selected_candidate_indices, inputs.node_count);
    write_poly_field(out, "training.selected_features", inputs.selected_features, inputs.node_count);
    write_poly_field(out, "training.selected_bins", inputs.selected_bins, inputs.node_count);

    write_poly_field(out, "training.leaf_gradients", inputs.leaf_gradients, inputs.node_count);
    write_poly_field(out, "training.leaf_hessians", inputs.leaf_hessians, inputs.node_count);
    write_poly_field(out, "training.leaf_denominators", inputs.leaf_denominators, inputs.node_count);
    write_poly_field(out, "training.encoded_negative_leaf_gradients", inputs.encoded_negative_leaf_gradients, inputs.node_count);
    write_poly_field(out, "training.encoded_leaf_values", inputs.encoded_leaf_values, inputs.node_count);
    write_poly_field(out, "training.leaf_remainders", inputs.leaf_remainders, inputs.node_count);

    write_poly_field(out, "training.cumulative_scores", inputs.cumulative_scores, boosting_domain_size);
    write_poly_field(out, "training.encoded_cumulative_scores", inputs.encoded_cumulative_scores, boosting_domain_size);
    write_poly_field(out, "training.softmax_shifts", inputs.softmax_shifts, inputs.iteration_count * inputs.query_count);
    write_poly_field(out, "training.shift_selector", inputs.shift_selector, boosting_domain_size);
    write_poly_field(out, "training.lifted_softmax_shifts", inputs.lifted_softmax_shifts, boosting_domain_size);
    write_poly_field(out, "training.encoded_shifted_scores", inputs.encoded_shifted_scores, boosting_domain_size);
    write_poly_field(out, "training.softmax_exp_values", inputs.softmax_exp_values, boosting_domain_size);
    write_poly_field(out,
                     "training.softmax_denominators",
                     inputs.softmax_denominators,
                     inputs.iteration_count * inputs.query_count);
    write_poly_field(
        out, "training.lifted_softmax_denominators", inputs.lifted_softmax_denominators, boosting_domain_size);
    write_poly_field(out, "training.softmax_probabilities", inputs.softmax_probabilities, boosting_domain_size);
    write_poly_field(out, "training.softmax_remainders", inputs.softmax_remainders, boosting_domain_size);
    write_poly_field(out, "training.base_labels", inputs.base_labels, inputs.class_count * inputs.query_count);
    write_poly_field(out, "training.pseudo_residuals", inputs.pseudo_residuals, boosting_domain_size);
    write_poly_field(out, "training.encoded_diagonal_hessians", inputs.encoded_diagonal_hessians, boosting_domain_size);
    write_poly_field(
        out, "training.diagonal_hessian_remainders", inputs.diagonal_hessian_remainders, boosting_domain_size);
    write_poly_field(out, "training.diagonal_hessians", inputs.diagonal_hessians, boosting_domain_size);
}

forestTrainingProofInputs read_training_inputs_fields(const ParsedFields& fields)
{
    forestTrainingProofInputs inputs;

    inputs.tree_count = required_scalar_field(fields, "training.tree_count");
    inputs.node_count = required_scalar_field(fields, "training.node_count");
    inputs.feature_count = required_scalar_field(fields, "training.feature_count");
    inputs.bin_count = required_scalar_field(fields, "training.bin_count");
    inputs.iteration_count = required_scalar_field(fields, "training.iteration_count");
    inputs.class_count = required_scalar_field(fields, "training.class_count");
    inputs.query_count = required_scalar_field(fields, "training.query_count");
    inputs.input_bit_bound = required_scalar_field(fields, "training.input_bit_bound");
    inputs.full_upper_bound = required_scalar_field(fields, "training.full_upper_bound");
    inputs.structure_bit_bound = required_scalar_field(fields, "training.structure_bit_bound");
    inputs.hessian_bit_bound = required_scalar_field(fields, "training.hessian_bit_bound");
    inputs.gain_denominator_bit_bound = required_scalar_field(fields, "training.gain_denominator_bit_bound");
    inputs.leaf_denominator_bit_bound = required_scalar_field(fields, "training.leaf_denominator_bit_bound");
    inputs.gain_gap_bit_bound = required_scalar_field(fields, "training.gain_gap_bit_bound");
    inputs.loss_mode = forest_training_loss_from_code(optional_scalar_field(
        fields, "training.loss_mode", forest_training_loss_code(forestTrainingLoss::SecondOrderXgboost)));
    inputs.l2_regularization = required_scalar_field(fields, "training.l2_regularization");
    inputs.learning_rate_numerator = optional_scalar_field(fields, "training.learning_rate_numerator", 1);
    inputs.learning_rate_denominator = optional_scalar_field(fields, "training.learning_rate_denominator", 1);
    inputs.split_penalty = required_fr_field(fields, "training.split_penalty");
    inputs.split_gain_threshold = optional_fr_field(fields, "training.split_gain_threshold", Fr(0));
    inputs.max_depth = optional_scalar_field(
        fields, "training.max_depth", infer_max_depth_from_shape(inputs.tree_count, inputs.node_count));

    inputs.gradient_encoding = required_encoding_field(fields, "training.gradient_encoding");
    inputs.gain_numerator_encoding = required_encoding_field(fields, "training.gain_numerator_encoding");
    inputs.quantized_gain_encoding = required_encoding_field(fields, "training.quantized_gain_encoding");
    inputs.leaf_value_encoding = required_encoding_field(fields, "training.leaf_value_encoding");
    inputs.score_encoding = required_encoding_field(fields, "training.score_encoding");
    inputs.softmax_encoding = required_encoding_field(fields, "training.softmax_encoding");

    const size_t query_row_count = inputs.tree_count * inputs.query_count;
    const size_t candidate_width = inputs.feature_count * inputs.bin_count;
    const size_t boosting_domain_size = inputs.iteration_count * inputs.class_count * inputs.query_count;

    inputs.root_indices = required_poly_field(fields, "training.root_indices", inputs.tree_count);
    inputs.split_node_mask = required_poly_field(fields, "training.split_node_mask", inputs.node_count);
    inputs.node_depths = required_poly_field(fields, "training.node_depths", inputs.node_count);
    inputs.left_child_indices = required_poly_field(fields, "training.left_child_indices", inputs.node_count);
    inputs.right_child_indices = required_poly_field(fields, "training.right_child_indices", inputs.node_count);
    inputs.node_tree_ids = required_poly_field(fields, "training.node_tree_ids", inputs.node_count);
    inputs.feature_selector = required_poly_field(fields, "training.feature_selector", inputs.node_count * inputs.feature_count);
    inputs.lower_bounds = required_poly_field(fields, "training.lower_bounds", inputs.node_count * inputs.feature_count);
    inputs.upper_bounds = required_poly_field(fields, "training.upper_bounds", inputs.node_count * inputs.feature_count);
    inputs.left_lower = required_poly_field(fields, "training.left_lower", inputs.node_count * inputs.feature_count);
    inputs.left_upper = required_poly_field(fields, "training.left_upper", inputs.node_count * inputs.feature_count);
    inputs.right_lower = required_poly_field(fields, "training.right_lower", inputs.node_count * inputs.feature_count);
    inputs.right_upper = required_poly_field(fields, "training.right_upper", inputs.node_count * inputs.feature_count);
    inputs.thresholds = required_poly_field(fields, "training.thresholds", inputs.node_count);

    inputs.query_tree_ids = required_poly_field(fields, "training.query_tree_ids", query_row_count);
    inputs.lifted_inputs = required_poly_field(fields, "training.lifted_inputs", query_row_count * inputs.feature_count);
    inputs.leaf_indices = required_poly_field(fields, "training.leaf_indices", query_row_count);
    inputs.leaf_lower_bounds = required_poly_field(fields, "training.leaf_lower_bounds", query_row_count * inputs.feature_count);
    inputs.leaf_upper_bounds = required_poly_field(fields, "training.leaf_upper_bounds", query_row_count * inputs.feature_count);
    inputs.tree_scores = required_poly_field(fields, "training.tree_scores", query_row_count);
    inputs.training_node_tree_ids = required_poly_field(fields, "training.training_node_tree_ids", inputs.node_count);
    inputs.training_node_lower_bounds =
        required_poly_field(fields, "training.training_node_lower_bounds", inputs.node_count * inputs.feature_count);
    inputs.training_node_upper_bounds =
        required_poly_field(fields, "training.training_node_upper_bounds", inputs.node_count * inputs.feature_count);
    inputs.node_leaf_mask = required_poly_field(fields, "training.node_leaf_mask", inputs.node_count);
    inputs.node_leaf_values = required_poly_field(fields, "training.node_leaf_values", inputs.node_count);
    inputs.lifted_leaf_indices =
        required_poly_field(fields, "training.lifted_leaf_indices", query_row_count * inputs.feature_count);
    inputs.flattened_bucket_indices =
        required_poly_field(fields, "training.flattened_bucket_indices", query_row_count * inputs.feature_count);

    for (size_t channel = 0; channel < inputs.channels.size(); ++channel) {
        const std::string prefix = "training.channel" + std::to_string(channel);
        inputs.channels[channel].query_values = required_poly_field(fields, prefix + ".query_values", query_row_count);
        inputs.channels[channel].lifted_values =
            required_poly_field(fields, prefix + ".lifted_values", query_row_count * inputs.feature_count);
        inputs.channels[channel].leaf_histograms =
            required_poly_field(fields, prefix + ".leaf_histograms", inputs.node_count * candidate_width);
        inputs.channels[channel].full_histograms =
            required_poly_field(fields, prefix + ".full_histograms", inputs.node_count * candidate_width);
        inputs.channels[channel].split_histograms =
            required_poly_field(fields, prefix + ".split_histograms", inputs.node_count * candidate_width);
        inputs.channels[channel].left_statistics =
            required_poly_field(fields, prefix + ".left_statistics", inputs.node_count * candidate_width);
        inputs.channels[channel].right_statistics =
            required_poly_field(fields, prefix + ".right_statistics", inputs.node_count * candidate_width);
    }

    inputs.left_gradients = required_poly_field(fields, "training.left_gradients", inputs.node_count * candidate_width);
    inputs.right_gradients = required_poly_field(fields, "training.right_gradients", inputs.node_count * candidate_width);
    inputs.left_hessians = required_poly_field(fields, "training.left_hessians", inputs.node_count * candidate_width);
    inputs.right_hessians = required_poly_field(fields, "training.right_hessians", inputs.node_count * candidate_width);
    inputs.left_gradient_squares =
        required_poly_field(fields, "training.left_gradient_squares", inputs.node_count * candidate_width);
    inputs.right_gradient_squares =
        required_poly_field(fields, "training.right_gradient_squares", inputs.node_count * candidate_width);
    inputs.parent_gradient_squares =
        required_poly_field(fields, "training.parent_gradient_squares", inputs.node_count * candidate_width);
    inputs.gain_numerators = required_poly_field(fields, "training.gain_numerators", inputs.node_count * candidate_width);
    inputs.gain_denominators =
        required_poly_field(fields, "training.gain_denominators", inputs.node_count * candidate_width);
    inputs.encoded_gain_numerators =
        required_poly_field(fields, "training.encoded_gain_numerators", inputs.node_count * candidate_width);
    inputs.encoded_quantized_gains =
        required_poly_field(fields, "training.encoded_quantized_gains", inputs.node_count * candidate_width);
    inputs.gain_remainders = required_poly_field(fields, "training.gain_remainders", inputs.node_count * candidate_width);

    inputs.quantized_gains = required_poly_field(fields, "training.quantized_gains", inputs.node_count * candidate_width);
    inputs.tie_break_scores =
        required_poly_field(fields, "training.tie_break_scores", inputs.node_count * candidate_width);
    inputs.candidate_mask = required_poly_field(fields, "training.candidate_mask", inputs.node_count * candidate_width);
    inputs.active_node_mask = required_poly_field(fields, "training.active_node_mask", inputs.node_count);
    inputs.depth_limit_mask =
        optional_poly_field(fields, "training.depth_limit_mask", inputs.node_count, std::vector<Fr>(inputs.node_count, Fr(0)));
    inputs.selector = required_poly_field(fields, "training.selector", inputs.node_count * candidate_width);
    inputs.selected_gains = required_poly_field(fields, "training.selected_gains", inputs.node_count);
    inputs.selected_tie_break_scores =
        required_poly_field(fields, "training.selected_tie_break_scores", inputs.node_count);
    inputs.selected_candidate_indices =
        required_poly_field(fields, "training.selected_candidate_indices", inputs.node_count);
    inputs.selected_features = required_poly_field(fields, "training.selected_features", inputs.node_count);
    inputs.selected_bins = required_poly_field(fields, "training.selected_bins", inputs.node_count);

    inputs.leaf_gradients = required_poly_field(fields, "training.leaf_gradients", inputs.node_count);
    inputs.leaf_hessians = required_poly_field(fields, "training.leaf_hessians", inputs.node_count);
    inputs.leaf_denominators = required_poly_field(fields, "training.leaf_denominators", inputs.node_count);
    inputs.encoded_negative_leaf_gradients =
        required_poly_field(fields, "training.encoded_negative_leaf_gradients", inputs.node_count);
    inputs.encoded_leaf_values = required_poly_field(fields, "training.encoded_leaf_values", inputs.node_count);
    inputs.leaf_remainders = required_poly_field(fields, "training.leaf_remainders", inputs.node_count);

    inputs.cumulative_scores = required_poly_field(fields, "training.cumulative_scores", boosting_domain_size);
    inputs.encoded_cumulative_scores =
        required_poly_field(fields, "training.encoded_cumulative_scores", boosting_domain_size);
    inputs.softmax_shifts =
        required_poly_field(fields, "training.softmax_shifts", inputs.iteration_count * inputs.query_count);
    inputs.shift_selector = required_poly_field(fields, "training.shift_selector", boosting_domain_size);
    inputs.lifted_softmax_shifts = required_poly_field(fields, "training.lifted_softmax_shifts", boosting_domain_size);
    inputs.encoded_shifted_scores = required_poly_field(fields, "training.encoded_shifted_scores", boosting_domain_size);
    inputs.softmax_exp_values = required_poly_field(fields, "training.softmax_exp_values", boosting_domain_size);
    inputs.softmax_denominators =
        required_poly_field(fields, "training.softmax_denominators", inputs.iteration_count * inputs.query_count);
    inputs.lifted_softmax_denominators =
        required_poly_field(fields, "training.lifted_softmax_denominators", boosting_domain_size);
    inputs.softmax_probabilities = required_poly_field(fields, "training.softmax_probabilities", boosting_domain_size);
    inputs.softmax_remainders = required_poly_field(fields, "training.softmax_remainders", boosting_domain_size);
    inputs.base_labels = required_poly_field(fields, "training.base_labels", inputs.class_count * inputs.query_count);
    inputs.pseudo_residuals = required_poly_field(fields, "training.pseudo_residuals", boosting_domain_size);
    inputs.encoded_diagonal_hessians =
        required_poly_field(fields, "training.encoded_diagonal_hessians", boosting_domain_size);
    inputs.diagonal_hessian_remainders =
        required_poly_field(fields, "training.diagonal_hessian_remainders", boosting_domain_size);
    inputs.diagonal_hessians = required_poly_field(fields, "training.diagonal_hessians", boosting_domain_size);

    return inputs;
}

void write_inference_query_fields(std::ostream& out, const forestInferenceQueryBatch& queries)
{
    write_scalar_field(out, "inference.query_count", queries.query_count);
    write_scalar_field(out, "inference.feature_count", queries.feature_count);
    write_poly_field(out, "inference.query_tree_ids", queries.query_tree_ids, queries.query_count);
    write_poly_field(out, "inference.lifted_inputs", queries.lifted_inputs, queries.query_count * queries.feature_count);
}

forestInferenceQueryBatch read_inference_query_fields(const ParsedFields& fields)
{
    forestInferenceQueryBatch queries;
    queries.query_count = required_scalar_field(fields, "inference.query_count");
    queries.feature_count = required_scalar_field(fields, "inference.feature_count");
    queries.query_tree_ids = required_poly_field(fields, "inference.query_tree_ids", queries.query_count);
    queries.lifted_inputs =
        required_poly_field(fields, "inference.lifted_inputs", queries.query_count * queries.feature_count);
    return queries;
}

void write_optional_inference_witness_fields(std::ostream& out,
                                             const forestTrainingThenInferenceInputs& inputs)
{
    write_scalar_field(out, "inference_witness.present", inputs.explicit_inference_witness.has_value() ? 1u : 0u);
    if (!inputs.explicit_inference_witness.has_value()) {
        return;
    }

    const forestExportedInferenceWitnessBatch& witness = *inputs.explicit_inference_witness;
    write_poly_field(out, "inference_witness.lower_bounds", witness.lower_bounds, witness.query_count * witness.feature_count);
    write_poly_field(out, "inference_witness.upper_bounds", witness.upper_bounds, witness.query_count * witness.feature_count);
    write_poly_field(out, "inference_witness.scores", witness.scores, witness.query_count);
    write_poly_field(out, "inference_witness.leaf_indices", witness.leaf_indices, witness.query_count);
}

std::optional<forestExportedInferenceWitnessBatch> read_optional_inference_witness_fields(
    const ParsedFields& fields,
    const forestInferenceQueryBatch& queries)
{
    const auto it = fields.find("inference_witness.present");
    if (it == fields.end()) {
        return std::nullopt;
    }

    if (it->second.size() != 1) {
        throw std::runtime_error(
            "forest_training_then_inference_io: scalar field 'inference_witness.present' must contain exactly one token");
    }
    const size_t present = parse_size_t_token(it->second[0], "inference_witness.present");
    if (present == 0) {
        return std::nullopt;
    }
    if (present != 1) {
        throw std::runtime_error("forest_training_then_inference_io: inference_witness.present must be 0 or 1");
    }

    forestExportedInferenceWitnessBatch witness;
    witness.query_tree_ids = queries.query_tree_ids;
    witness.lifted_inputs = queries.lifted_inputs;
    witness.query_count = queries.query_count;
    witness.feature_count = queries.feature_count;
    witness.lower_bounds =
        required_poly_field(fields, "inference_witness.lower_bounds", queries.query_count * queries.feature_count);
    witness.upper_bounds =
        required_poly_field(fields, "inference_witness.upper_bounds", queries.query_count * queries.feature_count);
    witness.scores = required_poly_field(fields, "inference_witness.scores", queries.query_count);
    witness.leaf_indices = required_poly_field(fields, "inference_witness.leaf_indices", queries.query_count);
    return witness;
}

void write_exported_leaf_table_fields(std::ostream& out, const forestExportedLeafTable& table)
{
    write_scalar_field(out, "exported.row_count", table.row_count);
    write_scalar_field(out, "exported.feature_count", table.feature_count);
    write_poly_field(out, "exported.tree_indices", table.tree_indices, table.row_count);
    write_poly_field(out, "exported.lower_bounds", table.lower_bounds, table.row_count * table.feature_count);
    write_poly_field(out, "exported.upper_bounds", table.upper_bounds, table.row_count * table.feature_count);
    write_poly_field(out, "exported.scores", table.scores, table.row_count);
    write_poly_field(out, "exported.leaf_indices", table.leaf_indices, table.row_count);
}

forestExportedLeafTable read_exported_leaf_table_fields(const ParsedFields& fields)
{
    forestExportedLeafTable table;
    table.row_count = required_scalar_field(fields, "exported.row_count");
    table.feature_count = required_scalar_field(fields, "exported.feature_count");
    table.tree_indices = required_poly_field(fields, "exported.tree_indices", table.row_count);
    table.lower_bounds = required_poly_field(fields, "exported.lower_bounds", table.row_count * table.feature_count);
    table.upper_bounds = required_poly_field(fields, "exported.upper_bounds", table.row_count * table.feature_count);
    table.scores = required_poly_field(fields, "exported.scores", table.row_count);
    table.leaf_indices = required_poly_field(fields, "exported.leaf_indices", table.row_count);
    return table;
}

} // namespace

forestTrainingThenInferenceCaseSummary summarize_forest_training_then_inference_inputs(
    const forestTrainingThenInferenceInputs& inputs)
{
    forestTrainingThenInferenceCaseSummary summary;
    summary.training_tree_count = inputs.training_inputs.tree_count;
    summary.training_node_count = inputs.training_inputs.node_count;
    summary.training_feature_count = inputs.training_inputs.feature_count;
    summary.training_query_count = inputs.training_inputs.query_count;
    summary.inference_query_count = inputs.inference_queries.query_count;
    summary.explicit_inference_witness_provided = inputs.explicit_inference_witness.has_value();
    summary.exported_leaf_row_count = count_exported_leaf_rows_from_training_inputs(inputs.training_inputs);

    validate_case_query_batch_shape(inputs.inference_queries);
    if (inputs.explicit_inference_witness.has_value()) {
        validate_case_witness_matches_queries(*inputs.explicit_inference_witness, inputs.inference_queries);
    }
    return summary;
}

void validate_forest_training_then_inference_inputs(const forestTrainingThenInferenceInputs& inputs)
{
    (void)summarize_forest_training_then_inference_inputs(inputs);
}

std::string forest_training_then_inference_case_format_description()
{
    return
        "Plain-text case format for the combined committed-forest training and exported inference workflow.\n"
        "Each non-comment line starts with one of:\n"
        "- format <tag>\n"
        "- scalar <name> <value>\n"
        "- fr <name> <decimal-field-element>\n"
        "- encoding <name> offset_binary <signed:0|1> <total_bits> <fractional_bits>\n"
        "- poly <name> <domain_size> <eval_0> ... <eval_n-1>\n"
        "\n"
        "The case contains two namespaces:\n"
        "- training.* for the full forest_training_workflow witness bundle\n"
        "- inference.* for the query batch proved against the exported leaf table\n"
        "- optional inference_witness.* for an explicit exported-inference witness that avoids automatic row matching\n"
        "\n"
        "Important input requirements:\n"
        "- training.* must already satisfy the shape and semantic requirements of forest_training_workflow\n"
        "- inference.feature_count must equal training.feature_count\n"
        "- each inference query must match exactly one exported leaf row from the committed forest\n"
        "- inference.query_count and inference.feature_count must be non-zero powers of two\n"
        "- when inference_witness.present = 1, the provided bounds / score / leaf-index witness must match inference.* and will be used directly; otherwise the workflow derives that witness automatically from the exported leaf table.\n";
}

std::string forest_training_then_inference_bundle_format_description()
{
    return
        "Plain-text bundle format for the combined committed-forest training and exported inference workflow.\n"
        "Each non-comment line starts with one of:\n"
        "- format <tag>\n"
        "- scalar <name> <value>\n"
        "- poly <name> <domain_size> <eval_0> ... <eval_n-1>\n"
        "- proof <name> <hex-encoded-proof-bytes>\n"
        "\n"
        "The bundle contains:\n"
        "- one independent forest well-formedness proof artifact\n"
        "- the training proof artifact\n"
        "- one exported-inference proof artifact\n"
        "- the exported leaf table used as the bridge into the fixed-table inference route.\n";
}

void write_forest_training_then_inference_case(const std::string& path,
                                               const forestTrainingThenInferenceInputs& inputs)
{
    validate_forest_training_then_inference_inputs(inputs);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("forest_training_then_inference_io: failed to open output file: " + path);
    }

    out << "format " << kCaseFormatTag << "\n";
    write_training_inputs_fields(out, inputs.training_inputs);
    write_inference_query_fields(out, inputs.inference_queries);
    write_optional_inference_witness_fields(out, inputs);
}

forestTrainingThenInferenceInputs read_forest_training_then_inference_case(const std::string& path)
{
    const ParsedFields fields = parse_fields_from_file(path, kCaseFormatTag);

    forestTrainingThenInferenceInputs inputs;
    inputs.training_inputs = read_training_inputs_fields(fields);
    inputs.inference_queries = read_inference_query_fields(fields);
    inputs.explicit_inference_witness = read_optional_inference_witness_fields(fields, inputs.inference_queries);
    validate_forest_training_then_inference_inputs(inputs);
    return inputs;
}

void write_forest_training_then_inference_bundle(const std::string& path,
                                                 const forestTrainingThenInferenceArtifact& artifact)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("forest_training_then_inference_io: failed to open output file: " + path);
    }

    out << "format " << kBundleFormatTag << "\n";
    write_exported_leaf_table_fields(out, artifact.exported_leaf_table);
    write_proof_field(out, "bundle.well_formedness_proof", artifact.well_formedness_artifact);
    write_proof_field(out, "bundle.training_proof", artifact.training_artifact);
    write_proof_field(out, "bundle.inference_proof", artifact.inference_artifact);
}

forestTrainingThenInferenceArtifact read_forest_training_then_inference_bundle(const std::string& path)
{
    const ParsedFields fields = parse_fields_from_file(path, kBundleFormatTag);

    forestTrainingThenInferenceArtifact artifact;
    artifact.exported_leaf_table = read_exported_leaf_table_fields(fields);
    artifact.well_formedness_artifact = read_proof_field(fields, "bundle.well_formedness_proof");
    artifact.training_artifact = read_proof_field(fields, "bundle.training_proof");
    artifact.inference_artifact = read_proof_field(fields, "bundle.inference_proof");
    return artifact;
}
