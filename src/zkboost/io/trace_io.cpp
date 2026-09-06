#include "zkboost/io/trace_io.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kTrainingFormatTag = "zkboost_binary_training_trace_v1";
constexpr const char* kInferenceFormatTag = "zkboost_binary_inference_trace_v1";

using ParsedFields = std::map<std::string, std::vector<std::string>>;

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

std::int64_t parse_int64_token(const std::string& token, const char* context)
{
    std::int64_t out = 0;
    try {
        out = static_cast<std::int64_t>(std::stoll(token));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(context) + ": invalid int64 token");
    }
    if (std::to_string(out) != token) {
        throw std::runtime_error(std::string(context) + ": non-canonical int64 token");
    }
    return out;
}

const std::vector<std::string>& required_field(const ParsedFields& fields, const std::string& name)
{
    const auto it = fields.find(name);
    if (it == fields.end()) {
        throw std::runtime_error("zkboost_trace_io: missing field '" + name + "'");
    }
    return it->second;
}

const std::vector<std::string>* optional_field(const ParsedFields& fields, const std::string& name)
{
    const auto it = fields.find(name);
    if (it == fields.end()) {
        return nullptr;
    }
    return &it->second;
}

size_t required_scalar_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 1) {
        throw std::runtime_error("zkboost_trace_io: scalar field '" + name + "' must contain exactly one token");
    }
    return parse_size_t_token(values[0], name.c_str());
}

size_t optional_scalar_field(const ParsedFields& fields, const std::string& name, size_t default_value)
{
    const std::vector<std::string>* values = optional_field(fields, name);
    if (values == nullptr) {
        return default_value;
    }
    if (values->size() != 1) {
        throw std::runtime_error("zkboost_trace_io: scalar field '" + name + "' must contain exactly one token");
    }
    return parse_size_t_token(values->front(), name.c_str());
}

fixedPointEncoding required_encoding_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 4) {
        throw std::runtime_error("zkboost_trace_io: encoding field '" + name + "' must contain 4 tokens");
    }
    if (values[0] != "offset_binary") {
        throw std::runtime_error("zkboost_trace_io: unsupported encoding kind in '" + name + "'");
    }

    fixedPointEncoding encoding;
    encoding.kind = fixedPointEncodingKind::OffsetBinary;
    encoding.signed_values = (parse_size_t_token(values[1], name.c_str()) != 0);
    encoding.total_bits = parse_size_t_token(values[2], name.c_str());
    encoding.fractional_bits = parse_size_t_token(values[3], name.c_str());
    validate_fixed_point_encoding(encoding, name.c_str());
    return encoding;
}

std::vector<std::int64_t> required_i64vec_field(const ParsedFields& fields,
                                                const std::string& name,
                                                size_t expected_count)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.empty()) {
        throw std::runtime_error("zkboost_trace_io: i64vec field '" + name + "' is empty");
    }
    const size_t stored_count = parse_size_t_token(values[0], name.c_str());
    if (stored_count != expected_count) {
        throw std::runtime_error("zkboost_trace_io: i64vec field '" + name + "' has unexpected element count");
    }
    if (values.size() != stored_count + 1) {
        throw std::runtime_error("zkboost_trace_io: i64vec field '" + name + "' has inconsistent token count");
    }

    std::vector<std::int64_t> out;
    out.reserve(stored_count);
    for (size_t i = 0; i < stored_count; ++i) {
        out.push_back(parse_int64_token(values[i + 1], name.c_str()));
    }
    return out;
}

std::vector<std::int64_t> optional_i64vec_field(const ParsedFields& fields,
                                                const std::string& name,
                                                size_t expected_count)
{
    const std::vector<std::string>* values = optional_field(fields, name);
    if (values == nullptr) {
        return {};
    }
    if (values->empty()) {
        throw std::runtime_error("zkboost_trace_io: i64vec field '" + name + "' is empty");
    }
    const size_t stored_count = parse_size_t_token(values->front(), name.c_str());
    if (stored_count != expected_count) {
        throw std::runtime_error("zkboost_trace_io: i64vec field '" + name + "' has unexpected element count");
    }
    if (values->size() != stored_count + 1) {
        throw std::runtime_error("zkboost_trace_io: i64vec field '" + name + "' has inconsistent token count");
    }

    std::vector<std::int64_t> out;
    out.reserve(stored_count);
    for (size_t i = 0; i < stored_count; ++i) {
        out.push_back(parse_int64_token((*values)[i + 1], name.c_str()));
    }
    return out;
}

ParsedFields parse_fields_from_file(const std::string& path, const std::string& expected_tag)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("zkboost_trace_io: failed to open input file: " + path);
    }

    ParsedFields fields;
    std::string line;
    while (std::getline(in, line)) {
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
            if (!(iss >> tag) || tag != expected_tag) {
                throw std::runtime_error("zkboost_trace_io: unexpected format tag in " + path);
            }
            fields["__format__"] = {tag};
            continue;
        }

        std::string name;
        if (!(iss >> name)) {
            throw std::runtime_error("zkboost_trace_io: malformed line in " + path);
        }
        if (fields.count(name) != 0) {
            throw std::runtime_error("zkboost_trace_io: duplicate field '" + name + "'");
        }

        std::vector<std::string> values;
        std::string value;
        while (iss >> value) {
            values.push_back(value);
        }

        if (kind != "scalar" && kind != "encoding" && kind != "i64vec") {
            throw std::runtime_error("zkboost_trace_io: unknown field kind '" + kind + "'");
        }
        fields[name] = std::move(values);
    }

    if (fields.count("__format__") == 0) {
        throw std::runtime_error("zkboost_trace_io: missing format line in " + path);
    }
    return fields;
}

void write_scalar_field(std::ostream& out, const std::string& name, size_t value)
{
    out << "scalar " << name << " " << value << "\n";
}

void write_encoding_field(std::ostream& out, const std::string& name, const fixedPointEncoding& encoding)
{
    validate_fixed_point_encoding(encoding, name.c_str());
    out << "encoding " << name << " offset_binary " << (encoding.signed_values ? 1 : 0) << " " << encoding.total_bits
        << " " << encoding.fractional_bits << "\n";
}

void write_i64vec_field(std::ostream& out, const std::string& name, const std::vector<std::int64_t>& values)
{
    out << "i64vec " << name << " " << values.size();
    for (std::int64_t value : values) {
        out << " " << value;
    }
    out << "\n";
}

void validate_binary_vector(const std::vector<std::int64_t>& values, const char* name)
{
    for (std::int64_t value : values) {
        if (value != 0 && value != 1) {
            throw std::runtime_error(std::string("zkboost_trace_io: field '") + name + "' must be binary");
        }
    }
}

} // namespace

void validate_zkboost_training_trace(const zkboostTrainingTrace& trace)
{
    if (trace.hyperparameters.round_count == 0) {
        throw std::runtime_error("zkboost_trace_io: round_count must be non-zero");
    }
    if (trace.hyperparameters.tree_count == 0) {
        throw std::runtime_error("zkboost_trace_io: tree_count must be non-zero");
    }
    if (trace.tree_shape.node_count == 0 || trace.tree_shape.leaf_count == 0) {
        throw std::runtime_error("zkboost_trace_io: tree shape must be non-empty");
    }
    if (trace.feature_count == 0 || trace.query_count == 0) {
        throw std::runtime_error("zkboost_trace_io: feature_count and query_count must be non-zero");
    }
    if (trace.hyperparameters.tree_count % trace.hyperparameters.round_count != 0) {
        throw std::runtime_error("zkboost_trace_io: tree_count must be divisible by round_count");
    }

    validate_fixed_point_encoding(trace.fixed_point.score_encoding, "zkboost.score_encoding");
    validate_fixed_point_encoding(trace.fixed_point.probability_encoding, "zkboost.probability_encoding");
    validate_fixed_point_encoding(trace.fixed_point.gradient_encoding, "zkboost.gradient_encoding");
    validate_fixed_point_encoding(trace.fixed_point.hessian_encoding, "zkboost.hessian_encoding");
    validate_fixed_point_encoding(trace.fixed_point.gain_encoding, "zkboost.gain_encoding");
    validate_fixed_point_encoding(trace.fixed_point.leaf_value_encoding, "zkboost.leaf_value_encoding");

    if (trace.labels.size() != trace.query_count) {
        throw std::runtime_error("zkboost_trace_io: labels size mismatch");
    }
    validate_binary_vector(trace.labels, "training.labels");

    if (trace.binned_inputs.size() != trace.query_count * trace.feature_count) {
        throw std::runtime_error("zkboost_trace_io: binned_inputs size mismatch");
    }
    if (!trace.raw_inputs.empty() && trace.raw_inputs.size() != trace.query_count * trace.feature_count) {
        throw std::runtime_error("zkboost_trace_io: raw_inputs size mismatch");
    }
    if (!trace.prebin_feature_mins.empty() && trace.prebin_feature_mins.size() != trace.feature_count) {
        throw std::runtime_error("zkboost_trace_io: prebin_feature_mins size mismatch");
    }
    if (!trace.prebin_feature_maxs.empty() && trace.prebin_feature_maxs.size() != trace.feature_count) {
        throw std::runtime_error("zkboost_trace_io: prebin_feature_maxs size mismatch");
    }
    if (!trace.prebin_deltas.empty() && trace.prebin_deltas.size() != trace.feature_count) {
        throw std::runtime_error("zkboost_trace_io: prebin_deltas size mismatch");
    }
    if (!trace.prebin_edges.empty() &&
        trace.prebin_edges.size() != trace.feature_count * trace.hyperparameters.bin_count) {
        throw std::runtime_error("zkboost_trace_io: prebin_edges size mismatch");
    }
    if (!trace.prebin_bin_ids.empty() &&
        trace.prebin_bin_ids.size() != trace.query_count * trace.feature_count) {
        throw std::runtime_error("zkboost_trace_io: prebin_bin_ids size mismatch");
    }
    const bool has_prebin =
        !trace.raw_inputs.empty() ||
        !trace.prebin_feature_mins.empty() ||
        !trace.prebin_feature_maxs.empty() ||
        !trace.prebin_deltas.empty() ||
        !trace.prebin_edges.empty() ||
        !trace.prebin_bin_ids.empty();
    if (has_prebin) {
        if (trace.raw_inputs.empty() || trace.prebin_feature_mins.empty() || trace.prebin_feature_maxs.empty() ||
            trace.prebin_deltas.empty() || trace.prebin_edges.empty() || trace.prebin_bin_ids.empty()) {
            throw std::runtime_error("zkboost_trace_io: prebin fields must be present together");
        }
        if (trace.prebin_dummy_feature >= trace.feature_count) {
            throw std::runtime_error("zkboost_trace_io: prebin_dummy_feature out of range");
        }
        if (trace.prebin_dummy_bin >= trace.hyperparameters.bin_count) {
            throw std::runtime_error("zkboost_trace_io: prebin_dummy_bin out of range");
        }
    }
    if (trace.root_indices.size() != trace.hyperparameters.tree_count) {
        throw std::runtime_error("zkboost_trace_io: root_indices size mismatch");
    }

    const size_t forest_node_count = trace.hyperparameters.tree_count * trace.tree_shape.node_count;
    if (trace.split_node_mask.size() != forest_node_count || trace.left_child_indices.size() != forest_node_count ||
        trace.right_child_indices.size() != forest_node_count ||
        trace.node_split_features.size() != forest_node_count ||
        trace.node_threshold_bins.size() != forest_node_count || trace.node_dummy_mask.size() != forest_node_count ||
        trace.node_leaf_mask.size() != forest_node_count) {
        throw std::runtime_error("zkboost_trace_io: node lane size mismatch");
    }
    validate_binary_vector(trace.split_node_mask, "training.split_node_mask");
    validate_binary_vector(trace.node_dummy_mask, "training.node_dummy_mask");
    validate_binary_vector(trace.node_leaf_mask, "training.node_leaf_mask");

    if (trace.leaf_weights.size() != trace.hyperparameters.tree_count * trace.tree_shape.leaf_count) {
        throw std::runtime_error("zkboost_trace_io: leaf_weights size mismatch");
    }
    if (trace.leaf_indices.size() != trace.hyperparameters.tree_count * trace.query_count) {
        throw std::runtime_error("zkboost_trace_io: leaf_indices size mismatch");
    }
    if (trace.path_node_indices.size() != trace.hyperparameters.tree_count * trace.query_count * (trace.tree_shape.max_depth + 1)) {
        throw std::runtime_error("zkboost_trace_io: path_node_indices size mismatch");
    }
    if (trace.path_decisions.size() != trace.hyperparameters.tree_count * trace.query_count * trace.tree_shape.max_depth) {
        throw std::runtime_error("zkboost_trace_io: path_decisions size mismatch");
    }
    validate_binary_vector(trace.path_decisions, "training.path_decisions");
    if (trace.round_start_scores.size() != trace.hyperparameters.round_count * trace.query_count) {
        throw std::runtime_error("zkboost_trace_io: round_start_scores size mismatch");
    }
    if (trace.round_probabilities.size() != trace.hyperparameters.round_count * trace.query_count) {
        throw std::runtime_error("zkboost_trace_io: round_probabilities size mismatch");
    }
    if (trace.round_gradients.size() != trace.hyperparameters.round_count * trace.query_count) {
        throw std::runtime_error("zkboost_trace_io: round_gradients size mismatch");
    }
    if (trace.round_hessians.size() != trace.hyperparameters.round_count * trace.query_count) {
        throw std::runtime_error("zkboost_trace_io: round_hessians size mismatch");
    }
    const size_t histogram_lane_count =
        trace.hyperparameters.tree_count * trace.tree_shape.node_count * trace.feature_count *
        trace.hyperparameters.bin_count;
    if (trace.node_hist_counts.size() != histogram_lane_count) {
        throw std::runtime_error("zkboost_trace_io: node_hist_counts size mismatch");
    }
    if (trace.node_hist_gradients.size() != histogram_lane_count) {
        throw std::runtime_error("zkboost_trace_io: node_hist_gradients size mismatch");
    }
    if (trace.node_hist_hessians.size() != histogram_lane_count) {
        throw std::runtime_error("zkboost_trace_io: node_hist_hessians size mismatch");
    }
}

void validate_zkboost_inference_trace(const zkboostInferenceTrace& trace)
{
    if (trace.query_count == 0 || trace.feature_count == 0 || trace.tree_count == 0) {
        throw std::runtime_error("zkboost_trace_io: inference shape must be non-zero");
    }
    if (trace.max_depth == 0) {
        throw std::runtime_error("zkboost_trace_io: inference max_depth must be non-zero");
    }
    if (trace.query_tree_ids.size() != trace.query_count) {
        throw std::runtime_error("zkboost_trace_io: query_tree_ids size mismatch");
    }
    if (trace.binned_inputs.size() != trace.query_count * trace.feature_count) {
        throw std::runtime_error("zkboost_trace_io: inference binned_inputs size mismatch");
    }
    if (trace.leaf_indices.size() != trace.tree_count * trace.query_count) {
        throw std::runtime_error("zkboost_trace_io: inference leaf_indices size mismatch");
    }
    if (trace.path_node_indices.size() != trace.tree_count * trace.query_count * (trace.max_depth + 1)) {
        throw std::runtime_error("zkboost_trace_io: inference path_node_indices size mismatch");
    }
    if (trace.path_decisions.size() != trace.tree_count * trace.query_count * trace.max_depth) {
        throw std::runtime_error("zkboost_trace_io: inference path_decisions size mismatch");
    }
    validate_binary_vector(trace.path_decisions, "inference.path_decisions");
}

zkboostTraceSummary summarize_zkboost_traces(const zkboostTrainingTrace& training,
                                             const zkboostInferenceTrace& inference)
{
    validate_zkboost_training_trace(training);
    validate_zkboost_inference_trace(inference);

    zkboostTraceSummary summary;
    summary.round_count = training.hyperparameters.round_count;
    summary.tree_count = training.hyperparameters.tree_count;
    summary.node_count = training.hyperparameters.tree_count * training.tree_shape.node_count;
    summary.leaf_count = training.hyperparameters.tree_count * training.tree_shape.leaf_count;
    summary.feature_count = training.feature_count;
    summary.bin_count = training.hyperparameters.bin_count;
    summary.training_query_count = training.query_count;
    summary.inference_query_count = inference.query_count;
    return summary;
}

std::string zkboost_trace_summary_description(const zkboostTraceSummary& summary)
{
    std::ostringstream out;
    out << "CertXGB trace summary\n";
    out << "rounds: " << summary.round_count << "\n";
    out << "trees: " << summary.tree_count << "\n";
    out << "total forest nodes: " << summary.node_count << "\n";
    out << "total forest leaves: " << summary.leaf_count << "\n";
    out << "features: " << summary.feature_count << "\n";
    out << "bins: " << summary.bin_count << "\n";
    out << "training queries: " << summary.training_query_count << "\n";
    out << "inference queries: " << summary.inference_query_count << "\n";
    return out.str();
}

std::string zkboost_trace_format_description()
{
    return "Formats:\n"
           "  - zkboost_binary_training_trace_v1\n"
           "  - zkboost_binary_inference_trace_v1\n"
           "These traces currently cover schema, path witnesses, and validation helpers.\n";
}

void write_zkboost_training_trace(const std::string& path, const zkboostTrainingTrace& trace)
{
    validate_zkboost_training_trace(trace);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("zkboost_trace_io: failed to open output file: " + path);
    }

    out << "format " << kTrainingFormatTag << "\n";
    write_scalar_field(out, "training.round_count", trace.hyperparameters.round_count);
    write_scalar_field(out, "training.tree_count", trace.hyperparameters.tree_count);
    write_scalar_field(out, "training.max_depth", trace.hyperparameters.max_depth);
    write_scalar_field(out, "training.bin_count", trace.hyperparameters.bin_count);
    write_i64vec_field(out, "training.learning_rate", {trace.hyperparameters.learning_rate});
    write_i64vec_field(out, "training.l2_regularization", {trace.hyperparameters.l2_regularization});
    write_i64vec_field(out, "training.split_penalty", {trace.hyperparameters.split_penalty});
    write_scalar_field(out, "training.tree_shape.max_depth", trace.tree_shape.max_depth);
    write_scalar_field(out, "training.tree_shape.leaf_count", trace.tree_shape.leaf_count);
    write_scalar_field(out, "training.tree_shape.internal_node_count", trace.tree_shape.internal_node_count);
    write_scalar_field(out, "training.tree_shape.node_count", trace.tree_shape.node_count);
    write_scalar_field(out, "training.feature_count", trace.feature_count);
    write_scalar_field(out, "training.query_count", trace.query_count);
    write_scalar_field(out, "training.input_bit_bound", trace.fixed_point.input_bit_bound);
    write_scalar_field(out, "training.full_upper_bound", trace.fixed_point.full_upper_bound);
    write_scalar_field(out, "training.structure_bit_bound", trace.fixed_point.structure_bit_bound);
    out << "i64vec training.base_score 1 " << trace.base_score << "\n";

    write_encoding_field(out, "training.score_encoding", trace.fixed_point.score_encoding);
    write_encoding_field(out, "training.probability_encoding", trace.fixed_point.probability_encoding);
    write_encoding_field(out, "training.gradient_encoding", trace.fixed_point.gradient_encoding);
    write_encoding_field(out, "training.hessian_encoding", trace.fixed_point.hessian_encoding);
    write_encoding_field(out, "training.gain_encoding", trace.fixed_point.gain_encoding);
    write_encoding_field(out, "training.leaf_value_encoding", trace.fixed_point.leaf_value_encoding);

    write_i64vec_field(out, "training.labels", trace.labels);
    if (!trace.raw_inputs.empty()) {
        write_i64vec_field(out, "training.raw_inputs", trace.raw_inputs);
        write_i64vec_field(out, "training.prebin_feature_mins", trace.prebin_feature_mins);
        write_i64vec_field(out, "training.prebin_feature_maxs", trace.prebin_feature_maxs);
        write_i64vec_field(out, "training.prebin_deltas", trace.prebin_deltas);
        write_i64vec_field(out, "training.prebin_edges", trace.prebin_edges);
        write_i64vec_field(out, "training.prebin_bin_ids", trace.prebin_bin_ids);
        write_scalar_field(out, "training.prebin_dummy_feature", trace.prebin_dummy_feature);
        write_scalar_field(out, "training.prebin_dummy_bin", trace.prebin_dummy_bin);
        write_i64vec_field(out, "training.prebin_dummy_threshold", {trace.prebin_dummy_threshold});
    }
    write_i64vec_field(out, "training.binned_inputs", trace.binned_inputs);
    write_i64vec_field(out, "training.root_indices", trace.root_indices);
    write_i64vec_field(out, "training.split_node_mask", trace.split_node_mask);
    write_i64vec_field(out, "training.left_child_indices", trace.left_child_indices);
    write_i64vec_field(out, "training.right_child_indices", trace.right_child_indices);
    write_i64vec_field(out, "training.node_split_features", trace.node_split_features);
    write_i64vec_field(out, "training.node_threshold_bins", trace.node_threshold_bins);
    write_i64vec_field(out, "training.node_dummy_mask", trace.node_dummy_mask);
    write_i64vec_field(out, "training.node_leaf_mask", trace.node_leaf_mask);
    write_i64vec_field(out, "training.leaf_weights", trace.leaf_weights);
    write_i64vec_field(out, "training.leaf_indices", trace.leaf_indices);
    write_i64vec_field(out, "training.path_node_indices", trace.path_node_indices);
    write_i64vec_field(out, "training.path_decisions", trace.path_decisions);
    write_i64vec_field(out, "training.round_start_scores", trace.round_start_scores);
    write_i64vec_field(out, "training.round_probabilities", trace.round_probabilities);
    write_i64vec_field(out, "training.round_gradients", trace.round_gradients);
    write_i64vec_field(out, "training.round_hessians", trace.round_hessians);
    write_i64vec_field(out, "training.node_hist_counts", trace.node_hist_counts);
    write_i64vec_field(out, "training.node_hist_gradients", trace.node_hist_gradients);
    write_i64vec_field(out, "training.node_hist_hessians", trace.node_hist_hessians);
}

zkboostTrainingTrace read_zkboost_training_trace(const std::string& path)
{
    const ParsedFields fields = parse_fields_from_file(path, kTrainingFormatTag);

    zkboostTrainingTrace trace;
    trace.hyperparameters.round_count = required_scalar_field(fields, "training.round_count");
    trace.hyperparameters.tree_count = required_scalar_field(fields, "training.tree_count");
    trace.hyperparameters.max_depth = required_scalar_field(fields, "training.max_depth");
    trace.hyperparameters.bin_count = required_scalar_field(fields, "training.bin_count");
    trace.hyperparameters.learning_rate = required_i64vec_field(fields, "training.learning_rate", 1).at(0);
    trace.hyperparameters.l2_regularization = required_i64vec_field(fields, "training.l2_regularization", 1).at(0);
    trace.hyperparameters.split_penalty = required_i64vec_field(fields, "training.split_penalty", 1).at(0);
    trace.tree_shape.max_depth = required_scalar_field(fields, "training.tree_shape.max_depth");
    trace.tree_shape.leaf_count = required_scalar_field(fields, "training.tree_shape.leaf_count");
    trace.tree_shape.internal_node_count = required_scalar_field(fields, "training.tree_shape.internal_node_count");
    trace.tree_shape.node_count = required_scalar_field(fields, "training.tree_shape.node_count");
    trace.feature_count = required_scalar_field(fields, "training.feature_count");
    trace.query_count = required_scalar_field(fields, "training.query_count");
    trace.fixed_point.input_bit_bound = required_scalar_field(fields, "training.input_bit_bound");
    trace.fixed_point.full_upper_bound = required_scalar_field(fields, "training.full_upper_bound");
    trace.fixed_point.structure_bit_bound = required_scalar_field(fields, "training.structure_bit_bound");
    trace.base_score = required_i64vec_field(fields, "training.base_score", 1).at(0);

    trace.fixed_point.score_encoding = required_encoding_field(fields, "training.score_encoding");
    trace.fixed_point.probability_encoding = required_encoding_field(fields, "training.probability_encoding");
    trace.fixed_point.gradient_encoding = required_encoding_field(fields, "training.gradient_encoding");
    trace.fixed_point.hessian_encoding = required_encoding_field(fields, "training.hessian_encoding");
    trace.fixed_point.gain_encoding = required_encoding_field(fields, "training.gain_encoding");
    trace.fixed_point.leaf_value_encoding = required_encoding_field(fields, "training.leaf_value_encoding");

    trace.labels = required_i64vec_field(fields, "training.labels", trace.query_count);
    trace.raw_inputs = optional_i64vec_field(fields, "training.raw_inputs", trace.query_count * trace.feature_count);
    trace.prebin_feature_mins = optional_i64vec_field(fields, "training.prebin_feature_mins", trace.feature_count);
    trace.prebin_feature_maxs = optional_i64vec_field(fields, "training.prebin_feature_maxs", trace.feature_count);
    trace.prebin_deltas = optional_i64vec_field(fields, "training.prebin_deltas", trace.feature_count);
    trace.prebin_edges =
        optional_i64vec_field(fields, "training.prebin_edges", trace.query_count == 0 ? 0 : trace.feature_count * trace.hyperparameters.bin_count);
    trace.prebin_bin_ids =
        optional_i64vec_field(fields, "training.prebin_bin_ids", trace.query_count * trace.feature_count);
    trace.prebin_dummy_feature = optional_scalar_field(fields, "training.prebin_dummy_feature", 0);
    trace.prebin_dummy_bin = optional_scalar_field(fields, "training.prebin_dummy_bin", 0);
    const std::vector<std::int64_t> prebin_dummy_threshold =
        optional_i64vec_field(fields, "training.prebin_dummy_threshold", 1);
    if (!prebin_dummy_threshold.empty()) {
        trace.prebin_dummy_threshold = prebin_dummy_threshold.front();
    }
    trace.binned_inputs =
        required_i64vec_field(fields, "training.binned_inputs", trace.query_count * trace.feature_count);
    trace.root_indices =
        required_i64vec_field(fields, "training.root_indices", trace.hyperparameters.tree_count);
    const size_t forest_node_count = trace.hyperparameters.tree_count * trace.tree_shape.node_count;
    trace.split_node_mask =
        required_i64vec_field(fields, "training.split_node_mask", forest_node_count);
    trace.left_child_indices =
        required_i64vec_field(fields, "training.left_child_indices", forest_node_count);
    trace.right_child_indices =
        required_i64vec_field(fields, "training.right_child_indices", forest_node_count);
    trace.node_split_features =
        required_i64vec_field(fields, "training.node_split_features", forest_node_count);
    trace.node_threshold_bins =
        required_i64vec_field(fields, "training.node_threshold_bins", forest_node_count);
    trace.node_dummy_mask =
        required_i64vec_field(fields, "training.node_dummy_mask", forest_node_count);
    trace.node_leaf_mask =
        required_i64vec_field(fields, "training.node_leaf_mask", forest_node_count);
    trace.leaf_weights = required_i64vec_field(
        fields, "training.leaf_weights", trace.hyperparameters.tree_count * trace.tree_shape.leaf_count);
    trace.leaf_indices = required_i64vec_field(
        fields, "training.leaf_indices", trace.hyperparameters.tree_count * trace.query_count);
    trace.path_node_indices = required_i64vec_field(
        fields,
        "training.path_node_indices",
        trace.hyperparameters.tree_count * trace.query_count * (trace.tree_shape.max_depth + 1));
    trace.path_decisions = required_i64vec_field(
        fields,
        "training.path_decisions",
        trace.hyperparameters.tree_count * trace.query_count * trace.tree_shape.max_depth);
    trace.round_start_scores = required_i64vec_field(
        fields, "training.round_start_scores", trace.hyperparameters.round_count * trace.query_count);
    trace.round_probabilities = required_i64vec_field(
        fields, "training.round_probabilities", trace.hyperparameters.round_count * trace.query_count);
    trace.round_gradients = required_i64vec_field(
        fields, "training.round_gradients", trace.hyperparameters.round_count * trace.query_count);
    trace.round_hessians = required_i64vec_field(
        fields, "training.round_hessians", trace.hyperparameters.round_count * trace.query_count);
    const size_t histogram_lane_count =
        trace.hyperparameters.tree_count * trace.tree_shape.node_count * trace.feature_count *
        trace.hyperparameters.bin_count;
    trace.node_hist_counts =
        required_i64vec_field(fields, "training.node_hist_counts", histogram_lane_count);
    trace.node_hist_gradients =
        required_i64vec_field(fields, "training.node_hist_gradients", histogram_lane_count);
    trace.node_hist_hessians =
        required_i64vec_field(fields, "training.node_hist_hessians", histogram_lane_count);

    validate_zkboost_training_trace(trace);
    return trace;
}

void write_zkboost_inference_trace(const std::string& path, const zkboostInferenceTrace& trace)
{
    validate_zkboost_inference_trace(trace);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("zkboost_trace_io: failed to open output file: " + path);
    }

    out << "format " << kInferenceFormatTag << "\n";
    write_scalar_field(out, "inference.query_count", trace.query_count);
    write_scalar_field(out, "inference.feature_count", trace.feature_count);
    write_scalar_field(out, "inference.tree_count", trace.tree_count);
    write_scalar_field(out, "inference.max_depth", trace.max_depth);
    write_i64vec_field(out, "inference.query_tree_ids", trace.query_tree_ids);
    write_i64vec_field(out, "inference.binned_inputs", trace.binned_inputs);
    write_i64vec_field(out, "inference.leaf_indices", trace.leaf_indices);
    write_i64vec_field(out, "inference.path_node_indices", trace.path_node_indices);
    write_i64vec_field(out, "inference.path_decisions", trace.path_decisions);
}

zkboostInferenceTrace read_zkboost_inference_trace(const std::string& path)
{
    const ParsedFields fields = parse_fields_from_file(path, kInferenceFormatTag);

    zkboostInferenceTrace trace;
    trace.query_count = required_scalar_field(fields, "inference.query_count");
    trace.feature_count = required_scalar_field(fields, "inference.feature_count");
    trace.tree_count = required_scalar_field(fields, "inference.tree_count");
    trace.max_depth = required_scalar_field(fields, "inference.max_depth");
    trace.query_tree_ids = required_i64vec_field(fields, "inference.query_tree_ids", trace.query_count);
    trace.binned_inputs = required_i64vec_field(fields, "inference.binned_inputs", trace.query_count * trace.feature_count);
    trace.leaf_indices = required_i64vec_field(fields, "inference.leaf_indices", trace.tree_count * trace.query_count);
    trace.path_node_indices = required_i64vec_field(
        fields,
        "inference.path_node_indices",
        trace.tree_count * trace.query_count * (trace.max_depth + 1));
    trace.path_decisions = required_i64vec_field(
        fields,
        "inference.path_decisions",
        trace.tree_count * trace.query_count * trace.max_depth);

    validate_zkboost_inference_trace(trace);
    return trace;
}
