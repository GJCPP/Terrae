#include "forest_training_trace_io.hpp"

#include "fixed_point.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kSemanticTraceFormatTag = "forest_training_semantic_trace_case_v1";
constexpr const char* kTrainingOnlyFormatTag = "forest_training_semantic_trace_v1";
constexpr const char* kInferenceOnlyFormatTag = "forest_inference_query_trace_v1";

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

const std::vector<std::string>& required_field(const ParsedFields& fields, const std::string& name)
{
    const auto it = fields.find(name);
    if (it == fields.end()) {
        throw std::runtime_error("forest_training_trace_io: missing field '" + name + "'");
    }
    return it->second;
}

size_t required_scalar_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 1) {
        throw std::runtime_error("forest_training_trace_io: scalar field '" + name + "' must contain exactly one token");
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
        throw std::runtime_error("forest_training_trace_io: scalar field '" + name + "' must contain exactly one token");
    }
    return parse_size_t_token(it->second[0], name.c_str());
}

Fr required_fr_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 1) {
        throw std::runtime_error("forest_training_trace_io: fr field '" + name + "' must contain exactly one token");
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
        throw std::runtime_error("forest_training_trace_io: fr field '" + name + "' must contain exactly one token");
    }
    return parse_fr_token(it->second[0], name.c_str());
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

fixedPointEncoding required_encoding_field(const ParsedFields& fields, const std::string& name)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.size() != 4) {
        throw std::runtime_error("forest_training_trace_io: encoding field '" + name + "' must contain 4 tokens");
    }
    if (values[0] != "offset_binary") {
        throw std::runtime_error("forest_training_trace_io: unsupported encoding kind in '" + name + "'");
    }

    fixedPointEncoding encoding;
    encoding.kind = fixedPointEncodingKind::OffsetBinary;
    encoding.signed_values = (parse_size_t_token(values[1], name.c_str()) != 0);
    encoding.total_bits = parse_size_t_token(values[2], name.c_str());
    encoding.fractional_bits = parse_size_t_token(values[3], name.c_str());
    validate_fixed_point_encoding(encoding, name.c_str());
    return encoding;
}

std::vector<std::int64_t> required_i64vec_field(const ParsedFields& fields, const std::string& name, size_t expected_count)
{
    const std::vector<std::string>& values = required_field(fields, name);
    if (values.empty()) {
        throw std::runtime_error("forest_training_trace_io: i64vec field '" + name + "' is empty");
    }
    const size_t stored_count = parse_size_t_token(values[0], name.c_str());
    if (stored_count != expected_count) {
        throw std::runtime_error("forest_training_trace_io: i64vec field '" + name + "' has unexpected element count");
    }
    if (values.size() != stored_count + 1) {
        throw std::runtime_error("forest_training_trace_io: i64vec field '" + name + "' has inconsistent token count");
    }

    std::vector<std::int64_t> out;
    out.reserve(stored_count);
    for (size_t i = 0; i < stored_count; ++i) {
        out.push_back(parse_int64_token(values[i + 1], name.c_str()));
    }
    return out;
}

ParsedFields parse_fields_from_file(const std::string& path, const std::string& expected_tag)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("forest_training_trace_io: failed to open input file: " + path);
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
                throw std::runtime_error("forest_training_trace_io: unexpected format tag in " + path);
            }
            fields["__format__"] = {tag};
            continue;
        }

        std::string name;
        if (!(iss >> name)) {
            throw std::runtime_error("forest_training_trace_io: malformed line in " + path);
        }
        if (fields.count(name) != 0) {
            throw std::runtime_error("forest_training_trace_io: duplicate field '" + name + "'");
        }

        std::vector<std::string> values;
        std::string value;
        while (iss >> value) {
            values.push_back(value);
        }

        if (kind != "scalar" && kind != "fr" && kind != "encoding" && kind != "i64vec") {
            throw std::runtime_error("forest_training_trace_io: unknown field kind '" + kind + "'");
        }
        fields[name] = std::move(values);
    }

    if (fields.count("__format__") == 0) {
        throw std::runtime_error("forest_training_trace_io: missing format line in " + path);
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

void write_i64vec_field(std::ostream& out, const std::string& name, const std::vector<std::int64_t>& values)
{
    out << "i64vec " << name << " " << values.size();
    for (std::int64_t value : values) {
        out << " " << value;
    }
    out << "\n";
}

void write_training_trace_fields(std::ostream& out, const forestTrainingSemanticTrace& trace)
{
    write_scalar_field(out, "training.tree_count", trace.tree_count);
    write_scalar_field(out, "training.node_count", trace.node_count);
    write_scalar_field(out, "training.feature_count", trace.feature_count);
    write_scalar_field(out, "training.bin_count", trace.bin_count);
    write_scalar_field(out, "training.iteration_count", trace.iteration_count);
    write_scalar_field(out, "training.class_count", trace.class_count);
    write_scalar_field(out, "training.query_count", trace.query_count);
    write_scalar_field(out, "training.input_bit_bound", trace.input_bit_bound);
    write_scalar_field(out, "training.full_upper_bound", trace.full_upper_bound);
    write_scalar_field(out, "training.structure_bit_bound", trace.structure_bit_bound);
    write_scalar_field(out, "training.hessian_bit_bound", trace.hessian_bit_bound);
    write_scalar_field(out, "training.gain_denominator_bit_bound", trace.gain_denominator_bit_bound);
    write_scalar_field(out, "training.leaf_denominator_bit_bound", trace.leaf_denominator_bit_bound);
    write_scalar_field(out, "training.gain_gap_bit_bound", trace.gain_gap_bit_bound);
    write_scalar_field(out, "training.loss_mode", forest_training_loss_code(trace.loss_mode));
    write_scalar_field(out, "training.l2_regularization", trace.l2_regularization);
    write_scalar_field(out, "training.learning_rate_numerator", trace.learning_rate_numerator);
    write_scalar_field(out, "training.learning_rate_denominator", trace.learning_rate_denominator);
    write_fr_field(out, "training.split_penalty", trace.split_penalty);
    write_fr_field(out, "training.split_gain_threshold", trace.split_gain_threshold);
    write_scalar_field(out, "training.max_depth", trace.max_depth);

    write_encoding_field(out, "training.gradient_encoding", trace.gradient_encoding);
    write_encoding_field(out, "training.gain_numerator_encoding", trace.gain_numerator_encoding);
    write_encoding_field(out, "training.quantized_gain_encoding", trace.quantized_gain_encoding);
    write_encoding_field(out, "training.leaf_value_encoding", trace.leaf_value_encoding);
    write_encoding_field(out, "training.score_encoding", trace.score_encoding);
    write_encoding_field(out, "training.softmax_encoding", trace.softmax_encoding);

    write_i64vec_field(out, "training.root_indices", trace.root_indices);
    write_i64vec_field(out, "training.split_node_mask", trace.split_node_mask);
    write_i64vec_field(out, "training.left_child_indices", trace.left_child_indices);
    write_i64vec_field(out, "training.right_child_indices", trace.right_child_indices);
    write_i64vec_field(out, "training.node_tree_ids", trace.node_tree_ids);
    write_i64vec_field(out, "training.feature_selector", trace.feature_selector);
    write_i64vec_field(out, "training.thresholds", trace.thresholds);
    write_i64vec_field(out, "training.node_leaf_mask", trace.node_leaf_mask);
    write_i64vec_field(out, "training.node_leaf_values", trace.node_leaf_values);
    write_i64vec_field(out, "training.training_inputs", trace.training_inputs);
    write_i64vec_field(out, "training.base_labels", trace.base_labels);
    write_i64vec_field(out, "training.leaf_indices", trace.leaf_indices);
    write_i64vec_field(out, "training.candidate_mask", trace.candidate_mask);
    write_i64vec_field(out, "training.selected_candidate_indices", trace.selected_candidate_indices);
}

forestTrainingSemanticTrace read_training_trace_fields(const ParsedFields& fields)
{
    forestTrainingSemanticTrace trace;

    trace.tree_count = required_scalar_field(fields, "training.tree_count");
    trace.node_count = required_scalar_field(fields, "training.node_count");
    trace.feature_count = required_scalar_field(fields, "training.feature_count");
    trace.bin_count = required_scalar_field(fields, "training.bin_count");
    trace.iteration_count = required_scalar_field(fields, "training.iteration_count");
    trace.class_count = required_scalar_field(fields, "training.class_count");
    trace.query_count = required_scalar_field(fields, "training.query_count");
    trace.input_bit_bound = required_scalar_field(fields, "training.input_bit_bound");
    trace.full_upper_bound = required_scalar_field(fields, "training.full_upper_bound");
    trace.structure_bit_bound = required_scalar_field(fields, "training.structure_bit_bound");
    trace.hessian_bit_bound = required_scalar_field(fields, "training.hessian_bit_bound");
    trace.gain_denominator_bit_bound = required_scalar_field(fields, "training.gain_denominator_bit_bound");
    trace.leaf_denominator_bit_bound = required_scalar_field(fields, "training.leaf_denominator_bit_bound");
    trace.gain_gap_bit_bound = required_scalar_field(fields, "training.gain_gap_bit_bound");
    // Backward compatibility: traces written before loss_mode existed used the
    // second-order XGBoost objective.
    trace.loss_mode = forest_training_loss_from_code(optional_scalar_field(
        fields, "training.loss_mode", forest_training_loss_code(forestTrainingLoss::SecondOrderXgboost)));
    trace.l2_regularization = required_scalar_field(fields, "training.l2_regularization");
    trace.learning_rate_numerator = optional_scalar_field(fields, "training.learning_rate_numerator", 1);
    trace.learning_rate_denominator = optional_scalar_field(fields, "training.learning_rate_denominator", 1);
    trace.split_penalty = required_fr_field(fields, "training.split_penalty");
    trace.split_gain_threshold = optional_fr_field(fields, "training.split_gain_threshold", Fr(0));
    trace.max_depth = optional_scalar_field(
        fields, "training.max_depth", infer_max_depth_from_shape(trace.tree_count, trace.node_count));

    trace.gradient_encoding = required_encoding_field(fields, "training.gradient_encoding");
    trace.gain_numerator_encoding = required_encoding_field(fields, "training.gain_numerator_encoding");
    trace.quantized_gain_encoding = required_encoding_field(fields, "training.quantized_gain_encoding");
    trace.leaf_value_encoding = required_encoding_field(fields, "training.leaf_value_encoding");
    trace.score_encoding = required_encoding_field(fields, "training.score_encoding");
    trace.softmax_encoding = required_encoding_field(fields, "training.softmax_encoding");

    const size_t node_feature_count = trace.node_count * trace.feature_count;
    const size_t query_row_count = trace.tree_count * trace.query_count;
    const size_t training_input_count = trace.query_count * trace.feature_count;
    const size_t base_label_count = trace.class_count * trace.query_count;
    const size_t candidate_count = trace.node_count * trace.feature_count * trace.bin_count;

    trace.root_indices = required_i64vec_field(fields, "training.root_indices", trace.tree_count);
    trace.split_node_mask = required_i64vec_field(fields, "training.split_node_mask", trace.node_count);
    trace.left_child_indices = required_i64vec_field(fields, "training.left_child_indices", trace.node_count);
    trace.right_child_indices = required_i64vec_field(fields, "training.right_child_indices", trace.node_count);
    trace.node_tree_ids = required_i64vec_field(fields, "training.node_tree_ids", trace.node_count);
    trace.feature_selector = required_i64vec_field(fields, "training.feature_selector", node_feature_count);
    trace.thresholds = required_i64vec_field(fields, "training.thresholds", trace.node_count);
    trace.node_leaf_mask = required_i64vec_field(fields, "training.node_leaf_mask", trace.node_count);
    trace.node_leaf_values = required_i64vec_field(fields, "training.node_leaf_values", trace.node_count);
    trace.training_inputs = required_i64vec_field(fields, "training.training_inputs", training_input_count);
    trace.base_labels = required_i64vec_field(fields, "training.base_labels", base_label_count);
    trace.leaf_indices = required_i64vec_field(fields, "training.leaf_indices", query_row_count);
    trace.candidate_mask = required_i64vec_field(fields, "training.candidate_mask", candidate_count);
    trace.selected_candidate_indices =
        required_i64vec_field(fields, "training.selected_candidate_indices", trace.node_count);

    return trace;
}

void write_inference_query_fields(std::ostream& out, const forestSemanticInferenceQueryBatch& queries)
{
    write_scalar_field(out, "inference.query_count", queries.query_count);
    write_scalar_field(out, "inference.feature_count", queries.feature_count);
    write_i64vec_field(out, "inference.query_tree_ids", queries.query_tree_ids);
    write_i64vec_field(out, "inference.lifted_inputs", queries.lifted_inputs);
    if (!queries.sample_indices.empty()) {
        write_i64vec_field(out, "inference.sample_indices", queries.sample_indices);
    }
}

forestSemanticInferenceQueryBatch read_inference_query_fields(const ParsedFields& fields)
{
    forestSemanticInferenceQueryBatch out;
    out.query_count = required_scalar_field(fields, "inference.query_count");
    out.feature_count = required_scalar_field(fields, "inference.feature_count");
    out.query_tree_ids = required_i64vec_field(fields, "inference.query_tree_ids", out.query_count);
    out.lifted_inputs = required_i64vec_field(fields, "inference.lifted_inputs", out.query_count * out.feature_count);
    const auto sample_indices_it = fields.find("inference.sample_indices");
    if (sample_indices_it != fields.end()) {
        const std::vector<std::string>& values = sample_indices_it->second;
        if (values.empty()) {
            throw std::runtime_error("forest_training_trace_io: i64vec field 'inference.sample_indices' is empty");
        }
        const size_t stored_count = parse_size_t_token(values[0], "inference.sample_indices");
        if (values.size() != stored_count + 1) {
            throw std::runtime_error(
                "forest_training_trace_io: i64vec field 'inference.sample_indices' has inconsistent token count");
        }
        out.sample_indices.reserve(stored_count);
        for (size_t i = 0; i < stored_count; ++i) {
            out.sample_indices.push_back(parse_int64_token(values[i + 1], "inference.sample_indices"));
        }
    }
    return out;
}

size_t count_non_zero_entries(const std::vector<std::int64_t>& values)
{
    size_t count = 0;
    for (std::int64_t value : values) {
        if (value != 0) {
            ++count;
        }
    }
    return count;
}

} // namespace

void write_forest_training_semantic_trace(const std::string& path, const forestTrainingSemanticTrace& trace)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("forest_training_trace_io: failed to open output file: " + path);
    }
    out << "format " << kTrainingOnlyFormatTag << "\n";
    write_training_trace_fields(out, trace);
}

forestTrainingSemanticTrace read_forest_training_semantic_trace(const std::string& path)
{
    const ParsedFields fields = parse_fields_from_file(path, kTrainingOnlyFormatTag);
    return read_training_trace_fields(fields);
}

void write_forest_semantic_inference_query_trace(const std::string& path,
                                                 const forestSemanticInferenceQueryBatch& queries)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("forest_training_trace_io: failed to open output file: " + path);
    }
    out << "format " << kInferenceOnlyFormatTag << "\n";
    write_inference_query_fields(out, queries);
}

forestSemanticInferenceQueryBatch read_forest_semantic_inference_query_trace(const std::string& path)
{
    const ParsedFields fields = parse_fields_from_file(path, kInferenceOnlyFormatTag);
    return read_inference_query_fields(fields);
}

forestTrainingSemanticTraceCaseSummary summarize_forest_training_semantic_trace_case(
    const forestTrainingSemanticTraceCase& trace_case)
{
    forestTrainingSemanticTraceCaseSummary out;
    out.tree_count = trace_case.training_trace.tree_count;
    out.node_count = trace_case.training_trace.node_count;
    out.feature_count = trace_case.training_trace.feature_count;
    out.bin_count = trace_case.training_trace.bin_count;
    out.class_count = trace_case.training_trace.class_count;
    out.training_query_count = trace_case.training_trace.query_count;
    out.inference_query_count = trace_case.inference_queries.query_count;
    out.trace_leaf_assignment_count =
        trace_case.training_trace.tree_count * trace_case.training_trace.query_count;
    out.trace_candidate_count =
        trace_case.training_trace.node_count * trace_case.training_trace.feature_count * trace_case.training_trace.bin_count;
    out.active_split_node_count = count_non_zero_entries(trace_case.training_trace.split_node_mask);
    out.active_candidate_count = count_non_zero_entries(trace_case.training_trace.candidate_mask);
    out.leaf_node_count = count_non_zero_entries(trace_case.training_trace.node_leaf_mask);
    out.exported_leaf_row_count = out.leaf_node_count;
    out.derived_training_query_row_count = trace_case.training_trace.tree_count * trace_case.training_trace.query_count;
    out.derived_bucket_address_count =
        out.derived_training_query_row_count * trace_case.training_trace.feature_count;
    out.derived_boosting_entry_count =
        trace_case.training_trace.iteration_count * trace_case.training_trace.class_count *
        trace_case.training_trace.query_count;
    return out;
}

void validate_forest_training_semantic_trace_case(const forestTrainingSemanticTraceCase& trace_case)
{
    (void)assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
}

std::string forest_training_semantic_trace_case_format_description()
{
    std::ostringstream out;
    out << "format " << kSemanticTraceFormatTag << "\n";
    out << "# scalar <name> <size_t>\n";
    out << "# fr <name> <base10 field element>\n";
    out << "# encoding <name> offset_binary <signed:0|1> <total_bits> <fractional_bits>\n";
    out << "# i64vec <name> <count> <v0> <v1> ...\n";
    out << "#\n";
    out << "# This semantic-trace format stores higher-level committed-forest training inputs:\n";
    out << "# - dataset rows once rather than pre-expanded across trees\n";
    out << "# - node leaf values at the forest level\n";
    out << "# - tree structure plus split rules rather than explicit hyper-box encodings\n";
    out << "# - inferred leaf assignments over (tree, sample)\n";
    out << "# - candidate masks / selected candidates\n";
    out << "# The repository's adapter then derives the current explicit training.* witnesses,\n";
    out << "# inference query batch, and case format from this file.\n";
    return out.str();
}

std::string forest_training_semantic_trace_contract_description()
{
    std::ostringstream out;
    out << "Required semantic-trace fields\n";
    out << "- shape and encoding parameters: tree/node/feature/bin/iteration/class/query counts,\n";
    out << "  input/range bit bounds, public loss mode, regularization, split penalty, and fixed-point encodings.\n";
    out << "  loss_mode=0 selects first-order RSS (default for new traces); loss_mode=1 selects legacy XGBoost.\n";
    out << "- committed forest structure fields: root indices, split-node mask, child indices,\n";
    out << "  node tree ids, feature selectors, thresholds, and node leaf mask.\n";
    out << "- committed forest value fields: node leaf values.\n";
    out << "- dataset-level fields: one query_count x feature_count training-input matrix,\n";
    out << "  one class_count x query_count base-label matrix, and one inference query batch.\n";
    out << "- inference query batch fields: query_tree_ids, lifted_inputs, and optional\n";
    out << "  sample_indices for standalone-subset inference benchmarking.\n";
    out << "- training trace fields: inferred leaf indices over (tree, sample), candidate mask,\n";
    out << "  and selected candidate indices.\n";
    out << "\n";
    out << "Derived internally by the repository\n";
    out << "- expanded training query ids and lifted dataset rows.\n";
    out << "- node boxes, child boxes, and leaf bounds reconstructed from tree structure and split rules.\n";
    out << "- per-query tree scores sourced from committed leaf rows.\n";
    out << "- flattened histogram bucket addresses.\n";
    out << "- count / pseudo-residual / diagonal-Hessian channels.\n";
    out << "- leaf and full histograms, split statistics, exact gains, quantized gains,\n";
    out << "  tie-break witnesses, and selected feature/bin outputs.\n";
    out << "- leaf gradient / Hessian totals, encoded leaf values, and leaf remainders.\n";
    out << "- cumulative scores, shifted softmax witnesses, pseudo-residuals, and diagonal Hessians.\n";
    out << "\n";
    out << "Current semantic responsibility boundary\n";
    out << "- external training code must decide the committed forest, leaf assignments, and split choices.\n";
    out << "- this repository derives the current witness-side proof tensors from that semantic layer.\n";
    return out.str();
}

std::string forest_training_semantic_trace_case_summary_description(
    const forestTrainingSemanticTraceCaseSummary& summary)
{
    std::ostringstream out;
    out << "Semantic trace summary\n";
    out << "  trees: " << summary.tree_count << "\n";
    out << "  nodes: " << summary.node_count << "\n";
    out << "  features: " << summary.feature_count << "\n";
    out << "  bins: " << summary.bin_count << "\n";
    out << "  classes: " << summary.class_count << "\n";
    out << "  training queries: " << summary.training_query_count << "\n";
    out << "  inference queries: " << summary.inference_query_count << "\n";
    out << "  trace leaf assignments: " << summary.trace_leaf_assignment_count << "\n";
    out << "  active split nodes: " << summary.active_split_node_count << "\n";
    out << "  leaf nodes: " << summary.leaf_node_count << "\n";
    out << "  candidate slots: " << summary.trace_candidate_count << "\n";
    out << "  active candidates: " << summary.active_candidate_count << "\n";
    out << "Derived workflow summary\n";
    out << "  training query rows: " << summary.derived_training_query_row_count << "\n";
    out << "  bucket addresses: " << summary.derived_bucket_address_count << "\n";
    out << "  boosting entries: " << summary.derived_boosting_entry_count << "\n";
    out << "  exported leaf rows: " << summary.exported_leaf_row_count << "\n";
    out << "  inference proofs: 1\n";
    return out.str();
}

void write_forest_training_semantic_trace_case(const std::string& path,
                                               const forestTrainingSemanticTraceCase& trace_case)
{
    validate_forest_training_semantic_trace_case(trace_case);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("forest_training_trace_io: failed to open output file: " + path);
    }

    out << "format " << kSemanticTraceFormatTag << "\n";
    write_training_trace_fields(out, trace_case.training_trace);
    write_inference_query_fields(out, trace_case.inference_queries);
}

forestTrainingSemanticTraceCase read_forest_training_semantic_trace_case(const std::string& path)
{
    const ParsedFields fields = parse_fields_from_file(path, kSemanticTraceFormatTag);

    forestTrainingSemanticTraceCase out;
    out.training_trace = read_training_trace_fields(fields);
    out.inference_queries = read_inference_query_fields(fields);
    validate_forest_training_semantic_trace_case(out);
    return out;
}

forestTrainingSemanticTraceCase read_forest_training_semantic_trace_case(const std::string& training_trace_path,
                                                                         const std::string& inference_trace_path)
{
    forestTrainingSemanticTraceCase out;
    out.training_trace = read_forest_training_semantic_trace(training_trace_path);
    out.inference_queries = read_forest_semantic_inference_query_trace(inference_trace_path);
    validate_forest_training_semantic_trace_case(out);
    return out;
}

forestTrainingThenInferenceInputs read_forest_training_then_inference_inputs_from_trace_case(const std::string& path)
{
    const forestTrainingSemanticTraceCase trace_case = read_forest_training_semantic_trace_case(path);
    return assemble_forest_training_then_inference_inputs_from_trace(
        trace_case.training_trace, trace_case.inference_queries);
}

forestTrainingThenInferenceInputs read_forest_training_then_inference_inputs_from_split_traces(
    const std::string& training_trace_path,
    const std::string& inference_trace_path)
{
    const forestTrainingSemanticTraceCase trace_case =
        read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path);
    return assemble_forest_training_then_inference_inputs_from_trace(
        trace_case.training_trace, trace_case.inference_queries);
}
