#include "bls12_381_wrapper.hpp"
#include "exp_check.hpp"
#include "forest_training_demo_data.hpp"
#include "forest_training_then_inference_io.hpp"
#include "forest_training_trace_io.hpp"
#include "forest_well_formedness_workflow.hpp"
#include "less_check.hpp"
#include "matrix_sum.hpp"
#include "test_kzg_srs.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

using namespace mcl::bn;

namespace {

std::vector<size_t> training_less_check_bit_bounds(const forestTrainingProofPublicInputs& inputs)
{
    const size_t softmax_denominator_bit_bound =
        std::max(inputs.hessian_bit_bound + 1, inputs.softmax_encoding.fractional_bits + 1);
    std::vector<size_t> bounds;
    bounds.reserve(32);
    for (size_t bits = 1; bits <= less_check_limb_bits(); ++bits) {
        bounds.push_back(bits);
    }
    const std::vector<size_t> semantic_bounds = {
        inputs.input_bit_bound,
        inputs.full_upper_bound,
        inputs.structure_bit_bound,
        fixed_point_encoded_bit_bound(inputs.gradient_encoding),
        fixed_point_encoded_bit_bound(inputs.gain_numerator_encoding),
        fixed_point_encoded_bit_bound(inputs.quantized_gain_encoding),
        fixed_point_encoded_bit_bound(inputs.leaf_value_encoding),
        fixed_point_encoded_bit_bound(inputs.score_encoding),
        fixed_point_encoded_bit_bound(inputs.softmax_encoding),
        inputs.hessian_bit_bound,
        inputs.hessian_bit_bound + inputs.softmax_encoding.fractional_bits + 1,
        inputs.gain_denominator_bit_bound,
        inputs.leaf_denominator_bit_bound,
        inputs.gain_gap_bit_bound,
        softmax_denominator_bit_bound,
        log2N(inputs.feature_count),
        log2N(inputs.bin_count),
        log2N(inputs.node_count),
    };
    bounds.insert(bounds.end(), semantic_bounds.begin(), semantic_bounds.end());
    return bounds;
}

std::vector<size_t> cube_block_rows(size_t outer_count, size_t middle_count, bool end_rows)
{
    std::vector<size_t> rows;
    rows.reserve(outer_count);
    for (size_t outer = 0; outer < outer_count; ++outer) {
        rows.push_back(outer * middle_count + (end_rows ? middle_count - 1 : 0));
    }
    return rows;
}

void preprocess_table_lookup_shape(kzgProtocol& kzg, size_t row_count, size_t column_count)
{
    if (row_count == 0 || column_count == 0 || !isPowerOfTwo(row_count) || !isPowerOfTwo(column_count)) {
        return;
    }
    preprocess_table_lookup_row_labels(kzg, row_count, column_count);
}

void preprocess_training_verifier_public_tables(kzgProtocol& kzg, const forestTrainingProofPublicInputs& inputs)
{
    preprocess_less_check_range_tables(kzg, training_less_check_bit_bounds(inputs));
    preprocess_matrix_column_prefix_selectors(kzg, inputs.iteration_count, inputs.class_count * inputs.query_count);
    preprocess_non_positive_exp_check_table(kzg, inputs.score_encoding);

    const size_t cube_row_count = inputs.iteration_count * inputs.class_count;
    preprocess_matrix_row_subset_selector(
        kzg,
        cube_row_count,
        inputs.query_count,
        cube_block_rows(inputs.iteration_count, inputs.class_count, false));
    preprocess_matrix_row_subset_selector(
        kzg,
        cube_row_count,
        inputs.query_count,
        cube_block_rows(inputs.iteration_count, inputs.class_count, true));
    preprocess_table_lookup_row_labels(kzg, cube_row_count, inputs.query_count);

    preprocess_hist_agg_index_table(kzg, inputs.node_count * inputs.feature_count * inputs.bin_count);
    preprocess_hist_agg_index_table(kzg, inputs.node_count);

    const size_t candidate_width = inputs.feature_count * inputs.bin_count;
    preprocess_table_lookup_shape(kzg, inputs.node_count, 1);
    preprocess_table_lookup_shape(kzg, inputs.node_count, inputs.feature_count);
    preprocess_table_lookup_shape(kzg, inputs.node_count, candidate_width);
    preprocess_table_lookup_shape(kzg, inputs.tree_count, 1);
    preprocess_table_lookup_shape(kzg, inputs.tree_count, inputs.feature_count);
    if (inputs.score_encoding.total_bits < std::numeric_limits<size_t>::digits - 1) {
        preprocess_table_lookup_shape(kzg, size_t(1) << inputs.score_encoding.total_bits, 2);
    }
}

void print_usage(const char* argv0)
{
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv0 << " write-demo-case <case-path>\n";
    std::cerr << "  " << argv0 << " write-demo-trace-case <trace-path>\n";
    std::cerr << "  " << argv0 << " write-demo-training-trace <training-trace-path>\n";
    std::cerr << "  " << argv0 << " write-demo-inference-trace <inference-trace-path>\n";
    std::cerr << "  " << argv0 << " describe-trace-format\n";
    std::cerr << "  " << argv0 << " describe-trace-contract\n";
    std::cerr << "  " << argv0 << " describe-formats\n";
    std::cerr << "  " << argv0 << " validate-trace-case <trace-path>\n";
    std::cerr << "  " << argv0 << " validate-training-trace <training-trace-path>\n";
    std::cerr << "  " << argv0 << " validate-inference-trace <inference-trace-path>\n";
    std::cerr << "  " << argv0 << " trace-to-case <trace-path> <case-path>\n";
    std::cerr << "  " << argv0 << " split-traces-to-case <training-trace-path> <inference-trace-path> <case-path>\n";
    std::cerr << "  " << argv0 << " required-srs-from-split-traces <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0 << " soundness-from-split-traces <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0 << " prover <srs-path> <case-path> <bundle-path>\n";
    std::cerr << "  " << argv0 << " verifier <srs-path> <case-path> <bundle-path>\n";
    std::cerr << "  " << argv0 << " prover-from-trace-case <srs-path> <trace-path> <bundle-path>\n";
    std::cerr << "  " << argv0 << " verifier-from-trace-case <srs-path> <trace-path> <bundle-path>\n";
    std::cerr << "  " << argv0 << " prover-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <bundle-path>\n";
    std::cerr << "  " << argv0 << " verifier-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <bundle-path>\n";
    std::cerr << "  " << argv0
              << " prove-training-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <proof-path>\n";
    std::cerr << "  " << argv0
              << " prove-inference-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <proof-path>\n";
    std::cerr << "  " << argv0
              << " prove-classification-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <proof-path>\n";
    std::cerr << "  " << argv0
              << " profile-verifier-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <bundle-path>\n";
    std::cerr << "  " << argv0
              << " profile-training-verifier-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <proof-path>\n";
    std::cerr << "  " << argv0
              << " profile-inference-verifier-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <proof-path>\n";
    std::cerr << "  " << argv0
              << " profile-classification-verifier-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <proof-path>\n";
    std::cerr << "  " << argv0 << " profile-from-split-traces <srs-path> <training-trace-path> <inference-trace-path> <bundle-path>\n";
    std::cerr << "  " << argv0 << " profile-commit-forest-from-split-traces <srs-path> <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0 << " profile-well-formedness-from-split-traces <srs-path> <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0 << " profile-training-from-split-traces <srs-path> <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0 << " profile-inference-from-split-traces <srs-path> <training-trace-path> <inference-trace-path>\n";
}

template <class Fn>
double seconds_for(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    std::forward<Fn>(fn)();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

kzgPublicParameters load_required_kzg_public_parameters(const std::string& srs_path,
                                                        const kzgSrsRequirement& requirement)
{
    return load_kzg_public_parameters(srs_path, requirement.deg1_plus_1, requirement.deg2_plus_1);
}

kzgSrsRequirement max_kzg_srs_requirement(const kzgSrsRequirement& left, const kzgSrsRequirement& right)
{
    kzgSrsRequirement out;
    out.deg1_plus_1 = std::max(left.deg1_plus_1, right.deg1_plus_1);
    out.deg2_plus_1 = std::max(left.deg2_plus_1, right.deg2_plus_1);
    return out;
}

std::string current_executable_path()
{
    std::vector<char> buffer(4096, '\0');
    const ssize_t len = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (len <= 0) {
        throw std::runtime_error("example.training_then_inference_bundle: failed to resolve /proc/self/exe");
    }
    buffer[static_cast<size_t>(len)] = '\0';
    return std::string(buffer.data(), static_cast<size_t>(len));
}

class verifierProfileSummary {
public:
    bool accepted = false;
    bool proof_fully_consumed = false;
    std::map<std::string, double> timings;
    std::map<std::string, std::uint64_t> cost_metrics;
};

verifierProfileSummary parse_verifier_profile_output(const std::string& text)
{
    verifierProfileSummary out;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("  accepted: ", 0) == 0) {
            out.accepted = line.substr(std::string("  accepted: ").size()) == "yes";
            continue;
        }
        if (line.rfind("  proof fully consumed: ", 0) == 0) {
            out.proof_fully_consumed = line.substr(std::string("  proof fully consumed: ").size()) == "yes";
            continue;
        }
        if (line.rfind("timing ", 0) != 0) {
            if (line.rfind("metric cost_", 0) == 0) {
                std::istringstream metric_line(line);
                std::string tag;
                std::string key;
                std::uint64_t value = 0;
                if (metric_line >> tag >> key >> value) {
                    out.cost_metrics.emplace(key.substr(std::string("cost_").size()), value);
                }
            }
            continue;
        }
        std::istringstream timing_line(line);
        std::string tag;
        std::string key;
        double value = 0.0;
        if (!(timing_line >> tag >> key >> value)) {
            continue;
        }
        out.timings.emplace(key, value);
    }
    return out;
}

void merge_verifier_cost_metrics(proof_cost::Report& report, const verifierProfileSummary& verifier)
{
    for (const auto& [name, value] : verifier.cost_metrics) {
        // The child emits every counter both unprefixed and under verifier_*.
        // Keep only the phase-qualified copy: the parent already owns the single
        // unprefixed proof-statement totals from its prover execution.
        if (name.rfind("verifier_", 0) == 0) {
            report.add(name, value);
        }
    }
}

void print_semantic_workload_metrics(const forestTrainingSemanticTraceCase& trace_case)
{
    const forestTrainingSemanticTrace& training = trace_case.training_trace;
    const size_t internal_node_count = static_cast<size_t>(
        std::count(training.split_node_mask.begin(), training.split_node_mask.end(), std::int64_t(1)));
    const size_t leaf_node_count = static_cast<size_t>(
        std::count(training.node_leaf_mask.begin(), training.node_leaf_mask.end(), std::int64_t(1)));
    const size_t candidate_count = static_cast<size_t>(
        std::count(training.candidate_mask.begin(), training.candidate_mask.end(), std::int64_t(1)));
    const size_t inference_sample_count = training.tree_count == 0
                                              ? 0
                                              : trace_case.inference_queries.query_count / training.tree_count;
    std::cout << "metric record_count_padded " << training.query_count << "\n";
    std::cout << "metric class_count " << training.class_count << "\n";
    std::cout << "metric max_depth " << training.max_depth << "\n";
    std::cout << "metric internal_node_count " << internal_node_count << "\n";
    std::cout << "metric semantic_leaf_node_count " << leaf_node_count << "\n";
    std::cout << "metric candidate_count " << candidate_count << "\n";
    std::cout << "metric inference_sample_count " << inference_sample_count << "\n";
    std::cout << "metric checked_path_count " << trace_case.inference_queries.query_count << "\n";
}

void add_terrae_application_cost_metrics(proof_cost::Report& report,
                                         const forestTrainingThenInferenceInputs& inputs,
                                         const kzgSrsRequirement& requirement)
{
    const size_t tree_count = inputs.training_inputs.tree_count;
    const size_t node_count = inputs.training_inputs.node_count;
    const size_t feature_count = inputs.training_inputs.feature_count;
    const size_t logical_fields = 2 * tree_count + 7 * node_count + 3 * node_count * feature_count;
    const size_t padded_fields = 2 * nextPow2(tree_count) + 7 * nextPow2(node_count) +
                                 3 * nextPow2(node_count * feature_count);
    report.add("application_commitment_count", 12);
    report.add("logical_committed_field_elements", logical_fields);
    report.add("padded_committed_field_elements", padded_fields);
    report.add("committed_polynomial_domain_sum", padded_fields);
    report.set_max("max_committed_polynomial_domain",
                   std::max(nextPow2(tree_count), nextPow2(node_count * feature_count)));
    report.add("required_srs_deg1_plus_1", requirement.deg1_plus_1);
    report.add("required_srs_deg2_plus_1", requirement.deg2_plus_1);
}

void write_serialized_proof_artifact(const std::string& path, const ProofArtifact& artifact)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("example.training_then_inference_bundle: failed to open proof output: " + path);
    }
    out.write(reinterpret_cast<const char*>(artifact.proof.bytes.data()),
              static_cast<std::streamsize>(artifact.proof.bytes.size()));
    if (!out) {
        throw std::runtime_error("example.training_then_inference_bundle: failed to write proof: " + path);
    }
}

ProofArtifact read_serialized_proof_artifact(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("example.training_then_inference_bundle: failed to open proof input: " + path);
    }
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) {
        throw std::runtime_error("example.training_then_inference_bundle: failed to read proof: " + path);
    }
    SerializedProof proof;
    proof.bytes.assign(bytes.begin(), bytes.end());
    return make_proof_artifact(std::move(proof));
}

verifierProfileSummary run_fresh_profile_command(const std::string& mode,
                                                 const std::string& srs_path,
                                                 const std::string& training_trace_path,
                                                 const std::string& inference_trace_path,
                                                 const std::string& artifact_path)
{
    char temp_output[] = "/tmp/terrae_profile_verifier_XXXXXX";
    const int temp_fd = ::mkstemp(temp_output);
    if (temp_fd < 0) {
        throw std::runtime_error("example.training_then_inference_bundle: failed to create verifier temp output");
    }
    ::close(temp_fd);

    const pid_t child = ::fork();
    if (child < 0) {
        ::unlink(temp_output);
        throw std::runtime_error("example.training_then_inference_bundle: fork failed");
    }

    if (child == 0) {
        const int output_fd = ::open(temp_output, O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (output_fd < 0) {
            _exit(127);
        }
        if (::dup2(output_fd, STDOUT_FILENO) < 0) {
            _exit(127);
        }
        ::close(output_fd);

        const std::string executable = current_executable_path();
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(executable.c_str()));
        argv.push_back(const_cast<char*>(mode.c_str()));
        argv.push_back(const_cast<char*>(srs_path.c_str()));
        argv.push_back(const_cast<char*>(training_trace_path.c_str()));
        argv.push_back(const_cast<char*>(inference_trace_path.c_str()));
        argv.push_back(const_cast<char*>(artifact_path.c_str()));
        argv.push_back(nullptr);
        ::execv(executable.c_str(), argv.data());
        _exit(127);
    }

    int status = 0;
    if (::waitpid(child, &status, 0) < 0) {
        ::unlink(temp_output);
        throw std::runtime_error("example.training_then_inference_bundle: waitpid failed");
    }
    std::ifstream input(temp_output);
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ::unlink(temp_output);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("example.training_then_inference_bundle: verifier child failed\n" + text);
    }
    return parse_verifier_profile_output(text);
}

std::size_t current_rss_bytes()
{
    std::ifstream statm("/proc/self/statm");
    std::size_t total_pages = 0;
    std::size_t resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) {
        return 0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 0;
    }
    return resident_pages * static_cast<std::size_t>(page_size);
}

struct stageMemorySnapshot {
    std::size_t baseline_rss_bytes = 0;
    std::size_t peak_rss_bytes = 0;
    std::size_t peak_delta_bytes = 0;
    std::size_t samples = 0;
};

class stageMemorySampler {
public:
    stageMemorySampler() = default;
    stageMemorySampler(const stageMemorySampler&) = delete;
    stageMemorySampler& operator=(const stageMemorySampler&) = delete;

    ~stageMemorySampler()
    {
        stop();
    }

    void start()
    {
        baseline_rss_bytes_ = current_rss_bytes();
        peak_rss_bytes_ = baseline_rss_bytes_;
        samples_ = 1;
    }

    stageMemorySnapshot stop()
    {
        sample_once();
        return snapshot();
    }

private:
    void sample_once()
    {
        const std::size_t rss = current_rss_bytes();
        ++samples_;
        if (rss > peak_rss_bytes_) {
            peak_rss_bytes_ = rss;
        }
    }

    stageMemorySnapshot snapshot() const
    {
        const std::size_t peak = peak_rss_bytes_;
        return {
            baseline_rss_bytes_,
            peak,
            peak > baseline_rss_bytes_ ? peak - baseline_rss_bytes_ : 0,
            samples_,
        };
    }

    std::size_t baseline_rss_bytes_ = 0;
    std::size_t peak_rss_bytes_ = 0;
    std::size_t samples_ = 0;
};

void print_stage_memory_metrics(const stageMemorySnapshot& snapshot)
{
    std::cout << "metric proof_stage_rss_baseline_bytes " << snapshot.baseline_rss_bytes << "\n";
    std::cout << "metric proof_stage_peak_rss_bytes " << snapshot.peak_rss_bytes << "\n";
    std::cout << "metric proof_stage_peak_delta_bytes " << snapshot.peak_delta_bytes << "\n";
    std::cout << "metric proof_stage_memory_samples " << snapshot.samples << "\n";
}

int run_write_demo_case(const std::string& case_path)
{
    const forestTrainingSemanticTraceCase trace_case = build_demo_training_trace_case();
    write_forest_training_then_inference_case_from_trace(
        case_path, trace_case.training_trace, trace_case.inference_queries);
    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
    const forestTrainingThenInferenceCaseSummary summary =
        summarize_forest_training_then_inference_inputs(inputs);

    std::cout << "Wrote demo case\n";
    std::cout << "  path: " << case_path << "\n";
    std::cout << "  exported leaf rows: " << summary.exported_leaf_row_count << "\n";
    std::cout << "  inference proof count: 1\n";
    return 0;
}

int run_write_demo_trace_case(const std::string& trace_path)
{
    const forestTrainingSemanticTraceCase trace_case = build_demo_training_trace_case();
    write_forest_training_semantic_trace_case(trace_path, trace_case);
    const forestTrainingSemanticTraceCaseSummary summary =
        summarize_forest_training_semantic_trace_case(trace_case);

    std::cout << "Wrote demo semantic trace\n";
    std::cout << "  path: " << trace_path << "\n";
    std::cout << forest_training_semantic_trace_case_summary_description(summary);
    return 0;
}

int run_write_demo_training_trace(const std::string& trace_path)
{
    const forestTrainingSemanticTrace trace = build_demo_training_trace();
    write_forest_training_semantic_trace(trace_path, trace);

    std::cout << "Wrote demo training semantic trace\n";
    std::cout << "  path: " << trace_path << "\n";
    std::cout << "  trees: " << trace.tree_count << "\n";
    std::cout << "  nodes: " << trace.node_count << "\n";
    std::cout << "  training queries: " << trace.query_count << "\n";
    return 0;
}

int run_write_demo_inference_trace(const std::string& trace_path)
{
    const forestSemanticInferenceQueryBatch queries = build_demo_inference_query_trace();
    write_forest_semantic_inference_query_trace(trace_path, queries);

    std::cout << "Wrote demo inference semantic trace\n";
    std::cout << "  path: " << trace_path << "\n";
    std::cout << "  inference queries: " << queries.query_count << "\n";
    std::cout << "  features: " << queries.feature_count << "\n";
    return 0;
}

int run_describe_formats()
{
    std::cout << "[case]\n" << forest_training_then_inference_case_format_description() << "\n";
    std::cout << "[bundle]\n" << forest_training_then_inference_bundle_format_description() << "\n";
    return 0;
}

int run_describe_trace_format()
{
    std::cout << "[semantic-trace-case]\n" << forest_training_semantic_trace_case_format_description() << "\n";
    std::cout << "[training-trace]\n";
    std::cout << "format forest_training_semantic_trace_v1\n";
    std::cout << "# Training-only semantic trace.\n";
    std::cout << "[inference-trace]\n";
    std::cout << "format forest_inference_query_trace_v1\n";
    std::cout << "# Inference-only semantic trace.\n";
    return 0;
}

int run_describe_trace_contract()
{
    std::cout << "[semantic-trace-contract]\n" << forest_training_semantic_trace_contract_description() << "\n";
    return 0;
}

int run_prover(const std::string& srs_path, const std::string& case_path, const std::string& bundle_path)
{
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    const forestTrainingThenInferenceInputs inputs =
        read_forest_training_then_inference_case(case_path);
    const forestTrainingThenInferenceCaseSummary summary =
        summarize_forest_training_then_inference_inputs(inputs);

    const forestTrainingThenInferenceArtifact artifact = prove_training_then_inference_with_runtime(
        public_params, inputs, "example.training_then_inference_bundle");
    write_forest_training_then_inference_bundle(bundle_path, artifact);

    std::cout << "Training-then-inference example\n";
    std::cout << "  mode: prover\n";
    std::cout << "  case: " << case_path << "\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << "  exported leaf rows: " << summary.exported_leaf_row_count << "\n";
    std::cout << "  inference proof count: 1\n";
    return 0;
}

int run_verifier(const std::string& srs_path, const std::string& case_path, const std::string& bundle_path)
{
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    const forestTrainingThenInferenceInputs inputs =
        read_forest_training_then_inference_case(case_path);
    const forestTrainingThenInferenceArtifact artifact =
        read_forest_training_then_inference_bundle(bundle_path);
    const forestTrainingThenInferenceVerifierBundle verifier_bundle =
        prepare_training_then_inference_verifier_bundle(
            public_params, inputs, artifact, "example.training_then_inference_bundle");
    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, verifier_bundle, "example.training_then_inference_bundle");

    std::cout << "Training-then-inference example\n";
    std::cout << "  mode: verifier\n";
    std::cout << "  case: " << case_path << "\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << "  accepted: " << (result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (result.proof_fully_consumed ? "yes" : "no") << "\n";
    return (result.accepted && result.proof_fully_consumed) ? 0 : 1;
}

int run_validate_trace_case(const std::string& trace_path)
{
    const forestTrainingSemanticTraceCase trace_case =
        read_forest_training_semantic_trace_case(trace_path);
    const forestTrainingSemanticTraceCaseSummary summary =
        summarize_forest_training_semantic_trace_case(trace_case);

    std::cout << "Validated semantic trace\n";
    std::cout << "  path: " << trace_path << "\n";
    std::cout << forest_training_semantic_trace_case_summary_description(summary);
    return 0;
}

int run_validate_training_trace(const std::string& trace_path)
{
    const forestTrainingSemanticTrace trace = read_forest_training_semantic_trace(trace_path);

    std::cout << "Validated training semantic trace\n";
    std::cout << "  path: " << trace_path << "\n";
    std::cout << "  trees: " << trace.tree_count << "\n";
    std::cout << "  nodes: " << trace.node_count << "\n";
    std::cout << "  training queries: " << trace.query_count << "\n";
    return 0;
}

int run_validate_inference_trace(const std::string& trace_path)
{
    const forestSemanticInferenceQueryBatch queries = read_forest_semantic_inference_query_trace(trace_path);

    std::cout << "Validated inference semantic trace\n";
    std::cout << "  path: " << trace_path << "\n";
    std::cout << "  inference queries: " << queries.query_count << "\n";
    std::cout << "  features: " << queries.feature_count << "\n";
    return 0;
}

int run_trace_to_case(const std::string& trace_path, const std::string& case_path)
{
    const forestTrainingSemanticTraceCase trace_case =
        read_forest_training_semantic_trace_case(trace_path);
    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(
            trace_case.training_trace, trace_case.inference_queries);
    const forestTrainingSemanticTraceCaseSummary summary =
        summarize_forest_training_semantic_trace_case(trace_case);

    write_forest_training_then_inference_case(case_path, inputs);

    std::cout << "Converted semantic trace to case\n";
    std::cout << "  trace: " << trace_path << "\n";
    std::cout << "  case: " << case_path << "\n";
    std::cout << forest_training_semantic_trace_case_summary_description(summary);
    return 0;
}

int run_split_traces_to_case(const std::string& training_trace_path,
                             const std::string& inference_trace_path,
                             const std::string& case_path)
{
    const forestTrainingSemanticTraceCase trace_case =
        read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path);
    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(
            trace_case.training_trace, trace_case.inference_queries);
    const forestTrainingSemanticTraceCaseSummary summary =
        summarize_forest_training_semantic_trace_case(trace_case);

    write_forest_training_then_inference_case(case_path, inputs);

    std::cout << "Converted split semantic traces to case\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  case: " << case_path << "\n";
    std::cout << forest_training_semantic_trace_case_summary_description(summary);
    return 0;
}

int run_prover_from_trace_case(const std::string& srs_path, const std::string& trace_path, const std::string& bundle_path)
{
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    const forestTrainingSemanticTraceCase trace_case =
        read_forest_training_semantic_trace_case(trace_path);
    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(
            trace_case.training_trace, trace_case.inference_queries);
    const forestTrainingSemanticTraceCaseSummary summary =
        summarize_forest_training_semantic_trace_case(trace_case);

    const forestTrainingThenInferenceArtifact artifact = prove_training_then_inference_with_runtime(
        public_params, inputs, "example.training_then_inference_bundle");
    write_forest_training_then_inference_bundle(bundle_path, artifact);

    std::cout << "Training-then-inference example\n";
    std::cout << "  mode: prover-from-trace-case\n";
    std::cout << "  trace: " << trace_path << "\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << forest_training_semantic_trace_case_summary_description(summary);
    std::cout << "  produced inference proof: 1\n";
    return 0;
}

int run_required_srs_from_split_traces(const std::string& training_trace_path,
                                       const std::string& inference_trace_path)
{
    forestTrainingThenInferenceInputs inputs =
        read_forest_training_then_inference_inputs_from_split_traces(training_trace_path, inference_trace_path);
    const kzgSrsRequirement training_requirement = required_training_then_inference_srs(inputs);
    const kzgSrsRequirement inference_requirement = required_exported_forest_classification_srs(inputs);
    const kzgSrsRequirement requirement = max_kzg_srs_requirement(training_requirement, inference_requirement);

    std::cout << "Terrae required SRS\n";
    std::cout << "  mode: required-srs-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "metric required_srs_size " << requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg1_plus_1 " << requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg2_plus_1 " << requirement.deg2_plus_1 << "\n";
    std::cout << "metric training_required_srs_deg1_plus_1 " << training_requirement.deg1_plus_1 << "\n";
    std::cout << "metric training_required_srs_deg2_plus_1 " << training_requirement.deg2_plus_1 << "\n";
    std::cout << "metric inference_required_srs_deg1_plus_1 " << inference_requirement.deg1_plus_1 << "\n";
    std::cout << "metric inference_required_srs_deg2_plus_1 " << inference_requirement.deg2_plus_1 << "\n";
    return 0;
}

forestCommittedNodePublicCommitments counting_committed_forest_public_inputs(
    const forestTrainingProofInputs& inputs)
{
    forestCommittedNodePublicCommitments out;
    out.tree_count = inputs.tree_count;
    out.node_count = inputs.node_count;
    out.feature_count = inputs.feature_count;
    return out;
}

void print_exact_soundness_report(const std::string& stage, const proof_cost::Report& report)
{
    static const std::string prefix = "soundness_session_";
    for (const auto& [name, value] : report.counters()) {
        if (name.rfind(prefix, 0) == 0) {
            std::cout << "metric soundness_exact_" << stage << "_" << name.substr(prefix.size())
                      << " " << value << "\n";
        }
    }
}

template <class BuildFn>
proof_cost::Report run_counting_program(const kzgSrsRequirement& requirement, BuildFn&& build)
{
    proof_cost::Report report;
    proof_cost::Scope scope(report);
    ProofRuntime runtime = ProofRuntime::counter(requirement.deg1_plus_1, requirement.deg2_plus_1);
    ProofProgram program(runtime);
    build(runtime, program);
    // Placeholder proof values need not satisfy the equations. Finalization is
    // still exhaustive and emits counters before the count-only KZG no-op.
    (void)program.finalize();
    return report;
}

int run_soundness_from_split_traces(const std::string& training_trace_path,
                                    const std::string& inference_trace_path)
{
    const forestTrainingThenInferenceInputs inputs =
        read_forest_training_then_inference_inputs_from_split_traces(
            training_trace_path, inference_trace_path);
    const forestCommittedNodeTable committed_forest =
        forest_committed_node_table_from_training_inputs(inputs.training_inputs);
    const forestExportedLeafTable exported_leaf_table =
        export_leaf_table_from_committed_forest(committed_forest);
    const forestCommittedNodePublicCommitments public_forest =
        counting_committed_forest_public_inputs(inputs.training_inputs);
    const forestTrainingProofPublicInputs training_public =
        forest_training_proof_public_inputs_from_inputs(public_forest, inputs.training_inputs);
    const forestWellFormednessInputs well_formedness =
        forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs);
    const forestWellFormednessPublicInputs well_formedness_public =
        forest_well_formedness_public_inputs_from_inputs(public_forest, well_formedness);

    const kzgSrsRequirement well_formedness_requirement =
        required_forest_well_formedness_srs(well_formedness_public);
    const proof_cost::Report well_formedness_report =
        run_counting_program(well_formedness_requirement, [&](ProofRuntime& runtime, ProofProgram& program) {
            add_forest_well_formedness_verifier_workflow_to_program(
                runtime, program, well_formedness_public,
                "example.training_then_inference_bundle.well_formedness");
        });

    const kzgSrsRequirement training_requirement =
        required_training_then_inference_srs(inputs);
    const forestTrainingProofInputs verifier_training =
        forest_training_proof_verifier_inputs_from_public_inputs(training_public);
    const proof_cost::Report training_report =
        run_counting_program(training_requirement, [&](ProofRuntime& runtime, ProofProgram& program) {
            const std::string prefix =
                "example.training_then_inference_bundle.training_with_well_formedness";
            add_forest_well_formedness_verifier_workflow_to_program(
                runtime, program, well_formedness_public, prefix + ".well_formedness");
            add_forest_training_proof_workflow_to_program(
                runtime, program, verifier_training, prefix + ".training", &training_public.committed_forest);
        });

    const size_t score_bit_bound =
        fixed_point_encoded_bit_bound(inputs.training_inputs.score_encoding);
    const std::optional<size_t> leaf_table_degree_bound =
        degree_bound_for_domain(exported_leaf_table.row_count * exported_leaf_table.feature_count);
    const forestClassificationPublicTable classification_table =
        build_forest_classification_public_table_metadata_from_exported_forest(
            "example.training_then_inference_bundle.inference_export.inference_table",
            exported_leaf_table,
            inputs.training_inputs.input_bit_bound,
            inputs.training_inputs.iteration_count,
            inputs.training_inputs.class_count,
            score_bit_bound,
            leaf_table_degree_bound);
    forestExportedInferenceBatchShape batch_shape;
    batch_shape.query_count = inputs.inference_queries.query_count;
    batch_shape.feature_count = inputs.inference_queries.feature_count;
    batch_shape.iteration_count = inputs.training_inputs.iteration_count;
    batch_shape.class_count = inputs.training_inputs.class_count;
    batch_shape.sample_count =
        batch_shape.query_count / (batch_shape.iteration_count * batch_shape.class_count);

    const kzgSrsRequirement inference_requirement =
        required_exported_forest_classification_srs(classification_table, batch_shape);
    const proof_cost::Report classification_report =
        run_counting_program(inference_requirement, [&](ProofRuntime&, ProofProgram& program) {
            add_exported_forest_classification_verifier_workflow_to_program(
                program,
                classification_table,
                batch_shape,
                "example.training_then_inference_bundle.inference");
        });

    std::cout << "Terrae exact soundness counter\n";
    std::cout << "  mode: soundness-from-split-traces\n";
    std::cout << "  SRS loaded: no\n";
    std::cout << "  proof generated: no\n";
    std::cout << "metric leaf_node_count " << exported_leaf_table.row_count << "\n";
    print_exact_soundness_report("well_formedness", well_formedness_report);
    print_exact_soundness_report("training", training_report);
    print_exact_soundness_report("classification", classification_report);
    return 0;
}

int run_verifier_from_trace_case(const std::string& srs_path, const std::string& trace_path, const std::string& bundle_path)
{
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    const forestTrainingThenInferenceInputs inputs =
        read_forest_training_then_inference_inputs_from_trace_case(trace_path);
    const forestTrainingThenInferenceArtifact artifact =
        read_forest_training_then_inference_bundle(bundle_path);
    const forestTrainingThenInferenceVerifierBundle verifier_bundle =
        prepare_training_then_inference_verifier_bundle(
            public_params, inputs, artifact, "example.training_then_inference_bundle");
    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, verifier_bundle, "example.training_then_inference_bundle");

    std::cout << "Training-then-inference example\n";
    std::cout << "  mode: verifier-from-trace-case\n";
    std::cout << "  trace: " << trace_path << "\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << "  accepted: " << (result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (result.proof_fully_consumed ? "yes" : "no") << "\n";
    return (result.accepted && result.proof_fully_consumed) ? 0 : 1;
}

int run_prover_from_split_traces(const std::string& srs_path,
                                 const std::string& training_trace_path,
                                 const std::string& inference_trace_path,
                                 const std::string& bundle_path)
{
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    const forestTrainingSemanticTraceCase trace_case =
        read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path);
    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(
            trace_case.training_trace, trace_case.inference_queries);
    const forestTrainingSemanticTraceCaseSummary summary =
        summarize_forest_training_semantic_trace_case(trace_case);

    const forestTrainingThenInferenceArtifact artifact = prove_training_then_inference_with_runtime(
        public_params, inputs, "example.training_then_inference_bundle");
    write_forest_training_then_inference_bundle(bundle_path, artifact);

    std::cout << "Training-then-inference example\n";
    std::cout << "  mode: prover-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << forest_training_semantic_trace_case_summary_description(summary);
    std::cout << "  produced inference proof: 1\n";
    return 0;
}

int run_verifier_from_split_traces(const std::string& srs_path,
                                   const std::string& training_trace_path,
                                   const std::string& inference_trace_path,
                                   const std::string& bundle_path)
{
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    const forestTrainingThenInferenceInputs inputs =
        read_forest_training_then_inference_inputs_from_split_traces(training_trace_path, inference_trace_path);
    const forestTrainingThenInferenceArtifact artifact =
        read_forest_training_then_inference_bundle(bundle_path);
    const forestTrainingThenInferenceVerifierBundle verifier_bundle =
        prepare_training_then_inference_verifier_bundle(
            public_params, inputs, artifact, "example.training_then_inference_bundle");
    const forestTrainingThenInferenceVerificationResult result =
        verify_training_then_inference_with_runtime(
            public_params, verifier_bundle, "example.training_then_inference_bundle");

    std::cout << "Training-then-inference example\n";
    std::cout << "  mode: verifier-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << "  accepted: " << (result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (result.proof_fully_consumed ? "yes" : "no") << "\n";
    return (result.accepted && result.proof_fully_consumed) ? 0 : 1;
}

int run_profile_verifier_from_split_traces(const std::string& srs_path,
                                           const std::string& training_trace_path,
                                           const std::string& inference_trace_path,
                                           const std::string& bundle_path)
{
    kzgPublicParameters public_params;
    forestTrainingThenInferenceInputs inputs;
    forestTrainingThenInferenceArtifact artifact;
    forestTrainingThenInferenceVerifierBundle verifier_bundle;
    forestTrainingThenInferenceVerificationResult verify_result;

    const double read_split_traces_seconds = seconds_for([&] {
        inputs = read_forest_training_then_inference_inputs_from_split_traces(training_trace_path, inference_trace_path);
    });
    const kzgSrsRequirement srs_requirement = required_training_then_inference_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    const double ensure_srs_seconds = seconds_for([&] {
        ensure_kzg_srs_covers(public_params, srs_requirement, "example.training_then_inference_bundle");
    });
    const double read_bundle_seconds =
        seconds_for([&] { artifact = read_forest_training_then_inference_bundle(bundle_path); });
    forestWellFormednessVerifierPreprocessing well_formedness_verifier;
    forestTrainingProofVerifierPreprocessing training_verifier;
    forestExportedInferenceVerifierPreprocessing inference_verifier;
    const double prepare_verifier_bundle_seconds = seconds_for([&] {
        verifier_bundle = prepare_training_then_inference_verifier_bundle(
            public_params, inputs, artifact, "example.training_then_inference_bundle");
        kzgProtocol preprocessing_kzg;
        preprocessing_kzg.setup(public_params);
        preprocess_training_verifier_public_tables(preprocessing_kzg, verifier_bundle.training_public_inputs);

        const forestWellFormednessPublicInputs well_formedness_public_inputs =
            forest_well_formedness_public_inputs_from_training_public_inputs(
                verifier_bundle.training_public_inputs);
        well_formedness_verifier = prepare_forest_well_formedness_verifier_preprocessing(
            public_params,
            verifier_bundle.well_formedness_artifact,
            well_formedness_public_inputs,
            "example.training_then_inference_bundle.well_formedness");
        training_verifier = prepare_forest_training_proof_verifier_preprocessing(
            public_params,
            verifier_bundle.training_artifact,
            verifier_bundle.training_public_inputs,
            "example.training_then_inference_bundle.training");
        inference_verifier = prepare_exported_forest_inference_verifier_preprocessing(
            public_params,
            verifier_bundle.inference_artifact,
            verifier_bundle.inference_table,
            verifier_bundle.inference_batch_shape,
            "example.training_then_inference_bundle.inference");
    });
    const double verify_well_formedness_seconds = seconds_for([&] {
        verify_result.well_formedness_result =
            verify_forest_well_formedness_preprocessed(well_formedness_verifier);
    });
    const double verify_training_seconds = seconds_for([&] {
        verify_result.training_result = verify_forest_training_proof_preprocessed(training_verifier);
    });
    const double verify_inference_seconds = seconds_for([&] {
        verify_result.inference_result = verify_exported_forest_inference_preprocessed(inference_verifier);
    });
    verify_result.accepted = verify_result.well_formedness_result.accepted &&
                             verify_result.training_result.accepted &&
                             verify_result.inference_result.accepted;
    verify_result.proof_fully_consumed =
        verify_result.well_formedness_result.proof_fully_consumed &&
        verify_result.training_result.proof_fully_consumed && verify_result.inference_result.proof_fully_consumed;

    std::cout << "Training-then-inference verifier profile\n";
    std::cout << "  mode: profile-verifier-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << "  accepted: " << (verify_result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (verify_result.proof_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric leaf_node_count " << artifact.exported_leaf_table.row_count << "\n";
    std::cout << "metric query_count " << inputs.inference_queries.query_count << "\n";
    std::cout << "metric well_formedness_proof_bytes "
              << verifier_bundle.well_formedness_artifact.proof.bytes.size() << "\n";
    std::cout << "metric training_proof_bytes " << verifier_bundle.training_artifact.proof.bytes.size() << "\n";
    std::cout << "metric inference_proof_bytes " << verifier_bundle.inference_artifact.proof.bytes.size() << "\n";
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing ensure_srs " << ensure_srs_seconds << "\n";
    std::cout << "timing read_bundle " << read_bundle_seconds << "\n";
    std::cout << "timing prepare_verifier_bundle " << prepare_verifier_bundle_seconds << "\n";
    std::cout << "timing verify_well_formedness_online " << verify_well_formedness_seconds << "\n";
    std::cout << "timing verify_training_online " << verify_training_seconds << "\n";
    std::cout << "timing verify_inference_online " << verify_inference_seconds << "\n";
    std::cout << "timing verify_online_total "
              << (verify_well_formedness_seconds + verify_training_seconds + verify_inference_seconds) << "\n";
    return (verify_result.accepted && verify_result.proof_fully_consumed) ? 0 : 1;
}

int run_profile_training_verifier_from_split_traces(const std::string& srs_path,
                                                    const std::string& training_trace_path,
                                                    const std::string& inference_trace_path,
                                                    const std::string& proof_path)
{
    kzgPublicParameters public_params;
    forestTrainingThenInferenceInputs inputs;
    ProofArtifact training_artifact;
    VerificationResult verify_result;
    proof_cost::Report cost_report;

    const double read_split_traces_seconds = seconds_for([&] {
        inputs = read_forest_training_then_inference_inputs_from_split_traces(training_trace_path, inference_trace_path);
    });
    const kzgSrsRequirement srs_requirement = required_training_then_inference_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    const double ensure_srs_seconds = seconds_for([&] {
        ensure_kzg_srs_covers(public_params, srs_requirement, "example.training_then_inference_bundle");
    });
    const double read_proof_seconds =
        seconds_for([&] { training_artifact = read_serialized_proof_artifact(proof_path); });
    forestCommittedNodePublicCommitments committed_forest_public_inputs;
    forestTrainingProofPublicInputs training_public_inputs;
    forestWellFormednessPublicInputs well_formedness_public_inputs;
    forestTrainingWithWellFormednessVerifierPreprocessing training_verifier;
    const double prepare_verifier_bundle_seconds = seconds_for([&] {
        committed_forest_public_inputs = commit_committed_forest_public_commitments(
            public_params,
            forest_committed_node_table_from_training_inputs(inputs.training_inputs),
            "committed_forest");
        training_public_inputs =
            forest_training_proof_public_inputs_from_inputs(committed_forest_public_inputs, inputs.training_inputs);
        well_formedness_public_inputs = forest_well_formedness_public_inputs_from_inputs(
            committed_forest_public_inputs,
            forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs));

        kzgProtocol preprocessing_kzg;
        preprocessing_kzg.setup(public_params);
        preprocess_training_verifier_public_tables(preprocessing_kzg, training_public_inputs);

        training_verifier = prepare_forest_training_with_well_formedness_verifier_preprocessing(
            public_params,
            training_artifact,
            training_public_inputs,
            well_formedness_public_inputs,
            "example.training_then_inference_bundle.training_with_well_formedness");
    });
    const double verify_training_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope phase_scope("verifier");
        verify_result = verify_forest_training_with_well_formedness_preprocessed(
            training_verifier);
    });

    std::cout << "Training verifier profile\n";
    std::cout << "  mode: profile-training-verifier-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  proof: " << proof_path << "\n";
    std::cout << "  accepted: " << (verify_result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (verify_result.proof_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric training_proof_bytes " << training_artifact.proof.bytes.size() << "\n";
    std::cout << "metric training_with_well_formedness_proof_bytes "
              << training_artifact.proof.bytes.size() << "\n";
    std::cout << "metric training_includes_well_formedness 1\n";
    proof_cost::print_metrics(cost_report);
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing ensure_srs " << ensure_srs_seconds << "\n";
    std::cout << "timing read_proof " << read_proof_seconds << "\n";
    std::cout << "timing prepare_verifier_bundle " << prepare_verifier_bundle_seconds << "\n";
    std::cout << "timing verify_training_online " << verify_training_seconds << "\n";
    std::cout << "timing verify_training_with_well_formedness_online "
              << verify_training_seconds << "\n";
    std::cout << "timing verify_online_total " << verify_training_seconds << "\n";
    return (verify_result.accepted && verify_result.proof_fully_consumed) ? 0 : 1;
}

int run_prove_training_from_split_traces(const std::string& srs_path,
                                         const std::string& training_trace_path,
                                         const std::string& inference_trace_path,
                                         const std::string& proof_path)
{
    kzgPublicParameters public_params;
    forestTrainingSemanticTraceCase trace_case;
    forestTrainingThenInferenceInputs inputs;
    forestExportedInferencePlan plan;
    forestExportedInferencePlanProfile plan_profile;
    forestWellFormednessInputs well_formedness_inputs;
    ProofArtifact training_artifact;
    proof_cost::Report cost_report;

    const double read_split_traces_seconds =
        seconds_for([&] { trace_case = read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path); });
    const double assemble_inputs_seconds = seconds_for([&] {
        inputs = assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
        well_formedness_inputs = forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs);
    });
    const kzgSrsRequirement srs_requirement = required_training_then_inference_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    seconds_for([&] {
        plan = assemble_exported_inference_plan_with_profile(
            public_params,
            inputs,
            plan_profile,
            "example.training_then_inference_bundle.training_export");
    });
    const double prove_training_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope phase_scope("prover");
        training_artifact = prove_forest_training_with_well_formedness_with_runtime(
            public_params,
            inputs.training_inputs,
            well_formedness_inputs,
            plan.committed_forest_public_inputs,
            "example.training_then_inference_bundle.training_with_well_formedness");
    });
    const double write_proof_seconds =
        seconds_for([&] { write_serialized_proof_artifact(proof_path, training_artifact); });

    std::cout << "Training prover profile\n";
    std::cout << "  mode: prove-training-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  proof: " << proof_path << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric training_proof_bytes " << training_artifact.proof.bytes.size() << "\n";
    std::cout << "metric training_with_well_formedness_proof_bytes "
              << training_artifact.proof.bytes.size() << "\n";
    std::cout << "metric training_includes_well_formedness 1\n";
    print_semantic_workload_metrics(trace_case);
    add_terrae_application_cost_metrics(cost_report, inputs, srs_requirement);
    proof_cost::print_metrics(cost_report);
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing assemble_inputs " << assemble_inputs_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing prove_training " << prove_training_seconds << "\n";
    std::cout << "timing prove_training_with_well_formedness "
              << prove_training_seconds << "\n";
    std::cout << "timing write_proof " << write_proof_seconds << "\n";
    return 0;
}

int run_profile_inference_verifier_from_split_traces(const std::string& srs_path,
                                                     const std::string& training_trace_path,
                                                     const std::string& inference_trace_path,
                                                     const std::string& proof_path)
{
    kzgPublicParameters public_params;
    forestTrainingSemanticTraceCase trace_case;
    forestTrainingThenInferenceInputs inputs;
    forestExportedInferencePlan plan;
    forestExportedInferencePlanProfile plan_profile;
    ProofArtifact inference_artifact;
    VerificationResult verify_result;
    proof_cost::Report cost_report;

    const double read_split_traces_seconds = seconds_for([&] {
        trace_case = read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path);
    });
    const double assemble_inputs_seconds = seconds_for([&] {
        inputs = assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
    });
    const kzgSrsRequirement srs_requirement = required_exported_forest_classification_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    const double ensure_srs_seconds = seconds_for([&] {
        ensure_kzg_srs_covers(public_params, srs_requirement, "example.training_then_inference_bundle");
    });
    const double read_proof_seconds =
        seconds_for([&] { inference_artifact = read_serialized_proof_artifact(proof_path); });
    seconds_for([&] {
        plan = assemble_exported_inference_plan_with_profile(
            public_params,
            inputs,
            plan_profile,
            "example.training_then_inference_bundle.inference_export");
    });
    const forestExportedInferenceBatchShape batch_shape = {
        plan.witness_batch.query_count,
        plan.witness_batch.feature_count,
        plan.witness_batch.iteration_count,
        plan.witness_batch.class_count,
        plan.witness_batch.sample_count,
    };
    forestExportedClassificationVerifierPreprocessing verifier;
    const double prepare_verifier_bundle_seconds = seconds_for([&] {
        verifier = prepare_exported_forest_classification_verifier_preprocessing(
            public_params,
            inference_artifact,
            plan.classification_table,
            batch_shape,
            "example.training_then_inference_bundle.inference_classification");
    });
    const double verify_inference_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope phase_scope("verifier");
        verify_result = verify_exported_forest_classification_preprocessed(verifier);
    });

    std::cout << "Inference verifier profile\n";
    std::cout << "  mode: profile-inference-verifier-from-split-traces\n";
    std::cout << "  proof scope: routing+sum+argmax classification\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  proof: " << proof_path << "\n";
    std::cout << "  accepted: " << (verify_result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (verify_result.proof_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric leaf_node_count " << plan.exported_leaf_table.row_count << "\n";
    std::cout << "metric tree_query_row_count " << plan.witness_batch.query_count << "\n";
    std::cout << "metric query_count " << plan.witness_batch.query_count << "\n";
    std::cout << "metric sample_count " << plan.witness_batch.sample_count << "\n";
    std::cout << "metric inference_proof_bytes " << inference_artifact.proof.bytes.size() << "\n";
    std::cout << "metric classification_proof_bytes " << inference_artifact.proof.bytes.size() << "\n";
    proof_cost::print_metrics(cost_report);
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing assemble_inputs " << assemble_inputs_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing ensure_srs " << ensure_srs_seconds << "\n";
    std::cout << "timing read_proof " << read_proof_seconds << "\n";
    std::cout << "timing prepare_verifier_bundle " << prepare_verifier_bundle_seconds << "\n";
    std::cout << "timing verify_inference_online " << verify_inference_seconds << "\n";
    std::cout << "timing verify_classification_online " << verify_inference_seconds << "\n";
    std::cout << "timing verify_online_total " << verify_inference_seconds << "\n";
    return (verify_result.accepted && verify_result.proof_fully_consumed) ? 0 : 1;
}

int run_prove_inference_from_split_traces(const std::string& srs_path,
                                          const std::string& training_trace_path,
                                          const std::string& inference_trace_path,
                                          const std::string& proof_path)
{
    kzgPublicParameters public_params;
    forestTrainingSemanticTraceCase trace_case;
    forestTrainingThenInferenceInputs inputs;
    forestExportedInferencePlan plan;
    forestExportedInferencePlanProfile plan_profile;
    ProofArtifact inference_artifact;

    const double read_split_traces_seconds =
        seconds_for([&] { trace_case = read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path); });
    const double assemble_inputs_seconds = seconds_for([&] {
        inputs = assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
    });
    const kzgSrsRequirement srs_requirement = required_exported_forest_classification_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    seconds_for([&] {
        plan = assemble_exported_inference_plan_with_profile(
            public_params,
            inputs,
            plan_profile,
            "example.training_then_inference_bundle.inference_export");
    });
    const double prove_inference_seconds = seconds_for([&] {
        inference_artifact = prove_exported_forest_classification_with_runtime(
            public_params,
            plan.classification_table,
            plan.witness_batch,
            "example.training_then_inference_bundle.inference_classification");
    });
    const double write_proof_seconds =
        seconds_for([&] { write_serialized_proof_artifact(proof_path, inference_artifact); });

    std::cout << "Inference prover profile\n";
    std::cout << "  mode: prove-inference-from-split-traces\n";
    std::cout << "  proof scope: routing+sum+argmax classification\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  proof: " << proof_path << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric leaf_node_count " << plan.exported_leaf_table.row_count << "\n";
    std::cout << "metric tree_query_row_count " << plan.witness_batch.query_count << "\n";
    std::cout << "metric query_count " << plan.witness_batch.query_count << "\n";
    std::cout << "metric sample_count " << plan.witness_batch.sample_count << "\n";
    std::cout << "metric inference_proof_bytes " << inference_artifact.proof.bytes.size() << "\n";
    std::cout << "metric classification_proof_bytes " << inference_artifact.proof.bytes.size() << "\n";
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing assemble_inputs " << assemble_inputs_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing prove_inference " << prove_inference_seconds << "\n";
    std::cout << "timing prove_classification " << prove_inference_seconds << "\n";
    std::cout << "timing write_proof " << write_proof_seconds << "\n";
    return 0;
}

int run_prove_classification_from_split_traces(const std::string& srs_path,
                                               const std::string& training_trace_path,
                                               const std::string& inference_trace_path,
                                               const std::string& proof_path)
{
    const forestTrainingSemanticTraceCase trace_case =
        read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path);
    const forestTrainingThenInferenceInputs inputs =
        assemble_forest_training_then_inference_inputs_from_trace(
            trace_case.training_trace, trace_case.inference_queries);
    const kzgSrsRequirement requirement = required_exported_forest_classification_srs(inputs);
    const kzgPublicParameters public_params = load_required_kzg_public_parameters(srs_path, requirement);
    forestExportedInferencePlanProfile profile;
    const forestExportedInferencePlan plan = assemble_exported_inference_plan_with_profile(
        public_params, inputs, profile, "example.training_then_inference_bundle.classification_export");
    ProofArtifact artifact;
    proof_cost::Report cost_report;
    const double prove_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope phase_scope("prover");
        artifact = prove_exported_forest_classification_with_runtime(
            public_params, plan.classification_table, plan.witness_batch,
            "example.training_then_inference_bundle.classification");
    });
    const double write_seconds = seconds_for([&] { write_serialized_proof_artifact(proof_path, artifact); });

    std::cout << "Classification prover profile\n";
    std::cout << "  mode: prove-classification-from-split-traces\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric leaf_node_count " << plan.exported_leaf_table.row_count << "\n";
    std::cout << "metric tree_query_row_count " << plan.witness_batch.query_count << "\n";
    std::cout << "metric sample_count " << plan.witness_batch.sample_count << "\n";
    std::cout << "metric classification_proof_bytes " << artifact.proof.bytes.size() << "\n";
    print_semantic_workload_metrics(trace_case);
    add_terrae_application_cost_metrics(cost_report, inputs, requirement);
    proof_cost::print_metrics(cost_report);
    std::cout << "timing prove_classification " << prove_seconds << "\n";
    std::cout << "timing write_proof " << write_seconds << "\n";
    return 0;
}

int run_profile_classification_verifier_from_split_traces(const std::string& srs_path,
                                                          const std::string& training_trace_path,
                                                          const std::string& inference_trace_path,
                                                          const std::string& proof_path)
{
    forestTrainingThenInferenceInputs inputs =
        read_forest_training_then_inference_inputs_from_split_traces(training_trace_path, inference_trace_path);
    const kzgSrsRequirement requirement = required_exported_forest_classification_srs(inputs);
    const kzgPublicParameters public_params = load_required_kzg_public_parameters(srs_path, requirement);
    const ProofArtifact artifact = read_serialized_proof_artifact(proof_path);
    forestExportedInferencePlanProfile profile;
    const forestExportedInferencePlan plan = assemble_exported_inference_plan_with_profile(
        public_params, inputs, profile, "example.training_then_inference_bundle.classification_export");
    const forestExportedInferenceBatchShape shape = {
        plan.witness_batch.query_count,
        plan.witness_batch.feature_count,
        plan.witness_batch.iteration_count,
        plan.witness_batch.class_count,
        plan.witness_batch.sample_count,
    };
    VerificationResult result;
    forestExportedClassificationVerifierPreprocessing verifier;
    proof_cost::Report cost_report;
    const double prepare_verifier_bundle_seconds = seconds_for([&] {
        verifier = prepare_exported_forest_classification_verifier_preprocessing(
            public_params,
            artifact,
            plan.classification_table,
            shape,
            "example.training_then_inference_bundle.classification");
    });
    const double verify_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope phase_scope("verifier");
        result = verify_exported_forest_classification_preprocessed(verifier);
    });

    std::cout << "Classification verifier profile\n";
    std::cout << "  mode: profile-classification-verifier-from-split-traces\n";
    std::cout << "  accepted: " << (result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (result.proof_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric leaf_node_count " << plan.exported_leaf_table.row_count << "\n";
    std::cout << "metric tree_query_row_count " << plan.witness_batch.query_count << "\n";
    std::cout << "metric sample_count " << plan.witness_batch.sample_count << "\n";
    std::cout << "metric classification_proof_bytes " << artifact.proof.bytes.size() << "\n";
    proof_cost::print_metrics(cost_report);
    std::cout << "timing prepare_verifier_bundle " << prepare_verifier_bundle_seconds << "\n";
    std::cout << "timing verify_classification_online " << verify_seconds << "\n";
    std::cout << "timing verify_online_total " << verify_seconds << "\n";
    return (result.accepted && result.proof_fully_consumed) ? 0 : 1;
}

int run_profile_from_split_traces(const std::string& srs_path,
                                  const std::string& training_trace_path,
                                  const std::string& inference_trace_path,
                                  const std::string& bundle_path)
{
    kzgPublicParameters public_params;
    forestTrainingSemanticTraceCase trace_case;
    forestTrainingThenInferenceInputs inputs;
    forestExportedInferencePlan plan;
    forestExportedInferencePlanProfile plan_profile;
    forestTrainingThenInferenceArtifact artifact;
    forestTrainingThenInferenceVerificationResult verify_result;
    forestWellFormednessInputs well_formedness_inputs;

    const double read_split_traces_seconds =
        seconds_for([&] { trace_case = read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path); });
    const double assemble_inputs_seconds = seconds_for([&] {
        inputs = assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
        well_formedness_inputs = forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs);
    });
    const kzgSrsRequirement srs_requirement = required_training_then_inference_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    const double ensure_srs_seconds = seconds_for([&] {
        ensure_kzg_srs_covers(public_params, srs_requirement, "forest_training_then_inference_workflow");
    });
    seconds_for([&] {
        plan = assemble_exported_inference_plan_with_profile(
            public_params,
            inputs,
            plan_profile,
            "example.training_then_inference_bundle.export");
    });
    const double prove_well_formedness_seconds = seconds_for([&] {
        artifact.well_formedness_artifact = prove_forest_well_formedness_with_runtime(
            public_params,
            well_formedness_inputs,
            plan.committed_forest_public_inputs,
            "example.training_then_inference_bundle.well_formedness");
    });
    const double prove_training_seconds = seconds_for([&] {
        artifact.training_artifact = prove_forest_training_proof_with_runtime(
            public_params,
            inputs.training_inputs,
            plan.committed_forest_public_inputs,
            "example.training_then_inference_bundle.training");
    });
    const double prove_inference_seconds = seconds_for([&] {
        artifact.inference_artifact = prove_exported_forest_inference_with_runtime(
            public_params,
            plan.inference_table,
            plan.witness_batch,
            "example.training_then_inference_bundle.inference");
    });
    artifact.exported_leaf_table = plan.exported_leaf_table;
    const double write_bundle_seconds =
        seconds_for([&] { write_forest_training_then_inference_bundle(bundle_path, artifact); });
    const verifierProfileSummary verifier_profile = run_fresh_profile_command(
        "profile-verifier-from-split-traces", srs_path, training_trace_path, inference_trace_path, bundle_path);
    const double verify_well_formedness_seconds =
        verifier_profile.timings.count("verify_well_formedness_online") == 0
            ? 0.0 : verifier_profile.timings.at("verify_well_formedness_online");
    const double verify_training_seconds =
        verifier_profile.timings.count("verify_training_online") == 0 ? 0.0
                                                                      : verifier_profile.timings.at("verify_training_online");
    const double verify_inference_seconds =
        verifier_profile.timings.count("verify_inference_online") == 0 ? 0.0
                                                                       : verifier_profile.timings.at("verify_inference_online");
    verify_result.accepted = verifier_profile.accepted;
    verify_result.proof_fully_consumed = verifier_profile.proof_fully_consumed;
    const bool all_accepted = verify_result.accepted;
    const bool all_fully_consumed = verify_result.proof_fully_consumed;

    std::cout << "Training-then-inference profile\n";
    std::cout << "  mode: profile-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << "  accepted: " << (all_accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (all_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric leaf_node_count " << plan.exported_leaf_table.row_count << "\n";
    std::cout << "metric committed_forest_rows " << plan.committed_forest.node_count << "\n";
    std::cout << "metric committed_forest_columns 12\n";
    std::cout << "metric committed_forest_domain_size "
              << nextPow2(plan.committed_forest.node_count * plan.committed_forest.feature_count) << "\n";
    std::cout << "metric committed_forest_commitment_count 13\n";
    std::cout << "metric well_formedness_proof_bytes "
              << artifact.well_formedness_artifact.proof.bytes.size() << "\n";
    std::cout << "metric training_proof_bytes " << artifact.training_artifact.proof.bytes.size() << "\n";
    std::cout << "metric inference_proof_bytes " << artifact.inference_artifact.proof.bytes.size() << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing assemble_inputs " << assemble_inputs_seconds << "\n";
    std::cout << "timing ensure_srs " << ensure_srs_seconds << "\n";
    std::cout << "timing prove_well_formedness " << prove_well_formedness_seconds << "\n";
    std::cout << "timing prove_training " << prove_training_seconds << "\n";
    std::cout << "timing prove_inference " << prove_inference_seconds << "\n";
    std::cout << "timing write_bundle " << write_bundle_seconds << "\n";
    std::cout << "timing verify_well_formedness_online " << verify_well_formedness_seconds << "\n";
    std::cout << "timing verify_training_online " << verify_training_seconds << "\n";
    std::cout << "timing verify_inference_online " << verify_inference_seconds << "\n";
    std::cout << "timing prove_online_total "
              << (prove_well_formedness_seconds + prove_training_seconds + prove_inference_seconds) << "\n";
    std::cout << "timing verify_online_total "
              << (verify_well_formedness_seconds + verify_training_seconds + verify_inference_seconds) << "\n";
    return (all_accepted && all_fully_consumed) ? 0 : 1;
}

int run_profile_commit_forest_from_split_traces(const std::string& srs_path,
                                                const std::string& training_trace_path,
                                                const std::string& inference_trace_path)
{
    kzgPublicParameters public_params;
    forestTrainingSemanticTraceCase trace_case;
    forestTrainingThenInferenceInputs inputs;
    forestCommittedNodeTable committed_forest;
    forestCommittedNodePublicCommitments public_inputs;

    const double read_split_traces_seconds =
        seconds_for([&] { trace_case = read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path); });
    const double assemble_inputs_seconds = seconds_for([&] {
        inputs = assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
    });
    const kzgSrsRequirement srs_requirement = required_exported_forest_inference_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    const double ensure_srs_seconds = seconds_for([&] {
        ensure_kzg_srs_covers(
            public_params, srs_requirement, "forest_training_then_inference_workflow");
    });
    seconds_for([&] {
        committed_forest = forest_committed_node_table_from_training_inputs(inputs.training_inputs);
    });
    seconds_for([&] {
        public_inputs = commit_committed_forest_public_commitments(
            public_params, committed_forest, "committed_forest");
    });

    std::cout << "Commit forest profile\n";
    std::cout << "  mode: profile-commit-forest-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric committed_forest_rows " << committed_forest.node_count << "\n";
    std::cout << "metric committed_forest_columns 12\n";
    std::cout << "metric committed_forest_domain_size "
              << nextPow2(committed_forest.node_count * committed_forest.feature_count) << "\n";
    std::cout << "metric committed_forest_commitment_count 13\n";
    std::cout << "metric committed_forest_artifact_bytes " << (12 * 48) << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing assemble_inputs " << assemble_inputs_seconds << "\n";
    std::cout << "timing ensure_srs " << ensure_srs_seconds << "\n";
    (void)public_inputs;
    return 0;
}

int run_profile_well_formedness_from_split_traces(const std::string& srs_path,
                                                  const std::string& training_trace_path,
                                                  const std::string& inference_trace_path)
{
    kzgPublicParameters public_params;
    forestTrainingSemanticTraceCase trace_case;
    forestTrainingThenInferenceInputs inputs;
    forestExportedInferencePlan plan;
    forestExportedInferencePlanProfile plan_profile;
    forestWellFormednessInputs well_formedness_inputs;
    ProofArtifact well_formedness_artifact;
    VerificationResult verify_result;
    stageMemorySnapshot proof_stage_memory;
    proof_cost::Report cost_report;

    const double read_split_traces_seconds =
        seconds_for([&] { trace_case = read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path); });
    const double assemble_inputs_seconds = seconds_for([&] {
        inputs = assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
        well_formedness_inputs = forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs);
    });
    const kzgSrsRequirement srs_requirement = required_training_then_inference_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    const double ensure_srs_seconds = seconds_for([&] {
        ensure_kzg_srs_covers(public_params, srs_requirement, "forest_training_then_inference_workflow");
    });
    seconds_for([&] {
        plan = assemble_exported_inference_plan_with_profile(
            public_params,
            inputs,
            plan_profile,
            "example.training_then_inference_bundle.well_formedness_export");
    });

    double prove_well_formedness_seconds = 0.0;
    double verify_well_formedness_seconds = 0.0;
    {
        stageMemorySampler sampler;
        sampler.start();
        prove_well_formedness_seconds = seconds_for([&] {
            proof_cost::Scope cost_scope(cost_report);
            proof_cost::PrefixScope phase_scope("prover_well_formedness");
            well_formedness_artifact = prove_forest_well_formedness_with_runtime(
                public_params,
                well_formedness_inputs,
                plan.committed_forest_public_inputs,
                "example.training_then_inference_bundle.well_formedness_only");
        });
        forestWellFormednessVerifierPreprocessing well_formedness_verifier;
        seconds_for([&] {
            const forestWellFormednessPublicInputs well_formedness_public_inputs =
                forest_well_formedness_public_inputs_from_inputs(
                    plan.committed_forest_public_inputs,
                    well_formedness_inputs);
            well_formedness_verifier = prepare_forest_well_formedness_verifier_preprocessing(
                public_params,
                well_formedness_artifact,
                well_formedness_public_inputs,
                "example.training_then_inference_bundle.well_formedness_only");
        });
        verify_well_formedness_seconds = seconds_for([&] {
            proof_cost::Scope cost_scope(cost_report);
            proof_cost::PrefixScope phase_scope("verifier_well_formedness");
            verify_result =
                verify_forest_well_formedness_preprocessed(well_formedness_verifier);
        });
        proof_stage_memory = sampler.stop();
    }

    std::cout << "Well-formedness proof profile\n";
    std::cout << "  mode: profile-well-formedness-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  accepted: " << (verify_result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (verify_result.proof_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric committed_forest_rows " << plan.committed_forest.node_count << "\n";
    std::cout << "metric committed_forest_columns 12\n";
    std::cout << "metric committed_forest_domain_size "
              << nextPow2(plan.committed_forest.node_count * plan.committed_forest.feature_count) << "\n";
    std::cout << "metric well_formedness_proof_bytes " << well_formedness_artifact.proof.bytes.size() << "\n";
    print_semantic_workload_metrics(trace_case);
    add_terrae_application_cost_metrics(cost_report, inputs, srs_requirement);
    proof_cost::print_metrics(cost_report);
    print_stage_memory_metrics(proof_stage_memory);
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing assemble_inputs " << assemble_inputs_seconds << "\n";
    std::cout << "timing ensure_srs " << ensure_srs_seconds << "\n";
    std::cout << "timing prove_well_formedness " << prove_well_formedness_seconds << "\n";
    std::cout << "timing verify_well_formedness_online " << verify_well_formedness_seconds << "\n";
    return (verify_result.accepted && verify_result.proof_fully_consumed) ? 0 : 1;
}

int run_profile_training_from_split_traces(const std::string& srs_path,
                                           const std::string& training_trace_path,
                                           const std::string& inference_trace_path)
{
    kzgPublicParameters public_params;
    forestTrainingSemanticTraceCase trace_case;
    forestTrainingThenInferenceInputs inputs;
    forestExportedInferencePlan plan;
    forestExportedInferencePlanProfile plan_profile;
    forestWellFormednessInputs well_formedness_inputs;
    ProofArtifact training_artifact;
    VerificationResult training_verify_result;
    stageMemorySnapshot proof_stage_memory;
    proof_cost::Report cost_report;

    const double read_split_traces_seconds =
        seconds_for([&] { trace_case = read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path); });
    const double assemble_inputs_seconds = seconds_for([&] {
        inputs = assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
        well_formedness_inputs = forest_well_formedness_inputs_from_training_inputs(inputs.training_inputs);
    });
    const kzgSrsRequirement srs_requirement = required_training_then_inference_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    const double ensure_srs_seconds = seconds_for([&] {
        ensure_kzg_srs_covers(public_params, srs_requirement, "forest_training_then_inference_workflow");
    });
    seconds_for([&] {
        plan = assemble_exported_inference_plan_with_profile(
            public_params,
            inputs,
            plan_profile,
            "example.training_then_inference_bundle.training_export");
    });

    double prove_well_formedness_seconds = 0.0;
    double prove_training_seconds = 0.0;
    double verify_well_formedness_seconds = 0.0;
    double verify_training_seconds = 0.0;
    {
        stageMemorySampler sampler;
        sampler.start();
        prove_training_seconds = seconds_for([&] {
            proof_cost::Scope cost_scope(cost_report);
            proof_cost::PrefixScope phase_scope("prover_training");
            training_artifact = prove_forest_training_with_well_formedness_with_runtime(
                public_params,
                inputs.training_inputs,
                well_formedness_inputs,
                plan.committed_forest_public_inputs,
                "example.training_then_inference_bundle.training_with_well_formedness");
        });
        forestTrainingProofPublicInputs training_public_inputs =
            forest_training_proof_public_inputs_from_inputs(
                plan.committed_forest_public_inputs, inputs.training_inputs);
        const forestWellFormednessPublicInputs well_formedness_public_inputs =
            forest_well_formedness_public_inputs_from_inputs(
                plan.committed_forest_public_inputs, well_formedness_inputs);
        forestTrainingWithWellFormednessVerifierPreprocessing verifier;
        seconds_for([&] {
            verifier = prepare_forest_training_with_well_formedness_verifier_preprocessing(
                public_params,
                training_artifact,
                training_public_inputs,
                well_formedness_public_inputs,
                "example.training_then_inference_bundle.training_with_well_formedness");
        });
        verify_training_seconds = seconds_for([&] {
            proof_cost::Scope cost_scope(cost_report);
            proof_cost::PrefixScope phase_scope("verifier_training");
            training_verify_result =
                verify_forest_training_with_well_formedness_preprocessed(verifier);
        });
        proof_stage_memory = sampler.stop();
    }
    const bool accepted = training_verify_result.accepted;
    const bool fully_consumed = training_verify_result.proof_fully_consumed;
    const std::size_t training_with_well_formedness_proof_bytes =
        training_artifact.proof.bytes.size();

    std::cout << "Training proof profile\n";
    std::cout << "  mode: profile-training-from-split-traces\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  accepted: " << (accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (fully_consumed ? "yes" : "no") << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric committed_forest_rows " << plan.committed_forest.node_count << "\n";
    std::cout << "metric committed_forest_columns 12\n";
    std::cout << "metric committed_forest_domain_size "
              << nextPow2(plan.committed_forest.node_count * plan.committed_forest.feature_count) << "\n";
    std::cout << "metric well_formedness_proof_bytes 0\n";
    std::cout << "metric training_proof_bytes " << training_artifact.proof.bytes.size() << "\n";
    std::cout << "metric training_with_well_formedness_proof_bytes "
              << training_with_well_formedness_proof_bytes << "\n";
    print_semantic_workload_metrics(trace_case);
    add_terrae_application_cost_metrics(cost_report, inputs, srs_requirement);
    proof_cost::print_metrics(cost_report);
    print_stage_memory_metrics(proof_stage_memory);
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing assemble_inputs " << assemble_inputs_seconds << "\n";
    std::cout << "timing ensure_srs " << ensure_srs_seconds << "\n";
    std::cout << "timing prove_well_formedness " << prove_well_formedness_seconds << "\n";
    std::cout << "timing prove_training " << prove_training_seconds << "\n";
    std::cout << "timing prove_training_with_well_formedness "
              << (prove_well_formedness_seconds + prove_training_seconds) << "\n";
    std::cout << "timing verify_well_formedness_online " << verify_well_formedness_seconds << "\n";
    std::cout << "timing verify_training_online " << verify_training_seconds << "\n";
    std::cout << "timing verify_training_with_well_formedness_online "
              << (verify_well_formedness_seconds + verify_training_seconds) << "\n";
    return (accepted && fully_consumed) ? 0 : 1;
}

int run_profile_inference_from_split_traces(const std::string& srs_path,
                                            const std::string& training_trace_path,
                                            const std::string& inference_trace_path)
{
    kzgPublicParameters public_params;
    forestTrainingSemanticTraceCase trace_case;
    forestTrainingThenInferenceInputs inputs;
    forestExportedInferencePlan plan;
    forestExportedInferencePlanProfile plan_profile;
    ProofArtifact inference_artifact;
    VerificationResult verify_result;
    stageMemorySnapshot proof_stage_memory;
    proof_cost::Report cost_report;

    const double read_split_traces_seconds =
        seconds_for([&] { trace_case = read_forest_training_semantic_trace_case(training_trace_path, inference_trace_path); });
    const double assemble_inputs_seconds = seconds_for([&] {
        inputs = assemble_forest_training_then_inference_inputs_from_trace(trace_case.training_trace, trace_case.inference_queries);
    });
    const kzgSrsRequirement srs_requirement = required_exported_forest_classification_srs(inputs);
    const double load_srs_seconds =
        seconds_for([&] { public_params = load_required_kzg_public_parameters(srs_path, srs_requirement); });
    const double ensure_srs_seconds = seconds_for([&] {
        ensure_kzg_srs_covers(public_params, srs_requirement, "forest_training_then_inference_workflow");
    });
    seconds_for([&] {
        plan = assemble_exported_inference_plan_with_profile(
            public_params,
            inputs,
            plan_profile,
            "example.training_then_inference_bundle.inference_export");
    });

    double prove_inference_seconds = 0.0;
    double verify_inference_seconds = 0.0;
    {
        stageMemorySampler sampler;
        sampler.start();
        prove_inference_seconds = seconds_for([&] {
            proof_cost::Scope cost_scope(cost_report);
            proof_cost::PrefixScope phase_scope("prover_classification");
            inference_artifact = prove_exported_forest_classification_with_runtime(
                public_params,
                plan.classification_table,
                plan.witness_batch,
                "example.training_then_inference_bundle.inference_classification");
        });
        char temp_proof_path[] = "/tmp/terrae_inference_proof_XXXXXX";
        const int temp_proof_fd = ::mkstemp(temp_proof_path);
        if (temp_proof_fd < 0) {
            throw std::runtime_error("example.training_then_inference_bundle: failed to create inference proof temp file");
        }
        ::close(temp_proof_fd);
        write_serialized_proof_artifact(temp_proof_path, inference_artifact);
        const verifierProfileSummary verifier_profile = run_fresh_profile_command(
            "profile-inference-verifier-from-split-traces",
            srs_path,
            training_trace_path,
            inference_trace_path,
            temp_proof_path);
        merge_verifier_cost_metrics(cost_report, verifier_profile);
        ::unlink(temp_proof_path);
        verify_result.accepted = verifier_profile.accepted;
        verify_result.proof_fully_consumed = verifier_profile.proof_fully_consumed;
        verify_inference_seconds =
            verifier_profile.timings.count("verify_inference_online") == 0
                ? 0.0
                : verifier_profile.timings.at("verify_inference_online");
        proof_stage_memory = sampler.stop();
    }

    std::cout << "Inference proof profile\n";
    std::cout << "  mode: profile-inference-from-split-traces\n";
    std::cout << "  proof scope: routing+sum+argmax classification\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  accepted: " << (verify_result.accepted ? "yes" : "no") << "\n";
    std::cout << "  proof fully consumed: " << (verify_result.proof_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "metric tree_count " << inputs.training_inputs.tree_count << "\n";
    std::cout << "metric node_count " << inputs.training_inputs.node_count << "\n";
    std::cout << "metric feature_count " << inputs.training_inputs.feature_count << "\n";
    std::cout << "metric bin_count " << inputs.training_inputs.bin_count << "\n";
    std::cout << "metric leaf_node_count " << plan.exported_leaf_table.row_count << "\n";
    std::cout << "metric tree_query_row_count " << plan.witness_batch.query_count << "\n";
    std::cout << "metric query_count " << plan.witness_batch.query_count << "\n";
    std::cout << "metric sample_count " << plan.witness_batch.sample_count << "\n";
    std::cout << "metric inference_proof_bytes " << inference_artifact.proof.bytes.size() << "\n";
    std::cout << "metric classification_proof_bytes " << inference_artifact.proof.bytes.size() << "\n";
    print_semantic_workload_metrics(trace_case);
    add_terrae_application_cost_metrics(cost_report, inputs, srs_requirement);
    proof_cost::print_metrics(cost_report);
    print_stage_memory_metrics(proof_stage_memory);
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing read_split_traces " << read_split_traces_seconds << "\n";
    std::cout << "timing assemble_inputs " << assemble_inputs_seconds << "\n";
    std::cout << "timing ensure_srs " << ensure_srs_seconds << "\n";
    std::cout << "timing prove_inference " << prove_inference_seconds << "\n";
    std::cout << "timing prove_classification " << prove_inference_seconds << "\n";
    std::cout << "timing verify_inference_online " << verify_inference_seconds << "\n";
    std::cout << "timing verify_classification_online " << verify_inference_seconds << "\n";
    return (verify_result.accepted && verify_result.proof_fully_consumed) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    init_groups();

    try {
        if (argc == 2 && std::string(argv[1]) == "describe-formats") {
            return run_describe_formats();
        }
        if (argc == 2 && std::string(argv[1]) == "describe-trace-format") {
            return run_describe_trace_format();
        }
        if (argc == 2 && std::string(argv[1]) == "describe-trace-contract") {
            return run_describe_trace_contract();
        }
        if (argc == 3 && std::string(argv[1]) == "write-demo-case") {
            return run_write_demo_case(argv[2]);
        }
        if (argc == 3 && std::string(argv[1]) == "write-demo-trace-case") {
            return run_write_demo_trace_case(argv[2]);
        }
        if (argc == 3 && std::string(argv[1]) == "write-demo-training-trace") {
            return run_write_demo_training_trace(argv[2]);
        }
        if (argc == 3 && std::string(argv[1]) == "write-demo-inference-trace") {
            return run_write_demo_inference_trace(argv[2]);
        }
        if (argc == 3 && std::string(argv[1]) == "validate-trace-case") {
            return run_validate_trace_case(argv[2]);
        }
        if (argc == 3 && std::string(argv[1]) == "validate-training-trace") {
            return run_validate_training_trace(argv[2]);
        }
        if (argc == 3 && std::string(argv[1]) == "validate-inference-trace") {
            return run_validate_inference_trace(argv[2]);
        }
        if (argc == 4 && std::string(argv[1]) == "trace-to-case") {
            return run_trace_to_case(argv[2], argv[3]);
        }
        if (argc == 5 && std::string(argv[1]) == "split-traces-to-case") {
            return run_split_traces_to_case(argv[2], argv[3], argv[4]);
        }
        if (argc == 4 && std::string(argv[1]) == "required-srs-from-split-traces") {
            return run_required_srs_from_split_traces(argv[2], argv[3]);
        }
        if (argc == 4 && std::string(argv[1]) == "soundness-from-split-traces") {
            return run_soundness_from_split_traces(argv[2], argv[3]);
        }
        if (argc == 5 && std::string(argv[1]) == "prover") {
            return run_prover(argv[2], argv[3], argv[4]);
        }
        if (argc == 5 && std::string(argv[1]) == "verifier") {
            return run_verifier(argv[2], argv[3], argv[4]);
        }
        if (argc == 5 && std::string(argv[1]) == "prover-from-trace-case") {
            return run_prover_from_trace_case(argv[2], argv[3], argv[4]);
        }
        if (argc == 5 && std::string(argv[1]) == "verifier-from-trace-case") {
            return run_verifier_from_trace_case(argv[2], argv[3], argv[4]);
        }
        if (argc == 6 && std::string(argv[1]) == "prover-from-split-traces") {
            return run_prover_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "verifier-from-split-traces") {
            return run_verifier_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "prove-training-from-split-traces") {
            return run_prove_training_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "prove-inference-from-split-traces") {
            return run_prove_inference_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "prove-classification-from-split-traces") {
            return run_prove_classification_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "profile-verifier-from-split-traces") {
            return run_profile_verifier_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "profile-training-verifier-from-split-traces") {
            return run_profile_training_verifier_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "profile-inference-verifier-from-split-traces") {
            return run_profile_inference_verifier_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "profile-classification-verifier-from-split-traces") {
            return run_profile_classification_verifier_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 6 && std::string(argv[1]) == "profile-from-split-traces") {
            return run_profile_from_split_traces(argv[2], argv[3], argv[4], argv[5]);
        }
        if (argc == 5 && std::string(argv[1]) == "profile-commit-forest-from-split-traces") {
            return run_profile_commit_forest_from_split_traces(argv[2], argv[3], argv[4]);
        }
        if (argc == 5 && std::string(argv[1]) == "profile-well-formedness-from-split-traces") {
            return run_profile_well_formedness_from_split_traces(argv[2], argv[3], argv[4]);
        }
        if (argc == 5 && std::string(argv[1]) == "profile-training-from-split-traces") {
            return run_profile_training_from_split_traces(argv[2], argv[3], argv[4]);
        }
        if (argc == 5 && std::string(argv[1]) == "profile-inference-from-split-traces") {
            return run_profile_inference_from_split_traces(argv[2], argv[3], argv[4]);
        }
        print_usage(argv[0]);
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
