#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "forest_committed_inference_workflow.hpp"
#include "forest_training_demo_data.hpp"
#include "forest_training_trace_io.hpp"
#include "kzg.hpp"
#include "test_kzg_srs.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

void print_usage(const char* argv0)
{
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv0 << "\n";
    std::cerr << "  " << argv0 << " profile-synthetic-scale\n";
    std::cerr << "  " << argv0 << " <srs-path> <training-trace-path> <inference-trace-path>\n";
}

template <class Fn>
double seconds_for(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    std::forward<Fn>(fn)();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

template <class Fn>
double average_seconds_for(size_t repeats, Fn&& fn)
{
    if (repeats == 0) {
        throw std::runtime_error("repeats must be non-zero");
    }
    double total = 0.0;
    for (size_t i = 0; i < repeats; ++i) {
        total += seconds_for(fn);
    }
    return total / static_cast<double>(repeats);
}

forestTrainingSemanticTrace synthetic_training_trace(size_t tree_count)
{
    forestTrainingSemanticTrace trace;
    trace.tree_count = tree_count;
    trace.node_count = 4 * tree_count;
    trace.feature_count = 1;
    trace.bin_count = 4;
    trace.iteration_count = tree_count;
    trace.class_count = 1;
    trace.query_count = 4;
    trace.input_bit_bound = 3;
    trace.full_upper_bound = 4;
    trace.structure_bit_bound = 3;
    trace.gradient_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    trace.gain_numerator_encoding = fixedPointEncoding::signed_offset_binary(10, 0);
    trace.quantized_gain_encoding = fixedPointEncoding::signed_offset_binary(10, 2);
    trace.leaf_value_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    trace.score_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    trace.softmax_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    trace.hessian_bit_bound = 3;
    trace.gain_denominator_bit_bound = 3;
    trace.leaf_denominator_bit_bound = 2;
    trace.gain_gap_bit_bound = 2;
    trace.loss_mode = forestTrainingLoss::SecondOrderXgboost;
    trace.l2_regularization = 1;

    trace.training_inputs = {0, 1, 2, 3};
    trace.base_labels = {1, 1, 1, 1};
    trace.root_indices.resize(tree_count);
    trace.split_node_mask.assign(trace.node_count, 0);
    trace.left_child_indices.assign(trace.node_count, 0);
    trace.right_child_indices.assign(trace.node_count, 0);
    trace.node_tree_ids.assign(trace.node_count, 0);
    trace.feature_selector.assign(trace.node_count * trace.feature_count, 0);
    trace.thresholds.assign(trace.node_count, 0);
    trace.node_leaf_mask.assign(trace.node_count, 0);
    trace.node_leaf_values.assign(trace.node_count, 0);
    trace.leaf_indices.assign(tree_count * trace.query_count, 0);
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);
    trace.selected_candidate_indices.assign(trace.node_count, 0);

    for (size_t tree = 0; tree < tree_count; ++tree) {
        const size_t root = 4 * tree;
        trace.root_indices[tree] = static_cast<std::int64_t>(root);
        trace.split_node_mask[root] = 1;
        trace.left_child_indices[root] = static_cast<std::int64_t>(root + 1);
        trace.right_child_indices[root] = static_cast<std::int64_t>(root + 3);
        trace.node_tree_ids[root] = static_cast<std::int64_t>(tree + 1);
        trace.node_tree_ids[root + 1] = static_cast<std::int64_t>(tree + 1);
        trace.node_tree_ids[root + 2] = static_cast<std::int64_t>(tree + 1);
        trace.node_tree_ids[root + 3] = static_cast<std::int64_t>(tree + 1);
        trace.feature_selector[root] = 1;
        trace.thresholds[root] = 2;
        trace.node_leaf_mask[root + 1] = 1;
        trace.node_leaf_mask[root + 3] = 1;
        trace.node_leaf_values[root + 1] = 5;
        trace.node_leaf_values[root + 3] = 9;
        trace.selected_candidate_indices[root] = 2;

        const size_t query_offset = tree * trace.query_count;
        trace.leaf_indices[query_offset + 0] = static_cast<std::int64_t>(root + 1);
        trace.leaf_indices[query_offset + 1] = static_cast<std::int64_t>(root + 1);
        trace.leaf_indices[query_offset + 2] = static_cast<std::int64_t>(root + 3);
        trace.leaf_indices[query_offset + 3] = static_cast<std::int64_t>(root + 3);
    }
    return trace;
}

forestSemanticInferenceQueryBatch synthetic_inference_trace(size_t tree_count)
{
    forestSemanticInferenceQueryBatch trace;
    trace.query_count = 4 * tree_count;
    trace.feature_count = 1;
    trace.query_tree_ids.resize(trace.query_count);
    trace.lifted_inputs.resize(trace.query_count);
    for (size_t tree = 0; tree < tree_count; ++tree) {
        const size_t offset = 4 * tree;
        trace.query_tree_ids[offset + 0] = static_cast<std::int64_t>(tree + 1);
        trace.query_tree_ids[offset + 1] = static_cast<std::int64_t>(tree + 1);
        trace.query_tree_ids[offset + 2] = static_cast<std::int64_t>(tree + 1);
        trace.query_tree_ids[offset + 3] = static_cast<std::int64_t>(tree + 1);
        trace.lifted_inputs[offset + 0] = 0;
        trace.lifted_inputs[offset + 1] = 1;
        trace.lifted_inputs[offset + 2] = 2;
        trace.lifted_inputs[offset + 3] = 3;
    }
    return trace;
}

int run_profile_synthetic_scale()
{
    constexpr size_t kVerifyRepeats = 7;
    std::cout << "Committed forest inference synthetic T-scale\n";
    for (size_t tree_count : std::vector<size_t>{1, 2, 4, 8}) {
        const forestTrainingSemanticTrace training_trace = synthetic_training_trace(tree_count);
        const forestSemanticInferenceQueryBatch inference_trace = synthetic_inference_trace(tree_count);
        const forestCommittedInferenceInputs inputs =
            assemble_forest_committed_inference_inputs_from_trace(training_trace, inference_trace);
        const kzgSrsRequirement requirement = required_committed_forest_inference_srs(inputs);
        const kzgPublicParameters public_params =
            testing_kzg_public_parameters(requirement.deg1_plus_1, requirement.deg2_plus_1);
        const forestCommittedInferencePublicInputs preprocessed_public_inputs =
            forest_committed_inference_public_inputs_from_inputs(public_params, inputs);
        forestCommittedInferencePublicInputs raw_public_inputs = preprocessed_public_inputs;
        raw_public_inputs.query_commitments.reset();

        ProofArtifact artifact;
        artifact = prove_committed_forest_inference_with_runtime(
            public_params, inputs, "example.synthetic.committed_forest_inference");

        VerificationResult raw_result;
        const double raw_seconds = average_seconds_for(kVerifyRepeats, [&] {
            raw_result = verify_committed_forest_inference_with_runtime(public_params,
                                                                        artifact,
                                                                        raw_public_inputs,
                                                                        "example.synthetic.committed_forest_inference");
        });
        VerificationResult preprocessed_result;
        const double preprocessed_seconds = average_seconds_for(kVerifyRepeats, [&] {
            preprocessed_result = verify_committed_forest_inference_with_runtime(
                public_params,
                artifact,
                preprocessed_public_inputs,
                "example.synthetic.committed_forest_inference");
        });
        if (!raw_result.accepted || !raw_result.proof_fully_consumed || !preprocessed_result.accepted ||
            !preprocessed_result.proof_fully_consumed) {
            throw std::runtime_error("synthetic committed inference verification failed");
        }
        std::cout << "tree_count=" << tree_count << " query_count=" << inference_trace.query_count
                  << " verify_repeats=" << kVerifyRepeats << " verify_raw_avg=" << raw_seconds
                  << " verify_preprocessed_avg=" << preprocessed_seconds << "\n";
    }
    return 0;
}

int run_demo()
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(4096, 4096);
    forestTrainingSemanticTrace trace = build_demo_training_trace();
    trace.right_child_indices = {3, 0, 0, 0};
    trace.node_leaf_mask = {0, 1, 0, 1};
    trace.node_leaf_values = {0, 0, 0, 0};
    trace.leaf_indices = {1, 1, 3, 3};
    const forestSemanticInferenceQueryBatch inference_trace = build_demo_inference_query_trace();
    const forestCommittedInferenceInputs inputs =
        assemble_forest_committed_inference_inputs_from_trace(trace, inference_trace);
    const forestCommittedInferencePublicInputs public_inputs =
        forest_committed_inference_public_inputs_from_inputs(public_params, inputs);

    const ProofArtifact artifact =
        prove_committed_forest_inference_with_runtime(public_params, inputs, "example.committed_forest_inference");
    const VerificationResult result = verify_committed_forest_inference_with_runtime(
        public_params, artifact, public_inputs, "example.committed_forest_inference");

    if (!result.accepted || !result.proof_fully_consumed) {
        std::cerr << "committed forest inference proof verification failed" << std::endl;
        return 1;
    }

    std::cout << "committed forest inference proof succeeded" << std::endl;
    return 0;
}

int run_trace_mode(const std::string& srs_path,
                   const std::string& training_trace_path,
                   const std::string& inference_trace_path)
{
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    const forestTrainingSemanticTrace training_trace = read_forest_training_semantic_trace(training_trace_path);
    const forestSemanticInferenceQueryBatch inference_trace =
        read_forest_semantic_inference_query_trace(inference_trace_path);

    forestCommittedInferenceInputs inputs =
        assemble_forest_committed_inference_inputs_from_trace(training_trace, inference_trace);
    const forestCommittedInferencePublicInputs preprocessed_public_inputs =
        forest_committed_inference_public_inputs_from_inputs(public_params, inputs);
    forestCommittedInferencePublicInputs raw_public_inputs = preprocessed_public_inputs;
    raw_public_inputs.query_commitments.reset();

    ProofArtifact artifact;
    const double prove_seconds = seconds_for([&] {
        artifact = prove_committed_forest_inference_with_runtime(
            public_params,
            inputs,
            "example.committed_forest_inference");
    });

    VerificationResult raw_result;
    const double verify_seconds_raw = seconds_for([&] {
        raw_result = verify_committed_forest_inference_with_runtime(
            public_params,
            artifact,
            raw_public_inputs,
            "example.committed_forest_inference");
    });

    VerificationResult preprocessed_result;
    const double verify_seconds_preprocessed = seconds_for([&] {
        preprocessed_result = verify_committed_forest_inference_with_runtime(
            public_params,
            artifact,
            preprocessed_public_inputs,
            "example.committed_forest_inference");
    });

    std::cout << "Committed forest inference example\n";
    std::cout << "  training trace: " << training_trace_path << "\n";
    std::cout << "  inference trace: " << inference_trace_path << "\n";
    std::cout << "  tree_count: " << training_trace.tree_count * training_trace.class_count * training_trace.iteration_count
              << "\n";
    std::cout << "  node_count: " << training_trace.node_count << "\n";
    std::cout << "  query_count: " << inference_trace.query_count << "\n";
    std::cout << "  proof_bytes: " << artifact.proof.bytes.size() << "\n";
    std::cout << "  prove_seconds: " << prove_seconds << "\n";
    std::cout << "  verify_seconds_raw_public_polys: " << verify_seconds_raw << "\n";
    std::cout << "  verify_seconds_preprocessed_commitments: " << verify_seconds_preprocessed << "\n";
    std::cout << "  raw accepted: " << (raw_result.accepted ? "yes" : "no") << "\n";
    std::cout << "  raw proof fully consumed: " << (raw_result.proof_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "  preprocessed accepted: " << (preprocessed_result.accepted ? "yes" : "no") << "\n";
    std::cout << "  preprocessed proof fully consumed: " << (preprocessed_result.proof_fully_consumed ? "yes" : "no")
              << "\n";
    return (raw_result.accepted && raw_result.proof_fully_consumed && preprocessed_result.accepted &&
            preprocessed_result.proof_fully_consumed)
               ? 0
               : 1;
}

} // namespace

int main(int argc, char** argv)
{
    init_groups();

    try {
        if (argc == 1) {
            return run_demo();
        }
        if (argc == 2 && std::string(argv[1]) == "profile-synthetic-scale") {
            return run_profile_synthetic_scale();
        }
        if (argc != 4) {
            print_usage(argv[0]);
            return 1;
        }

        return run_trace_mode(argv[1], argv[2], argv[3]);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
