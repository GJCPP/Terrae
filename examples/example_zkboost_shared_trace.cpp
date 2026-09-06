#include "bls12_381_wrapper.hpp"
#include "forest_training_trace_io.hpp"
#include "test_kzg_srs.hpp"
#include "zkboost/shared_trace/cert_gbdt.hpp"
#include "zkboost/shared_trace/forest_commitments.hpp"
#include "zkboost/shared_trace/inference_workflow.hpp"
#include "zkboost/shared_trace/shared_prediction.hpp"
#include "zkboost/shared_trace/validate_forest_plonk.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

void print_semantic_workload_metrics(const forestTrainingSemanticTrace& training,
                                     const forestSemanticInferenceQueryBatch& inference)
{
    const size_t internal_node_count = static_cast<size_t>(
        std::count(training.split_node_mask.begin(), training.split_node_mask.end(), std::int64_t(1)));
    const size_t leaf_node_count = static_cast<size_t>(
        std::count(training.node_leaf_mask.begin(), training.node_leaf_mask.end(), std::int64_t(1)));
    const size_t inference_sample_count = training.tree_count == 0 ? 0 : inference.query_count / training.tree_count;
    std::cout << "metric record_count_padded " << training.query_count << "\n";
    std::cout << "metric class_count " << training.class_count << "\n";
    std::cout << "metric max_depth " << training.max_depth << "\n";
    std::cout << "metric internal_node_count " << internal_node_count << "\n";
    std::cout << "metric semantic_leaf_node_count " << leaf_node_count << "\n";
    std::cout << "metric inference_sample_count " << inference_sample_count << "\n";
    std::cout << "metric checked_path_count " << inference.query_count << "\n";
}

int run_profile_inference_with_mode(const std::string& srs_path,
                                    const std::string& training_trace_path,
                                    const std::string& inference_trace_path,
                                    zkboostSharedValidateForestPublicInputMode public_input_mode,
                                    const char* mode_label);

void print_usage(const char* argv0)
{
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv0
              << " profile-validate-forest <srs-path> <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0
              << " profile-commit-forest <srs-path> <training-trace-path>\n";
    std::cerr << "  " << argv0
              << " profile-inference <srs-path> <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0
              << " profile-inference-preprocessed <srs-path> <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0
              << " required-srs-preprocessed <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0
              << " required-component-srs <training-trace-path> <inference-trace-path>\n";
    std::cerr << "  " << argv0
              << " profile-synthetic-inference-scale\n";
    std::cerr << "  " << argv0
              << " profile-cert-gbdt <srs-path> <training-trace-path> <inference-trace-path>\n";
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

forestTrainingSemanticTrace synthetic_two_class_training_trace(size_t iteration_count)
{
    forestTrainingSemanticTrace trace;
    trace.tree_count = 2 * iteration_count;
    trace.node_count = 4 * trace.tree_count;
    trace.feature_count = 1;
    trace.bin_count = 4;
    trace.iteration_count = iteration_count;
    trace.class_count = 2;
    trace.query_count = 4;
    trace.input_bit_bound = 3;
    trace.full_upper_bound = 4;
    trace.structure_bit_bound = 3;
    trace.gradient_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    trace.gain_numerator_encoding = fixedPointEncoding::signed_offset_binary(10, 0);
    trace.quantized_gain_encoding = fixedPointEncoding::signed_offset_binary(10, 2);
    trace.leaf_value_encoding = fixedPointEncoding::signed_offset_binary(6, 0);
    size_t score_bit_bound = 6;
    const size_t max_aggregate_score = 5 * iteration_count;
    while ((size_t(1) << (score_bit_bound - 1)) - 1 < max_aggregate_score) {
        ++score_bit_bound;
    }
    trace.score_encoding =
        fixedPointEncoding::signed_offset_binary(score_bit_bound, 0);
    trace.softmax_encoding = fixedPointEncoding::signed_offset_binary(6, 2);
    trace.hessian_bit_bound = 3;
    trace.gain_denominator_bit_bound = 3;
    trace.leaf_denominator_bit_bound = 2;
    trace.gain_gap_bit_bound = 2;
    trace.loss_mode = forestTrainingLoss::SecondOrderXgboost;
    trace.l2_regularization = 1;

    trace.training_inputs = {0, 1, 2, 3};
    trace.base_labels = {
        1, 1, 0, 0,
        0, 0, 1, 1,
    };
    trace.root_indices.resize(trace.tree_count);
    trace.split_node_mask.assign(trace.node_count, 0);
    trace.left_child_indices.assign(trace.node_count, 0);
    trace.right_child_indices.assign(trace.node_count, 0);
    trace.node_tree_ids.assign(trace.node_count, 0);
    trace.feature_selector.assign(trace.node_count * trace.feature_count, 0);
    trace.thresholds.assign(trace.node_count, 0);
    trace.node_leaf_mask.assign(trace.node_count, 0);
    trace.node_leaf_values.assign(trace.node_count, 0);
    trace.leaf_indices.assign(trace.tree_count * trace.query_count, 0);
    trace.candidate_mask.assign(trace.node_count * trace.feature_count * trace.bin_count, 0);
    trace.selected_candidate_indices.assign(trace.node_count, 0);

    for (size_t tree = 0; tree < trace.tree_count; ++tree) {
        const size_t root = 4 * tree;
        const bool class_one_tree = (tree % 2) == 1;
        trace.root_indices[tree] = static_cast<std::int64_t>(root);
        trace.split_node_mask[root] = 1;
        trace.left_child_indices[root] = static_cast<std::int64_t>(root + 1);
        trace.right_child_indices[root] = static_cast<std::int64_t>(root + 2);
        trace.left_child_indices[root + 1] = static_cast<std::int64_t>(root + 1);
        trace.right_child_indices[root + 1] = static_cast<std::int64_t>(root + 1);
        trace.left_child_indices[root + 2] = static_cast<std::int64_t>(root + 2);
        trace.right_child_indices[root + 2] = static_cast<std::int64_t>(root + 2);
        trace.left_child_indices[root + 3] = static_cast<std::int64_t>(root + 3);
        trace.right_child_indices[root + 3] = static_cast<std::int64_t>(root + 3);
        trace.node_tree_ids[root] = static_cast<std::int64_t>(tree + 1);
        trace.node_tree_ids[root + 1] = static_cast<std::int64_t>(tree + 1);
        trace.node_tree_ids[root + 2] = static_cast<std::int64_t>(tree + 1);
        trace.node_tree_ids[root + 3] = static_cast<std::int64_t>(tree + 1);
        trace.feature_selector[root] = 1;
        trace.thresholds[root] = 2;
        trace.node_leaf_mask[root + 1] = 1;
        trace.node_leaf_mask[root + 2] = 1;
        trace.node_leaf_values[root + 1] = class_one_tree ? -5 : 5;
        trace.node_leaf_values[root + 2] = class_one_tree ? 5 : -5;
        trace.selected_candidate_indices[root] = 1;

        const size_t leaf_offset = tree * trace.query_count;
        trace.leaf_indices[leaf_offset + 0] = static_cast<std::int64_t>(root + 1);
        trace.leaf_indices[leaf_offset + 1] = static_cast<std::int64_t>(root + 1);
        trace.leaf_indices[leaf_offset + 2] = static_cast<std::int64_t>(root + 2);
        trace.leaf_indices[leaf_offset + 3] = static_cast<std::int64_t>(root + 2);
    }
    return trace;
}

forestSemanticInferenceQueryBatch synthetic_two_class_inference_trace(size_t tree_count)
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

kzgPublicParameters load_required_zkboost_public_parameters(const std::string& srs_path,
                                                            const zkboostSharedKzgSrsRequirement& requirement)
{
    if (requirement.deg1_plus_1 == 0 || requirement.deg2_plus_1 == 0) {
        throw std::runtime_error("CertXGB SRS requirement must be non-zero");
    }
    return load_kzg_public_parameters(srs_path, requirement.deg1_plus_1, requirement.deg2_plus_1);
}

int run_profile_validate_forest(const std::string& srs_path,
                                const std::string& training_trace_path,
                                const std::string& inference_trace_path)
{
    forestTrainingSemanticTrace training;
    forestSemanticInferenceQueryBatch inference;
    kzgPublicParameters public_params;

    const double read_training_seconds = seconds_for([&] {
        training = read_forest_training_semantic_trace(training_trace_path);
    });
    const double read_inference_seconds = seconds_for([&] {
        inference = read_forest_semantic_inference_query_trace(inference_trace_path);
    });
    const zkboostSharedKzgSrsRequirement required_srs_requirement{
        required_zkboost_shared_validate_forest_plonk_srs_size(training, inference),
        2,
    };
    const double load_srs_seconds = seconds_for([&] {
        public_params = load_required_zkboost_public_parameters(srs_path, required_srs_requirement);
    });

    zkboostSharedTracePrediction prediction;
    zkboostSharedValidateForestPlonkInstance instance;
    plonkProof proof;
    bool accepted = false;

    const double prediction_seconds = seconds_for([&] {
        prediction = derive_zkboost_shared_trace_prediction(training, inference);
    });
    const double build_seconds = seconds_for([&] {
        instance = build_zkboost_shared_validate_forest_plonk_instance(
            training,
            inference,
            public_params);
    });
    const double prove_seconds = seconds_for([&] {
        proof = prove_zkboost_shared_validate_forest_plonk(instance);
    });
    const double verify_seconds = seconds_for([&] {
        accepted = verify_zkboost_shared_validate_forest_plonk(instance, proof);
    });
    if (!accepted) {
        throw std::runtime_error("shared ValidateForest proof did not verify");
    }

    const SerializedProof serialized = serialize_plonk_proof(proof);

    std::cout << "CertXGB shared trace ValidateForest\n";
    std::cout << "  statement_level: forest_routing\n";
    std::cout << "  training_trace: " << training_trace_path << "\n";
    std::cout << "  inference_trace: " << inference_trace_path << "\n";
    std::cout << "  shared_accuracy: " << prediction.accuracy_numerator
              << "/" << prediction.accuracy_denominator << "\n";
    std::cout << "  shared_prediction_digest: " << prediction.prediction_digest << "\n";
    std::cout << "metric tree_count " << instance.summary.tree_count << "\n";
    std::cout << "metric sample_count " << instance.summary.sample_count << "\n";
    std::cout << "metric node_count " << instance.summary.node_count << "\n";
    std::cout << "metric checked_paths " << instance.summary.checked_paths << "\n";
    std::cout << "metric split_node_count " << instance.summary.split_node_count << "\n";
    std::cout << "metric leaf_node_count " << instance.summary.leaf_node_count << "\n";
    std::cout << "metric committed_table_comparison_rows "
              << instance.committed_table.comparison_row_count << "\n";
    std::cout << "metric committed_table_total_lookup_rows "
              << instance.committed_table.total_lookup_row_count << "\n";
    std::cout << "metric committed_table_domain_size "
              << instance.committed_table.table_domain_size << "\n";
    std::cout << "metric committed_table_commitment_count 3\n";
    std::cout << "metric public_input_count " << instance.public_inputs.size() << "\n";
    std::cout << "metric proof_bytes " << serialized.bytes.size() << "\n";
    std::cout << "metric accuracy_numerator " << prediction.accuracy_numerator << "\n";
    std::cout << "metric accuracy_denominator " << prediction.accuracy_denominator << "\n";
    std::cout << "metric required_srs_size " << required_srs_requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg1_plus_1 " << required_srs_requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg2_plus_1 " << required_srs_requirement.deg2_plus_1 << "\n";
    std::cout << "timing read_training_trace " << read_training_seconds << "\n";
    std::cout << "timing read_inference_trace " << read_inference_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing shared_prediction " << prediction_seconds << "\n";
    std::cout << "timing build_instance " << build_seconds << "\n";
    std::cout << "timing prove " << prove_seconds << "\n";
    std::cout << "timing verify " << verify_seconds << "\n";
    std::cout << "timing prove_verify_total " << (prove_seconds + verify_seconds) << "\n";
    return 0;
}

int run_profile_cert_gbdt(const std::string& srs_path,
                          const std::string& training_trace_path,
                          const std::string& inference_trace_path)
{
    forestTrainingSemanticTrace training;
    forestSemanticInferenceQueryBatch inference;
    kzgPublicParameters public_params;

    const double read_training_seconds = seconds_for([&] {
        training = read_forest_training_semantic_trace(training_trace_path);
    });
    const double read_inference_seconds = seconds_for([&] {
        inference = read_forest_semantic_inference_query_trace(inference_trace_path);
    });
    const zkboostSharedKzgSrsRequirement required_srs_requirement =
        required_zkboost_shared_cert_gbdt_srs_requirement(training, inference);
    const double load_srs_seconds = seconds_for([&] {
        public_params = load_required_zkboost_public_parameters(srs_path, required_srs_requirement);
    });

    zkboostSharedCertGBDTInstance instance;
    zkboostSharedCertGBDTProof proof;
    zkboostSharedCertGBDTVerifierBundle verifier_bundle;
    zkboostSharedCertGBDTVerifierPreprocessing verifier_preprocessing;
    bool accepted = false;
    proof_cost::Report cost_report;

    const double build_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope phase_scope("preprocess");
        instance = build_zkboost_shared_cert_gbdt_instance(
            training,
            inference,
            public_params);
    });
    const zkboostSharedCertGBDTLinkChecks link_checks =
        check_zkboost_shared_cert_gbdt_links(instance);
    if (!link_checks.all_ok()) {
        std::cerr << "shared CertGBDT link preflight:"
                  << " forest_to_fetch=" << link_checks.forest_to_fetch
                  << " forest_wf_commitment=" << link_checks.forest_well_formedness_commitment_binding
                  << " forest_wf_to_fetch=" << link_checks.forest_well_formedness_to_fetch
                  << " forest_wf_to_splits=" << link_checks.forest_well_formedness_to_splits
                  << " forest_wf_to_leaf_values=" << link_checks.forest_well_formedness_to_leaf_values
                  << " forest_to_boosting_state=" << link_checks.forest_to_boosting_state
                  << " forest_to_histograms=" << link_checks.forest_to_histograms
                  << " boosting_state_to_histograms=" << link_checks.boosting_state_to_histograms
                  << " histograms_to_weighted_logup=" << link_checks.histograms_to_weighted_logup
                  << " histograms_to_leaf_values=" << link_checks.histograms_to_leaf_values
                  << " forest_to_leaf_values=" << link_checks.forest_to_leaf_values
                  << " histograms_to_splits=" << link_checks.histograms_to_splits
                  << " forest_to_splits=" << link_checks.forest_to_splits
                  << "\n";
        throw std::runtime_error("shared CertGBDT link preflight failed");
    }
    std::cout << "shared CertGBDT link preflight passed\n" << std::flush;
    const double prove_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope component_scope("prover_cross_session_batch");
        proof = prove_zkboost_shared_cert_gbdt(instance);
    });
    seconds_for([&] {
        verifier_bundle = prepare_zkboost_shared_cert_gbdt_verifier_bundle(instance);
        verifier_preprocessing =
            prepare_zkboost_shared_cert_gbdt_verifier_preprocessing(
                verifier_bundle, proof);
    });
    const double verify_seconds = seconds_for([&] {
        accepted = verify_zkboost_shared_cert_gbdt_preprocessed(
            verifier_preprocessing);
    });
    if (!accepted) {
        throw std::runtime_error("shared CertGBDT proof did not verify");
    }

    const size_t proof_bytes =
        serialized_zkboost_shared_cert_gbdt_proof_bytes(proof);

    std::cout << "CertXGB shared trace CertGBDT\n";
    std::cout << "  statement_level: training_with_forest_well_formedness\n";
    std::cout << "  training_trace: " << training_trace_path << "\n";
    std::cout << "  inference_trace: " << inference_trace_path << "\n";
    std::cout << "metric tree_count " << instance.summary.tree_count << "\n";
    std::cout << "metric sample_count " << instance.summary.sample_count << "\n";
    std::cout << "metric node_count " << instance.summary.node_count << "\n";
    std::cout << "metric feature_count " << instance.summary.feature_count << "\n";
    std::cout << "metric bin_count " << instance.summary.bin_count << "\n";
    std::cout << "metric checked_paths " << instance.summary.checked_paths << "\n";
    std::cout << "metric candidate_count " << instance.summary.candidate_count << "\n";
    std::cout << "metric forest_public_input_count " << instance.forest.public_inputs.size() << "\n";
    std::cout << "metric forest_well_formedness_public_input_count "
              << instance.forest_well_formedness.public_inputs.size() << "\n";
    std::cout << "metric boosting_public_input_count "
              << instance.boosting_state.public_inputs.size() << "\n";
    std::cout << "metric histogram_public_input_count "
              << instance.histograms.public_inputs.size() << "\n";
    std::cout << "metric leaf_value_public_input_count "
              << instance.leaf_values.public_inputs.size() << "\n";
    std::cout << "metric split_public_input_count "
              << instance.splits.public_inputs.size() << "\n";
    std::cout << "metric fetch_proof_bytes " << proof.forest_fetch.bytes.size() << "\n";
    std::cout << "metric cross_session_opening_proof_bytes "
              << proof.cross_session_opening_batch.bytes.size() << "\n";
    std::cout << "metric proof_bytes " << proof_bytes << "\n";
    std::cout << "metric required_srs_size " << required_srs_requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg1_plus_1 " << required_srs_requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg2_plus_1 " << required_srs_requirement.deg2_plus_1 << "\n";
    print_semantic_workload_metrics(training, inference);
    cost_report.add("application_commitment_count", 3);
    cost_report.add("required_srs_deg1_plus_1", required_srs_requirement.deg1_plus_1);
    cost_report.add("required_srs_deg2_plus_1", required_srs_requirement.deg2_plus_1);
    proof_cost::print_metrics(cost_report);
    std::cout << "timing read_training_trace " << read_training_seconds << "\n";
    std::cout << "timing read_inference_trace " << read_inference_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing build_instance " << build_seconds << "\n";
    std::cout << "timing prove " << prove_seconds << "\n";
    std::cout << "timing verify " << verify_seconds << "\n";
    std::cout << "timing prove_verify_total " << (prove_seconds + verify_seconds) << "\n";
    return 0;
}

int run_profile_commit_forest(const std::string& srs_path,
                              const std::string& training_trace_path)
{
    forestTrainingSemanticTrace training;
    kzgPublicParameters public_params;

    const double read_training_seconds = seconds_for([&] {
        training = read_forest_training_semantic_trace(training_trace_path);
    });
    const zkboostSharedKzgSrsRequirement required_srs_requirement =
        required_zkboost_shared_forest_commitment_srs_requirement_from_training_trace(training);
    const double load_srs_seconds = seconds_for([&] {
        public_params = load_required_zkboost_public_parameters(srs_path, required_srs_requirement);
    });

    zkboostSharedCommittedForest committed_forest;
    seconds_for([&] {
        committed_forest = build_zkboost_shared_committed_forest(
            public_params,
            training);
    });

    std::cout << "CertXGB shared trace CommitForest\n";
    std::cout << "  statement_level: commit_forest\n";
    std::cout << "  training_trace: " << training_trace_path << "\n";
    std::cout << "metric tree_count " << committed_forest.summary.tree_count << "\n";
    std::cout << "metric node_count " << committed_forest.summary.node_count << "\n";
    std::cout << "metric feature_count " << committed_forest.summary.feature_count << "\n";
    std::cout << "metric bin_count " << committed_forest.summary.bin_count << "\n";
    std::cout << "metric max_depth " << committed_forest.summary.max_depth << "\n";
    std::cout << "metric internal_node_count_per_tree "
              << committed_forest.summary.internal_node_count_per_tree << "\n";
    std::cout << "metric leaf_count_per_tree "
              << committed_forest.summary.leaf_count_per_tree << "\n";
    std::cout << "metric split_domain_size " << committed_forest.summary.split_domain_size << "\n";
    std::cout << "metric leaf_domain_size " << committed_forest.summary.leaf_domain_size << "\n";
    std::cout << "metric forest_commitment_count " << committed_forest.summary.commitment_count << "\n";
    std::cout << "metric forest_commitment_bytes " << committed_forest.summary.commitment_bytes << "\n";
    std::cout << "metric required_srs_size " << required_srs_requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg1_plus_1 " << required_srs_requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg2_plus_1 " << required_srs_requirement.deg2_plus_1 << "\n";
    std::cout << "timing read_training_trace " << read_training_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    return 0;
}

int run_profile_inference(const std::string& srs_path,
                          const std::string& training_trace_path,
                          const std::string& inference_trace_path)
{
    return run_profile_inference_with_mode(srs_path,
                                           training_trace_path,
                                           inference_trace_path,
                                           zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs,
                                           "profile-inference");
}

int run_profile_inference_with_mode(
    const std::string& srs_path,
    const std::string& training_trace_path,
    const std::string& inference_trace_path,
    zkboostSharedValidateForestPublicInputMode public_input_mode,
    const char* mode_label)
{
    forestTrainingSemanticTrace training;
    forestSemanticInferenceQueryBatch inference;
    kzgPublicParameters public_params;

    const double read_training_seconds = seconds_for([&] {
        training = read_forest_training_semantic_trace(training_trace_path);
    });
    const double read_inference_seconds = seconds_for([&] {
        inference = read_forest_semantic_inference_query_trace(inference_trace_path);
    });
    const zkboostSharedKzgSrsRequirement required_srs_requirement =
        required_zkboost_shared_inference_srs_requirement(training, inference, public_input_mode);
    const double load_srs_seconds = seconds_for([&] {
        public_params = load_required_zkboost_public_parameters(srs_path, required_srs_requirement);
    });

    zkboostSharedCommittedForest committed_forest;
    zkboostSharedInferenceInstance instance;
    zkboostSharedInferenceProof proof;
    zkboostSharedInferenceVerifierBundle verifier_bundle;
    zkboostSharedInferenceVerifierPreprocessing verifier_preprocessing;
    bool accepted = false;
    proof_cost::Report cost_report;

    seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope phase_scope("preprocess_commit_forest");
        committed_forest = build_zkboost_shared_committed_forest(
            public_params,
            training);
    });
    const double build_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        proof_cost::PrefixScope phase_scope("preprocess");
        instance = build_zkboost_shared_inference_instance_from_committed_forest(
            training,
            inference,
            committed_forest,
            public_params,
            public_input_mode);
    });
    double prove_inference_plonk_seconds = 0.0;
    const double prove_seconds = seconds_for([&] {
        proof_cost::Scope cost_scope(cost_report);
        prove_inference_plonk_seconds = seconds_for([&] {
            proof_cost::PrefixScope component_scope("prover_cross_session_batch");
            proof = prove_zkboost_shared_inference(instance);
        });
    });
    seconds_for([&] {
        verifier_bundle = prepare_zkboost_shared_inference_verifier_bundle(instance);
        verifier_preprocessing =
            prepare_zkboost_shared_inference_verifier_preprocessing(
                verifier_bundle, proof);
    });
    double verify_inference_plonk_seconds = 0.0;
    const double verify_seconds = seconds_for([&] {
        verify_inference_plonk_seconds = seconds_for([&] {
            accepted = verify_zkboost_shared_inference_preprocessed(
                verifier_preprocessing);
        });
    });
    if (!accepted) {
        throw std::runtime_error("shared inference proof did not verify");
    }

    const SerializedProof serialized_inference_plonk =
        serialize_plonk_proof(proof.inference_plonk);
    const size_t proof_bytes =
        serialized_zkboost_shared_inference_proof_bytes(proof);

    std::cout << "CertXGB shared trace Inference\n";
    std::cout << "  mode: " << mode_label << "\n";
    std::cout << "  statement_level: inference\n";
    std::cout << "  input_mode: "
              << (public_input_mode == zkboostSharedValidateForestPublicInputMode::PreprocessedConstants
                      ? "preprocessed_constants"
                      : "online_public_inputs")
              << "\n";
    std::cout << "  training_trace: " << training_trace_path << "\n";
    std::cout << "  inference_trace: " << inference_trace_path << "\n";
    std::cout << "  shared_accuracy: " << instance.prediction.accuracy_numerator
              << "/" << instance.prediction.accuracy_denominator << "\n";
    std::cout << "  shared_prediction_digest: " << instance.prediction.prediction_digest << "\n";
    std::cout << "metric tree_count " << instance.summary.tree_count << "\n";
    std::cout << "metric sample_count " << instance.summary.sample_count << "\n";
    std::cout << "metric node_count " << instance.summary.node_count << "\n";
    std::cout << "metric feature_count " << instance.summary.feature_count << "\n";
    std::cout << "metric bin_count " << instance.summary.bin_count << "\n";
    std::cout << "metric checked_paths " << instance.summary.checked_paths << "\n";
    std::cout << "metric forest_max_depth "
              << instance.committed_forest.summary.max_depth << "\n";
    std::cout << "metric forest_commitment_count "
              << instance.committed_forest.summary.commitment_count << "\n";
    std::cout << "metric forest_commitment_bytes "
              << instance.committed_forest.summary.commitment_bytes << "\n";
    std::cout << "metric inference_plonk_public_input_count "
              << instance.inference_plonk.public_inputs.size() << "\n";
    std::cout << "metric inference_plonk_proof_bytes "
              << serialized_inference_plonk.bytes.size() -
                     2 * G1::getSerializedByteSize()
              << "\n";
    std::cout << "metric fetch_proof_bytes " << proof.forest_fetch.bytes.size() << "\n";
    std::cout << "metric cross_session_opening_proof_bytes "
              << proof.cross_session_opening_batch.bytes.size() << "\n";
    std::cout << "metric proof_bytes " << proof_bytes << "\n";
    std::cout << "metric accuracy_numerator " << instance.prediction.accuracy_numerator << "\n";
    std::cout << "metric accuracy_denominator " << instance.prediction.accuracy_denominator << "\n";
    std::cout << "metric required_srs_size " << required_srs_requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg1_plus_1 " << required_srs_requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg2_plus_1 " << required_srs_requirement.deg2_plus_1 << "\n";
    print_semantic_workload_metrics(training, inference);
    cost_report.add("application_commitment_count", instance.committed_forest.summary.commitment_count);
    cost_report.add("required_srs_deg1_plus_1", required_srs_requirement.deg1_plus_1);
    cost_report.add("required_srs_deg2_plus_1", required_srs_requirement.deg2_plus_1);
    proof_cost::print_metrics(cost_report);
    std::cout << "timing read_training_trace " << read_training_seconds << "\n";
    std::cout << "timing read_inference_trace " << read_inference_seconds << "\n";
    std::cout << "timing load_srs " << load_srs_seconds << "\n";
    std::cout << "timing build_instance " << build_seconds << "\n";
    std::cout << "timing prove_inference_plonk " << prove_inference_plonk_seconds << "\n";
    std::cout << "timing verify_inference_plonk " << verify_inference_plonk_seconds << "\n";
    std::cout << "timing prove " << prove_seconds << "\n";
    std::cout << "timing verify " << verify_seconds << "\n";
    std::cout << "timing prove_verify_total " << (prove_seconds + verify_seconds) << "\n";
    return 0;
}

int run_required_srs_preprocessed(const std::string& training_trace_path,
                                  const std::string& inference_trace_path)
{
    forestTrainingSemanticTrace training;
    forestSemanticInferenceQueryBatch inference;

    const double read_training_seconds = seconds_for([&] {
        training = read_forest_training_semantic_trace(training_trace_path);
    });
    const double read_inference_seconds = seconds_for([&] {
        inference = read_forest_semantic_inference_query_trace(inference_trace_path);
    });

    const zkboostSharedKzgSrsRequirement commit_requirement =
        required_zkboost_shared_forest_commitment_srs_requirement_from_training_trace(training);
    const zkboostSharedKzgSrsRequirement inference_requirement =
        required_zkboost_shared_inference_srs_requirement(
            training,
            inference,
            zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);
    const zkboostSharedKzgSrsRequirement training_requirement =
        required_zkboost_shared_cert_gbdt_srs_requirement(training, inference);
    const zkboostSharedKzgSrsRequirement requirement =
        max_zkboost_shared_kzg_srs_requirement(
            max_zkboost_shared_kzg_srs_requirement(commit_requirement, inference_requirement),
            training_requirement);

    std::cout << "CertXGB required SRS\n";
    std::cout << "  mode: required-srs-preprocessed\n";
    std::cout << "  training_trace: " << training_trace_path << "\n";
    std::cout << "  inference_trace: " << inference_trace_path << "\n";
    std::cout << "metric required_srs_size " << requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg1_plus_1 " << requirement.deg1_plus_1 << "\n";
    std::cout << "metric required_srs_deg2_plus_1 " << requirement.deg2_plus_1 << "\n";
    std::cout << "metric commit_required_srs_deg1_plus_1 " << commit_requirement.deg1_plus_1 << "\n";
    std::cout << "metric commit_required_srs_deg2_plus_1 " << commit_requirement.deg2_plus_1 << "\n";
    std::cout << "metric inference_required_srs_deg1_plus_1 " << inference_requirement.deg1_plus_1 << "\n";
    std::cout << "metric inference_required_srs_deg2_plus_1 " << inference_requirement.deg2_plus_1 << "\n";
    std::cout << "metric training_required_srs_deg1_plus_1 " << training_requirement.deg1_plus_1 << "\n";
    std::cout << "metric training_required_srs_deg2_plus_1 " << training_requirement.deg2_plus_1 << "\n";
    std::cout << "timing read_training_trace " << read_training_seconds << "\n";
    std::cout << "timing read_inference_trace " << read_inference_seconds << "\n";
    return 0;
}

int run_required_component_srs(const std::string& training_trace_path,
                               const std::string& inference_trace_path)
{
    const forestTrainingSemanticTrace training =
        read_forest_training_semantic_trace(training_trace_path);
    const forestSemanticInferenceQueryBatch inference =
        read_forest_semantic_inference_query_trace(inference_trace_path);
    const zkboostSharedBoostingStateView boosting =
        build_zkboost_shared_boosting_state_view(training);
    const zkboostSharedHistogramView histograms =
        build_zkboost_shared_histogram_view(training, boosting);
    const zkboostSharedLeafValueView leaf_values =
        build_zkboost_shared_leaf_value_view(training, histograms);
    const zkboostSharedSplitView splits =
        build_zkboost_shared_split_view(training, histograms);
    const zkboostSharedValidateForestView forest =
        build_zkboost_shared_validate_forest_view(training, inference);
    const zkboostSharedPredictionView prediction =
        build_zkboost_shared_prediction_view(training, inference);

    std::cout << "CertXGB component SRS\n";
    std::cout << "metric forest_plonk_domain "
              << required_zkboost_shared_validate_forest_plonk_srs_size_from_view(
                     forest,
                     training.bin_count,
                     zkboostSharedValidateForestPublicInputMode::PreprocessedConstants)
              << "\n";
    std::cout << "metric forest_well_formedness_plonk_domain "
              << required_zkboost_shared_forest_well_formedness_plonk_srs_size(training)
              << "\n";
    std::cout << "metric boosting_state_plonk_domain "
              << required_zkboost_shared_boosting_state_plonk_srs_size_from_view(
                     training, boosting)
              << "\n";
    std::cout << "metric histogram_plonk_domain "
              << required_zkboost_shared_histogram_plonk_srs_size_from_view(
                     training, histograms)
              << "\n";
    std::cout << "metric histogram_weighted_logup_domain "
              << required_zkboost_shared_weighted_logup_srs_size_from_view(
                     training, histograms)
              << "\n";
    std::cout << "metric leaf_value_plonk_domain "
              << required_zkboost_shared_leaf_value_plonk_srs_size_from_view(
                     training, leaf_values)
              << "\n";
    std::cout << "metric split_plonk_domain "
              << required_zkboost_shared_split_plonk_srs_size_from_view(
                     training, splits)
              << "\n";
    std::cout << "metric prediction_plonk_domain "
              << required_zkboost_shared_prediction_plonk_srs_size_from_view(
                     training,
                     prediction,
                     zkboostSharedValidateForestPublicInputMode::PreprocessedConstants)
              << "\n";
    return 0;
}

int run_profile_synthetic_inference_scale()
{
    constexpr size_t kVerifyRepeats = 7;
    std::cout << "CertXGB shared trace synthetic T-scale\n";
    for (size_t iteration_count : std::vector<size_t>{1, 2, 4, 8}) {
        const forestTrainingSemanticTrace training = synthetic_two_class_training_trace(iteration_count);
        const forestSemanticInferenceQueryBatch inference = synthetic_two_class_inference_trace(training.tree_count);

        const zkboostSharedKzgSrsRequirement online_requirement =
            required_zkboost_shared_inference_srs_requirement(
                training,
                inference,
                zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);
        const zkboostSharedKzgSrsRequirement preprocessed_requirement =
            required_zkboost_shared_inference_srs_requirement(
                training,
                inference,
                zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);
        const kzgPublicParameters public_params = testing_kzg_public_parameters(
            std::max(online_requirement.deg1_plus_1, preprocessed_requirement.deg1_plus_1),
            std::max(online_requirement.deg2_plus_1, preprocessed_requirement.deg2_plus_1));

        const zkboostSharedInferenceInstance online_instance =
            build_zkboost_shared_inference_instance(
                training,
                inference,
                public_params,
                zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs);
        const zkboostSharedInferenceProof online_proof =
            prove_zkboost_shared_inference(online_instance);
        const zkboostSharedInferenceVerifierBundle online_verifier_bundle =
            prepare_zkboost_shared_inference_verifier_bundle(online_instance);

        bool online_accepted = false;
        const double online_verify_seconds = average_seconds_for(kVerifyRepeats, [&] {
            online_accepted =
                verify_zkboost_shared_inference_preprocessed(online_verifier_bundle, online_proof);
        });
        if (!online_accepted) {
            throw std::runtime_error("synthetic online shared inference proof did not verify");
        }

        const zkboostSharedInferenceInstance preprocessed_instance =
            build_zkboost_shared_inference_instance(
                training,
                inference,
                public_params,
                zkboostSharedValidateForestPublicInputMode::PreprocessedConstants);
        const zkboostSharedInferenceProof preprocessed_proof =
            prove_zkboost_shared_inference(preprocessed_instance);
        const zkboostSharedInferenceVerifierBundle preprocessed_verifier_bundle =
            prepare_zkboost_shared_inference_verifier_bundle(preprocessed_instance);

        bool preprocessed_accepted = false;
        const double preprocessed_verify_seconds = average_seconds_for(kVerifyRepeats, [&] {
            preprocessed_accepted =
                verify_zkboost_shared_inference_preprocessed(preprocessed_verifier_bundle, preprocessed_proof);
        });
        if (!preprocessed_accepted) {
            throw std::runtime_error("synthetic preprocessed shared inference proof did not verify");
        }

        std::cout << "iteration_count=" << iteration_count << " tree_count=" << training.tree_count
                  << " query_count=" << inference.query_count << " verify_repeats=" << kVerifyRepeats
                  << " inference_public_inputs_online=" << online_instance.inference_plonk.public_inputs.size()
                  << " inference_public_inputs_preprocessed=" << preprocessed_instance.inference_plonk.public_inputs.size()
                  << " verify_online_avg=" << online_verify_seconds
                  << " verify_preprocessed_avg=" << preprocessed_verify_seconds << "\n";
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    init_groups();
    try {
        if (argc == 5 && std::string(argv[1]) == "profile-validate-forest") {
            return run_profile_validate_forest(argv[2], argv[3], argv[4]);
        }
        if (argc == 4 && std::string(argv[1]) == "profile-commit-forest") {
            return run_profile_commit_forest(argv[2], argv[3]);
        }
        if (argc == 5 && std::string(argv[1]) == "profile-inference") {
            return run_profile_inference(argv[2], argv[3], argv[4]);
        }
        if (argc == 5 && std::string(argv[1]) == "profile-inference-preprocessed") {
            return run_profile_inference_with_mode(argv[2],
                                                   argv[3],
                                                   argv[4],
                                                   zkboostSharedValidateForestPublicInputMode::PreprocessedConstants,
                                                   "profile-inference-preprocessed");
        }
        if (argc == 4 && std::string(argv[1]) == "required-srs-preprocessed") {
            return run_required_srs_preprocessed(argv[2], argv[3]);
        }
        if (argc == 4 && std::string(argv[1]) == "required-component-srs") {
            return run_required_component_srs(argv[2], argv[3]);
        }
        if (argc == 2 && std::string(argv[1]) == "profile-synthetic-inference-scale") {
            return run_profile_synthetic_inference_scale();
        }
        if (argc == 5 && std::string(argv[1]) == "profile-cert-gbdt") {
            return run_profile_cert_gbdt(argv[2], argv[3], argv[4]);
        }
        print_usage(argv[0]);
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
