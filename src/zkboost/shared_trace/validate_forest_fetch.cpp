#include "zkboost/shared_trace/validate_forest_fetch.hpp"

#include "algebra/algebra_utils.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_validate_forest_fetch: " + message);
    }
}

struct forestCommitmentHandles {
    kzgCommitment split_features;
    kzgCommitment split_thresholds;
    kzgCommitment leaf_values;
};

struct fetchOpeningSet {
    size_t commitment_index = 0;
    std::vector<Fr> points;
    std::vector<Fr> values;
};

Fr split_row_point(size_t domain_size, size_t index)
{
    require_condition(index < domain_size, "split row index exceeds split domain");
    return subgroup_label(domain_size, index);
}

Fr leaf_row_point(size_t domain_size, size_t index)
{
    require_condition(index < domain_size, "leaf row index exceeds leaf domain");
    return subgroup_label(domain_size, index);
}

forestCommitmentHandles commit_or_adopt_forest_commitments(
    kzgProtocol& kzg,
    const zkboostSharedTreeEnsemble& ensemble,
    const zkboostSharedForestCommitments& commitments,
    bool prover_mode)
{
    const size_t split_degree =
        canonical_padded_zkboost_shared_forest_length(ensemble.split_features.size()) - 1;
    const size_t leaf_degree =
        canonical_padded_zkboost_shared_forest_length(ensemble.leaf_values.size()) - 1;

    auto bind_column = [&](const char* key,
                           const std::vector<std::int64_t>& raw_values,
                           const Group1& expected_group,
                           size_t max_degree) {
        if (prover_mode) {
            kzgCommitment expected;
            expected.com = expected_group;
            expected.index = 0;
            const kzgCommitment com =
                kzg.commit(poly_from_evals(encode_zkboost_shared_forest_column_padded(raw_values)),
                           expected);
            kzg.prove_degree(com, max_degree);
            return com;
        }

        const kzgCommitment com = kzg.adopt_public_commitment(key, expected_group);
        kzg.prove_degree(com, max_degree);
        return com;
    };

    forestCommitmentHandles out;
    out.split_features = bind_column("zkboost.shared.fetch.logical.split_features",
                                     ensemble.split_features,
                                     commitments.com_f,
                                     split_degree);
    out.split_thresholds = bind_column("zkboost.shared.fetch.logical.split_thresholds",
                                       ensemble.split_thresholds,
                                       commitments.com_t,
                                       split_degree);
    out.leaf_values = bind_column("zkboost.shared.fetch.logical.leaf_values",
                                  ensemble.leaf_values,
                                  commitments.com_w,
                                  leaf_degree);
    return out;
}

void validate_fetch_shapes(const zkboostSharedValidateForestFetchInstance& instance)
{
    require_condition(instance.view.summary.tree_count == instance.summary.tree_count,
                      "tree_count mismatch");
    require_condition(instance.view.summary.node_count == instance.summary.node_count,
                      "node_count mismatch");
    require_condition(instance.view.summary.feature_count == instance.summary.feature_count,
                      "feature_count mismatch");
    require_condition(instance.view.summary.sample_count == instance.summary.sample_count,
                      "sample_count mismatch");
    require_condition(instance.view.summary.max_depth == instance.summary.max_depth,
                      "max_depth mismatch");
    require_condition(instance.view.summary.checked_paths == instance.summary.checked_paths,
                      "checked_paths mismatch");
    require_condition(instance.view.fetch.split_field_reads.size() ==
                          instance.summary.split_field_read_count,
                      "split_field_reads length mismatch");
    require_condition(instance.view.fetch.path_reads.size() == instance.summary.path_read_count,
                      "path_reads length mismatch");
    require_condition(instance.view.fetch.leaf_value_reads.size() ==
                          instance.summary.leaf_value_read_count,
                      "leaf_value_reads length mismatch");
}

bool verify_opened_forest_reads(const zkboostSharedValidateForestFetchInstance& instance,
                                kzgProtocol& kzg,
                                const forestCommitmentHandles& handles)
{
    const size_t split_domain = instance.summary.split_domain_size;
    const size_t leaf_domain = instance.summary.leaf_domain_size;
    const size_t internal_count = instance.ensemble.internal_node_count_per_tree;
    const size_t leaf_count = instance.ensemble.leaf_count_per_tree;

    for (const zkboostSharedForestSplitFieldReadClaim& split_read : instance.view.fetch.split_field_reads) {
        const size_t row = split_read.tree * internal_count + split_read.internal_node_index;
        const Fr point = split_row_point(split_domain, row);
        kzg.add_opening_check(handles.split_features, point, fr_from_dec(std::to_string(split_read.feature)));
        kzg.add_opening_check(handles.split_thresholds, point, fr_from_dec(std::to_string(split_read.threshold)));
    }

    for (const zkboostSharedForestPathReadClaim& path_read : instance.view.fetch.path_reads) {
        const size_t row = path_read.tree * internal_count + path_read.internal_node_index;
        const Fr point = split_row_point(split_domain, row);
        kzg.add_opening_check(handles.split_features, point, fr_from_dec(std::to_string(path_read.feature)));
        kzg.add_opening_check(handles.split_thresholds, point, fr_from_dec(std::to_string(path_read.threshold)));
    }

    for (const zkboostSharedForestLeafValueReadClaim& leaf_read : instance.view.fetch.leaf_value_reads) {
        require_condition(leaf_read.logical_leaf_id > 0, "logical leaf id must be positive");
        const size_t row = leaf_read.tree * leaf_count +
                           static_cast<size_t>(leaf_read.logical_leaf_id - 1);
        const Fr point = leaf_row_point(leaf_domain, row);
        kzg.add_opening_check(handles.leaf_values, point, fr_from_dec(std::to_string(leaf_read.tree_score)));
    }

    return true;
}

void add_unique_fetch_opening(fetchOpeningSet& set, const Fr& point, const Fr& value)
{
    for (size_t i = 0; i < set.points.size(); ++i) {
        if (set.points[i] != point) {
            continue;
        }
        require_condition(set.values[i] == value, "conflicting fetch opening value");
        return;
    }
    set.points.push_back(point);
    set.values.push_back(value);
}

std::vector<fetchOpeningSet> collect_fetch_opening_sets(
    const zkboostSharedValidateForestFetchInstance& instance)
{
    const size_t split_domain = instance.summary.split_domain_size;
    const size_t leaf_domain = instance.summary.leaf_domain_size;
    const size_t internal_count = instance.ensemble.internal_node_count_per_tree;
    const size_t leaf_count = instance.ensemble.leaf_count_per_tree;

    std::vector<fetchOpeningSet> sets(3);
    sets[0].commitment_index = 0;
    sets[1].commitment_index = 1;
    sets[2].commitment_index = 2;

    for (const zkboostSharedForestSplitFieldReadClaim& split_read : instance.view.fetch.split_field_reads) {
        const size_t row = split_read.tree * internal_count + split_read.internal_node_index;
        const Fr point = split_row_point(split_domain, row);
        add_unique_fetch_opening(sets[0], point, fr_from_dec(std::to_string(split_read.feature)));
        add_unique_fetch_opening(sets[1], point, fr_from_dec(std::to_string(split_read.threshold)));
    }

    for (const zkboostSharedForestPathReadClaim& path_read : instance.view.fetch.path_reads) {
        const size_t row = path_read.tree * internal_count + path_read.internal_node_index;
        const Fr point = split_row_point(split_domain, row);
        add_unique_fetch_opening(sets[0], point, fr_from_dec(std::to_string(path_read.feature)));
        add_unique_fetch_opening(sets[1], point, fr_from_dec(std::to_string(path_read.threshold)));
    }

    for (const zkboostSharedForestLeafValueReadClaim& leaf_read : instance.view.fetch.leaf_value_reads) {
        require_condition(leaf_read.logical_leaf_id > 0, "logical leaf id must be positive");
        const size_t row = leaf_read.tree * leaf_count +
                           static_cast<size_t>(leaf_read.logical_leaf_id - 1);
        const Fr point = leaf_row_point(leaf_domain, row);
        add_unique_fetch_opening(sets[2], point, fr_from_dec(std::to_string(leaf_read.tree_score)));
    }

    std::vector<fetchOpeningSet> out;
    out.reserve(sets.size());
    for (fetchOpeningSet& set : sets) {
        if (!set.points.empty()) {
            out.push_back(std::move(set));
        }
    }
    return out;
}

std::vector<Fr> collect_fetch_union_points(const std::vector<fetchOpeningSet>& sets)
{
    std::vector<Fr> union_points;
    for (const fetchOpeningSet& set : sets) {
        for (const Fr& point : set.points) {
            if (!has_recorded_opening(union_points, point)) {
                union_points.push_back(point);
            }
        }
    }
    return union_points;
}

std::string fetch_opening_group_key(const std::vector<Fr>& union_points,
                                    const std::vector<Fr>& points)
{
    std::string key(union_points.size(), '0');
    for (const Fr& point : points) {
        bool found = false;
        for (size_t i = 0; i < union_points.size(); ++i) {
            if (union_points[i] == point) {
                key[i] = '1';
                found = true;
                break;
            }
        }
        require_condition(found, "fetch opening point missing from union");
    }
    return key;
}

zkboostSharedValidateForestFetchSummary summarize_fetch_instance(
    const zkboostSharedValidateForestView& view,
    const zkboostSharedTreeEnsemble& ensemble)
{
    zkboostSharedValidateForestFetchSummary out;
    out.tree_count = view.summary.tree_count;
    out.node_count = view.summary.node_count;
    out.feature_count = view.summary.feature_count;
    out.sample_count = view.summary.sample_count;
    out.max_depth = view.summary.max_depth;
    out.checked_paths = view.summary.checked_paths;
    out.split_domain_size =
        canonical_padded_zkboost_shared_forest_length(ensemble.split_features.size());
    out.leaf_domain_size =
        canonical_padded_zkboost_shared_forest_length(ensemble.leaf_values.size());
    out.split_field_read_count = view.fetch.split_field_reads.size();
    out.path_read_count = view.fetch.path_reads.size();
    out.leaf_value_read_count = view.fetch.leaf_value_reads.size();
    return out;
}

size_t count_distinct_fetch_opening_points(const zkboostSharedValidateForestView& view,
                                           const zkboostSharedTreeEnsemble& ensemble)
{
    const size_t split_domain =
        canonical_padded_zkboost_shared_forest_length(ensemble.split_features.size());
    const size_t leaf_domain =
        canonical_padded_zkboost_shared_forest_length(ensemble.leaf_values.size());
    const size_t internal_count = ensemble.internal_node_count_per_tree;
    const size_t leaf_count = ensemble.leaf_count_per_tree;

    std::vector<Fr> union_points;
    union_points.reserve(view.fetch.split_field_reads.size() +
                         view.fetch.path_reads.size() +
                         view.fetch.leaf_value_reads.size());

    const auto add_point = [&](const Fr& point) {
        if (!has_recorded_opening(union_points, point)) {
            union_points.push_back(point);
        }
    };

    for (const zkboostSharedForestSplitFieldReadClaim& split_read : view.fetch.split_field_reads) {
        const size_t row = split_read.tree * internal_count + split_read.internal_node_index;
        add_point(split_row_point(split_domain, row));
    }
    for (const zkboostSharedForestPathReadClaim& path_read : view.fetch.path_reads) {
        const size_t row = path_read.tree * internal_count + path_read.internal_node_index;
        add_point(split_row_point(split_domain, row));
    }
    for (const zkboostSharedForestLeafValueReadClaim& leaf_read : view.fetch.leaf_value_reads) {
        require_condition(leaf_read.logical_leaf_id > 0, "logical leaf id must be positive");
        const size_t row = leaf_read.tree * leaf_count +
                           static_cast<size_t>(leaf_read.logical_leaf_id - 1);
        add_point(leaf_row_point(leaf_domain, row));
    }
    return union_points.size();
}

class validateForestPublicInputLayout {
public:
    explicit validateForestPublicInputLayout(const zkboostSharedValidateForestSummary& summary)
    {
        const size_t q = summary.checked_paths;
        const size_t n = summary.node_count;
        const size_t f = summary.feature_count;
        const size_t d = summary.max_depth;
        trace_leaf_offset = 0;
        tree_score_offset = trace_leaf_offset + q;
        split_mask_offset = tree_score_offset + q;
        split_feature_offset = split_mask_offset + n;
        split_threshold_offset = split_feature_offset + n;
        query_input_offset = split_threshold_offset + n;
        logical_leaf_offset = query_input_offset + q * f;
        path_feature_offset = logical_leaf_offset + q;
        path_threshold_offset = path_feature_offset + q * d;
        path_permuted_input_offset = path_threshold_offset + q * d;
        path_branch_offset = path_permuted_input_offset + q * d;
        path_internal_node_offset = path_branch_offset + q * d;
        total_size = path_internal_node_offset + q * d;
    }

    size_t trace_leaf_offset = 0;
    size_t tree_score_offset = 0;
    size_t split_mask_offset = 0;
    size_t split_feature_offset = 0;
    size_t split_threshold_offset = 0;
    size_t query_input_offset = 0;
    size_t logical_leaf_offset = 0;
    size_t path_feature_offset = 0;
    size_t path_threshold_offset = 0;
    size_t path_permuted_input_offset = 0;
    size_t path_branch_offset = 0;
    size_t path_internal_node_offset = 0;
    size_t total_size = 0;
};

Fr link_i64_to_fr(std::int64_t value)
{
    return fr_from_dec(std::to_string(value));
}

bool public_input_equals_i64(const std::vector<Fr>& public_inputs,
                             size_t index,
                             std::int64_t value)
{
    return index < public_inputs.size() && public_inputs[index] == link_i64_to_fr(value);
}

bool link_online_validate_forest_inputs_to_fetch(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedValidateForestFetchInstance& fetch_instance)
{
    const validateForestPublicInputLayout layout(forest_instance.summary);
    const std::vector<Fr>& inputs = forest_instance.public_inputs;
    if (inputs.size() < layout.total_size) {
        return false;
    }

    const zkboostSharedValidateForestView& view = fetch_instance.view;
    const size_t q = forest_instance.summary.checked_paths;
    const size_t n = forest_instance.summary.node_count;
    const size_t f = forest_instance.summary.feature_count;
    const size_t d = forest_instance.summary.max_depth;

    if (view.routing.trace_leaf_indices.size() != q ||
        view.routing.tree_scores.size() != q ||
        view.fetch.public_split_masks.size() != n ||
        view.fetch.public_split_features.size() != n ||
        view.fetch.public_split_thresholds.size() != n ||
        view.routing.public_query_inputs.size() != q * f ||
        view.routing.path_claims.size() != q ||
        view.fetch.path_reads.size() != q * d ||
        view.fetch.leaf_value_reads.size() != q) {
        return false;
    }

    for (size_t query = 0; query < q; ++query) {
        if (!public_input_equals_i64(inputs,
                                     layout.trace_leaf_offset + query,
                                     view.routing.trace_leaf_indices.at(query)) ||
            !public_input_equals_i64(inputs,
                                     layout.tree_score_offset + query,
                                     view.routing.tree_scores.at(query))) {
            return false;
        }
    }

    for (size_t node = 0; node < n; ++node) {
        if (!public_input_equals_i64(inputs,
                                     layout.split_mask_offset + node,
                                     view.fetch.public_split_masks.at(node)) ||
            !public_input_equals_i64(inputs,
                                     layout.split_feature_offset + node,
                                     view.fetch.public_split_features.at(node)) ||
            !public_input_equals_i64(inputs,
                                     layout.split_threshold_offset + node,
                                     view.fetch.public_split_thresholds.at(node))) {
            return false;
        }
    }

    for (const zkboostSharedForestSplitFieldReadClaim& read : view.fetch.split_field_reads) {
        if (read.node >= n ||
            !public_input_equals_i64(inputs, layout.split_feature_offset + read.node, read.feature) ||
            !public_input_equals_i64(inputs, layout.split_threshold_offset + read.node, read.threshold)) {
            return false;
        }
    }

    for (size_t offset = 0; offset < q * f; ++offset) {
        if (!public_input_equals_i64(inputs,
                                     layout.query_input_offset + offset,
                                     view.routing.public_query_inputs.at(offset))) {
            return false;
        }
    }

    for (size_t query = 0; query < q; ++query) {
        const zkboostSharedForestPathClaim& claim = view.routing.path_claims.at(query);
        if (claim.path_features.size() != d ||
            claim.path_thresholds.size() != d ||
            claim.permuted_inputs.size() != d ||
            claim.path_branch_bits.size() != d) {
            return false;
        }
        if (!public_input_equals_i64(inputs,
                                     layout.logical_leaf_offset + query,
                                     claim.logical_leaf_id)) {
            return false;
        }
        std::int64_t expected_internal_node = 0;
        for (size_t step = 0; step < d; ++step) {
            const size_t path_offset = query * d + step;
            if (!public_input_equals_i64(inputs,
                                         layout.path_feature_offset + path_offset,
                                         claim.path_features.at(step)) ||
                !public_input_equals_i64(inputs,
                                         layout.path_threshold_offset + path_offset,
                                         claim.path_thresholds.at(step)) ||
                !public_input_equals_i64(inputs,
                                         layout.path_permuted_input_offset + path_offset,
                                         claim.permuted_inputs.at(step)) ||
                !public_input_equals_i64(inputs,
                                         layout.path_branch_offset + path_offset,
                                         claim.path_branch_bits.at(step))) {
                return false;
            }
            if (!public_input_equals_i64(inputs,
                                         layout.path_internal_node_offset + path_offset,
                                         expected_internal_node)) {
                return false;
            }
            expected_internal_node =
                (2 * expected_internal_node) + 1 + claim.path_branch_bits.at(step);
        }
    }

    for (const zkboostSharedForestPathReadClaim& read : view.fetch.path_reads) {
        if (read.query >= q || read.step >= d) {
            return false;
        }
        const size_t path_offset = read.query * d + read.step;
        if (!public_input_equals_i64(inputs,
                                     layout.logical_leaf_offset + read.query,
                                     read.logical_leaf_id) ||
            !public_input_equals_i64(inputs,
                                     layout.path_feature_offset + path_offset,
                                     read.feature) ||
            !public_input_equals_i64(inputs,
                                     layout.path_threshold_offset + path_offset,
                                     read.threshold)) {
            return false;
        }
        if (!public_input_equals_i64(inputs,
                                     layout.path_internal_node_offset + path_offset,
                                     static_cast<std::int64_t>(read.internal_node_index))) {
            return false;
        }
    }

    for (const zkboostSharedForestLeafValueReadClaim& read : view.fetch.leaf_value_reads) {
        if (read.query >= q ||
            !public_input_equals_i64(inputs,
                                     layout.logical_leaf_offset + read.query,
                                     read.logical_leaf_id) ||
            !public_input_equals_i64(inputs,
                                     layout.tree_score_offset + read.query,
                                     read.tree_score)) {
            return false;
        }
    }

    return true;
}

} // namespace

bool add_zkboost_shared_validate_forest_fetch_to_kzg(
    kzgProtocol& kzg,
    const zkboostSharedValidateForestFetchInstance& instance)
{
    validate_fetch_shapes(instance);
    const std::shared_ptr<ProofSession> session = kzg.proof_session();
    if (!session) {
        throw std::runtime_error(
            "add_zkboost_shared_validate_forest_fetch_to_kzg: missing proof session");
    }
    const bool prover_mode = session->mode() == ProofMode::Prove;
    const forestCommitmentHandles handles =
        commit_or_adopt_forest_commitments(kzg,
                                           instance.ensemble,
                                           instance.commitments,
                                           prover_mode);
    return verify_opened_forest_reads(instance, kzg, handles);
}

zkboostSharedValidateForestFetchInstance build_zkboost_shared_validate_forest_fetch_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters)
{
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    return build_zkboost_shared_validate_forest_fetch_instance_from_view(
        view,
        shared_tree_ensemble_from_training_trace(training),
        public_parameters);
}

zkboostSharedValidateForestFetchInstance build_zkboost_shared_validate_forest_fetch_instance_from_view(
    const zkboostSharedValidateForestView& view,
    const zkboostSharedTreeEnsemble& ensemble,
    const kzgPublicParameters& public_parameters)
{
    return build_zkboost_shared_validate_forest_fetch_instance_from_view(
        view,
        ensemble,
        commit_zkboost_shared_forest(public_parameters, ensemble),
        public_parameters);
}

zkboostSharedValidateForestFetchInstance build_zkboost_shared_validate_forest_fetch_instance_from_view(
    const zkboostSharedValidateForestView& view,
    const zkboostSharedTreeEnsemble& ensemble,
    const zkboostSharedForestCommitments& commitments,
    const kzgPublicParameters& public_parameters)
{
    validate_zkboost_shared_tree_ensemble(ensemble);

    zkboostSharedValidateForestFetchInstance out;
    out.summary = summarize_fetch_instance(view, ensemble);
    out.view = view;
    out.ensemble = ensemble;
    out.commitments = commitments;
    out.linked_input_digest = digest_zkboost_shared_validate_forest_public_inputs(view);
    out.public_parameters = public_parameters;
    return out;
}

zkboostSharedKzgSrsRequirement required_zkboost_shared_validate_forest_fetch_srs_requirement(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    return required_zkboost_shared_validate_forest_fetch_srs_requirement_from_view(
        build_zkboost_shared_validate_forest_view(training, inference),
        shared_tree_ensemble_from_training_trace(training));
}

zkboostSharedKzgSrsRequirement required_zkboost_shared_validate_forest_fetch_srs_requirement_from_view(
    const zkboostSharedValidateForestView& view,
    const zkboostSharedTreeEnsemble& ensemble)
{
    zkboostSharedKzgSrsRequirement out =
        required_zkboost_shared_validate_forest_fetch_srs_requirement_from_ensemble(ensemble);
    out.opening_deg2_plus_1_floor =
        std::max(out.opening_deg2_plus_1_floor, count_distinct_fetch_opening_points(view, ensemble) + 1);
    const size_t degree_shift_requirement =
        out.min_degree_bound_plus_1 == 0 ? size_t(1) : out.deg1_plus_1 - out.min_degree_bound_plus_1 + 1;
    out.deg2_plus_1 = std::max(out.opening_deg2_plus_1_floor, degree_shift_requirement);
    return out;
}

zkboostSharedKzgSrsRequirement required_zkboost_shared_validate_forest_fetch_srs_requirement_from_ensemble(
    const zkboostSharedTreeEnsemble& ensemble)
{
    return required_zkboost_shared_forest_commitment_srs_requirement(ensemble);
}

size_t required_zkboost_shared_validate_forest_fetch_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    return required_zkboost_shared_validate_forest_fetch_srs_requirement(training, inference).deg1_plus_1;
}

size_t required_zkboost_shared_validate_forest_fetch_srs_size(
    const forestTrainingSemanticTrace& training)
{
    return required_zkboost_shared_validate_forest_fetch_srs_requirement_from_ensemble(
               shared_tree_ensemble_from_training_trace(training))
        .deg1_plus_1;
}

size_t required_zkboost_shared_validate_forest_fetch_srs_size_from_ensemble(
    const zkboostSharedTreeEnsemble& ensemble)
{
    return required_zkboost_shared_validate_forest_fetch_srs_requirement_from_ensemble(ensemble).deg1_plus_1;
}

SerializedProof prove_zkboost_shared_validate_forest_fetch(
    const zkboostSharedValidateForestFetchInstance& instance)
{
    validate_fetch_shapes(instance);

    kzgProtocol prover;
    prover.setup(instance.public_parameters);
    const std::shared_ptr<ProofSession> session = ProofSession::prover();
    prover.set_proof_session(session);

    const forestCommitmentHandles handles =
        commit_or_adopt_forest_commitments(prover,
                                           instance.ensemble,
                                           instance.commitments,
                                           true);
    if (!verify_opened_forest_reads(instance, prover, handles)) {
        throw std::runtime_error("prove_zkboost_shared_validate_forest_fetch: opened values do not match logical fetch claims");
    }
    if (!prover.finalize()) {
        throw std::runtime_error("prove_zkboost_shared_validate_forest_fetch: KZG finalize failed");
    }
    return session->take_proof();
}

bool verify_zkboost_shared_validate_forest_fetch(
    const zkboostSharedValidateForestFetchInstance& instance,
    const SerializedProof& proof)
{
    try {
        validate_fetch_shapes(instance);
    } catch (const std::runtime_error&) {
        return false;
    }

    kzgProtocol verifier;
    verifier.setup(instance.public_parameters);
    const std::shared_ptr<ProofSession> session = ProofSession::verifier(proof);
    verifier.set_proof_session(session);

    try {
        const forestCommitmentHandles handles =
            commit_or_adopt_forest_commitments(verifier,
                                               instance.ensemble,
                                               instance.commitments,
                                               false);
        if (!verify_opened_forest_reads(instance, verifier, handles)) {
            return false;
        }
        return verifier.finalize() && session->proof_fully_consumed();
    } catch (const std::exception&) {
        return false;
    }
}

zkboostSharedValidateForestFetchVerifierBundle prepare_zkboost_shared_validate_forest_fetch_verifier_bundle(
    const zkboostSharedValidateForestFetchInstance& instance)
{
    validate_fetch_shapes(instance);

    const size_t split_degree =
        canonical_padded_zkboost_shared_forest_length(instance.ensemble.split_features.size()) - 1;
    const size_t leaf_degree =
        canonical_padded_zkboost_shared_forest_length(instance.ensemble.leaf_values.size()) - 1;

    std::vector<kzgCommitment> commitments(3);
    commitments[0].com = instance.commitments.com_f;
    commitments[0].index = 0;
    commitments[1].com = instance.commitments.com_t;
    commitments[1].index = 1;
    commitments[2].com = instance.commitments.com_w;
    commitments[2].index = 2;
    const std::vector<size_t> degrees = {split_degree, split_degree, leaf_degree};

    ProofTranscript transcript;
    for (size_t i = 0; i < commitments.size(); ++i) {
        transcript.absorb_string("kzg.commitment", transcript_representation(commitments[i]));
        transcript.absorb_string("kzg.degree.commitment", transcript_representation(commitments[i]));
        transcript.absorb_size("kzg.degree.bound", degrees[i]);
    }

    const Fr degree_challenge = transcript.squeeze_fr("kzg.degree.challenge");
    std::map<size_t, Group1> aggregate_by_shift;
    Fr factor(1);
    for (size_t i = 0; i < commitments.size(); ++i) {
        require_condition(degrees[i] <= instance.public_parameters.deg1,
                          "fetch degree bound exceeds SRS degree");
        const size_t shift = instance.public_parameters.deg1 - degrees[i];
        require_condition(shift < instance.public_parameters.g2_powers.size(),
                          "fetch degree proof requires unavailable G2 SRS power");
        aggregate_by_shift[shift] += factor * commitments[i].com;
        Fr::mul(factor, factor, degree_challenge);
    }

    zkboostSharedValidateForestFetchVerifierBundle out;
    for (const auto& [shift, aggregate] : aggregate_by_shift) {
        proof_cost::add("pairing_count");
        out.degree_combined_commitment += pairing(aggregate, instance.public_parameters.g2_powers[shift]);
    }
    out.transcript_prefix = transcript;

    const std::vector<fetchOpeningSet> opening_sets = collect_fetch_opening_sets(instance);
    if (opening_sets.empty()) {
        return out;
    }
    const std::vector<Fr> union_points = collect_fetch_union_points(opening_sets);
    require_condition(!union_points.empty(), "fetch opening union must be non-empty");
    require_condition(instance.public_parameters.g2_powers.size() >= union_points.size() + 1,
                      "fetch opening proof requires unavailable G2 SRS powers");

    const poly common_vanishing = poly_as_vanishing(union_points);
    out.common_vanishing_commitment =
        commit_g2_with_public_parameters(instance.public_parameters, common_vanishing);

    std::map<std::string, std::vector<size_t>> group_indices_by_key;
    out.opening_set_terms.reserve(opening_sets.size());
    for (size_t i = 0; i < opening_sets.size(); ++i) {
        const fetchOpeningSet& set = opening_sets[i];
        Group1 interpolation_commitment;
        if (set.points.size() == 1) {
            interpolation_commitment = set.values[0] * Group1::one();
        } else {
            interpolation_commitment =
                commit_g1_with_public_parameters(
                    instance.public_parameters,
                    poly_from_interpolation(set.points, set.values));
        }
        out.opening_set_terms.push_back(commitments[set.commitment_index].com - interpolation_commitment);
        group_indices_by_key[fetch_opening_group_key(union_points, set.points)].push_back(i);
    }

    out.opening_groups.reserve(group_indices_by_key.size());
    for (const auto& [key, set_indices] : group_indices_by_key) {
        std::vector<Fr> points;
        points.reserve(union_points.size());
        for (size_t i = 0; i < key.size(); ++i) {
            if (key[i] == '1') {
                points.push_back(union_points[i]);
            }
        }
        const poly set_vanishing = poly_as_vanishing(points);
        const auto [auxiliary, remainder] = div(common_vanishing, set_vanishing);
        require_condition(remainder.is_zero(), "fetch common vanishing is not divisible by opening set");

        zkboostSharedValidateForestFetchVerifierOpeningGroup group;
        group.set_indices = set_indices;
        group.auxiliary_commitment =
            commit_g2_with_public_parameters(instance.public_parameters, auxiliary);
        out.opening_groups.push_back(std::move(group));
    }

    return out;
}

bool verify_zkboost_shared_validate_forest_fetch_preprocessed(
    const zkboostSharedValidateForestFetchVerifierBundle& verifier_bundle,
    const SerializedProof& proof)
{
    try {
        auto transcript = std::make_shared<ProofTranscript>(verifier_bundle.transcript_prefix);
        const std::shared_ptr<ProofSession> session = ProofSession::verifier(proof, transcript);

        const Group1 degree_proof = session->exchange_group1("kzg.degree.proof", {});
        proof_cost::add("pairing_count");
        if (pairing(degree_proof, Group2::one()) != verifier_bundle.degree_combined_commitment) {
            return false;
        }

        const size_t proof_count = verifier_bundle.opening_set_terms.empty() ? 0 : 1;
        std::vector<Group1> opening_proofs;
        if (proof_count != 0) {
            const Fr opening_batch_challenge =
                transcript->squeeze_nonzero_fr("kzg.opening.batch.challenge");
            opening_proofs =
                session->exchange_group1_vector("kzg.opening.proof", proof_count, {});

            std::vector<Fr> factors(verifier_bundle.opening_set_terms.size(), Fr(1));
            Fr factor(1);
            for (size_t i = 0; i < factors.size(); ++i) {
                factors[i] = factor;
                Fr::mul(factor, factor, opening_batch_challenge);
            }

            GroupT rhs;
            for (const zkboostSharedValidateForestFetchVerifierOpeningGroup& group :
                 verifier_bundle.opening_groups) {
                Group1 aggregate = Group1::zero();
                for (const size_t set_index : group.set_indices) {
                    if (set_index >= verifier_bundle.opening_set_terms.size()) {
                        return false;
                    }
                    aggregate += factors[set_index] * verifier_bundle.opening_set_terms[set_index];
                }
                proof_cost::add("pairing_count");
                rhs += pairing(aggregate, group.auxiliary_commitment);
            }
            proof_cost::add("pairing_count");
            const GroupT lhs = pairing(opening_proofs[0], verifier_bundle.common_vanishing_commitment);
            if (lhs != rhs) {
                return false;
            }
        }

        return session->proof_fully_consumed();
    } catch (const std::exception&) {
        return false;
    }
}

bool link_zkboost_shared_validate_forest_to_fetch(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedValidateForestFetchInstance& fetch_instance)
{
    if (forest_instance.summary.tree_count != fetch_instance.summary.tree_count ||
        forest_instance.summary.node_count != fetch_instance.summary.node_count ||
        forest_instance.summary.feature_count != fetch_instance.summary.feature_count ||
        forest_instance.summary.sample_count != fetch_instance.summary.sample_count ||
        forest_instance.summary.max_depth != fetch_instance.summary.max_depth ||
        forest_instance.summary.checked_paths != fetch_instance.summary.checked_paths) {
        return false;
    }
    if (forest_instance.public_input_mode ==
            zkboostSharedValidateForestPublicInputMode::OnlinePublicInputs &&
        !forest_instance.public_inputs.empty()) {
        return link_online_validate_forest_inputs_to_fetch(forest_instance, fetch_instance);
    }
    return forest_instance.linked_input_digest == fetch_instance.linked_input_digest;
}
