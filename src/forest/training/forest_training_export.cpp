#include "forest_training_export.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>
#include <vector>

namespace {

void validate_committed_forest_shape(const forestCommittedNodeTable& committed_forest)
{
    if (committed_forest.tree_count == 0 || committed_forest.node_count == 0 || committed_forest.feature_count == 0) {
        throw std::runtime_error(
            "export_leaf_table_from_committed_forest: tree_count, node_count, and feature_count must be non-zero");
    }
    validate_matrix_shape(committed_forest.tree_count, 1, "export_leaf_table_from_committed_forest root indices");
    validate_matrix_shape(committed_forest.node_count, 1, "export_leaf_table_from_committed_forest node vectors");
    validate_matrix_shape(committed_forest.node_count, committed_forest.feature_count,
                          "export_leaf_table_from_committed_forest feature selector");
    validate_matrix_shape(
        committed_forest.node_count, committed_forest.feature_count, "export_leaf_table_from_committed_forest node boxes");
}

size_t next_power_of_two(size_t value)
{
    size_t out = 1;
    while (out < value) {
        out <<= 1;
    }
    return out;
}

poly canonical_tree_id_polynomial(size_t tree_count)
{
    std::vector<Fr> evals;
    evals.reserve(tree_count);
    for (size_t tree = 0; tree < tree_count; ++tree) {
        evals.push_back(field_from_size_t(tree + 1));
    }
    return poly_from_evals(evals);
}

} // namespace

forestCommittedNodeTable forest_committed_node_table_from_training_inputs(const forestTrainingProofInputs& inputs)
{
    forestCommittedNodeTable out;
    out.root_indices = inputs.root_indices;
    out.split_node_mask = inputs.split_node_mask;
    out.node_depths = inputs.node_depths;
    out.left_child_indices = inputs.left_child_indices;
    out.right_child_indices = inputs.right_child_indices;
    out.node_tree_ids = inputs.training_node_tree_ids;
    out.feature_selector = inputs.feature_selector;
    out.lower_bounds = inputs.training_node_lower_bounds;
    out.upper_bounds = inputs.training_node_upper_bounds;
    out.thresholds = inputs.thresholds;
    out.node_leaf_mask = inputs.node_leaf_mask;
    out.node_leaf_values = inputs.node_leaf_values;
    out.tree_count = inputs.tree_count;
    out.node_count = inputs.node_count;
    out.feature_count = inputs.feature_count;
    return out;
}

kzgSrsRequirement required_committed_forest_public_commitments_srs(
    const forestCommittedNodeTable& committed_forest)
{
    validate_committed_forest_shape(committed_forest);

    kzgSrsRequirement requirement;
    update_kzg_srs_requirement(requirement, committed_forest.tree_count);
    update_kzg_srs_requirement(requirement, committed_forest.node_count);
    update_kzg_srs_requirement(requirement, committed_forest.node_count * committed_forest.feature_count);
    return requirement;
}

kzgSrsRequirement required_committed_forest_public_commitments_srs(
    const forestTrainingProofInputs& inputs)
{
    return required_committed_forest_public_commitments_srs(
        forest_committed_node_table_from_training_inputs(inputs));
}

forestCommittedNodePublicCommitments commit_committed_forest_public_commitments(
    const kzgPublicParameters& public_params,
    const forestCommittedNodeTable& committed_forest,
    const std::string& key_prefix)
{
    validate_committed_forest_shape(committed_forest);

    kzgProtocol kzg;
    kzg.setup(public_params);

    forestCommittedNodePublicCommitments out;
    out.root_indices =
        kzg.commit_public(key_prefix + ".root_indices", committed_forest.root_indices, committed_forest.tree_count - 1).com;
    out.expected_tree_ids =
        kzg.commit_public(key_prefix + ".expected_tree_ids",
                          canonical_tree_id_polynomial(committed_forest.tree_count),
                          committed_forest.tree_count - 1)
            .com;
    out.split_node_mask =
        kzg.commit_public(key_prefix + ".split_node_mask", committed_forest.split_node_mask, committed_forest.node_count - 1).com;
    out.node_depths =
        kzg.commit_public(key_prefix + ".node_depths", committed_forest.node_depths, committed_forest.node_count - 1).com;
    out.left_child_indices =
        kzg.commit_public(key_prefix + ".left_child_indices", committed_forest.left_child_indices, committed_forest.node_count - 1).com;
    out.right_child_indices =
        kzg.commit_public(key_prefix + ".right_child_indices", committed_forest.right_child_indices, committed_forest.node_count - 1).com;
    out.node_tree_ids =
        kzg.commit_public(key_prefix + ".node_tree_ids", committed_forest.node_tree_ids, committed_forest.node_count - 1).com;
    out.feature_selector = kzg.commit_public(key_prefix + ".feature_selector",
                                             committed_forest.feature_selector,
                                             committed_forest.node_count * committed_forest.feature_count - 1)
                               .com;
    out.lower_bounds = kzg.commit_public(key_prefix + ".lower_bounds",
                                         committed_forest.lower_bounds,
                                         committed_forest.node_count * committed_forest.feature_count - 1)
                           .com;
    out.upper_bounds = kzg.commit_public(key_prefix + ".upper_bounds",
                                         committed_forest.upper_bounds,
                                         committed_forest.node_count * committed_forest.feature_count - 1)
                           .com;
    out.thresholds =
        kzg.commit_public(key_prefix + ".thresholds", committed_forest.thresholds, committed_forest.node_count - 1).com;
    out.node_leaf_mask =
        kzg.commit_public(key_prefix + ".node_leaf_mask", committed_forest.node_leaf_mask, committed_forest.node_count - 1).com;
    out.node_leaf_values =
        kzg.commit_public(key_prefix + ".node_leaf_values", committed_forest.node_leaf_values, committed_forest.node_count - 1).com;
    out.tree_count = committed_forest.tree_count;
    out.node_count = committed_forest.node_count;
    out.feature_count = committed_forest.feature_count;
    return out;
}

forestExportedLeafTable export_leaf_table_from_committed_forest(const forestCommittedNodeTable& committed_forest)
{
    validate_committed_forest_shape(committed_forest);

    const std::vector<Fr> tree_evals = committed_forest.node_tree_ids.evaluations_on_domain(committed_forest.node_count);
    const std::vector<Fr> lower_evals =
        committed_forest.lower_bounds.evaluations_on_domain(committed_forest.node_count * committed_forest.feature_count);
    const std::vector<Fr> upper_evals =
        committed_forest.upper_bounds.evaluations_on_domain(committed_forest.node_count * committed_forest.feature_count);
    const std::vector<Fr> leaf_mask_evals =
        committed_forest.node_leaf_mask.evaluations_on_domain(committed_forest.node_count);
    const std::vector<Fr> leaf_value_evals =
        committed_forest.node_leaf_values.evaluations_on_domain(committed_forest.node_count);

    std::vector<Fr> exported_tree_evals;
    std::vector<Fr> exported_lower_evals;
    std::vector<Fr> exported_upper_evals;
    std::vector<Fr> exported_score_evals;
    std::vector<Fr> exported_leaf_index_evals;

    for (size_t row = 0; row < committed_forest.node_count; ++row) {
        const size_t leaf_mask = canonical_size_t_from_fr(
            leaf_mask_evals[row], "export_leaf_table_from_committed_forest leaf mask");
        if (leaf_mask > 1) {
            throw std::runtime_error("export_leaf_table_from_committed_forest: leaf mask must be boolean");
        }
        if (leaf_mask == 0) {
            continue;
        }

        exported_tree_evals.push_back(tree_evals[row]);
        exported_score_evals.push_back(leaf_value_evals[row]);
        exported_leaf_index_evals.push_back(field_from_size_t(row));
        for (size_t feature = 0; feature < committed_forest.feature_count; ++feature) {
            const size_t index = row * committed_forest.feature_count + feature;
            exported_lower_evals.push_back(lower_evals[index]);
            exported_upper_evals.push_back(upper_evals[index]);
        }
    }

    if (exported_tree_evals.empty()) {
        throw std::runtime_error("export_leaf_table_from_committed_forest: forest has no leaf rows");
    }
    const size_t padded_row_count = next_power_of_two(exported_tree_evals.size());
    if (padded_row_count != exported_tree_evals.size()) {
        const Fr dummy_tree_id = Fr(0);
        const Fr zero = Fr(0);
        size_t next_dummy_leaf_index = committed_forest.node_count;
        while (exported_tree_evals.size() < padded_row_count) {
            exported_tree_evals.push_back(dummy_tree_id);
            exported_score_evals.push_back(zero);
            exported_leaf_index_evals.push_back(field_from_size_t(next_dummy_leaf_index++));
            for (size_t feature = 0; feature < committed_forest.feature_count; ++feature) {
                (void)feature;
                exported_lower_evals.push_back(zero);
                exported_upper_evals.push_back(zero);
            }
        }
    }

    forestExportedLeafTable out;
    out.tree_indices = poly_from_evals(exported_tree_evals);
    out.lower_bounds = poly_from_evals(exported_lower_evals);
    out.upper_bounds = poly_from_evals(exported_upper_evals);
    out.scores = poly_from_evals(exported_score_evals);
    out.leaf_indices = poly_from_evals(exported_leaf_index_evals);
    out.row_count = exported_tree_evals.size();
    out.feature_count = committed_forest.feature_count;
    return out;
}

forestInferencePublicTable build_forest_inference_public_table_from_exported_forest(
    const kzgPublicParameters& public_params,
    std::string key_prefix,
    const forestExportedLeafTable& table,
    size_t input_bit_bound,
    std::optional<size_t> degree_bound)
{
    return build_forest_inference_public_table(public_params,
                                               std::move(key_prefix),
                                               table.tree_indices,
                                               table.lower_bounds,
                                               table.upper_bounds,
                                               table.scores,
                                               table.row_count,
                                               table.feature_count,
                                               input_bit_bound,
                                               table.leaf_indices,
                                               degree_bound);
}

forestClassificationPublicTable build_forest_classification_public_table_from_exported_forest(
    const kzgPublicParameters& public_params,
    std::string key_prefix,
    const forestExportedLeafTable& table,
    size_t input_bit_bound,
    size_t iteration_count,
    size_t class_count,
    size_t score_bit_bound,
    std::optional<size_t> degree_bound)
{
    return build_forest_classification_public_table(public_params,
                                                    std::move(key_prefix),
                                                    table.tree_indices,
                                                    table.lower_bounds,
                                                    table.upper_bounds,
                                                    table.scores,
                                                    table.row_count,
                                                    table.feature_count,
                                                    input_bit_bound,
                                                    iteration_count,
                                                    class_count,
                                                    score_bit_bound,
                                                    table.leaf_indices,
                                                    degree_bound);
}

namespace {

cqTableLookupPublicTable make_counting_cq_table(std::string key,
                                                size_t row_count,
                                                size_t column_count,
                                                std::optional<size_t> degree_bound)
{
    cqTableLookupPublicTable out;
    out.key = std::move(key);
    out.row_count = row_count;
    out.column_count = column_count;
    out.domain_size = row_count * column_count;
    out.degree_bound = degree_bound;
    return out;
}

cqConcatenatedTableLookupPublicTable make_counting_concatenated_table(
    std::string key,
    size_t row_count,
    const std::vector<size_t>& column_counts,
    std::optional<size_t> degree_bound)
{
    cqConcatenatedTableLookupPublicTable out;
    out.row_count = row_count;
    out.column_counts = column_counts;
    out.merged_column_count = cq_concatenated_table_lookup_merged_width(column_counts);
    out.column_offsets =
        cq_concatenated_table_lookup_offsets(out.merged_column_count, column_counts);
    out.merged_table =
        make_counting_cq_table(std::move(key), row_count, out.merged_column_count, degree_bound);
    return out;
}

} // namespace

forestClassificationPublicTable build_forest_classification_public_table_metadata_from_exported_forest(
    std::string key_prefix,
    const forestExportedLeafTable& table,
    size_t input_bit_bound,
    size_t iteration_count,
    size_t class_count,
    size_t score_bit_bound,
    std::optional<size_t> degree_bound)
{
    if (table.row_count == 0 || table.feature_count == 0) {
        throw std::runtime_error(
            "build_forest_classification_public_table_metadata_from_exported_forest: empty table shape");
    }

    forestClassificationPublicTable out;
    out.layout_table =
        build_forest_tree_layout_public_table(key_prefix + ".layout", iteration_count, class_count);
    // The normal public builder commits this fixed layout. Presence of the
    // optional value makes the counter adopt the same public commitment.
    out.layout_table.base_layout_commitment = Group1::zero();

    out.legality_table.tree_count = iteration_count * class_count;
    out.legality_table.box_consistency_table.row_count = table.row_count;
    out.legality_table.box_consistency_table.feature_count = table.feature_count;
    out.legality_table.box_consistency_table.includes_leaf_index = true;

    forestRowFetchPublicTable& fetch = out.inference_table.row_fetch_table;
    fetch.row_count = table.row_count;
    fetch.feature_count = table.feature_count;
    fetch.includes_leaf_index = true;
    const std::string fetch_key = key_prefix + ".inference.step.fetch";
    fetch.lookup_table = make_counting_concatenated_table(
        fetch_key,
        table.row_count,
        {1, table.feature_count, table.feature_count, 1, 1},
        degree_bound);
    fetch.lower_lookup_table = make_counting_cq_table(
        fetch_key + ".lower", table.row_count, table.feature_count, degree_bound);
    fetch.upper_lookup_table = make_counting_cq_table(
        fetch_key + ".upper", table.row_count, table.feature_count, degree_bound);
    fetch.scalar_lookup_table = make_counting_concatenated_table(
        fetch_key + ".scalar", table.row_count, {1, 1, 1}, degree_bound);

    if (table.row_count >= 2) {
        forestCompressedRowLookupPublicTable compressed;
        compressed.key = fetch_key + ".compressed_row";
        compressed.row_count = table.row_count;
        compressed.feature_count = table.feature_count;
        compressed.degree_bound = degree_bound;
        compressed.components.reserve(3 + 2 * table.feature_count);
        const auto add_component = [&](const std::string& suffix) {
            forestCompressedRowComponent component;
            component.key = compressed.key + suffix;
            compressed.components.push_back(std::move(component));
        };
        add_component(".tree");
        add_component(".score");
        add_component(".leaf");
        for (size_t column = 0; column < table.feature_count; ++column) {
            add_component(".lower." + std::to_string(column));
        }
        for (size_t column = 0; column < table.feature_count; ++column) {
            add_component(".upper." + std::to_string(column));
        }
        fetch.compressed_row_lookup_table = std::move(compressed);
    }

    out.inference_table.input_bit_bound = input_bit_bound;
    out.prediction_table =
        build_forest_prediction_public_table(iteration_count, class_count, score_bit_bound);
    return out;
}
