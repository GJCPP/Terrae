#include "forest_split_update.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

void add_boolean_check(zeroProtocol& zero_check, const virCommitment& value, size_t domain_size)
{
    virCommitment boolean_zero = value * (value - virCommitment(Fr(1)));
    zero_check.add_check(boolean_zero, domain_size);
}

} // namespace

void forestSplitUpdateProtocol::add_check(const forestSplitUpdateWitness& witness,
                                          size_t tree_count,
                                          size_t node_count,
                                          size_t feature_count,
                                          size_t full_upper_bound,
                                          size_t bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("forestSplitUpdateProtocol not setup");
    }
    if (params_.forest_structure == nullptr || params_.hist_agg == nullptr || params_.row_sum == nullptr ||
        params_.zero_check == nullptr) {
        throw std::runtime_error("forestSplitUpdateProtocol dependencies not setup");
    }

    validate_matrix_shape(tree_count, 1, "forestSplitUpdateProtocol root indices");
    validate_matrix_shape(node_count, 1, "forestSplitUpdateProtocol node masks");
    validate_matrix_shape(node_count, feature_count, "forestSplitUpdateProtocol feature matrices");
    if (bit_bound == 0) {
        throw std::runtime_error("forestSplitUpdateProtocol: bit_bound must be non-zero");
    }

    witnesses_.push_back(witness);
    checks_.push_back({tree_count, node_count, feature_count, full_upper_bound, bit_bound});
}

bool forestSplitUpdateProtocol::finalize()
{
    begin_finalize("forestSplitUpdateProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestSplitUpdateCheckRecord& record = checks_[i];
        const forestSplitUpdateWitness& witness = witnesses_[i];
        const size_t matrix_domain_size = record.node_count * record.feature_count;
        const virCommitment feature_indices =
            matrix_column_index_expression(record.node_count, record.feature_count);

        params_.forest_structure->add_masked_check(witness.root_indices,
                                                   witness.active_node_mask,
                                                   witness.left_child_indices,
                                                   witness.right_child_indices,
                                                   witness.node_tree_ids,
                                                   witness.feature_selector,
                                                   witness.lower_bounds,
                                                   witness.upper_bounds,
                                                   witness.left_lower,
                                                   witness.left_upper,
                                                   witness.right_lower,
                                                   witness.right_upper,
                                                   witness.thresholds,
                                                   record.tree_count,
                                                   record.node_count,
                                                   record.feature_count,
                                                   record.full_upper_bound,
                                                   record.bit_bound);

        params_.hist_agg->add_check(witness.left_child_indices,
                                    witness.active_node_mask,
                                    witness.left_child_occupancy,
                                    record.node_count,
                                    record.node_count);
        params_.hist_agg->add_check(witness.right_child_indices,
                                    witness.active_node_mask,
                                    witness.right_child_occupancy,
                                    record.node_count,
                                    record.node_count);

        params_.row_sum->add_check(witness.feature_selector * feature_indices,
                                   witness.selected_features,
                                   record.node_count,
                                   record.feature_count);

        add_boolean_check(*params_.zero_check, witness.active_node_mask, record.node_count);
        add_boolean_check(*params_.zero_check, witness.current_leaf_mask, record.node_count);
        add_boolean_check(*params_.zero_check, witness.current_internal_mask, record.node_count);
        add_boolean_check(*params_.zero_check, witness.next_leaf_mask, record.node_count);
        add_boolean_check(*params_.zero_check, witness.next_internal_mask, record.node_count);
        add_boolean_check(*params_.zero_check, witness.next_active_node_mask, record.node_count);

        virCommitment next_active_zero =
            witness.next_active_node_mask - witness.left_child_occupancy - witness.right_child_occupancy;
        params_.zero_check->add_check(next_active_zero, record.node_count);

        virCommitment next_leaf_zero =
            witness.next_leaf_mask - witness.current_leaf_mask + witness.active_node_mask - witness.next_active_node_mask;
        params_.zero_check->add_check(next_leaf_zero, record.node_count);

        virCommitment next_internal_zero =
            witness.next_internal_mask - witness.current_internal_mask - witness.active_node_mask;
        params_.zero_check->add_check(next_internal_zero, record.node_count);

        virCommitment current_role_overlap = witness.current_leaf_mask * witness.current_internal_mask;
        params_.zero_check->add_check(current_role_overlap, record.node_count);

        virCommitment next_role_overlap = witness.next_leaf_mask * witness.next_internal_mask;
        params_.zero_check->add_check(next_role_overlap, record.node_count);

        virCommitment active_subset_current_leaf =
            witness.active_node_mask * (witness.current_leaf_mask - virCommitment(Fr(1)));
        params_.zero_check->add_check(active_subset_current_leaf, record.node_count);

        virCommitment next_active_subset_next_leaf =
            witness.next_active_node_mask * (witness.next_leaf_mask - virCommitment(Fr(1)));
        params_.zero_check->add_check(next_active_subset_next_leaf, record.node_count);

        virCommitment next_active_not_old_leaf = witness.next_active_node_mask * witness.current_leaf_mask;
        params_.zero_check->add_check(next_active_not_old_leaf, record.node_count);

        virCommitment next_active_not_old_internal = witness.next_active_node_mask * witness.current_internal_mask;
        params_.zero_check->add_check(next_active_not_old_internal, record.node_count);

        virCommitment threshold_zero =
            witness.active_node_mask * (witness.thresholds - witness.selected_bins - virCommitment(Fr(1)));
        params_.zero_check->add_check(threshold_zero, record.node_count);

        (void)matrix_domain_size;
    }

    return true;
}
