#include "forest_structure.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

void forestStructureProtocol::add_check(const virCommitment& root_indices,
                                        const virCommitment& node_leaf_mask,
                                        const virCommitment& depths,
                                        const virCommitment& left_child_indices,
                                        const virCommitment& right_child_indices,
                                        const virCommitment& node_tree_ids,
                                        const virCommitment& feature_selector,
                                        const virCommitment& lower_bounds,
                                        const virCommitment& upper_bounds,
                                        const virCommitment& left_lower,
                                        const virCommitment& left_upper,
                                        const virCommitment& right_lower,
                                        const virCommitment& right_upper,
                                        const virCommitment& thresholds,
                                        size_t tree_count,
                                        size_t node_count,
                                        size_t feature_count,
                                        size_t full_upper_bound,
                                        size_t bit_bound)
{
    // The unmasked API is just the masked API with an all-ones parent mask.
    add_masked_check(root_indices,
                     virCommitment(Fr(1)),
                     node_leaf_mask,
                     depths,
                     left_child_indices,
                     right_child_indices,
                     node_tree_ids,
                     feature_selector,
                     lower_bounds,
                     upper_bounds,
                     left_lower,
                     left_upper,
                     right_lower,
                     right_upper,
                     thresholds,
                     tree_count,
                     node_count,
                     feature_count,
                     full_upper_bound,
                     bit_bound);
    checks_.back().uses_mask = false;
}

void forestStructureProtocol::add_check(const virCommitment& root_indices,
                                        const virCommitment& left_child_indices,
                                        const virCommitment& right_child_indices,
                                        const virCommitment& node_tree_ids,
                                        const virCommitment& feature_selector,
                                        const virCommitment& lower_bounds,
                                        const virCommitment& upper_bounds,
                                        const virCommitment& left_lower,
                                        const virCommitment& left_upper,
                                        const virCommitment& right_lower,
                                        const virCommitment& right_upper,
                                        const virCommitment& thresholds,
                                        size_t tree_count,
                                        size_t node_count,
                                        size_t feature_count,
                                        size_t full_upper_bound,
                                        size_t bit_bound)
{
    add_masked_check(root_indices,
                     virCommitment(Fr(1)),
                     left_child_indices,
                     right_child_indices,
                     node_tree_ids,
                     feature_selector,
                     lower_bounds,
                     upper_bounds,
                     left_lower,
                     left_upper,
                     right_lower,
                     right_upper,
                     thresholds,
                     tree_count,
                     node_count,
                     feature_count,
                     full_upper_bound,
                     bit_bound);
    checks_.back().uses_mask = false;
}

void forestStructureProtocol::add_masked_check(const virCommitment& root_indices,
                                               const virCommitment& parent_mask,
                                               const virCommitment& node_leaf_mask,
                                               const virCommitment& depths,
                                               const virCommitment& left_child_indices,
                                               const virCommitment& right_child_indices,
                                               const virCommitment& node_tree_ids,
                                               const virCommitment& feature_selector,
                                               const virCommitment& lower_bounds,
                                               const virCommitment& upper_bounds,
                                               const virCommitment& left_lower,
                                               const virCommitment& left_upper,
                                               const virCommitment& right_lower,
                                               const virCommitment& right_upper,
                                               const virCommitment& thresholds,
                                               size_t tree_count,
                                               size_t node_count,
                                               size_t feature_count,
                                               size_t full_upper_bound,
                                               size_t bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("forestStructureProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.column_replication == nullptr ||
        params_.forest_depth_consistency == nullptr || params_.root_box_range == nullptr ||
        params_.root_selector_consistency == nullptr ||
        params_.node_role_consistency == nullptr || params_.parent_child_box_inheritance == nullptr ||
        params_.parent_child_tree_consistency == nullptr) {
        throw std::runtime_error("forestStructureProtocol dependencies not setup");
    }

    validate_matrix_shape(tree_count, 1, "forestStructureProtocol root indices");
    validate_matrix_shape(node_count, 1, "forestStructureProtocol child indices");
    validate_matrix_shape(node_count, feature_count, "forestStructureProtocol node boxes");
    if (full_upper_bound == 0 || bit_bound == 0) {
        throw std::runtime_error("forestStructureProtocol: full_upper_bound and bit_bound must be non-zero");
    }

    checks_.push_back({tree_count, node_count, feature_count, full_upper_bound, bit_bound, true, true});
    root_indices_.push_back(root_indices);
    parent_masks_.push_back(parent_mask);
    node_leaf_masks_.push_back(node_leaf_mask);
    depths_.push_back(depths);
    left_child_indices_.push_back(left_child_indices);
    right_child_indices_.push_back(right_child_indices);
    node_tree_ids_.push_back(node_tree_ids);
    feature_selectors_.push_back(feature_selector);
    lower_bounds_.push_back(lower_bounds);
    upper_bounds_.push_back(upper_bounds);
    left_lowers_.push_back(left_lower);
    left_uppers_.push_back(left_upper);
    right_lowers_.push_back(right_lower);
    right_uppers_.push_back(right_upper);
    thresholds_.push_back(thresholds);

    const virCommitment valid = parent_mask + node_leaf_mask;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    kzgCommitment lifted_valid;
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        const Group1 group = proof_session->transport_group1({});
        lifted_valid = params_.kzg->adopt_commitment(group);
        params_.kzg->prove_degree(lifted_valid, node_count * feature_count - 1);
    } else {
        const std::shared_ptr<poly> valid_poly = params_.kzg->materialize(valid);
        if (valid_poly == nullptr) {
            throw std::runtime_error("forestStructureProtocol: failed to materialize valid-node mask");
        }
        lifted_valid = params_.kzg->commit(
            valid_poly->column_replicated(node_count, feature_count), node_count * feature_count - 1);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            const Group1 group = proof_session->transport_group1([&] { return lifted_valid.com; });
            if (group != lifted_valid.com) {
                throw std::runtime_error(
                    "forestStructureProtocol: transported lifted-valid commitment mismatch");
            }
        }
    }
    lifted_valid_masks_.push_back(lifted_valid);
}

void forestStructureProtocol::add_masked_check(const virCommitment& root_indices,
                                               const virCommitment& parent_mask,
                                               const virCommitment& left_child_indices,
                                               const virCommitment& right_child_indices,
                                               const virCommitment& node_tree_ids,
                                               const virCommitment& feature_selector,
                                               const virCommitment& lower_bounds,
                                               const virCommitment& upper_bounds,
                                               const virCommitment& left_lower,
                                               const virCommitment& left_upper,
                                               const virCommitment& right_lower,
                                               const virCommitment& right_upper,
                                               const virCommitment& thresholds,
                                               size_t tree_count,
                                               size_t node_count,
                                               size_t feature_count,
                                               size_t full_upper_bound,
                                               size_t bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("forestStructureProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.column_replication == nullptr ||
        params_.forest_depth_consistency == nullptr || params_.root_box_range == nullptr ||
        params_.root_selector_consistency == nullptr ||
        params_.node_role_consistency == nullptr || params_.parent_child_box_inheritance == nullptr ||
        params_.parent_child_tree_consistency == nullptr) {
        throw std::runtime_error("forestStructureProtocol dependencies not setup");
    }

    validate_matrix_shape(tree_count, 1, "forestStructureProtocol root indices");
    validate_matrix_shape(node_count, 1, "forestStructureProtocol child indices");
    validate_matrix_shape(node_count, feature_count, "forestStructureProtocol node boxes");
    if (full_upper_bound == 0 || bit_bound == 0) {
        throw std::runtime_error("forestStructureProtocol: full_upper_bound and bit_bound must be non-zero");
    }

    checks_.push_back({tree_count, node_count, feature_count, full_upper_bound, bit_bound, true, false});
    root_indices_.push_back(root_indices);
    parent_masks_.push_back(parent_mask);
    node_leaf_masks_.push_back(virCommitment(Fr(0)));
    depths_.push_back(virCommitment(Fr(0)));
    left_child_indices_.push_back(left_child_indices);
    right_child_indices_.push_back(right_child_indices);
    node_tree_ids_.push_back(node_tree_ids);
    feature_selectors_.push_back(feature_selector);
    lower_bounds_.push_back(lower_bounds);
    upper_bounds_.push_back(upper_bounds);
    left_lowers_.push_back(left_lower);
    left_uppers_.push_back(left_upper);
    right_lowers_.push_back(right_lower);
    right_uppers_.push_back(right_upper);
    thresholds_.push_back(thresholds);
    lifted_valid_masks_.push_back({});
}

bool forestStructureProtocol::finalize()
{
    begin_finalize("forestStructureProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestStructureCheckRecord& record = checks_[i];

        auto add_zero = [this](virCommitment expression, size_t domain_size, const std::string& label) {
            params_.zero_check->add_check(expression, domain_size, label);
        };

        if (record.enforce_unique_roles_and_depth) {
            add_zero(node_leaf_masks_[i] * left_child_indices_[i],
                     record.node_count,
                     "forest_structure.leaf_no_left_child");
            add_zero(node_leaf_masks_[i] * right_child_indices_[i],
                     record.node_count,
                     "forest_structure.leaf_no_right_child");
        }

        if (record.uses_mask && record.enforce_unique_roles_and_depth) {
            const virCommitment valid = parent_masks_[i] + node_leaf_masks_[i];
            const virCommitment invalid = virCommitment(Fr(1)) - valid;
            const virCommitment lifted_valid(lifted_valid_masks_[i]);
            const virCommitment lifted_invalid = virCommitment(Fr(1)) - lifted_valid;

            params_.column_replication->add_check(
                valid, lifted_valid, record.node_count, record.feature_count);
            add_zero(invalid * left_child_indices_[i], record.node_count, "forest_structure.no_ghost.left_child");
            add_zero(invalid * right_child_indices_[i], record.node_count, "forest_structure.no_ghost.right_child");
            add_zero(invalid * node_tree_ids_[i], record.node_count, "forest_structure.no_ghost.tree_id");
            add_zero(invalid * depths_[i], record.node_count, "forest_structure.no_ghost.depth");
            add_zero(invalid * thresholds_[i], record.node_count, "forest_structure.no_ghost.threshold");
            add_zero(lifted_invalid * feature_selectors_[i],
                     record.node_count * record.feature_count,
                     "forest_structure.no_ghost.feature_selector");
            add_zero(lifted_invalid * lower_bounds_[i],
                     record.node_count * record.feature_count,
                     "forest_structure.no_ghost.lower_bounds");
            add_zero(lifted_invalid * upper_bounds_[i],
                     record.node_count * record.feature_count,
                     "forest_structure.no_ghost.upper_bounds");
        }

        // Root legality and tree membership.
        params_.root_box_range->add_check(root_indices_[i],
                                          lower_bounds_[i],
                                          upper_bounds_[i],
                                          record.tree_count,
                                          record.node_count,
                                          record.feature_count,
                                          record.full_upper_bound);
        params_.root_selector_consistency->add_check(
            root_indices_[i], node_tree_ids_[i], record.tree_count, record.node_count);

        if (record.uses_mask) {
            // For inactive parents, all child- and split-related obligations are
            // routed through the masked variants.
            if (record.enforce_unique_roles_and_depth) {
                params_.node_role_consistency->add_masked_check(root_indices_[i],
                                                                node_leaf_masks_[i],
                                                                left_child_indices_[i],
                                                                right_child_indices_[i],
                                                                parent_masks_[i],
                                                                record.tree_count,
                                                                record.node_count);
                params_.forest_depth_consistency->add_check(root_indices_[i],
                                                            parent_masks_[i],
                                                            left_child_indices_[i],
                                                            right_child_indices_[i],
                                                            depths_[i],
                                                            record.tree_count,
                                                            record.node_count);
            }
            params_.parent_child_box_inheritance->add_masked_check(feature_selectors_[i],
                                                                   parent_masks_[i],
                                                                   left_child_indices_[i],
                                                                   right_child_indices_[i],
                                                                   lower_bounds_[i],
                                                                   upper_bounds_[i],
                                                                   left_lowers_[i],
                                                                   left_uppers_[i],
                                                                   right_lowers_[i],
                                                                   right_uppers_[i],
                                                                   thresholds_[i],
                                                                   record.node_count,
                                                                   record.feature_count,
                                                                   record.bit_bound);
            params_.parent_child_tree_consistency->add_masked_check(left_child_indices_[i],
                                                                    right_child_indices_[i],
                                                                    parent_masks_[i],
                                                                    node_tree_ids_[i],
                                                                    record.node_count);
            continue;
        }

        // Active-only structure path.
        if (record.enforce_unique_roles_and_depth) {
            params_.node_role_consistency->add_check(
                root_indices_[i],
                node_leaf_masks_[i],
                left_child_indices_[i],
                right_child_indices_[i],
                record.tree_count,
                record.node_count);
            params_.forest_depth_consistency->add_check(root_indices_[i],
                                                        virCommitment(Fr(1)),
                                                        left_child_indices_[i],
                                                        right_child_indices_[i],
                                                        depths_[i],
                                                        record.tree_count,
                                                        record.node_count);
        }
        params_.parent_child_box_inheritance->add_check(feature_selectors_[i],
                                                        left_child_indices_[i],
                                                        right_child_indices_[i],
                                                        lower_bounds_[i],
                                                        upper_bounds_[i],
                                                        left_lowers_[i],
                                                        left_uppers_[i],
                                                        right_lowers_[i],
                                                        right_uppers_[i],
                                                        thresholds_[i],
                                                        record.node_count,
                                                        record.feature_count,
                                                        record.bit_bound);
        params_.parent_child_tree_consistency->add_check(
            left_child_indices_[i], right_child_indices_[i], node_tree_ids_[i], record.node_count);
    }

    return true;
}
