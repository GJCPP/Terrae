#pragma once

#include <vector>

#include "column_replication.hpp"
#include "forest_depth_consistency.hpp"
#include "node_role_consistency.hpp"
#include "parent_child_box_inheritance.hpp"
#include "parent_child_tree_consistency.hpp"
#include "protocol_lifecycle.hpp"
#include "root_box_range.hpp"
#include "root_selector_consistency.hpp"

using namespace mcl::bn;

class forestStructureCheckRecord {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t full_upper_bound = 0;
    size_t bit_bound = 0;
    bool uses_mask = false;
    bool enforce_unique_roles_and_depth = true;
};

class forestStructureParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
    forestDepthConsistencyProtocol* forest_depth_consistency = nullptr;
    rootBoxRangeProtocol* root_box_range = nullptr;
    rootSelectorConsistencyProtocol* root_selector_consistency = nullptr;
    nodeRoleConsistencyProtocol* node_role_consistency = nullptr;
    parentChildBoxInheritanceProtocol* parent_child_box_inheritance = nullptr;
    parentChildTreeConsistencyProtocol* parent_child_tree_consistency = nullptr;
};

// Thin composition protocol for the currently supported witness-side forest
// legality checks.
//
// It combines:
// - root box full-range checks
// - root uniqueness / tree membership checks
// - incoming-role uniqueness across roots / left children / right children
// - root/child depth consistency for acyclicity
// - parent/child box inheritance
// - parent/child same-tree linkage
//
// This is the current closest thing to a reusable `forestStructureProtocol`.
class forestStructureProtocol : protected ProtocolLifecycle {
public:
    ~forestStructureProtocol() noexcept
    {
        auto_finalize("forestStructureProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero_check,
               columnReplicationProtocol& column_replication,
               rootBoxRangeProtocol& root_box_range,
               rootSelectorConsistencyProtocol& root_selector_consistency,
               nodeRoleConsistencyProtocol& node_role_consistency,
               forestDepthConsistencyProtocol& forest_depth_consistency,
               parentChildBoxInheritanceProtocol& parent_child_box_inheritance,
               parentChildTreeConsistencyProtocol& parent_child_tree_consistency)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero_check;
        params_.column_replication = &column_replication;
        params_.forest_depth_consistency = &forest_depth_consistency;
        params_.root_box_range = &root_box_range;
        params_.root_selector_consistency = &root_selector_consistency;
        params_.node_role_consistency = &node_role_consistency;
        params_.parent_child_box_inheritance = &parent_child_box_inheritance;
        params_.parent_child_tree_consistency = &parent_child_tree_consistency;
        mark_setup();
    }

    void add_check(const virCommitment& root_indices,
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
                   size_t bit_bound);

    void add_check(const virCommitment& root_indices,
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
                   size_t bit_bound);

    void add_masked_check(const virCommitment& root_indices,
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
                          size_t bit_bound);

    void add_masked_check(const virCommitment& root_indices,
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
                          size_t bit_bound);

    bool finalize();

private:
    forestStructureParam params_;
    std::vector<forestStructureCheckRecord> checks_;
    std::vector<virCommitment> root_indices_;
    std::vector<virCommitment> parent_masks_;
    std::vector<virCommitment> node_leaf_masks_;
    std::vector<virCommitment> depths_;
    std::vector<virCommitment> left_child_indices_;
    std::vector<virCommitment> right_child_indices_;
    std::vector<virCommitment> node_tree_ids_;
    std::vector<virCommitment> feature_selectors_;
    std::vector<virCommitment> lower_bounds_;
    std::vector<virCommitment> upper_bounds_;
    std::vector<virCommitment> left_lowers_;
    std::vector<virCommitment> left_uppers_;
    std::vector<virCommitment> right_lowers_;
    std::vector<virCommitment> right_uppers_;
    std::vector<virCommitment> thresholds_;
    std::vector<kzgCommitment> lifted_valid_masks_;
};
