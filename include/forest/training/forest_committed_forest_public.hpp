#pragma once

#include <cstddef>

#include "bls12_381_wrapper.hpp"

class forestCommittedNodePublicCommitments {
public:
    Group1 root_indices;
    Group1 expected_tree_ids;
    Group1 split_node_mask;
    Group1 node_depths;
    Group1 left_child_indices;
    Group1 right_child_indices;
    Group1 node_tree_ids;
    Group1 feature_selector;
    Group1 lower_bounds;
    Group1 upper_bounds;
    Group1 thresholds;
    Group1 node_leaf_mask;
    Group1 node_leaf_values;
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
};
