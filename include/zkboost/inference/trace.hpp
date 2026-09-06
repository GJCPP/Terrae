#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class zkboostInferenceTrace {
public:
    size_t query_count = 0;
    size_t feature_count = 0;
    size_t tree_count = 0;
    size_t max_depth = 0;

    std::vector<std::int64_t> query_tree_ids;
    std::vector<std::int64_t> binned_inputs;
    std::vector<std::int64_t> leaf_indices;
    std::vector<std::int64_t> path_node_indices;
    std::vector<std::int64_t> path_decisions;
};
