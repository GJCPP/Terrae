#pragma once

#include <optional>
#include <string>

#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"

// Public metadata for the canonical tree ordering used by the current forest
// inference / prediction workflow.
//
// The expected tree-id layout is the sequential vector
//   [0, 1, ..., T*K - 1]
// interpreted in `(iteration, class)` order. A query-side tree-index vector of
// length `T*K*b` is valid when it is exactly the column-replication of that
// base layout across the `b` sample slots.
class forestTreeLayoutPublicTable {
public:
    std::string key_prefix;
    size_t iteration_count = 0;
    size_t class_count = 0;
    std::optional<Group1> base_layout_commitment;

    size_t tree_count() const
    {
        return iteration_count * class_count;
    }

    forestTreeLayoutPublicTable verifier_view() const
    {
        return *this;
    }
};

forestTreeLayoutPublicTable build_forest_tree_layout_public_table(std::string key_prefix,
                                                                  size_t iteration_count,
                                                                  size_t class_count);

forestTreeLayoutPublicTable build_forest_tree_layout_public_table(const kzgPublicParameters& public_params,
                                                                  std::string key_prefix,
                                                                  size_t iteration_count,
                                                                  size_t class_count);
