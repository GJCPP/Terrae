#pragma once

#include <string_view>
#include <vector>

#include "kzg.hpp"

struct InterleavedBatchNode {
    std::vector<virCommitment> commitments;
    size_t domain_size = 0;
};

virCommitment interleave_commitments(const virCommitment& even_values,
                                     const virCommitment& odd_values,
                                     size_t domain_size,
                                     std::string_view protocol_name);

InterleavedBatchNode merge_interleaved_nodes(const std::vector<InterleavedBatchNode>& leaves,
                                             std::string_view protocol_name);

size_t merged_interleaved_domain_size(const std::vector<size_t>& leaf_domain_sizes,
                                      std::string_view protocol_name);
