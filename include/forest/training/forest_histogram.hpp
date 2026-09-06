#pragma once

#include <vector>

#include "hist_agg.hpp"
#include "protocol_lifecycle.hpp"
#include "tree_histogram_consistency.hpp"

using namespace mcl::bn;

class forestHistogramCheckRecord {
public:
    size_t query_domain_size = 0;
    size_t node_count = 0;
    size_t histogram_width = 0;
    size_t channel_count = 0;
    bool uses_parent_mask = false;
    bool parent_mask_certified = false;
};

class forestHistogramChannelBatch {
public:
    std::vector<virCommitment> values;
    std::vector<virCommitment> leaf_histograms;
    std::vector<virCommitment> full_histograms;
};

class forestHistogramParam {
public:
    histAggProtocol* hist_agg = nullptr;
    treeHistogramConsistencyProtocol* tree_histogram_consistency = nullptr;
};

// Thin composition protocol for the current forest histogram story.
//
// It combines:
// - leaf-local histogram construction via flattened bucket addresses
// - full-tree histogram consistency via routed child aggregation
//
// The caller is responsible for proving that the flattened bucket indices are
// derived from valid leaf assignments. This protocol only proves:
// - `H_leaf` aggregates the supplied `(bucket_index, value)` pairs
// - `H_full` decomposes as `H_leaf + H_left + H_right`
//
// The multi-channel overloads let several additive payload channels share the
// same bucket-address and routing witnesses. This matches the common training
// case where count / gradient / Hessian histograms all use the same bucket map.
class forestHistogramProtocol : protected ProtocolLifecycle {
public:
    ~forestHistogramProtocol() noexcept
    {
        auto_finalize("forestHistogramProtocol", [this] { return finalize(); });
    }

    void setup(histAggProtocol& hist_agg, treeHistogramConsistencyProtocol& tree_histogram_consistency)
    {
        params_.hist_agg = &hist_agg;
        params_.tree_histogram_consistency = &tree_histogram_consistency;
        mark_setup();
    }

    void add_check(const virCommitment& flattened_bucket_indices,
                   const virCommitment& values,
                   const virCommitment& leaf_histogram,
                   const virCommitment& full_histogram,
                   const virCommitment& left_child_indices,
                   const virCommitment& right_child_indices,
                   size_t query_domain_size,
                   size_t node_count,
                   size_t histogram_width);

    void add_masked_check(const virCommitment& flattened_bucket_indices,
                          const virCommitment& values,
                          const virCommitment& leaf_histogram,
                          const virCommitment& full_histogram,
                          const virCommitment& parent_mask,
                          const virCommitment& left_child_indices,
                          const virCommitment& right_child_indices,
                          size_t query_domain_size,
                          size_t node_count,
                          size_t histogram_width);

    void add_check(const virCommitment& flattened_bucket_indices,
                   const forestHistogramChannelBatch& channels,
                   const virCommitment& left_child_indices,
                   const virCommitment& right_child_indices,
                   size_t query_domain_size,
                   size_t node_count,
                   size_t histogram_width);

    void add_masked_check(const virCommitment& flattened_bucket_indices,
                          const forestHistogramChannelBatch& channels,
                          const virCommitment& parent_mask,
                          const virCommitment& left_child_indices,
                          const virCommitment& right_child_indices,
                          size_t query_domain_size,
                          size_t node_count,
                          size_t histogram_width);

    void add_masked_check_with_certified_parent_mask(
        const virCommitment& flattened_bucket_indices,
        const forestHistogramChannelBatch& channels,
        const virCommitment& parent_mask,
        const virCommitment& left_child_indices,
        const virCommitment& right_child_indices,
        size_t query_domain_size,
        size_t node_count,
        size_t histogram_width);

    bool finalize();

private:
    forestHistogramParam params_;
    std::vector<forestHistogramCheckRecord> checks_;
    std::vector<virCommitment> flattened_bucket_indices_;
    std::vector<std::vector<virCommitment>> value_channels_;
    std::vector<std::vector<virCommitment>> leaf_histogram_channels_;
    std::vector<std::vector<virCommitment>> full_histogram_channels_;
    std::vector<virCommitment> parent_masks_;
    std::vector<virCommitment> left_child_indices_;
    std::vector<virCommitment> right_child_indices_;
};
