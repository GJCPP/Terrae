#pragma once

#include <vector>

#include "index_routing.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class treeHistogramConsistencyCheckRecord {
public:
    size_t node_count = 0;
    size_t histogram_width = 0;
    size_t channel_count = 0;
    bool uses_parent_mask = false;
    bool enforce_parent_mask_boolean = true;
    std::vector<kzgCommitment> left_histogram_commitments;
    std::vector<kzgCommitment> right_histogram_commitments;
};

class treeHistogramConsistencyParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    indexRoutingProtocol* index_routing = nullptr;
};

class treeHistogramConsistencyVerifier {
public:
    void add_check(const treeHistogramConsistencyCheckRecord& record);
    bool finalize() const;

    std::vector<treeHistogramConsistencyCheckRecord> checks;
    const treeHistogramConsistencyParam* params = nullptr;
};

class treeHistogramConsistencyProver {
public:
    treeHistogramConsistencyCheckRecord add_check(const std::vector<virCommitment>& full_histograms,
                                                 const virCommitment& left_child_indices,
                                                 const virCommitment& right_child_indices,
                                                 const virCommitment* parent_mask,
                                                 size_t node_count,
                                                 size_t histogram_width);

    std::vector<treeHistogramConsistencyCheckRecord> checks;
    const treeHistogramConsistencyParam* params = nullptr;
};

// Proves that a forest-sized full histogram is the sum of:
// - a leaf-local histogram already zero on non-leaf rows, and
// - left/right child histograms routed from the full histogram under a parent mask.
//
// Masked statement:
// - `parent_mask[parent] = 1` marks internal parent rows
// - masked routing derives `left_hist[parent] = H_full[left_child[parent]]`
//   and similarly for the right child
// - inactive rows route zero child histograms
// - the final decomposition is `H_full = H_leaf + left_hist + right_hist`
//
// This yields:
// - internal rows: `H_full[parent] = H_full[left] + H_full[right]`
// - leaf rows: `H_full[row] = H_leaf[row]`
class treeHistogramConsistencyProtocol : protected ProtocolLifecycle {
public:
    ~treeHistogramConsistencyProtocol() noexcept
    {
        auto_finalize("treeHistogramConsistencyProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero_check, indexRoutingProtocol& index_routing)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero_check;
        params_.index_routing = &index_routing;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& full_histogram,
                   const virCommitment& leaf_histogram,
                   const virCommitment& left_child_indices,
                   const virCommitment& right_child_indices,
                   size_t node_count,
                   size_t histogram_width);

    void add_check(const std::vector<virCommitment>& full_histograms,
                   const std::vector<virCommitment>& leaf_histograms,
                   const virCommitment& left_child_indices,
                   const virCommitment& right_child_indices,
                   size_t node_count,
                   size_t histogram_width);

    void add_masked_check(const virCommitment& full_histogram,
                          const virCommitment& leaf_histogram,
                          const virCommitment& parent_mask,
                          const virCommitment& left_child_indices,
                          const virCommitment& right_child_indices,
                          size_t node_count,
                          size_t histogram_width);

    void add_masked_check(const std::vector<virCommitment>& full_histograms,
                          const std::vector<virCommitment>& leaf_histograms,
                          const virCommitment& parent_mask,
                          const virCommitment& left_child_indices,
                          const virCommitment& right_child_indices,
                          size_t node_count,
                          size_t histogram_width);

    void add_masked_check_with_certified_parent_mask(
        const std::vector<virCommitment>& full_histograms,
        const std::vector<virCommitment>& leaf_histograms,
        const virCommitment& parent_mask,
        const virCommitment& left_child_indices,
        const virCommitment& right_child_indices,
        size_t node_count,
        size_t histogram_width);

    bool finalize();

private:
    void add_check_internal(const std::vector<virCommitment>& full_histograms,
                            const std::vector<virCommitment>& leaf_histograms,
                            const virCommitment* parent_mask,
                            const virCommitment& left_child_indices,
                            const virCommitment& right_child_indices,
                            size_t node_count,
                            size_t histogram_width,
                            bool enforce_parent_mask_boolean = true);

    treeHistogramConsistencyParam params_;
    treeHistogramConsistencyProver prover_;
    treeHistogramConsistencyVerifier verifier_;
    std::vector<std::vector<virCommitment>> full_histogram_batches_;
    std::vector<std::vector<virCommitment>> leaf_histogram_batches_;
    std::vector<virCommitment> parent_masks_;
    std::vector<virCommitment> left_child_indices_;
    std::vector<virCommitment> right_child_indices_;
};
