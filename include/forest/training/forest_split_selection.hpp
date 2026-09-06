#pragma once

#include "masked_argmax.hpp"
#include "matrix_sum.hpp"
#include "nonzero_check.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestSplitSelectionWitness {
public:
    virCommitment quantized_gains;
    virCommitment tie_break_scores;
    virCommitment candidate_mask;
    virCommitment left_counts;
    virCommitment right_counts;
    virCommitment active_node_mask;
    virCommitment selector;
    virCommitment selected_gains;
    virCommitment selected_tie_break_scores;
    virCommitment selected_candidate_indices;
    virCommitment selected_features;
    virCommitment selected_bins;
};

class forestSplitSelectionCheckRecord {
public:
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t query_count = 0;
    size_t gain_gap_bit_bound = 0;
};

class forestSplitSelectionParam {
public:
    maskedArgmaxProtocol* masked_argmax = nullptr;
    matrixRowSumProtocol* row_sum = nullptr;
    zeroProtocol* zero_check = nullptr;
    lessCheckProtocol* less_check = nullptr;
    nonzeroCheckProtocol* nonzero_check = nullptr;
};

// Proves masked split selection over the `(node, feature, bin)` candidate grid.
//
// The protocol proves:
// - one active candidate is selected per active node row,
// - the selected candidate maximizes a deterministic tie-break score derived
//   from the quantized gain plus the reversed candidate index,
// - the selected flattened candidate index is derived from the selector,
// - and that flattened index decomposes into `(selected_feature, selected_bin)`.
class forestSplitSelectionProtocol : protected ProtocolLifecycle {
public:
    ~forestSplitSelectionProtocol() noexcept
    {
        auto_finalize("forestSplitSelectionProtocol", [this] { return finalize(); });
    }

    void setup(maskedArgmaxProtocol& masked_argmax,
               matrixRowSumProtocol& row_sum,
               zeroProtocol& zero_check,
               lessCheckProtocol& less_check)
    {
        setup(masked_argmax, row_sum, zero_check, less_check, nullptr);
    }

    void setup(maskedArgmaxProtocol& masked_argmax,
               matrixRowSumProtocol& row_sum,
               zeroProtocol& zero_check,
               lessCheckProtocol& less_check,
               nonzeroCheckProtocol& nonzero_check)
    {
        setup(masked_argmax, row_sum, zero_check, less_check, &nonzero_check);
    }

private:
    void setup(maskedArgmaxProtocol& masked_argmax,
               matrixRowSumProtocol& row_sum,
               zeroProtocol& zero_check,
               lessCheckProtocol& less_check,
               nonzeroCheckProtocol* nonzero_check)
    {
        params_.masked_argmax = &masked_argmax;
        params_.row_sum = &row_sum;
        params_.zero_check = &zero_check;
        params_.less_check = &less_check;
        params_.nonzero_check = nonzero_check;
        mark_setup();
    }

public:

    void add_check(const forestSplitSelectionWitness& witness,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count,
                   size_t query_count,
                   size_t gain_gap_bit_bound);

    bool finalize();

private:
    forestSplitSelectionParam params_;
    std::vector<forestSplitSelectionWitness> witnesses_;
    std::vector<forestSplitSelectionCheckRecord> checks_;
};
