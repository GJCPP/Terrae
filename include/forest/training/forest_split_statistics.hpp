#pragma once

#include <vector>

#include "column_replication.hpp"
#include "kzg.hpp"
#include "matrix_sum.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestSplitStatisticsChannelBatch {
public:
    std::vector<virCommitment> histograms;
    std::vector<virCommitment> left_statistics;
    std::vector<virCommitment> right_statistics;
};

class forestSplitStatisticsCheckRecord {
public:
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t channel_count = 0;
    std::vector<kzgCommitment> row_total_commitments;
    std::vector<kzgCommitment> replicated_total_commitments;
};

class forestSplitStatisticsParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    matrixRowPrefixSumProtocol* row_prefix = nullptr;
    matrixRowSumProtocol* row_sum = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
};

class forestSplitStatisticsVerifier {
public:
    void add_check(const forestSplitStatisticsCheckRecord& record);
    bool finalize() const;

    std::vector<forestSplitStatisticsCheckRecord> checks;
    const forestSplitStatisticsParam* params = nullptr;
};

class forestSplitStatisticsProver {
public:
    forestSplitStatisticsCheckRecord add_check(const forestSplitStatisticsChannelBatch& channels,
                                               size_t node_count,
                                               size_t feature_count,
                                               size_t bin_count);

    std::vector<forestSplitStatisticsCheckRecord> checks;
    const forestSplitStatisticsParam* params = nullptr;
};

// Proves split-candidate statistics derived from forest histograms.
//
// Histogram layout:
// - each channel histogram is a row-major `node_count x (feature_count * bin_count)` matrix
// - within each node row, every feature occupies one contiguous `bin_count` block
// - equivalently, the same witness is viewed as a
//   `(node_count * feature_count) x bin_count` matrix with one row per
//   `(node, feature)` pair
//
// Proven relations for every channel:
// - `left_statistics` is the inclusive row-wise prefix sum over bins
// - `right_statistics = total_per_(node,feature) - left_statistics`
//
// So each threshold column represents the usual split after that bin:
// - left side includes bins `<= threshold_bin`
// - right side includes bins `> threshold_bin`
class forestSplitStatisticsProtocol : protected ProtocolLifecycle {
public:
    ~forestSplitStatisticsProtocol() noexcept
    {
        auto_finalize("forestSplitStatisticsProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero_check,
               matrixRowPrefixSumProtocol& row_prefix,
               matrixRowSumProtocol& row_sum,
               columnReplicationProtocol& column_replication)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero_check;
        params_.row_prefix = &row_prefix;
        params_.row_sum = &row_sum;
        params_.column_replication = &column_replication;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& histogram,
                   const virCommitment& left_statistics,
                   const virCommitment& right_statistics,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count);

    void add_check(const forestSplitStatisticsChannelBatch& channels,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count);

    bool finalize();

private:
    forestSplitStatisticsParam params_;
    forestSplitStatisticsProver prover_;
    forestSplitStatisticsVerifier verifier_;
    std::vector<std::vector<virCommitment>> histogram_channels_;
    std::vector<std::vector<virCommitment>> left_statistic_channels_;
    std::vector<std::vector<virCommitment>> right_statistic_channels_;
};
