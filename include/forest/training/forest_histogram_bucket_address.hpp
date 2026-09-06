#pragma once

#include "column_replication.hpp"
#include "less_check.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestHistogramBucketAddressWitness {
public:
    virCommitment leaf_indices;
    virCommitment lifted_leaf_indices;
    virCommitment lifted_inputs;
    virCommitment flattened_bucket_indices;
};

class forestHistogramBucketAddressCheckRecord {
public:
    size_t query_row_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t node_count = 0;
    bool indices_certified = false;
};

class forestHistogramBucketAddressParam {
public:
    zeroProtocol* zero_check = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

// Proves the flattened histogram bucket address
//   leaf_index * (feature_count * bin_count) + feature_offset + feature_value
// for each `(query, feature)` position.
class forestHistogramBucketAddressProtocol : protected ProtocolLifecycle {
public:
    ~forestHistogramBucketAddressProtocol() noexcept
    {
        auto_finalize("forestHistogramBucketAddressProtocol", [this] { return finalize(); });
    }

    void setup(zeroProtocol& zero_check, columnReplicationProtocol& column_replication, lessCheckProtocol& less_check)
    {
        params_.zero_check = &zero_check;
        params_.column_replication = &column_replication;
        params_.less_check = &less_check;
        mark_setup();
    }

    void add_check(const forestHistogramBucketAddressWitness& witness,
                   size_t query_row_count,
                   size_t feature_count,
                   size_t bin_count,
                   size_t node_count);

    // Skips only the leaf/address membership checks that are supplied by
    // training inference and histogram aggregation in the composed refresh.
    void add_check_with_certified_indices(const forestHistogramBucketAddressWitness& witness,
                                          size_t query_row_count,
                                          size_t feature_count,
                                          size_t bin_count,
                                          size_t node_count);

    bool finalize();

private:
    forestHistogramBucketAddressParam params_;
    std::vector<forestHistogramBucketAddressWitness> witnesses_;
    std::vector<forestHistogramBucketAddressCheckRecord> checks_;
};
