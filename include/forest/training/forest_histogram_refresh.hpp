#pragma once

#include <vector>

#include "column_replication.hpp"
#include "forest_histogram.hpp"
#include "forest_histogram_bucket_address.hpp"
#include "forest_training_inference.hpp"
#include "protocol_lifecycle.hpp"

using namespace mcl::bn;

class forestHistogramRefreshChannelBatch {
public:
    std::vector<virCommitment> query_values;
    std::vector<virCommitment> lifted_values;
    std::vector<virCommitment> leaf_histograms;
    std::vector<virCommitment> full_histograms;
};

class forestHistogramRefreshWitness {
public:
    forestTrainingInferenceWitness inference;
    virCommitment lifted_leaf_indices;
    virCommitment flattened_bucket_indices;
    forestHistogramRefreshChannelBatch channels;
    virCommitment parent_mask;
    virCommitment left_child_indices;
    virCommitment right_child_indices;
};

class forestHistogramRefreshCheckRecord {
public:
    size_t query_row_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t input_bit_bound = 0;
    size_t channel_count = 0;
    bool parent_mask_certified = false;
};

class forestHistogramRefreshParam {
public:
    forestTrainingInferenceProtocol* training_inference = nullptr;
    forestHistogramBucketAddressProtocol* bucket_address = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
    forestHistogramProtocol* forest_histogram = nullptr;
};

// Proves one histogram-refresh layer on top of training-side witness-forest
// inference.
//
// This composition layer proves:
// - per-query `leaf_index`, leaf box, and `tree_score` are sourced from the
//   committed witness-side forest,
// - flattened `(leaf, feature, bin)` bucket addresses are derived from those
//   sourced leaf indices plus lifted feature bins,
// - each per-query channel is column-replicated across the feature axis,
// - and the resulting lifted channel values aggregate into the supplied
//   leaf/full histograms.
//
// The protocol intentionally keeps the per-query channel values generic. It
// proves histogram refresh from inference-sourced leaf indices; later layers can
// separately prove where those query-level channels come from, such as
// pseudo-residual or Hessian refresh.
class forestHistogramRefreshProtocol : protected ProtocolLifecycle {
public:
    ~forestHistogramRefreshProtocol() noexcept
    {
        auto_finalize("forestHistogramRefreshProtocol", [this] { return finalize(); });
    }

    void setup(forestTrainingInferenceProtocol& training_inference,
               forestHistogramBucketAddressProtocol& bucket_address,
               columnReplicationProtocol& column_replication,
               forestHistogramProtocol& forest_histogram)
    {
        params_.training_inference = &training_inference;
        params_.bucket_address = &bucket_address;
        params_.column_replication = &column_replication;
        params_.forest_histogram = &forest_histogram;
        mark_setup();
    }

    void add_check(const forestHistogramRefreshWitness& witness,
                   size_t query_row_count,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count,
                   size_t input_bit_bound);

    // The complete training certificate verifies well-formedness against the
    // exact same committed forest, so its split mask is already Boolean.
    void add_check_with_certified_parent_mask(const forestHistogramRefreshWitness& witness,
                                              size_t query_row_count,
                                              size_t node_count,
                                              size_t feature_count,
                                              size_t bin_count,
                                              size_t input_bit_bound);

    bool finalize();

private:
    forestHistogramRefreshParam params_;
    std::vector<forestHistogramRefreshWitness> witnesses_;
    std::vector<forestHistogramRefreshCheckRecord> checks_;
};
