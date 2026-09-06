#include "forest_histogram_refresh.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

void validate_channel_batch(const forestHistogramRefreshChannelBatch& channels)
{
    if (channels.query_values.empty()) {
        throw std::runtime_error("forestHistogramRefreshProtocol: channel batch must be non-empty");
    }

    const size_t channel_count = channels.query_values.size();
    if (channels.lifted_values.size() != channel_count || channels.leaf_histograms.size() != channel_count ||
        channels.full_histograms.size() != channel_count) {
        throw std::runtime_error("forestHistogramRefreshProtocol: channel batch metadata size mismatch");
    }
}

} // namespace

void forestHistogramRefreshProtocol::add_check(const forestHistogramRefreshWitness& witness,
                                               size_t query_row_count,
                                               size_t node_count,
                                               size_t feature_count,
                                               size_t bin_count,
                                               size_t input_bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("forestHistogramRefreshProtocol not setup");
    }
    if (params_.training_inference == nullptr || params_.bucket_address == nullptr ||
        params_.column_replication == nullptr || params_.forest_histogram == nullptr) {
        throw std::runtime_error("forestHistogramRefreshProtocol dependencies not setup");
    }

    validate_matrix_shape(query_row_count, 1, "forestHistogramRefreshProtocol query vectors");
    validate_matrix_shape(query_row_count, feature_count, "forestHistogramRefreshProtocol lifted matrices");
    validate_channel_batch(witness.channels);

    witnesses_.push_back(witness);
    checks_.push_back({query_row_count,
                       node_count,
                       feature_count,
                       bin_count,
                       input_bit_bound,
                       witness.channels.query_values.size(),
                       false});
}

void forestHistogramRefreshProtocol::add_check_with_certified_parent_mask(
    const forestHistogramRefreshWitness& witness,
    size_t query_row_count,
    size_t node_count,
    size_t feature_count,
    size_t bin_count,
    size_t input_bit_bound)
{
    add_check(witness, query_row_count, node_count, feature_count, bin_count, input_bit_bound);
    checks_.back().parent_mask_certified = true;
}

bool forestHistogramRefreshProtocol::finalize()
{
    begin_finalize("forestHistogramRefreshProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestHistogramRefreshCheckRecord& record = checks_[i];
        const forestHistogramRefreshWitness& witness = witnesses_[i];

        params_.training_inference->add_check(
            witness.inference, record.query_row_count, record.node_count, record.feature_count, record.input_bit_bound);

        forestHistogramBucketAddressWitness bucket_witness;
        bucket_witness.leaf_indices = witness.inference.leaf_indices;
        bucket_witness.lifted_leaf_indices = witness.lifted_leaf_indices;
        bucket_witness.lifted_inputs = witness.inference.lifted_inputs;
        bucket_witness.flattened_bucket_indices = witness.flattened_bucket_indices;
        params_.bucket_address->add_check_with_certified_indices(
            bucket_witness, record.query_row_count, record.feature_count, record.bin_count, record.node_count);

        forestHistogramChannelBatch histogram_channels;
        histogram_channels.values.reserve(record.channel_count);
        histogram_channels.leaf_histograms.reserve(record.channel_count);
        histogram_channels.full_histograms.reserve(record.channel_count);

        for (size_t channel = 0; channel < record.channel_count; ++channel) {
            params_.column_replication->add_check(witness.channels.query_values[channel],
                                                  witness.channels.lifted_values[channel],
                                                  record.query_row_count,
                                                  record.feature_count);
            histogram_channels.values.push_back(witness.channels.lifted_values[channel]);
            histogram_channels.leaf_histograms.push_back(witness.channels.leaf_histograms[channel]);
            histogram_channels.full_histograms.push_back(witness.channels.full_histograms[channel]);
        }

        if (record.parent_mask_certified) {
            params_.forest_histogram->add_masked_check_with_certified_parent_mask(
                witness.flattened_bucket_indices,
                histogram_channels,
                witness.parent_mask,
                witness.left_child_indices,
                witness.right_child_indices,
                record.query_row_count * record.feature_count,
                record.node_count,
                record.feature_count * record.bin_count);
        } else {
            params_.forest_histogram->add_masked_check(witness.flattened_bucket_indices,
                                                       histogram_channels,
                                                       witness.parent_mask,
                                                       witness.left_child_indices,
                                                       witness.right_child_indices,
                                                       record.query_row_count * record.feature_count,
                                                       record.node_count,
                                                       record.feature_count * record.bin_count);
        }
    }

    return true;
}
