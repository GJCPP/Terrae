#include "forest_histogram.hpp"

#include "matrix_utils.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

bool forest_histogram_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_FOREST_HISTOGRAM");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void forest_histogram_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = forest_histogram_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "forest_histogram_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("forest_histogram_profile", label, "", elapsed.count(), "ok");
}

void validate_channel_batch(const forestHistogramChannelBatch& channels)
{
    if (channels.values.empty()) {
        throw std::runtime_error("forestHistogramProtocol: channel batch must be non-empty");
    }
    if (channels.values.size() != channels.leaf_histograms.size() ||
        channels.values.size() != channels.full_histograms.size()) {
        throw std::runtime_error("forestHistogramProtocol: channel batch metadata size mismatch");
    }
}

forestHistogramChannelBatch single_channel_batch(const virCommitment& values,
                                                 const virCommitment& leaf_histogram,
                                                 const virCommitment& full_histogram)
{
    forestHistogramChannelBatch channels;
    channels.values.push_back(values);
    channels.leaf_histograms.push_back(leaf_histogram);
    channels.full_histograms.push_back(full_histogram);
    return channels;
}

} // namespace

void forestHistogramProtocol::add_check(const virCommitment& flattened_bucket_indices,
                                        const virCommitment& values,
                                        const virCommitment& leaf_histogram,
                                        const virCommitment& full_histogram,
                                        const virCommitment& left_child_indices,
                                        const virCommitment& right_child_indices,
                                        size_t query_domain_size,
                                        size_t node_count,
                                        size_t histogram_width)
{
    add_masked_check(flattened_bucket_indices,
                     single_channel_batch(values, leaf_histogram, full_histogram),
                     virCommitment(Fr(1)),
                     left_child_indices,
                     right_child_indices,
                     query_domain_size,
                     node_count,
                     histogram_width);
    checks_.back().uses_parent_mask = false;
}

void forestHistogramProtocol::add_masked_check(const virCommitment& flattened_bucket_indices,
                                               const virCommitment& values,
                                               const virCommitment& leaf_histogram,
                                               const virCommitment& full_histogram,
                                               const virCommitment& parent_mask,
                                               const virCommitment& left_child_indices,
                                               const virCommitment& right_child_indices,
                                               size_t query_domain_size,
                                               size_t node_count,
                                               size_t histogram_width)
{
    add_masked_check(flattened_bucket_indices,
                     single_channel_batch(values, leaf_histogram, full_histogram),
                     parent_mask,
                     left_child_indices,
                     right_child_indices,
                     query_domain_size,
                     node_count,
                     histogram_width);
}

void forestHistogramProtocol::add_check(const virCommitment& flattened_bucket_indices,
                                        const forestHistogramChannelBatch& channels,
                                        const virCommitment& left_child_indices,
                                        const virCommitment& right_child_indices,
                                        size_t query_domain_size,
                                        size_t node_count,
                                        size_t histogram_width)
{
    add_masked_check(flattened_bucket_indices,
                     channels,
                     virCommitment(Fr(1)),
                     left_child_indices,
                     right_child_indices,
                     query_domain_size,
                     node_count,
                     histogram_width);
    checks_.back().uses_parent_mask = false;
}

void forestHistogramProtocol::add_masked_check(const virCommitment& flattened_bucket_indices,
                                               const forestHistogramChannelBatch& channels,
                                               const virCommitment& parent_mask,
                                               const virCommitment& left_child_indices,
                                               const virCommitment& right_child_indices,
                                               size_t query_domain_size,
                                               size_t node_count,
                                               size_t histogram_width)
{
    if (!is_setup) {
        throw std::runtime_error("forestHistogramProtocol not setup");
    }
    if (params_.hist_agg == nullptr || params_.tree_histogram_consistency == nullptr) {
        throw std::runtime_error("forestHistogramProtocol dependencies not setup");
    }

    validate_matrix_shape(node_count, 1, "forestHistogramProtocol child indices");
    validate_matrix_shape(node_count, histogram_width, "forestHistogramProtocol histograms");
    validate_channel_batch(channels);

    checks_.push_back({query_domain_size, node_count, histogram_width, channels.values.size(), true, false});
    flattened_bucket_indices_.push_back(flattened_bucket_indices);
    value_channels_.push_back(channels.values);
    leaf_histogram_channels_.push_back(channels.leaf_histograms);
    full_histogram_channels_.push_back(channels.full_histograms);
    parent_masks_.push_back(parent_mask);
    left_child_indices_.push_back(left_child_indices);
    right_child_indices_.push_back(right_child_indices);
}

void forestHistogramProtocol::add_masked_check_with_certified_parent_mask(
    const virCommitment& flattened_bucket_indices,
    const forestHistogramChannelBatch& channels,
    const virCommitment& parent_mask,
    const virCommitment& left_child_indices,
    const virCommitment& right_child_indices,
    size_t query_domain_size,
    size_t node_count,
    size_t histogram_width)
{
    add_masked_check(flattened_bucket_indices,
                     channels,
                     parent_mask,
                     left_child_indices,
                     right_child_indices,
                     query_domain_size,
                     node_count,
                     histogram_width);
    checks_.back().parent_mask_certified = true;
}

bool forestHistogramProtocol::finalize()
{
    begin_finalize("forestHistogramProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestHistogramCheckRecord& record = checks_[i];
        const size_t histogram_domain_size = record.node_count * record.histogram_width;

        forest_histogram_profile_scope("hist_agg.add_check", [&] {
            params_.hist_agg->add_check(flattened_bucket_indices_[i],
                                        value_channels_[i],
                                        leaf_histogram_channels_[i],
                                        record.query_domain_size,
                                        histogram_domain_size);
        });

        if (record.uses_parent_mask) {
            forest_histogram_profile_scope("tree_histogram_consistency.add_masked_check", [&] {
                if (record.parent_mask_certified) {
                    params_.tree_histogram_consistency->add_masked_check_with_certified_parent_mask(
                        full_histogram_channels_[i],
                        leaf_histogram_channels_[i],
                        parent_masks_[i],
                        left_child_indices_[i],
                        right_child_indices_[i],
                        record.node_count,
                        record.histogram_width);
                } else {
                    params_.tree_histogram_consistency->add_masked_check(full_histogram_channels_[i],
                                                                         leaf_histogram_channels_[i],
                                                                         parent_masks_[i],
                                                                         left_child_indices_[i],
                                                                         right_child_indices_[i],
                                                                         record.node_count,
                                                                         record.histogram_width);
                }
            });
            continue;
        }

        forest_histogram_profile_scope("tree_histogram_consistency.add_check", [&] {
            params_.tree_histogram_consistency->add_check(full_histogram_channels_[i],
                                                          leaf_histogram_channels_[i],
                                                          left_child_indices_[i],
                                                          right_child_indices_[i],
                                                          record.node_count,
                                                          record.histogram_width);
        });
    }

    return true;
}
