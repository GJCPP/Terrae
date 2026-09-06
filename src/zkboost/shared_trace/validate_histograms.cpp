#include "zkboost/shared_trace/validate_histograms.hpp"

#include "algebra_utils.hpp"

#include <functional>
#include <stdexcept>
#include <string>

namespace {

constexpr size_t kSharedHistogramChannelCount = 3;

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_validate_histograms: " + message);
    }
}

size_t checked_size(std::int64_t value, const std::string& name)
{
    require_condition(value >= 0, name + " must be non-negative");
    return static_cast<size_t>(value);
}

size_t candidate_column(size_t feature, size_t bin, size_t bin_count)
{
    return feature * bin_count + bin;
}

void validate_histogram_shapes(const forestTrainingSemanticTrace& training,
                               const zkboostSharedBoostingStateView& boosting)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(training.node_count > 0, "node_count must be non-zero");
    require_condition(training.feature_count > 0, "feature_count must be non-zero");
    require_condition(training.bin_count > 0, "bin_count must be non-zero");
    require_condition(training.query_count > 0, "query_count must be non-zero");
    require_condition(training.tree_count == training.iteration_count * training.class_count,
                      "tree_count must equal iteration_count * class_count");
    require_condition(training.leaf_indices.size() == training.tree_count * training.query_count,
                      "leaf_indices length mismatch");
    require_condition(training.training_inputs.size() == training.query_count * training.feature_count,
                      "training_inputs length mismatch");
    require_condition(training.split_node_mask.size() == training.node_count,
                      "split_node_mask length mismatch");
    require_condition(training.left_child_indices.size() == training.node_count,
                      "left_child_indices length mismatch");
    require_condition(training.right_child_indices.size() == training.node_count,
                      "right_child_indices length mismatch");
    require_condition(boosting.summary.iteration_count == training.iteration_count,
                      "boosting iteration_count mismatch");
    require_condition(boosting.summary.class_count == training.class_count,
                      "boosting class_count mismatch");
    require_condition(boosting.summary.sample_count == training.query_count,
                      "boosting sample_count mismatch");
    require_condition(boosting.summary.tree_count == training.tree_count,
                      "boosting tree_count mismatch");
    const size_t query_row_count = training.tree_count * training.query_count;
    require_condition(boosting.summary.cube_size == query_row_count,
                      "boosting cube_size mismatch");
    require_condition(boosting.pseudo_residuals.size() == query_row_count,
                      "pseudo_residuals length mismatch");
    require_condition(boosting.diagonal_hessians.size() == query_row_count,
                      "diagonal_hessians length mismatch");
}

std::vector<Fr> derive_count_query_values(size_t query_row_count)
{
    return std::vector<Fr>(query_row_count, Fr(1));
}

std::vector<Fr> derive_lifted_leaf_indices(const forestTrainingSemanticTrace& training)
{
    std::vector<Fr> out;
    out.reserve(training.tree_count * training.query_count * training.feature_count);
    for (std::int64_t leaf_index : training.leaf_indices) {
        const Fr field_leaf = fr_from_dec(std::to_string(leaf_index));
        for (size_t feature = 0; feature < training.feature_count; ++feature) {
            (void)feature;
            out.push_back(field_leaf);
        }
    }
    return out;
}

std::vector<Fr> derive_flattened_bucket_indices(const forestTrainingSemanticTrace& training)
{
    const size_t row_width = training.feature_count * training.bin_count;
    std::vector<Fr> out;
    out.reserve(training.tree_count * training.query_count * training.feature_count);
    for (size_t query_row = 0; query_row < training.tree_count * training.query_count; ++query_row) {
        const size_t sample = query_row % training.query_count;
        const size_t leaf_index =
            checked_size(training.leaf_indices.at(query_row), "leaf_indices");
        require_condition(leaf_index < training.node_count, "leaf index out of range");
        for (size_t feature = 0; feature < training.feature_count; ++feature) {
            const size_t input_index = sample * training.feature_count + feature;
            const size_t bin = checked_size(training.training_inputs.at(input_index), "training_inputs");
            require_condition(bin < training.bin_count, "training input bin out of range");
            out.push_back(field_from_size_t(
                leaf_index * row_width + candidate_column(feature, bin, training.bin_count)));
        }
    }
    return out;
}

std::vector<std::vector<Fr>> derive_leaf_histograms(
    const forestTrainingSemanticTrace& training,
    const std::vector<std::vector<Fr>>& query_channels)
{
    const size_t query_row_count = training.tree_count * training.query_count;
    const size_t histogram_width = training.feature_count * training.bin_count;
    std::vector<std::vector<Fr>> out(
        query_channels.size(),
        std::vector<Fr>(training.node_count * histogram_width, Fr(0)));

    for (const std::vector<Fr>& channel : query_channels) {
        require_condition(channel.size() == query_row_count, "query channel length mismatch");
    }

    for (size_t query_row = 0; query_row < query_row_count; ++query_row) {
        const size_t sample = query_row % training.query_count;
        const size_t leaf_index = checked_size(training.leaf_indices.at(query_row), "leaf_indices");
        require_condition(leaf_index < training.node_count, "leaf index out of range");
        for (size_t feature = 0; feature < training.feature_count; ++feature) {
            const size_t input_index = sample * training.feature_count + feature;
            const size_t bin = checked_size(training.training_inputs.at(input_index), "training_inputs");
            require_condition(bin < training.bin_count, "training input bin out of range");
            const size_t offset =
                leaf_index * histogram_width + candidate_column(feature, bin, training.bin_count);
            for (size_t channel = 0; channel < query_channels.size(); ++channel) {
                out.at(channel).at(offset) += query_channels.at(channel).at(query_row);
            }
        }
    }
    return out;
}

std::vector<Fr> derive_full_histogram_for_channel(const forestTrainingSemanticTrace& training,
                                                  const std::vector<Fr>& leaf_histogram)
{
    const size_t width = training.feature_count * training.bin_count;
    require_condition(leaf_histogram.size() == training.node_count * width,
                      "leaf histogram length mismatch");

    std::vector<Fr> out(training.node_count * width, Fr(0));
    std::vector<bool> computed(training.node_count, false);
    std::vector<bool> visiting(training.node_count, false);

    std::function<void(size_t)> compute = [&](size_t node) {
        require_condition(node < training.node_count, "node index out of range");
        if (computed.at(node)) {
            return;
        }
        require_condition(!visiting.at(node), "cycle in forest child pointers");
        visiting.at(node) = true;

        for (size_t col = 0; col < width; ++col) {
            out.at(node * width + col) = leaf_histogram.at(node * width + col);
        }

        if (checked_size(training.split_node_mask.at(node), "split_node_mask") != 0) {
            const size_t left = checked_size(training.left_child_indices.at(node), "left_child_indices");
            const size_t right = checked_size(training.right_child_indices.at(node), "right_child_indices");
            require_condition(left < training.node_count, "left child index out of range");
            require_condition(right < training.node_count, "right child index out of range");
            compute(left);
            compute(right);
            for (size_t col = 0; col < width; ++col) {
                out.at(node * width + col) +=
                    out.at(left * width + col) + out.at(right * width + col);
            }
        }

        visiting.at(node) = false;
        computed.at(node) = true;
    };

    for (size_t node = 0; node < training.node_count; ++node) {
        compute(node);
    }
    return out;
}

std::vector<Fr> derive_left_statistics(const std::vector<Fr>& histogram,
                                       size_t node_count,
                                       size_t feature_count,
                                       size_t bin_count)
{
    std::vector<Fr> out(node_count * feature_count * bin_count, Fr(0));
    for (size_t node = 0; node < node_count; ++node) {
        for (size_t feature = 0; feature < feature_count; ++feature) {
            Fr running(0);
            for (size_t bin = 0; bin < bin_count; ++bin) {
                const size_t index = node * feature_count * bin_count + feature * bin_count + bin;
                running += histogram.at(index);
                out.at(index) = running;
            }
        }
    }
    return out;
}

std::vector<Fr> derive_right_statistics(const std::vector<Fr>& left_statistics,
                                        size_t node_count,
                                        size_t feature_count,
                                        size_t bin_count)
{
    std::vector<Fr> out(node_count * feature_count * bin_count, Fr(0));
    for (size_t node = 0; node < node_count; ++node) {
        for (size_t feature = 0; feature < feature_count; ++feature) {
            const size_t total_index =
                node * feature_count * bin_count + feature * bin_count + (bin_count - 1);
            const Fr total = left_statistics.at(total_index);
            for (size_t bin = 0; bin < bin_count; ++bin) {
                const size_t index = node * feature_count * bin_count + feature * bin_count + bin;
                out.at(index) = total - left_statistics.at(index);
            }
        }
    }
    return out;
}

} // namespace

zkboostSharedHistogramView build_zkboost_shared_histogram_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& boosting)
{
    validate_histogram_shapes(training, boosting);

    zkboostSharedHistogramView view;
    view.summary.tree_count = training.tree_count;
    view.summary.node_count = training.node_count;
    view.summary.feature_count = training.feature_count;
    view.summary.bin_count = training.bin_count;
    view.summary.sample_count = training.query_count;
    view.summary.query_row_count = training.tree_count * training.query_count;
    view.summary.histogram_width = training.feature_count * training.bin_count;
    view.summary.histogram_size = training.node_count * view.summary.histogram_width;
    view.summary.channel_count = kSharedHistogramChannelCount;

    view.query_counts = derive_count_query_values(view.summary.query_row_count);
    view.query_pseudo_residuals = boosting.pseudo_residuals;
    view.query_diagonal_hessians = boosting.diagonal_hessians;
    view.lifted_leaf_indices = derive_lifted_leaf_indices(training);
    view.flattened_bucket_indices = derive_flattened_bucket_indices(training);

    const std::vector<std::vector<Fr>> query_channels = {
        view.query_counts,
        view.query_pseudo_residuals,
        view.query_diagonal_hessians,
    };
    view.leaf_histograms = derive_leaf_histograms(training, query_channels);
    view.full_histograms.reserve(kSharedHistogramChannelCount);
    view.left_statistics.reserve(kSharedHistogramChannelCount);
    view.right_statistics.reserve(kSharedHistogramChannelCount);
    for (size_t channel = 0; channel < kSharedHistogramChannelCount; ++channel) {
        view.full_histograms.push_back(
            derive_full_histogram_for_channel(training, view.leaf_histograms.at(channel)));
        view.left_statistics.push_back(
            derive_left_statistics(view.full_histograms.back(),
                                   training.node_count,
                                   training.feature_count,
                                   training.bin_count));
        view.right_statistics.push_back(
            derive_right_statistics(view.left_statistics.back(),
                                    training.node_count,
                                    training.feature_count,
                                    training.bin_count));
    }

    return view;
}

zkboostSharedHistogramView build_zkboost_shared_histogram_view(
    const forestTrainingSemanticTrace& training)
{
    return build_zkboost_shared_histogram_view(
        training,
        build_zkboost_shared_boosting_state_view(training));
}

zkboostSharedHistogramSummary validate_zkboost_shared_histograms(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& boosting)
{
    return build_zkboost_shared_histogram_view(training, boosting).summary;
}

zkboostSharedHistogramSummary validate_zkboost_shared_histograms(
    const forestTrainingSemanticTrace& training)
{
    return build_zkboost_shared_histogram_view(training).summary;
}
