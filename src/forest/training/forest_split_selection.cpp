#include "forest_split_selection.hpp"

#include "matrix_utils.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>

namespace {

bool forest_split_selection_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_FOREST_SPLIT_SELECTION");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void forest_split_selection_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = forest_split_selection_profile_enabled();
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
        std::cerr << "forest_split_selection_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("forest_split_selection_profile", label, "", elapsed.count(), "ok");
}

poly candidate_index_matrix_polynomial(size_t node_count, size_t feature_count, size_t bin_count)
{
    const size_t column_count = feature_count * bin_count;
    std::vector<Fr> evals;
    evals.reserve(node_count * column_count);
    for (size_t row = 0; row < node_count; ++row) {
        (void)row;
        for (size_t col = 0; col < column_count; ++col) {
            evals.push_back(field_from_size_t(col));
        }
    }
    return poly_from_evals(evals);
}

poly valid_threshold_matrix_polynomial(size_t node_count, size_t feature_count, size_t bin_count)
{
    const size_t column_count = feature_count * bin_count;
    std::vector<Fr> evals;
    evals.reserve(node_count * column_count);
    for (size_t row = 0; row < node_count; ++row) {
        (void)row;
        for (size_t col = 0; col < column_count; ++col) {
            evals.push_back(Fr((col % bin_count) + 1 < bin_count ? 1 : 0));
        }
    }
    return poly_from_evals(evals);
}

poly reverse_candidate_index_matrix_polynomial(size_t node_count, size_t feature_count, size_t bin_count)
{
    const size_t column_count = feature_count * bin_count;
    std::vector<Fr> evals;
    evals.reserve(node_count * column_count);
    for (size_t row = 0; row < node_count; ++row) {
        (void)row;
        for (size_t col = 0; col < column_count; ++col) {
            evals.push_back(field_from_size_t(column_count - 1 - col));
        }
    }
    return poly_from_evals(evals);
}

using SplitSelectionShapeKey = std::tuple<size_t, size_t, size_t>;

const poly& cached_candidate_index_matrix_polynomial(size_t node_count, size_t feature_count, size_t bin_count)
{
    static std::map<SplitSelectionShapeKey, poly> cache;
    const SplitSelectionShapeKey key{node_count, feature_count, bin_count};
    auto [it, inserted] = cache.emplace(key, poly());
    if (inserted) {
        it->second = candidate_index_matrix_polynomial(node_count, feature_count, bin_count);
    }
    return it->second;
}

const poly& cached_valid_threshold_matrix_polynomial(size_t node_count, size_t feature_count, size_t bin_count)
{
    static std::map<SplitSelectionShapeKey, poly> cache;
    const SplitSelectionShapeKey key{node_count, feature_count, bin_count};
    auto [it, inserted] = cache.emplace(key, poly());
    if (inserted) {
        it->second = valid_threshold_matrix_polynomial(node_count, feature_count, bin_count);
    }
    return it->second;
}

const poly& cached_reverse_candidate_index_matrix_polynomial(size_t node_count, size_t feature_count, size_t bin_count)
{
    static std::map<SplitSelectionShapeKey, poly> cache;
    const SplitSelectionShapeKey key{node_count, feature_count, bin_count};
    auto [it, inserted] = cache.emplace(key, poly());
    if (inserted) {
        it->second = reverse_candidate_index_matrix_polynomial(node_count, feature_count, bin_count);
    }
    return it->second;
}

void validate_forest_split_selection_parameters(size_t node_count,
                                                size_t feature_count,
                                                size_t bin_count,
                                                size_t query_count,
                                                size_t gain_gap_bit_bound)
{
    validate_matrix_shape(node_count, feature_count * bin_count, "forestSplitSelectionProtocol");
    if (!isPowerOfTwo(feature_count) || !isPowerOfTwo(bin_count) || !isPowerOfTwo(query_count)) {
        throw std::runtime_error(
            "forestSplitSelectionProtocol: feature_count, bin_count, and query_count must be powers of two");
    }
    if (gain_gap_bit_bound > static_cast<size_t>(Fr::getBitSize() - 1)) {
        throw std::runtime_error(
            "forestSplitSelectionProtocol: gain gap bit bound exceeds the safely representable integer range of Fr");
    }
}

} // namespace

void forestSplitSelectionProtocol::add_check(const forestSplitSelectionWitness& witness,
                                             size_t node_count,
                                             size_t feature_count,
                                             size_t bin_count,
                                             size_t query_count,
                                             size_t gain_gap_bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("forestSplitSelectionProtocol not setup");
    }
    if (params_.masked_argmax == nullptr || params_.row_sum == nullptr || params_.zero_check == nullptr ||
        params_.less_check == nullptr) {
        throw std::runtime_error("forestSplitSelectionProtocol dependencies not setup");
    }

    validate_forest_split_selection_parameters(
        node_count, feature_count, bin_count, query_count, gain_gap_bit_bound);

    witnesses_.push_back(witness);
    checks_.push_back({node_count, feature_count, bin_count, query_count, gain_gap_bit_bound});
}

bool forestSplitSelectionProtocol::finalize()
{
    begin_finalize("forestSplitSelectionProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestSplitSelectionCheckRecord& record = checks_[i];
        const forestSplitSelectionWitness& witness = witnesses_[i];
        const size_t column_count = record.feature_count * record.bin_count;
        const size_t tie_break_gap_bit_bound = record.gain_gap_bit_bound + log2N(column_count) + 1;

        const virCommitment candidate_indices(cached_candidate_index_matrix_polynomial(
            record.node_count, record.feature_count, record.bin_count));
        const virCommitment reverse_candidate_indices(cached_reverse_candidate_index_matrix_polynomial(
            record.node_count, record.feature_count, record.bin_count));
        const virCommitment tie_scale(field_from_size_t(column_count));
        virCommitment tie_break_zero =
            witness.tie_break_scores - witness.quantized_gains * tie_scale - reverse_candidate_indices;
        forest_split_selection_profile_scope("zero.tie_break_scores", [&] {
            params_.zero_check->add_check(tie_break_zero, record.node_count * column_count);
        });

        virCommitment selected_tie_break_zero =
            witness.active_node_mask *
            (witness.selected_tie_break_scores - witness.selected_gains * tie_scale -
             (virCommitment(field_from_size_t(column_count - 1)) - witness.selected_candidate_indices));
        forest_split_selection_profile_scope("zero.selected_tie_break_scores", [&] {
            params_.zero_check->add_check(selected_tie_break_zero, record.node_count);
        });

        maskedArgmaxCheckRecord argmax_record;
        forest_split_selection_profile_scope("masked_argmax.add_check", [&] {
            argmax_record = params_.masked_argmax->add_check(witness.tie_break_scores,
                                                             witness.selected_tie_break_scores,
                                                             witness.selector,
                                                             witness.candidate_mask,
                                                             witness.active_node_mask,
                                                             record.node_count,
                                                             column_count,
                                                             tie_break_gap_bit_bound);
        });

        const virCommitment valid_thresholds(cached_valid_threshold_matrix_polynomial(
            record.node_count, record.feature_count, record.bin_count));
        const virCommitment eligibility = valid_thresholds;
        virCommitment candidate_eligibility_zero =
            witness.candidate_mask * (virCommitment(Fr(1)) - eligibility);
        forest_split_selection_profile_scope("zero.candidate_eligibility", [&] {
            params_.zero_check->add_check(
                candidate_eligibility_zero, record.node_count * column_count,
                "forest_split_selection.candidate_eligibility");
        });
        virCommitment candidate_completeness_zero =
            (eligibility - witness.candidate_mask) * witness.left_counts * witness.right_counts;
        forest_split_selection_profile_scope("zero.candidate_completeness", [&] {
            params_.zero_check->add_check(
                candidate_completeness_zero, record.node_count * column_count,
                "forest_split_selection.candidate_completeness");
        });

        const size_t count_bit_bound = log2N(record.query_count) + 1;
        if (params_.nonzero_check != nullptr) {
            forest_split_selection_profile_scope("nonzero.left_count", [&] {
                params_.nonzero_check->add_masked_check(witness.left_counts,
                                                        witness.candidate_mask,
                                                        record.node_count * column_count,
                                                        "forest_split_selection.left_count_nonzero");
            });
            forest_split_selection_profile_scope("nonzero.right_count", [&] {
                params_.nonzero_check->add_masked_check(witness.right_counts,
                                                        witness.candidate_mask,
                                                        record.node_count * column_count,
                                                        "forest_split_selection.right_count_nonzero");
            });
        } else {
            forest_split_selection_profile_scope("less.left_count_nonzero", [&] {
                params_.less_check->add_masked_check(
                    witness.left_counts - virCommitment(Fr(1)),
                    witness.candidate_mask,
                    count_bit_bound,
                    record.node_count * column_count);
            });
            forest_split_selection_profile_scope("less.right_count_nonzero", [&] {
                params_.less_check->add_masked_check(
                    witness.right_counts - virCommitment(Fr(1)),
                    witness.candidate_mask,
                    count_bit_bound,
                    record.node_count * column_count);
            });
        }

        forest_split_selection_profile_scope("row_sum.selected_gains", [&] {
            params_.row_sum->add_check(
                witness.selector * witness.quantized_gains, witness.selected_gains, record.node_count, column_count);
        });
        forest_split_selection_profile_scope("row_sum.selected_candidate_indices", [&] {
            params_.row_sum->add_check(
                witness.selector * candidate_indices, witness.selected_candidate_indices, record.node_count, column_count);
        });

        const virCommitment feature_scale(field_from_size_t(record.bin_count));
        virCommitment index_split_zero =
            witness.selected_candidate_indices - witness.selected_features * feature_scale - witness.selected_bins;
        forest_split_selection_profile_scope("zero.index_split", [&] {
            params_.zero_check->add_check(index_split_zero, record.node_count);
        });

        forest_split_selection_profile_scope("less.selected_features", [&] {
            params_.less_check->add_check(witness.selected_features, log2N(record.feature_count), record.node_count);
        });
        forest_split_selection_profile_scope("less.selected_bins", [&] {
            params_.less_check->add_check(witness.selected_bins, log2N(record.bin_count), record.node_count);
        });
    }

    return true;
}
