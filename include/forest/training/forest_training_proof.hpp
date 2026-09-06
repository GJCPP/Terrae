#pragma once

#include "fixed_point.hpp"
#include "forest_boosting_state.hpp"
#include "forest_histogram_refresh.hpp"
#include "forest_leaf_value.hpp"
#include "forest_split_gain.hpp"
#include "forest_split_selection.hpp"
#include "forest_split_statistics.hpp"
#include "forest_quantized_split_gain.hpp"
#include "less_check.hpp"
#include "matrix_sum.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestTrainingProofWitness {
public:
    virCommitment root_indices;
    virCommitment split_node_mask;
    virCommitment node_leaf_mask;
    virCommitment depths;
    virCommitment left_child_indices;
    virCommitment right_child_indices;
    virCommitment node_tree_ids;
    virCommitment feature_selector;
    virCommitment lower_bounds;
    virCommitment upper_bounds;
    virCommitment left_lower;
    virCommitment left_upper;
    virCommitment right_lower;
    virCommitment right_upper;
    virCommitment thresholds;

    forestHistogramRefreshWitness histogram_refresh;
    forestSplitStatisticsChannelBatch split_statistics;
    forestSplitGainWitness split_gain;
    forestQuantizedSplitGainWitness quantized_split_gain;
    forestSplitSelectionWitness split_selection;
    virCommitment depth_limit_mask;
    forestLeafValueWitness leaf_value;
    forestBoostingStateWitness boosting_state;
};

class forestTrainingProofCheckRecord {
public:
    size_t tree_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    size_t query_count = 0;
    size_t query_row_count = 0;
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t input_bit_bound = 0;
    size_t full_upper_bound = 0;
    size_t structure_bit_bound = 0;
    fixedPointEncoding gradient_encoding;
    fixedPointEncoding gain_numerator_encoding;
    fixedPointEncoding quantized_gain_encoding;
    fixedPointEncoding leaf_value_encoding;
    fixedPointEncoding score_encoding;
    fixedPointEncoding softmax_encoding;
    size_t hessian_bit_bound = 0;
    size_t gain_denominator_bit_bound = 0;
    size_t leaf_denominator_bit_bound = 0;
    size_t gain_gap_bit_bound = 0;
    size_t l2_regularization = 0;
    Fr split_penalty = Fr(0);
    Fr split_gain_threshold = Fr(0);
    size_t max_depth = 0;
    size_t learning_rate_numerator = 1;
    size_t learning_rate_denominator = 1;
    forestTrainingLoss loss_mode = forestTrainingLoss::SecondOrderXgboost;
};

class forestTrainingProofParam {
public:
    forestHistogramRefreshProtocol* forest_histogram_refresh = nullptr;
    forestSplitStatisticsProtocol* forest_split_statistics = nullptr;
    forestSplitGainProtocol* forest_split_gain = nullptr;
    forestQuantizedSplitGainProtocol* forest_quantized_split_gain = nullptr;
    forestSplitSelectionProtocol* forest_split_selection = nullptr;
    forestLeafValueProtocol* forest_leaf_value = nullptr;
    forestBoostingStateProtocol* forest_boosting_state = nullptr;
    matrixRowSumProtocol* row_sum = nullptr;
    zeroProtocol* zero_check = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

// Top-level composition layer for the current committed-forest training route.
//
// This protocol does not introduce new low-level proof machinery. Instead it
// ties the existing training bricks to one shared committed forest:
// - training-time witness-side inference,
// - histogram refresh from the sourced leaf assignments,
// - split statistics / exact gain / quantized gain / split selection,
// - leaf-value reduction,
// - and the boosting-state score / residual / Hessian layer.
//
// Forest legality is deliberately owned by the independent well-formedness
// proof. A complete training certificate verifies both proofs against the same
// committed-forest public inputs, so no structural constraint is proved twice.
//
// The current statement shape assumes the common three-channel histogram
// story: count, pseudo-residual / gradient, and diagonal Hessian.
class forestTrainingProofProtocol : protected ProtocolLifecycle {
public:
    ~forestTrainingProofProtocol() noexcept
    {
        auto_finalize("forestTrainingProofProtocol", [this] { return finalize(); });
    }

    void setup(forestHistogramRefreshProtocol& forest_histogram_refresh,
               forestSplitStatisticsProtocol& forest_split_statistics,
               forestSplitGainProtocol& forest_split_gain,
               forestQuantizedSplitGainProtocol& forest_quantized_split_gain,
               forestSplitSelectionProtocol& forest_split_selection,
               forestLeafValueProtocol& forest_leaf_value,
               forestBoostingStateProtocol& forest_boosting_state,
               matrixRowSumProtocol& row_sum,
               zeroProtocol& zero_check,
               lessCheckProtocol& less_check)
    {
        params_.forest_histogram_refresh = &forest_histogram_refresh;
        params_.forest_split_statistics = &forest_split_statistics;
        params_.forest_split_gain = &forest_split_gain;
        params_.forest_quantized_split_gain = &forest_quantized_split_gain;
        params_.forest_split_selection = &forest_split_selection;
        params_.forest_leaf_value = &forest_leaf_value;
        params_.forest_boosting_state = &forest_boosting_state;
        params_.row_sum = &row_sum;
        params_.zero_check = &zero_check;
        params_.less_check = &less_check;
        mark_setup();
    }

    void add_check(const forestTrainingProofWitness& witness,
                   size_t tree_count,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count,
                   size_t iteration_count,
                   size_t class_count,
                   size_t query_count,
                   size_t input_bit_bound,
                   size_t full_upper_bound,
                   size_t structure_bit_bound,
                   const fixedPointEncoding& gradient_encoding,
                   const fixedPointEncoding& gain_numerator_encoding,
                   const fixedPointEncoding& quantized_gain_encoding,
                   const fixedPointEncoding& leaf_value_encoding,
                   const fixedPointEncoding& score_encoding,
                   const fixedPointEncoding& softmax_encoding,
                   size_t hessian_bit_bound,
                   size_t gain_denominator_bit_bound,
                   size_t leaf_denominator_bit_bound,
                   size_t gain_gap_bit_bound,
                   size_t l2_regularization,
                   const Fr& split_penalty = Fr(0),
                   const Fr& split_gain_threshold = Fr(0),
                   size_t max_depth = 0,
                   size_t learning_rate_numerator = 1,
                   size_t learning_rate_denominator = 1,
                   forestTrainingLoss loss_mode = forestTrainingLoss::SecondOrderXgboost);

    void add_check(const forestTrainingProofWitness& witness,
                   size_t tree_count,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count,
                   size_t iteration_count,
                   size_t class_count,
                   size_t query_count,
                   size_t input_bit_bound,
                   size_t full_upper_bound,
                   size_t structure_bit_bound,
                   const fixedPointEncoding& gradient_encoding,
                   const fixedPointEncoding& gain_numerator_encoding,
                   const fixedPointEncoding& quantized_gain_encoding,
                   const fixedPointEncoding& leaf_value_encoding,
                   const fixedPointEncoding& score_encoding,
                   size_t hessian_bit_bound,
                   size_t gain_denominator_bit_bound,
                   size_t leaf_denominator_bit_bound,
                   size_t gain_gap_bit_bound,
                   size_t l2_regularization,
                   const Fr& split_penalty = Fr(0),
                   const Fr& split_gain_threshold = Fr(0),
                   size_t max_depth = 0,
                   size_t learning_rate_numerator = 1,
                   size_t learning_rate_denominator = 1)
    {
        add_check(witness,
                  tree_count,
                  node_count,
                  feature_count,
                  bin_count,
                  iteration_count,
                  class_count,
                  query_count,
                  input_bit_bound,
                  full_upper_bound,
                  structure_bit_bound,
                  gradient_encoding,
                  gain_numerator_encoding,
                  quantized_gain_encoding,
                  leaf_value_encoding,
                  score_encoding,
                  score_encoding,
                  hessian_bit_bound,
                  gain_denominator_bit_bound,
                  leaf_denominator_bit_bound,
                  gain_gap_bit_bound,
                  l2_regularization,
                  split_penalty,
                  split_gain_threshold,
                  max_depth,
                  learning_rate_numerator,
                  learning_rate_denominator,
                  forestTrainingLoss::SecondOrderXgboost);
    }

    bool finalize();

private:
    forestTrainingProofParam params_;
    std::vector<forestTrainingProofWitness> witnesses_;
    std::vector<forestTrainingProofCheckRecord> checks_;
};
