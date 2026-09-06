#pragma once

#include "cube_argmax.hpp"
#include "cube_softmax.hpp"
#include "exp_check.hpp"
#include "fixed_point.hpp"
#include "forest_training_loss.hpp"
#include "fixed_point_division.hpp"
#include "matrix_sum.hpp"
#include "protocol_lifecycle.hpp"
#include "signed_range_check.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestBoostingStateWitness {
public:
    virCommitment tree_scores;
    virCommitment cumulative_scores;
    virCommitment encoded_cumulative_scores;
    virCommitment softmax_shifts;
    virCommitment shift_selector;
    virCommitment lifted_softmax_shifts;
    virCommitment encoded_shifted_scores;
    virCommitment softmax_exp_values;
    virCommitment softmax_denominators;
    virCommitment lifted_softmax_denominators;
    virCommitment softmax_probabilities;
    virCommitment softmax_remainders;
    virCommitment base_labels;
    virCommitment pseudo_residuals;
    virCommitment encoded_diagonal_hessians;
    virCommitment diagonal_hessian_remainders;
    virCommitment diagonal_hessians;
};

class forestBoostingStateCheckRecord {
public:
    size_t iteration_count = 0;
    size_t class_count = 0;
    size_t query_count = 0;
    fixedPointEncoding score_encoding;
    fixedPointEncoding softmax_encoding;
    size_t hessian_bit_bound = 0;
    forestTrainingLoss loss_mode = forestTrainingLoss::SecondOrderXgboost;
};

class forestBoostingStateParam {
public:
    zeroProtocol* zero_check = nullptr;
    matrixColumnPrefixSumProtocol* column_prefix = nullptr;
    matrixColumnSumProtocol* column_sum = nullptr;
    cubeMiddleArgmaxProtocol* cube_argmax = nullptr;
    cubeMiddleReplicationProtocol* replication = nullptr;
    cubeSumProtocol* cube_sum = nullptr;
    expCheckProtocol* exp_check = nullptr;
    fixedPointDivisionProtocol* fixed_point_division = nullptr;
    signedRangeCheckProtocol* signed_range_check = nullptr;
};

// Proves one global boosting-state layer over the `(iteration, class, sample)`
// score cube:
// - cumulative scores are the iteration-wise prefix sums of per-tree scores,
// - encoded cumulative scores are the signed fixed-point encoding of those sums,
// - encoded shifted scores are derived from row-max shifts over the class axis,
// - quantized exp outputs and denominators are accumulated over the class axis,
// - softmax probabilities are fixed-point encoded probabilities proven via integer division,
// - pseudo-residuals equal `probability_scaled - label * scale`,
// - and diagonal Hessians are fixed-point encoded values derived from
//   `floor(probability_scaled * (scale - probability_scaled) / scale)`.
//
// The label matrix is committed once over the base `(class, sample)` layout and
// then lifted across the iteration axis inside the reduction.
class forestBoostingStateProtocol : protected ProtocolLifecycle {
public:
    ~forestBoostingStateProtocol() noexcept
    {
        auto_finalize("forestBoostingStateProtocol", [this] { return finalize(); });
    }

    void setup(zeroProtocol& zero_check,
               matrixColumnPrefixSumProtocol& column_prefix,
               matrixColumnSumProtocol& column_sum,
               cubeMiddleArgmaxProtocol& cube_argmax,
               cubeMiddleReplicationProtocol& replication,
               cubeSumProtocol& cube_sum,
               expCheckProtocol& exp_check,
               fixedPointDivisionProtocol& fixed_point_division,
               signedRangeCheckProtocol& signed_range_check)
    {
        params_.zero_check = &zero_check;
        params_.column_prefix = &column_prefix;
        params_.column_sum = &column_sum;
        params_.cube_argmax = &cube_argmax;
        params_.replication = &replication;
        params_.cube_sum = &cube_sum;
        params_.exp_check = &exp_check;
        params_.fixed_point_division = &fixed_point_division;
        params_.signed_range_check = &signed_range_check;
        mark_setup();
    }

    void add_check(const forestBoostingStateWitness& witness,
                   size_t iteration_count,
                   size_t class_count,
                   size_t query_count,
                   const fixedPointEncoding& score_encoding,
                   const fixedPointEncoding& softmax_encoding,
                   size_t hessian_bit_bound,
                   forestTrainingLoss loss_mode = forestTrainingLoss::SecondOrderXgboost);

    void add_check(const forestBoostingStateWitness& witness,
                   size_t iteration_count,
                   size_t class_count,
                   size_t query_count,
                   const fixedPointEncoding& score_encoding)
    {
        add_check(witness, iteration_count, class_count, query_count, score_encoding, score_encoding, 1);
    }

    bool finalize();

private:
    forestBoostingStateParam params_;
    std::vector<forestBoostingStateWitness> witnesses_;
    std::vector<forestBoostingStateCheckRecord> checks_;
};
