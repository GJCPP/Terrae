#pragma once

#include "fixed_point.hpp"
#include "forest_training_loss.hpp"
#include "less_check.hpp"
#include "nonzero_check.hpp"
#include "protocol_lifecycle.hpp"
#include "signed_range_check.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestSplitGainWitness {
public:
    virCommitment left_counts;
    virCommitment right_counts;
    virCommitment left_gradients;
    virCommitment right_gradients;
    virCommitment left_hessians;
    virCommitment right_hessians;
    virCommitment left_gradient_squares;
    virCommitment right_gradient_squares;
    virCommitment parent_gradient_squares;
    virCommitment gain_numerators;
    virCommitment gain_denominators;
};

class forestSplitGainCheckRecord {
public:
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    fixedPointEncoding gradient_encoding;
    size_t hessian_bit_bound = 0;
    size_t denominator_bit_bound = 0;
    size_t l2_regularization = 0;
    Fr split_penalty = Fr(0);
    forestTrainingLoss loss_mode = forestTrainingLoss::SecondOrderXgboost;
    size_t first_order_hessian_scale = 1;
};

class forestSplitGainParam {
public:
    zeroProtocol* zero_check = nullptr;
    lessCheckProtocol* less_check = nullptr;
    signedRangeCheckProtocol* signed_range_check = nullptr;
    nonzeroCheckProtocol* nonzero_check = nullptr;
};

// Proves the exact regularized split-gain rational form over the current
// `(node, feature, bin)` candidate layout.
//
// Inputs:
// - left/right gradient statistics
// - left/right Hessian statistics
// - claimed gradient-square witnesses for left/right/parent
// - claimed exact gain numerator / denominator witnesses
//
// With public parameters:
// - `lambda = l2_regularization`
// - `split_penalty`
//
// The protocol proves:
// - `parent_gradient = left_gradient + right_gradient`
// - `parent_hessian = left_hessian + right_hessian`
// - each claimed square is the square of its corresponding gradient statistic
// - `gain_denominator =
//      (left_hessian + lambda) *
//      (right_hessian + lambda) *
//      (parent_hessian + lambda)`
// - `gain_numerator =
//      left_gradient^2  * (right_hessian + lambda) * (parent_hessian + lambda) +
//      right_gradient^2 * (left_hessian  + lambda) * (parent_hessian + lambda) -
//      parent_gradient^2 * (left_hessian + lambda) * (right_hessian + lambda) -
//      split_penalty * gain_denominator`
//
// In first-order RSS mode, the constant Hessian scale S is cancelled
// exactly using the already-certified count histograms:
// - gain_denominator = S * left_count * right_count * parent_count
// - gain_numerator uses the same formula above with counts in place of
//   Hessians and with zero regularization/penalty.
// This divides both the old numerator and denominator by S^2, so the
// rational gain and every subsequent quantized comparison are unchanged.
//
// This keeps gain computation exact as a rational pair. Quantization or
// argmax-style candidate selection can be layered above it later.
class forestSplitGainProtocol : protected ProtocolLifecycle {
public:
    ~forestSplitGainProtocol() noexcept
    {
        auto_finalize("forestSplitGainProtocol", [this] { return finalize(); });
    }

    void setup(zeroProtocol& zero_check, lessCheckProtocol& less_check, signedRangeCheckProtocol& signed_range_check)
    {
        setup(zero_check, less_check, signed_range_check, nullptr);
    }

    void setup(zeroProtocol& zero_check,
               lessCheckProtocol& less_check,
               signedRangeCheckProtocol& signed_range_check,
               nonzeroCheckProtocol& nonzero_check)
    {
        setup(zero_check, less_check, signed_range_check, &nonzero_check);
    }

private:
    void setup(zeroProtocol& zero_check,
               lessCheckProtocol& less_check,
               signedRangeCheckProtocol& signed_range_check,
               nonzeroCheckProtocol* nonzero_check)
    {
        params_.zero_check = &zero_check;
        params_.less_check = &less_check;
        params_.signed_range_check = &signed_range_check;
        params_.nonzero_check = nonzero_check;
        mark_setup();
    }

public:

    void add_check(const forestSplitGainWitness& witness,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count,
                   const fixedPointEncoding& gradient_encoding,
                   size_t hessian_bit_bound,
                   size_t denominator_bit_bound,
                   size_t l2_regularization,
                   const Fr& split_penalty = Fr(0));

    void add_check(const forestSplitGainWitness& witness,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count,
                   const fixedPointEncoding& gradient_encoding,
                   size_t hessian_bit_bound,
                   size_t denominator_bit_bound,
                   size_t l2_regularization,
                   const Fr& split_penalty,
                   forestTrainingLoss loss_mode,
                   const virCommitment& candidate_mask,
                   size_t first_order_hessian_scale);

    bool finalize();

private:
    forestSplitGainParam params_;
    std::vector<forestSplitGainWitness> witnesses_;
    std::vector<forestSplitGainCheckRecord> checks_;
    std::vector<virCommitment> candidate_masks_;
};
