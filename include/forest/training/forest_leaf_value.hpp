#pragma once

#include "fixed_point.hpp"
#include "forest_training_loss.hpp"
#include "fixed_point_division.hpp"
#include "less_check.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestLeafValueWitness {
public:
    virCommitment leaf_gradients;
    virCommitment leaf_hessians;
    virCommitment leaf_denominators;
    virCommitment encoded_negative_leaf_gradients;
    virCommitment encoded_leaf_values;
    virCommitment leaf_remainders;
};

class forestLeafValueCheckRecord {
public:
    size_t node_count = 0;
    fixedPointEncoding gradient_encoding;
    fixedPointEncoding leaf_value_encoding;
    size_t hessian_bit_bound = 0;
    size_t denominator_bit_bound = 0;
    size_t l2_regularization = 0;
    size_t learning_rate_numerator = 1;
    size_t learning_rate_denominator = 1;
    forestTrainingLoss loss_mode = forestTrainingLoss::SecondOrderXgboost;
};

class forestLeafValueParam {
public:
    zeroProtocol* zero_check = nullptr;
    lessCheckProtocol* less_check = nullptr;
    fixedPointDivisionProtocol* fixed_point_division = nullptr;
};

// Proves the usual regularized leaf value
//   w = floor((-G * scale * learning_rate_numerator) / ((H + lambda) * learning_rate_denominator))
// under explicit fixed-point encodings.
class forestLeafValueProtocol : protected ProtocolLifecycle {
public:
    ~forestLeafValueProtocol() noexcept
    {
        auto_finalize("forestLeafValueProtocol", [this] { return finalize(); });
    }

    void setup(zeroProtocol& zero_check,
               lessCheckProtocol& less_check,
               fixedPointDivisionProtocol& fixed_point_division)
    {
        params_.zero_check = &zero_check;
        params_.less_check = &less_check;
        params_.fixed_point_division = &fixed_point_division;
        mark_setup();
    }

    void add_check(const forestLeafValueWitness& witness,
                   size_t node_count,
                   const fixedPointEncoding& gradient_encoding,
                   const fixedPointEncoding& leaf_value_encoding,
                   size_t hessian_bit_bound,
                   size_t denominator_bit_bound,
                   size_t l2_regularization,
                   size_t learning_rate_numerator = 1,
                   size_t learning_rate_denominator = 1);

    void add_check(const forestLeafValueWitness& witness,
                   size_t node_count,
                   const fixedPointEncoding& gradient_encoding,
                   const fixedPointEncoding& leaf_value_encoding,
                   size_t hessian_bit_bound,
                   size_t denominator_bit_bound,
                   size_t l2_regularization,
                   size_t learning_rate_numerator,
                   size_t learning_rate_denominator,
                   forestTrainingLoss loss_mode,
                   const virCommitment& node_leaf_mask);

    bool finalize();

private:
    forestLeafValueParam params_;
    std::vector<forestLeafValueWitness> witnesses_;
    std::vector<forestLeafValueCheckRecord> checks_;
    std::vector<virCommitment> node_leaf_masks_;
};
