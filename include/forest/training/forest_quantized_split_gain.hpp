#pragma once

#include "fixed_point.hpp"
#include "fixed_point_division.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class forestQuantizedSplitGainWitness {
public:
    virCommitment exact_gain_numerators;
    virCommitment gain_denominators;
    virCommitment encoded_gain_numerators;
    virCommitment encoded_quantized_gains;
    virCommitment gain_remainders;
};

class forestQuantizedSplitGainCheckRecord {
public:
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bin_count = 0;
    fixedPointEncoding numerator_encoding;
    fixedPointEncoding quantized_gain_encoding;
    size_t denominator_bit_bound = 0;
    bool uses_mask = false;
};

class forestQuantizedSplitGainParam {
public:
    zeroProtocol* zero_check = nullptr;
    fixedPointDivisionProtocol* fixed_point_division = nullptr;
};

// Proves a quantized split-gain witness derived from the exact gain rational
// form produced by `forestSplitGainProtocol`.
//
// The protocol keeps the exact gain proof and the quantized comparison layer
// separate:
// - `exact_gain_numerators / gain_denominators` is the exact rational gain
// - `encoded_quantized_gains` is the fixed-point integer obtained by the chosen
//   floor-style quantization rule
//
// The caller also provides `encoded_gain_numerators`, which is just the exact
// numerator re-encoded in a signed fixed-point format, so the existing signed
// division layer can be reused directly.
class forestQuantizedSplitGainProtocol : protected ProtocolLifecycle {
public:
    ~forestQuantizedSplitGainProtocol() noexcept
    {
        auto_finalize("forestQuantizedSplitGainProtocol", [this] { return finalize(); });
    }

    void setup(zeroProtocol& zero_check, fixedPointDivisionProtocol& fixed_point_division)
    {
        params_.zero_check = &zero_check;
        params_.fixed_point_division = &fixed_point_division;
        mark_setup();
    }

    void add_check(const forestQuantizedSplitGainWitness& witness,
                   size_t node_count,
                   size_t feature_count,
                   size_t bin_count,
                   const fixedPointEncoding& numerator_encoding,
                   const fixedPointEncoding& quantized_gain_encoding,
                   size_t denominator_bit_bound);

    void add_masked_check(const forestQuantizedSplitGainWitness& witness,
                          const virCommitment& candidate_mask,
                          size_t node_count,
                          size_t feature_count,
                          size_t bin_count,
                          const fixedPointEncoding& numerator_encoding,
                          const fixedPointEncoding& quantized_gain_encoding,
                          size_t denominator_bit_bound);

    bool finalize();

private:
    forestQuantizedSplitGainParam params_;
    std::vector<forestQuantizedSplitGainWitness> witnesses_;
    std::vector<virCommitment> candidate_masks_;
    std::vector<forestQuantizedSplitGainCheckRecord> checks_;
};
