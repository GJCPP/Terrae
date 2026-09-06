#pragma once

#include <vector>

#include "fixed_point.hpp"
#include "less_check.hpp"
#include "protocol_lifecycle.hpp"

using namespace mcl::bn;

class signedRangeCheckParam {
public:
    kzgProtocol* kzg = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

void add_signed_range_check_srs_requirement(lessCheckSrsEstimator& less_check,
                                            const fixedPointEncoding& encoding,
                                            size_t domain_size);

// First iteration of signed fixed-point support.
//
// We keep the witness in its offset-binary encoded unsigned form and prove
// that it stays inside the valid unsigned interval for the chosen encoding.
//
// Example with total_bits = 4:
// - signed scaled values live in [-8, 7]
// - encoded witness values live in [0, 15]
class signedRangeCheckProtocol : protected ProtocolLifecycle {
public:
    ~signedRangeCheckProtocol() noexcept
    {
        auto_finalize("signedRangeCheckProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, lessCheckProtocol& less)
    {
        params_.kzg = &kzg;
        params_.less_check = &less;
        mark_setup();
    }

    void add_check(const virCommitment& encoded_values, const fixedPointEncoding& encoding, size_t domain_size);

    bool finalize();

private:
    signedRangeCheckParam params_;
};
