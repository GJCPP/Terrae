#pragma once

#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class rationalCheckParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
};

// Proves multiplicative division identities over a shared evaluation domain.
//
// Unmasked statement:
//   numerator = denominator * quotient
//
// Masked statement:
//   mask is boolean, and
//   mask * (numerator - denominator * quotient) = 0
//
// Example masked row:
//   numerator   = 0
//   denominator = 0
//   quotient    = 17
//   mask        = 0
//
// The row is disabled, so the division relation is not enforced there.
class rationalCheckProtocol : protected ProtocolLifecycle {
public:
    ~rationalCheckProtocol() noexcept
    {
        auto_finalize("rationalCheckProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        mark_setup();
    }

    void add_check(const virCommitment& numerator,
                   const virCommitment& denominator,
                   const virCommitment& quotient,
                   size_t domain_size);

    void add_masked_check(const virCommitment& numerator,
                          const virCommitment& denominator,
                          const virCommitment& quotient,
                          const virCommitment& mask,
                          size_t domain_size);

    bool finalize();

private:
    rationalCheckParam params_;
};
