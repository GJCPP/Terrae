#pragma once

#include <vector>

#include "kzg.hpp"
#include "less_check.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class divCheckRecord {
public:
    size_t bit_bound = 0;
    size_t divisor_power = 0;
    size_t domain_size = 0;
};

class divCheckParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

// Proves exact divisibility by a power of two together with the expected range
// bound on the quotient.
//
// Statement:
// - `values = quotient * 2^divisor_power`
// - `quotient` fits in `bit_bound - divisor_power` unsigned bits
class divCheckProtocol : protected ProtocolLifecycle {
public:
    ~divCheckProtocol() noexcept
    {
        auto_finalize("divCheckProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero, lessCheckProtocol& less)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.less_check = &less;
        mark_setup();
    }

    // Register one divisibility check over a shared power-of-two domain.
    void add_check(const virCommitment& values, size_t bit_bound, size_t divisor_power, size_t domain_size);

    bool finalize();

private:
    divCheckParam params_;
    std::vector<virCommitment> values_;
    std::vector<divCheckRecord> checks_;
};
