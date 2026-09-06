#pragma once

#include <string>
#include <vector>

#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class nonzeroCheckRecord {
public:
    size_t domain_size = 0;
    bool uses_mask = false;
    std::string label;
};

class nonzeroCheckProtocol : protected ProtocolLifecycle {
public:
    ~nonzeroCheckProtocol() noexcept
    {
        auto_finalize("nonzeroCheckProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero_check)
    {
        kzg_ = &kzg;
        zero_check_ = &zero_check;
        mark_setup();
    }

    void add_check(const virCommitment& values, size_t domain_size, const std::string& label = "");

    void add_masked_check(const virCommitment& values,
                          const virCommitment& mask,
                          size_t domain_size,
                          const std::string& label = "");

    bool finalize();

private:
    kzgProtocol* kzg_ = nullptr;
    zeroProtocol* zero_check_ = nullptr;
    std::vector<virCommitment> values_;
    std::vector<virCommitment> masks_;
    std::vector<nonzeroCheckRecord> checks_;
};
