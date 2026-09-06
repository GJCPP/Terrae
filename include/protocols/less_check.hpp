#pragma once

#include <map>
#include <vector>

#include "kzg.hpp"
#include "lookup.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class lessCheckRecord {
public:
    size_t bit_bound = 0;
    size_t domain_size = 0;
    bool uses_mask = false;
};

class lessCheckParam {
public:
    kzgProtocol* kzg = nullptr;
    lookupProtocol* lookup_check = nullptr;
    zeroProtocol* zero_check = nullptr;
};

size_t less_check_limb_bits();
std::vector<size_t> less_check_range_table_bit_bounds(size_t bit_bound);
void preprocess_less_check_range_tables(kzgProtocol& kzg, const std::vector<size_t>& bit_bounds);

class lessCheckSrsEstimator {
public:
    void add_check(size_t bit_bound, size_t domain_size, size_t count = 1);
    size_t required_domain_size() const;

private:
    std::map<size_t, std::vector<size_t>> domain_sizes_by_bit_bound_;
};

// Proves that each entry lies in the unsigned interval `[0, 2^bit_bound)`.
//
// Small ranges use a direct public lookup table. Larger ranges are decomposed
// into fixed-width limbs, each checked against a small public table, with a
// zero-check tying the limbs back to the original value.
class lessCheckProtocol : protected ProtocolLifecycle {
public:
    ~lessCheckProtocol() noexcept
    {
        auto_finalize("lessCheckProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, lookupProtocol& lookup)
    {
        params_.kzg = &kzg;
        params_.lookup_check = &lookup;
        mark_setup();
    }

    void setup(kzgProtocol& kzg, lookupProtocol& lookup, zeroProtocol& zero)
    {
        params_.kzg = &kzg;
        params_.lookup_check = &lookup;
        params_.zero_check = &zero;
        mark_setup();
    }

    // Unmasked range check.
    void add_check(const virCommitment& values, size_t bit_bound, size_t domain_size);

    // Masked range check. Inactive rows are multiplied by `mask` before the
    // lookup reduction, so only active rows must lie in range.
    void add_masked_check(const virCommitment& values,
                          const virCommitment& mask,
                          size_t bit_bound,
                          size_t domain_size);

    bool finalize();

private:
    lessCheckParam params_;
    std::vector<virCommitment> values_;
    std::vector<virCommitment> masks_;
    std::vector<lessCheckRecord> checks_;
};
