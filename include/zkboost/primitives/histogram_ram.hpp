#pragma once

#include <cstddef>
#include <vector>

#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "shuffle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class zkboostHistogramRamShape {
public:
    size_t query_domain_size = 0;
    size_t bucket_count = 0;
    size_t trace_domain_size = 0;
};

class zkboostHistogramRamRecord {
public:
    zkboostHistogramRamShape shape;

    kzgCommitment unsorted_read_commitment;
    kzgCommitment unsorted_write_commitment;
    kzgCommitment sorted_addr_commitment;
    kzgCommitment sorted_time_commitment;
    kzgCommitment sorted_read_commitment;
    kzgCommitment sorted_write_commitment;
    kzgCommitment sorted_delta_commitment;
    kzgCommitment sorted_key_diff_commitment;
    kzgCommitment sorted_addr_gap_commitment;
    kzgCommitment sorted_same_addr_commitment;
};

class zkboostHistogramRamParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    lessCheckProtocol* less_check = nullptr;
    shuffleProtocol* shuffle_check = nullptr;
};

// Non-PLONK RAM-style histogram proof for
//   s[i] = sum_{j: index[j] = i} val[j].
//
// The prover builds a RAM update trace with zero-initialized buckets, one
// read/write update per input row, and one terminal read per bucket. A shuffle
// proves equality between canonical operation rows and rows sorted by
// `(address, time)`. Range and zero checks prove sorted memory consistency.
class zkboostHistogramRamProtocol : protected ProtocolLifecycle {
public:
    ~zkboostHistogramRamProtocol() noexcept
    {
        auto_finalize("zkboostHistogramRamProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero,
               lessCheckProtocol& less,
               shuffleProtocol& shuffle)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.less_check = &less;
        params_.shuffle_check = &shuffle;
        mark_setup();
    }

    void add_check(const virCommitment& indices,
                   const virCommitment& values,
                   const virCommitment& sums,
                   size_t query_domain_size,
                   size_t bucket_count);

    bool finalize();

private:
    zkboostHistogramRamParam params_;
    std::vector<zkboostHistogramRamRecord> records_;
};

zkboostHistogramRamShape zkboost_histogram_ram_shape(size_t query_domain_size, size_t bucket_count);

