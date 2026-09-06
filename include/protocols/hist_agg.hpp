#pragma once

#include <vector>

#include "interleaved_batching.hpp"
#include "kzg.hpp"
#include "lagrange_utils.hpp"
#include "matrix_utils.hpp"
#include "protocol_lifecycle.hpp"
#include "profiling/soundness_counter.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

// Public index table `[0, 1, ..., table_domain_size - 1]` used by the lookup
// reduction that ties query indices to powers of the verifier challenge.
poly hist_agg_index_polynomial(size_t table_domain_size);
void preprocess_hist_agg_index_table(kzgProtocol& kzg, size_t table_domain_size);

// Power table `[1, r, r^2, ...]` over the histogram domain.
poly hist_agg_power_polynomial(const Fr& challenge, size_t domain_size);

class histAggCheckRecord {
public:
    size_t query_domain_size = 0;
    size_t table_domain_size = 0;
    Fr challenge = Fr(0);
    Fr histogram_claim = Fr(0);
    Fr query_claim = Fr(0);
    kzgCommitment power_table_commitment;
    kzgCommitment mapped_power_commitment;
};

class histAggParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    sumProtocol* sum_check = nullptr;
    tableLookupProtocol* table_lookup = nullptr;
    terraeSoundnessCounter* soundness_counter = nullptr;
};

class histAggVerifier {
public:
    Fr challenge();
    void add_check(const virCommitment& indices,
                   const virCommitment& values,
                   const virCommitment& histogram,
                   const histAggCheckRecord& record);
    bool finalize() const;

    std::vector<virCommitment> index_vectors;
    std::vector<virCommitment> value_vectors;
    std::vector<virCommitment> histograms;
    std::vector<histAggCheckRecord> checks;
    const histAggParam* params = nullptr;
    Fr sampled_challenge = Fr(0);
    bool has_sampled_challenge = false;
};

class histAggProver {
public:
    histAggCheckRecord add_check(const virCommitment& indices,
                                 const virCommitment& values,
                                 const virCommitment& histogram,
                                 const Fr& challenge,
                                 size_t query_domain_size,
                                 size_t table_domain_size);

    std::vector<histAggCheckRecord> checks;
    const histAggParam* params = nullptr;
};

// Proves that `histogram` is the aggregation of `values` keyed by `indices`.
//
// The reduction samples one challenge `r` and proves
//   sum_j histogram[j] * r^j = sum_i values[i] * r^{indices[i]}.
// A table lookup ties `indices[i]` to the corresponding `r^{indices[i]}` row.
class histAggProtocol : protected ProtocolLifecycle {
public:
    explicit histAggProtocol(tableLookupProtocol& table_lookup)
        : table_lookup_dependency_(&table_lookup)
    {
    }

    ~histAggProtocol() noexcept
    {
        auto_finalize("histAggProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero, sumProtocol& sum)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero;
        params_.sum_check = &sum;
        params_.table_lookup = table_lookup_dependency_;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void set_soundness_counter(terraeSoundnessCounter* counter) { params_.soundness_counter = counter; }

    void add_check(const virCommitment& indices,
                   const virCommitment& values,
                   const virCommitment& histogram,
                   size_t query_domain_size,
                   size_t table_domain_size);

    void add_check(const virCommitment& indices,
                   const std::vector<virCommitment>& values,
                   const std::vector<virCommitment>& histograms,
                   size_t query_domain_size,
                   size_t table_domain_size);

    bool finalize();

private:
    class pendingHistAggCheck {
    public:
        virCommitment indices;
        std::vector<virCommitment> values;
        std::vector<virCommitment> histograms;
        size_t query_domain_size = 0;
        size_t table_domain_size = 0;
    };

    histAggParam params_;
    histAggProver prover_;
    histAggVerifier verifier_;
    tableLookupProtocol* table_lookup_dependency_ = nullptr;
    std::vector<pendingHistAggCheck> pending_checks_;
};
