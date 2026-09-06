#pragma once

#include "fixed_point.hpp"
#include <vector>

#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "table_lookup.hpp"

using namespace mcl::bn;

void preprocess_non_positive_exp_check_table(kzgProtocol& kzg, const fixedPointEncoding& input_encoding);

class expCheckRecord {
public:
    size_t quantization_bits = 0;
    size_t domain_size = 0;
    bool uses_signed_input_encoding = false;
    bool assumes_non_positive_signed_inputs = false;
    fixedPointEncoding input_encoding;
};

class expCheckParam {
public:
    kzgProtocol* kzg = nullptr;
    tableLookupProtocol* table_lookup = nullptr;
};

class expCheckProtocol : protected ProtocolLifecycle {
public:
    ~expCheckProtocol() noexcept
    {
        auto_finalize("expCheckProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, tableLookupProtocol& table_lookup)
    {
        params_.kzg = &kzg;
        params_.table_lookup = &table_lookup;
        mark_setup();
    }

    void add_check(const virCommitment& input_values,
                   const virCommitment& output_values,
                   size_t quantization_bits,
                   size_t domain_size);

    // Signed overload: inputs are offset-binary encoded scaled integers.
    // Example: with total_bits = 4 and fractional_bits = 2, encoded input 9 means x = 0.25.
    void add_check(const virCommitment& input_values,
                   const virCommitment& output_values,
                   const fixedPointEncoding& input_encoding,
                   size_t domain_size);

    // Non-positive signed overload: inputs are offset-binary encoded scaled integers that
    // are already constrained elsewhere to decode to values <= 0. This is the stabilized
    // softmax path, so the exp table only needs to carry the non-positive half meaningfully.
    void add_non_positive_check(const virCommitment& input_values,
                                const virCommitment& output_values,
                                const fixedPointEncoding& input_encoding,
                                size_t domain_size);

    bool finalize();

private:
    expCheckParam params_;
    std::vector<virCommitment> input_values_;
    std::vector<virCommitment> output_values_;
    std::vector<expCheckRecord> checks_;
};
