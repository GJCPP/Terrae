#pragma once

#include <vector>

#include "less_check.hpp"
#include "kzg.hpp"
#include "protocol_lifecycle.hpp"

using namespace mcl::bn;

class boxCheckParam {
public:
    kzgProtocol* kzg = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

// Proves row-major box membership:
//   lower_bounds <= values < upper_bounds
//
// The protocol reduces each coordinate to two bounded gaps:
// - values - lower_bounds
// - upper_bounds - 1 - values
//
// Example:
//   values       = [[5, 8],
//                   [2, 7]]
//   lower_bounds = [[4, 8],
//                   [1, 6]]
//   upper_bounds = [[7, 9],
//                   [3, 8]]
//
// Then each entry lies inside its half-open interval.
class boxCheckProtocol : protected ProtocolLifecycle {
public:
    ~boxCheckProtocol() noexcept
    {
        auto_finalize("boxCheckProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, lessCheckProtocol& less_check)
    {
        params_.kzg = &kzg;
        params_.less_check = &less_check;
        mark_setup();
    }

    void add_check(const virCommitment& values,
                   const virCommitment& lower_bounds,
                   const virCommitment& upper_bounds,
                   size_t row_count,
                   size_t column_count,
                   size_t bit_bound);

    bool finalize();

private:
    boxCheckParam params_;
};
