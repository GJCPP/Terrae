#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"
#include "polynomial.hpp"
#include "protocol_lifecycle.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

// Vanishing factor for the relation
//   replicated(row, col) - base(row) = 0
// on every non-first column of each row.
poly replication_row_vanishing_factor(size_t n, size_t m);

class columnReplicationParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    size_t check_count = 0;
};

// Proves that `f_hat` is obtained by replicating the length-`n` vector `f`
// across `m` columns in row-major order.
//
// Example:
//   f     = [a, b]
//   f_hat = [[a, a, a],
//            [b, b, b]]
class columnReplicationProtocol : protected ProtocolLifecycle {
public:
    ~columnReplicationProtocol() noexcept
    {
        auto_finalize("columnReplicationProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, zeroProtocol& zero)
    {
        params.kzg = &kzg;
        params.zero_check = &zero;
        mark_setup();
    }

    // Register one replication statement over an `n x m` row-major matrix.
    void add_check(const virCommitment& f, const virCommitment& f_hat, size_t n, size_t m);

    bool finalize();

protected:
    columnReplicationParam params;
};
