#include "algebra_utils.hpp"
#include "column_replication.hpp"

#include <stdexcept>

poly replication_row_vanishing_factor(size_t n, size_t m)
{
    if (!isPowerOfTwo(n) || !isPowerOfTwo(m) || n == 0 || m == 0) {
        throw std::runtime_error("replication_row_vanishing_factor: n and m must be powers of two");
    }

    poly factor;
    for (size_t j = 0; j < m; ++j) {
        factor.add_term(j * n, Fr(1));
    }
    return factor;
}

void columnReplicationProtocol::add_check(const virCommitment& f,
                                          const virCommitment& f_hat,
                                          size_t n,
                                          size_t m)
{
    proof_cost::record_relation("replication", n * m);
    if (!is_setup) {
        throw std::runtime_error("columnReplicationProtocol not setup");
    }
    if (params.zero_check == nullptr) {
        throw std::runtime_error("columnReplicationProtocol zeroProtocol not setup");
    }
    if (!isPowerOfTwo(n) || !isPowerOfTwo(m) || n == 0 || m == 0) {
        throw std::runtime_error("columnReplicationProtocol: n and m must be powers of two");
    }

    const size_t nm = n * m;
    const Fr row_shift = inverse_root_of_unity(nm);

    // On H_nm \ H_n, replicated rows satisfy f_hat(wX) - f_hat(X) = 0.
    // Multiplying by z_Hn makes the same expression vanish on all of H_nm.
    virCommitment row_zero = (x_scale(f_hat, row_shift) - f_hat) * virCommitment(vanishing_poly(n));
    params.zero_check->add_check(row_zero, nm);

    // Column correctness means f_hat agrees with f on H_n.
    virCommitment column_zero = f_hat - f;
    params.zero_check->add_check(column_zero, n);
    ++params.check_count;
}

bool columnReplicationProtocol::finalize()
{
    begin_finalize("columnReplicationProtocol");

    return true;
}
