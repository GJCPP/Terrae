#pragma once

#include <vector>

#include "kzg.hpp"
#include "lookup.hpp"
#include "protocol_lifecycle.hpp"

using namespace mcl::bn;

// Proves that two tables with the same row count are row permutations of each
// other. Rows are first compressed to one scalar with a verifier challenge,
// then reduced to one multiset-equality lookup over the compressed rows.
class shuffleProtocol : protected ProtocolLifecycle {
public:
    ~shuffleProtocol() noexcept
    {
        auto_finalize("shuffleProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, lookupProtocol& lookup)
    {
        kzg_ = &kzg;
        lookup_ = &lookup;
        mark_setup();
    }

    void add_check(const std::vector<virCommitment>& lhs_columns,
                   const std::vector<virCommitment>& rhs_columns,
                   size_t row_count);

    void add_check(const virCommitment& lhs, const virCommitment& rhs, size_t row_count)
    {
        add_check(std::vector<virCommitment>{lhs}, std::vector<virCommitment>{rhs}, row_count);
    }

    bool finalize();

private:
    class pendingShuffleCheck {
    public:
        std::vector<virCommitment> lhs_columns;
        std::vector<virCommitment> rhs_columns;
        size_t row_count = 0;
    };

    kzgProtocol* kzg_ = nullptr;
    lookupProtocol* lookup_ = nullptr;
    std::vector<pendingShuffleCheck> pending_checks_;
};
