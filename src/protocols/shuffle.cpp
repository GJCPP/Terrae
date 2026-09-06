#include "shuffle.hpp"

#include "protocol_profile.hpp"

#include <stdexcept>
#include <string>

namespace {

virCommitment compress_shuffle_row(const std::vector<virCommitment>& columns, const Fr& challenge)
{
    if (columns.empty()) {
        throw std::runtime_error("shuffleProtocol: column batch must be non-empty");
    }

    virCommitment compressed = columns.front();
    Fr power = challenge;
    for (size_t index = 1; index < columns.size(); ++index) {
        compressed = compressed + columns[index] * virCommitment(power);
        Fr::mul(power, power, challenge);
    }
    return compressed;
}

} // namespace

void shuffleProtocol::add_check(const std::vector<virCommitment>& lhs_columns,
                                const std::vector<virCommitment>& rhs_columns,
                                size_t row_count)
{
    structured_protocol_profile::timed_void(
        "protocol_call",
        "shuffle.add_check",
        "row_count=" + std::to_string(row_count) + ";column_count=" + std::to_string(lhs_columns.size()),
        [&] {
            if (!is_setup) {
                throw std::runtime_error("shuffleProtocol not setup");
            }
            if (kzg_ == nullptr || lookup_ == nullptr) {
                throw std::runtime_error("shuffleProtocol dependencies not setup");
            }
            if (lhs_columns.empty() || rhs_columns.empty() || lhs_columns.size() != rhs_columns.size()) {
                throw std::runtime_error("shuffleProtocol: lhs/rhs column batch size mismatch");
            }
            if (row_count == 0 || !isPowerOfTwo(row_count)) {
                throw std::runtime_error("shuffleProtocol: row_count must be a non-zero power of two");
            }

            pendingShuffleCheck pending;
            pending.lhs_columns = lhs_columns;
            pending.rhs_columns = rhs_columns;
            pending.row_count = row_count;
            pending_checks_.push_back(std::move(pending));
        });
}

bool shuffleProtocol::finalize()
{
    begin_finalize("shuffleProtocol");
    if (pending_checks_.empty()) {
        return true;
    }

    const Fr challenge = kzg_->transcript().squeeze_nonzero_fr("shuffle.alpha");
    for (const pendingShuffleCheck& check : pending_checks_) {
        const virCommitment lhs = compress_shuffle_row(check.lhs_columns, challenge);
        const virCommitment rhs = compress_shuffle_row(check.rhs_columns, challenge);
        lookup_->add_check(lhs,
                           rhs,
                           check.row_count,
                           check.row_count,
                           "shuffle.row_count=" + std::to_string(check.row_count) +
                               ";columns=" + std::to_string(check.lhs_columns.size()));
    }
    pending_checks_.clear();
    return true;
}
