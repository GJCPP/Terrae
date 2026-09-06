#pragma once

#include <optional>
#include <vector>

#include "kzg.hpp"
#include "protocol_lifecycle.hpp"
#include "table_lookup.hpp"

using namespace mcl::bn;

class concatenatedTableLookupBlock {
public:
    concatenatedTableLookupBlock(const virCommitment& query, const virCommitment& table, size_t column_count)
        : query(query)
        , table(table)
        , column_count(column_count)
    {
    }

    virCommitment query;
    virCommitment table;
    size_t column_count = 0;
};

class concatenatedTableLookupCheckRecord {
public:
    size_t query_row_count = 0;
    size_t table_row_count = 0;
    size_t common_column_count = 0;
    Fr alpha = Fr(0);
    std::vector<size_t> column_counts;
};

class concatenatedTableLookupParam {
public:
    kzgProtocol* kzg = nullptr;
    tableLookupProtocol* table_lookup = nullptr;
};

// Proves that several query matrices all come from the same row positions across
// several table matrices with matching row counts but heterogeneous widths.
//
// Each block is a matrix lookup statement:
//   query_block ⊆ table_block
//
// Internally it aligns all blocks to a common width `m = lcm(m_i)`, forms one
// masked random linear combination over that width, and reduces the whole call
// to exactly one `tableLookupProtocol` instance.
class concatenatedTableLookupProtocol : protected ProtocolLifecycle {
public:
    ~concatenatedTableLookupProtocol() noexcept
    {
        auto_finalize("concatenatedTableLookupProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg, tableLookupProtocol& table_lookup)
    {
        params_.kzg = &kzg;
        params_.table_lookup = &table_lookup;
        mark_setup();
    }

    void add_check(const std::vector<concatenatedTableLookupBlock>& blocks,
                   size_t query_row_count,
                   size_t table_row_count);

    void add_check(const std::vector<concatenatedTableLookupBlock>& blocks,
                   size_t query_row_count,
                   size_t table_row_count,
                   const tableLookupWitness& witness);

    bool finalize();

private:
    void add_check_with_optional_witness(const std::vector<concatenatedTableLookupBlock>& blocks,
                                         size_t query_row_count,
                                         size_t table_row_count,
                                         const tableLookupWitness* witness);

    concatenatedTableLookupParam params_;
    std::vector<std::vector<concatenatedTableLookupBlock>> block_checks_;
    std::vector<concatenatedTableLookupCheckRecord> records_;
    std::vector<std::optional<tableLookupWitness>> witnesses_;
};
