#include "concatenated_table_lookup.hpp"

#include "lagrange_utils.hpp"
#include "matrix_utils.hpp"

#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_concatenated_table_lookup_shape(const std::vector<concatenatedTableLookupBlock>& blocks,
                                              size_t query_row_count,
                                              size_t table_row_count)
{
    if (blocks.empty()) {
        throw std::runtime_error("concatenatedTableLookupProtocol: at least one block is required");
    }

    for (const concatenatedTableLookupBlock& block : blocks) {
        validate_matrix_shape(query_row_count, block.column_count, "concatenatedTableLookupProtocol");
        validate_matrix_shape(table_row_count, block.column_count, "concatenatedTableLookupProtocol");
    }
}

size_t common_column_count_for_blocks(const std::vector<concatenatedTableLookupBlock>& blocks)
{
    size_t common_column_count = blocks.front().column_count;
    for (size_t i = 1; i < blocks.size(); ++i) {
        common_column_count = std::lcm(common_column_count, blocks[i].column_count);
    }
    return common_column_count;
}

virCommitment aligned_mask_expression(size_t row_count, size_t common_column_count, size_t original_column_count)
{
    if (common_column_count == 0 || original_column_count == 0 || common_column_count % original_column_count != 0) {
        throw std::runtime_error("concatenatedTableLookupProtocol: invalid aligned mask shape");
    }

    // A column-only mask repeated across every row of an (row_count x common_column_count) matrix.
    // `L_0^{stride}(X^(row_count * original_column_count))` selects exactly columns
    // 0, stride, 2*stride, ... without interpolating the width-only eval table.
    const size_t stride = common_column_count / original_column_count;
    return x_pow(subgroup_lagrange_expression(stride, 0), row_count * original_column_count);
}

std::string field_key(const Fr& value)
{
    return value.getStr(10);
}

std::string concatenated_row_key(const std::vector<std::vector<Fr>>& eval_blocks,
                                 const std::vector<concatenatedTableLookupBlock>& blocks,
                                 size_t row,
                                 size_t row_count)
{
    if (row >= row_count) {
        throw std::runtime_error("concatenatedTableLookupProtocol: row index out of range");
    }

    std::string key;
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        if (block_index != 0) {
            key += "||";
        }

        const size_t column_count = blocks[block_index].column_count;
        for (size_t column = 0; column < column_count; ++column) {
            if (column != 0) {
                key.push_back('|');
            }
            key += field_key(eval_blocks[block_index][matrix_index(row, column, column_count)]);
        }
    }
    return key;
}

tableLookupWitness build_joint_lookup_witness(kzgProtocol& kzg,
                                              const std::vector<concatenatedTableLookupBlock>& blocks,
                                              size_t query_row_count,
                                              size_t table_row_count)
{
    std::vector<std::vector<Fr>> query_evals;
    std::vector<std::vector<Fr>> table_evals;
    query_evals.reserve(blocks.size());
    table_evals.reserve(blocks.size());

    for (const concatenatedTableLookupBlock& block : blocks) {
        query_evals.push_back(kzg.evaluations(block.query,
                                              query_row_count * block.column_count,
                                              kzgProtocol::EvaluationMode::PreferEval));
        table_evals.push_back(kzg.evaluations(block.table,
                                              table_row_count * block.column_count,
                                              kzgProtocol::EvaluationMode::PreferEval));
    }

    std::map<std::string, size_t> first_row_by_key;
    for (size_t table_row = 0; table_row < table_row_count; ++table_row) {
        first_row_by_key.emplace(concatenated_row_key(table_evals, blocks, table_row, table_row_count), table_row);
    }

    tableLookupWitness witness;
    witness.row_labels.reserve(query_row_count);

    for (size_t query_row = 0; query_row < query_row_count; ++query_row) {
        const auto it = first_row_by_key.find(concatenated_row_key(query_evals, blocks, query_row, query_row_count));
        witness.row_labels.push_back(subgroup_label(table_row_count, it == first_row_by_key.end() ? 0 : it->second));
    }

    return witness;
}

virCommitment combined_aligned_expression(const std::vector<concatenatedTableLookupBlock>& blocks,
                                          size_t row_count,
                                          size_t common_column_count,
                                          const Fr& alpha,
                                          bool use_query_side)
{
    virCommitment combined(Fr(0));
    Fr alpha_power = alpha;

    for (const concatenatedTableLookupBlock& block : blocks) {
        const virCommitment mask = aligned_mask_expression(row_count, common_column_count, block.column_count);
        const virCommitment& source = use_query_side ? block.query : block.table;
        combined = combined + scale_by_constant(mask * source, alpha_power);
        Fr::mul(alpha_power, alpha_power, alpha);
    }

    return combined;
}

} // namespace

void concatenatedTableLookupProtocol::add_check(const std::vector<concatenatedTableLookupBlock>& blocks,
                                                size_t query_row_count,
                                                size_t table_row_count)
{
    add_check_with_optional_witness(blocks, query_row_count, table_row_count, nullptr);
}

void concatenatedTableLookupProtocol::add_check(const std::vector<concatenatedTableLookupBlock>& blocks,
                                                size_t query_row_count,
                                                size_t table_row_count,
                                                const tableLookupWitness& witness)
{
    add_check_with_optional_witness(blocks, query_row_count, table_row_count, &witness);
}

void concatenatedTableLookupProtocol::add_check_with_optional_witness(
    const std::vector<concatenatedTableLookupBlock>& blocks,
    size_t query_row_count,
    size_t table_row_count,
    const tableLookupWitness* witness)
{
    if (!is_setup) {
        throw std::runtime_error("concatenatedTableLookupProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.table_lookup == nullptr) {
        throw std::runtime_error("concatenatedTableLookupProtocol dependencies not setup");
    }

    validate_concatenated_table_lookup_shape(blocks, query_row_count, table_row_count);
    if (witness != nullptr && witness->row_labels.size() != query_row_count) {
        throw std::runtime_error("concatenatedTableLookupProtocol: witness row count mismatch");
    }

    concatenatedTableLookupCheckRecord record;
    record.query_row_count = query_row_count;
    record.table_row_count = table_row_count;
    record.common_column_count = common_column_count_for_blocks(blocks);
    record.alpha = params_.kzg->transcript().squeeze_nonzero_fr("concatenated_table_lookup.alpha");
    record.column_counts.reserve(blocks.size());
    for (const concatenatedTableLookupBlock& block : blocks) {
        record.column_counts.push_back(block.column_count);
    }

    block_checks_.push_back(blocks);
    records_.push_back(std::move(record));
    if (witness == nullptr) {
        witnesses_.push_back(std::nullopt);
    } else {
        witnesses_.push_back(*witness);
    }
}

bool concatenatedTableLookupProtocol::finalize()
{
    begin_finalize("concatenatedTableLookupProtocol");

    for (size_t check_index = 0; check_index < records_.size(); ++check_index) {
        const concatenatedTableLookupCheckRecord& record = records_[check_index];
        const std::vector<concatenatedTableLookupBlock>& blocks = block_checks_[check_index];
        const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
        const bool verifier_mode = proof_session && proof_session->mode() == ProofMode::Verify;

        const virCommitment combined_query =
            combined_aligned_expression(blocks, record.query_row_count, record.common_column_count, record.alpha, true);
        const virCommitment combined_table =
            combined_aligned_expression(blocks, record.table_row_count, record.common_column_count, record.alpha, false);

        if (verifier_mode) {
            params_.table_lookup->add_check(
                combined_query, combined_table, record.query_row_count, record.table_row_count, record.common_column_count);
        } else {
            const tableLookupWitness witness = witnesses_[check_index].has_value()
                                                   ? *witnesses_[check_index]
                                                   : build_joint_lookup_witness(
                                                         *params_.kzg, blocks, record.query_row_count, record.table_row_count);
            params_.table_lookup->add_check(combined_query,
                                            combined_table,
                                            record.query_row_count,
                                            record.table_row_count,
                                            record.common_column_count,
                                            witness);
        }
    }

    return true;
}
