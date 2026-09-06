#include "index_routing.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_index_routing_shape(size_t target_row_count, size_t source_row_count, size_t column_count)
{
    validate_matrix_shape(target_row_count, 1, "indexRoutingProtocol");
    validate_matrix_shape(source_row_count, 1, "indexRoutingProtocol");
    validate_matrix_shape(source_row_count, column_count, "indexRoutingProtocol");
    validate_matrix_shape(target_row_count, column_count, "indexRoutingProtocol");
}

poly index_table_polynomial(size_t source_row_count)
{
    validate_matrix_shape(source_row_count, 1, "indexRoutingProtocol index table");

    std::vector<Fr> evals;
    evals.reserve(source_row_count);
    for (size_t row = 0; row < source_row_count; ++row) {
        evals.push_back(field_from_size_t(row));
    }
    return poly_from_evals(evals);
}

bool try_size_t_from_fr(const Fr& value, size_t& out)
{
    const std::string repr = value.getStr(10);
    if (!repr.empty() && repr.front() == '-') {
        return false;
    }

    size_t parsed = 0;
    try {
        parsed = static_cast<size_t>(std::stoull(repr));
    } catch (const std::exception&) {
        return false;
    }

    if (field_from_size_t(parsed) != value) {
        return false;
    }

    out = parsed;
    return true;
}

std::vector<size_t> routed_positions_from_indices(const std::vector<Fr>& index_evals,
                                                  const std::vector<Fr>* maybe_mask_evals,
                                                  size_t source_row_count)
{
    std::vector<size_t> positions(index_evals.size(), 0);
    for (size_t row = 0; row < index_evals.size(); ++row) {
        // Inactive rows are routed through dummy row 0 only so the lookup
        // witness stays total; a later zero-check erases their output.
        const bool active = maybe_mask_evals == nullptr || (*maybe_mask_evals)[row] != Fr(0);
        if (!active) {
            positions[row] = 0;
            continue;
        }

        size_t parsed = 0;
        if (!try_size_t_from_fr(index_evals[row], parsed) || parsed >= source_row_count) {
            positions[row] = 0;
            continue;
        }
        positions[row] = parsed;
    }
    return positions;
}

poly routed_rows_polynomial_from_positions(const std::vector<Fr>& table_evals,
                                           const std::vector<size_t>& row_positions,
                                           size_t source_row_count,
                                           size_t column_count)
{
    std::vector<Fr> routed_evals(row_positions.size() * column_count, Fr(0));
    for (size_t target_row = 0; target_row < row_positions.size(); ++target_row) {
        const size_t source_row = row_positions[target_row];
        if (source_row >= source_row_count) {
            throw std::runtime_error("indexRoutingProtocol: routed source row out of range");
        }
        for (size_t column = 0; column < column_count; ++column) {
            routed_evals[matrix_index(target_row, column, column_count)] =
                table_evals[matrix_index(source_row, column, column_count)];
        }
    }
    return poly_from_evals(routed_evals);
}

poly lifted_mask_polynomial(const std::vector<Fr>& mask_evals, size_t column_count)
{
    std::vector<Fr> lifted(mask_evals.size() * column_count, Fr(0));
    for (size_t row = 0; row < mask_evals.size(); ++row) {
        for (size_t column = 0; column < column_count; ++column) {
            lifted[matrix_index(row, column, column_count)] = mask_evals[row];
        }
    }
    return poly_from_evals(lifted);
}

kzgCommitment adopt_transported_commitment(kzgProtocol& kzg,
                                           const std::shared_ptr<ProofSession>& proof_session,
                                           size_t degree_bound)
{
    const Group1 commitment_group = proof_session->transport_group1({});
    const kzgCommitment commitment = kzg.adopt_commitment(commitment_group);
    kzg.prove_degree(commitment, degree_bound);
    return commitment;
}

} // namespace

void indexRoutingVerifier::add_check(const indexRoutingCheckRecord& record)
{
    checks.push_back(record);
}

bool indexRoutingVerifier::finalize() const
{
    return true;
}

indexRoutingCheckRecord indexRoutingProver::add_check(const virCommitment& row_indices,
                                                      const virCommitment* mask,
                                                      const std::vector<virCommitment>& tables,
                                                      size_t target_row_count,
                                                      size_t source_row_count,
                                                      size_t column_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("indexRoutingProver not setup");
    }

    const std::vector<Fr> index_evals = params->kzg->evaluations(row_indices, target_row_count);
    if (tables.empty()) {
        throw std::runtime_error("indexRoutingProver: table batch must be non-empty");
    }

    std::vector<Fr> mask_evals;
    const std::vector<Fr>* maybe_mask_evals = nullptr;
    if (mask != nullptr) {
        mask_evals = params->kzg->evaluations(*mask, target_row_count);
        maybe_mask_evals = &mask_evals;
    }

    const std::vector<size_t> row_positions =
        routed_positions_from_indices(index_evals, maybe_mask_evals, source_row_count);

    indexRoutingCheckRecord record;
    record.target_row_count = target_row_count;
    record.source_row_count = source_row_count;
    record.column_count = column_count;
    record.channel_count = tables.size();
    record.uses_mask = mask != nullptr;
    record.routed_rows_commitments.reserve(tables.size());
    for (const virCommitment& table : tables) {
        const std::vector<Fr> table_evals = params->kzg->evaluations(table, source_row_count * column_count);
        record.routed_rows_commitments.push_back(params->kzg->commit(
            routed_rows_polynomial_from_positions(table_evals, row_positions, source_row_count, column_count),
            target_row_count * column_count - 1));
    }
    if (record.uses_mask) {
        record.lifted_mask_commitment =
            params->kzg->commit(lifted_mask_polynomial(mask_evals, column_count), target_row_count * column_count - 1);
    }

    checks.push_back(record);
    return record;
}

void indexRoutingProtocol::add_check(const virCommitment& row_indices,
                                     const virCommitment& table,
                                     const virCommitment& output,
                                     size_t target_row_count,
                                     size_t source_row_count,
                                     size_t column_count)
{
    add_check(row_indices,
              std::vector<virCommitment>{table},
              std::vector<virCommitment>{output},
              target_row_count,
              source_row_count,
              column_count);
}

void indexRoutingProtocol::add_check(const virCommitment& row_indices,
                                     const std::vector<virCommitment>& tables,
                                     const std::vector<virCommitment>& outputs,
                                     size_t target_row_count,
                                     size_t source_row_count,
                                     size_t column_count)
{
    add_check_internal(row_indices, nullptr, tables, outputs, target_row_count, source_row_count, column_count);
}

void indexRoutingProtocol::add_masked_check(const virCommitment& row_indices,
                                            const virCommitment& mask,
                                            const virCommitment& table,
                                            const virCommitment& output,
                                            size_t target_row_count,
                                            size_t source_row_count,
                                            size_t column_count)
{
    add_masked_check(row_indices,
                     mask,
                     std::vector<virCommitment>{table},
                     std::vector<virCommitment>{output},
                     target_row_count,
                     source_row_count,
                     column_count);
}

void indexRoutingProtocol::add_masked_check(const virCommitment& row_indices,
                                            const virCommitment& mask,
                                            const std::vector<virCommitment>& tables,
                                            const std::vector<virCommitment>& outputs,
                                            size_t target_row_count,
                                            size_t source_row_count,
                                            size_t column_count)
{
    add_check_internal(row_indices, &mask, tables, outputs, target_row_count, source_row_count, column_count);
}

void indexRoutingProtocol::add_masked_check_with_certified_mask(
    const virCommitment& row_indices,
    const virCommitment& mask,
    const std::vector<virCommitment>& tables,
    const std::vector<virCommitment>& outputs,
    size_t target_row_count,
    size_t source_row_count,
    size_t column_count)
{
    add_check_internal(
        row_indices, &mask, tables, outputs, target_row_count, source_row_count, column_count, false);
}

void indexRoutingProtocol::add_check_internal(const virCommitment& row_indices,
                                              const virCommitment* mask,
                                              const std::vector<virCommitment>& tables,
                                              const std::vector<virCommitment>& outputs,
                                              size_t target_row_count,
                                              size_t source_row_count,
                                              size_t column_count,
                                              bool enforce_mask_boolean)
{
    proof_cost::record_relation("routing", target_row_count * column_count * tables.size());
    if (!is_setup) {
        throw std::runtime_error("indexRoutingProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.column_replication == nullptr ||
        params_.concatenated_table_lookup == nullptr) {
        throw std::runtime_error("indexRoutingProtocol dependencies not setup");
    }

    validate_index_routing_shape(target_row_count, source_row_count, column_count);
    if (tables.empty() || tables.size() != outputs.size()) {
        throw std::runtime_error("indexRoutingProtocol: table batch metadata size mismatch");
    }

    indexRoutingCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.target_row_count = target_row_count;
        record.source_row_count = source_row_count;
        record.column_count = column_count;
        record.channel_count = tables.size();
        record.uses_mask = mask != nullptr;
        record.enforce_mask_boolean = enforce_mask_boolean;
        record.routed_rows_commitments.reserve(record.channel_count);
        for (size_t channel = 0; channel < record.channel_count; ++channel) {
            record.routed_rows_commitments.push_back(
                adopt_transported_commitment(*params_.kzg, proof_session, target_row_count * column_count - 1));
        }
        if (record.uses_mask) {
            record.lifted_mask_commitment =
                adopt_transported_commitment(*params_.kzg, proof_session, target_row_count * column_count - 1);
        }
    } else {
        record = prover_.add_check(row_indices, mask, tables, target_row_count, source_row_count, column_count);
        record.enforce_mask_boolean = enforce_mask_boolean;
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            for (size_t channel = 0; channel < record.channel_count; ++channel) {
                const Group1 routed_group =
                    proof_session->transport_group1([&] { return record.routed_rows_commitments[channel].com; });
                if (routed_group != record.routed_rows_commitments[channel].com) {
                    throw std::runtime_error("indexRoutingProtocol: transported routed-rows commitment mismatch");
                }
            }

            if (record.uses_mask) {
                const Group1 lifted_mask_group =
                    proof_session->transport_group1([&] { return record.lifted_mask_commitment.com; });
                if (lifted_mask_group != record.lifted_mask_commitment.com) {
                    throw std::runtime_error("indexRoutingProtocol: transported lifted-mask commitment mismatch");
                }
            }
        }
    }

    verifier_.add_check(record);
    row_indices_.push_back(row_indices);
    masks_.push_back(mask == nullptr ? virCommitment(Fr(1)) : *mask);
    table_batches_.push_back(tables);
    output_batches_.push_back(outputs);
}

bool indexRoutingProtocol::finalize()
{
    begin_finalize("indexRoutingProtocol");

    // Merge checks that route through the same index vector (and mask) into
    // one wide tuple lookup. The identity-index block then appears once, while
    // all routed columns are still forced to come from that exact source row.
    std::vector<std::vector<size_t>> groups;
    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        bool grouped = false;
        for (std::vector<size_t>& group : groups) {
            const size_t representative = group.front();
            const indexRoutingCheckRecord& left = verifier_.checks[representative];
            const indexRoutingCheckRecord& right = verifier_.checks[i];
            const bool same_mask = !left.uses_mask ||
                                   same_virtual_commitment_alias(masks_[representative], masks_[i]);
            if (left.target_row_count == right.target_row_count &&
                left.source_row_count == right.source_row_count &&
                left.uses_mask == right.uses_mask && same_mask &&
                same_virtual_commitment_alias(row_indices_[representative], row_indices_[i])) {
                group.push_back(i);
                grouped = true;
                break;
            }
        }
        if (!grouped) {
            groups.push_back({i});
        }
    }

    for (const std::vector<size_t>& group : groups) {
        const size_t representative = group.front();
        const indexRoutingCheckRecord& representative_record = verifier_.checks[representative];
        const virCommitment identity_table(
            index_table_polynomial(representative_record.source_row_count));
        const virCommitment effective_indices = representative_record.uses_mask
                                                    ? row_indices_[representative] * masks_[representative]
                                                    : row_indices_[representative];

        const tableLookupWitness* maybe_witness = nullptr;
        tableLookupWitness witness;
        if (!(params_.kzg->proof_session() &&
              params_.kzg->proof_session()->mode() == ProofMode::Verify)) {
            const std::vector<Fr> index_evals = params_.kzg->evaluations(
                row_indices_[representative], representative_record.target_row_count);
            std::vector<Fr> mask_evals;
            const std::vector<Fr>* maybe_mask_evals = nullptr;
            if (representative_record.uses_mask) {
                mask_evals = params_.kzg->evaluations(
                    masks_[representative], representative_record.target_row_count);
                maybe_mask_evals = &mask_evals;
            }
            const std::vector<size_t> row_positions = routed_positions_from_indices(
                index_evals, maybe_mask_evals, representative_record.source_row_count);
            witness = tableLookupWitness::from_positions(
                representative_record.source_row_count, row_positions);
            maybe_witness = &witness;
        }

        std::vector<concatenatedTableLookupBlock> blocks;
        blocks.emplace_back(effective_indices, identity_table, 1);
        for (const size_t check_index : group) {
            const indexRoutingCheckRecord& record = verifier_.checks[check_index];
            for (size_t channel = 0; channel < record.channel_count; ++channel) {
                blocks.emplace_back(virCommitment(record.routed_rows_commitments[channel]),
                                    table_batches_[check_index][channel],
                                    record.column_count);
            }
        }
        if (maybe_witness == nullptr) {
            params_.concatenated_table_lookup->add_check(
                blocks,
                representative_record.target_row_count,
                representative_record.source_row_count);
        } else {
            params_.concatenated_table_lookup->add_check(
                blocks,
                representative_record.target_row_count,
                representative_record.source_row_count,
                *maybe_witness);
        }

        for (const size_t check_index : group) {
            const indexRoutingCheckRecord& record = verifier_.checks[check_index];
            for (size_t channel = 0; channel < record.channel_count; ++channel) {
                const virCommitment routed_rows(record.routed_rows_commitments[channel]);
                if (record.uses_mask) {
                    const virCommitment lifted_mask(record.lifted_mask_commitment);
                    virCommitment output_zero =
                        output_batches_[check_index][channel] - lifted_mask * routed_rows;
                    params_.zero_check->add_check(
                        output_zero,
                        record.target_row_count * record.column_count,
                        "index_routing.output[" + std::to_string(check_index) + "," +
                            std::to_string(channel) + "]");
                } else {
                    virCommitment output_zero = output_batches_[check_index][channel] - routed_rows;
                    params_.zero_check->add_check(
                        output_zero,
                        record.target_row_count * record.column_count,
                        "index_routing.output[" + std::to_string(check_index) + "," +
                            std::to_string(channel) + "]");
                }
            }

            if (record.uses_mask) {
                if (record.enforce_mask_boolean) {
                    params_.zero_check->add_boolean_check(
                        masks_[check_index],
                        record.target_row_count,
                        "index_routing.mask_boolean[" + std::to_string(check_index) + "]");
                }
                const virCommitment lifted_mask(record.lifted_mask_commitment);
                params_.column_replication->add_check(
                    masks_[check_index], lifted_mask, record.target_row_count, record.column_count);
            }
        }
    }

    return verifier_.finalize();
}
