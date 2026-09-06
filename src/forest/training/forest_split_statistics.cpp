#include "forest_split_statistics.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

void validate_split_statistics_parameters(size_t node_count, size_t feature_count, size_t bin_count)
{
    validate_matrix_shape(node_count, feature_count * bin_count, "forestSplitStatisticsProtocol histograms");
    validate_matrix_shape(node_count * feature_count, bin_count, "forestSplitStatisticsProtocol reshaped histogram");
}

void validate_channel_batch(const forestSplitStatisticsChannelBatch& channels)
{
    if (channels.histograms.empty()) {
        throw std::runtime_error("forestSplitStatisticsProtocol: channel batch must be non-empty");
    }
    if (channels.histograms.size() != channels.left_statistics.size() ||
        channels.histograms.size() != channels.right_statistics.size()) {
        throw std::runtime_error("forestSplitStatisticsProtocol: channel batch metadata size mismatch");
    }
}

forestSplitStatisticsChannelBatch single_channel_batch(const virCommitment& histogram,
                                                       const virCommitment& left_statistics,
                                                       const virCommitment& right_statistics)
{
    forestSplitStatisticsChannelBatch channels;
    channels.histograms.push_back(histogram);
    channels.left_statistics.push_back(left_statistics);
    channels.right_statistics.push_back(right_statistics);
    return channels;
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

void forestSplitStatisticsVerifier::add_check(const forestSplitStatisticsCheckRecord& record)
{
    checks.push_back(record);
}

bool forestSplitStatisticsVerifier::finalize() const
{
    return true;
}

forestSplitStatisticsCheckRecord forestSplitStatisticsProver::add_check(const forestSplitStatisticsChannelBatch& channels,
                                                                       size_t node_count,
                                                                       size_t feature_count,
                                                                       size_t bin_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("forestSplitStatisticsProver not setup");
    }

    forestSplitStatisticsCheckRecord record;
    record.node_count = node_count;
    record.feature_count = feature_count;
    record.bin_count = bin_count;
    record.channel_count = channels.histograms.size();
    record.row_total_commitments.reserve(record.channel_count);
    record.replicated_total_commitments.reserve(record.channel_count);

    const size_t row_count = node_count * feature_count;
    struct ChannelWitness {
        poly row_total;
        poly replicated_total;
        Group1 row_total_group;
        Group1 replicated_total_group;
    };
    std::vector<ChannelWitness> channel_witnesses(record.channel_count);
    for (size_t channel = 0; channel < record.channel_count; ++channel) {
        const std::shared_ptr<poly> histogram_poly = params->kzg->materialize(channels.histograms[channel]);
        if (histogram_poly == nullptr) {
            throw std::runtime_error("forestSplitStatisticsProver: failed to materialize histogram");
        }

        ChannelWitness witness;
        witness.row_total = matrix_row_sum_polynomial(*histogram_poly, row_count, bin_count);
        witness.replicated_total = witness.row_total.column_replicated(row_count, bin_count);
        witness.row_total_group = params->kzg->commit_g1_group(witness.row_total);
        witness.replicated_total_group = params->kzg->commit_g1_group(witness.replicated_total);
        channel_witnesses[channel] = std::move(witness);
    }

    for (ChannelWitness& witness : channel_witnesses) {
        kzgCommitment row_total_commitment;
        row_total_commitment.com = witness.row_total_group;
        row_total_commitment = params->kzg->commit(witness.row_total, row_total_commitment);
        params->kzg->prove_degree(row_total_commitment, row_count - 1);
        record.row_total_commitments.push_back(row_total_commitment);

        kzgCommitment replicated_total_commitment;
        replicated_total_commitment.com = witness.replicated_total_group;
        replicated_total_commitment = params->kzg->commit(witness.replicated_total, replicated_total_commitment);
        params->kzg->prove_degree(replicated_total_commitment, row_count * bin_count - 1);
        record.replicated_total_commitments.push_back(replicated_total_commitment);
    }

    checks.push_back(record);
    return record;
}

void forestSplitStatisticsProtocol::add_check(const virCommitment& histogram,
                                              const virCommitment& left_statistics,
                                              const virCommitment& right_statistics,
                                              size_t node_count,
                                              size_t feature_count,
                                              size_t bin_count)
{
    add_check(single_channel_batch(histogram, left_statistics, right_statistics), node_count, feature_count, bin_count);
}

void forestSplitStatisticsProtocol::add_check(const forestSplitStatisticsChannelBatch& channels,
                                              size_t node_count,
                                              size_t feature_count,
                                              size_t bin_count)
{
    if (!is_setup) {
        throw std::runtime_error("forestSplitStatisticsProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.row_prefix == nullptr ||
        params_.row_sum == nullptr || params_.column_replication == nullptr) {
        throw std::runtime_error("forestSplitStatisticsProtocol dependencies not setup");
    }

    validate_split_statistics_parameters(node_count, feature_count, bin_count);
    validate_channel_batch(channels);

    forestSplitStatisticsCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        record.node_count = node_count;
        record.feature_count = feature_count;
        record.bin_count = bin_count;
        record.channel_count = channels.histograms.size();
        record.row_total_commitments.reserve(record.channel_count);
        record.replicated_total_commitments.reserve(record.channel_count);

        const size_t row_count = node_count * feature_count;
        for (size_t channel = 0; channel < record.channel_count; ++channel) {
            record.row_total_commitments.push_back(adopt_transported_commitment(*params_.kzg, proof_session, row_count - 1));
            record.replicated_total_commitments.push_back(
                adopt_transported_commitment(*params_.kzg, proof_session, row_count * bin_count - 1));
        }
    } else {
        record = prover_.add_check(channels, node_count, feature_count, bin_count);
        if (proof_session && proof_session->mode() == ProofMode::Prove) {
            for (size_t channel = 0; channel < record.channel_count; ++channel) {
                const Group1 row_total_group =
                    proof_session->transport_group1([&] { return record.row_total_commitments[channel].com; });
                if (row_total_group != record.row_total_commitments[channel].com) {
                    throw std::runtime_error(
                        "forestSplitStatisticsProtocol: transported row-total commitment mismatch");
                }

                const Group1 replicated_total_group =
                    proof_session->transport_group1([&] { return record.replicated_total_commitments[channel].com; });
                if (replicated_total_group != record.replicated_total_commitments[channel].com) {
                    throw std::runtime_error(
                        "forestSplitStatisticsProtocol: transported replicated-total commitment mismatch");
                }
            }
        }
    }

    verifier_.add_check(record);
    histogram_channels_.push_back(channels.histograms);
    left_statistic_channels_.push_back(channels.left_statistics);
    right_statistic_channels_.push_back(channels.right_statistics);
}

bool forestSplitStatisticsProtocol::finalize()
{
    begin_finalize("forestSplitStatisticsProtocol");

    for (size_t i = 0; i < verifier_.checks.size(); ++i) {
        const forestSplitStatisticsCheckRecord& record = verifier_.checks[i];
        const size_t row_count = record.node_count * record.feature_count;
        const size_t matrix_domain_size = row_count * record.bin_count;

        for (size_t channel = 0; channel < record.channel_count; ++channel) {
            const virCommitment row_total(record.row_total_commitments[channel]);
            const virCommitment replicated_total(record.replicated_total_commitments[channel]);

            params_.row_prefix->add_check(histogram_channels_[i][channel],
                                          left_statistic_channels_[i][channel],
                                          row_count,
                                          record.bin_count);
            params_.row_sum->add_check(histogram_channels_[i][channel], row_total, row_count, record.bin_count);
            params_.column_replication->add_check(row_total, replicated_total, row_count, record.bin_count);

            virCommitment split_zero =
                replicated_total - left_statistic_channels_[i][channel] - right_statistic_channels_[i][channel];
            params_.zero_check->add_check(split_zero, matrix_domain_size);
        }
    }

    return verifier_.finalize();
}
