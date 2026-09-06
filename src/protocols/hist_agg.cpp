#include "hist_agg.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void validate_hist_agg_shape(size_t query_domain_size, size_t table_domain_size, const char* context)
{
    if (!isPowerOfTwo(query_domain_size) || !isPowerOfTwo(table_domain_size) || query_domain_size == 0 ||
        table_domain_size == 0) {
        throw std::runtime_error(
            std::string(context) + ": query_domain_size and table_domain_size must be non-zero powers of two");
    }
}

void validate_hist_agg_batch(const std::vector<virCommitment>& values,
                             const std::vector<virCommitment>& histograms)
{
    if (values.empty() || histograms.empty() || values.size() != histograms.size()) {
        throw std::runtime_error("histAggProtocol: channel batch metadata size mismatch");
    }
}

std::string hist_agg_index_key(size_t table_domain_size)
{
    return "hist_agg.index:" + std::to_string(table_domain_size);
}

std::string hist_agg_cache_key(const kzgProtocol& kzg, size_t table_domain_size)
{
    return transcript_representation(kzg.g1_power(0)) + ":" + transcript_representation(kzg.g1_power(1)) + ":" +
           hist_agg_index_key(table_domain_size);
}

void cache_hist_agg_index_group(const kzgProtocol& kzg, size_t table_domain_size, const Group1& commitment)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = hist_agg_cache_key(kzg, table_domain_size);
    const auto it = cache.find(key);
    if (it == cache.end()) {
        cache.emplace(key, commitment);
        return;
    }
    if (it->second != commitment) {
        throw std::runtime_error("histAggProtocol: inconsistent cached index-table commitment");
    }
}

Group1 preprocessed_hist_agg_index_group(const kzgProtocol& kzg, size_t table_domain_size)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = hist_agg_cache_key(kzg, table_domain_size);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const Group1 commitment = kzg.commit_g1_group(hist_agg_index_polynomial(table_domain_size));
    cache.emplace(key, commitment);
    return commitment;
}

bool hist_agg_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_HIST_AGG");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

bool hist_agg_failure_debug_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_DEBUG_HIST_AGG_FAILURE");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void hist_agg_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = hist_agg_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "hist_agg_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("hist_agg_profile", label, "", elapsed.count(), "ok");
}

bool try_size_t_from_fr(const Fr& value, size_t& out)
{
    try {
        const uint64_t parsed = value.getUint64();
        if (parsed > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        const size_t as_size = static_cast<size_t>(parsed);
        if (field_from_size_t(as_size) != value) {
            return false;
        }
        out = as_size;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// Both vectors below represent already-committed tables in the protocol.  The
// direct gathers only construct the identical prover witness more efficiently;
// they do not replace the subsequent power-table recurrence or table lookup.
std::vector<Fr> subgroup_label_table_evals(size_t table_domain_size)
{
    std::vector<Fr> labels(table_domain_size, Fr(0));
    Fr current(1);
    const Fr root = subgroup_generator(table_domain_size);
    for (size_t index = 0; index < table_domain_size; ++index) {
        labels[index] = current;
        Fr::mul(current, current, root);
    }
    return labels;
}

std::vector<Fr> mapped_power_evals_from_indices(const std::vector<Fr>& index_evals,
                                                const std::vector<Fr>& power_table_evals)
{
    const size_t table_domain_size = power_table_evals.size();
    std::vector<Fr> mapped(index_evals.size(), Fr(0));
    for (size_t i = 0; i < index_evals.size(); ++i) {
        size_t index = 0;
        if (!try_size_t_from_fr(index_evals[i], index) || index >= table_domain_size) {
            // Retain the dummy value used by the lookup failure path.
            continue;
        }
        mapped[i] = power_table_evals[index];
    }
    return mapped;
}

tableLookupWitness lookup_witness_from_indices(const std::vector<Fr>& index_evals,
                                               const std::vector<Fr>& table_row_labels)
{
    const size_t table_domain_size = table_row_labels.size();
    tableLookupWitness witness;
    witness.row_labels.reserve(index_evals.size());
    size_t invalid_count = 0;
    std::vector<std::pair<size_t, Fr>> first_invalid;
    for (const Fr& value : index_evals) {
        size_t index = 0;
        if (!try_size_t_from_fr(value, index) || index >= table_domain_size) {
            // Match the implicit-scan fallback semantics: invalid indices use a
            // dummy subgroup label and are later rejected by the reduced lookup.
            if (hist_agg_failure_debug_enabled()) {
                ++invalid_count;
                if (first_invalid.size() < 8) {
                    first_invalid.emplace_back(witness.row_labels.size(), value);
                }
            }
            witness.row_labels.push_back(Fr(1));
            continue;
        }
        witness.row_labels.push_back(table_row_labels[index]);
    }
    if (invalid_count != 0) {
        std::cerr << "hist_agg_debug invalid_indices"
                  << " query_domain_size=" << index_evals.size()
                  << " table_domain_size=" << table_domain_size
                  << " invalid_count=" << invalid_count;
        for (const auto& [position, value] : first_invalid) {
            std::cerr << " sample=" << position << ":" << value.getStr(10);
        }
        std::cerr << "\n";
    }
    return witness;
}

Fr sum_products(const std::vector<Fr>& left, const std::vector<Fr>& right, const char* context)
{
    if (left.size() != right.size()) {
        throw std::runtime_error(std::string(context) + ": evaluation vector size mismatch");
    }

    Fr sum = Fr(0);
    for (size_t i = 0; i < left.size(); ++i) {
        Fr term;
        Fr::mul(term, left[i], right[i]);
        Fr::add(sum, sum, term);
    }
    return sum;
}

histAggCheckRecord make_hist_agg_record_from_shared(const std::vector<Fr>& value_evals,
                                                    const std::vector<Fr>& histogram_evals,
                                                    const std::vector<Fr>& mapped_power_evals,
                                                    const std::vector<Fr>& power_table_evals,
                                                    const Fr& challenge,
                                                    size_t query_domain_size,
                                                    size_t table_domain_size,
                                                    const kzgCommitment& power_table_commitment,
                                                    const kzgCommitment& mapped_power_commitment)
{
    histAggCheckRecord record;
    record.query_domain_size = query_domain_size;
    record.table_domain_size = table_domain_size;
    record.challenge = challenge;
    record.histogram_claim = sum_products(power_table_evals, histogram_evals, "histAggProtocol");
    record.query_claim = sum_products(mapped_power_evals, value_evals, "histAggProtocol");
    record.power_table_commitment = power_table_commitment;
    record.mapped_power_commitment = mapped_power_commitment;
    return record;
}

void add_power_table_constraints(zeroProtocol& zero,
                                 const virCommitment& power_table,
                                 const Fr& challenge,
                                 size_t table_domain_size)
{
    validate_hist_agg_shape(1, table_domain_size, "histAggProtocol");

    // Anchor `power_table[0] = 1`.
    virCommitment anchor_zero = power_table - virCommitment(Fr(1));
    zero.add_check(anchor_zero, 1);

    if (table_domain_size == 1) {
        return;
    }

    const Fr previous_shift = inverse_root_of_unity(table_domain_size);
    // Recurrence:
    //   power_table[i] = challenge * power_table[i - 1]
    // away from the subgroup-zero boundary.
    virCommitment recurrence_zero =
        (power_table - scale_by_constant(x_scale(power_table, previous_shift), challenge)) *
        (virCommitment(Fr(1)) - subgroup_zero_lagrange_expression(table_domain_size));
    zero.add_check(recurrence_zero, table_domain_size);
}

} // namespace

poly hist_agg_index_polynomial(size_t table_domain_size)
{
    validate_hist_agg_shape(1, table_domain_size, "hist_agg_index_polynomial");

    std::vector<Fr> evals;
    evals.reserve(table_domain_size);
    for (size_t i = 0; i < table_domain_size; ++i) {
        evals.push_back(field_from_size_t(i));
    }
    return poly_from_evals(evals);
}

void preprocess_hist_agg_index_table(kzgProtocol& kzg, size_t table_domain_size)
{
    validate_hist_agg_shape(table_domain_size, table_domain_size, "preprocess_hist_agg_index_table");
    (void)preprocessed_hist_agg_index_group(kzg, table_domain_size);
}

poly hist_agg_power_polynomial(const Fr& challenge, size_t domain_size)
{
    validate_hist_agg_shape(1, domain_size, "hist_agg_power_polynomial");

    std::vector<Fr> evals;
    evals.reserve(domain_size);
    Fr current = Fr(1);
    for (size_t i = 0; i < domain_size; ++i) {
        evals.push_back(current);
        Fr::mul(current, current, challenge);
    }
    return poly_from_evals(evals);
}

Fr histAggVerifier::challenge()
{
    if (has_sampled_challenge) {
        return sampled_challenge;
    }
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("histAggVerifier not setup");
    }
    sampled_challenge = params->kzg->transcript().squeeze_nonzero_fr("hist_agg.r");
    has_sampled_challenge = true;
    return sampled_challenge;
}

void histAggVerifier::add_check(const virCommitment& indices,
                                const virCommitment& values,
                                const virCommitment& histogram,
                                const histAggCheckRecord& record)
{
    index_vectors.push_back(indices);
    value_vectors.push_back(values);
    histograms.push_back(histogram);
    checks.push_back(record);
}

bool histAggVerifier::finalize() const
{
    for (const histAggCheckRecord& check : checks) {
        if (check.histogram_claim != check.query_claim) {
            return false;
        }
    }
    return true;
}

histAggCheckRecord histAggProver::add_check(const virCommitment& indices,
                                            const virCommitment& values,
                                            const virCommitment& histogram,
                                            const Fr& challenge,
                                            size_t query_domain_size,
                                            size_t table_domain_size)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("histAggProver not setup");
    }

    const std::vector<Fr> index_evals = params->kzg->evaluations(indices, query_domain_size);
    const std::vector<Fr> value_evals = params->kzg->evaluations(values, query_domain_size);
    const std::vector<Fr> histogram_evals = params->kzg->evaluations(histogram, table_domain_size);

    const poly power_table_poly = hist_agg_power_polynomial(challenge, table_domain_size);
    const std::vector<Fr> power_table_evals = power_table_poly.evaluations_on_domain(table_domain_size);
    const std::vector<Fr> mapped_power_evals =
        mapped_power_evals_from_indices(index_evals, power_table_evals);

    histAggCheckRecord record;
    record.query_domain_size = query_domain_size;
    record.table_domain_size = table_domain_size;
    record.challenge = challenge;
    record.histogram_claim = sum_products(power_table_evals, histogram_evals, "histAggProver");
    record.query_claim = sum_products(mapped_power_evals, value_evals, "histAggProver");
    record.power_table_commitment = params->kzg->commit(power_table_poly, table_domain_size - 1);
    record.mapped_power_commitment = params->kzg->commit(poly_from_evals(mapped_power_evals), query_domain_size - 1);

    checks.push_back(record);
    return record;
}

void histAggProtocol::add_check(const virCommitment& indices,
                                const virCommitment& values,
                                const virCommitment& histogram,
                                size_t query_domain_size,
                                size_t table_domain_size)
{
    proof_cost::record_relation("histogram_aggregation", query_domain_size);
    if (params_.soundness_counter != nullptr) {
        params_.soundness_counter->record_histogram(table_domain_size, 1);
    }
    structured_protocol_profile::timed_void(
        "protocol_call",
        "hist_agg.add_check",
        "query_domain_size=" + std::to_string(query_domain_size) + ";table_domain_size=" +
            std::to_string(table_domain_size) + ";channel_count=1",
        [&] {
            if (!is_setup) {
                throw std::runtime_error("histAggProtocol not setup");
            }
            if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.sum_check == nullptr ||
                params_.table_lookup == nullptr) {
                throw std::runtime_error("histAggProtocol dependencies not setup");
            }

            validate_hist_agg_shape(query_domain_size, table_domain_size, "histAggProtocol");

            pendingHistAggCheck pending;
            pending.indices = indices;
            pending.values.push_back(values);
            pending.histograms.push_back(histogram);
            pending.query_domain_size = query_domain_size;
            pending.table_domain_size = table_domain_size;
            pending_checks_.push_back(std::move(pending));
        });
}

void histAggProtocol::add_check(const virCommitment& indices,
                                const std::vector<virCommitment>& values,
                                const std::vector<virCommitment>& histograms,
                                size_t query_domain_size,
                                size_t table_domain_size)
{
    proof_cost::record_relation("histogram_aggregation", query_domain_size * values.size());
    if (params_.soundness_counter != nullptr) {
        params_.soundness_counter->record_histogram(table_domain_size, values.size());
    }
    structured_protocol_profile::timed_void(
        "protocol_call",
        "hist_agg.add_check",
        "query_domain_size=" + std::to_string(query_domain_size) + ";table_domain_size=" +
            std::to_string(table_domain_size) + ";channel_count=" + std::to_string(values.size()),
        [&] {
            if (!is_setup) {
                throw std::runtime_error("histAggProtocol not setup");
            }
            if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.sum_check == nullptr ||
                params_.table_lookup == nullptr) {
                throw std::runtime_error("histAggProtocol dependencies not setup");
            }

            validate_hist_agg_shape(query_domain_size, table_domain_size, "histAggProtocol");
            validate_hist_agg_batch(values, histograms);

            pendingHistAggCheck pending;
            pending.indices = indices;
            pending.values = values;
            pending.histograms = histograms;
            pending.query_domain_size = query_domain_size;
            pending.table_domain_size = table_domain_size;
            pending_checks_.push_back(std::move(pending));
        });
}

bool histAggProtocol::finalize()
{
    begin_finalize("histAggProtocol");

    if (!pending_checks_.empty()) {
        struct SharedTableData {
            bool has_index_table_commitment = false;
            bool has_power_table_commitment = false;
            kzgCommitment index_table_commitment;
            kzgCommitment power_table_commitment;
            std::vector<Fr> power_table_evals;
            std::vector<Fr> table_row_labels;
            virCommitment index_table;
            virCommitment power_table;
            bool constraints_added = false;
        };

        const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
        Fr challenge = Fr(0);
        hist_agg_profile_scope("challenge", [&] { challenge = verifier_.challenge(); });

        std::map<size_t, SharedTableData> shared_by_table_size;
        for (const pendingHistAggCheck& pending : pending_checks_) {
            SharedTableData& shared = shared_by_table_size[pending.table_domain_size];
            if (!shared.has_index_table_commitment) {
                const bool verifier_mode = proof_session && proof_session->mode() == ProofMode::Verify;
                if (verifier_mode) {
                    hist_agg_profile_scope("verify.register_preprocessed_index_table", [&] {
                        shared.index_table_commitment =
                            params_.kzg->adopt_public_commitment(
                                hist_agg_index_key(pending.table_domain_size),
                                preprocessed_hist_agg_index_group(*params_.kzg, pending.table_domain_size));
                        if (!params_.kzg->has_degree_bound(
                                virCommitment(shared.index_table_commitment), pending.table_domain_size - 1)) {
                            params_.kzg->prove_degree(shared.index_table_commitment, pending.table_domain_size - 1);
                        }
                    });
                } else {
                    hist_agg_profile_scope("commit_public_index_table", [&] {
                        const poly index_poly = hist_agg_index_polynomial(pending.table_domain_size);
                        const Group1 index_group =
                            preprocessed_hist_agg_index_group(*params_.kzg, pending.table_domain_size);
                        shared.index_table_commitment =
                            params_.kzg->commit_public_with_group(hist_agg_index_key(pending.table_domain_size),
                                                                  index_poly,
                                                                  index_group,
                                                                  pending.table_domain_size - 1);
                    });
                    cache_hist_agg_index_group(
                        *params_.kzg, pending.table_domain_size, shared.index_table_commitment.com);
                }
                shared.index_table = virCommitment(shared.index_table_commitment);
                shared.has_index_table_commitment = true;
            }
            if (!shared.has_power_table_commitment) {
                if (proof_session && proof_session->mode() == ProofMode::Verify) {
                    Group1 power_table_group;
                    hist_agg_profile_scope("verify.transport_power_table_commitment", [&] {
                        power_table_group = proof_session->transport_group1({});
                    });
                    hist_agg_profile_scope("verify.adopt_power_table_commitment", [&] {
                        shared.power_table_commitment = params_.kzg->adopt_commitment(power_table_group);
                        params_.kzg->prove_degree(shared.power_table_commitment, pending.table_domain_size - 1);
                    });
                } else {
                    poly power_table_poly;
                    hist_agg_profile_scope("prove.build_power_table_poly", [&] {
                        power_table_poly = hist_agg_power_polynomial(challenge, pending.table_domain_size);
                    });
                    hist_agg_profile_scope("prove.evaluate_power_table", [&] {
                        shared.power_table_evals = power_table_poly.evaluations_on_domain(pending.table_domain_size);
                    });
                    hist_agg_profile_scope("prove.build_table_row_labels", [&] {
                        shared.table_row_labels = subgroup_label_table_evals(pending.table_domain_size);
                    });
                    hist_agg_profile_scope("prove.commit_power_table", [&] {
                        shared.power_table_commitment =
                            params_.kzg->commit(power_table_poly, pending.table_domain_size - 1);
                    });
                    if (proof_session && proof_session->mode() == ProofMode::Prove) {
                        Group1 power_table_group;
                        hist_agg_profile_scope("prove.transport_power_table_commitment", [&] {
                            power_table_group =
                                proof_session->transport_group1([&] { return shared.power_table_commitment.com; });
                        });
                        if (power_table_group != shared.power_table_commitment.com) {
                            throw std::runtime_error("histAggProtocol: transported power table commitment mismatch");
                        }
                    }
                }
                shared.power_table = virCommitment(shared.power_table_commitment);
                shared.has_power_table_commitment = true;
            }

            if (!shared.constraints_added) {
                hist_agg_profile_scope("add_power_table_constraints", [&] {
                    add_power_table_constraints(*params_.zero_check, shared.power_table, challenge, pending.table_domain_size);
                });
                shared.constraints_added = true;
            }

            std::optional<tableLookupWitness> lookup_witness;
            std::vector<Fr> mapped_power_evals;
            kzgCommitment mapped_power_commitment;
            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                Group1 mapped_power_group;
                hist_agg_profile_scope("verify.transport_mapped_power_commitment", [&] {
                    mapped_power_group = proof_session->transport_group1({});
                });
                hist_agg_profile_scope("verify.adopt_mapped_power_commitment", [&] {
                    mapped_power_commitment = params_.kzg->adopt_commitment(mapped_power_group);
                    params_.kzg->prove_degree(mapped_power_commitment, pending.query_domain_size - 1);
                });
            } else {
                std::vector<Fr> index_evals;
                hist_agg_profile_scope("prove.evaluate_indices", [&] {
                    index_evals = params_.kzg->evaluations(pending.indices, pending.query_domain_size);
                });
                hist_agg_profile_scope("prove.build_lookup_witness", [&] {
                    lookup_witness = lookup_witness_from_indices(index_evals, shared.table_row_labels);
                });
                hist_agg_profile_scope("prove.build_mapped_power", [&] {
                    mapped_power_evals =
                        mapped_power_evals_from_indices(index_evals, shared.power_table_evals);
                });
                hist_agg_profile_scope("prove.commit_mapped_power", [&] {
                    mapped_power_commitment =
                        params_.kzg->commit(poly_from_evals(mapped_power_evals), pending.query_domain_size - 1);
                });
                if (proof_session && proof_session->mode() == ProofMode::Prove) {
                    Group1 mapped_power_group;
                    hist_agg_profile_scope("prove.transport_mapped_power_commitment", [&] {
                        mapped_power_group =
                            proof_session->transport_group1([&] { return mapped_power_commitment.com; });
                    });
                    if (mapped_power_group != mapped_power_commitment.com) {
                        throw std::runtime_error("histAggProtocol: transported mapped power commitment mismatch");
                    }
                }
            }

            const virCommitment mapped_power(mapped_power_commitment);
            std::vector<histAggCheckRecord> records;
            records.reserve(pending.values.size());

            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                for (size_t channel = 0; channel < pending.values.size(); ++channel) {
                    histAggCheckRecord record;
                    record.query_domain_size = pending.query_domain_size;
                    record.table_domain_size = pending.table_domain_size;
                    record.challenge = challenge;
                    record.power_table_commitment = shared.power_table_commitment;
                    record.mapped_power_commitment = mapped_power_commitment;
                    record.histogram_claim = proof_session->exchange_fr("hist_agg.histogram_claim", {});
                    record.query_claim = proof_session->exchange_fr("hist_agg.query_claim", {});
                    records.push_back(record);
                }
            } else {
                records.resize(pending.values.size());
                hist_agg_profile_scope("prove.channels.build_records", [&] {
                    for (size_t channel = 0; channel < pending.values.size(); ++channel) {
                        const std::vector<Fr> value_evals =
                            params_.kzg->evaluations(pending.values[channel], pending.query_domain_size);
                        const std::vector<Fr> histogram_evals =
                            params_.kzg->evaluations(pending.histograms[channel], pending.table_domain_size);
                        records[channel] =
                            make_hist_agg_record_from_shared(value_evals,
                                                             histogram_evals,
                                                             mapped_power_evals,
                                                             shared.power_table_evals,
                                                             challenge,
                                                             pending.query_domain_size,
                                                             pending.table_domain_size,
                                                             shared.power_table_commitment,
                                                             mapped_power_commitment);
                    }
                    return;
                });
                if (proof_session && proof_session->mode() == ProofMode::Prove) {
                    for (size_t channel = 0; channel < pending.values.size(); ++channel) {
                        hist_agg_profile_scope("prove.channel.exchange_claims", [&] {
                            records[channel].histogram_claim = proof_session->exchange_fr(
                                "hist_agg.histogram_claim", [&] { return records[channel].histogram_claim; });
                            records[channel].query_claim =
                                proof_session->exchange_fr("hist_agg.query_claim", [&] { return records[channel].query_claim; });
                        });
                    }
                }
            }

            for (size_t channel = 0; channel < pending.values.size(); ++channel) {
                hist_agg_profile_scope("channel.verifier_add_check", [&] {
                    verifier_.add_check(pending.indices, pending.values[channel], pending.histograms[channel], records[channel]);
                });
                hist_agg_profile_scope("channel.sum_check_histogram", [&] {
                    virCommitment histogram_weighted = shared.power_table * pending.histograms[channel];
                    params_.sum_check->add_check(
                        histogram_weighted, records[channel].histogram_claim, pending.table_domain_size);
                });
                hist_agg_profile_scope("channel.sum_check_query", [&] {
                    virCommitment query_weighted = mapped_power * pending.values[channel];
                    params_.sum_check->add_check(query_weighted, records[channel].query_claim, pending.query_domain_size);
                });
            }

            virCommitment lookup_query;
            virCommitment lookup_table;
            hist_agg_profile_scope("build_lookup_commitments", [&] {
                lookup_query = interleave_commitments(
                    pending.indices, mapped_power, pending.query_domain_size, "histAggProtocol");
                lookup_table = interleave_commitments(
                    shared.index_table, shared.power_table, pending.table_domain_size, "histAggProtocol");
            });
            hist_agg_profile_scope("table_lookup.add_check", [&] {
                if (lookup_witness.has_value()) {
                    params_.table_lookup->add_check(lookup_query,
                                                    lookup_table,
                                                    pending.query_domain_size,
                                                    pending.table_domain_size,
                                                    2,
                                                    *lookup_witness);
                    return;
                }
                params_.table_lookup->add_check(
                    lookup_query, lookup_table, pending.query_domain_size, pending.table_domain_size, 2);
            });
        }

        pending_checks_.clear();
    }

    return verifier_.finalize();
}
