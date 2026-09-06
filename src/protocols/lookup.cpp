#include "lookup.hpp"

#include "algebra_utils.hpp"
#include "interleaved_batching.hpp"
#include "polynomial.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

bool lookup_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_LOOKUP");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

bool lookup_failure_debug_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_DEBUG_LOOKUP_FAILURE");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void lookup_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = lookup_profile_enabled();
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
        std::cerr << "lookup_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("lookup_profile", label, "", elapsed.count(), "ok");
}

std::vector<Fr> inverse_witness_evals(const std::vector<Fr>& values, const Fr& x)
{
    std::vector<Fr> witness(values.size(), Fr(0));
    Fr prefix_product(1);
    for (size_t i = 0; i < values.size(); ++i) {
        const Fr denom = x + values[i];
        if (denom.isZero()) {
            throw std::runtime_error("lookupProtocol: sampled challenge hits a pole");
        }
        witness[i] = prefix_product;
        Fr::mul(prefix_product, prefix_product, denom);
    }

    Fr inverse_suffix;
    Fr::inv(inverse_suffix, prefix_product);
    for (size_t offset = 0; offset < values.size(); ++offset) {
        const size_t i = values.size() - 1 - offset;
        const Fr denom = x + values[i];
        Fr inverse_value;
        Fr::mul(inverse_value, inverse_suffix, witness[i]);
        Fr::mul(inverse_suffix, inverse_suffix, denom);
        witness[i] = inverse_value;
    }
    return witness;
}

std::vector<Fr> multiplicity_evals(const std::vector<Fr>& query_values, const std::vector<Fr>& table_values)
{
    std::vector<Fr> multiplicities(table_values.size(), Fr(0));
    std::map<Fr, size_t> first_index_by_value;
    for (size_t i = 0; i < table_values.size(); ++i) {
        first_index_by_value.emplace(table_values[i], i);
    }

    for (const Fr& query_value : query_values) {
        const auto it = first_index_by_value.find(query_value);
        if (it != first_index_by_value.end()) {
            Fr::add(multiplicities[it->second], multiplicities[it->second], Fr(1));
        }
    }
    return multiplicities;
}

std::vector<Fr> contiguous_range_table_evals(size_t table_domain_size)
{
    std::vector<Fr> table_values;
    table_values.reserve(table_domain_size);
    Fr current(0);
    const Fr one(1);
    for (size_t i = 0; i < table_domain_size; ++i) {
        table_values.push_back(current);
        Fr::add(current, current, one);
    }
    return table_values;
}

std::vector<Fr> range_table_multiplicity_evals(const std::vector<Fr>& query_values, size_t table_domain_size)
{
    std::vector<Fr> multiplicities(table_domain_size, Fr(0));
    const Fr one(1);
    for (const Fr& query_value : query_values) {
        uint64_t parsed = 0;
        try {
            parsed = query_value.getUint64();
        } catch (const std::exception&) {
            continue;
        }
        if (parsed >= table_domain_size) {
            continue;
        }
        Fr::add(multiplicities[static_cast<size_t>(parsed)], multiplicities[static_cast<size_t>(parsed)], one);
    }
    return multiplicities;
}

bool is_less_check_range_lookup_label(const std::string& label)
{
    return label.rfind("less_check.", 0) == 0;
}

void report_missing_lookup_values(const std::vector<Fr>& query_values,
                                  const std::vector<Fr>& table_values,
                                  size_t query_domain_size,
                                  size_t table_domain_size)
{
    if (!lookup_failure_debug_enabled()) {
        return;
    }

    std::set<Fr> table_set(table_values.begin(), table_values.end());
    size_t missing_count = 0;
    std::vector<std::pair<size_t, Fr>> first_missing;
    for (size_t i = 0; i < query_values.size(); ++i) {
        if (table_set.find(query_values[i]) != table_set.end()) {
            continue;
        }
        ++missing_count;
        if (first_missing.size() < 16) {
            first_missing.emplace_back(i, query_values[i]);
        }
    }
    if (missing_count == 0) {
        return;
    }

    std::cerr << "lookup_debug missing_values"
              << " query_domain_size=" << query_domain_size
              << " table_domain_size=" << table_domain_size
              << " missing_count=" << missing_count;
    for (const auto& [index, value] : first_missing) {
        std::cerr << " sample=" << index << ":" << value.getStr(10);
    }
    std::cerr << "\n";
}


std::vector<Fr> weighted_inverse_witness_evals(const std::vector<Fr>& table_values,
                                               const std::vector<Fr>& multiplicities,
                                               const Fr& x)
{
    if (table_values.size() != multiplicities.size()) {
        throw std::runtime_error("lookupProtocol: table and multiplicity sizes mismatch");
    }

    std::vector<Fr> witness(table_values.size(), Fr(0));
    Fr prefix_product(1);
    for (size_t i = 0; i < table_values.size(); ++i) {
        const Fr denom = x + table_values[i];
        if (denom.isZero()) {
            throw std::runtime_error("lookupProtocol: sampled challenge hits a pole");
        }
        witness[i] = prefix_product;
        Fr::mul(prefix_product, prefix_product, denom);
    }

    Fr inverse_suffix;
    Fr::inv(inverse_suffix, prefix_product);
    for (size_t offset = 0; offset < table_values.size(); ++offset) {
        const size_t i = table_values.size() - 1 - offset;
        const Fr denom = x + table_values[i];
        Fr inverse_value;
        Fr::mul(inverse_value, inverse_suffix, witness[i]);
        Fr::mul(inverse_suffix, inverse_suffix, denom);
        Fr::mul(witness[i], multiplicities[i], inverse_value);
    }
    return witness;
}

Fr sum_evaluations(const std::vector<Fr>& evals)
{
    Fr sum = Fr(0);
    for (const Fr& value : evals) {
        Fr::add(sum, sum, value);
    }
    return sum;
}

std::string encode_fr_list(const std::vector<Fr>& values)
{
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i].getStr(10);
    }
    return out.str();
}

std::string encode_sparse_terms(const std::vector<std::pair<size_t, Fr>>& terms)
{
    std::ostringstream out;
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i != 0) {
            out << ";";
        }
        out << terms[i].first << ":" << terms[i].second.getStr(10);
    }
    return out.str();
}

std::string vir_commitment_structural_key(const virCommitment& commitment)
{
    switch (commitment.type) {
    case virCommitment::Type::KZG:
        return "K(" + transcript_representation(commitment.com) + ")";
    case virCommitment::Type::SUM:
        return "A(" + vir_commitment_structural_key(*commitment.left) + "," +
               vir_commitment_structural_key(*commitment.right) + ")";
    case virCommitment::Type::PRODUCT:
        return "M(" + vir_commitment_structural_key(*commitment.left) + "," +
               vir_commitment_structural_key(*commitment.right) + ")";
    case virCommitment::Type::FRACTION:
        return "D(" + vir_commitment_structural_key(*commitment.left) + "," +
               vir_commitment_structural_key(*commitment.right) + ")";
    case virCommitment::Type::SPARSE_POLY:
        if (commitment.leaf.is_sparse()) {
            return "P{s:" + encode_sparse_terms(commitment.leaf.terms()) + "}";
        }
        return "P{d:" + encode_fr_list(commitment.leaf.coefficients()) + "}";
    case virCommitment::Type::LAGRANGE:
        return "L(" + std::to_string(commitment.lagrange_subgroup_size) + "," +
               std::to_string(commitment.lagrange_index) + ")";
    case virCommitment::Type::LAGRANGE_LINEAR_COMBINATION: {
        std::string key = "LC(" + std::to_string(commitment.lagrange_subgroup_size);
        for (const auto& [index, coefficient] : commitment.lagrange_terms) {
            key += "," + std::to_string(index) + ":" + coefficient.getStr(16);
        }
        key += ")";
        return key;
    }
    case virCommitment::Type::X_POWER:
        return "XP(" + std::to_string(commitment.exponent) + "," +
               vir_commitment_structural_key(*commitment.left) + ")";
    case virCommitment::Type::X_SCALE:
        return "XS(" + commitment.scale.getStr(10) + "," +
               vir_commitment_structural_key(*commitment.left) + ")";
    case virCommitment::Type::INTERLEAVE:
        return "I(" + std::to_string(commitment.exponent) + "," +
               vir_commitment_structural_key(*commitment.left) + "," +
               vir_commitment_structural_key(*commitment.right) + ")";
    }

    throw std::runtime_error("lookupProtocol: unknown virCommitment type");
}

} // namespace

Fr lookupVerifier::challenge()
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("lookupVerifier not setup");
    }
    return params->kzg->transcript().squeeze_fr("lookup.challenge");
}

void lookupVerifier::add_check(const virCommitment& multiplicity, const lookupCheckRecord& record)
{
    multiplicities.push_back(multiplicity);
    checks.push_back(record);
}

bool lookupVerifier::finalize() const
{
    for (const lookupCheckRecord& check : checks) {
        if (check.query_claim != check.table_claim) {
            return false;
        }
    }
    return true;
}

std::pair<virCommitment, lookupCheckRecord> lookupProver::add_check(const virCommitment& f,
                                                                    const virCommitment& t,
                                                                    const Fr& challenge,
                                                                    size_t n,
                                                                    size_t N)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("lookupProver not setup");
    }

    const std::vector<Fr> f_evals = params->kzg->evaluations(f, n);
    const std::vector<Fr> t_evals = params->kzg->evaluations(t, N);

    const std::vector<Fr> c_evals = multiplicity_evals(f_evals, t_evals);
    const std::vector<Fr> query_inverse_evals = inverse_witness_evals(f_evals, challenge);
    const std::vector<Fr> table_weighted_inverse_evals =
        weighted_inverse_witness_evals(t_evals, c_evals, challenge);

    const poly c_poly = poly_from_evals(c_evals);
    const poly query_inverse = poly_from_evals(query_inverse_evals);
    const poly table_weighted_inverse = poly_from_evals(table_weighted_inverse_evals);

    const Fr query_claim = sum_evaluations(query_inverse_evals);
    const Fr table_claim = sum_evaluations(table_weighted_inverse_evals);

    lookupCheckRecord record;
    record.query_domain_size = n;
    record.table_domain_size = N;
    record.challenge = challenge;
    record.query_claim = query_claim;
    record.table_claim = table_claim;
    record.multiplicity_commitment = params->kzg->commit(c_poly, N - 1);
    record.query_inverse_commitment = params->kzg->commit(query_inverse, n - 1);
    record.table_weighted_inverse_commitment = params->kzg->commit(table_weighted_inverse, N - 1);

    checks.push_back(record);
    return {virCommitment(record.multiplicity_commitment), record};
}

void lookupProtocol::add_check(const virCommitment& f,
                               const virCommitment& t,
                               size_t n,
                               size_t N,
                               std::string debug_label)
{
    structured_protocol_profile::timed_void(
        "protocol_call",
        "lookup.add_check",
        "query_domain_size=" + std::to_string(n) + ";table_domain_size=" + std::to_string(N),
        [&] {
            if (!is_setup) {
                throw std::runtime_error("lookupProtocol not setup");
            }
            if (params.kzg == nullptr || params.zero_check == nullptr || params.sum_check == nullptr) {
                throw std::runtime_error("lookupProtocol subprotocols not setup");
            }
            if (!isPowerOfTwo(n) || !isPowerOfTwo(N) || n == 0 || N == 0) {
                throw std::runtime_error("lookupProtocol: domain sizes must be non-zero powers of two");
            }

            proof_cost::record_relation("lookup", n);
            if (params.soundness_counter != nullptr) {
                params.soundness_counter->record_logical_lookup(n, N);
            }
            pending_checks.push_back(pendingLookupCheck{f, t, n, N, std::move(debug_label)});
        });
}

void lookupProtocol::materialize_pending_checks()
{
    if (pending_checks.empty()) {
        return;
    }

    struct GroupedLookupChecks {
        virCommitment table;
        size_t table_domain_size = 0;
        std::vector<InterleavedBatchNode> query_nodes;
        bool table_is_contiguous_range = true;
    };

    std::vector<GroupedLookupChecks> grouped_checks;
    std::map<std::string, size_t> grouped_index_by_key;
    grouped_checks.reserve(pending_checks.size());

    lookup_profile_scope("materialize.group_pending", [&] {
        for (const pendingLookupCheck& pending : pending_checks) {
            if (lookup_failure_debug_enabled()) {
                std::cerr << "lookup_debug pending_check"
                          << " query_domain_size=" << pending.query_domain_size
                          << " table_domain_size=" << pending.table_domain_size
                          << " label=" << pending.debug_label
                          << "\n";
            }
            const std::string key =
                std::to_string(pending.table_domain_size) + ":" + vir_commitment_structural_key(pending.table);
            auto it = grouped_index_by_key.find(key);
            if (it == grouped_index_by_key.end()) {
                const size_t index = grouped_checks.size();
                grouped_index_by_key.emplace(key, index);
                grouped_checks.push_back(GroupedLookupChecks{
                    pending.table,
                    pending.table_domain_size,
                    {},
                    is_less_check_range_lookup_label(pending.debug_label)});
                grouped_checks.back().query_nodes.push_back(
                    InterleavedBatchNode{{pending.query}, pending.query_domain_size});
                continue;
            }
            grouped_checks[it->second].table_is_contiguous_range =
                grouped_checks[it->second].table_is_contiguous_range &&
                is_less_check_range_lookup_label(pending.debug_label);
            grouped_checks[it->second].query_nodes.push_back(
                InterleavedBatchNode{{pending.query}, pending.query_domain_size});
        }
    });

    lookup_profile_scope("materialize.add_grouped_checks", [&] {
        for (const GroupedLookupChecks& grouped : grouped_checks) {
        virCommitment merged_query;
        size_t merged_query_domain_size = 0;
        if (grouped.query_nodes.size() == 1) {
            merged_query = grouped.query_nodes[0].commitments[0];
            merged_query_domain_size = grouped.query_nodes[0].domain_size;
        } else {
            const InterleavedBatchNode merged_queries =
                merge_interleaved_nodes(grouped.query_nodes, "lookupProtocol");
            merged_query = merged_queries.commitments[0];
            merged_query_domain_size = merged_queries.domain_size;
        }
        add_materialized_check(merged_query,
                               grouped.table,
                               merged_query_domain_size,
                               grouped.table_domain_size,
                               grouped.table_is_contiguous_range);
        }
    });
    pending_checks.clear();
}

void lookupProtocol::add_materialized_check(const virCommitment& f,
                                            const virCommitment& t,
                                            size_t n,
                                            size_t N,
                                            bool table_is_contiguous_range)
{
    if (!isPowerOfTwo(n) || !isPowerOfTwo(N) || n == 0 || N == 0) {
        throw std::runtime_error("lookupProtocol: domain sizes must be non-zero powers of two");
    }
    if (params.soundness_counter != nullptr) {
        params.soundness_counter->record_materialized_lookup(n, N);
    }

    lookupCheckRecord record;
    const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
    std::vector<Fr> prover_f_evals;
    std::vector<Fr> prover_t_evals;
    std::vector<Fr> prover_c_evals;
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        // `exchange_*` makes the multiplicity commitment transcript-visible, so
        // both sides bind the later challenge to the same commitment.
        const Group1 multiplicity_group = proof_session->exchange_group1("lookup.multiplicity", {});
        record.multiplicity_commitment = params.kzg->adopt_commitment(multiplicity_group);
        params.kzg->prove_degree(record.multiplicity_commitment, N - 1);
    } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
        lookup_profile_scope("prove.prechallenge.evaluate_query", [&] {
            prover_f_evals = params.kzg->evaluations(f, n, kzgProtocol::EvaluationMode::PreferEval);
        });
        lookup_profile_scope("prove.prechallenge.evaluate_table", [&] {
            if (table_is_contiguous_range) {
                prover_t_evals = contiguous_range_table_evals(N);
            } else {
                prover_t_evals = params.kzg->evaluations(t, N, kzgProtocol::EvaluationMode::PreferEval);
            }
        });
        lookup_profile_scope("prove.prechallenge.multiplicity", [&] {
            if (table_is_contiguous_range) {
                prover_c_evals = range_table_multiplicity_evals(prover_f_evals, N);
            } else {
                prover_c_evals = multiplicity_evals(prover_f_evals, prover_t_evals);
            }
        });
        report_missing_lookup_values(prover_f_evals, prover_t_evals, n, N);
        const poly multiplicity_poly = poly_from_evals(prover_c_evals);
        Group1 expected_group;
        lookup_profile_scope("prove.commit_multiplicity", [&] {
            expected_group = params.kzg->commit_g1_group(multiplicity_poly);
        });
        const Group1 multiplicity_group =
            proof_session->exchange_group1("lookup.multiplicity", [&] { return expected_group; });
        if (multiplicity_group != expected_group) {
            throw std::runtime_error("lookupProtocol: exchanged multiplicity commitment mismatch");
        }
        kzgCommitment precomputed_multiplicity;
        precomputed_multiplicity.com = expected_group;
        record.multiplicity_commitment = params.kzg->commit(multiplicity_poly, precomputed_multiplicity);
        if (record.multiplicity_commitment.com != expected_group) {
            throw std::runtime_error("lookupProtocol: multiplicity commitment registration mismatch");
        }
        params.kzg->prove_degree(record.multiplicity_commitment, N - 1);
    }

    const Fr challenge = verifier.challenge();
    record.query_domain_size = n;
    record.table_domain_size = N;
    record.challenge = challenge;

    virCommitment c;
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        const Group1 query_inverse_group = proof_session->transport_group1({});
        record.query_inverse_commitment = params.kzg->adopt_commitment(query_inverse_group);
        params.kzg->prove_degree(record.query_inverse_commitment, n - 1);

        const Group1 table_weighted_inverse_group = proof_session->transport_group1({});
        record.table_weighted_inverse_commitment = params.kzg->adopt_commitment(table_weighted_inverse_group);
        params.kzg->prove_degree(record.table_weighted_inverse_commitment, N - 1);

        record.query_claim = proof_session->exchange_fr("lookup.query_claim", {});
        record.table_claim = proof_session->exchange_fr("lookup.table_claim", {});
        c = virCommitment(record.multiplicity_commitment);
    } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
        std::vector<Fr> f_evals;
        std::vector<Fr> t_evals;
        std::vector<Fr> c_evals;
        std::vector<Fr> query_inverse_evals;
        std::vector<Fr> table_weighted_inverse_evals;
        lookup_profile_scope("prove.evaluate_query", [&] { f_evals = std::move(prover_f_evals); });
        lookup_profile_scope("prove.evaluate_table", [&] { t_evals = std::move(prover_t_evals); });
        lookup_profile_scope("prove.multiplicity", [&] { c_evals = std::move(prover_c_evals); });
        lookup_profile_scope("prove.query_inverse", [&] { query_inverse_evals = inverse_witness_evals(f_evals, challenge); });
        lookup_profile_scope("prove.table_weighted_inverse", [&] {
            table_weighted_inverse_evals = weighted_inverse_witness_evals(t_evals, c_evals, challenge);
        });

        const poly query_inverse = poly_from_evals(query_inverse_evals);
        const poly table_weighted_inverse = poly_from_evals(table_weighted_inverse_evals);

        record.query_claim = sum_evaluations(query_inverse_evals);
        record.table_claim = sum_evaluations(table_weighted_inverse_evals);
        lookup_profile_scope("prove.commit_query_inverse", [&] {
            record.query_inverse_commitment = params.kzg->commit(query_inverse, n - 1);
        });
        lookup_profile_scope("prove.commit_table_weighted_inverse", [&] {
            record.table_weighted_inverse_commitment = params.kzg->commit(table_weighted_inverse, N - 1);
        });

        const Group1 query_inverse_group =
            proof_session->transport_group1([&] { return record.query_inverse_commitment.com; });
        const Group1 table_weighted_inverse_group =
            proof_session->transport_group1([&] { return record.table_weighted_inverse_commitment.com; });
        if (query_inverse_group != record.query_inverse_commitment.com ||
            table_weighted_inverse_group != record.table_weighted_inverse_commitment.com) {
            throw std::runtime_error("lookupProtocol: transported commitments mismatch");
        }
        record.query_claim = proof_session->exchange_fr("lookup.query_claim", [&] { return record.query_claim; });
        record.table_claim = proof_session->exchange_fr("lookup.table_claim", [&] { return record.table_claim; });
        prover.checks.push_back(record);
        c = virCommitment(record.multiplicity_commitment);
    } else {
        auto [prover_c, prover_record] = prover.add_check(f, t, challenge, n, N);
        c = prover_c;
        record = prover_record;
    }
    verifier.add_check(c, record);

    virCommitment query_inverse_vir(record.query_inverse_commitment);
    virCommitment table_weighted_inverse_vir(record.table_weighted_inverse_commitment);
    // Equality of the two scalar claims is checked in `lookupVerifier`.
    lookup_profile_scope("sum_check.query_inverse", [&] {
        params.sum_check->add_check(query_inverse_vir, record.query_claim, n);
    });
    lookup_profile_scope("sum_check.table_weighted_inverse", [&] {
        params.sum_check->add_check(table_weighted_inverse_vir, record.table_claim, N);
    });

    const virCommitment challenge_poly(record.challenge);
    // Pointwise inverse identities:
    //   (f + beta) * query_inverse = 1
    //   (t + beta) * table_weighted_inverse = multiplicity.
    virCommitment query_inverse_zero = (f + challenge_poly) * query_inverse_vir - Fr(1);
    virCommitment table_weighted_inverse_zero = (t + challenge_poly) * table_weighted_inverse_vir - c;
    lookup_profile_scope("zero_check.query_inverse", [&] {
        params.zero_check->add_check(query_inverse_zero, n);
    });
    lookup_profile_scope("zero_check.table_weighted_inverse", [&] {
        params.zero_check->add_check(table_weighted_inverse_zero, N);
    });
}
