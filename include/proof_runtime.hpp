#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <variant>

#include "argmax.hpp"
#include "box_check.hpp"
#include "concatenated_table_lookup.hpp"
#include "cq_concatenated_table_lookup.hpp"
#include "cq_lookup.hpp"
#include "column_replication.hpp"
#include "cube_argmax.hpp"
#include "cube_softmax.hpp"
#include "cube_sum.hpp"
#include "div_check.hpp"
#include "exp_check.hpp"
#include "forest_inference.hpp"
#include "forest_histogram.hpp"
#include "forest_histogram_refresh.hpp"
#include "forest_histogram_bucket_address.hpp"
#include "forest_training_inference.hpp"
#include "forest_boosting_state.hpp"
#include "forest_depth_consistency.hpp"
#include "forest_leaf_value.hpp"
#include "forest_quantized_split_gain.hpp"
#include "forest_split_gain.hpp"
#include "forest_split_selection.hpp"
#include "forest_training_proof.hpp"
#include "forest_split_update.hpp"
#include "forest_split_statistics.hpp"
#include "forest_classification.hpp"
#include "forest_prediction.hpp"
#include "forest_row_fetch.hpp"
#include "forest_structure.hpp"
#include "forest_tree_layout.hpp"
#include "fixed_point_division.hpp"
#include "hist_agg.hpp"
#include "index_routing.hpp"
#include "kzg.hpp"
#include "lagrange_utils.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "masked_argmax.hpp"
#include "matrix_sum.hpp"
#include "matrix_utils.hpp"
#include "nonzero_check.hpp"
#include "node_role_consistency.hpp"
#include "parent_child_box_inheritance.hpp"
#include "parent_child_tree_consistency.hpp"
#include "proof_artifact.hpp"
#include "protocol_profile.hpp"
#include "profiling/soundness_counter.hpp"
#include "proof_session.hpp"
#include "rational_check.hpp"
#include "root_box_range.hpp"
#include "root_selector_consistency.hpp"
#include "selector_check.hpp"
#include "shuffle.hpp"
#include "signed_range_check.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "terrae_env.hpp"
#include "tree_histogram_consistency.hpp"
#include "zero_check.hpp"
#include "zkboost/primitives/histogram_ram.hpp"

namespace proof_runtime_detail {

inline bool profile_enabled(const char* env_flag)
{
    const char* flag = terrae_getenv(env_flag);
    return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
}

template <class Fn>
inline void profile_scope(const char* env_flag, const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = profile_enabled(env_flag);
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    try {
        fn();
    } catch (const std::exception& ex) {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        if (stderr_enabled) {
            std::cerr << label << " " << elapsed.count() << "\n";
        }
        structured_protocol_profile::append_row(
            "runtime_scope",
            label,
            std::string("env_flag=") + env_flag,
            elapsed.count(),
            std::string("throw:") + ex.what());
        throw;
    } catch (...) {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        if (stderr_enabled) {
            std::cerr << label << " " << elapsed.count() << "\n";
        }
        structured_protocol_profile::append_row(
            "runtime_scope", label, std::string("env_flag=") + env_flag, elapsed.count(), "throw");
        throw;
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row(
        "runtime_scope", label, std::string("env_flag=") + env_flag, elapsed.count(), "ok");
}

} // namespace proof_runtime_detail

class PublicPolynomialInput {
public:
    PublicPolynomialInput(std::string key, const poly& polynomial, std::optional<size_t> degree_bound = std::nullopt)
        : key(std::move(key))
        , polynomial(polynomial)
        , degree_bound(degree_bound)
    {
    }

    std::string key;
    poly polynomial;
    std::optional<size_t> degree_bound;
};

class PublicCommitmentInput {
public:
    static PublicCommitmentInput prover(std::string key,
                                        const poly& polynomial,
                                        std::optional<size_t> degree_bound = std::nullopt)
    {
        return PublicCommitmentInput(std::move(key), polynomial, std::nullopt, degree_bound);
    }

    static PublicCommitmentInput verifier(std::string key,
                                          const Group1& commitment,
                                          std::optional<size_t> degree_bound = std::nullopt)
    {
        return PublicCommitmentInput(std::move(key), std::nullopt, commitment, degree_bound);
    }

    std::string key;
    std::optional<poly> polynomial;
    std::optional<Group1> commitment;
    std::optional<size_t> degree_bound;

private:
    PublicCommitmentInput(std::string key,
                          std::optional<poly> polynomial,
                          std::optional<Group1> commitment,
                          std::optional<size_t> degree_bound)
        : key(std::move(key))
        , polynomial(std::move(polynomial))
        , commitment(std::move(commitment))
        , degree_bound(degree_bound)
    {
    }
};

class WitnessCommitmentInput {
public:
    static WitnessCommitmentInput from_optional(std::string label,
                                                std::optional<poly> polynomial,
                                                std::optional<size_t> degree_bound = std::nullopt)
    {
        return WitnessCommitmentInput(std::move(label), std::move(polynomial), degree_bound);
    }

    static WitnessCommitmentInput prover(std::string label,
                                         const poly& polynomial,
                                         std::optional<size_t> degree_bound = std::nullopt)
    {
        return WitnessCommitmentInput(std::move(label), polynomial, degree_bound);
    }

    static WitnessCommitmentInput verifier(std::string label, std::optional<size_t> degree_bound = std::nullopt)
    {
        return WitnessCommitmentInput(std::move(label), std::nullopt, degree_bound);
    }

    std::string label;
    std::optional<poly> polynomial;
    std::optional<size_t> degree_bound;

private:
    WitnessCommitmentInput(std::string label,
                           std::optional<poly> polynomial,
                           std::optional<size_t> degree_bound)
        : label(std::move(label))
        , polynomial(std::move(polynomial))
        , degree_bound(degree_bound)
    {
    }
};

using ProtocolPolynomialInput = std::variant<PublicPolynomialInput, PublicCommitmentInput, WitnessCommitmentInput>;
using CqLookupPublicInput = std::variant<PublicPolynomialInput, cqLookupPublicTable>;
using CqTableLookupPublicInput = std::variant<PublicPolynomialInput, cqTableLookupPublicTable>;

inline virCommitment forest_tree_layout_expression(size_t tree_count)
{
    if (tree_count == 0 || !isPowerOfTwo(tree_count)) {
        throw std::runtime_error("forest_tree_layout_expression: tree_count must be a non-zero power of two");
    }

    std::vector<std::pair<size_t, Fr>> terms;
    terms.reserve(tree_count);
    for (size_t tree = 0; tree < tree_count; ++tree) {
        terms.emplace_back(tree, field_from_size_t(tree + 1));
    }
    return subgroup_lagrange_linear_combination_expression(tree_count, terms);
}

inline poly forest_tree_layout_polynomial(size_t tree_count)
{
    if (tree_count == 0 || !isPowerOfTwo(tree_count)) {
        throw std::runtime_error("forest_tree_layout_polynomial: tree_count must be a non-zero power of two");
    }

    poly layout;
    for (size_t tree = 0; tree < tree_count; ++tree) {
        layout = layout + subgroup_lagrange_polynomial(tree_count, tree) * field_from_size_t(tree + 1);
    }
    return layout;
}

class ProofRuntime {
public:
    static ProofRuntime prover(size_t deg1,
                               size_t deg2,
                               std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>())
    {
        (void)deg1;
        (void)deg2;
        (void)transcript;
        throw std::runtime_error(
            "ProofRuntime::prover(deg1, deg2) is disabled; load or generate explicit kzgPublicParameters and call "
            "ProofRuntime::prover(public_params) instead");
    }

    static ProofRuntime prover(
        const kzgPublicParameters& public_params,
        std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>(),
        ProofEncoding encoding = ProofEncoding::Binary)
    {
        return ProofRuntime(ProofMode::Prove, public_params, SerializedProof{}, std::move(transcript), encoding);
    }
    // The returned runtime borrows public_params; the caller must keep it alive.
    // prove_with_runtime confines the runtime to that lifetime.
    static ProofRuntime prover_borrowed(
        const kzgPublicParameters& public_params,
        std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>(),
        ProofEncoding encoding = ProofEncoding::Binary)
    {
        return ProofRuntime(ProofMode::Prove,
                            public_params,
                            SerializedProof{},
                            std::move(transcript),
                            encoding,
                            BorrowedSrsTag{});
    }

    static ProofRuntime counter(size_t deg1_plus_1, size_t deg2_plus_1,
                                std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>())
    {
        return ProofRuntime(CountOnlyTag{}, deg1_plus_1, deg2_plus_1, std::move(transcript));
    }

    static ProofRuntime verifier(
        const kzgPublicParameters& public_params,
        const SerializedProof& proof,
        std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>())
    {
        return ProofRuntime(ProofMode::Verify, public_params, proof, std::move(transcript));
    }

    static ProofRuntime verifier(
        kzgPublicParameters&& public_params,
        const SerializedProof& proof,
        std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>())
    {
        return ProofRuntime(ProofMode::Verify, std::move(public_params), proof, std::move(transcript));
    }

    // The returned runtime borrows public_params; the caller must keep it alive
    // until verification and runtime destruction have both completed.
    static ProofRuntime verifier_borrowed(
        const kzgPublicParameters& public_params,
        const SerializedProof& proof,
        std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>())
    {
        return ProofRuntime(ProofMode::Verify,
                            public_params,
                            proof,
                            std::move(transcript),
                            ProofEncoding::Binary,
                            BorrowedSrsTag{});
    }

    static std::unique_ptr<ProofRuntime> verifier_ptr(
        const kzgPublicParameters& public_params,
        const SerializedProof& proof,
        std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>())
    {
        return std::unique_ptr<ProofRuntime>(
            new ProofRuntime(ProofMode::Verify, public_params, proof, std::move(transcript)));
    }

    ProofMode mode() const noexcept
    {
        return mode_;
    }

    bool is_count_only() const noexcept
    {
        return session_ && session_->count_only();
    }

    kzgProtocol& kzg()
    {
        return kzg_;
    }

    const kzgProtocol& kzg() const
    {
        return kzg_;
    }

    std::shared_ptr<ProofSession> session() const
    {
        return session_;
    }

    kzgPublicParameters public_parameters() const
    {
        return kzg_.public_parameters();
    }

    SerializedProof take_proof() const
    {
        if (!session_) {
            throw std::runtime_error("ProofRuntime session not initialized");
        }
        return session_->take_proof();
    }

    bool proof_fully_consumed() const
    {
        if (!session_) {
            throw std::runtime_error("ProofRuntime session not initialized");
        }
        return session_->proof_fully_consumed();
    }

    std::vector<uint8_t> transcript_state() const
    {
        if (!session_) {
            throw std::runtime_error("ProofRuntime session not initialized");
        }
        return session_->transcript().state();
    }

    kzgCommitment commit_public(const std::string& key, const poly& p)
    {
        return kzg_.commit_public(key, p);
    }

    kzgCommitment commit_public(const std::string& key, const poly& p, size_t max_degree)
    {
        return kzg_.commit_public(key, p, max_degree);
    }

    kzgCommitment bind_public_commitment(const std::string& key,
                                         const Group1& commitment,
                                         std::optional<size_t> degree_bound = std::nullopt)
    {
        (void)degree_bound;
        return kzg_.adopt_public_commitment(key, commitment);
    }

    virCommitment bind_virtual_public_commitment(const std::string& key,
                                                 const Group1& commitment,
                                                 std::optional<size_t> degree_bound = std::nullopt)
    {
        return virCommitment(bind_public_commitment(key, commitment, degree_bound));
    }

    kzgCommitment bind(const PublicPolynomialInput& input)
    {
        if (input.degree_bound.has_value()) {
            return kzg_.commit_public(input.key, input.polynomial, *input.degree_bound);
        }
        return kzg_.commit_public(input.key, input.polynomial);
    }

    virCommitment bind_virtual(const PublicPolynomialInput& input)
    {
        return virCommitment(bind(input));
    }

    kzgCommitment bind(const PublicCommitmentInput& input)
    {
        if (mode_ == ProofMode::Prove) {
            if (!input.polynomial.has_value()) {
                throw std::runtime_error("ProofRuntime prover public commitment input missing polynomial: " + input.key);
            }
            if (input.degree_bound.has_value()) {
                return kzg_.commit_public(input.key, *input.polynomial, *input.degree_bound);
            }
            return kzg_.commit_public(input.key, *input.polynomial);
        }

        if (!input.commitment.has_value()) {
            throw std::runtime_error("ProofRuntime verifier public commitment input missing group commitment: " + input.key);
        }

        const kzgCommitment commitment = kzg_.adopt_public_commitment(input.key, *input.commitment);
        if (input.degree_bound.has_value()) {
            if (!kzg_.has_degree_bound(virCommitment(commitment), *input.degree_bound)) {
                kzg_.prove_degree(commitment, *input.degree_bound);
            }
        }
        return commitment;
    }

    virCommitment bind_virtual(const PublicCommitmentInput& input)
    {
        return virCommitment(bind(input));
    }

    kzgCommitment bind(const WitnessCommitmentInput& input)
    {
        if (!session_) {
            throw std::runtime_error("ProofRuntime session not initialized");
        }

        if (mode_ == ProofMode::Prove) {
            if (!input.polynomial.has_value()) {
                throw std::runtime_error("ProofRuntime prover witness input missing polynomial: " + input.label);
            }

            const kzgCommitment commitment = input.degree_bound.has_value()
                                                 ? kzg_.commit(*input.polynomial, *input.degree_bound)
                                                 : kzg_.commit(*input.polynomial);
            const Group1 transported = session_->transport_group1([&] { return commitment.com; });
            if (transported != commitment.com) {
                throw std::runtime_error("ProofRuntime witness commitment transport mismatch: " + input.label);
            }
            return commitment;
        }

        if (input.polynomial.has_value()) {
            throw std::runtime_error("ProofRuntime verifier witness input must not include polynomial: " + input.label);
        }

        const Group1 group_commitment = session_->transport_group1({});
        const kzgCommitment commitment = kzg_.adopt_commitment(group_commitment);
        if (input.degree_bound.has_value()) {
            kzg_.prove_degree(commitment, *input.degree_bound);
        }
        return commitment;
    }

    virCommitment bind_virtual(const WitnessCommitmentInput& input)
    {
        return virCommitment(bind(input));
    }

    kzgCommitment bind(const ProtocolPolynomialInput& input)
    {
        return std::visit([this](const auto& value) { return bind(value); }, input);
    }

    virCommitment bind_virtual(const ProtocolPolynomialInput& input)
    {
        return virCommitment(bind(input));
    }

    bool prepare_kzg_verifier_finalization()
    {
        return kzg_.prepare_verifier_finalization();
    }

    bool finalize_kzg()
    {
        return kzg_.finalize();
    }

private:
    struct BorrowedSrsTag {};
    struct CountOnlyTag {};

    ProofRuntime(CountOnlyTag,
                 size_t deg1_plus_1,
                 size_t deg2_plus_1,
                 std::shared_ptr<ProofTranscript> transcript)
        : mode_(ProofMode::Verify)
    {
        session_ = ProofSession::counter(std::move(transcript));
        kzg_.setup_count_only(deg1_plus_1, deg2_plus_1);
        kzg_.set_proof_session(session_);
    }
    ProofRuntime(ProofMode mode,
                 std::shared_ptr<ProofTranscript> transcript,
                 SerializedProof proof = {},
                 ProofEncoding prover_encoding = ProofEncoding::Binary)
        : mode_(mode)
    {
        session_ = mode == ProofMode::Prove
                       ? ProofSession::prover(std::move(transcript), prover_encoding)
                       : ProofSession::verifier(proof, std::move(transcript));
    }

    ProofRuntime(ProofMode mode,
                 const kzgPublicParameters& public_params,
                 const SerializedProof& proof,
                 std::shared_ptr<ProofTranscript> transcript,
                 ProofEncoding prover_encoding = ProofEncoding::Binary)
        : ProofRuntime(mode, std::move(transcript), proof, prover_encoding)
    {
        kzg_.setup(public_params);
        kzg_.set_proof_session(session_);
    }

    ProofRuntime(ProofMode mode,
                 const kzgPublicParameters& public_params,
                 const SerializedProof& proof,
                 std::shared_ptr<ProofTranscript> transcript,
                 ProofEncoding prover_encoding,
                 BorrowedSrsTag)
        : ProofRuntime(mode, std::move(transcript), proof, prover_encoding)
    {
        kzg_.setup_borrowed(public_params);
        kzg_.set_proof_session(session_);
    }

    ProofRuntime(ProofMode mode,
                 kzgPublicParameters&& public_params,
                 const SerializedProof& proof,
                 std::shared_ptr<ProofTranscript> transcript,
                 ProofEncoding prover_encoding = ProofEncoding::Binary)
        : ProofRuntime(mode, std::move(transcript), proof, prover_encoding)
    {
        kzg_.setup(std::move(public_params));
        kzg_.set_proof_session(session_);
    }

    ProofMode mode_;
    std::shared_ptr<ProofSession> session_;
    kzgProtocol kzg_;
};

class ProofProgram {
public:
    class CqLookupContext {
    public:
        CqLookupContext(ProofRuntime& runtime, const PublicPolynomialInput& table_input, size_t table_domain_size)
            : runtime_(runtime)
        {
            table_.setup(runtime_.kzg(), runtime_.bind_virtual(table_input), table_domain_size);
            lookup_.setup(runtime_.kzg(), table_);
        }

        CqLookupContext(ProofRuntime& runtime, const cqLookupPublicTable& table_input)
            : runtime_(runtime)
        {
            const cqLookupPublicTable verifier_safe_table =
                runtime_.mode() == ProofMode::Verify ? table_input.verifier_view() : table_input;
            const virCommitment table_commitment = runtime_.bind_virtual_public_commitment(
                verifier_safe_table.key, verifier_safe_table.commitment, verifier_safe_table.degree_bound);
            table_.setup(runtime_.kzg(), verifier_safe_table, table_commitment);
            lookup_.setup(runtime_.kzg(), table_);
        }

        cqLookupProtocol& lookup()
        {
            return lookup_;
        }

        cqLookupTable& table()
        {
            return table_;
        }

        void add_check(const ProtocolPolynomialInput& query, size_t query_domain_size)
        {
            lookup_.add_check(runtime_.bind_virtual(query), query_domain_size);
        }

        bool finalize()
        {
            return lookup_.finalize();
        }

        bool finalize_prepare_opening()
        {
            return lookup_.prepare_opening();
        }

        bool finalize_complete_opening(const Fr& opening_point)
        {
            return lookup_.complete_opening(opening_point);
        }

        bool needs_opening() const
        {
            return lookup_.needs_opening();
        }

    private:
        ProofRuntime& runtime_;
        cqLookupTable table_;
        cqLookupProtocol lookup_;
    };

    class CqTableLookupContext {
    public:
        CqTableLookupContext(ProofRuntime& runtime,
                             zeroProtocol& zero,
                             const PublicPolynomialInput& table_input,
                             size_t table_row_count,
                             size_t column_count)
            : runtime_(runtime)
            , table_lookup_(reduced_lookup_)
        {
            table_.setup(runtime_.kzg(), runtime_.bind_virtual(table_input), table_row_count, column_count);
            table_lookup_.setup(runtime_.kzg(), zero, table_);
        }

        CqTableLookupContext(ProofRuntime& runtime, zeroProtocol& zero, const cqTableLookupPublicTable& table_input)
            : runtime_(runtime)
            , table_lookup_(reduced_lookup_)
        {
            const cqTableLookupPublicTable verifier_safe_table =
                runtime_.mode() == ProofMode::Verify ? table_input.verifier_view() : table_input;
            const virCommitment table_commitment = runtime_.bind_virtual_public_commitment(
                verifier_safe_table.key, verifier_safe_table.commitment, verifier_safe_table.degree_bound);
            table_.setup(verifier_safe_table, table_commitment);
            table_lookup_.setup(runtime_.kzg(), zero, table_);
        }

        cqLookupProtocol& reduced_lookup()
        {
            return reduced_lookup_;
        }

        cqTableLookupProtocol& table_lookup()
        {
            return table_lookup_;
        }

        cqTableLookupTable& table()
        {
            return table_;
        }

        void add_check(const ProtocolPolynomialInput& query, size_t query_row_count)
        {
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.cq_table_lookup.bind_and_add_check", [&] {
                    table_lookup_.add_check(runtime_.bind_virtual(query), query_row_count);
                });
        }

        bool finalize()
        {
            bool ok = true;
            ok = table_lookup_.finalize() && ok;
            ok = reduced_lookup_.finalize() && ok;
            return ok;
        }

        bool finalize_prepare_opening()
        {
            bool ok = true;
            ok = table_lookup_.finalize() && ok;
            ok = reduced_lookup_.prepare_opening() && ok;
            return ok;
        }

        bool finalize_complete_opening(const Fr& opening_point)
        {
            return reduced_lookup_.complete_opening(opening_point);
        }

        bool needs_opening() const
        {
            return reduced_lookup_.needs_opening();
        }

    private:
        ProofRuntime& runtime_;
        cqTableLookupTable table_;
        cqLookupProtocol reduced_lookup_;
        cqTableLookupProtocol table_lookup_;
    };

    class CqConcatenatedTableLookupContext {
    public:
        CqConcatenatedTableLookupContext(ProofRuntime& runtime,
                                         zeroProtocol& zero,
                                         const cqConcatenatedTableLookupPublicTable& table_input)
            : runtime_(runtime)
            , table_input_(runtime_.mode() == ProofMode::Verify ? table_input.verifier_view() : table_input)
            , table_lookup_context_(runtime_, zero, table_input_.merged_table)
        {
        }

        void add_check(const std::vector<ProtocolPolynomialInput>& queries, size_t query_row_count)
        {
            if (queries.size() != table_input_.column_counts.size()) {
                throw std::runtime_error("CqConcatenatedTableLookupContext: query/count size mismatch");
            }

            std::vector<virCommitment> query_commitments;
            query_commitments.reserve(queries.size());
            for (const ProtocolPolynomialInput& query : queries) {
                query_commitments.push_back(runtime_.bind_virtual(query));
            }

            add_check(query_commitments, query_row_count);
        }

        void add_check(const std::vector<virCommitment>& query_commitments, size_t query_row_count)
        {
            if (query_commitments.size() != table_input_.column_counts.size()) {
                throw std::runtime_error("CqConcatenatedTableLookupContext: query/count size mismatch");
            }

            virCommitment merged_query;
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT",
                "inference_profile.cq_concatenated_lookup.build_query_expression",
                [&] {
                    merged_query = cq_concatenated_table_lookup_query_expression(
                        query_commitments, query_row_count, table_input_, "CqConcatenatedTableLookupContext");
                });
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT",
                "inference_profile.cq_concatenated_lookup.table_lookup_add_check",
                [&] { table_lookup_context_.table_lookup().add_check(merged_query, query_row_count); });
        }

        void add_check(const std::vector<virCommitment>& query_commitments,
                       size_t query_row_count,
                       const tableLookupWitness& witness)
        {
            if (query_commitments.size() != table_input_.column_counts.size()) {
                throw std::runtime_error("CqConcatenatedTableLookupContext: query/count size mismatch");
            }

            virCommitment merged_query;
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT",
                "inference_profile.cq_concatenated_lookup.build_query_expression",
                [&] {
                    merged_query = cq_concatenated_table_lookup_query_expression(
                        query_commitments, query_row_count, table_input_, "CqConcatenatedTableLookupContext");
                });
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT",
                "inference_profile.cq_concatenated_lookup.table_lookup_add_check",
                [&] { table_lookup_context_.table_lookup().add_check(merged_query, query_row_count, witness); });
        }

        bool finalize()
        {
            bool ok = false;
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.cq_concatenated_lookup.finalize", [&] {
                    ok = table_lookup_context_.finalize();
                });
            return ok;
        }

        bool finalize_prepare_opening()
        {
            return table_lookup_context_.finalize_prepare_opening();
        }

        bool finalize_complete_opening(const Fr& opening_point)
        {
            return table_lookup_context_.finalize_complete_opening(opening_point);
        }

        bool needs_opening() const
        {
            return table_lookup_context_.needs_opening();
        }

    private:
        ProofRuntime& runtime_;
        cqConcatenatedTableLookupPublicTable table_input_;
        CqTableLookupContext table_lookup_context_;
    };

    class ForestRowFetchContext {
    public:
        ForestRowFetchContext(ProofRuntime& runtime,
                              zeroProtocol& zero,
                              columnReplicationProtocol& column_replication,
                              matrixRowSumProtocol& row_sum,
                              const forestRowFetchPublicTable& table_input)
            : runtime_(runtime)
            , zero_(zero)
            , table_input_(runtime_.mode() == ProofMode::Verify ? table_input.verifier_view() : table_input)
            , column_replication_(column_replication)
            , row_sum_(row_sum)
            , lookup_context_(runtime_, zero, table_input_.lookup_table)
        {
        }

        // Schema-aware helper for the common forest fetch:
        //   tree | lower_bounds | upper_bounds | score
        void add_check(const ProtocolPolynomialInput& tree_indices,
                       const ProtocolPolynomialInput& lower_bounds,
                       const ProtocolPolynomialInput& upper_bounds,
                       const ProtocolPolynomialInput& scores,
                       size_t query_row_count)
        {
            if (table_input_.includes_leaf_index) {
                throw std::runtime_error("ForestRowFetchContext: leaf index query is required by this table");
            }

            add_bound_commitments(runtime_.bind_virtual(tree_indices),
                                  runtime_.bind_virtual(lower_bounds),
                                  runtime_.bind_virtual(upper_bounds),
                                  runtime_.bind_virtual(scores),
                                  query_row_count);
        }

        // Optional extension when the committed forest table also exports leaf indices:
        //   tree | lower_bounds | upper_bounds | score | leaf_index
        void add_check(const ProtocolPolynomialInput& tree_indices,
                       const ProtocolPolynomialInput& lower_bounds,
                       const ProtocolPolynomialInput& upper_bounds,
                       const ProtocolPolynomialInput& scores,
                       const ProtocolPolynomialInput& leaf_indices,
                       size_t query_row_count)
        {
            if (!table_input_.includes_leaf_index) {
                throw std::runtime_error("ForestRowFetchContext: this table does not include leaf indices");
            }

            add_bound_commitments(runtime_.bind_virtual(tree_indices),
                                  runtime_.bind_virtual(lower_bounds),
                                  runtime_.bind_virtual(upper_bounds),
                                  runtime_.bind_virtual(scores),
                                  runtime_.bind_virtual(leaf_indices),
                                  query_row_count);
        }

        bool finalize()
        {
            bool ok = false;
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_row_fetch.finalize", [&] {
                    if (used_compressed_lookup_) {
                        ok = finalize_compressed_lookup();
                    } else if (used_split_lookup_) {
                        ok = finalize_split_lookup();
                    } else {
                        ok = lookup_context_.finalize();
                    }
                });
            return ok;
        }

    public:
        void add_bound_commitments(const virCommitment& tree_indices,
                                   const virCommitment& lower_bounds,
                                   const virCommitment& upper_bounds,
                                   const virCommitment& scores,
                                   size_t query_row_count)
        {
            if (table_input_.includes_leaf_index) {
                throw std::runtime_error("ForestRowFetchContext: leaf index query is required by this table");
            }

            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_row_fetch.lookup_add_check", [&] {
                    lookup_context_.add_check({tree_indices, lower_bounds, upper_bounds, scores}, query_row_count);
                });
        }

        void add_bound_commitments(const virCommitment& tree_indices,
                                   const virCommitment& lower_bounds,
                                   const virCommitment& upper_bounds,
                                   const virCommitment& scores,
                                   const virCommitment& leaf_indices,
                                   size_t query_row_count)
        {
            if (!table_input_.includes_leaf_index) {
                throw std::runtime_error("ForestRowFetchContext: this table does not include leaf indices");
            }

            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_row_fetch.lookup_add_check", [&] {
                    std::optional<tableLookupWitness> witness = build_leaf_lookup_witness(leaf_indices, query_row_count);
                    if (should_use_compressed_lookup(query_row_count)) {
                        add_compressed_leaf_lookup_check(
                            tree_indices, lower_bounds, upper_bounds, scores, leaf_indices, query_row_count);
                        return;
                    }
                    if (should_use_split_lookup(query_row_count)) {
                        add_split_leaf_lookup_check(
                            tree_indices, lower_bounds, upper_bounds, scores, leaf_indices, query_row_count, witness);
                        return;
                    }

                    if (witness.has_value()) {
                        lookup_context_.add_check(
                            {tree_indices, lower_bounds, upper_bounds, scores, leaf_indices}, query_row_count, *witness);
                    } else {
                        lookup_context_.add_check(
                            {tree_indices, lower_bounds, upper_bounds, scores, leaf_indices}, query_row_count);
                    }
                });
        }

    private:
        bool split_lookup_available() const
        {
            return table_input_.includes_leaf_index && table_input_.lower_lookup_table.has_value() &&
                   table_input_.upper_lookup_table.has_value() && table_input_.scalar_lookup_table.has_value();
        }

        bool compressed_lookup_available() const
        {
            return table_input_.includes_leaf_index && table_input_.compressed_row_lookup_table.has_value();
        }

        bool should_use_compressed_lookup(size_t query_row_count) const
        {
            constexpr size_t kMinCompressedLookupMergedDomain = 65536;
            return compressed_lookup_available() &&
                   query_row_count * table_input_.lookup_table.merged_column_count >= kMinCompressedLookupMergedDomain;
        }

        bool should_use_split_lookup(size_t query_row_count) const
        {
            constexpr size_t kMinSplitLookupMergedDomain = 65536;
            return split_lookup_available() &&
                   query_row_count * table_input_.lookup_table.merged_column_count >= kMinSplitLookupMergedDomain;
        }

        void ensure_split_lookup_contexts()
        {
            if (split_lower_lookup_context_ && split_upper_lookup_context_ && split_scalar_lookup_context_) {
                return;
            }
            if (!split_lookup_available()) {
                throw std::runtime_error("ForestRowFetchContext: split lookup tables are not available");
            }
            split_lower_lookup_context_ =
                std::make_unique<CqTableLookupContext>(runtime_, zero_, *table_input_.lower_lookup_table);
            split_upper_lookup_context_ =
                std::make_unique<CqTableLookupContext>(runtime_, zero_, *table_input_.upper_lookup_table);
            split_scalar_lookup_context_ =
                std::make_unique<CqTableLookupContext>(runtime_, zero_, table_input_.scalar_lookup_table->merged_table);
        }

        std::optional<tableLookupWitness> build_leaf_lookup_witness(const virCommitment& leaf_indices,
                                                                     size_t query_row_count) const
        {
            if (runtime_.mode() != ProofMode::Prove || !table_input_.row_position_by_leaf_index) {
                return std::nullopt;
            }

            const std::vector<Fr> leaf_index_evals = runtime_.kzg().evaluations(leaf_indices, query_row_count);
            std::vector<size_t> row_positions;
            row_positions.reserve(query_row_count);
            for (const Fr& leaf_index_eval : leaf_index_evals) {
                const auto row_it = table_input_.row_position_by_leaf_index->find(leaf_index_eval);
                if (row_it == table_input_.row_position_by_leaf_index->end()) {
                    throw std::runtime_error(
                        "ForestRowFetchContext: inferred leaf index is missing from exported row table");
                }
                row_positions.push_back(row_it->second);
            }
            return tableLookupWitness::from_positions(table_input_.row_count, row_positions);
        }

        void bind_compressed_row_components(const forestCompressedRowLookupPublicTable& table)
        {
            for (const forestCompressedRowComponent& component : table.components) {
                runtime_.bind_virtual_public_commitment(component.key, component.commitment, table.row_count - 1);
            }
        }

        std::vector<Fr> compressed_row_weights(const forestCompressedRowLookupPublicTable& table, const Fr& challenge) const
        {
            std::vector<Fr> weights;
            weights.reserve(table.components.size());
            Fr current = Fr(1);
            for (size_t i = 0; i < table.components.size(); ++i) {
                weights.push_back(current);
                Fr::mul(current, current, challenge);
            }
            return weights;
        }

        virCommitment compressed_row_query_expression(const virCommitment& tree_indices,
                                                      const virCommitment& compressed_bounds,
                                                      const virCommitment& scores,
                                                      const virCommitment& leaf_indices,
                                                      const std::vector<Fr>& weights) const
        {
            const size_t feature_count = table_input_.feature_count;
            if (weights.size() != 3 + 2 * feature_count) {
                throw std::runtime_error("ForestRowFetchContext: compressed row weight count mismatch");
            }

            size_t weight_index = 0;
            virCommitment query = scale_by_constant(tree_indices, weights[weight_index++]);
            query = query + scale_by_constant(scores, weights[weight_index++]);
            query = query + scale_by_constant(leaf_indices, weights[weight_index++]);
            weight_index += 2 * feature_count;
            query = query + compressed_bounds;
            return query;
        }

        cqLookupPublicTable compressed_row_dynamic_table(const forestCompressedRowLookupPublicTable& table,
                                                         const std::vector<Fr>& weights)
        {
            if (weights.size() != table.components.size()) {
                throw std::runtime_error("ForestRowFetchContext: compressed row table weight count mismatch");
            }

            cqLookupPublicTable out;
            out.key = table.key + ".dynamic." + std::to_string(compressed_row_table_count_++);
            out.domain_size = table.row_count;
            out.degree_bound = table.row_count == 0 ? std::optional<size_t>() : std::optional<size_t>(table.row_count - 1);
            out.commitment = Group1::zero();
            out.table_g2 = Group2::zero();
            out.lagrange_at_zero = table.lagrange_at_zero;
            out.vanishing_g2 = table.vanishing_g2;
            out.lagrange_commitments = table.lagrange_commitments;
            out.a0_quotient_commitments = table.a0_quotient_commitments;

            bool has_evals = true;
            bool has_quotients = true;
            for (const forestCompressedRowComponent& component : table.components) {
                has_evals = has_evals && component.evals != nullptr;
                has_quotients = has_quotients && component.quotient_commitments != nullptr;
            }

            std::shared_ptr<std::vector<Fr>> evals;
            if (has_evals) {
                evals = std::make_shared<std::vector<Fr>>(table.row_count, Fr(0));
            }
            std::shared_ptr<std::vector<Group1>> quotient_commitments;
            if (has_quotients) {
                quotient_commitments = std::make_shared<std::vector<Group1>>(table.row_count, Group1::zero());
            }

            for (size_t component_index = 0; component_index < table.components.size(); ++component_index) {
                const forestCompressedRowComponent& component = table.components[component_index];
                const Fr& weight = weights[component_index];
                out.commitment += weight * component.commitment;
                out.table_g2 += weight * component.table_g2;

                if (evals != nullptr) {
                    if (component.evals->size() != table.row_count) {
                        throw std::runtime_error("ForestRowFetchContext: compressed row eval size mismatch");
                    }
                    for (size_t row = 0; row < table.row_count; ++row) {
                        (*evals)[row] += weight * (*component.evals)[row];
                    }
                }
                if (quotient_commitments != nullptr) {
                    if (component.quotient_commitments->size() != table.row_count) {
                        throw std::runtime_error("ForestRowFetchContext: compressed row quotient size mismatch");
                    }
                    for (size_t row = 0; row < table.row_count; ++row) {
                        (*quotient_commitments)[row] += weight * (*component.quotient_commitments)[row];
                    }
                }
            }

            if (evals != nullptr) {
                auto index_by_value = std::make_shared<std::map<Fr, size_t>>();
                for (size_t row = 0; row < evals->size(); ++row) {
                    if (!index_by_value->emplace((*evals)[row], row).second) {
                        throw std::runtime_error(
                            "ForestRowFetchContext: compressed row table challenge produced duplicate row values");
                    }
                }
                out.table_evals = evals;
                out.index_by_value = index_by_value;
            }
            out.quotient_commitments = quotient_commitments;
            return out;
        }

        virCommitment compressed_column_weight_expression(size_t query_row_count,
                                                          size_t column_count,
                                                          const std::vector<Fr>& weights,
                                                          size_t weight_offset) const
        {
            std::vector<std::pair<size_t, Fr>> terms;
            terms.reserve(column_count);
            for (size_t column = 0; column < column_count; ++column) {
                terms.emplace_back(column, weights[weight_offset + column]);
            }
            return x_pow(subgroup_lagrange_linear_combination_expression(column_count, terms), query_row_count);
        }

        virCommitment bind_compressed_bounds_witness(const virCommitment& lower_bounds,
                                                     const virCommitment& upper_bounds,
                                                     size_t query_row_count,
                                                     size_t column_count,
                                                     const std::vector<Fr>& weights,
                                                     size_t lower_weight_offset,
                                                     size_t upper_weight_offset)
        {
            const std::string label =
                "forest_row_fetch.compressed_bounds." + std::to_string(compressed_bounds_count_++);
            if (runtime_.mode() == ProofMode::Prove) {
                const size_t matrix_domain = query_row_count * column_count;
                const std::vector<Fr> lower_evals =
                    runtime_.kzg().evaluations(lower_bounds, matrix_domain, kzgProtocol::EvaluationMode::PreferEval);
                const std::vector<Fr> upper_evals =
                    runtime_.kzg().evaluations(upper_bounds, matrix_domain, kzgProtocol::EvaluationMode::PreferEval);
                std::vector<Fr> compressed(query_row_count, Fr(0));
                for (size_t row = 0; row < query_row_count; ++row) {
                    for (size_t column = 0; column < column_count; ++column) {
                        const size_t index = matrix_index(row, column, column_count);
                        compressed[row] += weights[lower_weight_offset + column] * lower_evals[index];
                        compressed[row] += weights[upper_weight_offset + column] * upper_evals[index];
                    }
                }
                return runtime_.bind_virtual(WitnessCommitmentInput::prover(
                    label, poly_from_evals(compressed), query_row_count - 1));
            }
            return runtime_.bind_virtual(WitnessCommitmentInput::verifier(label, query_row_count - 1));
        }

        virCommitment add_compressed_bounds_row_sum_check(const virCommitment& lower_bounds,
                                                          const virCommitment& upper_bounds,
                                                          size_t query_row_count,
                                                          size_t column_count,
                                                          const std::vector<Fr>& weights,
                                                          size_t lower_weight_offset,
                                                          size_t upper_weight_offset)
        {
            const virCommitment lower_weights =
                compressed_column_weight_expression(query_row_count, column_count, weights, lower_weight_offset);
            const virCommitment upper_weights =
                compressed_column_weight_expression(query_row_count, column_count, weights, upper_weight_offset);
            const virCommitment weighted_bounds =
                lower_bounds * lower_weights + upper_bounds * upper_weights;
            const virCommitment compressed_bounds = bind_compressed_bounds_witness(lower_bounds,
                                                                                   upper_bounds,
                                                                                   query_row_count,
                                                                                   column_count,
                                                                                   weights,
                                                                                   lower_weight_offset,
                                                                                   upper_weight_offset);
            row_sum_.add_check(weighted_bounds, compressed_bounds, query_row_count, column_count);
            return compressed_bounds;
        }

        void add_compressed_leaf_lookup_check(const virCommitment& tree_indices,
                                              const virCommitment& lower_bounds,
                                              const virCommitment& upper_bounds,
                                              const virCommitment& scores,
                                              const virCommitment& leaf_indices,
                                              size_t query_row_count)
        {
            if (!compressed_lookup_available()) {
                throw std::runtime_error("ForestRowFetchContext: compressed row lookup table is not available");
            }

            const forestCompressedRowLookupPublicTable& table = *table_input_.compressed_row_lookup_table;
            bind_compressed_row_components(table);
            const Fr compression_challenge =
                runtime_.kzg().transcript().squeeze_nonzero_fr("forest_row_fetch.compressed_row.challenge");
            const std::vector<Fr> weights = compressed_row_weights(table, compression_challenge);
            const size_t lower_weight_offset = 3;
            const size_t upper_weight_offset = 3 + table_input_.feature_count;
            const virCommitment compressed_bounds =
                add_compressed_bounds_row_sum_check(lower_bounds,
                                                    upper_bounds,
                                                    query_row_count,
                                                    table_input_.feature_count,
                                                    weights,
                                                    lower_weight_offset,
                                                    upper_weight_offset);
            const virCommitment compressed_query = compressed_row_query_expression(tree_indices,
                                                                                   compressed_bounds,
                                                                                   scores,
                                                                                   leaf_indices,
                                                                                   weights);
            cqLookupPublicTable dynamic_table = compressed_row_dynamic_table(table, weights);
            auto context = std::make_unique<CqLookupContext>(runtime_, dynamic_table);
            context->lookup().add_check(compressed_query, query_row_count);
            compressed_row_lookup_contexts_.push_back(std::move(context));
            used_compressed_lookup_ = true;
        }

        virCommitment bind_shared_row_labels(const std::optional<tableLookupWitness>& witness,
                                             size_t query_row_count)
        {
            const std::string label =
                "forest_row_fetch.shared_row_labels." + std::to_string(shared_row_label_count_++);
            if (runtime_.mode() == ProofMode::Prove) {
                if (!witness.has_value()) {
                    throw std::runtime_error("ForestRowFetchContext: split lookup prover requires row-label witness");
                }
                return runtime_.bind_virtual(WitnessCommitmentInput::prover(
                    label, poly_from_evals(witness->row_labels), query_row_count - 1));
            }
            return runtime_.bind_virtual(WitnessCommitmentInput::verifier(label, query_row_count - 1));
        }

        void add_split_table_lookup(CqTableLookupContext& context,
                                    const virCommitment& query,
                                    size_t query_row_count,
                                    size_t column_count,
                                    const virCommitment& shared_row_labels,
                                    const std::optional<tableLookupWitness>& witness)
        {
            cqTableLookupCheckRecord record;
            if (runtime_.mode() == ProofMode::Prove) {
                if (!witness.has_value()) {
                    throw std::runtime_error("ForestRowFetchContext: split lookup prover requires row-label witness");
                }
                context.table_lookup().add_check(query, query_row_count, *witness, record);
            } else {
                context.table_lookup().add_check(query, query_row_count, record);
            }
            column_replication_.add_check(
                shared_row_labels, virCommitment(record.row_index_commitment), query_row_count, column_count);
        }

        void add_split_leaf_lookup_check(const virCommitment& tree_indices,
                                         const virCommitment& lower_bounds,
                                         const virCommitment& upper_bounds,
                                         const virCommitment& scores,
                                         const virCommitment& leaf_indices,
                                         size_t query_row_count,
                                         const std::optional<tableLookupWitness>& witness)
        {
            ensure_split_lookup_contexts();
            used_split_lookup_ = true;
            const virCommitment shared_row_labels = bind_shared_row_labels(witness, query_row_count);

            add_split_table_lookup(*split_lower_lookup_context_,
                                   lower_bounds,
                                   query_row_count,
                                   table_input_.feature_count,
                                   shared_row_labels,
                                   witness);
            add_split_table_lookup(*split_upper_lookup_context_,
                                   upper_bounds,
                                   query_row_count,
                                   table_input_.feature_count,
                                   shared_row_labels,
                                   witness);

            virCommitment scalar_query;
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT",
                "inference_profile.forest_row_fetch.split_scalar_query_expression",
                [&] {
                    scalar_query = cq_concatenated_table_lookup_query_expression(
                        {tree_indices, scores, leaf_indices},
                        query_row_count,
                        *table_input_.scalar_lookup_table,
                        "ForestRowFetchContext split scalar lookup");
                });
            add_split_table_lookup(*split_scalar_lookup_context_,
                                   scalar_query,
                                   query_row_count,
                                   table_input_.scalar_lookup_table->merged_column_count,
                                   shared_row_labels,
                                   witness);
        }

        bool finalize_split_lookup()
        {
            bool ok = true;
            ok = split_lower_lookup_context_->finalize() && ok;
            ok = split_upper_lookup_context_->finalize() && ok;
            ok = split_scalar_lookup_context_->finalize() && ok;
            return ok;
        }

        bool finalize_compressed_lookup()
        {
            bool ok = true;
            for (const auto& context : compressed_row_lookup_contexts_) {
                ok = context->finalize() && ok;
            }
            return ok;
        }

        ProofRuntime& runtime_;
        zeroProtocol& zero_;
        forestRowFetchPublicTable table_input_;
        columnReplicationProtocol& column_replication_;
        matrixRowSumProtocol& row_sum_;
        CqConcatenatedTableLookupContext lookup_context_;
        std::vector<std::unique_ptr<CqLookupContext>> compressed_row_lookup_contexts_;
        std::unique_ptr<CqTableLookupContext> split_lower_lookup_context_;
        std::unique_ptr<CqTableLookupContext> split_upper_lookup_context_;
        std::unique_ptr<CqTableLookupContext> split_scalar_lookup_context_;
        bool used_compressed_lookup_ = false;
        bool used_split_lookup_ = false;
        size_t shared_row_label_count_ = 0;
        size_t compressed_row_table_count_ = 0;
        size_t compressed_bounds_count_ = 0;
    };

    class ForestInferenceContext {
    public:
        ForestInferenceContext(ProofRuntime& runtime,
                               zeroProtocol& zero,
                               columnReplicationProtocol& column_replication,
                               matrixRowSumProtocol& row_sum,
                               boxCheckProtocol& box_check,
                               const forestInferencePublicTable& table_input)
            : runtime_(runtime)
            , table_input_(runtime_.mode() == ProofMode::Verify ? table_input.verifier_view() : table_input)
            , row_fetch_context_(runtime_, zero, column_replication, row_sum, table_input_.row_fetch_table)
            , box_check_(box_check)
        {
        }

        void add_check(const ProtocolPolynomialInput& tree_indices,
                       const ProtocolPolynomialInput& lifted_inputs,
                       const ProtocolPolynomialInput& lower_bounds,
                       const ProtocolPolynomialInput& upper_bounds,
                       const ProtocolPolynomialInput& scores,
                       size_t query_row_count)
        {
            virCommitment tree_commitment;
            virCommitment input_commitment;
            virCommitment lower_commitment;
            virCommitment upper_commitment;
            virCommitment score_commitment;
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_inference.bind_inputs", [&] {
                    tree_commitment = runtime_.bind_virtual(tree_indices);
                    input_commitment = runtime_.bind_virtual(lifted_inputs);
                    lower_commitment = runtime_.bind_virtual(lower_bounds);
                    upper_commitment = runtime_.bind_virtual(upper_bounds);
                    score_commitment = runtime_.bind_virtual(scores);
                });

            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_inference.row_fetch", [&] {
                    row_fetch_context_.add_bound_commitments(
                        tree_commitment, lower_commitment, upper_commitment, score_commitment, query_row_count);
                });
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_inference.box_check", [&] {
                    box_check_.add_check(input_commitment,
                                         lower_commitment,
                                         upper_commitment,
                                         query_row_count,
                                         table_input_.row_fetch_table.feature_count,
                                         table_input_.input_bit_bound);
                });
        }

        void add_check(const ProtocolPolynomialInput& tree_indices,
                       const ProtocolPolynomialInput& lifted_inputs,
                       const ProtocolPolynomialInput& lower_bounds,
                       const ProtocolPolynomialInput& upper_bounds,
                       const ProtocolPolynomialInput& scores,
                       const ProtocolPolynomialInput& leaf_indices,
                       size_t query_row_count)
        {
            virCommitment tree_commitment;
            virCommitment input_commitment;
            virCommitment lower_commitment;
            virCommitment upper_commitment;
            virCommitment score_commitment;
            virCommitment leaf_commitment;
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_inference.bind_inputs", [&] {
                    tree_commitment = runtime_.bind_virtual(tree_indices);
                    input_commitment = runtime_.bind_virtual(lifted_inputs);
                    lower_commitment = runtime_.bind_virtual(lower_bounds);
                    upper_commitment = runtime_.bind_virtual(upper_bounds);
                    score_commitment = runtime_.bind_virtual(scores);
                    leaf_commitment = runtime_.bind_virtual(leaf_indices);
                });

            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_inference.row_fetch", [&] {
                    row_fetch_context_.add_bound_commitments(tree_commitment,
                                                             lower_commitment,
                                                             upper_commitment,
                                                             score_commitment,
                                                             leaf_commitment,
                                                             query_row_count);
                });
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_inference.box_check", [&] {
                    box_check_.add_check(input_commitment,
                                         lower_commitment,
                                         upper_commitment,
                                         query_row_count,
                                         table_input_.row_fetch_table.feature_count,
                                         table_input_.input_bit_bound);
                });
        }

        bool finalize()
        {
            bool ok = false;
            proof_runtime_detail::profile_scope(
                "TERRAE_PROFILE_INFERENCE_CONTEXT", "inference_profile.forest_inference.finalize_row_fetch", [&] {
                    ok = row_fetch_context_.finalize();
                });
            return ok;
        }

    public:
        void add_bound_commitments(const virCommitment& tree_indices,
                                   const virCommitment& lifted_inputs,
                                   const virCommitment& lower_bounds,
                                   const virCommitment& upper_bounds,
                                   const virCommitment& scores,
                                   size_t query_row_count)
        {
            row_fetch_context_.add_bound_commitments(
                tree_indices, lower_bounds, upper_bounds, scores, query_row_count);
            box_check_.add_check(lifted_inputs,
                                 lower_bounds,
                                 upper_bounds,
                                 query_row_count,
                                 table_input_.row_fetch_table.feature_count,
                                 table_input_.input_bit_bound);
        }

        void add_bound_commitments(const virCommitment& tree_indices,
                                   const virCommitment& lifted_inputs,
                                   const virCommitment& lower_bounds,
                                   const virCommitment& upper_bounds,
                                   const virCommitment& scores,
                                   const virCommitment& leaf_indices,
                                   size_t query_row_count)
        {
            row_fetch_context_.add_bound_commitments(
                tree_indices, lower_bounds, upper_bounds, scores, leaf_indices, query_row_count);
            box_check_.add_check(lifted_inputs,
                                 lower_bounds,
                                 upper_bounds,
                                 query_row_count,
                                 table_input_.row_fetch_table.feature_count,
                                 table_input_.input_bit_bound);
        }

    private:
        ProofRuntime& runtime_;
        forestInferencePublicTable table_input_;
        ForestRowFetchContext row_fetch_context_;
        boxCheckProtocol& box_check_;
    };

    class ForestTreeLayoutContext {
    public:
        ForestTreeLayoutContext(ProofRuntime& runtime,
                                columnReplicationProtocol& column_replication,
                                const forestTreeLayoutPublicTable& table_input)
            : runtime_(runtime)
            , table_input_(runtime_.mode() == ProofMode::Verify ? table_input.verifier_view() : table_input)
            , column_replication_(column_replication)
        {
        }

        void add_check(const ProtocolPolynomialInput& tree_indices, size_t query_row_count)
        {
            add_bound_commitments(runtime_.bind_virtual(tree_indices), query_row_count);
        }

        void add_bound_commitments(const virCommitment& tree_indices, size_t query_row_count)
        {
            const size_t tree_count = table_input_.tree_count();
            if (query_row_count == 0 || query_row_count % tree_count != 0) {
                throw std::runtime_error(
                    "ForestTreeLayoutContext: query_row_count must be a positive multiple of iteration_count * class_count");
            }

            const size_t batch_size = query_row_count / tree_count;
            if (!isPowerOfTwo(batch_size)) {
                throw std::runtime_error("ForestTreeLayoutContext: inferred batch size must be a non-zero power of two");
            }

            virCommitment base_layout;
            if (table_input_.base_layout_commitment.has_value()) {
                if (runtime_.mode() == ProofMode::Prove) {
                    base_layout = runtime_.bind_virtual(
                        PublicCommitmentInput::prover(table_input_.key_prefix + ".base_layout",
                                                      forest_tree_layout_polynomial(tree_count),
                                                      tree_count - 1));
                } else {
                    base_layout = runtime_.bind_virtual(
                        PublicCommitmentInput::verifier(table_input_.key_prefix + ".base_layout",
                                                        *table_input_.base_layout_commitment,
                                                        tree_count - 1));
                }
            } else {
                base_layout = forest_tree_layout_expression(tree_count);
            }
            column_replication_.add_check(base_layout, tree_indices, tree_count, batch_size);
        }

        bool finalize()
        {
            return true;
        }

    private:
        ProofRuntime& runtime_;
        forestTreeLayoutPublicTable table_input_;
        columnReplicationProtocol& column_replication_;
    };

    class ForestPredictionContext {
    public:
        ForestPredictionContext(ProofRuntime& runtime,
                                matrixColumnSumProtocol& column_sum,
                                cubeMiddleArgmaxProtocol& cube_middle_argmax,
                                const forestPredictionPublicTable& table_input)
            : runtime_(runtime)
            , table_input_(runtime_.mode() == ProofMode::Verify ? table_input.verifier_view() : table_input)
            , column_sum_(column_sum)
            , cube_middle_argmax_(cube_middle_argmax)
        {
        }

        void add_check(const ProtocolPolynomialInput& scores,
                       const ProtocolPolynomialInput& class_scores,
                       const ProtocolPolynomialInput& predicted_scores,
                       const ProtocolPolynomialInput& class_selector,
                       size_t query_row_count)
        {
            const virCommitment score_commitment = runtime_.bind_virtual(scores);
            add_bound_commitments(score_commitment,
                                  runtime_.bind_virtual(class_scores),
                                  runtime_.bind_virtual(predicted_scores),
                                  runtime_.bind_virtual(class_selector),
                                  query_row_count);
        }

        bool finalize()
        {
            return true;
        }

        void add_bound_commitments(const virCommitment& scores,
                                   const virCommitment& class_scores,
                                   const virCommitment& predicted_scores,
                                   const virCommitment& class_selector,
                                   size_t query_row_count)
        {
            const size_t trees_per_batch = table_input_.iteration_count * table_input_.class_count;
            if (query_row_count == 0 || query_row_count % trees_per_batch != 0) {
                throw std::runtime_error(
                    "ForestPredictionContext: query_row_count must be a positive multiple of iteration_count * class_count");
            }
            const size_t batch_size = query_row_count / trees_per_batch;
            if (!isPowerOfTwo(batch_size)) {
                throw std::runtime_error("ForestPredictionContext: inferred batch size must be a non-zero power of two");
            }

            // Scores are ordered by (iteration, class, sample):
            //   score_index = iteration * (K * b) + class * b + sample
            //
            // So after viewing the fetched score vector as a T x (K * b) matrix, a
            // column sum over the iteration axis yields the final class scores in the
            // category-major layout:
            //   [class 0 scores for all samples | class 1 scores | ... ].
            column_sum_.add_check(scores,
                                  class_scores,
                                  table_input_.iteration_count,
                                  table_input_.class_count * batch_size);

            // Reinterpret class_scores as a 1 x K x b cube, so cube-middle argmax
            // proves the maximum across the class axis for each sample slot.
            cube_middle_argmax_.add_check(class_scores,
                                          predicted_scores,
                                          class_selector,
                                          1,
                                          table_input_.class_count,
                                          batch_size,
                                          table_input_.score_bit_bound);
        }

    private:
        ProofRuntime& runtime_;
        forestPredictionPublicTable table_input_;
        matrixColumnSumProtocol& column_sum_;
        cubeMiddleArgmaxProtocol& cube_middle_argmax_;
    };

    class ForestClassificationContext {
    public:
        ForestClassificationContext(ProofRuntime& runtime,
                                    zeroProtocol& zero,
                                    boxCheckProtocol& box_check,
                                    columnReplicationProtocol& column_replication,
                                    matrixRowSumProtocol& row_sum,
                                    matrixColumnSumProtocol& column_sum,
                                    cubeMiddleArgmaxProtocol& cube_middle_argmax,
                                    const forestClassificationPublicTable& table_input)
            : runtime_(runtime)
            , table_input_(runtime_.mode() == ProofMode::Verify ? table_input.verifier_view() : table_input)
            , layout_context_(runtime_, column_replication, table_input_.layout_table)
            , inference_context_(runtime_, zero, column_replication, row_sum, box_check, table_input_.inference_table)
            , prediction_context_(runtime_, column_sum, cube_middle_argmax, table_input_.prediction_table)
        {
        }

        void add_check(const ProtocolPolynomialInput& tree_indices,
                       const ProtocolPolynomialInput& lifted_inputs,
                       const ProtocolPolynomialInput& lower_bounds,
                       const ProtocolPolynomialInput& upper_bounds,
                       const ProtocolPolynomialInput& scores,
                       const ProtocolPolynomialInput& class_scores,
                       const ProtocolPolynomialInput& predicted_scores,
                       const ProtocolPolynomialInput& class_selector,
                       size_t query_row_count)
        {
            const virCommitment tree_commitment = runtime_.bind_virtual(tree_indices);
            const virCommitment input_commitment = runtime_.bind_virtual(lifted_inputs);
            const virCommitment lower_commitment = runtime_.bind_virtual(lower_bounds);
            const virCommitment upper_commitment = runtime_.bind_virtual(upper_bounds);
            const virCommitment score_commitment = runtime_.bind_virtual(scores);
            const virCommitment class_score_commitment = runtime_.bind_virtual(class_scores);
            const virCommitment predicted_score_commitment = runtime_.bind_virtual(predicted_scores);
            const virCommitment class_selector_commitment = runtime_.bind_virtual(class_selector);

            add_bound_commitments(tree_commitment,
                                  input_commitment,
                                  lower_commitment,
                                  upper_commitment,
                                  score_commitment,
                                  class_score_commitment,
                                  predicted_score_commitment,
                                  class_selector_commitment,
                                  query_row_count);
        }

        void add_check(const ProtocolPolynomialInput& tree_indices,
                       const ProtocolPolynomialInput& lifted_inputs,
                       const ProtocolPolynomialInput& lower_bounds,
                       const ProtocolPolynomialInput& upper_bounds,
                       const ProtocolPolynomialInput& scores,
                       const ProtocolPolynomialInput& leaf_indices,
                       const ProtocolPolynomialInput& class_scores,
                       const ProtocolPolynomialInput& predicted_scores,
                       const ProtocolPolynomialInput& class_selector,
                       size_t query_row_count)
        {
            const virCommitment tree_commitment = runtime_.bind_virtual(tree_indices);
            const virCommitment input_commitment = runtime_.bind_virtual(lifted_inputs);
            const virCommitment lower_commitment = runtime_.bind_virtual(lower_bounds);
            const virCommitment upper_commitment = runtime_.bind_virtual(upper_bounds);
            const virCommitment score_commitment = runtime_.bind_virtual(scores);
            const virCommitment leaf_commitment = runtime_.bind_virtual(leaf_indices);
            const virCommitment class_score_commitment = runtime_.bind_virtual(class_scores);
            const virCommitment predicted_score_commitment = runtime_.bind_virtual(predicted_scores);
            const virCommitment class_selector_commitment = runtime_.bind_virtual(class_selector);

            add_bound_commitments(tree_commitment,
                                  input_commitment,
                                  lower_commitment,
                                  upper_commitment,
                                  score_commitment,
                                  leaf_commitment,
                                  class_score_commitment,
                                  predicted_score_commitment,
                                  class_selector_commitment,
                                  query_row_count);
        }

        bool finalize()
        {
            return layout_context_.finalize() && inference_context_.finalize() && prediction_context_.finalize();
        }

        void add_bound_commitments(const virCommitment& tree_indices,
                                   const virCommitment& lifted_inputs,
                                   const virCommitment& lower_bounds,
                                   const virCommitment& upper_bounds,
                                   const virCommitment& scores,
                                   const virCommitment& class_scores,
                                   const virCommitment& predicted_scores,
                                   const virCommitment& class_selector,
                                   size_t query_row_count)
        {
            layout_context_.add_bound_commitments(tree_indices, query_row_count);
            inference_context_.add_bound_commitments(
                tree_indices, lifted_inputs, lower_bounds, upper_bounds, scores, query_row_count);
            prediction_context_.add_bound_commitments(
                scores, class_scores, predicted_scores, class_selector, query_row_count);
        }

        void add_bound_commitments(const virCommitment& tree_indices,
                                   const virCommitment& lifted_inputs,
                                   const virCommitment& lower_bounds,
                                   const virCommitment& upper_bounds,
                                   const virCommitment& scores,
                                   const virCommitment& leaf_indices,
                                   const virCommitment& class_scores,
                                   const virCommitment& predicted_scores,
                                   const virCommitment& class_selector,
                                   size_t query_row_count)
        {
            layout_context_.add_bound_commitments(tree_indices, query_row_count);
            inference_context_.add_bound_commitments(
                tree_indices, lifted_inputs, lower_bounds, upper_bounds, scores, leaf_indices, query_row_count);
            prediction_context_.add_bound_commitments(
                scores, class_scores, predicted_scores, class_selector, query_row_count);
        }

    private:
        ProofRuntime& runtime_;
        forestClassificationPublicTable table_input_;
        ForestTreeLayoutContext layout_context_;
        ForestInferenceContext inference_context_;
        ForestPredictionContext prediction_context_;
    };

    explicit ProofProgram(ProofRuntime& runtime)
        : runtime_(runtime)
        , table_lookup_(lookup_)
        , concatenated_table_lookup_()
        , hist_agg_(table_lookup_)
        , cube_sum_(table_lookup_)
    {
        zero_.setup(runtime_.kzg());
        sum_.setup(runtime_.kzg(), zero_);
        zero_.set_soundness_counter(&soundness_counter_);
        sum_.set_soundness_counter(&soundness_counter_);
        nonzero_.setup(runtime_.kzg(), zero_);
        lookup_.setup(runtime_.kzg(), zero_, sum_);
        lookup_.set_soundness_counter(&soundness_counter_);
        less_.setup(runtime_.kzg(), lookup_, zero_);
        signed_range_.setup(runtime_.kzg(), less_);
        div_.setup(runtime_.kzg(), zero_, less_);
        rational_.setup(runtime_.kzg(), zero_);
        fixed_point_division_.setup(runtime_.kzg(), zero_, less_, signed_range_, nonzero_);
        table_lookup_.setup(runtime_.kzg(), zero_, sum_);
        table_lookup_.set_soundness_counter(&soundness_counter_);
        concatenated_table_lookup_.setup(runtime_.kzg(), table_lookup_);
        hist_agg_.setup(runtime_.kzg(), zero_, sum_);
        hist_agg_.set_soundness_counter(&soundness_counter_);
        exp_.setup(runtime_.kzg(), table_lookup_);
        column_replication_.setup(runtime_.kzg(), zero_);
        column_prefix_.setup(runtime_.kzg(), zero_, column_replication_);
        row_prefix_.setup(runtime_.kzg(), zero_);
        column_sum_.setup(runtime_.kzg(), zero_);
        row_sum_.setup(runtime_.kzg(), zero_);
        selector_.setup(runtime_.kzg(), zero_, row_sum_);
        shuffle_.setup(runtime_.kzg(), lookup_);
        zkboost_histogram_ram_.setup(runtime_.kzg(), zero_, less_, shuffle_);
        index_routing_.setup(runtime_.kzg(), zero_, column_replication_, concatenated_table_lookup_);
        root_selector_consistency_.setup(runtime_.kzg(), zero_, hist_agg_, index_routing_);
        root_selector_consistency_.set_soundness_counter(&soundness_counter_);
        node_role_consistency_.setup(runtime_.kzg(), zero_, hist_agg_);
        box_check_.setup(runtime_.kzg(), less_);
        forest_depth_consistency_.setup(runtime_.kzg(), zero_, index_routing_);
        parent_child_box_inheritance_.setup(runtime_.kzg(), zero_, selector_, column_replication_, index_routing_, less_);
        parent_child_tree_consistency_.setup(runtime_.kzg(), zero_, index_routing_);
        root_box_range_.setup(runtime_.kzg(), zero_, index_routing_);
        tree_histogram_consistency_.setup(runtime_.kzg(), zero_, index_routing_);
        forest_histogram_.setup(hist_agg_, tree_histogram_consistency_);
        forest_histogram_bucket_address_.setup(zero_, column_replication_, less_);
        forest_training_inference_.setup(runtime_.kzg(), less_, box_check_, concatenated_table_lookup_);
        forest_histogram_refresh_.setup(
            forest_training_inference_, forest_histogram_bucket_address_, column_replication_, forest_histogram_);
        forest_split_statistics_.setup(runtime_.kzg(), zero_, row_prefix_, row_sum_, column_replication_);
        forest_split_gain_.setup(zero_, less_, signed_range_, nonzero_);
        forest_quantized_split_gain_.setup(zero_, fixed_point_division_);
        forest_leaf_value_.setup(zero_, less_, fixed_point_division_);
        masked_argmax_.setup(runtime_.kzg(), zero_, selector_, row_sum_, column_replication_, less_);
        forest_split_selection_.setup(masked_argmax_, row_sum_, zero_, less_, nonzero_);
        forest_training_proof_.setup(forest_histogram_refresh_,
                                     forest_split_statistics_,
                                     forest_split_gain_,
                                     forest_quantized_split_gain_,
                                     forest_split_selection_,
                                     forest_leaf_value_,
                                     forest_boosting_state_,
                                     row_sum_,
                                     zero_,
                                     less_);
        forest_split_update_.setup(forest_structure_, hist_agg_, row_sum_, zero_);
        forest_structure_.setup(runtime_.kzg(),
                                zero_,
                                column_replication_,
                                root_box_range_,
                                root_selector_consistency_,
                                node_role_consistency_,
                                forest_depth_consistency_,
                                parent_child_box_inheritance_,
                                parent_child_tree_consistency_);
        argmax_.setup(runtime_.kzg(), selector_, row_sum_, column_replication_, less_);
        cube_sum_.setup(runtime_.kzg(), zero_);
        cube_middle_replication_.setup(runtime_.kzg(), zero_, table_lookup_);
        cube_middle_argmax_.setup(runtime_.kzg(), zero_, cube_sum_, cube_middle_replication_, less_);
        cube_middle_softmax_.setup(
            runtime_.kzg(), zero_, signed_range_, cube_middle_argmax_, exp_, cube_sum_, cube_middle_replication_);
        forest_boosting_state_.setup(
            zero_, column_prefix_, column_sum_, cube_middle_argmax_, cube_middle_replication_, cube_sum_, exp_, fixed_point_division_, signed_range_);
    }

    ProofRuntime& runtime()
    {
        return runtime_;
    }

    terraeSoundnessCounter& soundness_counter()
    {
        return soundness_counter_;
    }

    template <class Input>
    kzgCommitment bind(const Input& input)
    {
        return runtime_.bind(input);
    }

    template <class Input>
    virCommitment bind_virtual(const Input& input)
    {
        return runtime_.bind_virtual(input);
    }

    zeroProtocol& zero()
    {
        return zero_;
    }

    nonzeroCheckProtocol& nonzero_check()
    {
        return nonzero_;
    }

    sumProtocol& sum()
    {
        return sum_;
    }

    lookupProtocol& lookup()
    {
        return lookup_;
    }

    lessCheckProtocol& less_check()
    {
        return less_;
    }

    divCheckProtocol& div_check()
    {
        return div_;
    }

    rationalCheckProtocol& rational_check()
    {
        return rational_;
    }

    fixedPointDivisionProtocol& fixed_point_division()
    {
        return fixed_point_division_;
    }

    signedRangeCheckProtocol& signed_range_check()
    {
        return signed_range_;
    }

    selectorCheckProtocol& selector_check()
    {
        return selector_;
    }

    shuffleProtocol& shuffle()
    {
        return shuffle_;
    }

    rootSelectorConsistencyProtocol& root_selector_consistency()
    {
        return root_selector_consistency_;
    }

    nodeRoleConsistencyProtocol& node_role_consistency()
    {
        return node_role_consistency_;
    }

    rootBoxRangeProtocol& root_box_range()
    {
        return root_box_range_;
    }

    boxCheckProtocol& box_check()
    {
        return box_check_;
    }

    indexRoutingProtocol& index_routing()
    {
        return index_routing_;
    }

    argmaxProtocol& argmax()
    {
        return argmax_;
    }

    parentChildBoxInheritanceProtocol& parent_child_box_inheritance()
    {
        return parent_child_box_inheritance_;
    }

    parentChildTreeConsistencyProtocol& parent_child_tree_consistency()
    {
        return parent_child_tree_consistency_;
    }

    forestStructureProtocol& forest_structure()
    {
        return forest_structure_;
    }

    treeHistogramConsistencyProtocol& tree_histogram_consistency()
    {
        return tree_histogram_consistency_;
    }

    forestHistogramProtocol& forest_histogram()
    {
        return forest_histogram_;
    }

    forestHistogramBucketAddressProtocol& forest_histogram_bucket_address()
    {
        return forest_histogram_bucket_address_;
    }

    forestTrainingInferenceProtocol& forest_training_inference()
    {
        return forest_training_inference_;
    }

    forestHistogramRefreshProtocol& forest_histogram_refresh()
    {
        return forest_histogram_refresh_;
    }

    forestSplitStatisticsProtocol& forest_split_statistics()
    {
        return forest_split_statistics_;
    }

    forestSplitGainProtocol& forest_split_gain()
    {
        return forest_split_gain_;
    }

    forestQuantizedSplitGainProtocol& forest_quantized_split_gain()
    {
        return forest_quantized_split_gain_;
    }

    forestSplitSelectionProtocol& forest_split_selection()
    {
        return forest_split_selection_;
    }

    forestTrainingProofProtocol& forest_training_proof()
    {
        return forest_training_proof_;
    }

    forestSplitUpdateProtocol& forest_split_update()
    {
        return forest_split_update_;
    }

    forestLeafValueProtocol& forest_leaf_value()
    {
        return forest_leaf_value_;
    }

    forestBoostingStateProtocol& forest_boosting_state()
    {
        return forest_boosting_state_;
    }

    expCheckProtocol& exp_check()
    {
        return exp_;
    }

    tableLookupProtocol& table_lookup()
    {
        return table_lookup_;
    }

    concatenatedTableLookupProtocol& concatenated_table_lookup()
    {
        return concatenated_table_lookup_;
    }

    histAggProtocol& hist_agg()
    {
        return hist_agg_;
    }

    zkboostHistogramRamProtocol& zkboost_histogram_ram()
    {
        return zkboost_histogram_ram_;
    }

    matrixColumnPrefixSumProtocol& column_prefix()
    {
        return column_prefix_;
    }

    columnReplicationProtocol& column_replication()
    {
        return column_replication_;
    }

    matrixRowPrefixSumProtocol& row_prefix()
    {
        return row_prefix_;
    }

    matrixColumnSumProtocol& column_sum()
    {
        return column_sum_;
    }

    matrixRowSumProtocol& row_sum()
    {
        return row_sum_;
    }

    cubeSumProtocol& cube_sum()
    {
        return cube_sum_;
    }

    cubeMiddleArgmaxProtocol& cube_middle_argmax()
    {
        return cube_middle_argmax_;
    }

    cubeMiddleSoftmaxProtocol& cube_middle_softmax()
    {
        return cube_middle_softmax_;
    }

    CqLookupContext& create_cq_lookup_context(const PublicPolynomialInput& table_input, size_t table_domain_size)
    {
        auto context = std::make_unique<CqLookupContext>(runtime_, table_input, table_domain_size);
        CqLookupContext& ref = *context;
        cq_lookup_contexts_.push_back(std::move(context));
        return ref;
    }

    CqLookupContext& create_cq_lookup_context(const cqLookupPublicTable& table_input)
    {
        auto context = std::make_unique<CqLookupContext>(runtime_, table_input);
        CqLookupContext& ref = *context;
        cq_lookup_contexts_.push_back(std::move(context));
        return ref;
    }

    CqTableLookupContext& create_cq_table_lookup_context(const PublicPolynomialInput& table_input,
                                                         size_t table_row_count,
                                                         size_t column_count)
    {
        auto context =
            std::make_unique<CqTableLookupContext>(runtime_, zero_, table_input, table_row_count, column_count);
        CqTableLookupContext& ref = *context;
        cq_table_lookup_contexts_.push_back(std::move(context));
        return ref;
    }

    CqTableLookupContext& create_cq_table_lookup_context(const cqTableLookupPublicTable& table_input)
    {
        auto context = std::make_unique<CqTableLookupContext>(runtime_, zero_, table_input);
        CqTableLookupContext& ref = *context;
        cq_table_lookup_contexts_.push_back(std::move(context));
        return ref;
    }

    CqConcatenatedTableLookupContext& create_cq_concatenated_table_lookup_context(
        const cqConcatenatedTableLookupPublicTable& table_input)
    {
        auto context = std::make_unique<CqConcatenatedTableLookupContext>(runtime_, zero_, table_input);
        CqConcatenatedTableLookupContext& ref = *context;
        cq_concatenated_table_lookup_contexts_.push_back(std::move(context));
        return ref;
    }

    ForestRowFetchContext& create_forest_row_fetch_context(const forestRowFetchPublicTable& table_input)
    {
        auto context = std::make_unique<ForestRowFetchContext>(runtime_, zero_, column_replication_, row_sum_, table_input);
        ForestRowFetchContext& ref = *context;
        forest_row_fetch_contexts_.push_back(std::move(context));
        return ref;
    }

    ForestInferenceContext& create_forest_inference_context(const forestInferencePublicTable& table_input)
    {
        auto context =
            std::make_unique<ForestInferenceContext>(runtime_, zero_, column_replication_, row_sum_, box_check_, table_input);
        ForestInferenceContext& ref = *context;
        forest_inference_contexts_.push_back(std::move(context));
        return ref;
    }

    ForestTreeLayoutContext& create_forest_tree_layout_context(const forestTreeLayoutPublicTable& table_input)
    {
        auto context = std::make_unique<ForestTreeLayoutContext>(runtime_, column_replication_, table_input);
        ForestTreeLayoutContext& ref = *context;
        forest_tree_layout_contexts_.push_back(std::move(context));
        return ref;
    }

    ForestPredictionContext& create_forest_prediction_context(const forestPredictionPublicTable& table_input)
    {
        auto context =
            std::make_unique<ForestPredictionContext>(runtime_, column_sum_, cube_middle_argmax_, table_input);
        ForestPredictionContext& ref = *context;
        forest_prediction_contexts_.push_back(std::move(context));
        return ref;
    }

    ForestClassificationContext& create_forest_classification_context(const forestClassificationPublicTable& table_input)
    {
        auto context = std::make_unique<ForestClassificationContext>(
            runtime_, zero_, box_check_, column_replication_, row_sum_, column_sum_, cube_middle_argmax_, table_input);
        ForestClassificationContext& ref = *context;
        forest_classification_contexts_.push_back(std::move(context));
        return ref;
    }

    bool finalize()
    {
        bool ok = true;
        const char* first_failure = nullptr;
        const bool profile_finalize = []() {
            const char* flag = terrae_getenv("TERRAE_PROFILE_FINALIZE");
            return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
        }();
        auto record_finalize = [&](const char* name, bool stage_ok) {
            if (!stage_ok && first_failure == nullptr) {
                first_failure = name;
            }
            ok = stage_ok && ok;
        };
        auto run_finalize = [&](const char* name, auto&& fn) {
            const bool structured_enabled = structured_protocol_profile::enabled();
            if (!profile_finalize && !structured_enabled) {
                const bool stage_ok = fn();
                record_finalize(name, stage_ok);
                return;
            }
            const auto start = std::chrono::steady_clock::now();
            bool stage_ok = false;
            try {
                stage_ok = fn();
            } catch (const std::exception& ex) {
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                if (profile_finalize) {
                    std::cerr << "finalize_timing " << name << " " << elapsed.count() << " throw\n";
                }
                structured_protocol_profile::append_row(
                    "protocol_finalize", name, "", elapsed.count(), std::string("throw:") + ex.what());
                throw;
            } catch (...) {
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                if (profile_finalize) {
                    std::cerr << "finalize_timing " << name << " " << elapsed.count() << " throw\n";
                }
                structured_protocol_profile::append_row("protocol_finalize", name, "", elapsed.count(), "throw");
                throw;
            }
            const auto end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = end - start;
            if (profile_finalize) {
                std::cerr << "finalize_timing " << name << " " << elapsed.count() << " "
                          << (stage_ok ? "ok" : "fail") << "\n";
            }
            structured_protocol_profile::append_row(
                "protocol_finalize", name, "", elapsed.count(), stage_ok ? "ok" : "fail");
            record_finalize(name, stage_ok);
        };

        bool needs_shared_opening_stage = false;

        for (auto& context : cq_table_lookup_contexts_) {
            run_finalize("cq_table_lookup_context.prepare_opening", [&] { return context->finalize_prepare_opening(); });
        }
        for (auto& context : cq_lookup_contexts_) {
            run_finalize("cq_lookup_context.prepare_opening", [&] { return context->finalize_prepare_opening(); });
        }
        for (auto& context : cq_concatenated_table_lookup_contexts_) {
            run_finalize("cq_concatenated_table_lookup_context.prepare_opening",
                         [&] { return context->finalize_prepare_opening(); });
        }
        for (auto& context : forest_row_fetch_contexts_) {
            run_finalize("forest_row_fetch_context", [&] { return context->finalize(); });
        }
        for (auto& context : forest_inference_contexts_) {
            run_finalize("forest_inference_context", [&] { return context->finalize(); });
        }
        for (auto& context : forest_tree_layout_contexts_) {
            run_finalize("forest_tree_layout_context", [&] { return context->finalize(); });
        }
        for (auto& context : forest_prediction_contexts_) {
            run_finalize("forest_prediction_context", [&] { return context->finalize(); });
        }
        for (auto& context : forest_classification_contexts_) {
            run_finalize("forest_classification_context", [&] { return context->finalize(); });
        }

        run_finalize("forest_training_proof", [&] { return forest_training_proof_.finalize(); });
        run_finalize("forest_histogram_refresh", [&] { return forest_histogram_refresh_.finalize(); });
        run_finalize("forest_histogram", [&] { return forest_histogram_.finalize(); });
        run_finalize("forest_histogram_bucket_address", [&] { return forest_histogram_bucket_address_.finalize(); });
        run_finalize("forest_training_inference", [&] { return forest_training_inference_.finalize(); });
        run_finalize("forest_split_statistics", [&] { return forest_split_statistics_.finalize(); });
        run_finalize("forest_split_gain", [&] { return forest_split_gain_.finalize(); });
        run_finalize("forest_quantized_split_gain", [&] { return forest_quantized_split_gain_.finalize(); });
        run_finalize("forest_split_selection", [&] { return forest_split_selection_.finalize(); });
        run_finalize("forest_split_update", [&] { return forest_split_update_.finalize(); });
        run_finalize("forest_leaf_value", [&] { return forest_leaf_value_.finalize(); });
        run_finalize("forest_boosting_state", [&] { return forest_boosting_state_.finalize(); });
        run_finalize("forest_structure", [&] { return forest_structure_.finalize(); });
        run_finalize("tree_histogram_consistency", [&] { return tree_histogram_consistency_.finalize(); });
        run_finalize("root_selector_consistency", [&] { return root_selector_consistency_.finalize(); });
        run_finalize("node_role_consistency", [&] { return node_role_consistency_.finalize(); });
        run_finalize("forest_depth_consistency", [&] { return forest_depth_consistency_.finalize(); });
        run_finalize("root_box_range", [&] { return root_box_range_.finalize(); });
        run_finalize("parent_child_box_inheritance", [&] { return parent_child_box_inheritance_.finalize(); });
        run_finalize("parent_child_tree_consistency", [&] { return parent_child_tree_consistency_.finalize(); });
        run_finalize("masked_argmax", [&] { return masked_argmax_.finalize(); });
        run_finalize("argmax", [&] { return argmax_.finalize(); });
        run_finalize("cube_middle_argmax", [&] { return cube_middle_argmax_.finalize(); });
        run_finalize("cube_middle_softmax", [&] { return cube_middle_softmax_.finalize(); });
        run_finalize("cube_middle_replication", [&] { return cube_middle_replication_.finalize(); });
        run_finalize("exp_check", [&] { return exp_.finalize(); });
        run_finalize("div_check", [&] { return div_.finalize(); });
        run_finalize("rational_check", [&] { return rational_.finalize(); });
        run_finalize("fixed_point_division", [&] { return fixed_point_division_.finalize(); });
        run_finalize("box_check", [&] { return box_check_.finalize(); });
        run_finalize("signed_range_check", [&] { return signed_range_.finalize(); });
        run_finalize("selector_check", [&] { return selector_.finalize(); });
        run_finalize("zkboost_histogram_ram", [&] { return zkboost_histogram_ram_.finalize(); });
        run_finalize("hist_agg", [&] { return hist_agg_.finalize(); });
        run_finalize("index_routing", [&] { return index_routing_.finalize(); });
        run_finalize("concatenated_table_lookup", [&] { return concatenated_table_lookup_.finalize(); });
        run_finalize("column_replication", [&] { return column_replication_.finalize(); });
        run_finalize("cube_sum", [&] { return cube_sum_.finalize(); });
        run_finalize("table_lookup", [&] { return table_lookup_.finalize(); });
        run_finalize("less_check", [&] { return less_.finalize(); });
        run_finalize("shuffle", [&] { return shuffle_.finalize(); });
        run_finalize("lookup", [&] { return lookup_.finalize(); });
        run_finalize("column_prefix", [&] { return column_prefix_.finalize(); });
        run_finalize("row_prefix", [&] { return row_prefix_.finalize(); });
        run_finalize("column_sum", [&] { return column_sum_.finalize(); });
        run_finalize("row_sum", [&] { return row_sum_.finalize(); });
        run_finalize("sum_check", [&] { return sum_.finalize(); });
        run_finalize("nonzero_check", [&] { return nonzero_.finalize(); });
        run_finalize("zero_check.prepare_opening", [&] { return zero_.prepare_opening(); });
        needs_shared_opening_stage = zero_.needs_opening();
        for (const auto& context : cq_table_lookup_contexts_) {
            needs_shared_opening_stage = context->needs_opening() || needs_shared_opening_stage;
        }
        for (const auto& context : cq_lookup_contexts_) {
            needs_shared_opening_stage = context->needs_opening() || needs_shared_opening_stage;
        }
        for (const auto& context : cq_concatenated_table_lookup_contexts_) {
            needs_shared_opening_stage = context->needs_opening() || needs_shared_opening_stage;
        }
        if (needs_shared_opening_stage) {
            const Fr shared_opening = runtime_.kzg().shared_opening_challenge("protocol.opening.stage0", true);
            for (auto& context : cq_table_lookup_contexts_) {
                run_finalize("cq_table_lookup_context.complete_opening",
                             [&] { return context->finalize_complete_opening(shared_opening); });
            }
            for (auto& context : cq_lookup_contexts_) {
                run_finalize("cq_lookup_context.complete_opening",
                             [&] { return context->finalize_complete_opening(shared_opening); });
            }
            for (auto& context : cq_concatenated_table_lookup_contexts_) {
                run_finalize("cq_concatenated_table_lookup_context.complete_opening",
                             [&] { return context->finalize_complete_opening(shared_opening); });
            }
            run_finalize("zero_check.complete_opening", [&] { return zero_.complete_opening(shared_opening); });
        } else {
            run_finalize("zero_check.complete_opening", [&] { return zero_.complete_opening(Fr(0)); });
        }
        soundness_counter_.emit_metrics(runtime_.kzg().commitment_count());
        run_finalize("kzg", [&] { return runtime_.finalize_kzg(); });
        return ok;
    }

    bool prepare_verifier_preprocessing()
    {
        if (runtime_.mode() != ProofMode::Verify) {
            throw std::runtime_error("ProofProgram::prepare_verifier_preprocessing is verifier-only");
        }
        if (verification_prepared_) {
            throw std::runtime_error("ProofProgram verifier preprocessing already prepared");
        }

        bool ok = true;
        const bool profile_finalize = []() {
            const char* flag = terrae_getenv("TERRAE_PROFILE_FINALIZE");
            return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
        }();
        auto run_finalize = [&](const char* name, auto&& fn) {
            const bool structured_enabled = structured_protocol_profile::enabled();
            if (!profile_finalize && !structured_enabled) {
                ok = fn() && ok;
                return;
            }
            const auto start = std::chrono::steady_clock::now();
            bool stage_ok = false;
            try {
                stage_ok = fn();
            } catch (const std::exception& ex) {
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                if (profile_finalize) {
                    std::cerr << "finalize_timing " << name << " " << elapsed.count() << " throw\n";
                }
                structured_protocol_profile::append_row(
                    "protocol_finalize", name, "", elapsed.count(), std::string("throw:") + ex.what());
                throw;
            } catch (...) {
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                if (profile_finalize) {
                    std::cerr << "finalize_timing " << name << " " << elapsed.count() << " throw\n";
                }
                structured_protocol_profile::append_row("protocol_finalize", name, "", elapsed.count(), "throw");
                throw;
            }
            const auto end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = end - start;
            if (profile_finalize) {
                std::cerr << "finalize_timing " << name << " " << elapsed.count() << " "
                          << (stage_ok ? "ok" : "fail") << "\n";
            }
            structured_protocol_profile::append_row(
                "protocol_finalize", name, "", elapsed.count(), stage_ok ? "ok" : "fail");
            ok = stage_ok && ok;
        };

        prepared_needs_shared_opening_stage_ = false;

        for (auto& context : cq_table_lookup_contexts_) {
            run_finalize("cq_table_lookup_context.prepare_opening", [&] { return context->finalize_prepare_opening(); });
        }
        for (auto& context : cq_lookup_contexts_) {
            run_finalize("cq_lookup_context.prepare_opening", [&] { return context->finalize_prepare_opening(); });
        }
        for (auto& context : cq_concatenated_table_lookup_contexts_) {
            run_finalize("cq_concatenated_table_lookup_context.prepare_opening",
                         [&] { return context->finalize_prepare_opening(); });
        }
        for (auto& context : forest_row_fetch_contexts_) {
            run_finalize("forest_row_fetch_context", [&] { return context->finalize(); });
        }
        for (auto& context : forest_inference_contexts_) {
            run_finalize("forest_inference_context", [&] { return context->finalize(); });
        }
        for (auto& context : forest_tree_layout_contexts_) {
            run_finalize("forest_tree_layout_context", [&] { return context->finalize(); });
        }
        for (auto& context : forest_prediction_contexts_) {
            run_finalize("forest_prediction_context", [&] { return context->finalize(); });
        }
        for (auto& context : forest_classification_contexts_) {
            run_finalize("forest_classification_context", [&] { return context->finalize(); });
        }

        run_finalize("forest_training_proof", [&] { return forest_training_proof_.finalize(); });
        run_finalize("forest_histogram_refresh", [&] { return forest_histogram_refresh_.finalize(); });
        run_finalize("forest_histogram", [&] { return forest_histogram_.finalize(); });
        run_finalize("forest_histogram_bucket_address", [&] { return forest_histogram_bucket_address_.finalize(); });
        run_finalize("forest_training_inference", [&] { return forest_training_inference_.finalize(); });
        run_finalize("forest_split_statistics", [&] { return forest_split_statistics_.finalize(); });
        run_finalize("forest_split_gain", [&] { return forest_split_gain_.finalize(); });
        run_finalize("forest_quantized_split_gain", [&] { return forest_quantized_split_gain_.finalize(); });
        run_finalize("forest_split_selection", [&] { return forest_split_selection_.finalize(); });
        run_finalize("forest_split_update", [&] { return forest_split_update_.finalize(); });
        run_finalize("forest_leaf_value", [&] { return forest_leaf_value_.finalize(); });
        run_finalize("forest_boosting_state", [&] { return forest_boosting_state_.finalize(); });
        run_finalize("forest_structure", [&] { return forest_structure_.finalize(); });
        run_finalize("tree_histogram_consistency", [&] { return tree_histogram_consistency_.finalize(); });
        run_finalize("root_selector_consistency", [&] { return root_selector_consistency_.finalize(); });
        run_finalize("node_role_consistency", [&] { return node_role_consistency_.finalize(); });
        run_finalize("forest_depth_consistency", [&] { return forest_depth_consistency_.finalize(); });
        run_finalize("root_box_range", [&] { return root_box_range_.finalize(); });
        run_finalize("parent_child_box_inheritance", [&] { return parent_child_box_inheritance_.finalize(); });
        run_finalize("parent_child_tree_consistency", [&] { return parent_child_tree_consistency_.finalize(); });
        run_finalize("masked_argmax", [&] { return masked_argmax_.finalize(); });
        run_finalize("argmax", [&] { return argmax_.finalize(); });
        run_finalize("cube_middle_argmax", [&] { return cube_middle_argmax_.finalize(); });
        run_finalize("cube_middle_softmax", [&] { return cube_middle_softmax_.finalize(); });
        run_finalize("cube_middle_replication", [&] { return cube_middle_replication_.finalize(); });
        run_finalize("exp_check", [&] { return exp_.finalize(); });
        run_finalize("div_check", [&] { return div_.finalize(); });
        run_finalize("rational_check", [&] { return rational_.finalize(); });
        run_finalize("fixed_point_division", [&] { return fixed_point_division_.finalize(); });
        run_finalize("box_check", [&] { return box_check_.finalize(); });
        run_finalize("signed_range_check", [&] { return signed_range_.finalize(); });
        run_finalize("selector_check", [&] { return selector_.finalize(); });
        run_finalize("zkboost_histogram_ram", [&] { return zkboost_histogram_ram_.finalize(); });
        run_finalize("hist_agg", [&] { return hist_agg_.finalize(); });
        run_finalize("index_routing", [&] { return index_routing_.finalize(); });
        run_finalize("concatenated_table_lookup", [&] { return concatenated_table_lookup_.finalize(); });
        run_finalize("column_replication", [&] { return column_replication_.finalize(); });
        run_finalize("cube_sum", [&] { return cube_sum_.finalize(); });
        run_finalize("table_lookup", [&] { return table_lookup_.finalize(); });
        run_finalize("less_check", [&] { return less_.finalize(); });
        run_finalize("shuffle", [&] { return shuffle_.finalize(); });
        run_finalize("lookup", [&] { return lookup_.finalize(); });
        run_finalize("column_prefix", [&] { return column_prefix_.finalize(); });
        run_finalize("row_prefix", [&] { return row_prefix_.finalize(); });
        run_finalize("column_sum", [&] { return column_sum_.finalize(); });
        run_finalize("row_sum", [&] { return row_sum_.finalize(); });
        run_finalize("sum_check", [&] { return sum_.finalize(); });
        run_finalize("nonzero_check", [&] { return nonzero_.finalize(); });
        run_finalize("zero_check.prepare_opening", [&] { return zero_.prepare_opening(); });
        prepared_needs_shared_opening_stage_ = zero_.needs_opening();
        for (const auto& context : cq_table_lookup_contexts_) {
            prepared_needs_shared_opening_stage_ = context->needs_opening() || prepared_needs_shared_opening_stage_;
        }
        for (const auto& context : cq_lookup_contexts_) {
            prepared_needs_shared_opening_stage_ = context->needs_opening() || prepared_needs_shared_opening_stage_;
        }
        for (const auto& context : cq_concatenated_table_lookup_contexts_) {
            prepared_needs_shared_opening_stage_ =
                context->needs_opening() || prepared_needs_shared_opening_stage_;
        }

        verification_prepared_ = true;
        verification_prepare_ok_ = ok;
        return ok;
    }

    bool complete_verifier_opening_preprocessing()
    {
        if (runtime_.mode() != ProofMode::Verify) {
            throw std::runtime_error("ProofProgram::complete_verifier_opening_preprocessing is verifier-only");
        }
        if (!verification_prepared_) {
            throw std::runtime_error("ProofProgram verifier preprocessing has not been prepared");
        }
        if (verification_opening_completed_) {
            throw std::runtime_error("ProofProgram verifier opening preprocessing already completed");
        }
        if (verification_completed_) {
            throw std::runtime_error("ProofProgram verifier preprocessing already completed");
        }

        bool ok = verification_prepare_ok_;
        const bool profile_finalize = []() {
            const char* flag = terrae_getenv("TERRAE_PROFILE_FINALIZE");
            return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
        }();
        auto run_finalize = [&](const char* name, auto&& fn) {
            const bool structured_enabled = structured_protocol_profile::enabled();
            if (!profile_finalize && !structured_enabled) {
                ok = fn() && ok;
                return;
            }
            const auto start = std::chrono::steady_clock::now();
            bool stage_ok = false;
            try {
                stage_ok = fn();
            } catch (const std::exception& ex) {
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                if (profile_finalize) {
                    std::cerr << "finalize_timing " << name << " " << elapsed.count() << " throw\n";
                }
                structured_protocol_profile::append_row(
                    "protocol_finalize", name, "", elapsed.count(), std::string("throw:") + ex.what());
                throw;
            } catch (...) {
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                if (profile_finalize) {
                    std::cerr << "finalize_timing " << name << " " << elapsed.count() << " throw\n";
                }
                structured_protocol_profile::append_row("protocol_finalize", name, "", elapsed.count(), "throw");
                throw;
            }
            const auto end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = end - start;
            if (profile_finalize) {
                std::cerr << "finalize_timing " << name << " " << elapsed.count() << " "
                          << (stage_ok ? "ok" : "fail") << "\n";
            }
            structured_protocol_profile::append_row(
                "protocol_finalize", name, "", elapsed.count(), stage_ok ? "ok" : "fail");
            ok = stage_ok && ok;
        };

        if (prepared_needs_shared_opening_stage_) {
            const Fr shared_opening = runtime_.kzg().shared_opening_challenge("protocol.opening.stage0", true);
            for (auto& context : cq_table_lookup_contexts_) {
                run_finalize("cq_table_lookup_context.complete_opening",
                             [&] { return context->finalize_complete_opening(shared_opening); });
            }
            for (auto& context : cq_lookup_contexts_) {
                run_finalize("cq_lookup_context.complete_opening",
                             [&] { return context->finalize_complete_opening(shared_opening); });
            }
            for (auto& context : cq_concatenated_table_lookup_contexts_) {
                run_finalize("cq_concatenated_table_lookup_context.complete_opening",
                             [&] { return context->finalize_complete_opening(shared_opening); });
            }
            run_finalize("zero_check.complete_opening", [&] { return zero_.complete_opening(shared_opening); });
        } else {
            run_finalize("zero_check.complete_opening", [&] { return zero_.complete_opening(Fr(0)); });
        }

        verification_prepare_ok_ = ok;
        verification_opening_completed_ = true;
        return ok;
    }

    bool verify_preprocessed()
    {
        if (runtime_.mode() != ProofMode::Verify) {
            throw std::runtime_error("ProofProgram::verify_preprocessed is verifier-only");
        }
        if (!verification_prepared_) {
            throw std::runtime_error("ProofProgram verifier preprocessing has not been prepared");
        }
        if (verification_completed_) {
            throw std::runtime_error("ProofProgram verifier preprocessing already completed");
        }

        bool ok = verification_prepare_ok_;
        const bool profile_finalize = []() {
            const char* flag = terrae_getenv("TERRAE_PROFILE_FINALIZE");
            return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
        }();
        auto run_finalize = [&](const char* name, auto&& fn) {
            const bool structured_enabled = structured_protocol_profile::enabled();
            if (!profile_finalize && !structured_enabled) {
                ok = fn() && ok;
                return;
            }
            const auto start = std::chrono::steady_clock::now();
            bool stage_ok = false;
            try {
                stage_ok = fn();
            } catch (const std::exception& ex) {
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                if (profile_finalize) {
                    std::cerr << "finalize_timing " << name << " " << elapsed.count() << " throw\n";
                }
                structured_protocol_profile::append_row(
                    "protocol_finalize", name, "", elapsed.count(), std::string("throw:") + ex.what());
                throw;
            } catch (...) {
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end - start;
                if (profile_finalize) {
                    std::cerr << "finalize_timing " << name << " " << elapsed.count() << " throw\n";
                }
                structured_protocol_profile::append_row("protocol_finalize", name, "", elapsed.count(), "throw");
                throw;
            }
            const auto end = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = end - start;
            if (profile_finalize) {
                std::cerr << "finalize_timing " << name << " " << elapsed.count() << " "
                          << (stage_ok ? "ok" : "fail") << "\n";
            }
            structured_protocol_profile::append_row(
                "protocol_finalize", name, "", elapsed.count(), stage_ok ? "ok" : "fail");
            ok = stage_ok && ok;
        };

        if (!verification_opening_completed_) {
            ok = complete_verifier_opening_preprocessing() && ok;
        }
        run_finalize("kzg", [&] { return runtime_.finalize_kzg(); });
        verification_completed_ = true;
        return ok;
    }

private:
    ProofRuntime& runtime_;
    terraeSoundnessCounter soundness_counter_;
    zeroProtocol zero_;
    nonzeroCheckProtocol nonzero_;
    sumProtocol sum_;
    lookupProtocol lookup_;
    lessCheckProtocol less_;
    signedRangeCheckProtocol signed_range_;
    divCheckProtocol div_;
    rationalCheckProtocol rational_;
    fixedPointDivisionProtocol fixed_point_division_;
    selectorCheckProtocol selector_;
    shuffleProtocol shuffle_;
    forestHistogramProtocol forest_histogram_;
    forestHistogramBucketAddressProtocol forest_histogram_bucket_address_;
    forestTrainingInferenceProtocol forest_training_inference_;
    forestHistogramRefreshProtocol forest_histogram_refresh_;
    forestSplitStatisticsProtocol forest_split_statistics_;
    forestSplitGainProtocol forest_split_gain_;
    forestQuantizedSplitGainProtocol forest_quantized_split_gain_;
    forestSplitSelectionProtocol forest_split_selection_;
    forestTrainingProofProtocol forest_training_proof_;
    forestSplitUpdateProtocol forest_split_update_;
    forestLeafValueProtocol forest_leaf_value_;
    forestBoostingStateProtocol forest_boosting_state_;
    forestStructureProtocol forest_structure_;
    treeHistogramConsistencyProtocol tree_histogram_consistency_;
    rootSelectorConsistencyProtocol root_selector_consistency_;
    nodeRoleConsistencyProtocol node_role_consistency_;
    forestDepthConsistencyProtocol forest_depth_consistency_;
    rootBoxRangeProtocol root_box_range_;
    boxCheckProtocol box_check_;
    parentChildBoxInheritanceProtocol parent_child_box_inheritance_;
    parentChildTreeConsistencyProtocol parent_child_tree_consistency_;
    indexRoutingProtocol index_routing_;
    maskedArgmaxProtocol masked_argmax_;
    argmaxProtocol argmax_;
    tableLookupProtocol table_lookup_;
    concatenatedTableLookupProtocol concatenated_table_lookup_;
    histAggProtocol hist_agg_;
    zkboostHistogramRamProtocol zkboost_histogram_ram_;
    expCheckProtocol exp_;
    columnReplicationProtocol column_replication_;
    matrixColumnPrefixSumProtocol column_prefix_;
    matrixRowPrefixSumProtocol row_prefix_;
    matrixColumnSumProtocol column_sum_;
    matrixRowSumProtocol row_sum_;
    cubeSumProtocol cube_sum_;
    cubeMiddleReplicationProtocol cube_middle_replication_;
    cubeMiddleArgmaxProtocol cube_middle_argmax_;
    cubeMiddleSoftmaxProtocol cube_middle_softmax_;
    std::vector<std::unique_ptr<CqLookupContext>> cq_lookup_contexts_;
    std::vector<std::unique_ptr<CqTableLookupContext>> cq_table_lookup_contexts_;
    std::vector<std::unique_ptr<CqConcatenatedTableLookupContext>> cq_concatenated_table_lookup_contexts_;
    std::vector<std::unique_ptr<ForestRowFetchContext>> forest_row_fetch_contexts_;
    std::vector<std::unique_ptr<ForestInferenceContext>> forest_inference_contexts_;
    std::vector<std::unique_ptr<ForestTreeLayoutContext>> forest_tree_layout_contexts_;
    std::vector<std::unique_ptr<ForestPredictionContext>> forest_prediction_contexts_;
    std::vector<std::unique_ptr<ForestClassificationContext>> forest_classification_contexts_;
    bool verification_prepared_ = false;
    bool verification_opening_completed_ = false;
    bool verification_completed_ = false;
    bool verification_prepare_ok_ = false;
    bool prepared_needs_shared_opening_stage_ = false;
};

template <class SequenceFn>
bool run_proof_program(ProofRuntime& runtime, SequenceFn&& sequence)
{
    ProofProgram program(runtime);
    const bool profile_sequence = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_SEQUENCE");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    if (!profile_sequence) {
        std::forward<SequenceFn>(sequence)(program);
        return program.finalize();
    }
    const auto sequence_start = std::chrono::steady_clock::now();
    std::forward<SequenceFn>(sequence)(program);
    const auto sequence_end = std::chrono::steady_clock::now();
    const auto finalize_start = sequence_end;
    const bool ok = program.finalize();
    const auto finalize_end = std::chrono::steady_clock::now();
    std::cerr << "program_timing sequence "
              << std::chrono::duration<double>(sequence_end - sequence_start).count() << "\n";
    std::cerr << "program_timing finalize "
              << std::chrono::duration<double>(finalize_end - finalize_start).count() << "\n";
    return ok;
}

template <class RunnerFn>
ProofArtifact prove_with_runtime(size_t deg1, size_t deg2, RunnerFn&& runner)
{
    ProofRuntime prover = ProofRuntime::prover(deg1, deg2);
    if (!std::forward<RunnerFn>(runner)(prover)) {
        throw std::runtime_error("prove_with_runtime: prover execution failed");
    }
    return make_proof_artifact(prover.take_proof());
}

template <class RunnerFn>
ProofArtifact prove_with_runtime(const kzgPublicParameters& public_params, RunnerFn&& runner)
{
    ProofRuntime prover = ProofRuntime::prover_borrowed(public_params);
    if (!std::forward<RunnerFn>(runner)(prover)) {
        throw std::runtime_error("prove_with_runtime: prover execution failed");
    }
    return make_proof_artifact(prover.take_proof());
}

template <class RunnerFn>
ProofArtifact prove_with_runtime(const kzgPublicParameters& public_params,
                                 ProofEncoding encoding,
                                 RunnerFn&& runner)
{
    ProofRuntime prover =
        ProofRuntime::prover_borrowed(public_params, std::make_shared<ProofTranscript>(), encoding);
    if (!std::forward<RunnerFn>(runner)(prover)) {
        throw std::runtime_error("prove_with_runtime: prover execution failed");
    }
    return make_proof_artifact(prover.take_proof());
}

template <class RunnerFn>
VerificationResult verify_with_runtime(const kzgPublicParameters& public_params,
                                       const ProofArtifact& artifact,
                                       RunnerFn&& runner)
{
    ProofRuntime verifier = ProofRuntime::verifier(public_params, artifact.proof);
    const bool accepted = std::forward<RunnerFn>(runner)(verifier);
    const bool proof_fully_consumed = verifier.proof_fully_consumed();
    std::vector<uint8_t> transcript_state = verifier.transcript_state();
    VerificationResult result;
    result.accepted = accepted;
    result.proof_fully_consumed = proof_fully_consumed;
    result.transcript_state = std::move(transcript_state);
    return result;
}
