#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "profiling/proof_cost.hpp"

// This observer records the terms used by the concrete algebraic soundness
// bound. It is deliberately outside the proof transcript: recording must not
// create commitments, sample challenges, or alter proof serialization.
enum class terraeSoundnessStage : std::size_t {
    Unscoped = 0,
    Inference,
    WellFormedness,
    Training,
    Count,
};

class terraeSoundnessCounter {
public:
    class StageScope {
    public:
        StageScope(terraeSoundnessCounter& counter, terraeSoundnessStage stage)
            : counter_(counter), previous_(counter.current_stage_)
        {
            counter_.current_stage_ = stage;
        }

        ~StageScope()
        {
            counter_.current_stage_ = previous_;
        }

        StageScope(const StageScope&) = delete;
        StageScope& operator=(const StageScope&) = delete;

    private:
        terraeSoundnessCounter& counter_;
        terraeSoundnessStage previous_;
    };

    StageScope scoped_stage(terraeSoundnessStage stage)
    {
        return StageScope(*this, stage);
    }

    void set_stage(terraeSoundnessStage stage)
    {
        current_stage_ = stage;
    }

    void record_zero_claim(std::size_t domain_size)
    {
        StageStats& out = stats();
        ++out.zero_claims;
        out.zero_max_domain_size = std::max(out.zero_max_domain_size,
                                            static_cast<std::uint64_t>(domain_size));
    }

    void record_sum_claim(bool direct_opening, std::size_t domain_size)
    {
        StageStats& out = stats();
        if (direct_opening) {
            ++out.sum_direct_opening_claims;
            out.sum_direct_opening_max_domain_size =
                std::max(out.sum_direct_opening_max_domain_size,
                         static_cast<std::uint64_t>(domain_size));
        } else {
            ++out.sum_batched_claims;
            out.sum_batched_max_domain_size =
                std::max(out.sum_batched_max_domain_size,
                         static_cast<std::uint64_t>(domain_size));
        }
    }

    // A logical lookup is registered before deterministic grouping and
    // interleaving. Appendix K uses this conservative count in its ledger.
    void record_logical_lookup(std::size_t query_rows, std::size_t table_rows)
    {
        StageStats& out = stats();
        ++out.logical_lookup_instances;
        out.logical_lookup_query_rows += query_rows;
        out.logical_lookup_table_rows += table_rows;
        out.logical_lookup_max_query_rows =
            std::max(out.logical_lookup_max_query_rows, static_cast<std::uint64_t>(query_rows));
        out.logical_lookup_max_table_rows =
            std::max(out.logical_lookup_max_table_rows, static_cast<std::uint64_t>(table_rows));
    }

    // A materialized lookup is the actual lookup instance after deterministic
    // grouping/interleaving. It is reported separately for implementation audit.
    void record_materialized_lookup(std::size_t query_rows, std::size_t table_rows)
    {
        StageStats& out = stats();
        ++out.materialized_lookup_instances;
        out.materialized_lookup_query_rows += query_rows;
        out.materialized_lookup_table_rows += table_rows;
        out.materialized_lookup_max_query_rows =
            std::max(out.materialized_lookup_max_query_rows,
                     static_cast<std::uint64_t>(query_rows));
        out.materialized_lookup_max_table_rows =
            std::max(out.materialized_lookup_max_table_rows,
                     static_cast<std::uint64_t>(table_rows));
    }

    void record_histogram(std::size_t address_space, std::size_t channel_count)
    {
        StageStats& out = stats();
        out.histogram_checks += channel_count;
        out.histogram_address_space_sum_minus_one += channel_count * (address_space - 1);
        out.histogram_max_address_space = std::max(out.histogram_max_address_space,
                                                   static_cast<std::uint64_t>(address_space));
    }

    // Root uniqueness is deterministic routing plus a zero-check in the
    // implementation, not a randomized histogram/power-sum test. Keep this
    // explicit so exported bounds do not silently charge a TK/p term.
    void record_deterministic_root_uniqueness(std::size_t tree_count)
    {
        StageStats& out = stats();
        ++out.deterministic_root_uniqueness_checks;
        out.deterministic_root_tree_ids += tree_count;
    }

    void emit_metrics(std::size_t commitment_count) const
    {
        StageStats session{};
        for (const StageStats& in : stats_) {
            session.zero_claims += in.zero_claims;
            session.zero_max_domain_size = std::max(session.zero_max_domain_size, in.zero_max_domain_size);
            session.sum_batched_claims += in.sum_batched_claims;
            session.sum_batched_max_domain_size =
                std::max(session.sum_batched_max_domain_size, in.sum_batched_max_domain_size);
            session.sum_direct_opening_claims += in.sum_direct_opening_claims;
            session.sum_direct_opening_max_domain_size =
                std::max(session.sum_direct_opening_max_domain_size,
                         in.sum_direct_opening_max_domain_size);
            session.logical_lookup_instances += in.logical_lookup_instances;
            session.logical_lookup_query_rows += in.logical_lookup_query_rows;
            session.logical_lookup_table_rows += in.logical_lookup_table_rows;
            session.logical_lookup_max_query_rows =
                std::max(session.logical_lookup_max_query_rows, in.logical_lookup_max_query_rows);
            session.logical_lookup_max_table_rows =
                std::max(session.logical_lookup_max_table_rows, in.logical_lookup_max_table_rows);
            session.materialized_lookup_instances += in.materialized_lookup_instances;
            session.materialized_lookup_query_rows += in.materialized_lookup_query_rows;
            session.materialized_lookup_table_rows += in.materialized_lookup_table_rows;
            session.materialized_lookup_max_query_rows =
                std::max(session.materialized_lookup_max_query_rows,
                         in.materialized_lookup_max_query_rows);
            session.materialized_lookup_max_table_rows =
                std::max(session.materialized_lookup_max_table_rows,
                         in.materialized_lookup_max_table_rows);
            session.histogram_checks += in.histogram_checks;
            session.histogram_address_space_sum_minus_one += in.histogram_address_space_sum_minus_one;
            session.histogram_max_address_space = std::max(session.histogram_max_address_space, in.histogram_max_address_space);
            session.deterministic_root_uniqueness_checks += in.deterministic_root_uniqueness_checks;
            session.deterministic_root_tree_ids += in.deterministic_root_tree_ids;
        }
        emit_stage_metrics("session", session, commitment_count);
        for (std::size_t i = 0; i < static_cast<std::size_t>(terraeSoundnessStage::Count); ++i) {
            emit_stage_metrics(stage_name(static_cast<terraeSoundnessStage>(i)), stats_[i], 0);
        }
    }

private:
    struct StageStats {
        std::uint64_t zero_claims = 0, zero_max_domain_size = 0;
        std::uint64_t sum_batched_claims = 0, sum_batched_max_domain_size = 0;
        std::uint64_t sum_direct_opening_claims = 0, sum_direct_opening_max_domain_size = 0;
        std::uint64_t logical_lookup_instances = 0, logical_lookup_query_rows = 0;
        std::uint64_t logical_lookup_table_rows = 0, logical_lookup_max_query_rows = 0;
        std::uint64_t logical_lookup_max_table_rows = 0;
        std::uint64_t materialized_lookup_instances = 0, materialized_lookup_query_rows = 0;
        std::uint64_t materialized_lookup_table_rows = 0, materialized_lookup_max_query_rows = 0;
        std::uint64_t materialized_lookup_max_table_rows = 0;
        std::uint64_t histogram_checks = 0, histogram_address_space_sum_minus_one = 0, histogram_max_address_space = 0;
        std::uint64_t deterministic_root_uniqueness_checks = 0, deterministic_root_tree_ids = 0;
    };

    static std::uint64_t coefficient(std::uint64_t count) { return count == 0 ? 0 : count - 1; }

    static std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs)
    {
        if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
            throw std::overflow_error("terraeSoundnessCounter metric overflow");
        }
        return lhs * rhs;
    }

    static std::uint64_t exact_log2(std::uint64_t value)
    {
        if (value == 0) return 0;
        if ((value & (value - 1)) != 0) {
            throw std::runtime_error("terraeSoundnessCounter domain bound must be a power of two");
        }
        std::uint64_t result = 0;
        while (value > 1) {
            value >>= 1;
            ++result;
        }
        return result;
    }

    static void emit_stage_metrics(const std::string& stage,
                                   const StageStats& in,
                                   std::uint64_t commitment_count)
    {
        constexpr std::uint64_t constraint_degree = 4;
        const std::string prefix = "soundness_" + stage + "_";
        const std::uint64_t zero_coefficient = coefficient(in.zero_claims);
        const std::uint64_t sum_coefficient = coefficient(in.sum_batched_claims);
        const std::uint64_t evaluated_identities =
            (in.zero_claims == 0 ? 0 : 1) + (in.sum_batched_claims == 0 ? 0 : 1);
        const std::uint64_t evaluation_coefficient = constraint_degree * evaluated_identities;
        const std::uint64_t lookup_compression_coefficient = 2 * in.logical_lookup_instances;
        const std::uint64_t c_pi = zero_coefficient + sum_coefficient + in.histogram_checks +
                                   evaluation_coefficient + lookup_compression_coefficient;
        const std::uint64_t domain_bound = std::max({in.zero_max_domain_size,
                                                     in.sum_batched_max_domain_size,
                                                     in.histogram_max_address_space,
                                                     in.logical_lookup_max_query_rows,
                                                     in.logical_lookup_max_table_rows,
                                                     in.materialized_lookup_max_query_rows,
                                                     in.materialized_lookup_max_table_rows});

        proof_cost::add(prefix + "commitment_count", commitment_count);
        proof_cost::add(prefix + "zero_claims", in.zero_claims);
        proof_cost::add(prefix + "zero_batch_coefficient", zero_coefficient);
        proof_cost::set_max(prefix + "zero_max_domain_size", in.zero_max_domain_size);
        proof_cost::add(prefix + "sum_batched_claims", in.sum_batched_claims);
        proof_cost::add(prefix + "sum_batch_coefficient", sum_coefficient);
        proof_cost::set_max(prefix + "sum_batched_max_domain_size", in.sum_batched_max_domain_size);
        proof_cost::add(prefix + "sum_direct_opening_claims", in.sum_direct_opening_claims);
        proof_cost::set_max(prefix + "sum_direct_opening_max_domain_size",
                            in.sum_direct_opening_max_domain_size);
        proof_cost::add(prefix + "logical_lookup_instances", in.logical_lookup_instances);
        proof_cost::add(prefix + "logical_lookup_query_rows", in.logical_lookup_query_rows);
        proof_cost::add(prefix + "logical_lookup_table_rows", in.logical_lookup_table_rows);
        proof_cost::set_max(prefix + "logical_lookup_max_query_rows", in.logical_lookup_max_query_rows);
        proof_cost::set_max(prefix + "logical_lookup_max_table_rows", in.logical_lookup_max_table_rows);
        proof_cost::add(prefix + "materialized_lookup_instances", in.materialized_lookup_instances);
        proof_cost::add(prefix + "materialized_lookup_query_rows", in.materialized_lookup_query_rows);
        proof_cost::add(prefix + "materialized_lookup_table_rows", in.materialized_lookup_table_rows);
        proof_cost::set_max(prefix + "materialized_lookup_max_query_rows",
                            in.materialized_lookup_max_query_rows);
        proof_cost::set_max(prefix + "materialized_lookup_max_table_rows",
                            in.materialized_lookup_max_table_rows);
        // Backward-compatible aliases for the previously emitted logical count.
        proof_cost::add(prefix + "lookup_instances", in.logical_lookup_instances);
        proof_cost::add(prefix + "lookup_query_rows", in.logical_lookup_query_rows);
        proof_cost::add(prefix + "lookup_table_rows", in.logical_lookup_table_rows);
        proof_cost::set_max(prefix + "lookup_max_table_rows", in.logical_lookup_max_table_rows);
        proof_cost::add(prefix + "histogram_checks", in.histogram_checks);
        proof_cost::add(prefix + "histogram_coefficient", in.histogram_address_space_sum_minus_one);
        proof_cost::set_max(prefix + "histogram_max_address_space", in.histogram_max_address_space);
        proof_cost::add(prefix + "deterministic_root_uniqueness_checks", in.deterministic_root_uniqueness_checks);
        proof_cost::add(prefix + "deterministic_root_tree_ids", in.deterministic_root_tree_ids);
        proof_cost::add(prefix + "algebraic_coefficient",
                        zero_coefficient + sum_coefficient + in.histogram_address_space_sum_minus_one);
        proof_cost::add(prefix + "evaluated_identity_count", evaluated_identities);
        proof_cost::add(prefix + "evaluation_coefficient", evaluation_coefficient);
        proof_cost::add(prefix + "lookup_compression_coefficient", lookup_compression_coefficient);
        proof_cost::add(prefix + "constraint_degree", constraint_degree);
        proof_cost::set_max(prefix + "domain_bound", domain_bound);
        proof_cost::add(prefix + "domain_log2", exact_log2(domain_bound));
        proof_cost::add(prefix + "c_pi", c_pi);
        proof_cost::add(prefix + "round_by_round_numerator", checked_multiply(c_pi, domain_bound));
        proof_cost::add(prefix + "quotient_degree_bound_exclusive",
                        checked_multiply(constraint_degree - 1, domain_bound));
        proof_cost::add(prefix + "tested_identity_degree_bound_exclusive",
                        checked_multiply(constraint_degree, domain_bound));
    }
    static std::string stage_name(terraeSoundnessStage stage)
    {
        switch (stage) {
        case terraeSoundnessStage::Inference: return "inference";
        case terraeSoundnessStage::WellFormedness: return "well_formedness";
        case terraeSoundnessStage::Training: return "training";
        case terraeSoundnessStage::Unscoped:
        default: return "unscoped";
        }
    }
    StageStats& stats() { return stats_[static_cast<std::size_t>(current_stage_)]; }

    std::array<StageStats, static_cast<std::size_t>(terraeSoundnessStage::Count)> stats_{};
    terraeSoundnessStage current_stage_ = terraeSoundnessStage::Unscoped;
};
