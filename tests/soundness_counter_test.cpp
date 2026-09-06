#include "profiling/soundness_counter.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace {

std::uint64_t metric(const proof_cost::Report& report, const std::string& name)
{
    const auto it = report.counters().find(name);
    if (it == report.counters().end()) {
        return 0;
    }
    return it->second;
}

TEST(SoundnessCounter, DerivesAppendixKInferenceLedger)
{
    proof_cost::Report report;
    proof_cost::Scope scope(report);
    terraeSoundnessCounter counter;
    counter.set_stage(terraeSoundnessStage::Inference);

    for (std::size_t i = 0; i < 32; ++i) {
        counter.record_zero_claim(128);
    }
    for (std::size_t i = 0; i < 8; ++i) {
        counter.record_sum_claim(true, 1024);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        counter.record_logical_lookup(64, 128);
    }
    counter.record_materialized_lookup(128, 128);
    counter.emit_metrics(25);

    EXPECT_EQ(metric(report, "soundness_session_commitment_count"), 25);
    EXPECT_EQ(metric(report, "soundness_session_zero_claims"), 32);
    EXPECT_EQ(metric(report, "soundness_session_sum_batched_claims"), 0);
    EXPECT_EQ(metric(report, "soundness_session_sum_direct_opening_claims"), 8);
    EXPECT_EQ(metric(report, "soundness_session_logical_lookup_instances"), 4);
    EXPECT_EQ(metric(report, "soundness_session_materialized_lookup_instances"), 1);
    EXPECT_EQ(metric(report, "soundness_session_evaluated_identity_count"), 1);
    EXPECT_EQ(metric(report, "soundness_session_domain_bound"), 128);
    EXPECT_EQ(metric(report, "soundness_session_domain_log2"), 7);
    EXPECT_EQ(metric(report, "soundness_session_c_pi"), 43);
    EXPECT_EQ(metric(report, "soundness_session_round_by_round_numerator"), 43 * 128);
    EXPECT_EQ(metric(report, "soundness_session_quotient_degree_bound_exclusive"), 3 * 128);
    EXPECT_EQ(metric(report, "soundness_session_tested_identity_degree_bound_exclusive"), 4 * 128);
}

TEST(SoundnessCounter, AggregatesStagesAndKeepsLogicalAndMaterializedLookupsDistinct)
{
    proof_cost::Report report;
    proof_cost::Scope scope(report);
    terraeSoundnessCounter counter;

    counter.set_stage(terraeSoundnessStage::WellFormedness);
    counter.record_zero_claim(16);
    counter.record_sum_claim(false, 32);
    counter.record_histogram(8, 3);
    counter.record_logical_lookup(16, 32);
    counter.record_logical_lookup(16, 32);

    counter.set_stage(terraeSoundnessStage::Training);
    counter.record_zero_claim(64);
    counter.record_materialized_lookup(128, 32);
    counter.record_deterministic_root_uniqueness(4);
    counter.emit_metrics(11);

    EXPECT_EQ(metric(report, "soundness_session_zero_claims"), 2);
    EXPECT_EQ(metric(report, "soundness_session_sum_batched_claims"), 1);
    EXPECT_EQ(metric(report, "soundness_session_histogram_checks"), 3);
    EXPECT_EQ(metric(report, "soundness_session_logical_lookup_instances"), 2);
    EXPECT_EQ(metric(report, "soundness_session_materialized_lookup_instances"), 1);
    EXPECT_EQ(metric(report, "soundness_session_evaluated_identity_count"), 2);
    EXPECT_EQ(metric(report, "soundness_session_domain_bound"), 128);
    EXPECT_EQ(metric(report, "soundness_session_c_pi"), 16);
    EXPECT_EQ(metric(report, "soundness_session_deterministic_root_uniqueness_checks"), 1);
    EXPECT_EQ(metric(report, "soundness_session_deterministic_root_tree_ids"), 4);
}

} // namespace
