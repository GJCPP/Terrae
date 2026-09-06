#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "protocol_profile.hpp"
#include "proof_runtime.hpp"
#include "terrae_env.hpp"

class kzgSrsRequirement {
public:
    size_t deg1_plus_1 = 1;
    size_t deg2_plus_1 = 1;
    size_t min_degree_bound_plus_1 = 0;
    size_t opening_deg2_plus_1_floor = 1;
};

inline kzgSrsRequirement merge_kzg_srs_requirements(const kzgSrsRequirement& left,
                                                    const kzgSrsRequirement& right)
{
    kzgSrsRequirement out;
    out.deg1_plus_1 = std::max(left.deg1_plus_1, right.deg1_plus_1);
    out.opening_deg2_plus_1_floor =
        std::max(left.opening_deg2_plus_1_floor, right.opening_deg2_plus_1_floor);
    if (left.min_degree_bound_plus_1 == 0) {
        out.min_degree_bound_plus_1 = right.min_degree_bound_plus_1;
    } else if (right.min_degree_bound_plus_1 == 0) {
        out.min_degree_bound_plus_1 = left.min_degree_bound_plus_1;
    } else {
        out.min_degree_bound_plus_1 =
            std::min(left.min_degree_bound_plus_1, right.min_degree_bound_plus_1);
    }
    const size_t degree_shift_requirement =
        out.min_degree_bound_plus_1 == 0
            ? size_t(1)
            : out.deg1_plus_1 - out.min_degree_bound_plus_1 + 1;
    out.deg2_plus_1 = std::max({left.deg2_plus_1,
                               right.deg2_plus_1,
                               out.opening_deg2_plus_1_floor,
                               degree_shift_requirement});
    return out;
}

inline ProtocolPolynomialInput witness_input(ProofRuntime& runtime,
                                             const std::string& label,
                                             const poly& polynomial,
                                             size_t degree_bound)
{
    if (runtime.mode() == ProofMode::Prove) {
        return WitnessCommitmentInput::prover(label, polynomial, degree_bound);
    }
    return WitnessCommitmentInput::verifier(label, degree_bound);
}

inline size_t degree_bound_for_domain(size_t domain_size, const char* context = "training_runtime_utils")
{
    if (domain_size == 0) {
        throw std::runtime_error(std::string(context) + ": domain size must be non-zero");
    }
    return domain_size - 1;
}

inline size_t vector_degree_bound(size_t row_count, const char* context = "training_runtime_utils")
{
    return degree_bound_for_domain(row_count, context);
}

inline size_t matrix_degree_bound(size_t row_count,
                                  size_t column_count,
                                  const char* context = "training_runtime_utils")
{
    return degree_bound_for_domain(row_count * column_count, context);
}

inline void update_kzg_srs_requirement(kzgSrsRequirement& requirement, size_t domain_size)
{
    if (domain_size == 0) {
        throw std::runtime_error("training_runtime_utils: domain size must be non-zero");
    }
    requirement.deg1_plus_1 = std::max(requirement.deg1_plus_1, domain_size);
    if (requirement.min_degree_bound_plus_1 == 0) {
        requirement.min_degree_bound_plus_1 = domain_size;
    } else {
        requirement.min_degree_bound_plus_1 = std::min(requirement.min_degree_bound_plus_1, domain_size);
    }
    requirement.deg2_plus_1 = std::max(
        requirement.opening_deg2_plus_1_floor,
        requirement.deg1_plus_1 - requirement.min_degree_bound_plus_1 + 1);
}

inline void require_kzg_distinct_opening_points(kzgSrsRequirement& requirement, size_t point_count)
{
    if (point_count == 0) {
        return;
    }
    requirement.opening_deg2_plus_1_floor =
        std::max(requirement.opening_deg2_plus_1_floor, point_count + 1);
    requirement.deg2_plus_1 = std::max(
        requirement.deg2_plus_1,
        std::max(requirement.opening_deg2_plus_1_floor,
                 requirement.min_degree_bound_plus_1 == 0
                     ? size_t(1)
                     : requirement.deg1_plus_1 - requirement.min_degree_bound_plus_1 + 1));
}

inline void require_kzg_constant_degree_claim(kzgSrsRequirement& requirement)
{
    if (requirement.min_degree_bound_plus_1 == 0) {
        requirement.min_degree_bound_plus_1 = 1;
    } else {
        requirement.min_degree_bound_plus_1 = std::min(requirement.min_degree_bound_plus_1, size_t(1));
    }
    requirement.deg2_plus_1 = std::max(
        requirement.opening_deg2_plus_1_floor,
        requirement.deg1_plus_1 - requirement.min_degree_bound_plus_1 + 1);
}

inline void ensure_kzg_srs_covers(const kzgPublicParameters& public_params,
                                  const kzgSrsRequirement& requirement,
                                  const std::string& context)
{
    const size_t have_deg1_plus_1 = public_params.deg1 + 1;
    const size_t have_deg2_plus_1 = public_params.deg2 + 1;
    if (have_deg1_plus_1 < requirement.deg1_plus_1 || have_deg2_plus_1 < requirement.deg2_plus_1) {
        throw std::runtime_error(context + ": loaded KZG SRS is too small; need deg1+1 >= " +
                                 std::to_string(requirement.deg1_plus_1) + ", deg2+1 >= " +
                                 std::to_string(requirement.deg2_plus_1) + ", but got deg1+1 = " +
                                 std::to_string(have_deg1_plus_1) + ", deg2+1 = " +
                                 std::to_string(have_deg2_plus_1));
    }
}

inline bool workflow_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_WORKFLOW");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
auto workflow_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = workflow_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        return fn();
    }
    const auto start = std::chrono::steady_clock::now();
    try {
        auto result = fn();
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        if (stderr_enabled) {
            std::cerr << "workflow_timing " << label << " " << elapsed.count() << "\n";
        }
        structured_protocol_profile::append_row("workflow_scope", label, "", elapsed.count(), "ok");
        return result;
    } catch (const std::exception& ex) {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        if (stderr_enabled) {
            std::cerr << "workflow_timing " << label << " " << elapsed.count() << "\n";
        }
        structured_protocol_profile::append_row(
            "workflow_scope", label, "", elapsed.count(), std::string("throw:") + ex.what());
        throw;
    } catch (...) {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        if (stderr_enabled) {
            std::cerr << "workflow_timing " << label << " " << elapsed.count() << "\n";
        }
        structured_protocol_profile::append_row("workflow_scope", label, "", elapsed.count(), "throw");
        throw;
    }
}

template <class Fn>
void workflow_profile_scope_void(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = workflow_profile_enabled();
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
            std::cerr << "workflow_timing " << label << " " << elapsed.count() << "\n";
        }
        structured_protocol_profile::append_row(
            "workflow_scope", label, "", elapsed.count(), std::string("throw:") + ex.what());
        throw;
    } catch (...) {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        if (stderr_enabled) {
            std::cerr << "workflow_timing " << label << " " << elapsed.count() << "\n";
        }
        structured_protocol_profile::append_row("workflow_scope", label, "", elapsed.count(), "throw");
        throw;
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "workflow_timing " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("workflow_scope", label, "", elapsed.count(), "ok");
}
