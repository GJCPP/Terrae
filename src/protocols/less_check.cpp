#include "less_check.hpp"

#include "algebra_utils.hpp"
#include "interleaved_batching.hpp"
#include "matrix_utils.hpp"
#include "polynomial.hpp"
#include "terrae_env.hpp"

#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t kLessCheckLimbBits = 16;

void validate_range_bit_bound(size_t bit_bound)
{
    const size_t max_safe_bit_bound = Fr::getBitSize() - 1;
    if (bit_bound > max_safe_bit_bound) {
        throw std::runtime_error("lessCheckProtocol: bit bound exceeds the safely representable integer range of Fr");
    }
    if (bit_bound >= std::numeric_limits<size_t>::digits) {
        throw std::runtime_error("lessCheckProtocol: bit bound is too large for size_t limb decomposition");
    }
}

size_t checked_range_domain_size(size_t bit_bound)
{
    validate_range_bit_bound(bit_bound);
    if (bit_bound > kLessCheckLimbBits) {
        throw std::runtime_error("lessCheckProtocol: direct range table bit bound exceeds limb width");
    }
    return size_t(1) << bit_bound;
}

Fr power_of_two_scale(size_t exponent)
{
    return pow_fr(fr_from_int(2), exponent);
}

poly range_table_polynomial(size_t bit_bound)
{
    const size_t domain_size = checked_range_domain_size(bit_bound);
    std::vector<Fr> evals;
    evals.reserve(domain_size);

    // Public table [0, 1, ..., 2^bit_bound - 1].
    Fr current = Fr(0);
    for (size_t i = 0; i < domain_size; ++i) {
        evals.push_back(current);
        Fr::add(current, current, Fr(1));
    }
    return poly_from_evals(evals);
}

std::string range_table_key(size_t bit_bound)
{
    return "less_check.range_table." + std::to_string(bit_bound);
}

std::string less_check_cache_key(const kzgProtocol& kzg, size_t bit_bound)
{
    return transcript_representation(kzg.g1_power(0)) + ":" + transcript_representation(kzg.g1_power(1)) + ":" +
           range_table_key(bit_bound);
}

void cache_range_table_group(const kzgProtocol& kzg, size_t bit_bound, const Group1& commitment)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = less_check_cache_key(kzg, bit_bound);
    const auto it = cache.find(key);
    if (it == cache.end()) {
        cache.emplace(key, commitment);
        return;
    }
    if (it->second != commitment) {
        throw std::runtime_error("lessCheckProtocol: inconsistent cached range-table commitment");
    }
}

Group1 preprocessed_range_table_group(const kzgProtocol& kzg, size_t bit_bound)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = less_check_cache_key(kzg, bit_bound);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const Group1 commitment = kzg.commit_g1_group(range_table_polynomial(bit_bound));
    cache.emplace(key, commitment);
    return commitment;
}

bool verifier_session(const std::shared_ptr<ProofSession>& proof_session)
{
    return proof_session && proof_session->mode() == ProofMode::Verify;
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

void transport_prover_commitment(const std::shared_ptr<ProofSession>& proof_session,
                                 const kzgCommitment& commitment,
                                 const char* context)
{
    if (!proof_session || proof_session->mode() != ProofMode::Prove) {
        return;
    }
    const Group1 transported = proof_session->transport_group1([&] { return commitment.com; });
    if (transported != commitment.com) {
        throw std::runtime_error(std::string(context) + ": transported commitment mismatch");
    }
}

virCommitment range_table_commitment(kzgProtocol& kzg,
                                     const std::shared_ptr<ProofSession>& proof_session,
                                     size_t bit_bound)
{
    const size_t table_domain_size = checked_range_domain_size(bit_bound);
    if (verifier_session(proof_session)) {
        const kzgCommitment commitment =
            kzg.adopt_public_commitment(range_table_key(bit_bound), preprocessed_range_table_group(kzg, bit_bound));
        if (!kzg.has_degree_bound(virCommitment(commitment), table_domain_size - 1)) {
            kzg.prove_degree(commitment, table_domain_size - 1);
        }
        return virCommitment(commitment);
    }

    const poly table_poly = range_table_polynomial(bit_bound);
    const Group1 table_group = preprocessed_range_table_group(kzg, bit_bound);
    const kzgCommitment commitment =
        kzg.commit_public_with_group(range_table_key(bit_bound), table_poly, table_group, table_domain_size - 1);
    cache_range_table_group(kzg, bit_bound, commitment.com);
    return virCommitment(commitment);
}

std::vector<Fr> limb_evaluations(const std::vector<Fr>& value_evals, size_t offset, size_t limb_bits)
{
    const uint64_t limb_mask = (uint64_t(1) << limb_bits) - 1;
    std::vector<Fr> limb_evals;
    limb_evals.reserve(value_evals.size());
    for (const Fr& value : value_evals) {
        uint64_t parsed = 0;
        try {
            parsed = value.getUint64();
        } catch (const std::exception&) {
            throw std::runtime_error("lessCheckProtocol limb decomposition: value is not a canonical field integer");
        }
        limb_evals.push_back(field_from_size_t((parsed >> offset) & limb_mask));
    }
    return limb_evals;
}

bool less_failure_debug_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_DEBUG_LESS_FAILURE");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

bool less_profile_enabled()
{
    const char* flag = terrae_getenv("TERRAE_PROFILE_LESS");
    return flag != nullptr && flag[0] != '\0' && std::string(flag) != "0" && std::string(flag) != "false";
}

template <class Fn>
auto less_profile_scope(const char* label, Fn&& fn)
{
    if (!less_profile_enabled()) {
        return fn();
    }
    const auto start = std::chrono::steady_clock::now();
    auto out = fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cerr << "less_profile " << label << " " << elapsed.count() << "\n";
    return out;
}

template <class Fn>
void less_profile_scope_void(const char* label, Fn&& fn)
{
    if (!less_profile_enabled()) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    std::cerr << "less_profile " << label << " " << elapsed.count() << "\n";
}

} // namespace

size_t less_check_limb_bits()
{
    return kLessCheckLimbBits;
}

std::vector<size_t> less_check_range_table_bit_bounds(size_t bit_bound)
{
    validate_range_bit_bound(bit_bound);
    if (bit_bound <= kLessCheckLimbBits) {
        return {bit_bound};
    }

    std::set<size_t> bounds;
    const size_t limb_count = (bit_bound + kLessCheckLimbBits - 1) / kLessCheckLimbBits;
    for (size_t limb = 0; limb < limb_count; ++limb) {
        const size_t offset = limb * kLessCheckLimbBits;
        bounds.insert(std::min(kLessCheckLimbBits, bit_bound - offset));
    }
    return std::vector<size_t>(bounds.begin(), bounds.end());
}

void preprocess_less_check_range_tables(kzgProtocol& kzg, const std::vector<size_t>& bit_bounds)
{
    std::set<size_t> table_bounds;
    for (const size_t bit_bound : bit_bounds) {
        const std::vector<size_t> required = less_check_range_table_bit_bounds(bit_bound);
        table_bounds.insert(required.begin(), required.end());
    }
    for (const size_t table_bit_bound : table_bounds) {
        (void)preprocessed_range_table_group(kzg, table_bit_bound);
    }
}

void lessCheckSrsEstimator::add_check(size_t bit_bound, size_t domain_size, size_t count)
{
    if (count == 0) {
        return;
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("lessCheckSrsEstimator: domain size must be a non-zero power of two");
    }
    validate_range_bit_bound(bit_bound);

    std::vector<size_t>& domains = domain_sizes_by_bit_bound_[bit_bound];
    domains.reserve(domains.size() + count);
    for (size_t i = 0; i < count; ++i) {
        domains.push_back(domain_size);
    }
}

size_t lessCheckSrsEstimator::required_domain_size() const
{
    if (domain_sizes_by_bit_bound_.empty()) {
        return 1;
    }

    size_t max_bit_bound = 0;
    for (const auto& [bit_bound, _] : domain_sizes_by_bit_bound_) {
        max_bit_bound = std::max(max_bit_bound, bit_bound);
    }

    if (max_bit_bound <= kLessCheckLimbBits) {
        std::vector<size_t> leaves;
        for (const auto& [_, domains] : domain_sizes_by_bit_bound_) {
            leaves.insert(leaves.end(), domains.begin(), domains.end());
        }
        return std::max(merged_interleaved_domain_size(leaves, "lessCheckSrsEstimator"),
                        checked_range_domain_size(max_bit_bound));
    }

    std::map<size_t, std::vector<size_t>> lookup_leaves_by_limb_bits;
    size_t largest_domain = 1;
    for (const auto& [bit_bound, domains] : domain_sizes_by_bit_bound_) {
        const size_t merged_values_domain =
            merged_interleaved_domain_size(domains, "lessCheckSrsEstimator");
        largest_domain = std::max(largest_domain, merged_values_domain);

        const size_t limb_count = (bit_bound + kLessCheckLimbBits - 1) / kLessCheckLimbBits;
        for (size_t limb = 0; limb < limb_count; ++limb) {
            const size_t offset = limb * kLessCheckLimbBits;
            const size_t limb_bits = std::min(kLessCheckLimbBits, bit_bound - offset);
            lookup_leaves_by_limb_bits[limb_bits].push_back(merged_values_domain);
        }
    }

    for (const auto& [limb_bits, domains] : lookup_leaves_by_limb_bits) {
        largest_domain = std::max(largest_domain,
                                  merged_interleaved_domain_size(domains, "lessCheckSrsEstimator"));
        largest_domain = std::max(largest_domain, checked_range_domain_size(limb_bits));
    }
    return largest_domain;
}

void lessCheckProtocol::add_check(const virCommitment& values, size_t bit_bound, size_t domain_size)
{
    proof_cost::record_relation("range_check", domain_size);
    if (!is_setup) {
        throw std::runtime_error("lessCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.lookup_check == nullptr) {
        throw std::runtime_error("lessCheckProtocol subprotocols not setup");
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("lessCheckProtocol: domain size must be a non-zero power of two");
    }
    validate_range_bit_bound(bit_bound);

    lessCheckRecord record;
    record.bit_bound = bit_bound;
    record.domain_size = domain_size;
    record.uses_mask = false;
    values_.push_back(values);
    masks_.push_back(virCommitment(Fr(1)));
    checks_.push_back(record);
}

void lessCheckProtocol::add_masked_check(const virCommitment& values,
                                         const virCommitment& mask,
                                         size_t bit_bound,
                                         size_t domain_size)
{
    proof_cost::record_relation("range_check", domain_size);
    if (!is_setup) {
        throw std::runtime_error("lessCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.lookup_check == nullptr) {
        throw std::runtime_error("lessCheckProtocol subprotocols not setup");
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("lessCheckProtocol: domain size must be a non-zero power of two");
    }
    validate_range_bit_bound(bit_bound);

    lessCheckRecord record;
    record.bit_bound = bit_bound;
    record.domain_size = domain_size;
    record.uses_mask = true;
    values_.push_back(values);
    masks_.push_back(mask);
    checks_.push_back(record);
}

bool lessCheckProtocol::finalize()
{
    begin_finalize("lessCheckProtocol");

    if (checks_.empty()) {
        return true;
    }

    size_t max_bit_bound = 0;
    for (const lessCheckRecord& check : checks_) {
        validate_range_bit_bound(check.bit_bound);
        max_bit_bound = std::max(max_bit_bound, check.bit_bound);
    }

    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    const bool use_limb_decomposition = max_bit_bound > kLessCheckLimbBits;

    if (use_limb_decomposition && params_.zero_check == nullptr) {
        throw std::runtime_error("lessCheckProtocol: limb range checks require zeroProtocol setup");
    }

    if (!use_limb_decomposition) {
        const size_t table_domain_size = checked_range_domain_size(max_bit_bound);
        virCommitment table_commitment = less_profile_scope("direct.range_table_commitment", [&] {
            return range_table_commitment(*params_.kzg, proof_session, max_bit_bound);
        });

        std::vector<InterleavedBatchNode> leaves;
        leaves.reserve(checks_.size());

        less_profile_scope_void("direct.build_leaves", [&] {
            for (size_t i = 0; i < checks_.size(); ++i) {
                const lessCheckRecord& check = checks_[i];
                // Scale smaller ranges into the largest shared table so every batched
                // leaf can be discharged by one lookup.
                virCommitment scaled_values = check.uses_mask ? values_[i] * masks_[i] : values_[i];
                if (check.bit_bound != max_bit_bound) {
                    scaled_values = scaled_values * virCommitment(power_of_two_scale(max_bit_bound - check.bit_bound));
                }

                if (less_failure_debug_enabled()) {
                    const Fr upper_bound = field_from_size_t(table_domain_size);
                    const std::vector<Fr> evals = params_.kzg->evaluations(
                        scaled_values, check.domain_size, kzgProtocol::EvaluationMode::PreferEval);
                    std::cerr << "less_debug check"
                              << " check_index=" << i
                              << " bit_bound=" << check.bit_bound
                              << " max_bit_bound=" << max_bit_bound
                              << " domain_size=" << check.domain_size
                              << " uses_mask=" << (check.uses_mask ? 1 : 0)
                              << "\n";
                    size_t missing_count = 0;
                    std::vector<std::pair<size_t, Fr>> first_missing;
                    for (size_t j = 0; j < evals.size(); ++j) {
                        if (evals[j].isLess(upper_bound)) {
                            continue;
                        }
                        ++missing_count;
                        if (first_missing.size() < 8) {
                            first_missing.emplace_back(j, evals[j]);
                        }
                    }
                    if (missing_count != 0) {
                        std::cerr << "less_debug out_of_range"
                                  << " check_index=" << i
                                  << " bit_bound=" << check.bit_bound
                                  << " max_bit_bound=" << max_bit_bound
                                  << " domain_size=" << check.domain_size
                                  << " uses_mask=" << (check.uses_mask ? 1 : 0)
                                  << " missing_count=" << missing_count;
                        for (const auto& [index, value] : first_missing) {
                            std::cerr << " sample=" << index << ":" << value.getStr(10);
                        }
                        std::cerr << "\n";
                    }
                }

                leaves.push_back(InterleavedBatchNode{{scaled_values}, check.domain_size});
            }
        });

        const InterleavedBatchNode merged = less_profile_scope("direct.merge_leaves", [&] {
            return merge_interleaved_nodes(leaves, "lessCheckProtocol");
        });
        // After scaling, all range checks become one lookup against the single
        // public integer table for `max_bit_bound`.
        less_profile_scope_void("direct.lookup_add_check", [&] {
            params_.lookup_check->add_check(merged.commitments[0],
                                            table_commitment,
                                            merged.domain_size,
                                            table_domain_size,
                                            "less_check.max_bit_bound=" + std::to_string(max_bit_bound));
        });
        return true;
    }

    std::map<size_t, std::vector<InterleavedBatchNode>> checked_value_leaves_by_bit_bound;
    less_profile_scope_void("limb.group_checked_values", [&] {
        for (size_t i = 0; i < checks_.size(); ++i) {
            const lessCheckRecord& check = checks_[i];
            const virCommitment checked_values = check.uses_mask ? values_[i] * masks_[i] : values_[i];
            checked_value_leaves_by_bit_bound[check.bit_bound].push_back(
                InterleavedBatchNode{{checked_values}, check.domain_size});
        }
    });

    std::map<size_t, std::vector<InterleavedBatchNode>> lookup_leaves_by_limb_bits;
    for (auto& [bit_bound, checked_value_leaves] : checked_value_leaves_by_bit_bound) {
        const InterleavedBatchNode merged_values = less_profile_scope("limb.merge_checked_values", [&] {
            return merge_interleaved_nodes(checked_value_leaves, "lessCheckProtocol");
        });
        const virCommitment checked_values = merged_values.commitments[0];
        const size_t limb_count = (bit_bound + kLessCheckLimbBits - 1) / kLessCheckLimbBits;
        std::vector<virCommitment> limbs;
        limbs.reserve(limb_count);

        if (verifier_session(proof_session)) {
            less_profile_scope_void("limb.adopt_transported_commitments", [&] {
                for (size_t limb = 0; limb < limb_count; ++limb) {
                    const size_t offset = limb * kLessCheckLimbBits;
                    const size_t limb_bits = std::min(kLessCheckLimbBits, bit_bound - offset);
                    const kzgCommitment limb_commitment =
                        adopt_transported_commitment(*params_.kzg, proof_session, merged_values.domain_size - 1);
                    const virCommitment limb_virtual(limb_commitment);
                    limbs.push_back(limb_virtual);
                    lookup_leaves_by_limb_bits[limb_bits].push_back(
                        InterleavedBatchNode{{limb_virtual}, merged_values.domain_size});
                }
            });
        } else {
            const std::vector<Fr> checked_evals = less_profile_scope("limb.evaluate_merged_values", [&] {
                return params_.kzg->evaluations(checked_values,
                                                merged_values.domain_size,
                                                kzgProtocol::EvaluationMode::PreferEval);
            });
            for (size_t limb = 0; limb < limb_count; ++limb) {
                const size_t offset = limb * kLessCheckLimbBits;
                const size_t limb_bits = std::min(kLessCheckLimbBits, bit_bound - offset);
                const std::vector<Fr> limb_evals = less_profile_scope("limb.decompose_values", [&] {
                    return limb_evaluations(checked_evals, offset, limb_bits);
                });
                const kzgCommitment limb_commitment = less_profile_scope("limb.commit", [&] {
                    return params_.kzg->commit(poly_from_evals(limb_evals), merged_values.domain_size - 1);
                });
                transport_prover_commitment(proof_session, limb_commitment, "lessCheckProtocol limb");
                const virCommitment limb_virtual(limb_commitment);
                limbs.push_back(limb_virtual);
                lookup_leaves_by_limb_bits[limb_bits].push_back(
                    InterleavedBatchNode{{limb_virtual}, merged_values.domain_size});
            }
        }

        virCommitment reconstructed(Fr(0));
        for (size_t limb = 0; limb < limbs.size(); ++limb) {
            const size_t offset = limb * kLessCheckLimbBits;
            reconstructed = reconstructed + limbs[limb] * virCommitment(power_of_two_scale(offset));
        }
        virCommitment recomposition = checked_values - reconstructed;
        less_profile_scope_void("limb.add_recomposition_zero", [&] {
            params_.zero_check->add_check(recomposition, merged_values.domain_size, "less_check.limb_recomposition");
        });
    }

    for (auto& [limb_bits, leaves] : lookup_leaves_by_limb_bits) {
        const size_t table_domain_size = checked_range_domain_size(limb_bits);
        const virCommitment table_commitment = less_profile_scope("limb.range_table_commitment", [&] {
            return range_table_commitment(*params_.kzg, proof_session, limb_bits);
        });
        const InterleavedBatchNode merged = less_profile_scope("limb.merge_lookup_leaves", [&] {
            return merge_interleaved_nodes(leaves, "lessCheckProtocol");
        });
        less_profile_scope_void("limb.lookup_add_check", [&] {
            params_.lookup_check->add_check(merged.commitments[0],
                                            table_commitment,
                                            merged.domain_size,
                                            table_domain_size,
                                            "less_check.limb_bits=" + std::to_string(limb_bits));
        });
    }
    return true;
}
