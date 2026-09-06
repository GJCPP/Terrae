#include "zkboost/primitives/histogram_ram.hpp"

#include "algebra_utils.hpp"
#include "lagrange_utils.hpp"
#include "matrix_utils.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr size_t kRamRangeProofBitBound = 48;

class ramRow {
public:
    size_t addr = 0;
    size_t time = 0;
    Fr read = Fr(0);
    Fr write = Fr(0);
    Fr delta = Fr(0);
};

size_t next_power_of_two(size_t n)
{
    if (n == 0) {
        return 1;
    }
    size_t out = 1;
    while (out < n) {
        if (out > std::numeric_limits<size_t>::max() / 2) {
            throw std::runtime_error("zkboostHistogramRamProtocol: trace domain size overflow");
        }
        out *= 2;
    }
    return out;
}

size_t checked_add(size_t left, size_t right, const char* context)
{
    if (left > std::numeric_limits<size_t>::max() - right) {
        throw std::runtime_error(std::string(context) + ": size overflow");
    }
    return left + right;
}

size_t checked_mul(size_t left, size_t right, const char* context)
{
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
        throw std::runtime_error(std::string(context) + ": size overflow");
    }
    return left * right;
}

size_t ceil_log2_at_least_one(size_t n)
{
    if (n <= 1) {
        return 1;
    }
    size_t bits = 0;
    size_t value = n - 1;
    while (value > 0) {
        ++bits;
        value >>= 1;
    }
    return bits;
}

size_t ram_fixed_range_bit_bound(size_t required_bit_bound)
{
    if (required_bit_bound <= kRamRangeProofBitBound) {
        return kRamRangeProofBitBound;
    }
    if (required_bit_bound < std::numeric_limits<size_t>::digits) {
        return std::numeric_limits<size_t>::digits - 1;
    }
    throw std::runtime_error("zkboostHistogramRamProtocol: range bound exceeds supported RAM proof width");
}

Fr power_of_two_field(size_t exponent)
{
    return pow_fr(fr_from_int(2), exponent);
}

void add_fixed_width_range_check(lessCheckProtocol& less_check,
                                 const virCommitment& values,
                                 size_t exact_bit_bound,
                                 size_t domain_size)
{
    const size_t proof_bit_bound = ram_fixed_range_bit_bound(exact_bit_bound);
    virCommitment checked_values = values;
    if (proof_bit_bound != exact_bit_bound) {
        // Scaling by 2^(proof-exact) lets a fixed-width range proof enforce
        // the original exact upper bound while keeping limb count constant.
        checked_values = checked_values * virCommitment(power_of_two_field(proof_bit_bound - exact_bit_bound));
    }
    less_check.add_check(checked_values, proof_bit_bound, domain_size);
}

void validate_shape(size_t query_domain_size, size_t bucket_count)
{
    if (query_domain_size == 0 || bucket_count == 0 ||
        !isPowerOfTwo(query_domain_size) || !isPowerOfTwo(bucket_count)) {
        throw std::runtime_error(
            "zkboostHistogramRamProtocol: query_domain_size and bucket_count must be non-zero powers of two");
    }
}

poly poly_from_evaluations(const std::vector<Fr>& evals)
{
    return poly_from_evals(evals);
}

std::vector<Fr> zero_evals(size_t n)
{
    return std::vector<Fr>(n, Fr(0));
}

poly segment_selector(size_t domain_size, size_t begin, size_t end)
{
    std::vector<Fr> evals(domain_size, Fr(0));
    for (size_t i = begin; i < end; ++i) {
        evals.at(i) = Fr(1);
    }
    return poly_from_evaluations(evals);
}

poly update_time_polynomial(size_t trace_domain_size, size_t query_domain_size)
{
    std::vector<Fr> evals = zero_evals(trace_domain_size);
    for (size_t i = 0; i < query_domain_size; ++i) {
        evals.at(i) = field_from_size_t(i + 1);
    }
    return poly_from_evaluations(evals);
}

poly bucket_addr_polynomial(size_t trace_domain_size, size_t begin, size_t bucket_count)
{
    std::vector<Fr> evals = zero_evals(trace_domain_size);
    for (size_t i = 0; i < bucket_count; ++i) {
        evals.at(begin + i) = field_from_size_t(i);
    }
    return poly_from_evaluations(evals);
}

poly constant_on_segment(size_t trace_domain_size, size_t begin, size_t end, const Fr& value)
{
    std::vector<Fr> evals = zero_evals(trace_domain_size);
    for (size_t i = begin; i < end; ++i) {
        evals.at(i) = value;
    }
    return poly_from_evaluations(evals);
}

Fr trace_shift_to_segment_start(size_t trace_domain_size, size_t segment_begin)
{
    Fr shift = get_root_of_unity(trace_domain_size);
    shift = pow_fr(shift, segment_begin);
    Fr inv;
    Fr::inv(inv, shift);
    return inv;
}

std::string static_commitment_cache_key(const kzgProtocol& kzg, const std::string& local_key)
{
    return transcript_representation(kzg.g1_power(0)) + ":" +
           transcript_representation(kzg.g1_power(1)) + ":zkboost_histogram_ram:" + local_key;
}

template <class BuildPolyFn>
kzgCommitment preprocessed_static_commitment(kzgProtocol& kzg,
                                             const std::string& local_key,
                                             BuildPolyFn&& build_poly)
{
    static std::map<std::string, Group1> group_cache;
    const std::string cache_key = static_commitment_cache_key(kzg, local_key);
    const std::shared_ptr<ProofSession> proof_session = kzg.proof_session();
    const bool verifier_mode = proof_session && proof_session->mode() == ProofMode::Verify;

    const auto cached = group_cache.find(cache_key);
    if (verifier_mode && cached != group_cache.end()) {
        return kzg.adopt_commitment(cached->second);
    }

    const poly witness = std::forward<BuildPolyFn>(build_poly)();
    Group1 group;
    if (cached == group_cache.end()) {
        group = kzg.commit_g1_group(witness);
        group_cache.emplace(cache_key, group);
    } else {
        group = cached->second;
    }

    if (verifier_mode) {
        return kzg.adopt_commitment(group);
    }

    kzgCommitment precomputed;
    precomputed.com = group;
    return kzg.commit(witness, precomputed);
}

std::string shape_static_key(const zkboostHistogramRamShape& shape, const std::string& label)
{
    return "q=" + std::to_string(shape.query_domain_size) +
           ":m=" + std::to_string(shape.bucket_count) +
           ":n=" + std::to_string(shape.trace_domain_size) +
           ":" + label;
}

template <class BuildPolyFn>
virCommitment static_virtual(kzgProtocol& kzg,
                             const zkboostHistogramRamShape& shape,
                             const std::string& label,
                             BuildPolyFn&& build_poly)
{
    return virCommitment(
        preprocessed_static_commitment(kzg, shape_static_key(shape, label), std::forward<BuildPolyFn>(build_poly)));
}

kzgCommitment commit_or_transport(kzgProtocol& kzg,
                                  const std::string& label,
                                  const poly& witness,
                                  size_t degree_bound)
{
    const std::shared_ptr<ProofSession> proof_session = kzg.proof_session();
    if (proof_session && proof_session->mode() == ProofMode::Verify) {
        const Group1 group = proof_session->transport_group1({});
        const kzgCommitment commitment = kzg.adopt_commitment(group);
        kzg.prove_degree(commitment, degree_bound);
        return commitment;
    }

    kzgCommitment commitment = kzg.commit(witness, degree_bound);
    if (proof_session && proof_session->mode() == ProofMode::Prove) {
        const Group1 group = proof_session->transport_group1([&] { return commitment.com; });
        if (group != commitment.com) {
            throw std::runtime_error(label + ": transported commitment mismatch");
        }
    }
    return commitment;
}

std::vector<ramRow> build_unsorted_rows(const std::vector<Fr>& index_evals,
                                        const std::vector<Fr>& value_evals,
                                        const std::vector<Fr>& sum_evals,
                                        const zkboostHistogramRamShape& shape)
{
    std::vector<ramRow> rows(shape.trace_domain_size);
    std::vector<Fr> memory(shape.bucket_count, Fr(0));

    for (size_t i = 0; i < shape.query_domain_size; ++i) {
        const size_t addr = canonical_size_t_from_fr(index_evals.at(i), "zkboostHistogramRamProtocol index");
        if (addr >= shape.bucket_count) {
            throw std::runtime_error("zkboostHistogramRamProtocol: index out of bucket range");
        }
        ramRow row;
        row.addr = addr;
        row.time = i + 1;
        row.read = memory.at(addr);
        row.delta = value_evals.at(i);
        row.write = row.read + row.delta;
        memory.at(addr) = row.write;
        rows.at(i) = row;
    }

    const size_t init_begin = shape.query_domain_size;
    for (size_t bucket = 0; bucket < shape.bucket_count; ++bucket) {
        ramRow row;
        row.addr = bucket;
        row.time = 0;
        rows.at(init_begin + bucket) = row;
    }

    const size_t terminal_begin = shape.query_domain_size + shape.bucket_count;
    for (size_t bucket = 0; bucket < shape.bucket_count; ++bucket) {
        ramRow row;
        row.addr = bucket;
        row.time = shape.query_domain_size + 1;
        row.read = sum_evals.at(bucket);
        row.write = sum_evals.at(bucket);
        rows.at(terminal_begin + bucket) = row;
    }

    return rows;
}

std::vector<ramRow> sorted_rows(std::vector<ramRow> rows)
{
    std::stable_sort(rows.begin(), rows.end(), [](const ramRow& left, const ramRow& right) {
        if (left.addr != right.addr) {
            return left.addr < right.addr;
        }
        return left.time < right.time;
    });
    return rows;
}

std::vector<Fr> column_addr(const std::vector<ramRow>& rows)
{
    std::vector<Fr> out;
    out.reserve(rows.size());
    for (const ramRow& row : rows) {
        out.push_back(field_from_size_t(row.addr));
    }
    return out;
}

std::vector<Fr> column_time(const std::vector<ramRow>& rows)
{
    std::vector<Fr> out;
    out.reserve(rows.size());
    for (const ramRow& row : rows) {
        out.push_back(field_from_size_t(row.time));
    }
    return out;
}

std::vector<Fr> column_read(const std::vector<ramRow>& rows)
{
    std::vector<Fr> out;
    out.reserve(rows.size());
    for (const ramRow& row : rows) {
        out.push_back(row.read);
    }
    return out;
}

std::vector<Fr> column_write(const std::vector<ramRow>& rows)
{
    std::vector<Fr> out;
    out.reserve(rows.size());
    for (const ramRow& row : rows) {
        out.push_back(row.write);
    }
    return out;
}

std::vector<Fr> column_delta(const std::vector<ramRow>& rows)
{
    std::vector<Fr> out;
    out.reserve(rows.size());
    for (const ramRow& row : rows) {
        out.push_back(row.delta);
    }
    return out;
}

std::vector<Fr> sorted_key_diffs(const std::vector<ramRow>& rows, size_t time_base)
{
    std::vector<Fr> out(rows.size(), Fr(0));
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
        const size_t current = rows.at(i).addr * time_base + rows.at(i).time;
        const size_t next = rows.at(i + 1).addr * time_base + rows.at(i + 1).time;
        out.at(i) = field_from_size_t(next - current);
    }
    return out;
}

std::vector<Fr> sorted_addr_gaps(const std::vector<ramRow>& rows)
{
    std::vector<Fr> out(rows.size(), Fr(0));
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
        if (rows.at(i).addr != rows.at(i + 1).addr) {
            out.at(i) = field_from_size_t(rows.at(i + 1).addr - rows.at(i).addr - 1);
        }
    }
    return out;
}

std::vector<Fr> sorted_same_addr_flags(const std::vector<ramRow>& rows)
{
    std::vector<Fr> out(rows.size(), Fr(0));
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
        if (rows.at(i).addr == rows.at(i + 1).addr) {
            out.at(i) = Fr(1);
        }
    }
    return out;
}

} // namespace

zkboostHistogramRamShape zkboost_histogram_ram_shape(size_t query_domain_size, size_t bucket_count)
{
    validate_shape(query_domain_size, bucket_count);
    zkboostHistogramRamShape shape;
    shape.query_domain_size = query_domain_size;
    shape.bucket_count = bucket_count;
    shape.trace_domain_size =
        next_power_of_two(checked_add(query_domain_size,
                                      checked_mul(2, bucket_count, "zkboostHistogramRamProtocol trace rows"),
                                      "zkboostHistogramRamProtocol trace rows"));
    return shape;
}

void zkboostHistogramRamProtocol::add_check(const virCommitment& indices,
                                            const virCommitment& values,
                                            const virCommitment& sums,
                                            size_t query_domain_size,
                                            size_t bucket_count)
{
    if (!is_setup) {
        throw std::runtime_error("zkboostHistogramRamProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.less_check == nullptr ||
        params_.shuffle_check == nullptr) {
        throw std::runtime_error("zkboostHistogramRamProtocol dependencies not setup");
    }

    const zkboostHistogramRamShape shape = zkboost_histogram_ram_shape(query_domain_size, bucket_count);
    const size_t n = shape.trace_domain_size;
    const size_t q = shape.query_domain_size;
    const size_t m = shape.bucket_count;
    const size_t init_begin = q;
    const size_t terminal_begin = checked_add(q, m, "zkboostHistogramRamProtocol terminal rows");
    const size_t terminal_end = checked_add(q,
                                            checked_mul(2, m, "zkboostHistogramRamProtocol terminal rows"),
                                            "zkboostHistogramRamProtocol terminal rows");
    const size_t time_base = checked_add(q, 2, "zkboostHistogramRamProtocol time base");
    const size_t key_range = checked_mul(m, time_base, "zkboostHistogramRamProtocol key range");

    zkboostHistogramRamRecord record;
    record.shape = shape;

    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    if (!(proof_session && proof_session->mode() == ProofMode::Verify)) {
        const std::vector<Fr> index_evals = params_.kzg->evaluations(indices, q);
        const std::vector<Fr> value_evals = params_.kzg->evaluations(values, q);
        const std::vector<Fr> sum_evals = params_.kzg->evaluations(sums, m);
        const std::vector<ramRow> unsorted = build_unsorted_rows(index_evals, value_evals, sum_evals, shape);
        const std::vector<ramRow> sorted = sorted_rows(unsorted);

        record.unsorted_read_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.unsorted_read",
                                poly_from_evaluations(column_read(unsorted)), n - 1);
        record.unsorted_write_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.unsorted_write",
                                poly_from_evaluations(column_write(unsorted)), n - 1);
        record.sorted_addr_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_addr",
                                poly_from_evaluations(column_addr(sorted)), n - 1);
        record.sorted_time_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_time",
                                poly_from_evaluations(column_time(sorted)), n - 1);
        record.sorted_read_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_read",
                                poly_from_evaluations(column_read(sorted)), n - 1);
        record.sorted_write_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_write",
                                poly_from_evaluations(column_write(sorted)), n - 1);
        record.sorted_delta_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_delta",
                                poly_from_evaluations(column_delta(sorted)), n - 1);
        record.sorted_key_diff_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_key_diff",
                                poly_from_evaluations(sorted_key_diffs(sorted, time_base)), n - 1);
        record.sorted_addr_gap_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_addr_gap",
                                poly_from_evaluations(sorted_addr_gaps(sorted)), n - 1);
        record.sorted_same_addr_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_same_addr",
                                poly_from_evaluations(sorted_same_addr_flags(sorted)), n - 1);
    } else {
        record.unsorted_read_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.unsorted_read", poly(), n - 1);
        record.unsorted_write_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.unsorted_write", poly(), n - 1);
        record.sorted_addr_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_addr", poly(), n - 1);
        record.sorted_time_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_time", poly(), n - 1);
        record.sorted_read_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_read", poly(), n - 1);
        record.sorted_write_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_write", poly(), n - 1);
        record.sorted_delta_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_delta", poly(), n - 1);
        record.sorted_key_diff_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_key_diff", poly(), n - 1);
        record.sorted_addr_gap_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_addr_gap", poly(), n - 1);
        record.sorted_same_addr_commitment =
            commit_or_transport(*params_.kzg, "zkboostHistogramRamProtocol.sorted_same_addr", poly(), n - 1);
    }

    records_.push_back(record);

    kzgProtocol& kzg = *params_.kzg;
    const virCommitment update_selector = static_virtual(kzg, shape, "update_selector", [&] {
        return segment_selector(n, 0, q);
    });
    const virCommitment init_selector = static_virtual(kzg, shape, "init_selector", [&] {
        return segment_selector(n, init_begin, terminal_begin);
    });
    const virCommitment terminal_selector = static_virtual(kzg, shape, "terminal_selector", [&] {
        return segment_selector(n, terminal_begin, terminal_end);
    });
    const virCommitment padding_selector = static_virtual(kzg, shape, "padding_selector", [&] {
        return segment_selector(n, terminal_end, n);
    });
    const virCommitment non_last_selector(virCommitment(Fr(1)) - subgroup_lagrange_expression(n, n - 1));

    const virCommitment lifted_indices = x_pow(indices, n / q);
    const virCommitment lifted_values = x_pow(values, n / q);
    const virCommitment shifted_sums =
        x_scale(x_pow(sums, n / m), trace_shift_to_segment_start(n, terminal_begin));

    const virCommitment init_addr = static_virtual(kzg, shape, "init_addr", [&] {
        return bucket_addr_polynomial(n, init_begin, m);
    });
    const virCommitment terminal_addr = static_virtual(kzg, shape, "terminal_addr", [&] {
        return bucket_addr_polynomial(n, terminal_begin, m);
    });
    const virCommitment update_time = static_virtual(kzg, shape, "update_time", [&] {
        return update_time_polynomial(n, q);
    });
    const virCommitment terminal_time = static_virtual(kzg, shape, "terminal_time", [&] {
        return constant_on_segment(n, terminal_begin, terminal_end, field_from_size_t(q + 1));
    });

    const virCommitment unsorted_addr = update_selector * lifted_indices + init_selector * init_addr +
                                        terminal_selector * terminal_addr;
    const virCommitment unsorted_time = update_time + terminal_time;
    const virCommitment unsorted_delta = update_selector * lifted_values;
    const virCommitment unsorted_read(record.unsorted_read_commitment);
    const virCommitment unsorted_write(record.unsorted_write_commitment);

    const virCommitment sorted_addr(record.sorted_addr_commitment);
    const virCommitment sorted_time(record.sorted_time_commitment);
    const virCommitment sorted_read(record.sorted_read_commitment);
    const virCommitment sorted_write(record.sorted_write_commitment);
    const virCommitment sorted_delta(record.sorted_delta_commitment);
    const virCommitment sorted_key_diff(record.sorted_key_diff_commitment);
    const virCommitment sorted_addr_gap(record.sorted_addr_gap_commitment);
    const virCommitment sorted_same(record.sorted_same_addr_commitment);

    virCommitment init_read_zero = init_selector * unsorted_read;
    virCommitment init_write_zero = init_selector * unsorted_write;
    virCommitment update_delta_zero = update_selector * (unsorted_write - unsorted_read - unsorted_delta);
    virCommitment terminal_read_zero = terminal_selector * (unsorted_read - shifted_sums);
    virCommitment terminal_write_zero = terminal_selector * (unsorted_write - shifted_sums);
    virCommitment padding_read_zero = padding_selector * unsorted_read;
    virCommitment padding_write_zero = padding_selector * unsorted_write;

    params_.zero_check->add_check(init_read_zero, n, "histogram_ram.init_read");
    params_.zero_check->add_check(init_write_zero, n, "histogram_ram.init_write");
    params_.zero_check->add_check(update_delta_zero, n, "histogram_ram.update_delta");
    params_.zero_check->add_check(terminal_read_zero, n, "histogram_ram.terminal_read");
    params_.zero_check->add_check(terminal_write_zero, n, "histogram_ram.terminal_write");
    params_.zero_check->add_check(padding_read_zero, n, "histogram_ram.padding_read");
    params_.zero_check->add_check(padding_write_zero, n, "histogram_ram.padding_write");

    params_.shuffle_check->add_check(
        {unsorted_addr, unsorted_time, unsorted_read, unsorted_write, unsorted_delta},
        {sorted_addr, sorted_time, sorted_read, sorted_write, sorted_delta},
        n);

    const Fr time_base_fr = field_from_size_t(time_base);
    const virCommitment sorted_key = scale_by_constant(sorted_addr, time_base_fr) + sorted_time;
    const virCommitment next_sorted_key = x_scale(sorted_key, get_root_of_unity(n));
    const virCommitment next_sorted_addr = x_scale(sorted_addr, get_root_of_unity(n));
    const virCommitment next_sorted_read = x_scale(sorted_read, get_root_of_unity(n));

    virCommitment key_diff_zero = non_last_selector * (next_sorted_key - sorted_key - sorted_key_diff);
    virCommitment last_key_diff_zero = subgroup_lagrange_expression(n, n - 1) * sorted_key_diff;
    virCommitment same_boolean_zero = sorted_same * (sorted_same - virCommitment(Fr(1)));
    virCommitment same_addr_zero = non_last_selector * sorted_same * (next_sorted_addr - sorted_addr);
    virCommitment addr_gap_zero = non_last_selector * (virCommitment(Fr(1)) - sorted_same) *
                                  (next_sorted_addr - sorted_addr - virCommitment(Fr(1)) - sorted_addr_gap);
    virCommitment memory_zero = non_last_selector * sorted_same * (next_sorted_read - sorted_write);

    params_.zero_check->add_check(key_diff_zero, n, "histogram_ram.key_diff");
    params_.zero_check->add_check(last_key_diff_zero, n, "histogram_ram.last_key_diff");
    params_.zero_check->add_check(same_boolean_zero, n, "histogram_ram.same_boolean");
    params_.zero_check->add_check(same_addr_zero, n, "histogram_ram.same_addr");
    params_.zero_check->add_check(addr_gap_zero, n, "histogram_ram.addr_gap");
    params_.zero_check->add_check(memory_zero, n, "histogram_ram.memory");

    add_fixed_width_range_check(*params_.less_check, sorted_addr, ceil_log2_at_least_one(m), n);
    add_fixed_width_range_check(*params_.less_check, sorted_time, ceil_log2_at_least_one(q + 2), n);
    add_fixed_width_range_check(*params_.less_check, sorted_key_diff, ceil_log2_at_least_one(key_range), n);
    add_fixed_width_range_check(*params_.less_check, sorted_addr_gap, ceil_log2_at_least_one(m), n);
}

bool zkboostHistogramRamProtocol::finalize()
{
    begin_finalize("zkboostHistogramRamProtocol");
    return true;
}
