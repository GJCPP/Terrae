#include "interleaved_batching.hpp"

#include "algebra_utils.hpp"
#include "polynomial.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string protocol_error(std::string_view protocol_name, std::string_view message)
{
    return std::string(protocol_name) + ": " + std::string(message);
}

struct InterleavedBatchBucket {
    std::vector<InterleavedBatchNode> nodes;
    size_t head = 0;
};

size_t bucket_index_for_domain_size(size_t domain_size, std::string_view protocol_name)
{
    if (!isPowerOfTwo(domain_size)) {
        throw std::runtime_error(protocol_error(protocol_name, "domain size must be a power of two"));
    }

    return log2N(domain_size);
}

void ensure_bucket(std::vector<InterleavedBatchBucket>& buckets, size_t index)
{
    if (index >= buckets.size()) {
        buckets.resize(index + 1);
    }
}

size_t active_bucket_size(const InterleavedBatchBucket& bucket)
{
    return bucket.nodes.size() - bucket.head;
}

InterleavedBatchNode pop_bucket_front(InterleavedBatchBucket& bucket, std::string_view protocol_name)
{
    if (active_bucket_size(bucket) == 0) {
        throw std::runtime_error(protocol_error(protocol_name, "attempted to pop from an empty bucket"));
    }

    InterleavedBatchNode node = bucket.nodes[bucket.head];
    ++bucket.head;
    if (bucket.head == bucket.nodes.size()) {
        bucket.nodes.clear();
        bucket.head = 0;
    }
    return node;
}

virCommitment lift_to_domain(const virCommitment& values,
                             size_t current_domain_size,
                             size_t target_domain_size,
                             std::string_view protocol_name)
{
    if (current_domain_size == target_domain_size) {
        return values;
    }
    if (current_domain_size == 0 || target_domain_size == 0 || target_domain_size % current_domain_size != 0) {
        throw std::runtime_error(protocol_error(protocol_name, "invalid domain lift"));
    }
    const size_t factor = target_domain_size / current_domain_size;
    if (!isPowerOfTwo(factor)) {
        throw std::runtime_error(protocol_error(protocol_name, "domain lift factor must be a power of two"));
    }
    return x_pow(values, factor);
}

InterleavedBatchNode merge_nodes(const InterleavedBatchNode& left,
                                 const InterleavedBatchNode& right,
                                 std::string_view protocol_name)
{
    if (left.commitments.size() != right.commitments.size()) {
        throw std::runtime_error(protocol_error(protocol_name, "merged nodes must carry the same commitment arity"));
    }

    const size_t pair_domain_size = std::max(left.domain_size, right.domain_size);
    if (pair_domain_size > std::numeric_limits<size_t>::max() / 2) {
        throw std::runtime_error(protocol_error(protocol_name, "merged domain size overflow"));
    }

    InterleavedBatchNode merged;
    merged.commitments.reserve(left.commitments.size());
    merged.domain_size = 2 * pair_domain_size;
    for (size_t i = 0; i < left.commitments.size(); ++i) {
        const virCommitment lifted_left =
            lift_to_domain(left.commitments[i], left.domain_size, pair_domain_size, protocol_name);
        const virCommitment lifted_right =
            lift_to_domain(right.commitments[i], right.domain_size, pair_domain_size, protocol_name);
        merged.commitments.push_back(interleave_commitments(lifted_left, lifted_right, pair_domain_size, protocol_name));
    }
    return merged;
}

void push_with_bucket_carry(std::vector<InterleavedBatchBucket>& buckets,
                            size_t index,
                            InterleavedBatchNode node,
                            std::string_view protocol_name)
{
    ensure_bucket(buckets, index);
    buckets[index].nodes.push_back(std::move(node));

    size_t carry_index = index;
    while (active_bucket_size(buckets[carry_index]) >= 2) {
        const InterleavedBatchNode left = pop_bucket_front(buckets[carry_index], protocol_name);
        const InterleavedBatchNode right = pop_bucket_front(buckets[carry_index], protocol_name);
        InterleavedBatchNode merged = merge_nodes(left, right, protocol_name);
        ++carry_index;
        ensure_bucket(buckets, carry_index);
        buckets[carry_index].nodes.push_back(std::move(merged));
    }
}

size_t active_node_count(const std::vector<InterleavedBatchBucket>& buckets)
{
    size_t total = 0;
    for (const InterleavedBatchBucket& bucket : buckets) {
        total += active_bucket_size(bucket);
    }
    return total;
}

} // namespace

virCommitment interleave_commitments(const virCommitment& even_values,
                                     const virCommitment& odd_values,
                                     size_t domain_size,
                                     std::string_view protocol_name)
{
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error(protocol_error(protocol_name, "interleave domain must be a non-zero power of two"));
    }

    virCommitment out;
    out.type = virCommitment::Type::INTERLEAVE;
    out.left = std::make_shared<const virCommitment>(even_values);
    out.right = std::make_shared<const virCommitment>(odd_values);
    out.exponent = domain_size;
    return out;
}

InterleavedBatchNode merge_interleaved_nodes(const std::vector<InterleavedBatchNode>& leaves,
                                             std::string_view protocol_name)
{
    if (leaves.empty()) {
        throw std::runtime_error(protocol_error(protocol_name, "no merged query produced"));
    }

    std::vector<InterleavedBatchBucket> buckets;
    for (const InterleavedBatchNode& leaf : leaves) {
        if (leaf.commitments.empty()) {
            throw std::runtime_error(protocol_error(protocol_name, "node must carry at least one commitment"));
        }
        const size_t bucket_index = bucket_index_for_domain_size(leaf.domain_size, protocol_name);
        ensure_bucket(buckets, bucket_index);
        buckets[bucket_index].nodes.push_back(leaf);
    }

    for (size_t i = 0; i < buckets.size(); ++i) {
        while (active_bucket_size(buckets[i]) >= 2) {
            const InterleavedBatchNode left = pop_bucket_front(buckets[i], protocol_name);
            const InterleavedBatchNode right = pop_bucket_front(buckets[i], protocol_name);
            push_with_bucket_carry(buckets, i + 1, merge_nodes(left, right, protocol_name), protocol_name);
        }
    }

    while (active_node_count(buckets) > 1) {
        size_t first_bucket = buckets.size();
        size_t second_bucket = buckets.size();

        for (size_t i = 0; i < buckets.size(); ++i) {
            if (active_bucket_size(buckets[i]) == 0) {
                continue;
            }
            if (first_bucket == buckets.size()) {
                first_bucket = i;
                continue;
            }
            second_bucket = i;
            break;
        }

        if (second_bucket == buckets.size()) {
            throw std::runtime_error(protocol_error(protocol_name, "failed to find two nodes to merge"));
        }

        const InterleavedBatchNode left = pop_bucket_front(buckets[first_bucket], protocol_name);
        const InterleavedBatchNode right = pop_bucket_front(buckets[second_bucket], protocol_name);
        push_with_bucket_carry(buckets, second_bucket + 1, merge_nodes(left, right, protocol_name), protocol_name);
    }

    for (const InterleavedBatchBucket& bucket : buckets) {
        if (active_bucket_size(bucket) == 0) {
            continue;
        }
        size_t source_domain_size_sum = 0;
        for (const InterleavedBatchNode& leaf : leaves) {
            source_domain_size_sum += leaf.domain_size;
        }
        return bucket.nodes[bucket.head];
    }

    throw std::runtime_error(protocol_error(protocol_name, "no merged query produced"));
}

size_t merged_interleaved_domain_size(const std::vector<size_t>& leaf_domain_sizes,
                                      std::string_view protocol_name)
{
    if (leaf_domain_sizes.empty()) {
        throw std::runtime_error(protocol_error(protocol_name, "no merged query produced"));
    }

    size_t total = 0;
    for (const size_t domain_size : leaf_domain_sizes) {
        if (!isPowerOfTwo(domain_size) || domain_size == 0) {
            throw std::runtime_error(protocol_error(protocol_name, "domain size must be a non-zero power of two"));
        }
        if (domain_size > std::numeric_limits<size_t>::max() - total) {
            throw std::runtime_error(protocol_error(protocol_name, "merged domain size overflow"));
        }
        total += domain_size;
    }

    size_t out = 1;
    while (out < total) {
        if (out > std::numeric_limits<size_t>::max() / 2) {
            throw std::runtime_error(protocol_error(protocol_name, "merged domain size overflow"));
        }
        out *= 2;
    }
    return out;
}
