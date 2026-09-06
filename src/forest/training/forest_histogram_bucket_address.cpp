#include "forest_histogram_bucket_address.hpp"

#include "matrix_utils.hpp"

#include <stdexcept>

namespace {

poly feature_offset_matrix_polynomial(size_t query_row_count, size_t feature_count, size_t bin_count)
{
    std::vector<Fr> evals;
    evals.reserve(query_row_count * feature_count);
    for (size_t row = 0; row < query_row_count; ++row) {
        (void)row;
        for (size_t feature = 0; feature < feature_count; ++feature) {
            evals.push_back(field_from_size_t(feature * bin_count));
        }
    }
    return poly_from_evals(evals);
}

void validate_histogram_bucket_address_parameters(size_t query_row_count,
                                                  size_t feature_count,
                                                  size_t bin_count,
                                                  size_t node_count)
{
    validate_matrix_shape(query_row_count, feature_count, "forestHistogramBucketAddressProtocol");
    if (!isPowerOfTwo(bin_count) || !isPowerOfTwo(node_count)) {
        throw std::runtime_error("forestHistogramBucketAddressProtocol: bin_count and node_count must be powers of two");
    }
}

} // namespace

void forestHistogramBucketAddressProtocol::add_check(const forestHistogramBucketAddressWitness& witness,
                                                     size_t query_row_count,
                                                     size_t feature_count,
                                                     size_t bin_count,
                                                     size_t node_count)
{
    if (!is_setup) {
        throw std::runtime_error("forestHistogramBucketAddressProtocol not setup");
    }
    if (params_.zero_check == nullptr || params_.column_replication == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("forestHistogramBucketAddressProtocol dependencies not setup");
    }

    validate_histogram_bucket_address_parameters(query_row_count, feature_count, bin_count, node_count);

    witnesses_.push_back(witness);
    checks_.push_back({query_row_count, feature_count, bin_count, node_count, false});
}

void forestHistogramBucketAddressProtocol::add_check_with_certified_indices(
    const forestHistogramBucketAddressWitness& witness,
    size_t query_row_count,
    size_t feature_count,
    size_t bin_count,
    size_t node_count)
{
    add_check(witness, query_row_count, feature_count, bin_count, node_count);
    checks_.back().indices_certified = true;
}

bool forestHistogramBucketAddressProtocol::finalize()
{
    begin_finalize("forestHistogramBucketAddressProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestHistogramBucketAddressCheckRecord& record = checks_[i];
        const forestHistogramBucketAddressWitness& witness = witnesses_[i];
        const size_t matrix_domain_size = record.query_row_count * record.feature_count;
        const virCommitment feature_offsets(
            feature_offset_matrix_polynomial(record.query_row_count, record.feature_count, record.bin_count));
        const virCommitment leaf_scale(field_from_size_t(record.feature_count * record.bin_count));

        params_.column_replication->add_check(
            witness.leaf_indices, witness.lifted_leaf_indices, record.query_row_count, record.feature_count);

        virCommitment address_zero = witness.flattened_bucket_indices -
                                     witness.lifted_leaf_indices * leaf_scale - feature_offsets - witness.lifted_inputs;
        params_.zero_check->add_check(address_zero, matrix_domain_size);

        if (!record.indices_certified) {
            params_.less_check->add_check(
                witness.leaf_indices, log2N(record.node_count), record.query_row_count);
            params_.less_check->add_check(
                witness.flattened_bucket_indices,
                log2N(record.node_count) + log2N(record.feature_count) + log2N(record.bin_count),
                matrix_domain_size);
        }
        params_.less_check->add_check(witness.lifted_inputs, log2N(record.bin_count), matrix_domain_size);
    }

    return true;
}
