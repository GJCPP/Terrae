#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "hist_agg.hpp"
#include "interleaved_batching.hpp"
#include "kzg.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
#include "sum_check.hpp"
#include "table_lookup.hpp"
#include "test_kzg_srs.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mcl::bn;

namespace {

poly poly_from_domain_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

poly histogram_poly_from_inputs(const std::vector<int>& indices, const std::vector<int>& values, size_t bucket_count)
{
    std::vector<Fr> histogram(bucket_count, Fr(0));
    for (size_t i = 0; i < indices.size(); ++i) {
        Fr::add(histogram[static_cast<size_t>(indices[i])], histogram[static_cast<size_t>(indices[i])], fr_from_int(values[i]));
    }
    return poly_from_evals(histogram);
}

poly perturb_evaluation(const poly& p, size_t domain_size, size_t index, const Fr& delta)
{
    std::vector<Fr> evals = p.evaluations_on_domain(domain_size);
    Fr::add(evals[index], evals[index], delta);
    return poly_from_evals(evals);
}

} // namespace

class HistAggProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(HistAggProtocolTest, HelperPowerPolynomialMatchesExpectedSequence)
{
    const Fr r = fr_from_int(3);
    const poly powers = hist_agg_power_polynomial(r, 4);
    const std::vector<Fr> expected = {
        fr_from_int(1),
        fr_from_int(3),
        fr_from_int(9),
        fr_from_int(27),
    };

    EXPECT_EQ(powers.evaluations_on_domain(4), expected);
}

TEST_F(HistAggProtocolTest, RejectsNonPowerOfTwoShapes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(1024, 1024));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol protocol(table_lookup);
    protocol.setup(kzg, zero, sum);

    const virCommitment values(fr_from_int(0));
    EXPECT_THROW(protocol.add_check(values, values, values, 3, 4), std::runtime_error);
    EXPECT_THROW(protocol.add_check(values, values, values, 4, 3), std::runtime_error);
}

TEST_F(HistAggProtocolTest, ValidHistogramAggregationProofFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol protocol(table_lookup);
    protocol.setup(kzg, zero, sum);

    const poly indices = poly_from_domain_ints({0, 2, 1, 2});
    const poly values = poly_from_domain_ints({5, 7, 11, 13});
    const poly histogram = histogram_poly_from_inputs({0, 2, 1, 2}, {5, 7, 11, 13}, 4);

    protocol.add_check(
        virCommitment(kzg.commit(indices)), virCommitment(kzg.commit(values)), virCommitment(kzg.commit(histogram)), 4, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(HistAggProtocolTest, WrongHistogramFailsAtProtocolCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol protocol(table_lookup);
    protocol.setup(kzg, zero, sum);

    const poly indices = poly_from_domain_ints({0, 2, 1, 2});
    const poly values = poly_from_domain_ints({5, 7, 11, 13});
    poly histogram = histogram_poly_from_inputs({0, 2, 1, 2}, {5, 7, 11, 13}, 4);
    histogram = perturb_evaluation(histogram, 4, 2, fr_from_int(1));

    protocol.add_check(
        virCommitment(kzg.commit(indices)), virCommitment(kzg.commit(values)), virCommitment(kzg.commit(histogram)), 4, 4);

    EXPECT_FALSE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(HistAggProtocolTest, OutOfRangeIndexFailsLookupReduction)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol protocol(table_lookup);
    protocol.setup(kzg, zero, sum);

    const poly indices = poly_from_domain_ints({0, 4, 1, 2});
    const poly values = poly_from_domain_ints({5, 7, 11, 13});
    const poly histogram = poly_from_domain_ints({5, 11, 13, 0});

    protocol.add_check(
        virCommitment(kzg.commit(indices)), virCommitment(kzg.commit(values)), virCommitment(kzg.commit(histogram)), 4, 4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_FALSE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(HistAggProtocolTest, SupportsBatchedChecks)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol protocol(table_lookup);
    protocol.setup(kzg, zero, sum);

    const poly first_indices = poly_from_domain_ints({0, 2, 1, 2});
    const poly first_values = poly_from_domain_ints({5, 7, 11, 13});
    const poly first_histogram = histogram_poly_from_inputs({0, 2, 1, 2}, {5, 7, 11, 13}, 4);

    const poly second_indices = poly_from_domain_ints({3, 3, 0, 1, 3, 2, 0, 0});
    const poly second_values = poly_from_domain_ints({17, 19, 23, 29, 31, 37, 41, 43});
    const poly second_histogram =
        histogram_poly_from_inputs({3, 3, 0, 1, 3, 2, 0, 0}, {17, 19, 23, 29, 31, 37, 41, 43}, 4);

    protocol.add_check(virCommitment(kzg.commit(first_indices)),
                       virCommitment(kzg.commit(first_values)),
                       virCommitment(kzg.commit(first_histogram)),
                       4,
                       4);
    protocol.add_check(virCommitment(kzg.commit(second_indices)),
                       virCommitment(kzg.commit(second_values)),
                       virCommitment(kzg.commit(second_histogram)),
                       8,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(HistAggProtocolTest, SupportsSharedIndexBatchAcrossMultipleChannels)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(4096, 4096));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol reduced_lookup;
    reduced_lookup.setup(kzg, zero, sum);

    tableLookupProtocol table_lookup(reduced_lookup);
    table_lookup.setup(kzg, zero, sum);

    histAggProtocol protocol(table_lookup);
    protocol.setup(kzg, zero, sum);

    const std::vector<int> shared_indices = {0, 2, 1, 2};
    const poly indices = poly_from_domain_ints(shared_indices);

    const poly values_a = poly_from_domain_ints({5, 7, 11, 13});
    const poly values_b = poly_from_domain_ints({1, 3, 5, 7});
    const poly histogram_a = histogram_poly_from_inputs(shared_indices, {5, 7, 11, 13}, 4);
    const poly histogram_b = histogram_poly_from_inputs(shared_indices, {1, 3, 5, 7}, 4);

    protocol.add_check(virCommitment(kzg.commit(indices)),
                       {virCommitment(kzg.commit(values_a)), virCommitment(kzg.commit(values_b))},
                       {virCommitment(kzg.commit(histogram_a)), virCommitment(kzg.commit(histogram_b))},
                       4,
                       4);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(table_lookup.finalize());
    EXPECT_TRUE(reduced_lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
