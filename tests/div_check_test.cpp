#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "div_check.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
#include "sum_check.hpp"
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

class TestableZeroProtocol : public zeroProtocol {
public:
    size_t tracked_check_count() const
    {
        return params.domain_sizes.size();
    }

    const std::vector<size_t>& tracked_domain_sizes() const
    {
        return params.domain_sizes;
    }
};

} // namespace

class DivCheckProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(DivCheckProtocolTest, AddCheckRejectsInvalidParameters)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    divCheckProtocol div;
    div.setup(kzg, zero, less);

    const virCommitment values(fr_from_int(0));
    EXPECT_THROW(div.add_check(values, 4, 5, 4), std::runtime_error);
    EXPECT_THROW(div.add_check(values, 4, 2, 3), std::runtime_error);
    EXPECT_THROW(div.add_check(values, Fr::getBitSize(), 1, 4), std::runtime_error);
}

TEST_F(DivCheckProtocolTest, ValidChecksPassForPublicPowerOfTwoDivisors)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    TestableZeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    divCheckProtocol div;
    div.setup(kzg, zero, less);

    div.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 4, 8, 12}))), 4, 2, 4);
    div.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 2, 4, 6}))), 3, 1, 4);
    div.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0}))), 0, 0, 1);

    EXPECT_TRUE(div.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(DivCheckProtocolTest, BatchesChecksWithSameDivisorPowerThroughInterleaving)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    TestableZeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    divCheckProtocol div;
    div.setup(kzg, zero, less);

    div.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 2}))), 2, 1, 2);
    div.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 2, 4, 6}))), 3, 1, 4);
    div.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 4, 8, 12}))), 4, 2, 4);

    EXPECT_TRUE(div.finalize());
    ASSERT_EQ(zero.tracked_check_count(), 2u);
    EXPECT_EQ(zero.tracked_domain_sizes()[0], 8u);
    EXPECT_EQ(zero.tracked_domain_sizes()[1], 4u);

    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(DivCheckProtocolTest, NonDivisibleValueFailsRangeReducedQuotientCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    divCheckProtocol div;
    div.setup(kzg, zero, less);

    div.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 4, 6, 12}))), 4, 2, 4);

    EXPECT_TRUE(div.finalize());
    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(DivCheckProtocolTest, ProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(256, 256));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_lookup;
    prover_lookup.setup(prover_kzg, prover_zero, prover_sum);

    lessCheckProtocol prover_less;
    prover_less.setup(prover_kzg, prover_lookup);

    divCheckProtocol prover_div;
    prover_div.setup(prover_kzg, prover_zero, prover_less);

    const poly divisible_by_four = poly_from_domain_ints({0, 4, 8, 12});
    const poly divisible_by_two = poly_from_domain_ints({0, 2, 4, 6});
    const kzgCommitment prover_divisible_by_four_commitment = prover_kzg.commit(divisible_by_four);
    const kzgCommitment prover_divisible_by_two_commitment = prover_kzg.commit(divisible_by_two);

    prover_div.add_check(virCommitment(prover_divisible_by_four_commitment), 4, 2, 4);
    prover_div.add_check(virCommitment(prover_divisible_by_two_commitment), 3, 1, 4);

    EXPECT_TRUE(prover_div.finalize());
    EXPECT_TRUE(prover_less.finalize());
    EXPECT_TRUE(prover_lookup.finalize());
    EXPECT_TRUE(prover_sum.finalize());
    EXPECT_TRUE(prover_zero.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    zeroProtocol verifier_zero;
    verifier_zero.setup(verifier_kzg);

    sumProtocol verifier_sum;
    verifier_sum.setup(verifier_kzg, verifier_zero);

    lookupProtocol verifier_lookup;
    verifier_lookup.setup(verifier_kzg, verifier_zero, verifier_sum);

    lessCheckProtocol verifier_less;
    verifier_less.setup(verifier_kzg, verifier_lookup);

    divCheckProtocol verifier_div;
    verifier_div.setup(verifier_kzg, verifier_zero, verifier_less);

    const kzgCommitment verifier_divisible_by_four_commitment =
        verifier_kzg.adopt_commitment(prover_divisible_by_four_commitment.com);
    const kzgCommitment verifier_divisible_by_two_commitment =
        verifier_kzg.adopt_commitment(prover_divisible_by_two_commitment.com);

    verifier_div.add_check(virCommitment(verifier_divisible_by_four_commitment), 4, 2, 4);
    verifier_div.add_check(virCommitment(verifier_divisible_by_two_commitment), 3, 1, 4);

    EXPECT_TRUE(verifier_div.finalize());
    EXPECT_TRUE(verifier_less.finalize());
    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}
