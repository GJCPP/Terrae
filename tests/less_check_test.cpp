#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
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

poly poly_from_domain_size_ts(const std::vector<size_t>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (size_t value : values) {
        Fr encoded;
        encoded.setStr(std::to_string(value), 10);
        evals.push_back(encoded);
    }
    return poly_from_evals(evals);
}

class TestableLookupProtocol : public lookupProtocol {
public:
    const lookupProver& prover_ref() const { return prover; }
    const lookupVerifier& verifier_ref() const { return verifier; }
};

} // namespace

class LessCheckProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(LessCheckProtocolTest, AddCheckRejectsInvalidDomainSize)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    const virCommitment values(fr_from_int(0));
    const virCommitment mask(fr_from_int(1));
    EXPECT_THROW(less.add_check(values, 3, 3), std::runtime_error);
    EXPECT_THROW(less.add_check(values, 3, 0), std::runtime_error);
    EXPECT_THROW(less.add_check(values, Fr::getBitSize(), 2), std::runtime_error);
    EXPECT_THROW(less.add_masked_check(values, mask, 3, 3), std::runtime_error);
    EXPECT_THROW(less.add_masked_check(values, mask, 3, 0), std::runtime_error);
    EXPECT_THROW(less.add_masked_check(values, mask, Fr::getBitSize(), 2), std::runtime_error);
}

TEST_F(LessCheckProtocolTest, SupportsSingletonDomainChecks)
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

    const poly one_bit_value = poly_from_domain_ints({1});
    const poly two_bit_value = poly_from_domain_ints({2});

    less.add_check(virCommitment(kzg.commit(one_bit_value)), 1, 1);
    less.add_check(virCommitment(kzg.commit(two_bit_value)), 2, 1);

    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LessCheckProtocolTest, ValidChecksBatchAcrossDifferentBitBounds)
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

    const poly three_bit_values = poly_from_domain_ints({0, 3, 5, 7});
    const poly two_bit_values = poly_from_domain_ints({0, 1, 2, 3});

    less.add_check(virCommitment(kzg.commit(three_bit_values)), 3, 4);
    less.add_check(virCommitment(kzg.commit(two_bit_values)), 2, 4);

    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LessCheckProtocolTest, MaskedChecksIgnoreInactiveOutOfRangeRows)
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

    const poly values = poly_from_domain_ints({0, 1, 5, 3});
    const poly mask = poly_from_domain_ints({1, 1, 0, 1});

    less.add_masked_check(virCommitment(kzg.commit(values)), virCommitment(kzg.commit(mask)), 2, 4);

    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LessCheckProtocolTest, MaskedChecksFailWhenActiveRowsExceedBound)
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

    const poly values = poly_from_domain_ints({0, 1, 4, 3});
    const poly mask = poly_from_domain_ints({1, 1, 1, 1});

    less.add_masked_check(virCommitment(kzg.commit(values)), virCommitment(kzg.commit(mask)), 2, 4);

    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LessCheckProtocolTest, OutOfRangeValueFailsLookupReduction)
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

    const poly valid_three_bit_values = poly_from_domain_ints({0, 3, 5, 7});
    const poly invalid_two_bit_values = poly_from_domain_ints({0, 1, 2, 4});

    less.add_check(virCommitment(kzg.commit(valid_three_bit_values)), 3, 4);
    less.add_check(virCommitment(kzg.commit(invalid_two_bit_values)), 2, 4);

    EXPECT_TRUE(less.finalize());
    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LessCheckProtocolTest, ProveAndVerifyWithSerializedProof)
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

    const poly three_bit_values = poly_from_domain_ints({0, 3, 5, 7});
    const poly two_bit_values = poly_from_domain_ints({0, 1, 2, 3});
    const kzgCommitment prover_three_bit_commitment = prover_kzg.commit(three_bit_values);
    const kzgCommitment prover_two_bit_commitment = prover_kzg.commit(two_bit_values);

    prover_less.add_check(virCommitment(prover_three_bit_commitment), 3, 4);
    prover_less.add_check(virCommitment(prover_two_bit_commitment), 2, 4);

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

    const kzgCommitment verifier_three_bit_commitment = verifier_kzg.adopt_commitment(prover_three_bit_commitment.com);
    const kzgCommitment verifier_two_bit_commitment = verifier_kzg.adopt_commitment(prover_two_bit_commitment.com);

    verifier_less.add_check(virCommitment(verifier_three_bit_commitment), 3, 4);
    verifier_less.add_check(virCommitment(verifier_two_bit_commitment), 2, 4);

    EXPECT_TRUE(verifier_less.finalize());
    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(LessCheckProtocolTest, ProveAndVerifyLargeBitBoundWithLimbs)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(131072, 131072));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_lookup;
    prover_lookup.setup(prover_kzg, prover_zero, prover_sum);

    lessCheckProtocol prover_less;
    prover_less.setup(prover_kzg, prover_lookup, prover_zero);

    const poly large_values = poly_from_domain_size_ts({0, (size_t(1) << 12) + 7, (size_t(1) << 24) + 11, (size_t(1) << 35) + 19});
    const kzgCommitment prover_values_commitment = prover_kzg.commit(large_values);

    prover_less.add_check(virCommitment(prover_values_commitment), 36, 4);

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
    verifier_less.setup(verifier_kzg, verifier_lookup, verifier_zero);

    const kzgCommitment verifier_values_commitment = verifier_kzg.adopt_commitment(prover_values_commitment.com);
    verifier_less.add_check(virCommitment(verifier_values_commitment), 36, 4);

    EXPECT_TRUE(verifier_less.finalize());
    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(LessCheckProtocolTest, MergesSmallerDomainsFirst)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    TestableLookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    lessCheckProtocol less;
    less.setup(kzg, lookup);

    less.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 1}))), 1, 2);
    less.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 1}))), 1, 2);
    less.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 2, 3}))), 2, 4);
    less.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 2, 3}))), 2, 4);
    less.add_check(virCommitment(kzg.commit(poly_from_domain_ints({0, 1, 2, 3, 4, 5, 6, 7}))), 3, 8);

    EXPECT_TRUE(less.finalize());
    EXPECT_TRUE(lookup.finalize());

    ASSERT_EQ(lookup.prover_ref().checks.size(), 1u);
    EXPECT_EQ(lookup.prover_ref().checks[0].query_domain_size, 32u);
    EXPECT_EQ(lookup.prover_ref().checks[0].table_domain_size, 8u);

    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
