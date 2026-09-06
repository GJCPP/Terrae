#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "algebra_utils.hpp"
#include "kzg.hpp"
#include "polynomial.hpp"
#include "sum_check.hpp"
#include "zero_check.hpp"

#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

using namespace mcl::bn;

namespace {

poly poly_from_ints(const std::vector<int>& ints)
{
    std::vector<Fr> coeffs;
    coeffs.reserve(ints.size());
    for (int v : ints) {
        coeffs.push_back(fr_from_int(v));
    }
    poly p;
    p.init_with_coeff(coeffs);
    return p;
}

poly random_poly(std::mt19937& rng, int max_degree, int coeff_abs_max)
{
    std::uniform_int_distribution<int> deg_dist(0, max_degree);
    std::uniform_int_distribution<int> coeff_dist(-coeff_abs_max, coeff_abs_max);

    const int degree = deg_dist(rng);
    std::vector<Fr> coeffs(static_cast<size_t>(degree) + 1, Fr(0));
    for (int i = 0; i <= degree; ++i) {
        coeffs[static_cast<size_t>(i)] = fr_from_int(coeff_dist(rng));
    }
    if (coeffs.back().isZero()) {
        coeffs.back() = Fr(1);
    }

    poly p;
    p.init_with_coeff(coeffs);
    return p;
}

Fr sum_over_domain(const poly& p, size_t n)
{
    Fr omega = get_root_of_unity(n);
    Fr x = Fr(1);
    Fr sum = Fr(0);
    for (size_t i = 0; i < n; ++i) {
        sum += p.eval(x);
        Fr::mul(x, x, omega);
    }
    return sum;
}

class TestableSumProtocol : public sumProtocol {
public:
    const sumParam& params_ref() const { return params; }
    const sumProver& prover_ref() const { return prover; }
    const sumVerifier& verifier_ref() const { return verifier; }
};

class TestableKzgProtocol : public kzgProtocol {
public:
    const kzgParams& params_ref() const { return params; }
};

class TestableZeroProtocol : public zeroProtocol {
public:
    void mark_finalized_for_test() { is_finalized = true; }
    const zeroParam& params_ref() const { return params; }
};

} // namespace

class SumCheckTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(SumCheckTest, AddCheck_RejectsNonPowerOfTwoDomain)
{
    TestableSumProtocol protocol;
    virCommitment v;

    EXPECT_THROW(protocol.add_check(v, Fr(6), 3), std::runtime_error);
    EXPECT_THROW(protocol.add_check(v, Fr(6), 0), std::runtime_error);
}

TEST_F(SumCheckTest, DestructorAutoFinalizesSilentlyOnSuccess)
{
    testing::internal::CaptureStderr();
    {
        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(64, 64));

        TestableZeroProtocol zero;
        zero.setup(kzg);

        TestableSumProtocol protocol;
        protocol.setup(kzg, zero);
    }
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(err.empty());
}

TEST_F(SumCheckTest, AddCheck_TracksStateAndFinalizesOneBatchedRemainder)
{
    TestableKzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    TestableZeroProtocol zero;
    zero.setup(kzg);

    TestableSumProtocol protocol;
    protocol.setup(kzg, zero);

    // f(x) = x^5 - x = x * (x^4 - 1), so division by v_4 has quotient x.
    const poly f = poly_from_ints({0, -1, 0, 0, 0, 1});
    constexpr size_t n = 4;
    const Fr claim = fr_from_int(0);

    virCommitment v(kzg.commit(f));
    protocol.add_check(v, claim, n);

    ASSERT_EQ(protocol.params_ref().domain_sizes.size(), 1u);
    EXPECT_EQ(protocol.params_ref().domain_sizes[0], n);
    ASSERT_EQ(protocol.verifier_ref().commits.size(), 1u);
    ASSERT_EQ(protocol.verifier_ref().claims.size(), 1u);
    EXPECT_EQ(protocol.verifier_ref().claims[0].getStr(), claim.getStr());
    EXPECT_EQ(protocol.prover_ref().quotients.size(), 0u);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.params_ref().domain_sizes.empty());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SumCheckTest, AddCheck_UsesDirectOpeningWhenDegreeAlreadyProven)
{
    TestableKzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    TestableZeroProtocol zero;
    zero.setup(kzg);

    TestableSumProtocol protocol;
    protocol.setup(kzg, zero);

    const poly f = poly_from_ints({3, 2, 1});
    constexpr size_t n = 8;
    const Fr claim = sum_over_domain(f, n);

    const kzgCommitment com = kzg.commit(f, n - 1);
    virCommitment v(com);
    protocol.add_check(v, claim, n);

    EXPECT_TRUE(protocol.prover_ref().quotients.empty());
    EXPECT_TRUE(zero.params_ref().domain_sizes.empty());
    ASSERT_EQ(protocol.verifier_ref().direct_opening_checks.size(), 1u);
    ASSERT_EQ(kzg.params_ref().x[com.index].size(), 1u);
    ASSERT_EQ(kzg.params_ref().y[com.index].size(), 1u);
    EXPECT_TRUE(kzg.params_ref().x[com.index][0].isZero());
    EXPECT_EQ(kzg.params_ref().y[com.index][0].getStr(), (claim / n).getStr());

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SumCheckTest, AddCheck_UsesDirectOpeningForDegreeBoundedVirtualCommitment)
{
    TestableKzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    TestableZeroProtocol zero;
    zero.setup(kzg);

    TestableSumProtocol protocol;
    protocol.setup(kzg, zero);

    const poly f = poly_from_ints({3, 2, 1});
    constexpr size_t n = 8;
    const kzgCommitment com = kzg.commit(f, 2);
    virCommitment v = x_scale(virCommitment(com), fr_from_int(5)) + virCommitment(fr_from_int(7));
    const Fr claim = sum_over_domain(*kzg.materialize(v), n);

    protocol.add_check(v, claim, n);

    EXPECT_TRUE(protocol.prover_ref().quotients.empty());
    EXPECT_TRUE(zero.params_ref().domain_sizes.empty());
    ASSERT_EQ(protocol.verifier_ref().direct_opening_checks.size(), 1u);
    ASSERT_EQ(kzg.params_ref().x[com.index].size(), 1u);
    EXPECT_TRUE(kzg.params_ref().x[com.index][0].isZero());

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SumCheckTest, Finalize_FailsWhenDirectOpeningClaimIsWrong)
{
    TestableKzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    TestableZeroProtocol zero;
    zero.setup(kzg);

    TestableSumProtocol protocol;
    protocol.setup(kzg, zero);

    const poly f = poly_from_ints({3, 2, 1});
    constexpr size_t n = 8;
    Fr wrong_claim = sum_over_domain(f, n);
    wrong_claim += Fr(1);

    virCommitment v(kzg.commit(f, n - 1));
    protocol.add_check(v, wrong_claim, n);

    EXPECT_FALSE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SumCheckTest, Finalize_CannotBeCalledTwice)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    TestableZeroProtocol zero;
    zero.setup(kzg);

    TestableSumProtocol protocol;
    protocol.setup(kzg, zero);

    // f(x) = x^3 - x = x * (x^2 - 1), so prover quotient is well-defined and non-empty.
    poly f = poly_from_ints({0, -1, 0, 1});
    virCommitment v(kzg.commit(f));
    protocol.add_check(v, fr_from_int(0), 2);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_THROW(protocol.finalize(), std::runtime_error);
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(SumCheckTest, ProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(128, 128));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    constexpr size_t n = 8;
    const poly f = poly_from_ints({5, -2, 0, 1, 3});
    const Fr claim = sum_over_domain(f, n);
    const kzgCommitment public_commitment = prover_kzg.commit(f);
    virCommitment prover_statement(public_commitment);

    prover_sum.add_check(prover_statement, claim, n);

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

    const kzgCommitment verifier_commitment = verifier_kzg.adopt_commitment(public_commitment.com);
    virCommitment verifier_statement(verifier_commitment);
    verifier_sum.add_check(verifier_statement, claim, n);

    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(SumCheckTest, RandomizedHeavy_ValidBatchesFinalize)
{
    std::mt19937 rng(20260324);
    const std::vector<size_t> domain_choices = {2, 4, 8, 16, 32};
    std::uniform_int_distribution<int> checks_dist(8, 16);
    std::uniform_int_distribution<int> domain_idx_dist(0, static_cast<int>(domain_choices.size()) - 1);

    constexpr int rounds = 40;
    for (int round = 0; round < rounds; ++round) {
        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(1024, 1024));

        TestableZeroProtocol zero;
        zero.setup(kzg);

        TestableSumProtocol protocol;
        protocol.setup(kzg, zero);

        const int check_count = checks_dist(rng);
        for (int i = 0; i < check_count; ++i) {
            const size_t n = domain_choices[static_cast<size_t>(domain_idx_dist(rng))];
            const poly f = random_poly(rng, 18, 20);
            const Fr claim = sum_over_domain(f, n);

            virCommitment v(kzg.commit(f));
            protocol.add_check(v, claim, n);
        }

        EXPECT_TRUE(protocol.finalize()) << "round=" << round;
        EXPECT_TRUE(zero.finalize()) << "round=" << round;
        EXPECT_TRUE(kzg.finalize()) << "round=" << round;
    }
}

TEST_F(SumCheckTest, RandomizedHeavy_LargeSingleBatchTracksState)
{
    std::mt19937 rng(424242);
    const std::vector<size_t> domain_choices = {2, 4, 8, 16, 32, 64};
    std::uniform_int_distribution<int> domain_idx_dist(0, static_cast<int>(domain_choices.size()) - 1);

    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(2048, 2048));

    TestableZeroProtocol zero;
    zero.setup(kzg);

    TestableSumProtocol protocol;
    protocol.setup(kzg, zero);

    constexpr size_t kChecks = 64;
    for (size_t i = 0; i < kChecks; ++i) {
        const size_t n = domain_choices[static_cast<size_t>(domain_idx_dist(rng))];
        const poly h = random_poly(rng, 20, 25);
        const poly f = identity_poly() * h * vanishing_poly(n);
        const Fr claim = sum_over_domain(f, n);

        virCommitment v(kzg.commit(f));
        protocol.add_check(v, claim, n);
    }

    EXPECT_EQ(protocol.params_ref().domain_sizes.size(), kChecks);
    EXPECT_EQ(protocol.verifier_ref().commits.size(), kChecks);
    EXPECT_EQ(protocol.verifier_ref().claims.size(), kChecks);
    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.params_ref().domain_sizes.empty());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}
