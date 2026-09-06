#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "algebra_utils.hpp"
#include "kzg.hpp"
#include "polynomial.hpp"
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
    // Keep leading coefficient non-zero to preserve intended degree.
    if (coeffs.back().isZero()) {
        coeffs.back() = Fr(1);
    }

    poly p;
    p.init_with_coeff(coeffs);
    return p;
}

} // namespace

class TestableZeroProtocol : public zeroProtocol {
public:
    const zeroVerifier& verifier_ref() const { return verifier; }
};

class ZeroCheckTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ZeroCheckTest, VanishingPoly_RequiresPowerOfTwo)
{
    EXPECT_THROW(vanishing_poly(0), std::runtime_error);
    EXPECT_THROW(vanishing_poly(3), std::runtime_error);
    EXPECT_THROW(vanishing_poly(6), std::runtime_error);
}

TEST_F(ZeroCheckTest, VanishingPoly_EvaluatesToZeroOnDomain)
{
    constexpr size_t n = 8;
    poly z = vanishing_poly(n);
    Fr omega = get_root_of_unity(n);
    Fr x = Fr(1);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_TRUE(z.eval(x).isZero()) << "Z_n(omega^" << i << ") should be zero";
        Fr::mul(x, x, omega);
    }
}

TEST_F(ZeroCheckTest, AddCheck_RejectsNonPowerOfTwoDomain)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    zeroProtocol protocol;
    protocol.setup(kzg);

    poly p = poly_from_ints({1, 2, 3});
    virCommitment v(kzg.commit(p));

    EXPECT_THROW(protocol.add_check(v, 3), std::runtime_error);
    EXPECT_THROW(protocol.add_check(v, 0), std::runtime_error);
}

TEST_F(ZeroCheckTest, DestructorAutoFinalizesSilentlyOnSuccess)
{
    testing::internal::CaptureStderr();
    {
        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(64, 64));

        zeroProtocol protocol;
        protocol.setup(kzg);
    }
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(err.empty());
}

TEST_F(ZeroCheckTest, DestructorReportsFinalizeFailure)
{
    testing::internal::CaptureStderr();
    {
        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(64, 64));

        zeroProtocol protocol;
        protocol.setup(kzg);

        poly p = poly_from_ints({1, 2, 3});
        virCommitment v(kzg.commit(p));
        protocol.add_check(v, 4);
    }
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("zeroProtocol finalize failed during destruction"), std::string::npos);
}

TEST_F(ZeroCheckTest, Finalize_SucceedsForValidSingleCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    // f(X) = (X^4 - 1) * (2 + 3X + X^2), so f is zero on <omega_4>.
    const poly z4 = vanishing_poly(4);
    const poly h = poly_from_ints({2, 3, 1});
    const poly f = h * z4;

    zeroProtocol protocol;
    protocol.setup(kzg);

    virCommitment v(kzg.commit(f));
    protocol.add_check(v, 4);
    EXPECT_TRUE(protocol.finalize());
}

TEST_F(ZeroCheckTest, FinalizeDerivesEtaFromTranscript)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    const poly f = poly_from_ints({1, 0, 1}) * vanishing_poly(2);

    TestableZeroProtocol protocol;
    protocol.setup(kzg);

    virCommitment v(kzg.commit(f));
    protocol.add_check(v, 2);

    ProofTranscript expected = kzg.transcript();
    const Fr expected_eta = expected.squeeze_fr("zero.eta");

    EXPECT_TRUE(protocol.finalize());
    EXPECT_EQ(protocol.verifier_ref().eta, expected_eta);
}

TEST_F(ZeroCheckTest, ProveAndVerifyWithSerializedProof)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(64, 64));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    const poly f = poly_from_ints({2, 3, 1}) * vanishing_poly(4);
    const kzgCommitment public_commitment = prover_kzg.commit(f);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);
    virCommitment prover_statement(public_commitment);
    prover_zero.add_check(prover_statement, 4);

    EXPECT_TRUE(prover_zero.finalize());
    EXPECT_TRUE(prover_kzg.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover_kzg.public_parameters();

    kzgProtocol verifier_kzg;
    verifier_kzg.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier_kzg.set_proof_session(verify_session);

    const kzgCommitment verifier_commitment = verifier_kzg.adopt_commitment(public_commitment.com);
    zeroProtocol verifier_zero;
    verifier_zero.setup(verifier_kzg);
    virCommitment verifier_statement(verifier_commitment);
    verifier_zero.add_check(verifier_statement, 4);

    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(ZeroCheckTest, Finalize_SucceedsForBatchedChecksWithDifferentDomains)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    // f1(X) = (X^2 - 1) * (3 + X)
    const poly f1 = poly_from_ints({3, 1}) * vanishing_poly(2);
    // f2(X) = (X^8 - 1) * (1 + 4X + 2X^2)
    const poly f2 = poly_from_ints({1, 4, 2}) * vanishing_poly(8);

    zeroProtocol protocol;
    protocol.setup(kzg);

    virCommitment v1(kzg.commit(f1));
    virCommitment v2(kzg.commit(f2));
    protocol.add_check(v1, 2);
    protocol.add_check(v2, 8);

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(ZeroCheckTest, Finalize_FailsForInvalidCheck)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    // f(X) = 1 + X is not identically zero on <omega_4>.
    const poly f = poly_from_ints({1, 1});

    zeroProtocol protocol;
    protocol.setup(kzg);

    virCommitment v(kzg.commit(f));
    protocol.add_check(v, 4);

    EXPECT_FALSE(protocol.finalize());
}

TEST_F(ZeroCheckTest, Finalize_CannotBeCalledTwice)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    const poly f = poly_from_ints({1, 0, 1}) * vanishing_poly(2);

    zeroProtocol protocol;
    protocol.setup(kzg);

    virCommitment v(kzg.commit(f));
    protocol.add_check(v, 2);

    EXPECT_NO_THROW(static_cast<void>(protocol.finalize()));
    EXPECT_THROW(protocol.finalize(), std::runtime_error);
}

TEST_F(ZeroCheckTest, RandomizedHeavy_MixedValidAndInvalidBatches)
{
    std::mt19937 rng(20260323);
    const std::vector<size_t> domain_choices = {2, 4, 8, 16, 32};
    std::uniform_int_distribution<int> checks_dist(6, 12);
    std::uniform_int_distribution<int> domain_idx_dist(0, static_cast<int>(domain_choices.size()) - 1);
    std::uniform_real_distribution<double> invalid_batch_dist(0.0, 1.0);

    // Stress many independent protocol instances with mixed valid/invalid batches.
    constexpr int rounds = 60;
    for (int round = 0; round < rounds; ++round) {
        kzgProtocol kzg;
        kzg.setup(testing_kzg_public_parameters(512, 512));

        zeroProtocol protocol;
        protocol.setup(kzg);

        const int check_count = checks_dist(rng);
        const bool should_make_invalid = (invalid_batch_dist(rng) < 0.35);
        const int bad_idx = should_make_invalid ? (round % check_count) : -1;

        for (int i = 0; i < check_count; ++i) {
            const size_t n = domain_choices[static_cast<size_t>(domain_idx_dist(rng))];
            poly h = random_poly(rng, 24, 20);
            poly f = h * vanishing_poly(n); // valid by construction

            // Corrupt exactly one check in this batch when needed.
            if (i == bad_idx) {
                f = f + Fr(1);
            }

            virCommitment v(kzg.commit(f));
            protocol.add_check(v, n);
        }

        const bool ok = protocol.finalize();
        if (should_make_invalid) {
            EXPECT_FALSE(ok) << "round=" << round << " should fail";
        } else {
            EXPECT_TRUE(ok) << "round=" << round << " should pass";
        }
    }
}
