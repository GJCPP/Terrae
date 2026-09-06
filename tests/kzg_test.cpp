/**
 * Tests for KZG Polynomial Commitment Scheme (PCS).
 */
#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "algebra_utils.hpp"
#include "kzg.hpp"
#include "lagrange_utils.hpp"
#include "polynomial.hpp"

#include <chrono>
#include <functional>
#include <gtest/gtest.h>
#include <initializer_list>
#include <set>
#include <random>
#include <string_view>
#include <vector>

using namespace mcl::bn;

namespace {

poly dense_poly(std::initializer_list<int> coeffs)
{
    std::vector<Fr> fr_coeffs;
    fr_coeffs.reserve(coeffs.size());
    for (int coeff : coeffs) {
        fr_coeffs.push_back(fr_from_int(coeff));
    }

    poly p;
    p.init_with_coeff(fr_coeffs);
    return p;
}

Fr open_at(kzgProtocol& protocol, const kzgCommitment& com, int x)
{
    return protocol.open(com, fr_from_int(x));
}

void expect_open_eq(kzgProtocol& protocol, const kzgCommitment& com, int x, std::string_view expected)
{
    EXPECT_EQ(open_at(protocol, com, x).getStr(), expected);
}

poly random_sparse_poly(std::mt19937& rng, int max_degree, int max_terms, int coeff_max)
{
    std::uniform_int_distribution<int> deg_dist(0, max_degree);
    std::uniform_int_distribution<int> coeff_dist(1, coeff_max);
    std::uniform_int_distribution<int> terms_dist(1, max_terms);

    poly s;
    const int term_count = terms_dist(rng);
    std::set<int> used_exp;
    for (int i = 0; i < term_count; ++i) {
        int exp = deg_dist(rng);
        if (used_exp.count(exp)) {
            continue;
        }
        used_exp.insert(exp);
        s.add_term(static_cast<size_t>(exp), fr_from_int(coeff_dist(rng)));
    }
    return s;
}

poly random_dense_poly(std::mt19937& rng, int max_degree, int coeff_max)
{
    std::uniform_int_distribution<int> deg_dist(0, max_degree);
    std::uniform_int_distribution<int> coeff_dist(0, coeff_max);

    int deg = deg_dist(rng);
    std::vector<Fr> coeffs(static_cast<size_t>(deg) + 1, Fr(0));
    for (int i = 0; i <= deg; ++i) {
        coeffs[static_cast<size_t>(i)] = fr_from_int(coeff_dist(rng));
    }
    if (coeffs.back().isZero()) {
        coeffs.back() = Fr(1);
    }

    poly p;
    p.init_with_coeff(coeffs);
    return p;
}

std::vector<Fr> random_scalar_vec(std::mt19937& rng, size_t n, int coeff_max)
{
    std::uniform_int_distribution<int> coeff_dist(0, coeff_max);
    std::vector<Fr> coeffs(n, Fr(0));
    for (size_t i = 0; i < n; ++i) {
        coeffs[i] = fr_from_int(coeff_dist(rng));
    }
    if (!coeffs.empty() && coeffs.back().isZero()) {
        coeffs.back() = Fr(1);
    }
    return coeffs;
}

double measure_ms(const std::function<void()>& fn, int rounds)
{
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    for (int i = 0; i < rounds; ++i) {
        fn();
    }
    const auto t1 = Clock::now();
    const std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    return elapsed.count() / static_cast<double>(rounds);
}

struct RandomExpr {
    virCommitment V;
    bool sparse_only = true;
};

RandomExpr random_expr_sum_product(std::mt19937& rng, kzgProtocol& protocol, int depth, bool allow_kzg_leaf)
{
    if (depth == 0) {
        std::uniform_int_distribution<int> leaf_type_dist(0, allow_kzg_leaf ? 1 : 0);
        int leaf_type = leaf_type_dist(rng);  // 0 = sparse, 1 = kzg
        if (leaf_type == 0) {
            return {virCommitment(random_sparse_poly(rng, 6, 3, 20)), true};
        }

        poly p = random_dense_poly(rng, 4, 15);
        return {virCommitment(protocol.commit(p)), false};
    }

    RandomExpr left = random_expr_sum_product(rng, protocol, depth - 1, allow_kzg_leaf);
    RandomExpr right = random_expr_sum_product(rng, protocol, depth - 1, allow_kzg_leaf);

    RandomExpr out;
    out.V = left.V + right.V;
    out.sparse_only = left.sparse_only && right.sparse_only;
    return out;
}

virCommitment random_sparse_expr_sum_product(std::mt19937& rng, int depth)
{
    if (depth == 0) {
        return virCommitment(random_sparse_poly(rng, 8, 4, 20));
    }
    std::uniform_int_distribution<int> op_dist(0, 1);  // 0 = sum, 1 = product
    virCommitment left = random_sparse_expr_sum_product(rng, depth - 1);
    virCommitment right = random_sparse_expr_sum_product(rng, depth - 1);
    if (op_dist(rng) == 0) {
        return left + right;
    }
    return left * right;
}

} // namespace

class TestableKzgProtocol : public kzgProtocol {
public:
    const kzgVerifier& verifier_ref() const { return verifier; }
};

class KzgTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_groups();
    }
};

TEST_F(KzgTest, Setup_RequiresCallBeforeCommit) {
    kzgProtocol protocol;
    poly p = dense_poly({1, 2, 3});

    EXPECT_THROW(protocol.commit(p), std::runtime_error);
}

TEST_F(KzgTest, Setup_RequiresCallBeforeFinalize) {
    kzgProtocol protocol;
    EXPECT_THROW(protocol.finalize(), std::runtime_error);
}

TEST_F(KzgTest, DestructorAutoFinalizesSilentlyOnSuccess) {
    testing::internal::CaptureStderr();
    {
        kzgProtocol protocol;
        protocol.setup(testing_kzg_public_parameters(10, 10));
    }
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(err.empty());
}

TEST_F(KzgTest, DestructorReportsFinalizeFailure) {
    testing::internal::CaptureStderr();
    {
        kzgProtocol protocol;
        protocol.setup(testing_kzg_public_parameters(10, 10));

        poly p = dense_poly({1, 2, 3});
        protocol.commit(p, 1);
    }
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("kzgProtocol finalize failed during destruction"), std::string::npos);
}

TEST_F(KzgTest, SinglePolynomial_SingleOpening_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p(x) = 1 + 2x + 3x^2
    poly p = dense_poly({1, 2, 3});

    kzgCommitment com = protocol.commit(p);

    // Verify y = p(5) = 1 + 10 + 75 = 86
    EXPECT_EQ(open_at(protocol, com, 5).getStr(), fr_from_dec("86").getStr());

}

TEST_F(KzgTest, SinglePolynomial_MultipleOpenings_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p(x) = 1 + 2x + 3x^2
    poly p = dense_poly({1, 2, 3});

    kzgCommitment com = protocol.commit(p);

    // Open at multiple points
    // p(1) = 6, p(2) = 17, p(3) = 34
    expect_open_eq(protocol, com, 1, "6");
    expect_open_eq(protocol, com, 2, "17");
    expect_open_eq(protocol, com, 3, "34");

}

TEST_F(KzgTest, AdoptedCommitmentsDoNotBreakLaterPublicMaterialization) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(32, 32));

    const poly first_public = dense_poly({1, 2});
    const poly second_public = dense_poly({3, 4, 5});

    const kzgCommitment first_commitment = protocol.commit_public("public.before_adopt", first_public, 1);
    const kzgCommitment adopted_commitment = protocol.adopt_commitment(Group1::one());
    const kzgCommitment second_commitment = protocol.commit_public("public.after_adopt", second_public, 2);

    EXPECT_EQ(protocol.materialize(virCommitment(first_commitment))->coefficients(), first_public.coefficients());
    EXPECT_EQ(protocol.materialize(virCommitment(second_commitment))->coefficients(), second_public.coefficients());
    EXPECT_THROW(protocol.materialize(virCommitment(adopted_commitment)), std::runtime_error);
}

TEST_F(KzgTest, MultiplePolynomials_MultiplyOpenings_Verifies) {
    // Random engine for repeatable tests
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> poly_dist(3, 9);  // 2 < n < 10
    std::uniform_int_distribution<int> deg_dist(1, 20);   // keep degree reasonable
    std::uniform_int_distribution<int> coeff_dist(0, 100);
    std::uniform_int_distribution<int> open_dist(2, 9);  // 1 < m_i < 10
    std::uniform_int_distribution<int> xval_dist(1, 10000);

    int n = poly_dist(rng);

    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(20, 128));

    struct PolyInfo {
        poly p;
        std::vector<Fr> coeffs;
        kzgCommitment com;
        std::vector<Fr> open_points;
        std::vector<Fr> opened_values;
    };

    std::vector<PolyInfo> polys(n);

    // Generate each polynomial
    for (int i = 0; i < n; ++i) {
        int deg = deg_dist(rng);
        std::vector<Fr> coeffs(deg + 1);
        for (int j = 0; j <= deg; ++j) {
            coeffs[j].setStr(std::to_string(coeff_dist(rng)), 10);
        }
        polys[i].coeffs = coeffs;
        polys[i].p.init_with_coeff(coeffs);
        polys[i].com = protocol.commit(polys[i].p);
    }

    // For each poly, open at random points, check correct values
    for (int i = 0; i < n; ++i) {
        int m = open_dist(rng);
        std::vector<Fr> xs(m), ys(m);
        std::set<std::string> used_xs;
        for (int k = 0; k < m; ++k) {
            // Pick unique x values per poly
            Fr x;
            std::string xstr;
            do {
                x.setStr(std::to_string(xval_dist(rng)), 10);
                xstr = x.getStr();
            } while (used_xs.count(xstr));
            used_xs.insert(xstr);

            Fr y = protocol.open(polys[i].com, x);

            // Manually evaluate
            Fr expected = Fr(0);
            for (int j = polys[i].coeffs.size() - 1; j >= 0; --j) {
                Fr::mul(expected, expected, x);
                Fr::add(expected, expected, polys[i].coeffs[j]);
            }
            EXPECT_EQ(y.getStr(), expected.getStr()) << "Polynomial " << i << " at x=" << xstr;

            xs[k] = x;
            ys[k] = y;
        }
        polys[i].open_points = xs;
        polys[i].opened_values = ys;
    }

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, ZeroPolynomial_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    poly p;
    p.init_zero();

    kzgCommitment com = protocol.commit(p);

    Fr x;
    x.setStr("123", 10);
    Fr y = protocol.open(com, x);

    EXPECT_TRUE(y.isZero());

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, ConstantPolynomial_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    poly p = dense_poly({42});

    kzgCommitment com = protocol.commit(p);

    expect_open_eq(protocol, com, 999, "42");

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, HighDegreePolynomial_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p(x) = 1 + 2x + 3x^2 + 4x^3 + 5x^4
    poly p = dense_poly({1, 2, 3, 4, 5});

    kzgCommitment com = protocol.commit(p);

    // p(2) = 1 + 4 + 12 + 32 + 80 = 129
    EXPECT_EQ(open_at(protocol, com, 2).getStr(), fr_from_dec("129").getStr());

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, CommitPolynomialLargerThanSrs_Throws) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(4, 4));

    poly p = dense_poly({1, 2, 3, 4, 5});

    EXPECT_THROW(protocol.commit(p), std::runtime_error);
}

TEST_F(KzgTest, FinalizeTwice_Throws) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    poly p = dense_poly({1, 2});

    kzgCommitment com = protocol.commit(p);
    open_at(protocol, com, 1);

    ASSERT_TRUE(protocol.finalize());
    EXPECT_THROW(protocol.finalize(), std::runtime_error);
}

// --- Degree bound tests ---

TEST_F(KzgTest, DegreeBound_SinglePolynomial_WithinBound_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p(x) = 1 + 2x + 3x^2 (degree 2)
    poly p = dense_poly({1, 2, 3});

    // Claim degree <= 3 (actual degree 2, within bound)
    kzgCommitment com = protocol.commit(p, 3);

    EXPECT_EQ(open_at(protocol, com, 5).getStr(), fr_from_dec("86").getStr());

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, DegreeBound_SinglePolynomial_ExactBound_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p(x) = 1 + 2x + 3x^2 (degree 2)
    poly p = dense_poly({1, 2, 3});

    // Claim degree <= 2 (exact bound)
    kzgCommitment com = protocol.commit(p, 2);

    expect_open_eq(protocol, com, 7, "162");  // p(7) = 1 + 14 + 147 = 162

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, DegreeBound_SinglePolynomial_ZeroPoly_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    poly p;
    p.init_zero();

    // Claim degree <= 0
    kzgCommitment com = protocol.commit(p, 0);

    Fr x;
    x.setStr("42", 10);
    Fr y = protocol.open(com, x);
    EXPECT_TRUE(y.isZero());

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, DegreeBound_SinglePolynomial_Constant_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    poly p = dense_poly({7});

    // Claim degree <= 0
    kzgCommitment com = protocol.commit(p, 0);

    expect_open_eq(protocol, com, 100, "7");

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, DegreeBound_MultiplePolynomials_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p1(x) = 1 + x (degree 1)
    poly p1 = dense_poly({1, 1});

    // p2(x) = 1 + x + x^2 (degree 2)
    poly p2 = dense_poly({1, 1, 1});

    kzgCommitment com1 = protocol.commit(p1, 1);
    kzgCommitment com2 = protocol.commit(p2, 2);

    expect_open_eq(protocol, com1, 3, "4");   // p1(3) = 4
    expect_open_eq(protocol, com2, 4, "21");  // p2(4) = 1 + 4 + 16 = 21

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, DegreeBound_MixedWithUnbounded_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // Polynomial with degree bound
    poly p1 = dense_poly({1, 2});
    kzgCommitment com1 = protocol.commit(p1, 1);

    // Polynomial without degree bound
    poly p2 = dense_poly({1, 0, 1});
    kzgCommitment com2 = protocol.commit(p2);

    expect_open_eq(protocol, com1, 5, "11");   // p1(5) = 1 + 10 = 11
    expect_open_eq(protocol, com2, 5, "26");   // p2(5) = 1 + 25 = 26

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, DegreeBound_ProveDegreeAfterCommit_Verifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    std::vector<Fr> coeffs = {Fr(1), Fr(2), Fr(3)};
    poly p;
    p.init_with_coeff(coeffs);

    kzgCommitment com = protocol.commit(p);
    protocol.prove_degree(com, 2);

    Fr x;
    x.setStr("1", 10);
    protocol.open(com, x);

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, DegreeBoundChallengeComesFromTranscript) {
    TestableKzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    poly p = dense_poly({1, 2, 3});
    kzgCommitment com = protocol.commit(p);
    protocol.prove_degree(com, 2);
    protocol.open(com, fr_from_int(1));

    ProofTranscript expected = protocol.transcript();
    const Fr expected_challenge = expected.squeeze_fr("kzg.degree.challenge");

    EXPECT_TRUE(protocol.finalize());
    EXPECT_EQ(protocol.verifier_ref().degree_challenge, expected_challenge);
}

TEST_F(KzgTest, ProveAndVerifyWithSerializedProof) {
    kzgProtocol prover;
    prover.setup(testing_kzg_public_parameters(10, 10));
    auto prove_session = ProofSession::prover();
    prover.set_proof_session(prove_session);

    poly p = dense_poly({1, 2, 3});
    kzgCommitment com = prover.commit(p, 2);
    EXPECT_EQ(prover.open(com, fr_from_int(5)), fr_from_dec("86"));
    EXPECT_TRUE(prover.finalize());

    const SerializedProof proof = prove_session->take_proof();
    const kzgPublicParameters public_params = prover.public_parameters();

    kzgProtocol verifier;
    verifier.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier.set_proof_session(verify_session);

    kzgCommitment verifier_com = verifier.adopt_commitment(com.com);
    verifier.prove_degree(verifier_com, 2);
    EXPECT_EQ(verifier.open(verifier_com, fr_from_int(5)), fr_from_dec("86"));
    EXPECT_TRUE(verifier.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(KzgTest, ProveAndVerifyWithSerializedProofBatchesSamePointAcrossCommitments) {
    const kzgPublicParameters public_params = testing_kzg_public_parameters(32, 32);

    kzgProtocol prover;
    prover.setup(public_params);
    auto prove_session = ProofSession::prover();
    prover.set_proof_session(prove_session);

    const poly p1 = dense_poly({1, 2, 3});
    const poly p2 = dense_poly({4, 5});
    const poly p3 = dense_poly({7, 0, 1});

    const kzgCommitment c1 = prover.commit(p1, 2);
    const kzgCommitment c2 = prover.commit(p2, 1);
    const kzgCommitment c3 = prover.commit(p3, 2);

    const Fr x = fr_from_int(5);
    EXPECT_EQ(prover.open(c1, x), p1.eval(x));
    EXPECT_EQ(prover.open(c2, x), p2.eval(x));
    EXPECT_EQ(prover.open(c3, x), p3.eval(x));
    EXPECT_TRUE(prover.finalize());

    const SerializedProof proof = prove_session->take_proof();

    kzgProtocol verifier;
    verifier.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier.set_proof_session(verify_session);

    const kzgCommitment vc1 = verifier.adopt_commitment(c1.com);
    verifier.prove_degree(vc1, 2);
    const kzgCommitment vc2 = verifier.adopt_commitment(c2.com);
    verifier.prove_degree(vc2, 1);
    const kzgCommitment vc3 = verifier.adopt_commitment(c3.com);
    verifier.prove_degree(vc3, 2);

    EXPECT_EQ(verifier.open(vc1, x), p1.eval(x));
    EXPECT_EQ(verifier.open(vc2, x), p2.eval(x));
    EXPECT_EQ(verifier.open(vc3, x), p3.eval(x));

    EXPECT_TRUE(verifier.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(KzgTest, ProveAndVerifyWithSerializedProofBatchesDifferentPointSetsAcrossCommitments) {
    const kzgPublicParameters public_params = testing_kzg_public_parameters(64, 64);

    kzgProtocol prover;
    prover.setup(public_params);
    auto prove_session = ProofSession::prover();
    prover.set_proof_session(prove_session);

    const poly p1 = dense_poly({1, 2, 3, 4});
    const poly p2 = dense_poly({5, 0, 1});
    const poly p3 = dense_poly({7, 8});

    const kzgCommitment c1 = prover.commit(p1, 3);
    const kzgCommitment c2 = prover.commit(p2, 2);
    const kzgCommitment c3 = prover.commit(p3, 1);

    EXPECT_EQ(prover.open(c1, fr_from_int(2)), p1.eval(fr_from_int(2)));
    EXPECT_EQ(prover.open(c1, fr_from_int(3)), p1.eval(fr_from_int(3)));
    EXPECT_EQ(prover.open(c2, fr_from_int(3)), p2.eval(fr_from_int(3)));
    EXPECT_EQ(prover.open(c2, fr_from_int(5)), p2.eval(fr_from_int(5)));
    EXPECT_EQ(prover.open(c3, fr_from_int(7)), p3.eval(fr_from_int(7)));
    EXPECT_TRUE(prover.finalize());

    const SerializedProof proof = prove_session->take_proof();

    kzgProtocol verifier;
    verifier.setup(public_params);
    auto verify_session = ProofSession::verifier(proof);
    verifier.set_proof_session(verify_session);

    const kzgCommitment vc1 = verifier.adopt_commitment(c1.com);
    verifier.prove_degree(vc1, 3);
    const kzgCommitment vc2 = verifier.adopt_commitment(c2.com);
    verifier.prove_degree(vc2, 2);
    const kzgCommitment vc3 = verifier.adopt_commitment(c3.com);
    verifier.prove_degree(vc3, 1);

    EXPECT_EQ(verifier.open(vc1, fr_from_int(2)), p1.eval(fr_from_int(2)));
    EXPECT_EQ(verifier.open(vc1, fr_from_int(3)), p1.eval(fr_from_int(3)));
    EXPECT_EQ(verifier.open(vc2, fr_from_int(3)), p2.eval(fr_from_int(3)));
    EXPECT_EQ(verifier.open(vc2, fr_from_int(5)), p2.eval(fr_from_int(5)));
    EXPECT_EQ(verifier.open(vc3, fr_from_int(7)), p3.eval(fr_from_int(7)));

    EXPECT_TRUE(verifier.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(KzgTest, FinalizeFailsWhenDeg2CannotCoverUnionOfBatchOpeningPoints) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(16, 4));

    const kzgCommitment c1 = protocol.commit(dense_poly({1, 2, 3}));
    const kzgCommitment c2 = protocol.commit(dense_poly({4, 5}));

    EXPECT_EQ(protocol.open(c1, fr_from_int(2)), dense_poly({1, 2, 3}).eval(fr_from_int(2)));
    EXPECT_EQ(protocol.open(c1, fr_from_int(3)), dense_poly({1, 2, 3}).eval(fr_from_int(3)));
    EXPECT_EQ(protocol.open(c2, fr_from_int(5)), dense_poly({4, 5}).eval(fr_from_int(5)));
    EXPECT_EQ(protocol.open(c2, fr_from_int(7)), dense_poly({4, 5}).eval(fr_from_int(7)));

    EXPECT_THROW(protocol.finalize(), std::runtime_error);
}

TEST_F(KzgTest, DegreeBound_ProveWrongDegree_Fails) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    std::vector<Fr> coeffs = {Fr(1), Fr(2), Fr(3), Fr(4), Fr(5)};
    poly p;
    p.init_with_coeff(coeffs);

    kzgCommitment com = protocol.commit(p);
    protocol.prove_degree(com, 3);

    EXPECT_FALSE(protocol.finalize());
}

// --- Virtual polynomial commitment tests ---

TEST_F(KzgTest, VirtualCommitment_Constant) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    virCommitment V(Fr(42));
    Fr x;
    x.setStr("5", 10);

    std::map<std::pair<size_t, Fr>, Fr> openings;
    auto get_zero = [](size_t, const Fr&) { return Fr(0); };
    Fr value = eval_virtual(V, x, get_zero, openings);

    EXPECT_TRUE(value.getStr() == "42");
    EXPECT_TRUE(openings.empty());
}

TEST_F(KzgTest, VirtualCommitment_Sum_OpensAndVerifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p1(x) = 1 + x, p2(x) = 2x
    std::vector<Fr> c1 = {Fr(1), Fr(1)};
    std::vector<Fr> c2 = {Fr(0), Fr(2)};
    poly p1, p2;
    p1.init_with_coeff(c1);
    p2.init_with_coeff(c2);

    virCommitment V1(protocol.commit(p1));
    virCommitment V2(protocol.commit(p2));
    virCommitment V = V1 + V2;  // p1 + p2 = 1 + 3x

    Fr x;
    x.setStr("5", 10);
    Fr y = protocol.open(V, x);

    // (p1 + p2)(5) = 1 + 15 = 16
    EXPECT_EQ(y.getStr(), "16");

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, VirtualCommitment_Product_OpensAndVerifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p1(x) = 1 + x, p2(x) = x
    std::vector<Fr> c1 = {Fr(1), Fr(1)};
    std::vector<Fr> c2 = {Fr(0), Fr(1)};
    poly p1, p2;
    p1.init_with_coeff(c1);
    p2.init_with_coeff(c2);

    virCommitment V1(protocol.commit(p1));
    virCommitment V2(protocol.commit(p2));
    virCommitment V = V1 * V2;  // p1 * p2 = x(1+x) = x + x^2

    Fr x;
    x.setStr("3", 10);
    Fr y = protocol.open(V, x);

    // (p1 * p2)(3) = 3 + 9 = 12
    EXPECT_EQ(y.getStr(), "12");

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, VirtualCommitment_Fraction_OpensAndVerifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p1(x) = 1 + 2x, p2(x) = 1 + x
    std::vector<Fr> c1 = {Fr(1), Fr(2)};
    std::vector<Fr> c2 = {Fr(1), Fr(1)};
    poly p1, p2;
    p1.init_with_coeff(c1);
    p2.init_with_coeff(c2);

    virCommitment V1(protocol.commit(p1));
    virCommitment V2(protocol.commit(p2));
    virCommitment V = V1 / V2;  // (1+2x)/(1+x)

    Fr x;
    x.setStr("4", 10);
    Fr y = protocol.open(V, x);

    // (p1/p2)(4) = 9/5. In Fr we need 9 * inv(5)
    Fr nine, five, inv5, expected;
    nine.setStr("9", 10);
    five.setStr("5", 10);
    Fr::inv(inv5, five);
    Fr::mul(expected, nine, inv5);
    EXPECT_EQ(y.getStr(), expected.getStr());

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, VirtualCommitment_Nested_OpensAndVerifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    // p1 = 1, p2 = 2, p3 = 3
    std::vector<Fr> c1 = {Fr(1)}, c2 = {Fr(2)}, c3 = {Fr(3)};
    poly p1, p2, p3;
    p1.init_with_coeff(c1);
    p2.init_with_coeff(c2);
    p3.init_with_coeff(c3);

    virCommitment V1(protocol.commit(p1));
    virCommitment V2(protocol.commit(p2));
    virCommitment V3(protocol.commit(p3));
    virCommitment V = (V1 + V2) * V3;  // (1+2)*3 = 9

    Fr x;
    x.setStr("100", 10);
    Fr y = protocol.open(V, x);

    EXPECT_EQ(y.getStr(), "9");

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, VirtualCommitment_XPower_OpensAndVerifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(16, 16));

    // p(x) = 1 + x + x^2
    std::vector<Fr> coeffs = {Fr(1), Fr(1), Fr(1)};
    poly p;
    p.init_with_coeff(coeffs);

    virCommitment V = x_pow(virCommitment(protocol.commit(p)), 3);  // p(x^3)

    Fr x;
    x.setStr("2", 10);
    Fr y = protocol.open(V, x);

    // p(2^3) = 1 + 8 + 64 = 73
    EXPECT_EQ(y.getStr(), "73");
    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, VirtualCommitment_XPower_ZeroExponentUsesOneAsInput) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(8, 8));

    poly s;
    s.add_term(0, Fr(5));
    s.add_term(1, Fr(7));  // 5 + 7x

    virCommitment V = x_pow(virCommitment(s), 0);
    Fr x;
    x.setStr("11", 10);

    std::map<std::pair<size_t, Fr>, Fr> openings;
    auto get_zero = [](size_t, const Fr&) { return Fr(0); };
    Fr value = eval_virtual(V, x, get_zero, openings);

    // Evaluate at x^0 = 1.
    EXPECT_EQ(value.getStr(), "12");
    EXPECT_TRUE(openings.empty());
}

TEST_F(KzgTest, VirtualCommitment_XScale_OpensAndVerifies) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(16, 16));

    poly p = dense_poly({2, 3, 4});  // 2 + 3x + 4x^2
    virCommitment V = x_scale(virCommitment(protocol.commit(p)), fr_from_int(5));

    Fr x = fr_from_int(7);
    Fr y = protocol.open(V, x);

    EXPECT_EQ(y, p.eval(fr_from_int(5) * x));
    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, HasDegreeBound_TracksVirtualExpressionsFromProofsAndStructure) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(32, 32));

    poly p = dense_poly({2, 3, 4});  // degree 2
    const kzgCommitment com = protocol.commit(p, 2);

    virCommitment bounded =
        x_scale(virCommitment(com), fr_from_int(5)) + x_pow(virCommitment(fr_from_int(7)), 9);
    EXPECT_TRUE(protocol.has_degree_bound(bounded, 2));
    EXPECT_FALSE(protocol.has_degree_bound(bounded, 1));

    virCommitment product = bounded * virCommitment(dense_poly({1, 0, 1}));  // multiply by degree-2 sparse poly
    EXPECT_TRUE(protocol.has_degree_bound(product, 4));
    EXPECT_FALSE(protocol.has_degree_bound(product, 3));

    virCommitment quotient = product / virCommitment(fr_from_int(3));
    EXPECT_TRUE(protocol.has_degree_bound(quotient, 4));

    virCommitment unproven = virCommitment(protocol.commit(p));
    EXPECT_FALSE(protocol.has_degree_bound(unproven, 2));
}

// --- Materialization tests ---

TEST_F(KzgTest, Materialize_SparseOnlySum_ReturnsSparsePolynomial) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    poly s1;
    s1.add_term(0, Fr(2));  // 2
    s1.add_term(3, Fr(5));  // + 5x^3

    poly s2;
    s2.add_term(1, Fr(7));  // 7x
    s2.add_term(3, Fr(1));  // + x^3

    virCommitment V = virCommitment(s1) + virCommitment(s2);
    std::shared_ptr<poly> mat = protocol.materialize(V);

    ASSERT_NE(mat, nullptr);
    ASSERT_TRUE(mat->is_sparse());

    Fr x;
    x.setStr("2", 10);
    // (2 + 5x^3) + (7x + x^3) at x=2 => 2 + 40 + 14 + 8 = 64
    EXPECT_EQ(mat->eval(x).getStr(), "64");
}

TEST_F(KzgTest, Materialize_SparseOnlyProduct_ReturnsSparsePolynomial) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(20, 20));

    poly s1;
    s1.add_term(0, Fr(1));  // 1
    s1.add_term(2, Fr(1));  // + x^2

    poly s2;
    s2.add_term(1, Fr(3));  // 3x
    s2.add_term(3, Fr(2));  // + 2x^3

    virCommitment V = virCommitment(s1) * virCommitment(s2);
    std::shared_ptr<poly> mat = protocol.materialize(V);

    ASSERT_NE(mat, nullptr);
    ASSERT_TRUE(mat->is_sparse());

    Fr x;
    x.setStr("2", 10);
    // (1 + x^2)(3x + 2x^3) at x=2 => 5 * 22 = 110
    EXPECT_EQ(mat->eval(x).getStr(), "110");
}

TEST_F(KzgTest, Materialize_KzgAndSparseSum_ReturnsDensePolynomial) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(10, 10));

    std::vector<Fr> coeffs = {Fr(1), Fr(2), Fr(3)};  // 1 + 2x + 3x^2
    poly p;
    p.init_with_coeff(coeffs);
    virCommitment Vkzg(protocol.commit(p));

    poly s;
    s.add_term(0, Fr(4));  // +4
    s.add_term(1, Fr(5));  // +5x
    virCommitment Vsparse(s);

    std::shared_ptr<poly> mat = protocol.materialize(Vkzg + Vsparse);
    ASSERT_NE(mat, nullptr);

    Fr x;
    x.setStr("3", 10);
    // (1 + 2x + 3x^2) + (4 + 5x) at x=3 => 34 + 19 = 53
    EXPECT_EQ(mat->eval(x).getStr(), "53");
}

TEST_F(KzgTest, Materialize_CachesVirtualCommitmentResult) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(16, 16));

    poly p = dense_poly({1, 2, 3});
    virCommitment V = virCommitment(protocol.commit(p)) + virCommitment(fr_from_int(5));

    std::shared_ptr<poly> first = protocol.materialize(V);
    std::shared_ptr<poly> second = protocol.materialize(V);

    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first.get(), second.get());
    EXPECT_EQ(V.materialized_cache.get(), first.get());
}

TEST_F(KzgTest, Materialize_NestedSparseTree_RemainsSparseAndCorrect) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(20, 20));

    poly s1;
    s1.add_term(0, Fr(1));  // 1
    s1.add_term(2, Fr(2));  // + 2x^2

    poly s2;
    s2.add_term(1, Fr(3));  // 3x

    poly s3;
    s3.add_term(0, Fr(5));  // 5

    // ((s1 + s2) * s3) is still sparse-only.
    virCommitment V = (virCommitment(s1) + virCommitment(s2)) * virCommitment(s3);
    std::shared_ptr<poly> mat = protocol.materialize(V);

    ASSERT_NE(mat, nullptr);
    ASSERT_TRUE(mat->is_sparse());

    Fr x;
    x.setStr("4", 10);
    // (1 + 2*16 + 3*4) * 5 = (45) * 5 = 225
    EXPECT_EQ(mat->eval(x).getStr(), "225");
}

TEST_F(KzgTest, Materialize_SparseXPower_RemainsSparseAndCorrect) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(32, 32));

    poly s;
    s.add_term(0, Fr(1));  // 1
    s.add_term(2, Fr(2));  // + 2x^2

    std::shared_ptr<poly> mat = protocol.materialize(x_pow(virCommitment(s), 3));

    ASSERT_NE(mat, nullptr);
    ASSERT_TRUE(mat->is_sparse());

    Fr x;
    x.setStr("2", 10);
    // (1 + 2x^2) with x -> x^3 gives 1 + 2x^6; at x=2 => 1 + 2*64 = 129
    EXPECT_EQ(mat->eval(x).getStr(), "129");
}

TEST_F(KzgTest, Materialize_XScale_MatchesScaledEvaluation) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(32, 32));

    poly s;
    s.add_term(0, Fr(1));
    s.add_term(2, Fr(2));
    s.add_term(5, Fr(3));

    Fr scale = fr_from_int(9);
    std::shared_ptr<poly> mat = protocol.materialize(x_scale(virCommitment(s), scale));

    ASSERT_NE(mat, nullptr);
    ASSERT_TRUE(mat->is_sparse());
    EXPECT_EQ(mat->eval(fr_from_int(2)), s.eval(scale * fr_from_int(2)));
    EXPECT_EQ(mat->eval(fr_from_int(11)), s.eval(scale * fr_from_int(11)));
}

TEST_F(KzgTest, EvaluationsOnDomain_MatchesMaterializedPolynomial) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(32, 32));

    poly p = dense_poly({2, 0, 1});
    virCommitment V = x_scale(virCommitment(protocol.commit(p)), fr_from_int(3)) + virCommitment(fr_from_int(7));

    const std::vector<Fr> expected = protocol.materialize(V)->evaluations_on_domain(8);
    const std::vector<Fr> actual = protocol.evaluations(V, 8);

    EXPECT_EQ(actual, expected);
}

TEST_F(KzgTest, PreferEval_SubgroupLagrangeXPowerMatchesMaterializedPolynomial) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(128, 128));

    const virCommitment selector = x_pow(subgroup_lagrange_expression(4, 1), 2);
    const std::vector<Fr> expected = protocol.materialize(selector)->evaluations_on_domain(16);
    const std::vector<Fr> actual =
        protocol.evaluations(selector, 16, kzgProtocol::EvaluationMode::PreferEval);

    EXPECT_EQ(actual, expected);
}

TEST_F(KzgTest, BatchOpeningCommitments_MatchNaiveQuotientCommitmentsOnDomain) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(64, 64));

    poly p = dense_poly({3, 1, 4, 1, 5});
    const size_t n = 8;
    const Fr omega = get_root_of_unity(n);

    const std::vector<Group1> batch = protocol.batch_opening_commitments(p, n);
    ASSERT_EQ(batch.size(), n);

    Fr root = Fr(1);
    const std::vector<Fr> evals = p.evaluations_on_domain(n);
    for (size_t i = 0; i < n; ++i) {
        const poly numerator = p - evals[i];
        poly divisor;
        divisor.add_term(1, Fr(1));
        divisor.add_term(0, -root);
        const auto [quotient, remainder] = div(numerator, divisor);
        ASSERT_TRUE(remainder.is_zero());
        EXPECT_EQ(batch[i], protocol.commit(quotient).com);
        Fr::mul(root, root, omega);
    }
}

TEST_F(KzgTest, BatchOpeningCommitments_VirtualKzgLeafMatchesPolynomialVariant) {
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(64, 64));

    poly p = dense_poly({2, 7, 1, 8});
    const kzgCommitment com = protocol.commit(p);
    const std::vector<Group1> expected = protocol.batch_opening_commitments(p, 8);
    const std::vector<Group1> actual = protocol.batch_opening_commitments(virCommitment(com), 8);

    EXPECT_EQ(actual, expected);
}

TEST_F(KzgTest, Materialize_RandomizedHeavy_OpenMatchesMaterializedEval) {
    std::mt19937 rng(20260323);
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(128, 128));

    std::uniform_int_distribution<int> depth_dist(2, 5);
    std::uniform_int_distribution<int> x_dist(1, 500);

    constexpr int kExpressions = 120;
    constexpr int kPointsPerExpr = 10;

    for (int i = 0; i < kExpressions; ++i) {
        virCommitment V = random_sparse_expr_sum_product(rng, depth_dist(rng));
        std::shared_ptr<poly> mat = protocol.materialize(V);
        ASSERT_NE(mat, nullptr) << "materialize returned null at expr " << i;

        for (int j = 0; j < kPointsPerExpr; ++j) {
            Fr x = fr_from_int(x_dist(rng));
            std::map<std::pair<size_t, Fr>, Fr> kzg_openings;
            auto get_kzg = [](size_t, const Fr&) { return Fr(0); };
            Fr open_y = eval_virtual(V, x, get_kzg, kzg_openings);
            Fr mat_y = mat->eval(x);
            EXPECT_EQ(open_y.getStr(), mat_y.getStr())
                << "Mismatch at expr " << i << ", point " << j << ", x=" << x.getStr();
            EXPECT_TRUE(kzg_openings.empty());
        }
    }
}

TEST_F(KzgTest, Materialize_RandomizedHeavy_SparseOnlyTreesStaySparse) {
    std::mt19937 rng(424242);
    kzgProtocol protocol;
    protocol.setup(testing_kzg_public_parameters(64, 64));

    std::uniform_int_distribution<int> depth_dist(2, 4);
    std::uniform_int_distribution<int> x_dist(1, 200);

    constexpr int kExpressions = 72;
    constexpr int kPointsPerExpr = 4;

    for (int i = 0; i < kExpressions; ++i) {
        RandomExpr expr = random_expr_sum_product(rng, protocol, depth_dist(rng), false);
        ASSERT_TRUE(expr.sparse_only);

        std::shared_ptr<poly> mat = protocol.materialize(expr.V);
        ASSERT_NE(mat, nullptr);
        ASSERT_TRUE(mat->is_sparse()) << "Expected sparse result for sparse-only tree " << i;

        for (int j = 0; j < kPointsPerExpr; ++j) {
            Fr x = fr_from_int(x_dist(rng));
            Fr open_y = protocol.open(expr.V, x);
            Fr mat_y = mat->eval(x);
            EXPECT_EQ(open_y.getStr(), mat_y.getStr())
                << "Mismatch at sparse-only expr " << i << ", point " << j;
        }
    }

    EXPECT_TRUE(protocol.finalize());
}

TEST_F(KzgTest, MsmPippenger_LargePolynomial_ShowsAdvantageOverNaive) {
    std::mt19937 rng(20260326);

    // Large degree to make MSM algorithmic gap visible.
    const size_t n = 8192;
    const int rounds = 3;
    const std::vector<Fr> coeffs = random_scalar_vec(rng, n, 1000000);

    kzgParams params(n, n);
    kzgProver prover;
    prover.params = &params;

    poly p;
    p.init_with_coeff(coeffs);

    // Warmup to reduce one-time overhead effects.
    Group1 warmup_pippenger = prover.commit(p, 0).com;
    Group1 warmup_naive = Group1::inner_prod(coeffs, params.g1_powers);
    EXPECT_EQ(warmup_pippenger, warmup_naive);

    Group1 commit_pippenger;
    Group1 commit_naive;

    const double pippenger_ms = measure_ms([&]() {
        commit_pippenger = prover.commit(p, 0).com;
    }, rounds);
    const double naive_ms = measure_ms([&]() {
        commit_naive = Group1::inner_prod(coeffs, params.g1_powers);
    }, rounds);

    EXPECT_EQ(commit_pippenger, commit_naive);

    // For a large polynomial, Pippenger should beat the naive baseline.
    // Keep a small margin so this stays stable on slightly noisy machines.
    EXPECT_LT(pippenger_ms, naive_ms * 0.98)
        << "Expected Pippenger to outperform naive MSM for large polynomial: "
        << "pippenger_ms=" << pippenger_ms << ", naive_ms=" << naive_ms;
}
