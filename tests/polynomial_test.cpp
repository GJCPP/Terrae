/**
 * Tests for NTT / Lagrange interpolation at roots of unity (lagrange_ntt).
 */
#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"
#include <mcl/ntt.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <cstring>

using namespace mcl::bn;

// Initialize BLS12-381 once for all tests (Fr and NTT depend on it).
class LagrangeNttTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }

    // Evaluate polynomial with coefficients c[0..n-1] at x: c[0] + c[1]*x + ... + c[n-1]*x^(n-1).
    static void evalPoly(Fr& out, const Fr* c, size_t n, const Fr& x)
    {
        if (n == 0) {
            out = Fr(0);
            return;
        }
        out = c[n - 1];
        for (int i = (int)n - 2; i >= 0; i--) {
            Fr::mul(out, out, x);
            Fr::add(out, out, c[i]);
        }
    }
};

TEST_F(LagrangeNttTest, InvalidN_ReturnsFalse)
{
    Fr omega;
    omega.setStr("1", 10);
    Fr arr[4];
    for (int i = 0; i < 4; i++) arr[i].setStr(std::to_string(i).c_str(), 10);

    EXPECT_FALSE(ntt(arr, 0, omega));
    EXPECT_FALSE(ntt(arr, 3, omega));  // not power of two
    EXPECT_FALSE(intt(arr, 0, omega));
    EXPECT_FALSE(intt(arr, 3, omega));
}

TEST_F(LagrangeNttTest, N1_IsIdentity)
{
    Fr omega;
    ASSERT_TRUE(get_root_of_unity(omega, 1));

    Fr a, b;
    a.setStr("42", 10);
    b = a;
    EXPECT_TRUE(ntt(&a, 1, omega));
    EXPECT_TRUE(intt(&a, 1, omega));
    EXPECT_EQ(a.getStr(), b.getStr());
}

TEST_F(LagrangeNttTest, Roundtrip_N2)
{
    Fr omega;
    ASSERT_TRUE(get_root_of_unity(omega, 2));

    Fr coeffs[2], original[2];
    coeffs[0].setStr("1", 10);
    coeffs[1].setStr("2", 10);
    original[0] = coeffs[0];
    original[1] = coeffs[1];

    ASSERT_TRUE(ntt(coeffs, 2, omega));
    ASSERT_TRUE(intt(coeffs, 2, omega));

    EXPECT_EQ(coeffs[0].getStr(), original[0].getStr());
    EXPECT_EQ(coeffs[1].getStr(), original[1].getStr());
}

TEST_F(LagrangeNttTest, Roundtrip_N4)
{
    Fr omega;
    ASSERT_TRUE(get_root_of_unity(omega, 4));

    Fr coeffs[4], original[4];
    coeffs[0].setStr("1", 10);
    coeffs[1].setStr("2", 10);
    coeffs[2].setStr("3", 10);
    coeffs[3].setStr("4", 10);
    for (size_t i = 0; i < 4; i++) original[i] = coeffs[i];

    ASSERT_TRUE(ntt(coeffs, 4, omega));
    ASSERT_TRUE(intt(coeffs, 4, omega));

    for (size_t i = 0; i < 4; i++)
        EXPECT_EQ(coeffs[i].getStr(), original[i].getStr()) << "i=" << i;
}

TEST_F(LagrangeNttTest, Roundtrip_N8)
{
    Fr omega;
    ASSERT_TRUE(get_root_of_unity(omega, 8));

    std::vector<Fr> coeffs(8);
    for (size_t i = 0; i < 8; i++) coeffs[i].setStr(std::to_string(i + 1).c_str(), 10);
    std::vector<Fr> original = coeffs;

    ASSERT_TRUE(ntt(coeffs.data(), 8, omega));
    ASSERT_TRUE(intt(coeffs.data(), 8, omega));

    for (size_t i = 0; i < 8; i++)
        EXPECT_EQ(coeffs[i].getStr(), original[i].getStr()) << "i=" << i;
}

TEST_F(LagrangeNttTest, ForwardNTT_MatchesManualEvaluation)
{
    Fr omega;
    ASSERT_TRUE(get_root_of_unity(omega, 4));

    // Coefficients: 1 + 2*x + 3*x^2 + 4*x^3
    Fr coeffs[4];
    coeffs[0].setStr("1", 10);
    coeffs[1].setStr("2", 10);
    coeffs[2].setStr("3", 10);
    coeffs[3].setStr("4", 10);

    Fr evaluations[4];
    for (size_t i = 0; i < 4; i++) evaluations[i] = coeffs[i];
    ASSERT_TRUE(ntt(evaluations, 4, omega));

    // Check evaluations[i] = poly(omega^i) for i = 0,1,2,3.
    Fr x = Fr(1);
    for (size_t i = 0; i < 4; i++) {
        Fr expected;
        evalPoly(expected, coeffs, 4, x);
        EXPECT_EQ(evaluations[i].getStr(), expected.getStr())
            << "evaluation at omega^" << i << " should equal poly(omega^" << i << ")";
        Fr::mul(x, x, omega);  // x = omega^(i+1)
    }
}

TEST_F(LagrangeNttTest, InverseNTT_RecoversCoefficientsFromEvaluations)
{
    Fr omega;
    ASSERT_TRUE(get_root_of_unity(omega, 4));

    // Start from evaluations at 1, omega, omega^2, omega^3 (we get them via NTT from known coeffs).
    Fr coeffs[4];
    coeffs[0].setStr("1", 10);
    coeffs[1].setStr("2", 10);
    coeffs[2].setStr("3", 10);
    coeffs[3].setStr("4", 10);  // polynomial 1
    Fr evals[4];
    for (size_t i = 0; i < 4; i++) evals[i] = coeffs[i];
    
    ASSERT_TRUE(ntt(evals, 4, omega));
    ASSERT_TRUE(intt(evals, 4, omega));

    EXPECT_EQ(evals[0].getStr(), "1");
    EXPECT_EQ(evals[1].getStr(), "2");
    EXPECT_EQ(evals[2].getStr(), "3");
    EXPECT_EQ(evals[3].getStr(), "4");
}

TEST_F(LagrangeNttTest, OmegaIsRootOfUnity)
{
    for (size_t n : { size_t(2), size_t(4), size_t(8), size_t(16) }) {
        Fr omega;
        ASSERT_TRUE(get_root_of_unity(omega, n)) << "n=" << n;
        Fr omega_n;
        Fr::pow(omega_n, omega, n);
        EXPECT_TRUE(omega_n.isOne()) << "omega^" << n << " should be 1, n=" << n;
    }
}

class PolyTest : public ::testing::Test {
protected:
    void SetUp() override {
        init_groups();
    }
};

TEST_F(PolyTest, InitWithCoeffWorks)
{
    // poly initialization: p(x) = 3x^2 + 2x + 1
    Fr coeffs[3];
    coeffs[0].setStr("1", 10);
    coeffs[1].setStr("2", 10);
    coeffs[2].setStr("3", 10);

    poly p;
    p.init_with_coeff(coeffs, 3);
    ASSERT_EQ(p.coefficients().size(), 3u);
    EXPECT_EQ(p.coefficients()[0].getStr(), "1");
    EXPECT_EQ(p.coefficients()[1].getStr(), "2");
    EXPECT_EQ(p.coefficients()[2].getStr(), "3");
}

TEST_F(PolyTest, AddPolynomials)
{
    // p(x) = x + 1, q(x) = 4x^2 + 5
    Fr pc[2], qc[3];
    pc[0].setStr("1", 10); pc[1].setStr("1", 10);
    qc[0].setStr("5", 10); qc[1].setStr("0", 10); qc[2].setStr("4", 10);

    poly p, q;
    p.init_with_coeff(pc, 2);
    q.init_with_coeff(qc, 3);

    poly sum = p + q;
    ASSERT_EQ(sum.coefficients().size(), 3u);
    // [1+5, 1+0, 0+4] = [6,1,4]
    EXPECT_EQ(sum.coefficients()[0].getStr(), "6");
    EXPECT_EQ(sum.coefficients()[1].getStr(), "1");
    EXPECT_EQ(sum.coefficients()[2].getStr(), "4");
}

TEST_F(PolyTest, SubtractPolynomials)
{
    // p(x) = 3x^2 + 3, q(x) = x^2 + 1
    Fr pc[3], qc[3];
    pc[0].setStr("3", 10); pc[1].setStr("0", 10); pc[2].setStr("3", 10);
    qc[0].setStr("1", 10); qc[1].setStr("0", 10); qc[2].setStr("1", 10);

    poly p, q;
    p.init_with_coeff(pc, 3);
    q.init_with_coeff(qc, 3);

    poly diff = p - q;
    ASSERT_EQ(diff.coefficients().size(), 3u);
    // [3-1, 0-0, 3-1] = [2,0,2]
    EXPECT_EQ(diff.coefficients()[0].getStr(), "2");
    EXPECT_TRUE(diff.coefficients()[1].isZero());
    EXPECT_EQ(diff.coefficients()[2].getStr(), "2");
}

TEST_F(PolyTest, MultiplyPolynomials)
{
    // p(x) = x + 1, q(x) = x + 2
    Fr pc[2], qc[2];
    pc[0].setStr("1", 10); pc[1].setStr("1", 10);
    qc[0].setStr("2", 10); qc[1].setStr("1", 10);

    poly p, q;
    p.init_with_coeff(pc, 2);
    q.init_with_coeff(qc, 2);

    poly prod = p * q;
    ASSERT_EQ(prod.coefficients().size(), 3u);
    // (x + 1)(x + 2) = x^2 + 3x + 2
    EXPECT_EQ(prod.coefficients()[0].getStr(), "2");
    EXPECT_EQ(prod.coefficients()[1].getStr(), "3");
    EXPECT_EQ(prod.coefficients()[2].getStr(), "1");
}

TEST_F(PolyTest, ZeroPolynomialMultiplication)
{
    poly zero, p;
    Fr pc[2];
    pc[0].setStr("1", 10); pc[1].setStr("1", 10);
    p.init_with_coeff(pc, 2);

    poly res1 = zero * p;
    poly res2 = p * zero;

    EXPECT_TRUE(res1.coefficients().empty());
    EXPECT_TRUE(res2.coefficients().empty());
}

TEST_F(PolyTest, TrimRemovesTrailingZeros)
{
    Fr coeffs[4];
    coeffs[0].setStr("1", 10);
    coeffs[1].setStr("0", 10);
    coeffs[2].setStr("2", 10);
    coeffs[3].setStr("0", 10);

    poly p;
    p.init_with_coeff(coeffs, 4);
    // The .trim() called inside should remove the last (zero) coefficient
    EXPECT_EQ(p.coefficients().size(), 3u);
    EXPECT_EQ(p.coefficients()[2].getStr(), "2");
}

TEST_F(PolyTest, NttAndInttConsistency)
{
    // p(x) = x^3 + 2x^2 + 3x + 4
    Fr coeffs[4];
    coeffs[0].setStr("4", 10);
    coeffs[1].setStr("3", 10);
    coeffs[2].setStr("2", 10);
    coeffs[3].setStr("1", 10);

    std::vector<Fr> v(coeffs, coeffs+4);
    Fr omega4;
    ASSERT_TRUE(get_root_of_unity(omega4, 4));
    ASSERT_TRUE(ntt(v.data(), 4, omega4));
    ASSERT_TRUE(intt(v.data(), 4, omega4));
    // Now v should equal original coeffs
    for(int i=0; i<4; ++i)
        EXPECT_EQ(v[i].getStr(), coeffs[i].getStr());
}

TEST_F(PolyTest, EvaluationsOnDomainFoldHighDegreeTerms)
{
    constexpr size_t domain_size = 4;
    const std::vector<Fr> dense_coeffs = {
        Fr(1), Fr(2), Fr(3), Fr(4), Fr(5), Fr(6), Fr(7), Fr(8), Fr(9), Fr(10),
    };

    poly dense;
    dense.init_with_coeff(dense_coeffs);

    poly sparse;
    sparse.add_term(0, Fr(3));
    sparse.add_term(4, Fr(5));
    sparse.add_term(9, Fr(7));

    for (const poly* candidate : {&dense, &sparse}) {
        const std::vector<Fr> evaluations = candidate->evaluations_on_domain(domain_size);
        ASSERT_EQ(evaluations.size(), domain_size);

        const Fr omega = get_root_of_unity(domain_size);
        Fr point(1);
        for (size_t i = 0; i < domain_size; ++i) {
            EXPECT_EQ(evaluations[i].getStr(), candidate->eval(point).getStr());
            Fr::mul(point, point, omega);
        }
    }
}

TEST_F(PolyTest, PolynomialDivision)
{
    // Divide (x^2 + 3x + 2) / (x + 1) = x + 2
    Fr num_coeffs[3], denom_coeffs[2];
    num_coeffs[0].setStr("2", 10);   // constant term
    num_coeffs[1].setStr("3", 10);   // x
    num_coeffs[2].setStr("1", 10);   // x^2

    denom_coeffs[0].setStr("1", 10); // constant term
    denom_coeffs[1].setStr("1", 10); // x

    poly numerator, denominator;
    numerator.init_with_coeff(num_coeffs, 3);
    denominator.init_with_coeff(denom_coeffs, 2);

    poly result = numerator / denominator;
    const std::vector<Fr>& res_coeffs = result.coefficients();

    // Expect result to be x + 2: i.e., res_coeffs[0]=2, res_coeffs[1]=1
    ASSERT_EQ(res_coeffs.size(), 2);
    EXPECT_EQ(res_coeffs[0].getStr(), "2");
    EXPECT_EQ(res_coeffs[1].getStr(), "1");

    // Divide by higher degree: (x + 2) / (x^2 + 3x + 2) = 0
    Fr num2[2];
    num2[0].setStr("2", 10); // constant
    num2[1].setStr("1", 10); // x
    poly lowdeg;
    lowdeg.init_with_coeff(num2, 2);

    poly result2 = lowdeg / numerator;
    EXPECT_TRUE(result2.coefficients().empty());

    // Division by zero denominator yields zero poly (empty)
    poly zero;
    poly result3 = numerator / zero;
    EXPECT_TRUE(result3.coefficients().empty());
}

TEST_F(PolyTest, ConstantArithmetic_RightOperand)
{
    // p(x) = 1 + 2x + 3x^2
    poly p;
    p.init_with_coeff(std::vector<Fr>{Fr(1), Fr(2), Fr(3)});
    Fr c(5);

    poly add = p + c; // 6 + 2x + 3x^2
    ASSERT_EQ(add.coefficients().size(), 3u);
    EXPECT_EQ(add.coefficients()[0].getStr(), "6");
    EXPECT_EQ(add.coefficients()[1].getStr(), "2");
    EXPECT_EQ(add.coefficients()[2].getStr(), "3");

    poly sub = p - c; // -4 + 2x + 3x^2
    EXPECT_EQ(sub.coefficients().size(), 3u);
    EXPECT_EQ(sub.coefficients()[1].getStr(), "2");
    EXPECT_EQ(sub.coefficients()[2].getStr(), "3");

    poly mul = p * c; // 5 + 10x + 15x^2
    ASSERT_EQ(mul.coefficients().size(), 3u);
    EXPECT_EQ(mul.coefficients()[0].getStr(), "5");
    EXPECT_EQ(mul.coefficients()[1].getStr(), "10");
    EXPECT_EQ(mul.coefficients()[2].getStr(), "15");

    poly div = p / c; // multiply by inv(5)
    Fr x;
    x.setStr("7", 10);
    Fr expected = p.eval(x);
    Fr inv5;
    Fr::inv(inv5, c);
    Fr::mul(expected, expected, inv5);
    EXPECT_EQ(div.eval(x).getStr(), expected.getStr());
}

TEST_F(PolyTest, ConstantArithmetic_LeftOperand)
{
    poly p;
    p.init_with_coeff(std::vector<Fr>{Fr(1), Fr(2), Fr(3)});
    Fr c(5);

    poly add = c + p; // 6 + 2x + 3x^2
    ASSERT_EQ(add.coefficients().size(), 3u);
    EXPECT_EQ(add.coefficients()[0].getStr(), "6");
    EXPECT_EQ(add.coefficients()[1].getStr(), "2");
    EXPECT_EQ(add.coefficients()[2].getStr(), "3");

    poly sub = c - p; // 4 - 2x - 3x^2
    ASSERT_EQ(sub.coefficients().size(), 3u);
    EXPECT_EQ(sub.coefficients()[0].getStr(), "4");
    Fr neg2, neg3;
    Fr::neg(neg2, Fr(2));
    Fr::neg(neg3, Fr(3));
    EXPECT_EQ(sub.coefficients()[1].getStr(), neg2.getStr());
    EXPECT_EQ(sub.coefficients()[2].getStr(), neg3.getStr());

    poly mul = c * p; // 5 + 10x + 15x^2
    ASSERT_EQ(mul.coefficients().size(), 3u);
    EXPECT_EQ(mul.coefficients()[0].getStr(), "5");
    EXPECT_EQ(mul.coefficients()[1].getStr(), "10");
    EXPECT_EQ(mul.coefficients()[2].getStr(), "15");
}

// ---------------------------------------------------------------------------
// Lagrange interpolation tests
// ---------------------------------------------------------------------------

class LagrangeInterpolationTest : public ::testing::Test {
protected:
    void SetUp() override {
        init_groups();
    }
};

TEST_F(LagrangeInterpolationTest, InitAsVanishing_ZerosAtAllPoints)
{
    // Z(X) = (X - x[0])(X - x[1])...(X - x[n-1]) vanishes at all x[i]
    Fr x[3];
    x[0].setStr("0", 10);
    x[1].setStr("1", 10);
    x[2].setStr("2", 10);

    poly Z;
    Z.init_as_vanishing(x, 3);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(Z.eval(x[i]).isZero())
            << "Vanishing polynomial Z(x[" << i << "]) should be 0";
    }

    // Z(X) = X(X-1)(X-2) = X^3 - 3X^2 + 2X => coeffs [0, 2, -3, 1]
    ASSERT_EQ(Z.coefficients().size(), 4u);
    EXPECT_TRUE(Z.coefficients()[0].isZero());
}

TEST_F(LagrangeInterpolationTest, InitAsVanishing_EmptySet_IsOne)
{
    Fr x[1];  // dummy, n=0
    poly Z;
    Z.init_as_vanishing(x, 0);

    ASSERT_EQ(Z.coefficients().size(), 1u);
    EXPECT_TRUE(Z.coefficients()[0].isOne());
}

TEST_F(LagrangeInterpolationTest, SinglePoint_ConstantPolynomial)
{
    // Single point (x0, y0) -> P(x) = y0 for all x
    Fr x[1], y[1];
    x[0].setStr("42", 10);
    y[0].setStr("7", 10);

    poly p;
    p.init_with_interpolation(x, y, 1);

    ASSERT_EQ(p.coefficients().size(), 1u);
    EXPECT_EQ(p.coefficients()[0].getStr(), "7");

    // Eval at any point should be 7
    Fr xx;
    xx.setStr("0", 10);
    EXPECT_EQ(p.eval(xx).getStr(), "7");
    xx.setStr("100", 10);
    EXPECT_EQ(p.eval(xx).getStr(), "7");
}

TEST_F(LagrangeInterpolationTest, TwoPoints_LinearPolynomial)
{
    // (0, 1) and (1, 3) -> P(x) = 1 + 2x
    Fr x[2], y[2];
    x[0].setStr("0", 10); y[0].setStr("1", 10);
    x[1].setStr("1", 10); y[1].setStr("3", 10);

    poly p;
    p.init_with_interpolation(x, y, 2);

    EXPECT_EQ(p.eval(x[0]).getStr(), y[0].getStr());
    EXPECT_EQ(p.eval(x[1]).getStr(), y[1].getStr());

    // P(0) = 1, P(1) = 3 -> linear: P(x) = 1 + 2x
    ASSERT_EQ(p.coefficients().size(), 2u);
    EXPECT_EQ(p.coefficients()[0].getStr(), "1");
    EXPECT_EQ(p.coefficients()[1].getStr(), "2");
}

TEST_F(LagrangeInterpolationTest, ThreePoints_QuadraticPolynomial)
{
    // (0, 1), (1, 2), (2, 5) -> P(x) = 1 + x^2 (quadratic through these points)
    Fr x[3], y[3];
    x[0].setStr("0", 10); y[0].setStr("1", 10);
    x[1].setStr("1", 10); y[1].setStr("2", 10);
    x[2].setStr("2", 10); y[2].setStr("5", 10);

    poly p;
    p.init_with_interpolation(x, y, 3);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(p.eval(x[i]).getStr(), y[i].getStr())
            << "P(x[" << i << "]) should equal y[" << i << "]";
    }

    // Verify coefficients: P(x) = 1 + x^2 => coeffs [1, 0, 1]
    ASSERT_EQ(p.coefficients().size(), 3u);
    EXPECT_EQ(p.coefficients()[0].getStr(), "1");
    EXPECT_TRUE(p.coefficients()[1].isZero());
    EXPECT_EQ(p.coefficients()[2].getStr(), "1");
}

TEST_F(LagrangeInterpolationTest, InterpolationPassesThroughAllPoints)
{
    // Arbitrary points: degree 3 polynomial
    std::vector<Fr> x = {Fr(0), Fr(1), Fr(2), Fr(3)};
    std::vector<Fr> y;
    y.push_back(Fr(1));
    Fr y1; y1.setStr("5", 10); y.push_back(y1);
    Fr y2; y2.setStr("17", 10); y.push_back(y2);
    Fr y3; y3.setStr("43", 10); y.push_back(y3);

    poly p = poly_from_interpolation(x, y);

    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_EQ(p.eval(x[i]).getStr(), y[i].getStr())
            << "P(x[" << i << "]) should equal y[" << i << "]";
    }
}

TEST_F(LagrangeInterpolationTest, Roundtrip_CoeffToEvalToInterpolation)
{
    // Build polynomial from coefficients, sample at distinct points, interpolate, compare
    Fr coeffs[4];
    coeffs[0].setStr("1", 10);
    coeffs[1].setStr("2", 10);
    coeffs[2].setStr("3", 10);
    coeffs[3].setStr("4", 10);

    poly original;
    original.init_with_coeff(coeffs, 4);

    // Sample at x = 0, 1, 2, 3
    std::vector<Fr> x = {Fr(0), Fr(1), Fr(2), Fr(3)};
    std::vector<Fr> y(4);
    for (size_t i = 0; i < 4; ++i) {
        y[i] = original.eval(x[i]);
    }

    poly recovered = poly_from_interpolation(x, y);

    const std::vector<Fr>& orig_c = original.coefficients();
    const std::vector<Fr>& rec_c = recovered.coefficients();
    ASSERT_EQ(orig_c.size(), rec_c.size());
    for (size_t i = 0; i < orig_c.size(); ++i) {
        EXPECT_EQ(orig_c[i].getStr(), rec_c[i].getStr())
            << "coefficient " << i << " should match";
    }
}

TEST_F(LagrangeInterpolationTest, ZeroPoints_ProducesZeroPoly)
{
    Fr x[1], y[1];  // dummy, n=0
    poly p;
    p.init_with_interpolation(x, y, 0);

    EXPECT_TRUE(p.coefficients().empty());
}

TEST_F(LagrangeInterpolationTest, VectorOverload_ThrowsOnMismatchedSizes)
{
    std::vector<Fr> x = {Fr(0), Fr(1)};
    std::vector<Fr> y = {Fr(1)};  // size 1, x has size 2

    EXPECT_THROW(poly_from_interpolation(x, y), std::runtime_error);
}

TEST_F(LagrangeInterpolationTest, InitWithInterpolation_AndPolyFromInterpolation_Agree)
{
    Fr x[3], y[3];
    x[0].setStr("0", 10); y[0].setStr("1", 10);
    x[1].setStr("1", 10); y[1].setStr("4", 10);
    x[2].setStr("2", 10); y[2].setStr("9", 10);

    poly p1, p2;
    p1.init_with_interpolation(x, y, 3);
    p2 = poly_from_interpolation(x, y, 3);

    ASSERT_EQ(p1.coefficients().size(), p2.coefficients().size());
    for (size_t i = 0; i < p1.coefficients().size(); ++i) {
        EXPECT_EQ(p1.coefficients()[i].getStr(), p2.coefficients()[i].getStr());
    }
}
