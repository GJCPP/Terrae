/**
 * Tests for sparse behavior in the unified poly class.
 */
#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"

#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace mcl::bn;

namespace {

Fr fr(int n)
{
    Fr r;
    r.setStr(std::to_string(n).c_str(), 10);
    return r;
}

bool same_coeff(const poly& a, const poly& b)
{
    const auto& ac = a.coefficients();
    const auto& bc = b.coefficients();
    if (ac.size() != bc.size()) return false;
    for (size_t i = 0; i < ac.size(); ++i) {
        if (ac[i] != bc[i]) return false;
    }
    return true;
}

poly random_sparse_poly(std::mt19937_64& rng, size_t max_deg, size_t term_count)
{
    std::uniform_int_distribution<size_t> exp_dist(0, max_deg);
    std::uniform_int_distribution<int> coeff_dist(-100, 100);
    std::vector<std::pair<size_t, Fr>> terms;
    terms.reserve(term_count);
    for (size_t i = 0; i < term_count; ++i) {
        terms.push_back({exp_dist(rng), fr(coeff_dist(rng))});
    }
    return poly_from_terms(terms);
}

poly random_dense_poly(std::mt19937_64& rng, size_t degree)
{
    std::uniform_int_distribution<int> coeff_dist(-100, 100);
    std::vector<Fr> coeffs(degree + 1, Fr(0));
    for (size_t i = 0; i <= degree; ++i) {
        coeffs[i] = fr(coeff_dist(rng));
    }
    poly p;
    p.init_with_coeff(coeffs);
    return p;
}

} // namespace

class SparsePolyTest : public ::testing::Test {
protected:
    void SetUp() override { init_groups(); }
};

TEST_F(SparsePolyTest, InitWithTerms_PreservesSparseRepresentation)
{
    poly p = poly_from_terms({{0, fr(1)}, {2, fr(1)}, {5, fr(1)}});
    ASSERT_TRUE(p.is_sparse());
    const auto& t = p.terms();
    ASSERT_EQ(t.size(), 3u);
    EXPECT_EQ(t[0].first, 0u);
    EXPECT_EQ(t[1].first, 2u);
    EXPECT_EQ(t[2].first, 5u);
}

TEST_F(SparsePolyTest, AddTerm_MergesAndCancels)
{
    poly p;
    p.add_term(4, fr(3));
    p.add_term(4, fr(-1));
    p.add_term(4, fr(-2));
    p.add_term(7, fr(0));
    EXPECT_TRUE(p.terms().empty());
    EXPECT_TRUE(p.is_zero());
}

TEST_F(SparsePolyTest, VanishingAndIdentityHelpersBuildSparsePolys)
{
    poly z = vanishing_poly(8);
    poly id = identity_poly();
    ASSERT_TRUE(z.is_sparse());
    ASSERT_TRUE(id.is_sparse());
    ASSERT_EQ(z.terms().size(), 2u);
    ASSERT_EQ(id.terms().size(), 1u);
}

TEST_F(SparsePolyTest, SparseDenseAdditionMatchesDenseGroundTruth)
{
    poly sparse = poly_from_terms({{0, fr(2)}, {3, fr(4)}});
    poly dense;
    dense.init_with_coeff(std::vector<Fr>{fr(1), fr(1), fr(1), fr(1)});

    poly sum = sparse + dense;
    poly expected;
    expected.init_with_coeff(std::vector<Fr>{fr(3), fr(1), fr(1), fr(5)});
    EXPECT_TRUE(same_coeff(sum, expected));
}

TEST_F(SparsePolyTest, SparseDenseMultiplicationMatchesDenseGroundTruth)
{
    poly sparse = poly_from_terms({{0, fr(2)}, {3, fr(4)}});
    poly dense;
    dense.init_with_coeff(std::vector<Fr>{fr(1), fr(2), fr(3)});

    poly product = sparse * dense;
    poly expected;
    expected.init_with_coeff(std::vector<Fr>{fr(2), fr(4), fr(6), fr(4), fr(8), fr(12)});
    EXPECT_TRUE(same_coeff(product, expected));
}

TEST_F(SparsePolyTest, DenseBySparseDivisionReturnsExpectedQuotientAndRemainder)
{
    poly numerator;
    numerator.init_with_coeff(std::vector<Fr>{fr(2), fr(3), fr(1)});
    poly denominator = poly_from_terms({{0, fr(1)}, {1, fr(1)}});

    const auto [q, r] = div(numerator, denominator);
    poly recomposed = q * denominator + r;
    EXPECT_TRUE(same_coeff(recomposed, numerator));
}

TEST_F(SparsePolyTest, EvalMatchesDenseCoefficients)
{
    poly p = poly_from_terms({{0, fr(1)}, {2, fr(1)}, {5, fr(1)}});
    Fr x = fr(3);
    EXPECT_EQ(p.eval(x).getStr(), "253");
}

TEST_F(SparsePolyTest, RandomizedSparseDenseOperatorsMatchDenseGroundTruth)
{
    std::mt19937_64 rng(20260327);
    for (int it = 0; it < 80; ++it) {
        poly sparse = random_sparse_poly(rng, 300, 32);
        poly dense = random_dense_poly(rng, 64);

        poly sparse_dense;
        sparse_dense.init_with_coeff(sparse.coefficients());
        poly dense_dense;
        dense_dense.init_with_coeff(dense.coefficients());

        EXPECT_TRUE(same_coeff(sparse + dense, sparse_dense + dense_dense)) << "iter=" << it << " add";
        EXPECT_TRUE(same_coeff(sparse - dense, sparse_dense - dense_dense)) << "iter=" << it << " sub";
        EXPECT_TRUE(same_coeff(sparse * dense, sparse_dense * dense_dense)) << "iter=" << it << " mul";
    }
}
