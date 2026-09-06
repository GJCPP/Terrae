#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "algebra_utils.hpp"
#include "column_replication.hpp"
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
    if (coeffs.back().isZero()) {
        coeffs.back() = Fr(1);
    }

    poly p;
    p.init_with_coeff(coeffs);
    return p;
}

} // namespace

class ColumnReplicationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ColumnReplicationTest, HelpersRejectNonPowerOfTwoShapes)
{
    poly f = poly_from_ints({1, 2, 3});
    EXPECT_THROW(static_cast<void>(replication_row_vanishing_factor(3, 2)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(replication_row_vanishing_factor(4, 3)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(f.column_replicated(3, 2)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(f.column_replicated(4, 3)), std::runtime_error);
}

TEST_F(ColumnReplicationTest, ReplicatedPolynomialMatchesMatrixLayout)
{
    const poly f = poly_from_ints({5, 2, 1});
    const size_t n = 4;
    const size_t m = 2;
    const poly f_hat = f.column_replicated(n, m);

    const Fr omega_nm = get_root_of_unity(n * m);
    Fr x = Fr(1);
    std::vector<Fr> evals;
    for (size_t i = 0; i < n * m; ++i) {
        evals.push_back(f_hat.eval(x));
        Fr::mul(x, x, omega_nm);
    }

    ASSERT_EQ(evals.size(), 8u);
    EXPECT_EQ(evals[0], evals[1]);
    EXPECT_EQ(evals[2], evals[3]);
    EXPECT_EQ(evals[4], evals[5]);
    EXPECT_EQ(evals[6], evals[7]);

    const Fr omega_n = get_root_of_unity(n);
    Fr row_x = Fr(1);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(evals[i * m], f.eval(row_x));
        Fr::mul(row_x, row_x, omega_n);
    }
}

TEST_F(ColumnReplicationTest, ReplicatedPolynomialUsesResidueModuloVanishing)
{
    const size_t n = 4;
    const size_t m = 2;

    poly f = poly_from_ints({1, 2, 3});
    f = f + vanishing_poly(n) * poly_from_ints({7, 5, 4});

    const poly f_hat = f.column_replicated(n, m);
    const poly residue = div(f, vanishing_poly(n)).second;

    const Fr omega_nm = get_root_of_unity(n * m);
    const Fr omega_n = get_root_of_unity(n);
    Fr x_nm = Fr(1);
    Fr x_n = Fr(1);
    for (size_t i = 0; i < n; ++i) {
        const Fr expected = residue.eval(x_n);
        EXPECT_EQ(f_hat.eval(x_nm), expected);
        EXPECT_EQ(f_hat.eval(omega_nm * x_nm), expected);
        Fr::mul(x_nm, x_nm, omega_nm * omega_nm);
        Fr::mul(x_n, x_n, omega_n);
    }
}

TEST_F(ColumnReplicationTest, ReplicationConstraintsMaterializeToZeroOnDomains)
{
    const size_t n = 8;
    const size_t m = 4;
    const size_t nm = n * m;

    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    const poly f = poly_from_ints({3, 0, 2, 1});
    const poly f_hat = f.column_replicated(n, m);

    const virCommitment f_com(kzg.commit(f));
    const virCommitment f_hat_com(kzg.commit(f_hat));
    const virCommitment row_zero =
        (x_scale(f_hat_com, fr_from_int(1) / get_root_of_unity(nm)) - f_hat_com) * virCommitment(vanishing_poly(n));
    const virCommitment column_zero = f_hat_com - f_com;

    const std::shared_ptr<poly> row_mat = kzg.materialize(row_zero);
    const std::shared_ptr<poly> col_mat = kzg.materialize(column_zero);
    ASSERT_NE(row_mat, nullptr);
    ASSERT_NE(col_mat, nullptr);
    EXPECT_TRUE(div(*row_mat, vanishing_poly(nm)).second.is_zero());
    EXPECT_TRUE(div(*col_mat, vanishing_poly(n)).second.is_zero());

    const Fr probe = fr_from_int(13);
    EXPECT_EQ(kzg.open(row_zero, probe), row_mat->eval(probe));
    EXPECT_EQ(kzg.open(column_zero, probe), col_mat->eval(probe));

    Fr x_nm = Fr(1);
    const Fr omega_nm = get_root_of_unity(nm);
    for (size_t i = 0; i < nm; ++i) {
        EXPECT_TRUE(row_mat->eval(x_nm).isZero()) << "row i=" << i;
        Fr::mul(x_nm, x_nm, omega_nm);
    }

    Fr x_n = Fr(1);
    const Fr omega_n = get_root_of_unity(n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_TRUE(col_mat->eval(x_n).isZero()) << "col i=" << i;
        Fr::mul(x_n, x_n, omega_n);
    }
}

TEST_F(ColumnReplicationTest, FinalizeSucceedsForValidReplication)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));
    zeroProtocol zero;
    zero.setup(kzg);

    const size_t n = 8;
    const size_t m = 4;
    const poly f = poly_from_ints({3, 0, 2, 1});
    const poly f_hat = f.column_replicated(n, m);

    columnReplicationProtocol protocol;
    protocol.setup(kzg, zero);

    const virCommitment f_com(kzg.commit(f));
    const virCommitment f_hat_com(kzg.commit(f_hat));
    protocol.add_check(f_com, f_hat_com, n, m);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ColumnReplicationTest, UnderlyingZeroCheckSupportsRowAndColumnConstraints)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    const size_t n = 8;
    const size_t m = 4;
    const size_t nm = n * m;
    const Fr row_shift = fr_from_int(1) / get_root_of_unity(nm);

    const poly f = poly_from_ints({3, 0, 2, 1});
    const poly f_hat = f.column_replicated(n, m);
    const virCommitment f_com(kzg.commit(f));
    const virCommitment f_hat_com(kzg.commit(f_hat));
    virCommitment row_zero = (x_scale(f_hat_com, row_shift) - f_hat_com) * virCommitment(vanishing_poly(n));
    virCommitment column_zero = f_hat_com - f_com;

    {
        zeroProtocol protocol;
        protocol.setup(kzg);
        protocol.add_check(row_zero, nm);
        EXPECT_TRUE(protocol.finalize());
    }
    {
        zeroProtocol protocol;
        protocol.setup(kzg);
        protocol.add_check(column_zero, n);
        EXPECT_TRUE(protocol.finalize());
    }
    {
        zeroProtocol protocol;
        protocol.setup(kzg);
        protocol.add_check(row_zero, nm);
        protocol.add_check(column_zero, n);
        EXPECT_TRUE(protocol.finalize());
    }
}

TEST_F(ColumnReplicationTest, FinalizeFailsForBrokenRows)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));
    zeroProtocol zero;
    zero.setup(kzg);

    const size_t n = 8;
    const size_t m = 4;
    const poly f = poly_from_ints({3, 1, 4});
    poly f_hat = f.column_replicated(n, m);
    f_hat = f_hat + poly_from_ints({0, 0, 0, 0, 0, 1});

    columnReplicationProtocol protocol;
    protocol.setup(kzg, zero);

    const virCommitment f_com(kzg.commit(f));
    const virCommitment f_hat_com(kzg.commit(f_hat));
    protocol.add_check(f_com, f_hat_com, n, m);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_FALSE(zero.finalize());
}

TEST_F(ColumnReplicationTest, FinalizeFailsForWrongFirstColumn)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));
    zeroProtocol zero;
    zero.setup(kzg);

    const size_t n = 8;
    const size_t m = 2;
    poly f = poly_from_ints({1, 5, 2});
    const poly f_hat = f.column_replicated(n, m);
    f = f + fr_from_int(7);

    columnReplicationProtocol protocol;
    protocol.setup(kzg, zero);

    const virCommitment f_com(kzg.commit(f));
    const virCommitment f_hat_com(kzg.commit(f_hat));
    protocol.add_check(f_com, f_hat_com, n, m);

    EXPECT_TRUE(protocol.finalize());
    EXPECT_FALSE(zero.finalize());
}

TEST_F(ColumnReplicationTest, FinalizeSupportsBatchedChecks)
{
    std::mt19937 rng(20260327);

    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(512, 512));
    zeroProtocol zero;
    zero.setup(kzg);

    columnReplicationProtocol protocol;
    protocol.setup(kzg, zero);

    const std::vector<std::pair<size_t, size_t>> shapes = {
        {4, 2},
        {8, 4},
        {16, 2},
    };

    for (const auto& [n, m] : shapes) {
        const poly f = random_poly(rng, static_cast<int>(n) - 1, 10);
        const poly f_hat = f.column_replicated(n, m);
        protocol.add_check(virCommitment(kzg.commit(f)), virCommitment(kzg.commit(f_hat)), n, m);
    }

    EXPECT_TRUE(protocol.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(ColumnReplicationTest, FinalizeCannotBeCalledTwice)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));
    zeroProtocol zero;
    zero.setup(kzg);

    const poly f = poly_from_ints({1, 2, 3});
    const poly f_hat = f.column_replicated(4, 2);

    columnReplicationProtocol protocol;
    protocol.setup(kzg, zero);
    protocol.add_check(virCommitment(kzg.commit(f)), virCommitment(kzg.commit(f_hat)), 4, 2);

    EXPECT_NO_THROW(static_cast<void>(protocol.finalize()));
    EXPECT_THROW(protocol.finalize(), std::runtime_error);
}
