#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "kzg.hpp"
#include "lagrange_utils.hpp"
#include "matrix_utils.hpp"
#include "polynomial.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace mcl::bn;

namespace {

std::vector<Fr> evaluations(const poly& p, size_t domain_size)
{
    return p.evaluations_on_domain(domain_size);
}

} // namespace

class LagrangeUtilsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(LagrangeUtilsTest, ZeroBasisPolynomialMatchesIndicator)
{
    const poly basis = subgroup_zero_lagrange_polynomial(4);
    EXPECT_EQ(evaluations(basis, 4),
              (std::vector<Fr>{Fr(1), Fr(0), Fr(0), Fr(0)}));
}

TEST_F(LagrangeUtilsTest, ShiftedBasisPolynomialMatchesIndicator)
{
    const poly basis = subgroup_lagrange_polynomial(4, 2);
    EXPECT_EQ(evaluations(basis, 4),
              (std::vector<Fr>{Fr(0), Fr(0), Fr(1), Fr(0)}));
}

TEST_F(LagrangeUtilsTest, VirtualExpressionMaterializesToExpectedBasis)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    const virCommitment expression = subgroup_lagrange_expression(4, 3);
    EXPECT_EQ(kzg.materialize(expression)->evaluations_on_domain(4),
              (std::vector<Fr>{Fr(0), Fr(0), Fr(0), Fr(1)}));
}

TEST_F(LagrangeUtilsTest, MatrixRowSelectorExpressionMatchesExpectedRow)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    const virCommitment expression = matrix_row_selector_expression(4, 2, 2);
    EXPECT_EQ(kzg.materialize(expression)->evaluations_on_domain(8),
              (std::vector<Fr>{Fr(0), Fr(0), Fr(0), Fr(0), Fr(1), Fr(1), Fr(0), Fr(0)}));
}

TEST_F(LagrangeUtilsTest, SelectorExpressionPreferEvalMatchesMaterializedPolynomial)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    const virCommitment expression = subgroup_lagrange_selector_expression(8, {1, 3, 6});
    EXPECT_EQ(kzg.evaluations(expression, 8, kzgProtocol::EvaluationMode::PreferEval),
              kzg.materialize(expression)->evaluations_on_domain(8));
}

TEST_F(LagrangeUtilsTest, MatrixColumnIndexExpressionMatchesExpectedColumns)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    const virCommitment expression = matrix_column_index_expression(2, 4);
    EXPECT_EQ(kzg.evaluations(expression, 8, kzgProtocol::EvaluationMode::PreferEval),
              (std::vector<Fr>{Fr(0), Fr(1), Fr(2), Fr(3), Fr(0), Fr(1), Fr(2), Fr(3)}));
}
