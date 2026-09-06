#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
#include "proof_runtime.hpp"
#include "shuffle.hpp"
#include "sum_check.hpp"
#include "test_kzg_srs.hpp"
#include "zero_check.hpp"
#include "zkboost/primitives/histogram_ram.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
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

class HistogramRamFixture {
public:
    kzgProtocol kzg;
    zeroProtocol zero;
    sumProtocol sum;
    lookupProtocol lookup;
    lessCheckProtocol less;
    shuffleProtocol shuffle;
    zkboostHistogramRamProtocol histogram;

    HistogramRamFixture()
    {
        kzg.setup(testing_kzg_public_parameters(65536, 65536));
        zero.setup(kzg);
        sum.setup(kzg, zero);
        lookup.setup(kzg, zero, sum);
        less.setup(kzg, lookup, zero);
        shuffle.setup(kzg, lookup);
        histogram.setup(kzg, zero, less, shuffle);
    }
};

} // namespace

class ZkboostHistogramRamProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(ZkboostHistogramRamProtocolTest, ComputesTraceShape)
{
    const zkboostHistogramRamShape shape = zkboost_histogram_ram_shape(4, 4);
    EXPECT_EQ(shape.query_domain_size, 4u);
    EXPECT_EQ(shape.bucket_count, 4u);
    EXPECT_EQ(shape.trace_domain_size, 16u);

    EXPECT_THROW((void)zkboost_histogram_ram_shape(3, 4), std::runtime_error);
    EXPECT_THROW((void)zkboost_histogram_ram_shape(4, 3), std::runtime_error);
}

TEST_F(ZkboostHistogramRamProtocolTest, ValidHistogramFinalizes)
{
    HistogramRamFixture fixture;

    const poly indices = poly_from_domain_ints({0, 2, 1, 2});
    const poly values = poly_from_domain_ints({5, 7, 11, 13});
    const poly sums = poly_from_domain_ints({5, 11, 20, 0});

    fixture.histogram.add_check(virCommitment(fixture.kzg.commit(indices)),
                                virCommitment(fixture.kzg.commit(values)),
                                virCommitment(fixture.kzg.commit(sums)),
                                4,
                                4);

    EXPECT_TRUE(fixture.histogram.finalize());
    EXPECT_TRUE(fixture.shuffle.finalize());
    EXPECT_TRUE(fixture.less.finalize());
    EXPECT_TRUE(fixture.lookup.finalize());
    EXPECT_TRUE(fixture.sum.finalize());
    EXPECT_TRUE(fixture.zero.finalize());
    EXPECT_TRUE(fixture.kzg.finalize());
}

TEST_F(ZkboostHistogramRamProtocolTest, WrongHistogramFailsMemoryConsistency)
{
    HistogramRamFixture fixture;

    const poly indices = poly_from_domain_ints({0, 2, 1, 2});
    const poly values = poly_from_domain_ints({5, 7, 11, 13});
    const poly wrong_sums = poly_from_domain_ints({5, 11, 21, 0});

    fixture.histogram.add_check(virCommitment(fixture.kzg.commit(indices)),
                                virCommitment(fixture.kzg.commit(values)),
                                virCommitment(fixture.kzg.commit(wrong_sums)),
                                4,
                                4);

    EXPECT_TRUE(fixture.histogram.finalize());
    EXPECT_TRUE(fixture.shuffle.finalize());
    EXPECT_TRUE(fixture.less.finalize());
    EXPECT_TRUE(fixture.lookup.finalize());
    EXPECT_TRUE(fixture.sum.finalize());
    EXPECT_FALSE(fixture.zero.finalize());
    EXPECT_TRUE(fixture.kzg.finalize());
}

TEST_F(ZkboostHistogramRamProtocolTest, RejectsOutOfRangeIndexWitness)
{
    HistogramRamFixture fixture;

    const poly indices = poly_from_domain_ints({0, 2, 4, 2});
    const poly values = poly_from_domain_ints({5, 7, 11, 13});
    const poly sums = poly_from_domain_ints({5, 0, 20, 0});

    EXPECT_THROW(
        fixture.histogram.add_check(virCommitment(fixture.kzg.commit(indices)),
                                    virCommitment(fixture.kzg.commit(values)),
                                    virCommitment(fixture.kzg.commit(sums)),
                                    4,
                                    4),
        std::runtime_error);
}

TEST_F(ZkboostHistogramRamProtocolTest, SupportsProofRuntimeAcrossModes)
{
    const kzgPublicParameters public_params = testing_kzg_public_parameters(65536, 65536);
    const poly indices = poly_from_domain_ints({0, 2, 1, 2});
    const poly values = poly_from_domain_ints({5, 7, 11, 13});
    const poly sums = poly_from_domain_ints({5, 11, 20, 0});

    ProofRuntime prover_runtime = ProofRuntime::prover(public_params);
    ProofProgram prover_program(prover_runtime);
    const virCommitment prover_indices =
        prover_program.bind_virtual(WitnessCommitmentInput::prover("histogram_ram.indices", indices, 3));
    const virCommitment prover_values =
        prover_program.bind_virtual(WitnessCommitmentInput::prover("histogram_ram.values", values, 3));
    const virCommitment prover_sums =
        prover_program.bind_virtual(WitnessCommitmentInput::prover("histogram_ram.sums", sums, 3));
    prover_program.zkboost_histogram_ram().add_check(prover_indices, prover_values, prover_sums, 4, 4);
    ASSERT_TRUE(prover_program.finalize());
    const SerializedProof proof = prover_runtime.take_proof();
    ASSERT_FALSE(proof.bytes.empty());

    ProofRuntime verifier_runtime = ProofRuntime::verifier(public_params, proof);
    ProofProgram verifier_program(verifier_runtime);
    const virCommitment verifier_indices =
        verifier_program.bind_virtual(WitnessCommitmentInput::verifier("histogram_ram.indices", 3));
    const virCommitment verifier_values =
        verifier_program.bind_virtual(WitnessCommitmentInput::verifier("histogram_ram.values", 3));
    const virCommitment verifier_sums =
        verifier_program.bind_virtual(WitnessCommitmentInput::verifier("histogram_ram.sums", 3));
    verifier_program.zkboost_histogram_ram().add_check(verifier_indices, verifier_values, verifier_sums, 4, 4);

    EXPECT_TRUE(verifier_program.finalize());
    EXPECT_TRUE(verifier_runtime.proof_fully_consumed());
}
