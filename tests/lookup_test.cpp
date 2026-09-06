#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "test_kzg_srs.hpp"
#include "kzg.hpp"
#include "lookup.hpp"
#include "polynomial.hpp"
#include "sum_check.hpp"
#include "zero_check.hpp"

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

std::vector<Fr> evals_from_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return evals;
}

class TestableLookupProtocol : public lookupProtocol {
public:
    const lookupProver& prover_ref() const { return prover; }
    const lookupVerifier& verifier_ref() const { return verifier; }
};

} // namespace

class LookupProtocolTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(LookupProtocolTest, AddCheckRejectsInvalidDomainSizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(64, 64));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    const virCommitment v(fr_from_int(0));
    EXPECT_THROW(lookup.add_check(v, v, 3, 4), std::runtime_error);
    EXPECT_THROW(lookup.add_check(v, v, 4, 0), std::runtime_error);
}

TEST_F(LookupProtocolTest, ValidLookupFinalizes)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    const poly f = poly_from_domain_ints({5, 7, 7, 11});
    const poly t = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});

    lookup.add_check(virCommitment(kzg.commit(f)), virCommitment(kzg.commit(t)), 4, 8);

    EXPECT_TRUE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LookupProtocolTest, InvalidQueryFails)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    lookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    const poly f = poly_from_domain_ints({5, 7, 29, 11});
    const poly t = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});

    lookup.add_check(virCommitment(kzg.commit(f)), virCommitment(kzg.commit(t)), 4, 8);

    EXPECT_FALSE(lookup.finalize());
    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LookupProtocolTest, ProveAndVerifyWithSerializedProof)
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

    const poly f = poly_from_domain_ints({5, 7, 7, 11});
    const poly t = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    const kzgCommitment prover_f_commitment = prover_kzg.commit(f);
    const kzgCommitment prover_t_commitment = prover_kzg.commit(t);

    prover_lookup.add_check(virCommitment(prover_f_commitment), virCommitment(prover_t_commitment), 4, 8);

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

    const kzgCommitment verifier_f_commitment = verifier_kzg.adopt_commitment(prover_f_commitment.com);
    const kzgCommitment verifier_t_commitment = verifier_kzg.adopt_commitment(prover_t_commitment.com);
    verifier_lookup.add_check(virCommitment(verifier_f_commitment), virCommitment(verifier_t_commitment), 4, 8);

    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(LookupProtocolTest, ProveAndVerifyWithSerializedProofBatchesChecks)
{
    kzgProtocol prover_kzg;
    prover_kzg.setup(testing_kzg_public_parameters(512, 512));
    auto prove_session = ProofSession::prover();
    prover_kzg.set_proof_session(prove_session);

    zeroProtocol prover_zero;
    prover_zero.setup(prover_kzg);

    sumProtocol prover_sum;
    prover_sum.setup(prover_kzg, prover_zero);

    lookupProtocol prover_lookup;
    prover_lookup.setup(prover_kzg, prover_zero, prover_sum);

    const poly t = poly_from_domain_ints({2, 3, 5, 7});
    const poly f1 = poly_from_domain_ints({2, 3});
    const poly f2 = poly_from_domain_ints({7});
    const kzgCommitment prover_f1_commitment = prover_kzg.commit(f1);
    const kzgCommitment prover_f2_commitment = prover_kzg.commit(f2);
    const kzgCommitment prover_t_commitment = prover_kzg.commit(t);

    prover_lookup.add_check(virCommitment(prover_f1_commitment), virCommitment(prover_t_commitment), 2, 4);
    prover_lookup.add_check(virCommitment(prover_f2_commitment), virCommitment(prover_t_commitment), 1, 4);

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

    const kzgCommitment verifier_f1_commitment = verifier_kzg.adopt_commitment(prover_f1_commitment.com);
    const kzgCommitment verifier_f2_commitment = verifier_kzg.adopt_commitment(prover_f2_commitment.com);
    const kzgCommitment verifier_t_commitment = verifier_kzg.adopt_commitment(prover_t_commitment.com);

    verifier_lookup.add_check(virCommitment(verifier_f1_commitment), virCommitment(verifier_t_commitment), 2, 4);
    verifier_lookup.add_check(virCommitment(verifier_f2_commitment), virCommitment(verifier_t_commitment), 1, 4);

    EXPECT_TRUE(verifier_lookup.finalize());
    EXPECT_TRUE(verifier_sum.finalize());
    EXPECT_TRUE(verifier_zero.finalize());
    EXPECT_TRUE(verifier_kzg.finalize());
    EXPECT_TRUE(verify_session->proof_fully_consumed());
    EXPECT_EQ(prove_session->transcript().state(), verify_session->transcript().state());
}

TEST_F(LookupProtocolTest, ProverGeneratesMultiplicityCommitment)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    TestableLookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    const poly f = poly_from_domain_ints({5, 7, 7, 11});
    const poly t = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});

    lookup.add_check(virCommitment(kzg.commit(f)), virCommitment(kzg.commit(t)), 4, 8);
    EXPECT_TRUE(lookup.finalize());

    ASSERT_EQ(lookup.prover_ref().checks.size(), 1u);
    ASSERT_EQ(lookup.verifier_ref().checks.size(), 1u);
    ASSERT_EQ(lookup.verifier_ref().multiplicities.size(), 1u);

    const lookupCheckRecord& check = lookup.prover_ref().checks[0];
    std::shared_ptr<poly> c_poly = kzg.materialize(lookup.verifier_ref().multiplicities[0]);
    EXPECT_EQ(c_poly->evaluations_on_domain(8), evals_from_ints({0, 1, 2, 1, 0, 0, 0, 0}));
    EXPECT_EQ(check.query_claim, check.table_claim);

    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LookupProtocolTest, BatchesChecksIntoOneCombinedLookup)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(256, 256));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    TestableLookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    const poly table = poly_from_domain_ints({2, 3, 5, 7});
    const poly f1 = poly_from_domain_ints({2, 3});
    const poly f2 = poly_from_domain_ints({7});
    const virCommitment table_commitment(kzg.commit(table));

    lookup.add_check(virCommitment(kzg.commit(f1)), table_commitment, 2, 4);
    lookup.add_check(virCommitment(kzg.commit(f2)), table_commitment, 1, 4);

    EXPECT_TRUE(lookup.finalize());

    ASSERT_EQ(lookup.prover_ref().checks.size(), 1u);
    ASSERT_EQ(lookup.verifier_ref().checks.size(), 1u);
    EXPECT_EQ(lookup.prover_ref().checks[0].query_domain_size, 4u);
    EXPECT_EQ(lookup.prover_ref().checks[0].table_domain_size, 4u);
    EXPECT_EQ(lookup.verifier_ref().checks[0].query_domain_size, 4u);
    EXPECT_EQ(lookup.verifier_ref().checks[0].table_domain_size, 4u);
    EXPECT_FALSE(lookup.prover_ref().checks[0].challenge.isZero());
    EXPECT_FALSE(lookup.verifier_ref().checks[0].challenge.isZero());
    EXPECT_EQ(lookup.verifier_ref().checks[0].query_claim, lookup.verifier_ref().checks[0].table_claim);

    EXPECT_TRUE(sum.finalize());
    EXPECT_TRUE(zero.finalize());
    EXPECT_TRUE(kzg.finalize());
}

TEST_F(LookupProtocolTest, FinalizeProducesNonzeroChallenge)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    TestableLookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    const virCommitment f(kzg.commit(poly_from_domain_ints({5, 7, 7, 11})));
    const virCommitment t(kzg.commit(poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23})));
    lookup.add_check(f, t, 4, 8);
    EXPECT_TRUE(lookup.finalize());

    ASSERT_EQ(lookup.verifier_ref().checks.size(), 1u);
    EXPECT_FALSE(lookup.verifier_ref().checks[0].challenge.isZero());
}

TEST_F(LookupProtocolTest, ProofModeBindsMultiplicityBeforeChallenge)
{
    kzgProtocol kzg;
    kzg.setup(testing_kzg_public_parameters(128, 128));
    auto prove_session = ProofSession::prover();
    kzg.set_proof_session(prove_session);

    zeroProtocol zero;
    zero.setup(kzg);

    sumProtocol sum;
    sum.setup(kzg, zero);

    TestableLookupProtocol lookup;
    lookup.setup(kzg, zero, sum);

    const virCommitment f(kzg.commit(poly_from_domain_ints({5, 7, 7, 11})));
    const virCommitment t(kzg.commit(poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23})));

    lookup.add_check(f, t, 4, 8);
    EXPECT_TRUE(lookup.finalize());

    ASSERT_EQ(lookup.verifier_ref().checks.size(), 1u);
    const lookupCheckRecord& record = lookup.verifier_ref().checks[0];

    ProofTranscript expected;
    expected.absorb_string("kzg.commitment", transcript_representation(f.com));
    expected.absorb_string("kzg.commitment", transcript_representation(t.com));
    expected.absorb_string("lookup.multiplicity", proof_encoding(record.multiplicity_commitment.com));
    expected.absorb_string("kzg.commitment", transcript_representation(record.multiplicity_commitment));
    expected.absorb_string("kzg.degree.commitment", transcript_representation(record.multiplicity_commitment));
    expected.absorb_size("kzg.degree.bound", 7);
    const Fr expected_challenge = expected.squeeze_fr("lookup.challenge");
    EXPECT_EQ(record.challenge, expected_challenge);
}
