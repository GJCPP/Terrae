#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"
#include "transcript.hpp"

#include <gtest/gtest.h>

using namespace mcl::bn;

namespace {

poly poly_from_ints(const std::vector<int>& ints)
{
    std::vector<Fr> coeffs;
    coeffs.reserve(ints.size());
    for (int value : ints) {
        coeffs.push_back(fr_from_int(value));
    }
    poly out;
    out.init_with_coeff(coeffs);
    return out;
}

} // namespace

class TranscriptTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        init_groups();
    }
};

TEST_F(TranscriptTest, SameMessageSequenceProducesSameChallenges)
{
    ProofTranscript left;
    ProofTranscript right;

    const poly p = poly_from_ints({1, 2, 3});
    left.absorb_string("msg", "alpha");
    left.absorb_string("poly", transcript_representation(p));
    right.absorb_string("msg", "alpha");
    right.absorb_string("poly", transcript_representation(p));

    EXPECT_EQ(left.squeeze_hash("challenge"), right.squeeze_hash("challenge"));
    EXPECT_EQ(left.squeeze_fr("next"), right.squeeze_fr("next"));
}

TEST_F(TranscriptTest, DifferentMessagesChangeChallenge)
{
    ProofTranscript left;
    ProofTranscript right;

    left.absorb_string("msg", "alpha");
    right.absorb_string("msg", "beta");

    EXPECT_NE(left.squeeze_hash("challenge"), right.squeeze_hash("challenge"));
}

TEST_F(TranscriptTest, SqueezeMutatesTranscriptState)
{
    ProofTranscript transcript;
    transcript.absorb_string("msg", "alpha");

    const Fr first = transcript.squeeze_fr("challenge");
    const Fr second = transcript.squeeze_fr("challenge");

    EXPECT_NE(first, second);
}

TEST_F(TranscriptTest, NonZeroChallengeAvoidsZeroOutput)
{
    ProofTranscript transcript;
    transcript.absorb_string("msg", "alpha");

    EXPECT_FALSE(transcript.squeeze_nonzero_fr("challenge").isZero());
}
