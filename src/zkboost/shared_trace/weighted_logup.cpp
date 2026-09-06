#include "zkboost/shared_trace/weighted_logup.hpp"

#include "algebra_utils.hpp"
#include "profiling/proof_cost.hpp"
#include "proof_runtime.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_weighted_logup: " + message);
    }
}

std::vector<Fr> padded_evals(const std::vector<Fr>& values, size_t domain_size)
{
    require_condition(values.size() <= domain_size, "column exceeds padded domain");
    std::vector<Fr> out(domain_size, Fr(0));
    std::copy(values.begin(), values.end(), out.begin());
    return out;
}

std::vector<Fr> index_evals(size_t logical_size, size_t domain_size)
{
    require_condition(logical_size <= domain_size, "index column exceeds padded domain");
    std::vector<Fr> out(domain_size, Fr(0));
    for (size_t i = 0; i < domain_size; ++i) {
        out[i] = field_from_size_t(i);
    }
    return out;
}

std::string digest_public_inputs(const std::vector<Fr>& public_inputs)
{
    ProofTranscript transcript("CertXGB weighted LogUp histogram link");
    transcript.absorb_size("public_inputs.size", public_inputs.size());
    for (const Fr& value : public_inputs) {
        transcript.absorb_string("public_input", value.getStr(10));
    }
    const std::array<uint8_t, 32> digest =
        transcript.squeeze_hash("weighted_logup.histogram.digest");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : digest) {
        out << std::setw(2) << static_cast<unsigned>(byte);
    }
    return out.str();
}

std::vector<Fr> weighted_inverse_evals(const std::vector<Fr>& numerators,
                                       const std::vector<Fr>& indices,
                                       const Fr& challenge)
{
    require_condition(numerators.size() == indices.size(), "weighted inverse shape mismatch");
    std::vector<Fr> out(numerators.size(), Fr(0));
    for (size_t i = 0; i < numerators.size(); ++i) {
        const Fr denominator = challenge + indices[i];
        require_condition(!denominator.isZero(), "Fiat-Shamir challenge hits a bucket pole");
        Fr inverse;
        Fr::inv(inverse, denominator);
        out[i] = numerators[i] * inverse;
    }
    return out;
}

Fr sum_evals(const std::vector<Fr>& values)
{
    Fr out(0);
    for (const Fr& value : values) {
        out += value;
    }
    return out;
}

virCommitment bind_public_column(ProofProgram& program,
                                 const char* key,
                                 const poly* prover_polynomial,
                                 const Group1& verifier_commitment,
                                 size_t degree_bound)
{
    if (program.runtime().mode() == ProofMode::Prove) {
        require_condition(prover_polynomial != nullptr, std::string(key) + " prover polynomial missing");
        return program.bind_virtual(
            PublicCommitmentInput::prover(key, *prover_polynomial, degree_bound));
    }
    return program.bind_virtual(
        PublicCommitmentInput::verifier(key, verifier_commitment, degree_bound));
}

struct weightedLogupColumns {
    const poly* query_indices = nullptr;
    const poly* histogram_indices = nullptr;
    std::vector<const poly*> query_values;
    std::vector<const poly*> histograms;
    const std::vector<Fr>* query_index_evals = nullptr;
    const std::vector<Fr>* histogram_index_evals = nullptr;
    std::vector<const std::vector<Fr>*> query_value_evals;
    std::vector<const std::vector<Fr>*> histogram_evals;
    Group1 query_indices_commitment;
    Group1 histogram_indices_commitment;
    std::vector<Group1> query_value_commitments;
    std::vector<Group1> histogram_commitments;
};

bool add_weighted_logup(ProofProgram& program,
                        const zkboostSharedWeightedLogupSummary& summary,
                        const weightedLogupColumns& columns)
{
    require_condition(summary.channel_count > 0, "channel_count must be non-zero");
    require_condition(columns.query_value_commitments.size() == summary.channel_count,
                      "query commitment channel mismatch");
    require_condition(columns.histogram_commitments.size() == summary.channel_count,
                      "histogram commitment channel mismatch");

    if (program.runtime().mode() == ProofMode::Prove) {
        const size_t logical_query_rows =
            summary.query_value_count * summary.channel_count;
        const size_t logical_table_rows =
            summary.histogram_size * summary.channel_count;
        const size_t padded_slots =
            (summary.query_domain_size + summary.histogram_domain_size) *
            summary.channel_count;
    }

    {
        const virCommitment query_indices = bind_public_column(
            program,
            "zkboost.weighted_logup.query_indices",
            columns.query_indices,
            columns.query_indices_commitment,
            summary.query_domain_size - 1);
        const virCommitment histogram_indices = bind_public_column(
            program,
            "zkboost.weighted_logup.histogram_indices",
            columns.histogram_indices,
            columns.histogram_indices_commitment,
            summary.histogram_domain_size - 1);

        std::vector<virCommitment> query_values;
        std::vector<virCommitment> histograms;
        query_values.reserve(summary.channel_count);
        histograms.reserve(summary.channel_count);
        for (size_t channel = 0; channel < summary.channel_count; ++channel) {
            const std::string query_key =
                "zkboost.weighted_logup.query_values." + std::to_string(channel);
            const std::string histogram_key =
                "zkboost.weighted_logup.histogram." + std::to_string(channel);
            const poly* query_poly =
                channel < columns.query_values.size() ? columns.query_values[channel] : nullptr;
            const poly* histogram_poly =
                channel < columns.histograms.size() ? columns.histograms[channel] : nullptr;
            query_values.push_back(bind_public_column(
                program,
                query_key.c_str(),
                query_poly,
                columns.query_value_commitments[channel],
                summary.query_domain_size - 1));
            histograms.push_back(bind_public_column(
                program,
                histogram_key.c_str(),
                histogram_poly,
                columns.histogram_commitments[channel],
                summary.histogram_domain_size - 1));
        }

        // All statement commitments are absorbed before beta is sampled.
        const Fr beta = program.runtime().kzg().transcript().squeeze_nonzero_fr(
            "zkboost.weighted_logup.beta");
        const virCommitment beta_expression(beta);

        for (size_t channel = 0; channel < summary.channel_count; ++channel) {
            std::optional<poly> query_inverse;
            std::optional<poly> histogram_inverse;
            Fr expected_claim(0);
            if (program.runtime().mode() == ProofMode::Prove) {
                require_condition(columns.query_index_evals != nullptr &&
                                      columns.histogram_index_evals != nullptr &&
                                      channel < columns.query_value_evals.size() &&
                                      channel < columns.histogram_evals.size(),
                                  "prover weighted LogUp evaluations missing");
                const std::vector<Fr> query_inverse_values = weighted_inverse_evals(
                    *columns.query_value_evals[channel],
                    *columns.query_index_evals,
                    beta);
                const std::vector<Fr> histogram_inverse_values = weighted_inverse_evals(
                    *columns.histogram_evals[channel],
                    *columns.histogram_index_evals,
                    beta);
                expected_claim = sum_evals(query_inverse_values);
                require_condition(expected_claim == sum_evals(histogram_inverse_values),
                                  "weighted LogUp identity does not hold");
                query_inverse = poly_from_evals(query_inverse_values);
                histogram_inverse = poly_from_evals(histogram_inverse_values);
            }

            const std::string query_inverse_label =
                "zkboost.weighted_logup.query_inverse." + std::to_string(channel);
            const std::string histogram_inverse_label =
                "zkboost.weighted_logup.histogram_inverse." + std::to_string(channel);
            virCommitment query_inverse_commitment = program.bind_virtual(
                WitnessCommitmentInput::from_optional(
                    query_inverse_label,
                    std::move(query_inverse),
                    summary.query_domain_size - 1));
            virCommitment histogram_inverse_commitment = program.bind_virtual(
                WitnessCommitmentInput::from_optional(
                    histogram_inverse_label,
                    std::move(histogram_inverse),
                    summary.histogram_domain_size - 1));

            const std::string claim_label =
                "zkboost.weighted_logup.sum_claim." + std::to_string(channel);
            const Fr claim = program.runtime().session()->exchange_fr(
                claim_label,
                [&] { return expected_claim; });

            // sum_i v_i/(beta+a_i) = sum_j H_j/(beta+j)
            program.sum().add_check(
                query_inverse_commitment,
                claim,
                summary.query_domain_size);
            program.sum().add_check(
                histogram_inverse_commitment,
                claim,
                summary.histogram_domain_size);

            // Pointwise inverse identities bind both rational sums to their
            // committed indices, values, and leaf histogram.
            virCommitment query_inverse_zero =
                (query_indices + beta_expression) * query_inverse_commitment -
                query_values[channel];
            virCommitment histogram_inverse_zero =
                (histogram_indices + beta_expression) * histogram_inverse_commitment -
                histograms[channel];
            program.zero().add_check(
                query_inverse_zero,
                summary.query_domain_size,
                "zkboost.weighted_logup.query_inverse");
            program.zero().add_check(
                histogram_inverse_zero,
                summary.histogram_domain_size,
                "zkboost.weighted_logup.histogram_inverse");
        }
    }
    return true;
}

bool run_weighted_logup(ProofRuntime& runtime,
                        const zkboostSharedWeightedLogupSummary& summary,
                        const weightedLogupColumns& columns)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        add_weighted_logup(program, summary, columns);
    });
}

weightedLogupColumns prover_columns(const zkboostSharedWeightedLogupInstance& instance)
{
    weightedLogupColumns out;
    out.query_indices = &instance.query_indices;
    out.histogram_indices = &instance.histogram_indices;
    out.query_index_evals = &instance.query_index_evals;
    out.histogram_index_evals = &instance.histogram_index_evals;
    for (size_t channel = 0; channel < instance.summary.channel_count; ++channel) {
        out.query_values.push_back(&instance.query_values[channel]);
        out.histograms.push_back(&instance.histograms[channel]);
        out.query_value_evals.push_back(&instance.query_value_evals[channel]);
        out.histogram_evals.push_back(&instance.histogram_evals[channel]);
    }
    out.query_indices_commitment = instance.query_indices_commitment;
    out.histogram_indices_commitment = instance.histogram_indices_commitment;
    out.query_value_commitments = instance.query_value_commitments;
    out.histogram_commitments = instance.histogram_commitments;
    return out;
}

weightedLogupColumns verifier_columns(
    const zkboostSharedWeightedLogupVerifierBundle& verifier_bundle)
{
    weightedLogupColumns out;
    out.query_indices_commitment = verifier_bundle.query_indices_commitment;
    out.histogram_indices_commitment = verifier_bundle.histogram_indices_commitment;
    out.query_value_commitments = verifier_bundle.query_value_commitments;
    out.histogram_commitments = verifier_bundle.histogram_commitments;
    return out;
}

} // namespace

bool add_zkboost_shared_weighted_logup_to_program(
    ProofProgram& program,
    const zkboostSharedWeightedLogupInstance& instance)
{
    return add_weighted_logup(program, instance.summary, prover_columns(instance));
}

bool add_zkboost_shared_weighted_logup_verifier_to_program(
    ProofProgram& program,
    const zkboostSharedWeightedLogupVerifierBundle& verifier_bundle)
{
    return add_weighted_logup(
        program, verifier_bundle.summary, verifier_columns(verifier_bundle));
}

bool add_zkboost_weighted_logup_histogram_to_program(
    ProofProgram& program,
    const virCommitment& indices,
    const virCommitment& values,
    const virCommitment& histogram,
    size_t query_domain_size,
    size_t histogram_domain_size)
{
    require_condition(query_domain_size > 0 && histogram_domain_size > 0,
                      "histogram domains must be non-zero");
    require_condition(isPowerOfTwo(query_domain_size) && isPowerOfTwo(histogram_domain_size),
                      "histogram domains must be powers of two");

    // Match the CertXGB composition from ZKBoost: the address column is
    // authenticated and range-constrained upstream, so Weighted LogUp only
    // proves the histogram aggregation identity here.

    // Unlike the full three-channel CertXGB instance, this standalone form
    // derives the canonical bucket-index column locally. Its commitment is
    // public and therefore does not enlarge the proof.
    const std::vector<Fr> histogram_index_evaluations =
        index_evals(histogram_domain_size, histogram_domain_size);
    const virCommitment histogram_indices = program.bind_virtual(
        PublicPolynomialInput(
            "zkboost.weighted_logup_histogram.bucket_indices." +
                std::to_string(histogram_domain_size),
            poly_from_evals(histogram_index_evaluations),
            histogram_domain_size - 1));

    const Fr beta = program.runtime().kzg().transcript().squeeze_nonzero_fr(
        "zkboost.weighted_logup_histogram.beta");
    const virCommitment beta_expression(beta);

    std::optional<poly> query_inverse;
    std::optional<poly> histogram_inverse;
    Fr expected_claim(0);
    if (program.runtime().mode() == ProofMode::Prove) {
        const std::vector<Fr> query_indices =
            program.runtime().kzg().evaluations(indices, query_domain_size);
        const std::vector<Fr> query_values =
            program.runtime().kzg().evaluations(values, query_domain_size);
        const std::vector<Fr> histogram_values =
            program.runtime().kzg().evaluations(histogram, histogram_domain_size);
        const std::vector<Fr> query_inverse_values =
            weighted_inverse_evals(query_values, query_indices, beta);
        const std::vector<Fr> histogram_inverse_values =
            weighted_inverse_evals(histogram_values, histogram_index_evaluations, beta);
        expected_claim = sum_evals(query_inverse_values);
        require_condition(expected_claim == sum_evals(histogram_inverse_values),
                          "weighted LogUp histogram identity does not hold");
        query_inverse = poly_from_evals(query_inverse_values);
        histogram_inverse = poly_from_evals(histogram_inverse_values);
    }

    virCommitment query_inverse_commitment = program.bind_virtual(
        WitnessCommitmentInput::from_optional(
            "zkboost.weighted_logup_histogram.query_inverse",
            std::move(query_inverse),
            query_domain_size - 1));
    virCommitment histogram_inverse_commitment = program.bind_virtual(
        WitnessCommitmentInput::from_optional(
            "zkboost.weighted_logup_histogram.histogram_inverse",
            std::move(histogram_inverse),
            histogram_domain_size - 1));
    const Fr claim = program.runtime().session()->exchange_fr(
        "zkboost.weighted_logup_histogram.sum_claim", [&] { return expected_claim; });

    program.sum().add_check(query_inverse_commitment, claim, query_domain_size);
    program.sum().add_check(histogram_inverse_commitment, claim, histogram_domain_size);

    virCommitment query_inverse_zero =
        (indices + beta_expression) * query_inverse_commitment - values;
    virCommitment histogram_inverse_zero =
        (histogram_indices + beta_expression) * histogram_inverse_commitment - histogram;
    program.zero().add_check(query_inverse_zero,
                             query_domain_size,
                             "zkboost.weighted_logup_histogram.query_inverse");
    program.zero().add_check(histogram_inverse_zero,
                             histogram_domain_size,
                             "zkboost.weighted_logup_histogram.histogram_inverse");
    return true;
}

zkboostSharedWeightedLogupInstance build_zkboost_shared_weighted_logup_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& view,
    const kzgPublicParameters& public_parameters)
{
    require_condition(view.summary.query_row_count == training.tree_count * training.query_count,
                      "query_row_count mismatch");
    require_condition(view.summary.feature_count == training.feature_count,
                      "feature_count mismatch");
    require_condition(view.summary.histogram_size ==
                          training.node_count * training.feature_count * training.bin_count,
                      "histogram_size mismatch");
    require_condition(view.summary.channel_count == 3, "expected three histogram channels");

    zkboostSharedWeightedLogupInstance out;
    out.summary.query_row_count = view.summary.query_row_count;
    out.summary.feature_count = training.feature_count;
    out.summary.query_value_count = view.summary.query_row_count * training.feature_count;
    out.summary.histogram_size = view.summary.histogram_size;
    out.summary.query_domain_size = nextPow2(out.summary.query_value_count);
    out.summary.histogram_domain_size = nextPow2(out.summary.histogram_size);
    out.summary.channel_count = view.summary.channel_count;
    out.public_parameters = public_parameters;

    require_condition(view.flattened_bucket_indices.size() == out.summary.query_value_count,
                      "flattened bucket index length mismatch");
    out.query_index_evals = padded_evals(
        view.flattened_bucket_indices,
        out.summary.query_domain_size);
    out.histogram_index_evals = index_evals(
        out.summary.histogram_size,
        out.summary.histogram_domain_size);

    const std::vector<const std::vector<Fr>*> query_channels = {
        &view.query_counts,
        &view.query_pseudo_residuals,
        &view.query_diagonal_hessians,
    };
    out.query_value_evals.resize(out.summary.channel_count);
    out.histogram_evals.resize(out.summary.channel_count);
    for (size_t channel = 0; channel < out.summary.channel_count; ++channel) {
        require_condition(query_channels[channel]->size() == view.summary.query_row_count,
                          "query channel length mismatch");
        std::vector<Fr> lifted;
        lifted.reserve(out.summary.query_value_count);
        for (size_t query = 0; query < view.summary.query_row_count; ++query) {
            for (size_t feature = 0; feature < training.feature_count; ++feature) {
                (void)feature;
                lifted.push_back(query_channels[channel]->at(query));
            }
        }
        out.query_value_evals[channel] = padded_evals(
            lifted,
            out.summary.query_domain_size);
        require_condition(view.leaf_histograms[channel].size() == out.summary.histogram_size,
                          "leaf histogram channel length mismatch");
        out.histogram_evals[channel] = padded_evals(
            view.leaf_histograms[channel],
            out.summary.histogram_domain_size);
    }

    out.query_indices = poly_from_evals(out.query_index_evals);
    out.histogram_indices = poly_from_evals(out.histogram_index_evals);
    out.query_indices_commitment =
        commit_g1_with_public_parameters(public_parameters, out.query_indices);
    out.histogram_indices_commitment =
        commit_g1_with_public_parameters(public_parameters, out.histogram_indices);
    for (size_t channel = 0; channel < out.summary.channel_count; ++channel) {
        out.query_values.push_back(poly_from_evals(out.query_value_evals[channel]));
        out.histograms.push_back(poly_from_evals(out.histogram_evals[channel]));
        out.query_value_commitments.push_back(
            commit_g1_with_public_parameters(public_parameters, out.query_values.back()));
        out.histogram_commitments.push_back(
            commit_g1_with_public_parameters(public_parameters, out.histograms.back()));
    }
    out.linked_histogram_digest = digest_public_inputs(
        encode_zkboost_shared_histogram_public_inputs(view, training));
    return out;
}

size_t required_zkboost_shared_weighted_logup_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& view)
{
    require_condition(view.summary.query_row_count == training.tree_count * training.query_count,
                      "query_row_count mismatch");
    return std::max(
        nextPow2(view.summary.query_row_count * training.feature_count),
        nextPow2(view.summary.histogram_size));
}

SerializedProof prove_zkboost_shared_weighted_logup(
    const zkboostSharedWeightedLogupInstance& instance)
{
    ProofRuntime runtime = ProofRuntime::prover(instance.public_parameters);
    if (!run_weighted_logup(runtime, instance.summary, prover_columns(instance))) {
        throw std::runtime_error("prove_zkboost_shared_weighted_logup: proof program failed");
    }
    return runtime.take_proof();
}

bool verify_zkboost_shared_weighted_logup(
    const zkboostSharedWeightedLogupInstance& instance,
    const SerializedProof& proof)
{
    return verify_zkboost_shared_weighted_logup_preprocessed(
        prepare_zkboost_shared_weighted_logup_verifier_bundle(instance),
        proof);
}

zkboostSharedWeightedLogupVerifierBundle prepare_zkboost_shared_weighted_logup_verifier_bundle(
    const zkboostSharedWeightedLogupInstance& instance)
{
    zkboostSharedWeightedLogupVerifierBundle out;
    out.summary = instance.summary;
    out.public_parameters = &instance.public_parameters;
    out.query_indices_commitment = instance.query_indices_commitment;
    out.histogram_indices_commitment = instance.histogram_indices_commitment;
    out.query_value_commitments = instance.query_value_commitments;
    out.histogram_commitments = instance.histogram_commitments;
    return out;
}

bool verify_zkboost_shared_weighted_logup_preprocessed(
    const zkboostSharedWeightedLogupVerifierBundle& verifier_bundle,
    const SerializedProof& proof)
{
    if (verifier_bundle.public_parameters == nullptr) {
        return false;
    }
    try {
        ProofRuntime runtime = ProofRuntime::verifier_borrowed(
            *verifier_bundle.public_parameters,
            proof);
        const bool ok = run_weighted_logup(
            runtime,
            verifier_bundle.summary,
            verifier_columns(verifier_bundle));
        return ok && runtime.proof_fully_consumed();
    } catch (const std::exception&) {
        return false;
    }
}

bool link_zkboost_shared_histograms_to_weighted_logup(
    const zkboostSharedHistogramPlonkInstance& histogram_instance,
    const zkboostSharedWeightedLogupInstance& weighted_logup_instance)
{
    try {
        if (histogram_instance.summary.query_row_count !=
                weighted_logup_instance.summary.query_row_count ||
            histogram_instance.summary.feature_count !=
                weighted_logup_instance.summary.feature_count ||
            histogram_instance.summary.histogram_size !=
                weighted_logup_instance.summary.histogram_size ||
            histogram_instance.summary.channel_count !=
                weighted_logup_instance.summary.channel_count ||
            weighted_logup_instance.summary.channel_count != 3) {
            return false;
        }

        const size_t q = histogram_instance.summary.query_row_count;
        const size_t f = histogram_instance.summary.feature_count;
        const size_t h = histogram_instance.summary.histogram_size;
        const size_t sample_count = histogram_instance.summary.sample_count;
        const size_t old_public_size = 3 * q + 12 * h;
        const size_t input_count = sample_count * f;
        const size_t flattened_count = q * f;
        const size_t flattened_offset = old_public_size + input_count;
        if (histogram_instance.public_inputs.size() !=
                flattened_offset + flattened_count ||
            weighted_logup_instance.summary.query_value_count != flattened_count) {
            return false;
        }

        std::vector<Fr> flattened(
            histogram_instance.public_inputs.begin() + flattened_offset,
            histogram_instance.public_inputs.end());
        const std::vector<Fr> expected_query_indices = padded_evals(
            flattened,
            weighted_logup_instance.summary.query_domain_size);
        const std::vector<Fr> expected_histogram_indices = index_evals(
            h,
            weighted_logup_instance.summary.histogram_domain_size);
        const poly expected_query_indices_poly =
            poly_from_evals(expected_query_indices);
        const poly expected_histogram_indices_poly =
            poly_from_evals(expected_histogram_indices);
        const Group1 expected_query_indices_commitment =
            commit_g1_with_public_parameters(
                weighted_logup_instance.public_parameters,
                expected_query_indices_poly);
        const Group1 expected_histogram_indices_commitment =
            commit_g1_with_public_parameters(
                weighted_logup_instance.public_parameters,
                expected_histogram_indices_poly);
        if (expected_query_indices_commitment != weighted_logup_instance.query_indices_commitment ||
            expected_histogram_indices_commitment != weighted_logup_instance.histogram_indices_commitment) {
            return false;
        }

        if (weighted_logup_instance.query_value_commitments.size() != 3 ||
            weighted_logup_instance.histogram_commitments.size() != 3) {
            return false;
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            std::vector<Fr> lifted;
            lifted.reserve(flattened_count);
            for (size_t query = 0; query < q; ++query) {
                const Fr value = channel == 0
                                     ? Fr(1)
                                     : histogram_instance.public_inputs.at(channel * q + query);
                for (size_t feature = 0; feature < f; ++feature) {
                    (void)feature;
                    lifted.push_back(value);
                }
            }
            const std::vector<Fr> expected_query_values = padded_evals(
                lifted,
                weighted_logup_instance.summary.query_domain_size);
            std::vector<Fr> histogram_values(
                histogram_instance.public_inputs.begin() + 3 * q + channel * h,
                histogram_instance.public_inputs.begin() + 3 * q + (channel + 1) * h);
            const std::vector<Fr> expected_histogram = padded_evals(
                histogram_values,
                weighted_logup_instance.summary.histogram_domain_size);
            const poly expected_query_values_poly =
                poly_from_evals(expected_query_values);
            const poly expected_histogram_poly =
                poly_from_evals(expected_histogram);
            const Group1 expected_query_values_commitment =
                commit_g1_with_public_parameters(
                    weighted_logup_instance.public_parameters,
                    expected_query_values_poly);
            const Group1 expected_histogram_commitment =
                commit_g1_with_public_parameters(
                    weighted_logup_instance.public_parameters,
                    expected_histogram_poly);
            if (expected_query_values_commitment != weighted_logup_instance.query_value_commitments.at(channel) ||
                expected_histogram_commitment != weighted_logup_instance.histogram_commitments.at(channel)) {
                return false;
            }
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
