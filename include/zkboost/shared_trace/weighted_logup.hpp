#pragma once

#include "commitments/proof_session.hpp"
#include "zkboost/shared_trace/validate_histograms.hpp"
#include "zkboost/shared_trace/validate_histograms_plonk.hpp"

#include <string>
#include <vector>

class ProofProgram;
class virCommitment;

class zkboostSharedWeightedLogupSummary {
public:
    size_t query_row_count = 0;
    size_t feature_count = 0;
    size_t query_value_count = 0;
    size_t histogram_size = 0;
    size_t query_domain_size = 0;
    size_t histogram_domain_size = 0;
    size_t channel_count = 0;
};

class zkboostSharedWeightedLogupInstance {
public:
    zkboostSharedWeightedLogupSummary summary;
    kzgPublicParameters public_parameters;

    std::vector<Fr> query_index_evals;
    std::vector<Fr> histogram_index_evals;
    std::vector<std::vector<Fr>> query_value_evals;
    std::vector<std::vector<Fr>> histogram_evals;

    poly query_indices;
    poly histogram_indices;
    std::vector<poly> query_values;
    std::vector<poly> histograms;

    Group1 query_indices_commitment;
    Group1 histogram_indices_commitment;
    std::vector<Group1> query_value_commitments;
    std::vector<Group1> histogram_commitments;
    std::string linked_histogram_digest;
};

class zkboostSharedWeightedLogupVerifierBundle {
public:
    zkboostSharedWeightedLogupSummary summary;
    const kzgPublicParameters* public_parameters = nullptr;
    Group1 query_indices_commitment;
    Group1 histogram_indices_commitment;
    std::vector<Group1> query_value_commitments;
    std::vector<Group1> histogram_commitments;
};

bool add_zkboost_shared_weighted_logup_to_program(
    ProofProgram& program,
    const zkboostSharedWeightedLogupInstance& instance);
bool add_zkboost_shared_weighted_logup_verifier_to_program(
    ProofProgram& program,
    const zkboostSharedWeightedLogupVerifierBundle& verifier_bundle);

// Proves a single histogram relation
//   histogram[b] = sum_{i : indices[i] = b} values[i].
// As in CertXGB, this helper assumes `indices` were already authenticated and
// constrained to [0, histogram_domain_size) by the upstream computation.
bool add_zkboost_weighted_logup_histogram_to_program(
    ProofProgram& program,
    const virCommitment& indices,
    const virCommitment& values,
    const virCommitment& histogram,
    size_t query_domain_size,
    size_t histogram_domain_size);

zkboostSharedWeightedLogupInstance build_zkboost_shared_weighted_logup_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& view,
    const kzgPublicParameters& public_parameters);

size_t required_zkboost_shared_weighted_logup_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& view);

SerializedProof prove_zkboost_shared_weighted_logup(
    const zkboostSharedWeightedLogupInstance& instance);

bool verify_zkboost_shared_weighted_logup(
    const zkboostSharedWeightedLogupInstance& instance,
    const SerializedProof& proof);

zkboostSharedWeightedLogupVerifierBundle prepare_zkboost_shared_weighted_logup_verifier_bundle(
    const zkboostSharedWeightedLogupInstance& instance);

bool verify_zkboost_shared_weighted_logup_preprocessed(
    const zkboostSharedWeightedLogupVerifierBundle& verifier_bundle,
    const SerializedProof& proof);

bool link_zkboost_shared_histograms_to_weighted_logup(
    const zkboostSharedHistogramPlonkInstance& histogram_instance,
    const zkboostSharedWeightedLogupInstance& weighted_logup_instance);
