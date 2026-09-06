#include "cube_argmax.hpp"

#include <stdexcept>

namespace {

void validate_cube_middle_argmax_parameters(size_t outer_count,
                                            size_t middle_count,
                                            size_t inner_count,
                                            size_t bit_bound)
{
    if (!isPowerOfTwo(outer_count) || !isPowerOfTwo(middle_count) || !isPowerOfTwo(inner_count) ||
        outer_count == 0 || middle_count == 0 || inner_count == 0) {
        throw std::runtime_error(
            "cubeMiddleArgmaxProtocol: outer_count, middle_count, and inner_count must be non-zero powers of two");
    }
    if (bit_bound > static_cast<size_t>(Fr::getBitSize() - 1)) {
        throw std::runtime_error(
            "cubeMiddleArgmaxProtocol: bit bound exceeds the safely representable integer range of Fr");
    }
}

size_t cube_domain_size(size_t outer_count, size_t middle_count, size_t inner_count)
{
    return outer_count * middle_count * inner_count;
}

} // namespace

void cubeMiddleArgmaxVerifier::add_check(const virCommitment& scores,
                                         const virCommitment& max_values,
                                         const virCommitment& selector,
                                         const cubeMiddleArgmaxCheckRecord& record)
{
    this->scores.push_back(scores);
    this->max_values.push_back(max_values);
    selectors.push_back(selector);
    checks.push_back(record);
}

bool cubeMiddleArgmaxVerifier::finalize() const
{
    return true;
}

cubeMiddleArgmaxCheckRecord cubeMiddleArgmaxProver::add_check(const virCommitment& max_values,
                                                              size_t outer_count,
                                                              size_t middle_count,
                                                              size_t inner_count,
                                                              size_t bit_bound)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("cubeMiddleArgmaxProver not setup");
    }

    cubeMiddleArgmaxCheckRecord record;
    record.outer_count = outer_count;
    record.middle_count = middle_count;
    record.inner_count = inner_count;
    record.bit_bound = bit_bound;

    const poly lifted_max_poly =
        cube_middle_replication_polynomial(*params->kzg->materialize(max_values), outer_count, middle_count, inner_count);
    record.lifted_max_commitment = params->kzg->commit(lifted_max_poly, cube_domain_size(outer_count, middle_count, inner_count) - 1);
    checks.push_back(record);
    return record;
}

void cubeMiddleArgmaxProtocol::add_check(const virCommitment& scores,
                                         const virCommitment& max_values,
                                         const virCommitment& selector,
                                         size_t outer_count,
                                         size_t middle_count,
                                         size_t inner_count,
                                         size_t bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("cubeMiddleArgmaxProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.cube_sum == nullptr ||
        params_.replication == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("cubeMiddleArgmaxProtocol dependencies not setup");
    }

    validate_cube_middle_argmax_parameters(outer_count, middle_count, inner_count, bit_bound);
    proof_cost::record_relation("argmax", cube_domain_size(outer_count, middle_count, inner_count));

    const size_t full_domain_size = cube_domain_size(outer_count, middle_count, inner_count);
    const virCommitment lifted_max = [&]() {
        cubeMiddleArgmaxCheckRecord record;
        const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
        if (proof_session && proof_session->mode() == ProofMode::Verify) {
            record.outer_count = outer_count;
            record.middle_count = middle_count;
            record.inner_count = inner_count;
            record.bit_bound = bit_bound;
            const Group1 lifted_max_group = proof_session->transport_group1({});
            record.lifted_max_commitment = params_.kzg->adopt_commitment(lifted_max_group);
            params_.kzg->prove_degree(record.lifted_max_commitment, full_domain_size - 1);
        } else {
            record = prover_.add_check(max_values, outer_count, middle_count, inner_count, bit_bound);
            if (proof_session && proof_session->mode() == ProofMode::Prove) {
                const Group1 lifted_max_group =
                    proof_session->transport_group1([&] { return record.lifted_max_commitment.com; });
                if (lifted_max_group != record.lifted_max_commitment.com) {
                    throw std::runtime_error("cubeMiddleArgmaxProtocol: transported lifted max commitment mismatch");
                }
            }
        }

        verifier_.add_check(scores, max_values, selector, record);
        return virCommitment(record.lifted_max_commitment);
    }();

    // Boolean selector over the full cube.
    virCommitment selector_boolean_zero = selector * (selector - virCommitment(Fr(1)));
    params_.zero_check->add_check(selector_boolean_zero, full_domain_size);

    // One selected entry per (outer, inner) column when summing along the middle axis.
    params_.cube_sum->add_check(selector, virCommitment(Fr(1)), outer_count, middle_count, inner_count);

    // The selected entries add up to the claimed maxima.
    params_.cube_sum->add_check(selector * scores, max_values, outer_count, middle_count, inner_count);

    params_.replication->add_check(max_values, lifted_max, outer_count, middle_count, inner_count);

    // Entry-wise domination of every candidate by the selected maximum.
    params_.less_check->add_check(lifted_max - scores, bit_bound, full_domain_size);
}

void cubeMiddleArgmaxProtocol::add_check(const virCommitment& scores,
                                         const virCommitment& max_values,
                                         const virCommitment& lifted_max_values,
                                         const virCommitment& selector,
                                         size_t outer_count,
                                         size_t middle_count,
                                         size_t inner_count,
                                         size_t bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("cubeMiddleArgmaxProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.cube_sum == nullptr ||
        params_.replication == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("cubeMiddleArgmaxProtocol dependencies not setup");
    }

    validate_cube_middle_argmax_parameters(outer_count, middle_count, inner_count, bit_bound);
    proof_cost::record_relation("argmax", cube_domain_size(outer_count, middle_count, inner_count));

    const size_t full_domain_size = cube_domain_size(outer_count, middle_count, inner_count);
    cubeMiddleArgmaxCheckRecord record;
    record.outer_count = outer_count;
    record.middle_count = middle_count;
    record.inner_count = inner_count;
    record.bit_bound = bit_bound;
    if (lifted_max_values.type != virCommitment::Type::KZG) {
        throw std::runtime_error("cubeMiddleArgmaxProtocol: external lifted max must be a direct KZG commitment");
    }
    record.lifted_max_commitment = lifted_max_values.com;
    verifier_.add_check(scores, max_values, selector, record);

    // Boolean selector over the full cube.
    virCommitment selector_boolean_zero = selector * (selector - virCommitment(Fr(1)));
    params_.zero_check->add_check(selector_boolean_zero, full_domain_size);

    // One selected entry per (outer, inner) column when summing along the middle axis.
    params_.cube_sum->add_check(selector, virCommitment(Fr(1)), outer_count, middle_count, inner_count);

    // The selected entries add up to the claimed maxima.
    params_.cube_sum->add_check(selector * scores, max_values, outer_count, middle_count, inner_count);

    params_.replication->add_check(max_values, lifted_max_values, outer_count, middle_count, inner_count);

    // Entry-wise domination of every candidate by the selected maximum.
    params_.less_check->add_check(lifted_max_values - scores, bit_bound, full_domain_size);
}

bool cubeMiddleArgmaxProtocol::finalize()
{
    begin_finalize("cubeMiddleArgmaxProtocol");
    return verifier_.finalize();
}
