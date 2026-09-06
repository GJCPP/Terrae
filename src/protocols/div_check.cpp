#include "div_check.hpp"

#include "algebra_utils.hpp"
#include "interleaved_batching.hpp"
#include "polynomial.hpp"

#include <map>
#include <stdexcept>
#include <vector>

namespace {

void validate_check_parameters(size_t bit_bound, size_t divisor_power, size_t domain_size)
{
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("divCheckProtocol: domain size must be a non-zero power of two");
    }
    if (bit_bound > static_cast<size_t>(Fr::getBitSize() - 1)) {
        throw std::runtime_error("divCheckProtocol: bit bound exceeds the safely representable integer range of Fr");
    }
    if (divisor_power > bit_bound) {
        throw std::runtime_error("divCheckProtocol: divisor power must not exceed bit bound");
    }
}

Fr power_of_two_scale(size_t exponent)
{
    return pow_fr(fr_from_int(2), exponent);
}

Fr inverse_power_of_two_scale(size_t exponent)
{
    const Fr scale = power_of_two_scale(exponent);
    Fr inverse;
    Fr::inv(inverse, scale);
    return inverse;
}

kzgCommitment quotient_commitment_for_check(kzgProtocol& kzg,
                                            const virCommitment& values,
                                            size_t divisor_power,
                                            size_t domain_size)
{
    const std::vector<Fr> value_evals =
        kzg.evaluations(values, domain_size, kzgProtocol::EvaluationMode::PreferEval);
    std::vector<Fr> quotient_evals;
    quotient_evals.reserve(value_evals.size());

    const Fr inverse_divisor = inverse_power_of_two_scale(divisor_power);
    for (const Fr& value : value_evals) {
        // Exact division by 2^k is linear in the field; later zero-checks make
        // sure this quotient really matches the original witness.
        quotient_evals.push_back(value * inverse_divisor);
    }

    const poly quotient_poly = poly_from_evals(quotient_evals);
    return kzg.commit(quotient_poly, domain_size - 1);
}

} // namespace

void divCheckProtocol::add_check(const virCommitment& values,
                                 size_t bit_bound,
                                 size_t divisor_power,
                                 size_t domain_size)
{
    if (!is_setup) {
        throw std::runtime_error("divCheckProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.zero_check == nullptr || params_.less_check == nullptr) {
        throw std::runtime_error("divCheckProtocol subprotocols not setup");
    }

    validate_check_parameters(bit_bound, divisor_power, domain_size);

    divCheckRecord record;
    record.bit_bound = bit_bound;
    record.divisor_power = divisor_power;
    record.domain_size = domain_size;
    values_.push_back(values);
    checks_.push_back(record);
}

bool divCheckProtocol::finalize()
{
    begin_finalize("divCheckProtocol");

    if (checks_.empty()) {
        return true;
    }

    const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
    std::vector<size_t> divisor_powers_in_order;
    std::map<size_t, std::vector<size_t>> grouped_check_indices;

    for (size_t i = 0; i < checks_.size(); ++i) {
        const size_t divisor_power = checks_[i].divisor_power;
        if (grouped_check_indices.find(divisor_power) == grouped_check_indices.end()) {
            divisor_powers_in_order.push_back(divisor_power);
        }
        grouped_check_indices[divisor_power].push_back(i);
    }

    for (size_t divisor_power : divisor_powers_in_order) {
        const std::vector<size_t>& group_indices = grouped_check_indices.at(divisor_power);
        size_t max_bit_bound = 0;
        for (size_t index : group_indices) {
            max_bit_bound = std::max(max_bit_bound, checks_[index].bit_bound);
        }

        std::vector<InterleavedBatchNode> leaves;
        leaves.reserve(group_indices.size());
        for (size_t index : group_indices) {
            const divCheckRecord& check = checks_[index];
            virCommitment scaled_values = values_[index];
            if (check.bit_bound != max_bit_bound) {
                scaled_values = scaled_values * virCommitment(power_of_two_scale(max_bit_bound - check.bit_bound));
            }

            leaves.push_back(InterleavedBatchNode{{scaled_values}, check.domain_size});
        }

        const InterleavedBatchNode merged = merge_interleaved_nodes(leaves, "divCheckProtocol");
        kzgCommitment quotient_commitment;

        if (proof_session && proof_session->mode() == ProofMode::Verify) {
            const Group1 quotient_group = proof_session->transport_group1({});
            quotient_commitment = params_.kzg->adopt_commitment(quotient_group);
            params_.kzg->prove_degree(quotient_commitment, merged.domain_size - 1);
        } else {
            const kzgCommitment prover_quotient_commitment =
                quotient_commitment_for_check(*params_.kzg, merged.commitments[0], divisor_power, merged.domain_size);
            quotient_commitment = prover_quotient_commitment;
            if (proof_session && proof_session->mode() == ProofMode::Prove) {
                const Group1 quotient_group =
                    proof_session->transport_group1([&] { return prover_quotient_commitment.com; });
                if (quotient_group != prover_quotient_commitment.com) {
                    throw std::runtime_error("divCheckProtocol: transported quotient commitment mismatch");
                }
            }
        }

        const virCommitment quotient_values(quotient_commitment);
        const virCommitment divisor_scale(power_of_two_scale(divisor_power));

        // First prove exact divisibility, then range-check the quotient.
        virCommitment divisibility_zero = merged.commitments[0] - quotient_values * divisor_scale;

        params_.zero_check->add_check(divisibility_zero, merged.domain_size);
        params_.less_check->add_check(quotient_values, max_bit_bound - divisor_power, merged.domain_size);
    }

    return true;
}
