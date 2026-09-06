#include "nonzero_check.hpp"

#include "polynomial.hpp"

#include <stdexcept>

namespace {

std::vector<Fr> inverse_witness_evaluations(kzgProtocol& kzg,
                                            const virCommitment& values,
                                            const virCommitment& mask,
                                            size_t domain_size,
                                            bool uses_mask,
                                            const std::string& label)
{
    std::vector<Fr> value_evals =
        kzg.evaluations(values, domain_size, kzgProtocol::EvaluationMode::PreferEval);
    std::vector<Fr> mask_evals;
    if (uses_mask) {
        mask_evals = kzg.evaluations(mask, domain_size, kzgProtocol::EvaluationMode::PreferEval);
    }

    std::vector<Fr> inverse_evals(domain_size, Fr(0));
    for (size_t i = 0; i < domain_size; ++i) {
        if (uses_mask && mask_evals[i].isZero()) {
            continue;
        }
        if (value_evals[i].isZero()) {
            throw std::runtime_error("nonzeroCheckProtocol: zero value in active row" +
                                     (label.empty() ? std::string() : std::string(" for ") + label));
        }
        Fr::inv(inverse_evals[i], value_evals[i]);
    }
    return inverse_evals;
}

} // namespace

void nonzeroCheckProtocol::add_check(const virCommitment& values,
                                     size_t domain_size,
                                     const std::string& label)
{
    proof_cost::record_relation("nonzero_check", domain_size);
    if (!is_setup) {
        throw std::runtime_error("nonzeroCheckProtocol not setup");
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("nonzeroCheckProtocol: domain size must be a non-zero power of two");
    }

    values_.push_back(values);
    masks_.push_back(virCommitment(Fr(1)));
    checks_.push_back({domain_size, false, label});
}

void nonzeroCheckProtocol::add_masked_check(const virCommitment& values,
                                            const virCommitment& mask,
                                            size_t domain_size,
                                            const std::string& label)
{
    proof_cost::record_relation("nonzero_check", domain_size);
    if (!is_setup) {
        throw std::runtime_error("nonzeroCheckProtocol not setup");
    }
    if (!isPowerOfTwo(domain_size) || domain_size == 0) {
        throw std::runtime_error("nonzeroCheckProtocol: domain size must be a non-zero power of two");
    }

    values_.push_back(values);
    masks_.push_back(mask);
    checks_.push_back({domain_size, true, label});
}

bool nonzeroCheckProtocol::finalize()
{
    begin_finalize("nonzeroCheckProtocol");
    if (kzg_ == nullptr || zero_check_ == nullptr) {
        throw std::runtime_error("nonzeroCheckProtocol dependencies not setup");
    }

    const std::shared_ptr<ProofSession> proof_session = kzg_->proof_session();
    for (size_t i = 0; i < checks_.size(); ++i) {
        const nonzeroCheckRecord& record = checks_[i];
        kzgCommitment inverse_commitment;

        if (proof_session && proof_session->mode() == ProofMode::Verify) {
            const Group1 inverse_group = proof_session->transport_group1({});
            inverse_commitment = kzg_->adopt_commitment(inverse_group);
            kzg_->prove_degree(inverse_commitment, record.domain_size - 1);
        } else {
            const std::vector<Fr> inverse_evals = inverse_witness_evaluations(
                *kzg_, values_[i], masks_[i], record.domain_size, record.uses_mask, record.label);
            const poly inverse_poly = poly_from_evals(inverse_evals);
            inverse_commitment = kzg_->commit(inverse_poly, record.domain_size - 1);
            if (proof_session && proof_session->mode() == ProofMode::Prove) {
                const Group1 transported = proof_session->transport_group1([&] { return inverse_commitment.com; });
                if (transported != inverse_commitment.com) {
                    throw std::runtime_error("nonzeroCheckProtocol: transported inverse commitment mismatch");
                }
            }
        }

        virCommitment nonzero_zero = values_[i] * virCommitment(inverse_commitment) - virCommitment(Fr(1));
        if (record.uses_mask) {
            nonzero_zero = masks_[i] * nonzero_zero;
        }
        zero_check_->add_check(nonzero_zero, record.domain_size, record.label);
    }

    return true;
}
