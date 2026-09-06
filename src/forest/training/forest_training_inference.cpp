#include "forest_training_inference.hpp"

#include "matrix_utils.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validate_forest_training_inference_parameters(size_t query_row_count,
                                                   size_t node_count,
                                                   size_t feature_count,
                                                   size_t input_bit_bound)
{
    validate_matrix_shape(query_row_count, 1, "forestTrainingInferenceProtocol query vectors");
    validate_matrix_shape(query_row_count, feature_count, "forestTrainingInferenceProtocol query boxes");
    validate_matrix_shape(node_count, 1, "forestTrainingInferenceProtocol node vectors");
    validate_matrix_shape(node_count, feature_count, "forestTrainingInferenceProtocol node boxes");
    if (input_bit_bound == 0) {
        throw std::runtime_error("forestTrainingInferenceProtocol: input bit bound must be non-zero");
    }
}

bool try_size_t_from_fr(const Fr& value, size_t& out)
{
    const std::string repr = value.getStr(10);
    if (!repr.empty() && repr.front() == '-') {
        return false;
    }

    size_t parsed = 0;
    try {
        parsed = static_cast<size_t>(std::stoull(repr));
    } catch (const std::exception&) {
        return false;
    }

    if (field_from_size_t(parsed) != value) {
        return false;
    }

    out = parsed;
    return true;
}

const poly& node_index_polynomial(size_t node_count)
{
    validate_matrix_shape(node_count, 1, "forestTrainingInferenceProtocol node index table");

    static std::map<size_t, poly> cache;
    const auto it = cache.find(node_count);
    if (it != cache.end()) {
        return it->second;
    }

    std::vector<Fr> evals;
    evals.reserve(node_count);
    for (size_t row = 0; row < node_count; ++row) {
        evals.push_back(field_from_size_t(row));
    }
    return cache.emplace(node_count, poly_from_evals(evals)).first->second;
}

std::vector<size_t> lookup_positions_from_leaf_indices(const std::vector<Fr>& leaf_index_evals, size_t node_count)
{
    std::vector<size_t> positions(leaf_index_evals.size(), 0);
    for (size_t row = 0; row < leaf_index_evals.size(); ++row) {
        size_t parsed = 0;
        if (!try_size_t_from_fr(leaf_index_evals[row], parsed) || parsed >= node_count) {
            positions[row] = 0;
            continue;
        }
        positions[row] = parsed;
    }
    return positions;
}

} // namespace

void forestTrainingInferenceProtocol::add_check(const forestTrainingInferenceWitness& witness,
                                                size_t query_row_count,
                                                size_t node_count,
                                                size_t feature_count,
                                                size_t input_bit_bound)
{
    if (!is_setup) {
        throw std::runtime_error("forestTrainingInferenceProtocol not setup");
    }
    if (params_.kzg == nullptr || params_.less_check == nullptr || params_.box_check == nullptr ||
        params_.concatenated_table_lookup == nullptr) {
        throw std::runtime_error("forestTrainingInferenceProtocol dependencies not setup");
    }

    validate_forest_training_inference_parameters(query_row_count, node_count, feature_count, input_bit_bound);

    witnesses_.push_back(witness);
    checks_.push_back({query_row_count, node_count, feature_count, input_bit_bound});
}

bool forestTrainingInferenceProtocol::finalize()
{
    begin_finalize("forestTrainingInferenceProtocol");

    for (size_t i = 0; i < checks_.size(); ++i) {
        const forestTrainingInferenceCheckRecord& record = checks_[i];
        const forestTrainingInferenceWitness& witness = witnesses_[i];
        const virCommitment node_indices(node_index_polynomial(record.node_count));
        const std::vector<concatenatedTableLookupBlock> blocks = {
            concatenatedTableLookupBlock(witness.leaf_indices, node_indices, 1),
            concatenatedTableLookupBlock(witness.query_tree_ids, witness.node_tree_ids, 1),
            concatenatedTableLookupBlock(witness.leaf_lower_bounds, witness.node_lower_bounds, record.feature_count),
            concatenatedTableLookupBlock(witness.leaf_upper_bounds, witness.node_upper_bounds, record.feature_count),
            concatenatedTableLookupBlock(witness.tree_scores, witness.node_leaf_values, 1),
            concatenatedTableLookupBlock(virCommitment(Fr(1)), witness.node_leaf_mask, 1),
        };

        const std::shared_ptr<ProofSession> proof_session = params_.kzg->proof_session();
        const bool verifier_mode = proof_session && proof_session->mode() == ProofMode::Verify;
        if (verifier_mode) {
            params_.concatenated_table_lookup->add_check(blocks, record.query_row_count, record.node_count);
        } else {
            const std::vector<Fr> leaf_index_evals = params_.kzg->evaluations(witness.leaf_indices, record.query_row_count);
            const std::vector<size_t> row_positions =
                lookup_positions_from_leaf_indices(leaf_index_evals, record.node_count);
            const tableLookupWitness lookup_witness =
                tableLookupWitness::from_positions(record.node_count, row_positions);
            params_.concatenated_table_lookup->add_check(
                blocks, record.query_row_count, record.node_count, lookup_witness);
        }

        params_.box_check->add_check(witness.lifted_inputs,
                                     witness.leaf_lower_bounds,
                                     witness.leaf_upper_bounds,
                                     record.query_row_count,
                                     record.feature_count,
                                     record.input_bit_bound);
    }

    return true;
}
