#include "table_lookup.hpp"

#include "algebra_utils.hpp"
#include "matrix_utils.hpp"
#include "protocol_profile.hpp"
#include "terrae_env.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

bool table_lookup_profile_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_PROFILE_TABLE_LOOKUP");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

bool table_lookup_failure_debug_enabled()
{
    static const bool enabled = []() {
        const char* flag = terrae_getenv("TERRAE_DEBUG_TABLE_LOOKUP_FAILURE");
        return flag != nullptr && std::string(flag) != "0" && std::string(flag) != "false";
    }();
    return enabled;
}

template <class Fn>
void table_lookup_profile_scope(const std::string& label, Fn&& fn)
{
    const bool stderr_enabled = table_lookup_profile_enabled();
    const bool structured_enabled = structured_protocol_profile::enabled();
    if (!stderr_enabled && !structured_enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    if (stderr_enabled) {
        std::cerr << "table_lookup_profile " << label << " " << elapsed.count() << "\n";
    }
    structured_protocol_profile::append_row("table_lookup_profile", label, "", elapsed.count(), "ok");
}

struct MatrixShapeData {
    std::shared_ptr<const std::vector<Fr>> row_label_evals;
    std::shared_ptr<const std::vector<Fr>> column_label_evals;
    poly row_label_poly;
    poly column_label_poly;
};

struct PendingRowIndexWitness {
    poly row_index_poly;
    Group1 row_index_group;
};

const MatrixShapeData& get_matrix_shape_data(size_t row_count, size_t column_count);

Fr fr_from_size(size_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

std::shared_ptr<const std::vector<Group1>> precompute_cq_quotient_commitments(kzgProtocol& kzg,
                                                                              const poly& p,
                                                                              size_t domain_size)
{
    const std::vector<Group1> quotient_openings = kzg.batch_opening_commitments(p, domain_size);
    auto quotient_commitments = std::make_shared<std::vector<Group1>>();
    quotient_commitments->reserve(domain_size);

    Fr lagrange_at_zero = fr_from_size(domain_size);
    Fr::inv(lagrange_at_zero, lagrange_at_zero);
    const Fr omega = get_root_of_unity(domain_size);
    Fr root = Fr(1);
    for (size_t i = 0; i < domain_size; ++i) {
        quotient_commitments->push_back((lagrange_at_zero * root) * quotient_openings[i]);
        Fr::mul(root, root, omega);
    }
    return quotient_commitments;
}

std::string matrix_shape_key(size_t row_count, size_t column_count)
{
    return std::to_string(row_count) + "x" + std::to_string(column_count);
}

std::string row_label_helper_key(size_t row_count, size_t column_count)
{
    return "matrix_row_labels:" + matrix_shape_key(row_count, column_count);
}

std::string cq_row_label_helper_key(const std::string& table_key)
{
    return table_key + ".row_label";
}

std::string row_label_cache_key(const kzgProtocol& kzg, size_t row_count, size_t column_count)
{
    const std::string srs_id = kzg.g1_power(1).value().getStr(16);
    return srs_id + ":" + row_label_helper_key(row_count, column_count);
}

void cache_row_label_group(const kzgProtocol& kzg, size_t row_count, size_t column_count, const Group1& commitment)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = row_label_cache_key(kzg, row_count, column_count);
    const auto it = cache.find(key);
    if (it == cache.end()) {
        cache.emplace(key, commitment);
        return;
    }
    if (it->second != commitment) {
        throw std::runtime_error("tableLookupProtocol: inconsistent cached row-label commitment");
    }
}

Group1 preprocessed_row_label_group(const kzgProtocol& kzg, size_t row_count, size_t column_count)
{
    static std::unordered_map<std::string, Group1> cache;
    const std::string key = row_label_cache_key(kzg, row_count, column_count);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const MatrixShapeData& shape = get_matrix_shape_data(row_count, column_count);
    const Group1 commitment = kzg.commit_g1_group(shape.row_label_poly);
    cache.emplace(key, commitment);
    return commitment;
}

poly column_label_poly_for_shape(size_t row_count, size_t column_count)
{
    if (column_count == 1) {
        return poly_from_coeff(std::vector<Fr>{Fr(1)});
    }

    poly label_poly;
    label_poly.add_term(row_count, Fr(1));
    return label_poly;
}

MatrixShapeData build_matrix_shape_data(size_t row_count, size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "matrix lookup");

    const size_t domain_size = row_count * column_count;
    const Fr row_root = subgroup_generator(row_count);
    const Fr column_root = subgroup_generator(column_count);

    auto row_labels = std::make_shared<std::vector<Fr>>(domain_size, Fr(0));
    auto column_labels = std::make_shared<std::vector<Fr>>(domain_size, Fr(0));

    Fr row_label = Fr(1);
    for (size_t row = 0; row < row_count; ++row) {
        Fr column_label = Fr(1);
        for (size_t column = 0; column < column_count; ++column) {
            // Every cell gets a pair `(row_label, column_label)` so later random
            // linear combinations can encode full row identity.
            const size_t index = matrix_index(row, column, column_count);
            (*row_labels)[index] = row_label;
            (*column_labels)[index] = column_label;
            Fr::mul(column_label, column_label, column_root);
        }
        Fr::mul(row_label, row_label, row_root);
    }

    MatrixShapeData data;
    data.row_label_evals = row_labels;
    data.column_label_evals = column_labels;
    data.row_label_poly = poly_from_evals(*row_labels);
    data.column_label_poly = column_label_poly_for_shape(row_count, column_count);
    return data;
}

const MatrixShapeData& get_matrix_shape_data(size_t row_count, size_t column_count)
{
    static std::map<std::string, MatrixShapeData> cache;
    const std::string key = matrix_shape_key(row_count, column_count);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(key, build_matrix_shape_data(row_count, column_count)).first->second;
}

std::vector<Fr> row_labels_from_positions(size_t table_row_count, const std::vector<size_t>& row_positions)
{
    std::vector<Fr> row_labels;
    row_labels.reserve(row_positions.size());
    for (size_t row_position : row_positions) {
        if (row_position >= table_row_count) {
            throw std::runtime_error("tableLookupWitness: row position out of range");
        }
        row_labels.push_back(subgroup_label(table_row_count, row_position));
    }
    return row_labels;
}

std::vector<Fr> expand_row_labels(const std::vector<Fr>& row_labels, size_t column_count)
{
    std::vector<Fr> expanded(row_labels.size() * column_count, Fr(0));
    for (size_t row = 0; row < row_labels.size(); ++row) {
        for (size_t column = 0; column < column_count; ++column) {
            expanded[matrix_index(row, column, column_count)] = row_labels[row];
        }
    }
    return expanded;
}

const std::map<Fr, size_t>& row_position_by_label(size_t table_row_count)
{
    static std::map<size_t, std::map<Fr, size_t>> cache;
    const auto it = cache.find(table_row_count);
    if (it != cache.end()) {
        return it->second;
    }

    std::map<Fr, size_t> mapping;
    for (size_t row = 0; row < table_row_count; ++row) {
        mapping.emplace(subgroup_label(table_row_count, row), row);
    }
    return cache.emplace(table_row_count, std::move(mapping)).first->second;
}

std::string field_key(const Fr& value)
{
    return value.getStr(10);
}

std::string matrix_row_key(const std::vector<Fr>& evals,
                           size_t row,
                           size_t row_count,
                           size_t column_count)
{
    if (row >= row_count) {
        throw std::runtime_error("tableLookupWitness: row index out of range");
    }

    std::string key;
    for (size_t column = 0; column < column_count; ++column) {
        if (column != 0) {
            key.push_back('|');
        }
        key += field_key(evals[matrix_index(row, column, column_count)]);
    }
    return key;
}

std::vector<Fr> build_row_labels_from_scan(const std::vector<Fr>& query_evals,
                                           const std::vector<Fr>& table_evals,
                                           size_t query_row_count,
                                           size_t table_row_count,
                                           size_t column_count)
{
    std::map<std::string, size_t> first_row_by_key;
    for (size_t table_row = 0; table_row < table_row_count; ++table_row) {
        first_row_by_key.emplace(
            matrix_row_key(table_evals, table_row, table_row_count, column_count), table_row);
    }

    std::vector<Fr> row_labels;
    row_labels.reserve(query_row_count);

    for (size_t query_row = 0; query_row < query_row_count; ++query_row) {
        const auto it =
            first_row_by_key.find(matrix_row_key(query_evals, query_row, query_row_count, column_count));
        if (it == first_row_by_key.end()) {
            // Missing rows fall back to subgroup label 1; the later reduced
            // lookup proof will reject because the labeled row is not in-table.
            row_labels.push_back(Fr(1));
        } else {
            row_labels.push_back(subgroup_label(table_row_count, it->second));
        }
    }

    return row_labels;
}

void report_missing_table_lookup_rows(const std::vector<Fr>& query_evals,
                                      const std::vector<Fr>& table_evals,
                                      size_t query_row_count,
                                      size_t table_row_count,
                                      size_t column_count)
{
    if (!table_lookup_failure_debug_enabled()) {
        return;
    }

    std::map<std::string, size_t> first_row_by_key;
    for (size_t table_row = 0; table_row < table_row_count; ++table_row) {
        first_row_by_key.emplace(
            matrix_row_key(table_evals, table_row, table_row_count, column_count), table_row);
    }

    size_t missing_count = 0;
    std::vector<size_t> first_missing;
    for (size_t query_row = 0; query_row < query_row_count; ++query_row) {
        if (first_row_by_key.find(matrix_row_key(query_evals, query_row, query_row_count, column_count)) !=
            first_row_by_key.end()) {
            continue;
        }
        ++missing_count;
        if (first_missing.size() < 8) {
            first_missing.push_back(query_row);
        }
    }
    if (missing_count == 0) {
        return;
    }

    std::cerr << "table_lookup_debug missing_rows"
              << " query_rows=" << query_row_count
              << " table_rows=" << table_row_count
              << " columns=" << column_count
              << " missing_count=" << missing_count;
    for (size_t row : first_missing) {
        std::cerr << " row=" << row << ":";
        for (size_t column = 0; column < column_count; ++column) {
            if (column != 0) {
                std::cerr << ",";
            }
            std::cerr << query_evals[matrix_index(row, column, column_count)].getStr(10);
        }
    }
    std::cerr << "\n";
}

void validate_witness_shape(const tableLookupWitness& witness,
                            size_t query_row_count,
                            const char* context)
{
    if (witness.row_labels.size() != query_row_count) {
        throw std::runtime_error(std::string(context) + ": witness row count mismatch");
    }
}

Group1 sparse_linear_combination(const std::vector<Group1>& bases,
                                 const std::vector<std::pair<size_t, Fr>>& terms,
                                 const char* context)
{
    std::vector<G1> selected_bases;
    std::vector<Fr> selected_scalars;
    selected_bases.reserve(terms.size());
    selected_scalars.reserve(terms.size());
    for (const auto& [index, coeff] : terms) {
        if (index >= bases.size()) {
            throw std::runtime_error(std::string(context) + ": sparse basis index out of range");
        }
        if (!coeff.isZero()) {
            selected_bases.push_back(bases[index].v);
            selected_scalars.push_back(coeff);
        }
    }

    if (selected_bases.empty()) {
        return Group1::zero();
    }
    if (selected_bases.size() >= 16) {
        G1 out;
        G1::mulVec(out, selected_bases.data(), selected_scalars.data(), selected_bases.size());
        return Group1(out);
    }

    Group1 sum = Group1::zero();
    for (size_t i = 0; i < selected_bases.size(); ++i) {
        sum += selected_scalars[i] * Group1(selected_bases[i]);
    }
    return sum;
}

kzgCommitment row_label_commitment_for_shape(kzgProtocol& kzg, size_t row_count, size_t column_count)
{
    const MatrixShapeData& shape = get_matrix_shape_data(row_count, column_count);
    const std::string key = row_label_helper_key(row_count, column_count);
    const std::shared_ptr<ProofSession> proof_session = kzg.proof_session();
    const bool verifier_mode = proof_session && proof_session->mode() == ProofMode::Verify;
    if (verifier_mode) {
        const kzgCommitment commitment =
            kzg.adopt_public_commitment(key, preprocessed_row_label_group(kzg, row_count, column_count));
        if (!kzg.has_degree_bound(virCommitment(commitment), row_count * column_count - 1)) {
            kzg.prove_degree(commitment, row_count * column_count - 1);
        }
        return commitment;
    }

    const Group1 row_label_group = preprocessed_row_label_group(kzg, row_count, column_count);
    const kzgCommitment commitment =
        kzg.commit_public_with_group(key, shape.row_label_poly, row_label_group, row_count * column_count - 1);
    cache_row_label_group(kzg, row_count, column_count, commitment.com);
    return commitment;
}

virCommitment build_labeled_commitment(const virCommitment& values,
                                       const poly& column_label_poly,
                                       const virCommitment& row_labels,
                                       const Fr& rho)
{
    Fr rho_square;
    Fr::mul(rho_square, rho, rho);
    return values + scale_by_constant(virCommitment(column_label_poly), rho) +
           scale_by_constant(row_labels, rho_square);
}

virCommitment row_step_constraint(const virCommitment& row_indices, size_t query_row_count, size_t column_count)
{
    const size_t domain_size = query_row_count * column_count;
    const Fr backward_shift = inverse_root_of_unity(domain_size);
    return (x_scale(row_indices, backward_shift) - row_indices) * virCommitment(vanishing_poly(query_row_count));
}

PendingRowIndexWitness build_pending_row_index_witness(kzgProtocol& kzg,
                                                       const std::vector<Fr>& row_labels,
                                                       size_t column_count)
{
    const std::vector<Fr> expanded_row_labels = expand_row_labels(row_labels, column_count);
    PendingRowIndexWitness out;
    out.row_index_poly = poly_from_evals(expanded_row_labels);
    out.row_index_group = kzg.commit_g1_group(out.row_index_poly);
    return out;
}

tableLookupCheckRecord make_table_lookup_record(kzgProtocol& kzg,
                                                const std::vector<Fr>& row_labels,
                                                const Fr& rho,
                                                size_t query_row_count,
                                                size_t table_row_count,
                                                size_t column_count)
{
    validate_witness_shape(tableLookupWitness{row_labels}, query_row_count, "tableLookupProver");

    tableLookupCheckRecord record;
    record.query_row_count = query_row_count;
    record.table_row_count = table_row_count;
    record.column_count = column_count;
    record.rho = rho;

    const PendingRowIndexWitness pending = build_pending_row_index_witness(kzg, row_labels, column_count);
    kzgCommitment precomputed_row_index;
    precomputed_row_index.com = pending.row_index_group;
    record.row_index_commitment = kzg.commit(pending.row_index_poly, precomputed_row_index);
    if (record.row_index_commitment.com != pending.row_index_group) {
        throw std::runtime_error("tableLookupProver: row index commitment registration mismatch");
    }
    kzg.prove_degree(record.row_index_commitment, query_row_count * column_count - 1);
    return record;
}

cqTableLookupCheckRecord make_cq_table_lookup_record(kzgProtocol& kzg,
                                                     const std::vector<Fr>& row_labels,
                                                     const Fr& rho,
                                                     size_t query_row_count,
                                                     size_t column_count)
{
    validate_witness_shape(tableLookupWitness{row_labels}, query_row_count, "cqTableLookupProver");

    cqTableLookupCheckRecord record;
    record.query_row_count = query_row_count;
    record.column_count = column_count;
    record.rho = rho;

    const PendingRowIndexWitness pending = build_pending_row_index_witness(kzg, row_labels, column_count);
    kzgCommitment precomputed_row_index;
    precomputed_row_index.com = pending.row_index_group;
    record.row_index_commitment = kzg.commit(pending.row_index_poly, precomputed_row_index);
    if (record.row_index_commitment.com != pending.row_index_group) {
        throw std::runtime_error("cqTableLookupProver: row index commitment registration mismatch");
    }
    kzg.prove_degree(record.row_index_commitment, query_row_count * column_count - 1);
    return record;
}

PendingRowIndexWitness make_table_lookup_prechallenge_witness(kzgProtocol& kzg,
                                                              const virCommitment& query,
                                                              const virCommitment& table,
                                                              size_t query_row_count,
                                                              size_t table_row_count,
                                                              size_t column_count,
                                                              const tableLookupWitness* witness)
{
    if (witness == nullptr) {
        const size_t query_domain_size = query_row_count * column_count;
        const size_t table_domain_size = table_row_count * column_count;
        const std::vector<Fr> query_evals =
            kzg.evaluations(query, query_domain_size, kzgProtocol::EvaluationMode::PreferEval);
        const std::vector<Fr> table_evals =
            kzg.evaluations(table, table_domain_size, kzgProtocol::EvaluationMode::PreferEval);
        report_missing_table_lookup_rows(
            query_evals, table_evals, query_row_count, table_row_count, column_count);
        const std::vector<Fr> row_labels =
            build_row_labels_from_scan(query_evals, table_evals, query_row_count, table_row_count, column_count);
        return build_pending_row_index_witness(kzg, row_labels, column_count);
    }

    validate_witness_shape(*witness, query_row_count, "tableLookupProtocol");
    return build_pending_row_index_witness(kzg, witness->row_labels, column_count);
}

PendingRowIndexWitness make_cq_table_lookup_prechallenge_witness(kzgProtocol& kzg,
                                                                 const virCommitment& query,
                                                                 size_t query_row_count,
                                                                 const cqTableLookupTable& table,
                                                                 const tableLookupWitness* witness)
{
    const size_t column_count = table.column_count;
    if (witness == nullptr) {
        if (table.table_evals == nullptr) {
            throw std::runtime_error("cqTableLookupProtocol: prover requires table evaluations for implicit witness construction");
        }
        const size_t column_count = table.column_count;
        const size_t query_domain_size = query_row_count * column_count;
        const std::vector<Fr> query_evals =
            kzg.evaluations(query, query_domain_size, kzgProtocol::EvaluationMode::PreferEval);
        const std::vector<Fr> row_labels =
            build_row_labels_from_scan(query_evals, *table.table_evals, query_row_count, table.row_count, column_count);
        return build_pending_row_index_witness(kzg, row_labels, column_count);
    }

    validate_witness_shape(*witness, query_row_count, "cqTableLookupProtocol");
    return build_pending_row_index_witness(kzg, witness->row_labels, column_count);
}

cqLookupSparseWitness cq_table_lookup_sparse_witness_from_witness(const cqTableLookupTable& table,
                                                                  const cqLookupTable& reduced_table,
                                                                  size_t table_row_count,
                                                                  size_t column_count,
                                                                  const tableLookupWitness& witness,
                                                                  const Fr& rho)
{
    if (table.table_evals == nullptr || table.table_quotient_commitments == nullptr ||
        table.row_label_quotient_commitments == nullptr || table.column_label_quotient_commitments == nullptr) {
        throw std::runtime_error("cqTableLookupProtocol: sparse witness requires table-side preprocessing");
    }
    if (table.row_label_evals == nullptr || table.column_label_evals == nullptr) {
        throw std::runtime_error("cqTableLookupProtocol: sparse witness requires label evaluations");
    }
    if (reduced_table.lagrange_commitments == nullptr || reduced_table.a0_quotient_commitments == nullptr) {
        throw std::runtime_error("cqTableLookupProtocol: sparse witness requires domain preprocessing");
    }

    const std::map<Fr, size_t>& row_position_lookup = row_position_by_label(table_row_count);
    std::map<size_t, Fr> frequency_by_index;
    for (const Fr& row_label : witness.row_labels) {
        const auto row_it = row_position_lookup.find(row_label);
        if (row_it == row_position_lookup.end()) {
            throw std::runtime_error("cqTableLookupProtocol: row label is not a valid subgroup label");
        }
        for (size_t column = 0; column < column_count; ++column) {
            const size_t index = matrix_index(row_it->second, column, column_count);
            auto [frequency_it, _] = frequency_by_index.emplace(index, Fr(0));
            Fr::add(frequency_it->second, frequency_it->second, Fr(1));
        }
    }

    const Fr rho_square = rho * rho;
    cqLookupSparseWitness out;
    out.multiplicity_terms.reserve(frequency_by_index.size());
    out.table_evals_at_terms.reserve(frequency_by_index.size());
    out.quotient_commitments_at_terms.reserve(frequency_by_index.size());

    for (const auto& [index, frequency] : frequency_by_index) {
        if (index >= table.table_evals->size() || index >= table.table_quotient_commitments->size() ||
            index >= table.row_label_quotient_commitments->size() ||
            index >= table.column_label_quotient_commitments->size()) {
            throw std::runtime_error("cqTableLookupProtocol: sparse witness table index out of range");
        }
        if (index >= table.row_label_evals->size() || index >= table.column_label_evals->size()) {
            throw std::runtime_error("cqTableLookupProtocol: sparse witness label index out of range");
        }
        const Fr row_label = (*table.row_label_evals)[index];
        const Fr column_label = (*table.column_label_evals)[index];

        Fr labeled_eval = (*table.table_evals)[index];
        Fr column_term;
        Fr::mul(column_term, rho, column_label);
        Fr::add(labeled_eval, labeled_eval, column_term);
        Fr row_term;
        Fr::mul(row_term, rho_square, row_label);
        Fr::add(labeled_eval, labeled_eval, row_term);

        const Group1 labeled_quotient = (*table.table_quotient_commitments)[index] +
                                        rho * (*table.column_label_quotient_commitments)[index] +
                                        rho_square * (*table.row_label_quotient_commitments)[index];

        out.multiplicity_terms.emplace_back(index, frequency);
        out.table_evals_at_terms.push_back(labeled_eval);
        out.quotient_commitments_at_terms.push_back(labeled_quotient);
    }

    out.multiplicity_commitment =
        sparse_linear_combination(*reduced_table.lagrange_commitments,
                                  out.multiplicity_terms,
                                  "cqTableLookupProtocol");
    return out;
}

} // namespace

void preprocess_table_lookup_row_labels(kzgProtocol& kzg, size_t row_count, size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "preprocess_table_lookup_row_labels");
    (void)preprocessed_row_label_group(kzg, row_count, column_count);
}

cqTableLookupPublicTable build_cq_table_lookup_public_table(const kzgPublicParameters& public_params,
                                                            std::string key,
                                                            const poly& table,
                                                            size_t row_count,
                                                            size_t column_count,
                                                            std::optional<size_t> degree_bound)
{
    validate_matrix_shape(row_count, column_count, "build_cq_table_lookup_public_table");
    const MatrixShapeData& shape = get_matrix_shape_data(row_count, column_count);
    kzgProtocol temp_kzg;
    temp_kzg.setup(public_params);

    cqTableLookupPublicTable out;
    out.key = std::move(key);
    out.row_count = row_count;
    out.column_count = column_count;
    out.domain_size = row_count * column_count;
    out.degree_bound = degree_bound;
    out.commitment = commit_g1_with_public_parameters(public_params, table);
    out.table_g2 = commit_g2_with_public_parameters(public_params, table);
    out.table_evals = std::make_shared<std::vector<Fr>>(table.evaluations_on_domain(out.domain_size));
    fill_cq_lookup_domain_precompute(
        public_params,
        out.domain_size,
        out.lagrange_at_zero,
        out.vanishing_g2,
        out.lagrange_commitments,
        out.a0_quotient_commitments);
    out.table_quotient_commitments = precompute_cq_quotient_commitments(temp_kzg, table, out.domain_size);
    out.row_label_g2 = commit_g2_with_public_parameters(public_params, shape.row_label_poly);
    out.column_label_g2 = commit_g2_with_public_parameters(public_params, shape.column_label_poly);
    out.row_label_g1 = commit_g1_with_public_parameters(public_params, shape.row_label_poly);
    out.row_label_quotient_commitments =
        precompute_cq_quotient_commitments(temp_kzg, shape.row_label_poly, out.domain_size);
    out.column_label_quotient_commitments =
        precompute_cq_quotient_commitments(temp_kzg, shape.column_label_poly, out.domain_size);
    return out;
}

tableLookupWitness tableLookupWitness::from_positions(size_t table_row_count,
                                                      const std::vector<size_t>& row_positions)
{
    tableLookupWitness witness;
    witness.row_labels = row_labels_from_positions(table_row_count, row_positions);
    return witness;
}

Fr tableLookupVerifier::challenge()
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("tableLookupVerifier not setup");
    }
    return params->kzg->transcript().squeeze_nonzero_fr("table_lookup.rho");
}

void tableLookupVerifier::add_check(const tableLookupCheckRecord& record)
{
    checks.push_back(record);
}

bool tableLookupVerifier::finalize() const
{
    return true;
}

tableLookupCheckRecord tableLookupProver::add_check(const virCommitment& query,
                                                    const virCommitment& table,
                                                    const Fr& rho,
                                                    size_t query_row_count,
                                                    size_t table_row_count,
                                                    size_t column_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("tableLookupProver not setup");
    }

    const size_t query_domain_size = query_row_count * column_count;
    const size_t table_domain_size = table_row_count * column_count;
    const std::vector<Fr> query_evals = params->kzg->evaluations(query, query_domain_size);
    const std::vector<Fr> table_evals = params->kzg->evaluations(table, table_domain_size);
    const std::vector<Fr> row_labels =
        build_row_labels_from_scan(query_evals, table_evals, query_row_count, table_row_count, column_count);

    tableLookupCheckRecord record = make_table_lookup_record(
        *params->kzg, row_labels, rho, query_row_count, table_row_count, column_count);
    checks.push_back(record);
    return record;
}

tableLookupCheckRecord tableLookupProver::add_check(const tableLookupWitness& witness,
                                                    const Fr& rho,
                                                    size_t query_row_count,
                                                    size_t table_row_count,
                                                    size_t column_count)
{
    if (params == nullptr || params->kzg == nullptr) {
        throw std::runtime_error("tableLookupProver not setup");
    }

    validate_witness_shape(witness, query_row_count, "tableLookupProver");
    tableLookupCheckRecord record = make_table_lookup_record(
        *params->kzg, witness.row_labels, rho, query_row_count, table_row_count, column_count);
    checks.push_back(record);
    return record;
}

void tableLookupProtocol::add_check(const virCommitment& query,
                                    const virCommitment& table,
                                    size_t query_rows,
                                    size_t table_rows,
                                    size_t columns)
{
    add_check_with_optional_witness(query, table, query_rows, table_rows, columns, nullptr, nullptr);
}

void tableLookupProtocol::add_check(const virCommitment& query,
                                    const virCommitment& table,
                                    size_t query_rows,
                                    size_t table_rows,
                                    size_t columns,
                                    const tableLookupWitness& witness)
{
    add_check_with_optional_witness(query, table, query_rows, table_rows, columns, &witness, nullptr);
}

void tableLookupProtocol::add_check(const virCommitment& query,
                                    const virCommitment& table,
                                    size_t query_rows,
                                    size_t table_rows,
                                    size_t columns,
                                    const tableLookupWitness& witness,
                                    tableLookupCheckRecord& out_record)
{
    add_check_with_optional_witness(query, table, query_rows, table_rows, columns, &witness, &out_record);
}

void tableLookupProtocol::add_check_with_optional_witness(const virCommitment& query,
                                                          const virCommitment& table,
                                                          size_t query_rows,
                                                          size_t table_rows,
                                                          size_t columns,
                                                          const tableLookupWitness* witness,
                                                          tableLookupCheckRecord* out_record)
{
    proof_cost::record_relation("table_lookup", query_rows * columns);
    structured_protocol_profile::timed_void(
        "protocol_call",
        "table_lookup.add_check",
        "query_rows=" + std::to_string(query_rows) + ";table_rows=" + std::to_string(table_rows) + ";columns=" +
            std::to_string(columns) + ";explicit_witness=" + std::to_string(witness != nullptr ? 1 : 0),
        [&] {
            if (!is_setup) {
                throw std::runtime_error("tableLookupProtocol not setup");
            }
            if (params.kzg == nullptr || params.zero_check == nullptr || params.sum_check == nullptr ||
                reduced_lookup == nullptr) {
                throw std::runtime_error("tableLookupProtocol dependencies not setup");
            }

            validate_matrix_shape(query_rows, columns, "tableLookupProtocol");
            validate_matrix_shape(table_rows, columns, "tableLookupProtocol");

            tableLookupCheckRecord record;
            const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                const Group1 row_index_group = proof_session->exchange_group1("table_lookup.row_index", {});
                record.row_index_commitment = params.kzg->adopt_commitment(row_index_group);
                params.kzg->prove_degree(record.row_index_commitment, query_rows * columns - 1);
            } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
                PendingRowIndexWitness pending;
                table_lookup_profile_scope("prove.prepare_row_index_witness", [&] {
                    pending = make_table_lookup_prechallenge_witness(
                        *params.kzg, query, table, query_rows, table_rows, columns, witness);
                });
                const Group1 row_index_group =
                    proof_session->exchange_group1("table_lookup.row_index", [&] { return pending.row_index_group; });
                if (row_index_group != pending.row_index_group) {
                    throw std::runtime_error("tableLookupProtocol: exchanged row index commitment mismatch");
                }
                table_lookup_profile_scope("prove.commit_row_index", [&] {
                    kzgCommitment precomputed_row_index;
                    precomputed_row_index.com = pending.row_index_group;
                    record.row_index_commitment = params.kzg->commit(pending.row_index_poly, precomputed_row_index);
                });
                if (record.row_index_commitment.com != pending.row_index_group) {
                    throw std::runtime_error("tableLookupProtocol: row index commitment registration mismatch");
                }
                table_lookup_profile_scope("prove.degree_row_index", [&] {
                    params.kzg->prove_degree(record.row_index_commitment, query_rows * columns - 1);
                });
            } else {
                record.query_row_count = query_rows;
                record.table_row_count = table_rows;
                record.column_count = columns;
            }

            const size_t query_domain_size = query_rows * columns;
            const size_t table_domain_size = table_rows * columns;
            const Fr rho = verifier.challenge();
            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                record.query_row_count = query_rows;
                record.table_row_count = table_rows;
                record.column_count = columns;
                record.rho = rho;
                verifier.add_check(record);
            } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
                record.query_row_count = query_rows;
                record.table_row_count = table_rows;
                record.column_count = columns;
                record.rho = rho;
                prover.checks.push_back(record);
                verifier.add_check(record);
            } else {
                const tableLookupCheckRecord prover_record =
                    witness == nullptr ? prover.add_check(query, table, rho, query_rows, table_rows, columns)
                                       : prover.add_check(*witness, rho, query_rows, table_rows, columns);
                record = prover_record;
                verifier.add_check(record);
            }
            if (out_record != nullptr) {
                *out_record = record;
            }

            const virCommitment row_index_vir(record.row_index_commitment);
            const virCommitment table_row_labels(row_label_commitment_for_shape(*params.kzg, table_rows, columns));

            const virCommitment labeled_query =
                build_labeled_commitment(query, column_label_poly_for_shape(query_rows, columns), row_index_vir, rho);
            const virCommitment labeled_table =
                build_labeled_commitment(table, column_label_poly_for_shape(table_rows, columns), table_row_labels, rho);

            table_lookup_profile_scope("reduced_lookup.add_check", [&] {
                reduced_lookup->add_check(labeled_query,
                                          labeled_table,
                                          query_domain_size,
                                          table_domain_size,
                                          "table_lookup.query_rows=" + std::to_string(query_rows) +
                                              ";table_rows=" + std::to_string(table_rows) +
                                              ";columns=" + std::to_string(columns));
            });

            virCommitment step_zero = row_step_constraint(row_index_vir, query_rows, columns);
            table_lookup_profile_scope("zero_check.add_check", [&] {
                params.zero_check->add_check(step_zero, query_domain_size);
            });
        });
}

bool tableLookupProtocol::finalize()
{
    begin_finalize("tableLookupProtocol");
    return verifier.finalize();
}

void cqTableLookupTable::setup(kzgProtocol& kzg,
                               const virCommitment& table,
                               size_t row_count,
                               size_t column_count)
{
    validate_matrix_shape(row_count, column_count, "cqTableLookupTable");

    public_key.clear();
    const MatrixShapeData& shape = get_matrix_shape_data(row_count, column_count);
    this->row_count = row_count;
    this->column_count = column_count;
    domain_size = row_count * column_count;
    table_commitment = table;
    table_evals = std::make_shared<std::vector<Fr>>(
        kzg.evaluations(table, domain_size, kzgProtocol::EvaluationMode::PreferEval));
    table_g2 = kzg.commit_g2(poly_from_evals(*table_evals));
    table_quotient_commitments = precompute_cq_quotient_commitments(kzg, poly_from_evals(*table_evals), domain_size);
    row_label_evals = shape.row_label_evals;
    column_label_evals = shape.column_label_evals;
    row_label_poly = shape.row_label_poly;
    column_label_poly = shape.column_label_poly;
    row_label_g2 = kzg.commit_g2(row_label_poly);
    column_label_g2 = kzg.commit_g2(column_label_poly);
    row_label_quotient_commitments = precompute_cq_quotient_commitments(kzg, row_label_poly, domain_size);
    column_label_quotient_commitments = precompute_cq_quotient_commitments(kzg, column_label_poly, domain_size);
}

void cqTableLookupTable::setup(const cqTableLookupPublicTable& table, const virCommitment& table_commitment)
{
    validate_matrix_shape(table.row_count, table.column_count, "cqTableLookupTable");
    if (table.domain_size != table.row_count * table.column_count) {
        throw std::runtime_error("cqTableLookupTable: domain size does not match row/column shape");
    }

    public_key = table.key;
    row_count = table.row_count;
    column_count = table.column_count;
    domain_size = table.domain_size;
    this->table_commitment = table_commitment;
    lagrange_at_zero = table.lagrange_at_zero;
    vanishing_g2 = table.vanishing_g2;
    table_g2 = table.table_g2;
    table_evals = table.table_evals;
    lagrange_commitments = table.lagrange_commitments;
    table_quotient_commitments = table.table_quotient_commitments;
    a0_quotient_commitments = table.a0_quotient_commitments;
    row_label_g2 = table.row_label_g2;
    column_label_g2 = table.column_label_g2;
    row_label_g1 = table.row_label_g1;
    row_label_quotient_commitments = table.row_label_quotient_commitments;
    column_label_quotient_commitments = table.column_label_quotient_commitments;

    if (table.table_evals != nullptr) {
        const MatrixShapeData& shape = get_matrix_shape_data(row_count, column_count);
        row_label_evals = shape.row_label_evals;
        column_label_evals = shape.column_label_evals;
        row_label_poly = shape.row_label_poly;
        column_label_poly = shape.column_label_poly;
        return;
    }

    row_label_evals.reset();
    column_label_evals.reset();
    row_label_poly = poly();
    column_label_poly = column_label_poly_for_shape(row_count, column_count);
}

Fr cqTableLookupVerifier::challenge()
{
    if (has_sampled_rho) {
        return sampled_rho;
    }
    if (params == nullptr || params->kzg == nullptr || params->table == nullptr) {
        throw std::runtime_error("cqTableLookupVerifier not setup");
    }

    sampled_rho = params->kzg->transcript().squeeze_nonzero_fr("cq_table_lookup.rho");
    has_sampled_rho = true;
    return sampled_rho;
}

void cqTableLookupVerifier::add_check(const cqTableLookupCheckRecord& record)
{
    checks.push_back(record);
}

bool cqTableLookupVerifier::finalize() const
{
    return true;
}

cqTableLookupCheckRecord cqTableLookupProver::add_check(const virCommitment& query,
                                                        size_t query_row_count,
                                                        const Fr& rho)
{
    if (params == nullptr || params->kzg == nullptr || params->table == nullptr) {
        throw std::runtime_error("cqTableLookupProver not setup");
    }
    if (params->table->table_evals == nullptr) {
        throw std::runtime_error("cqTableLookupProver requires table evaluations; verifier artifacts are not enough");
    }

    const size_t column_count = params->table->column_count;
    const size_t query_domain_size = query_row_count * column_count;
    const std::vector<Fr> query_evals = params->kzg->evaluations(query, query_domain_size);
    const std::vector<Fr> row_labels = build_row_labels_from_scan(
        query_evals, *params->table->table_evals, query_row_count, params->table->row_count, column_count);

    cqTableLookupCheckRecord record =
        make_cq_table_lookup_record(*params->kzg, row_labels, rho, query_row_count, column_count);
    checks.push_back(record);
    return record;
}

cqTableLookupCheckRecord cqTableLookupProver::add_check(const tableLookupWitness& witness,
                                                        size_t query_row_count,
                                                        const Fr& rho)
{
    if (params == nullptr || params->kzg == nullptr || params->table == nullptr) {
        throw std::runtime_error("cqTableLookupProver not setup");
    }

    validate_witness_shape(witness, query_row_count, "cqTableLookupProver");
    cqTableLookupCheckRecord record = make_cq_table_lookup_record(
        *params->kzg, witness.row_labels, rho, query_row_count, params->table->column_count);
    checks.push_back(record);
    return record;
}

virCommitment cq_row_label_commitment(kzgProtocol& kzg, const cqTableLookupTable& table)
{
    const std::shared_ptr<ProofSession> proof_session = kzg.proof_session();
    const bool verifier_mode = proof_session && proof_session->mode() == ProofMode::Verify;

    if (kzg.is_count_only() && !table.public_key.empty()) {
        const std::string key = cq_row_label_helper_key(table.public_key);
        const kzgCommitment commitment = kzg.adopt_public_commitment(key, Group1::zero());
        if (!kzg.has_degree_bound(virCommitment(commitment), table.domain_size - 1)) {
            kzg.prove_degree(commitment, table.domain_size - 1);
        }
        return virCommitment(commitment);
    }

    if (!table.public_key.empty() && table.row_label_g1 != Group1::zero()) {
        const std::string key = cq_row_label_helper_key(table.public_key);
        if (verifier_mode) {
            const kzgCommitment commitment = kzg.adopt_public_commitment(key, table.row_label_g1);
            if (!kzg.has_degree_bound(virCommitment(commitment), table.domain_size - 1)) {
                kzg.prove_degree(commitment, table.domain_size - 1);
            }
            return virCommitment(commitment);
        }
        const kzgCommitment commitment =
            kzg.commit_public_with_group(key, table.row_label_poly, table.row_label_g1, table.domain_size - 1);
        return virCommitment(commitment);
    }

    if (table.row_label_g1 != Group1::zero() && !table.row_label_poly.coefficients().empty()) {
        kzgCommitment precomputed;
        precomputed.com = table.row_label_g1;
        return virCommitment(kzg.commit(table.row_label_poly, precomputed));
    }
    return virCommitment(row_label_commitment_for_shape(kzg, table.row_count, table.column_count));
}

void cqTableLookupProtocol::ensure_reduced_lookup_ready(const Fr& rho, bool needs_dense_table)
{
    if (reduced_lookup_ready) {
        return;
    }
    if (params.kzg == nullptr || params.table == nullptr) {
        throw std::runtime_error("cqTableLookupProtocol not setup");
    }

    const virCommitment labeled_table =
        build_labeled_commitment(params.table->table_commitment,
                                 params.table->column_label_poly,
                                 cq_row_label_commitment(*params.kzg, *params.table),
                                 rho);
    const Group2 labeled_table_g2 =
        params.table->table_g2 + rho * params.table->column_label_g2 + (rho * rho) * params.table->row_label_g2;
    if (params.table->table_evals != nullptr && needs_dense_table) {
        // Prover path: materialize the same labeled table evaluations so CQ
        // preprocessing can be rebuilt without touching the original matrix
        // lookup reduction again.
        std::vector<Fr> labeled_table_evals(params.table->domain_size, Fr(0));
        Fr rho_square;
        Fr::mul(rho_square, rho, rho);
        for (size_t i = 0; i < labeled_table_evals.size(); ++i) {
            Fr labeled = (*params.table->table_evals)[i];
            Fr column_term;
            Fr::mul(column_term, rho, (*params.table->column_label_evals)[i]);
            Fr::add(labeled, labeled, column_term);
            Fr row_term;
            Fr::mul(row_term, rho_square, (*params.table->row_label_evals)[i]);
            Fr::add(labeled, labeled, row_term);
            labeled_table_evals[i] = labeled;
        }
        std::shared_ptr<std::vector<Group1>> labeled_quotients = std::make_shared<std::vector<Group1>>();
        if (params.table->table_quotient_commitments == nullptr ||
            params.table->column_label_quotient_commitments == nullptr ||
            params.table->row_label_quotient_commitments == nullptr) {
            reduced_table.setup_from_evaluations(*params.kzg, labeled_table, labeled_table_evals);
        } else {
            labeled_quotients->resize(params.table->table_quotient_commitments->size(), Group1::zero());
            const Fr rho_square = rho * rho;
            for (size_t i = 0; i < labeled_quotients->size(); ++i) {
                (*labeled_quotients)[i] = (*params.table->table_quotient_commitments)[i] +
                                          rho * (*params.table->column_label_quotient_commitments)[i] +
                                          rho_square * (*params.table->row_label_quotient_commitments)[i];
            }
            reduced_table.setup_from_evaluations_with_precomputed(
                *params.kzg,
                labeled_table,
                labeled_table_evals,
                labeled_table_g2,
                labeled_quotients,
                nullptr,
                params.table->lagrange_commitments,
                params.table->a0_quotient_commitments,
                params.table->lagrange_at_zero,
                params.table->vanishing_g2);
        }
    } else if (params.table->lagrange_commitments != nullptr && params.table->a0_quotient_commitments != nullptr &&
               params.table->lagrange_at_zero != Fr(0)) {
        reduced_table.setup_verifier(*params.kzg,
                                     labeled_table,
                                     params.table->domain_size,
                                     labeled_table_g2,
                                     params.table->lagrange_commitments,
                                     params.table->a0_quotient_commitments,
                                     params.table->lagrange_at_zero,
                                     params.table->vanishing_g2);
    } else {
        reduced_table.setup_verifier(*params.kzg,
                                     labeled_table,
                                     params.table->domain_size,
                                     labeled_table_g2,
                                     params.table->lagrange_commitments,
                                     params.table->a0_quotient_commitments,
                                     params.table->lagrange_at_zero,
                                     params.table->vanishing_g2);
    }
    if (reduced_lookup == nullptr) {
        throw std::runtime_error("cqTableLookupProtocol reduced lookup not provided");
    }
    reduced_lookup->setup(*params.kzg, reduced_table);
    reduced_lookup_ready = true;
}

void cqTableLookupProtocol::add_check(const virCommitment& query, size_t query_rows)
{
    add_check_with_optional_witness(query, query_rows, nullptr, nullptr);
}

void cqTableLookupProtocol::add_check(const virCommitment& query,
                                      size_t query_rows,
                                      cqTableLookupCheckRecord& out_record)
{
    add_check_with_optional_witness(query, query_rows, nullptr, &out_record);
}

void cqTableLookupProtocol::add_check(const virCommitment& query,
                                      size_t query_rows,
                                      const tableLookupWitness& witness)
{
    add_check_with_optional_witness(query, query_rows, &witness, nullptr);
}

void cqTableLookupProtocol::add_check(const virCommitment& query,
                                      size_t query_rows,
                                      const tableLookupWitness& witness,
                                      cqTableLookupCheckRecord& out_record)
{
    add_check_with_optional_witness(query, query_rows, &witness, &out_record);
}

void cqTableLookupProtocol::add_check_with_optional_witness(const virCommitment& query,
                                                            size_t query_rows,
                                                            const tableLookupWitness* witness,
                                                            cqTableLookupCheckRecord* out_record)
{
    structured_protocol_profile::timed_void(
        "protocol_call",
        "cq_table_lookup.add_check",
        "query_rows=" + std::to_string(query_rows) + ";table_rows=" + std::to_string(params.table->row_count) +
            ";columns=" + std::to_string(params.table->column_count) + ";explicit_witness=" +
            std::to_string(witness != nullptr ? 1 : 0),
        [&] {
            if (!is_setup) {
                throw std::runtime_error("cqTableLookupProtocol not setup");
            }
            if (params.kzg == nullptr || params.zero_check == nullptr || params.table == nullptr ||
                reduced_lookup == nullptr) {
                throw std::runtime_error("cqTableLookupProtocol dependencies not setup");
            }

            validate_matrix_shape(query_rows, params.table->column_count, "cqTableLookupProtocol");
            cqTableLookupCheckRecord record;
            const std::shared_ptr<ProofSession> proof_session = params.kzg->proof_session();
            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                const Group1 row_index_group = proof_session->exchange_group1("cq_table_lookup.row_index", {});
                record.row_index_commitment = params.kzg->adopt_commitment(row_index_group);
                params.kzg->prove_degree(record.row_index_commitment, query_rows * params.table->column_count - 1);
            } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
                const PendingRowIndexWitness pending = make_cq_table_lookup_prechallenge_witness(
                    *params.kzg, query, query_rows, *params.table, witness);
                const Group1 row_index_group = proof_session->exchange_group1(
                    "cq_table_lookup.row_index", [&] { return pending.row_index_group; });
                if (row_index_group != pending.row_index_group) {
                    throw std::runtime_error("cqTableLookupProtocol: exchanged row index commitment mismatch");
                }
                kzgCommitment precomputed_row_index;
                precomputed_row_index.com = pending.row_index_group;
                record.row_index_commitment = params.kzg->commit(pending.row_index_poly, precomputed_row_index);
                if (record.row_index_commitment.com != pending.row_index_group) {
                    throw std::runtime_error("cqTableLookupProtocol: row index commitment registration mismatch");
                }
                params.kzg->prove_degree(record.row_index_commitment, query_rows * params.table->column_count - 1);
            } else {
                record.query_row_count = query_rows;
                record.column_count = params.table->column_count;
            }

            const size_t query_domain_size = query_rows * params.table->column_count;
            const Fr rho = verifier.challenge();
            ensure_reduced_lookup_ready(rho, witness == nullptr);
            if (proof_session && proof_session->mode() == ProofMode::Verify) {
                record.query_row_count = query_rows;
                record.column_count = params.table->column_count;
                record.rho = rho;
                verifier.add_check(record);
            } else if (proof_session && proof_session->mode() == ProofMode::Prove) {
                record.query_row_count = query_rows;
                record.column_count = params.table->column_count;
                record.rho = rho;
                prover.checks.push_back(record);
                verifier.add_check(record);
            } else {
                const cqTableLookupCheckRecord prover_record =
                    witness == nullptr ? prover.add_check(query, query_rows, rho)
                                       : prover.add_check(*witness, query_rows, rho);
                record = prover_record;
                verifier.add_check(record);
            }
            if (out_record != nullptr) {
                *out_record = record;
            }

            const virCommitment row_index_vir(record.row_index_commitment);
            const virCommitment labeled_query =
                build_labeled_commitment(
                    query, column_label_poly_for_shape(query_rows, params.table->column_count), row_index_vir, rho);
            if (witness != nullptr && proof_session && proof_session->mode() == ProofMode::Prove) {
                const cqLookupSparseWitness sparse_witness =
                    cq_table_lookup_sparse_witness_from_witness(
                        *params.table, reduced_table, params.table->row_count, params.table->column_count, *witness, rho);
                reduced_lookup->add_check(labeled_query, query_domain_size, sparse_witness);
            } else if (witness != nullptr && !proof_session) {
                const cqLookupSparseWitness sparse_witness =
                    cq_table_lookup_sparse_witness_from_witness(
                        *params.table, reduced_table, params.table->row_count, params.table->column_count, *witness, rho);
                reduced_lookup->add_check(labeled_query, query_domain_size, sparse_witness);
            } else {
                reduced_lookup->add_check(labeled_query, query_domain_size);
            }

            virCommitment step_zero = row_step_constraint(row_index_vir, query_rows, params.table->column_count);
            params.zero_check->add_check(step_zero, query_domain_size);
        });
}

bool cqTableLookupProtocol::finalize()
{
    begin_finalize("cqTableLookupProtocol");
    return verifier.finalize();
}
