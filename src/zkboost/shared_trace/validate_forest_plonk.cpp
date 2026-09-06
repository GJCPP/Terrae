#include "zkboost/shared_trace/validate_forest_plonk.hpp"

#include "algebra/algebra_utils.hpp"
#include "commitments/transcript.hpp"
#include "zkboost/compiler/generic_circuit.hpp"

#include <array>
#include <iomanip>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

Fr shared_int64_to_fr(std::int64_t value)
{
    return fr_from_dec(std::to_string(value));
}

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_validate_forest_plonk: " + message);
    }
}

std::int64_t checked_i64(size_t value, const std::string& name)
{
    require_condition(value <= static_cast<size_t>(std::numeric_limits<std::int64_t>::max()),
                      name + " exceeds int64 range");
    return static_cast<std::int64_t>(value);
}

size_t query_input_offset(size_t feature_count, size_t query, size_t feature)
{
    return query * feature_count + feature;
}

std::int64_t query_feature_lookup_tag(size_t query)
{
    constexpr std::int64_t kTagBase = -6000000000LL;
    require_condition(query <= static_cast<size_t>(std::numeric_limits<std::int64_t>::max()),
                      "query index exceeds int64 range");
    const std::int64_t query_i64 = static_cast<std::int64_t>(query);
    require_condition(query_i64 <= std::numeric_limits<std::int64_t>::max() + kTagBase,
                      "query feature lookup tag underflows int64");
    return kTagBase - query_i64;
}

class sharedValidateForestCircuitCompiler : public zkboostGenericCircuitCompilerBase {
public:
    sharedValidateForestCircuitCompiler(const zkboostSharedValidateForestView& view,
                                        size_t bin_count,
                                        zkboostSharedValidateForestPublicInputMode public_input_mode)
        : view_(view)
        , bin_count_(bin_count)
        , public_input_mode_(public_input_mode)
    {
        require_condition(bin_count_ > 0, "bin_count must be non-zero");
        require_condition(view_.summary.feature_count > 0, "feature_count must be non-zero");
        require_condition(view_.fetch.public_split_masks.size() == view_.summary.node_count,
                          "public_split_masks length mismatch");
        require_condition(view_.fetch.public_split_features.size() == view_.summary.node_count,
                          "public_split_features length mismatch");
        require_condition(view_.fetch.public_split_thresholds.size() == view_.summary.node_count,
                          "public_split_thresholds length mismatch");
        require_condition(view_.routing.public_query_inputs.size() ==
                              view_.summary.checked_paths * view_.summary.feature_count,
                          "public_query_inputs length mismatch");
        zero_ = constant(0);
        one_ = constant(1);
        add_lookup_table();
        initialize_public_inputs();
    }

    void compile()
    {
        for (const zkboostSharedForestPathClaim& claim : view_.routing.path_claims) {
            compile_path_claim(claim);
        }
        compile_public_forest_field_claims();
    }

private:
    void add_lookup_table()
    {
        // The routing circuit keeps only the comparison relation as a VK-fixed
        // lookup table. Forest and input fetches are supplied via the separate
        // opening-based fetch proof.
        for (size_t threshold = 0; threshold < bin_count_; ++threshold) {
            for (size_t value = 0; value < bin_count_; ++value) {
                add_lookup_table_row(static_cast<std::int64_t>(threshold),
                                     static_cast<std::int64_t>(value),
                                     value < threshold ? 0 : 1);
            }
        }

        // ZKBoost's ValidateInference batches path-feature selection through a
        // permutation/lookup argument. Tag every (feature, value) pair by its
        // query so a path step needs one lookup instead of F one-hot products.
        for (size_t query = 0; query < view_.summary.checked_paths; ++query) {
            const std::int64_t tag = query_feature_lookup_tag(query);
            for (size_t feature = 0; feature < view_.summary.feature_count; ++feature) {
                add_lookup_table_row(
                    checked_i64(feature, "query feature lookup index"),
                    view_.routing.public_query_inputs.at(
                        query_input_offset(view_.summary.feature_count, query, feature)),
                    tag);
            }
        }
    }

    void initialize_public_inputs()
    {
        public_trace_leaf_indices_.reserve(view_.routing.trace_leaf_indices.size());
        for (std::int64_t value : view_.routing.trace_leaf_indices) {
            public_trace_leaf_indices_.push_back(bound_input_value(value));
        }

        public_tree_scores_.reserve(view_.routing.tree_scores.size());
        for (std::int64_t value : view_.routing.tree_scores) {
            public_tree_scores_.push_back(bound_input_value(value));
        }

        public_split_masks_.reserve(view_.fetch.public_split_masks.size());
        for (std::int64_t value : view_.fetch.public_split_masks) {
            public_split_masks_.push_back(bound_input_value(value));
        }

        public_split_features_.reserve(view_.fetch.public_split_features.size());
        for (std::int64_t value : view_.fetch.public_split_features) {
            public_split_features_.push_back(bound_input_value(value));
        }

        public_split_thresholds_.reserve(view_.fetch.public_split_thresholds.size());
        for (std::int64_t value : view_.fetch.public_split_thresholds) {
            public_split_thresholds_.push_back(bound_input_value(value));
        }

        public_query_inputs_.reserve(view_.routing.public_query_inputs.size());
        for (std::int64_t value : view_.routing.public_query_inputs) {
            public_query_inputs_.push_back(bound_input_value(value));
        }

        public_logical_leaf_ids_.reserve(view_.routing.path_claims.size());
        public_path_features_.reserve(view_.routing.path_claims.size() * view_.summary.max_depth);
        public_path_thresholds_.reserve(view_.routing.path_claims.size() * view_.summary.max_depth);
        public_path_permuted_inputs_.reserve(view_.routing.path_claims.size() * view_.summary.max_depth);
        public_path_branch_bits_.reserve(view_.routing.path_claims.size() * view_.summary.max_depth);
        public_path_internal_node_indices_.reserve(view_.routing.path_claims.size() *
                                                   view_.summary.max_depth);
        for (const zkboostSharedForestPathClaim& claim : view_.routing.path_claims) {
            public_logical_leaf_ids_.push_back(bound_input_value(claim.logical_leaf_id));
        }
        for (const zkboostSharedForestPathClaim& claim : view_.routing.path_claims) {
            for (size_t step = 0; step < view_.summary.max_depth; ++step) {
                public_path_features_.push_back(bound_input_value(claim.path_features.at(step)));
            }
        }
        for (const zkboostSharedForestPathClaim& claim : view_.routing.path_claims) {
            for (size_t step = 0; step < view_.summary.max_depth; ++step) {
                public_path_thresholds_.push_back(bound_input_value(claim.path_thresholds.at(step)));
            }
        }
        for (const zkboostSharedForestPathClaim& claim : view_.routing.path_claims) {
            for (size_t step = 0; step < view_.summary.max_depth; ++step) {
                public_path_permuted_inputs_.push_back(bound_input_value(claim.permuted_inputs.at(step)));
            }
        }
        for (const zkboostSharedForestPathClaim& claim : view_.routing.path_claims) {
            for (size_t step = 0; step < view_.summary.max_depth; ++step) {
                public_path_branch_bits_.push_back(bound_input_value(claim.path_branch_bits.at(step)));
            }
        }
        for (const zkboostSharedForestPathClaim& claim : view_.routing.path_claims) {
            std::int64_t internal_node_index = 0;
            for (size_t step = 0; step < view_.summary.max_depth; ++step) {
                public_path_internal_node_indices_.push_back(bound_input_value(internal_node_index));
                internal_node_index =
                    (2 * internal_node_index) + 1 + claim.path_branch_bits.at(step);
            }
        }
    }

    void compile_public_forest_field_claims()
    {
        // The split-field vectors are public inputs by construction. Their
        // binding to `com_T` is checked in the opening-based fetch proof.
    }

    zkboostAssignedCircuitValue bound_input_value(std::int64_t value)
    {
        if (public_input_mode_ == zkboostSharedValidateForestPublicInputMode::PreprocessedConstants) {
            return constant(value);
        }
        return public_value(value);
    }

    void compile_path_claim(const zkboostSharedForestPathClaim& claim)
    {
        require_condition(claim.tree < view_.summary.tree_count, "path tree out of range");
        require_condition(claim.sample < view_.summary.sample_count, "path sample out of range");
        const size_t query = claim.tree * view_.summary.sample_count + claim.sample;
        require_condition(query < public_trace_leaf_indices_.size(), "path trace leaf query out of range");
        require_condition(query < public_tree_scores_.size(), "path tree score out of range");
        require_condition((query + 1) * view_.summary.feature_count <= public_query_inputs_.size(),
                          "path query inputs out of range");

        require_condition(claim.path_features.size() == view_.summary.max_depth,
                          "path_features length must equal max_depth");
        require_condition(claim.path_features.size() == claim.path_thresholds.size(),
                          "path_features/path_thresholds length mismatch");
        require_condition(claim.path_features.size() == claim.permuted_inputs.size(),
                          "path_features/permuted_inputs length mismatch");
        require_condition(claim.path_features.size() == claim.path_branch_bits.size(),
                          "path_features/path_branch_bits length mismatch");
        constrain_equal(public_trace_leaf_indices_.at(query), constant(claim.trace_leaf_index));
        const zkboostAssignedCircuitValue logical_leaf_id = public_logical_leaf_ids_.at(query);

        zkboostAssignedCircuitValue leaf_from_bits = constant(checked_i64(size_t(1), "leaf equation base"));
        zkboostAssignedCircuitValue current_internal_node = zero_;
        for (size_t step = 0; step < claim.path_features.size(); ++step) {
            const size_t path_offset = query * view_.summary.max_depth + step;
            const zkboostAssignedCircuitValue feature = public_path_features_.at(path_offset);
            const zkboostAssignedCircuitValue threshold = public_path_thresholds_.at(path_offset);
            const zkboostAssignedCircuitValue permuted_input = public_path_permuted_inputs_.at(path_offset);
            const zkboostAssignedCircuitValue branch = public_path_branch_bits_.at(path_offset);
            constrain_equal(public_path_internal_node_indices_.at(path_offset),
                            current_internal_node);

            add_lookup(feature,
                       permuted_input,
                       constant(query_feature_lookup_tag(query)));

            // Every comparison-table output is either zero or one, so lookup
            // membership already enforces branch Booleanity.
            add_lookup(threshold, permuted_input, branch);
            leaf_from_bits = add_scaled(leaf_from_bits,
                                        Fr(1),
                                        branch,
                                        shared_int64_to_fr(std::int64_t(1)
                                                           << (view_.summary.max_depth - 1 - step)),
                                        Fr(0));
            current_internal_node = add_scaled(current_internal_node, Fr(2), branch, Fr(1), Fr(1));
        }
        constrain_equal(leaf_from_bits, logical_leaf_id);
        constrain_equal(public_tree_scores_.at(query), constant(claim.tree_score));
    }

    const zkboostSharedValidateForestView& view_;
    size_t bin_count_ = 0;
    std::vector<zkboostAssignedCircuitValue> public_trace_leaf_indices_;
    std::vector<zkboostAssignedCircuitValue> public_tree_scores_;
    std::vector<zkboostAssignedCircuitValue> public_split_masks_;
    std::vector<zkboostAssignedCircuitValue> public_split_features_;
    std::vector<zkboostAssignedCircuitValue> public_split_thresholds_;
    std::vector<zkboostAssignedCircuitValue> public_query_inputs_;
    std::vector<zkboostAssignedCircuitValue> public_logical_leaf_ids_;
    std::vector<zkboostAssignedCircuitValue> public_path_features_;
    std::vector<zkboostAssignedCircuitValue> public_path_thresholds_;
    std::vector<zkboostAssignedCircuitValue> public_path_permuted_inputs_;
    std::vector<zkboostAssignedCircuitValue> public_path_branch_bits_;
    std::vector<zkboostAssignedCircuitValue> public_path_internal_node_indices_;
    zkboostAssignedCircuitValue zero_;
    zkboostAssignedCircuitValue one_;
    zkboostSharedValidateForestPublicInputMode public_input_mode_;
};

void absorb_int64_vector(ProofTranscript& transcript,
                         const char* label,
                         const std::vector<std::int64_t>& values)
{
    transcript.absorb_size(std::string(label) + ".size", values.size());
    for (std::int64_t value : values) {
        transcript.absorb_string(label, std::to_string(value));
    }
}

std::string hex_digest(const std::array<uint8_t, 32>& digest)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint8_t byte : digest) {
        out << std::setw(2) << static_cast<unsigned>(byte);
    }
    return out.str();
}

} // namespace

zkboostSharedValidateForestCommittedTable
build_zkboost_shared_validate_forest_committed_table_metadata(
    size_t bin_count,
    size_t total_lookup_row_count,
    const plonkVerificationKey& verification_key)
{
    zkboostSharedValidateForestCommittedTable out;
    out.comparison_row_count = bin_count * bin_count;
    out.total_lookup_row_count = total_lookup_row_count;
    out.table_domain_size = verification_key.n;
    out.left_commitment = verification_key.table_1_commitment;
    out.right_commitment = verification_key.table_2_commitment;
    out.output_commitment = verification_key.table_3_commitment;
    return out;
}

std::vector<Fr> encode_zkboost_shared_validate_forest_public_inputs(
    const zkboostSharedValidateForestView& view)
{
    std::vector<Fr> out;
    out.reserve(view.routing.trace_leaf_indices.size() +
                view.routing.tree_scores.size() +
                view.fetch.public_split_masks.size() +
                view.fetch.public_split_features.size() +
                view.fetch.public_split_thresholds.size() +
                view.routing.public_query_inputs.size() +
                view.routing.path_claims.size() +
                5 * view.routing.path_claims.size() * view.summary.max_depth);
    for (std::int64_t value : view.routing.trace_leaf_indices) {
        out.push_back(shared_int64_to_fr(value));
    }
    for (std::int64_t value : view.routing.tree_scores) {
        out.push_back(shared_int64_to_fr(value));
    }
    for (std::int64_t value : view.fetch.public_split_masks) {
        out.push_back(shared_int64_to_fr(value));
    }
    for (std::int64_t value : view.fetch.public_split_features) {
        out.push_back(shared_int64_to_fr(value));
    }
    for (std::int64_t value : view.fetch.public_split_thresholds) {
        out.push_back(shared_int64_to_fr(value));
    }
    for (std::int64_t value : view.routing.public_query_inputs) {
        out.push_back(shared_int64_to_fr(value));
    }
    for (const zkboostSharedForestPathClaim& claim : view.routing.path_claims) {
        out.push_back(shared_int64_to_fr(claim.logical_leaf_id));
    }
    for (const zkboostSharedForestPathClaim& claim : view.routing.path_claims) {
        for (std::int64_t value : claim.path_features) {
            out.push_back(shared_int64_to_fr(value));
        }
    }
    for (const zkboostSharedForestPathClaim& claim : view.routing.path_claims) {
        for (std::int64_t value : claim.path_thresholds) {
            out.push_back(shared_int64_to_fr(value));
        }
    }
    for (const zkboostSharedForestPathClaim& claim : view.routing.path_claims) {
        for (std::int64_t value : claim.permuted_inputs) {
            out.push_back(shared_int64_to_fr(value));
        }
    }
    for (const zkboostSharedForestPathClaim& claim : view.routing.path_claims) {
        for (std::int64_t value : claim.path_branch_bits) {
            out.push_back(shared_int64_to_fr(value));
        }
    }
    for (const zkboostSharedForestPathClaim& claim : view.routing.path_claims) {
        std::int64_t internal_node_index = 0;
        for (std::int64_t branch : claim.path_branch_bits) {
            out.push_back(shared_int64_to_fr(internal_node_index));
            internal_node_index = (2 * internal_node_index) + 1 + branch;
        }
    }
    return out;
}

std::string digest_zkboost_shared_validate_forest_public_inputs(
    const zkboostSharedValidateForestView& view)
{
    ProofTranscript transcript("CertXGB shared ValidateForest link");
    transcript.absorb_size("summary.tree_count", view.summary.tree_count);
    transcript.absorb_size("summary.node_count", view.summary.node_count);
    transcript.absorb_size("summary.feature_count", view.summary.feature_count);
    transcript.absorb_size("summary.max_depth", view.summary.max_depth);
    transcript.absorb_size("summary.checked_paths", view.summary.checked_paths);
    absorb_int64_vector(transcript, "trace_leaf_indices", view.routing.trace_leaf_indices);
    absorb_int64_vector(transcript, "tree_scores", view.routing.tree_scores);
    absorb_int64_vector(transcript, "public_split_masks", view.fetch.public_split_masks);
    absorb_int64_vector(transcript, "public_split_features", view.fetch.public_split_features);
    absorb_int64_vector(transcript, "public_split_thresholds", view.fetch.public_split_thresholds);
    absorb_int64_vector(transcript, "public_query_inputs", view.routing.public_query_inputs);
    for (const zkboostSharedForestPathClaim& claim : view.routing.path_claims) {
        transcript.absorb_size("path.tree", claim.tree);
        transcript.absorb_size("path.sample", claim.sample);
        transcript.absorb_string("path.logical_leaf_id", std::to_string(claim.logical_leaf_id));
        absorb_int64_vector(transcript, "path.features", claim.path_features);
        absorb_int64_vector(transcript, "path.thresholds", claim.path_thresholds);
        absorb_int64_vector(transcript, "path.permuted_inputs", claim.permuted_inputs);
        absorb_int64_vector(transcript, "path.branch_bits", claim.path_branch_bits);
        transcript.absorb_string("path.tree_score", std::to_string(claim.tree_score));
    }
    return hex_digest(transcript.squeeze_hash("validate_forest.link.digest"));
}

zkboostGenericCircuit build_zkboost_shared_validate_forest_generic_circuit_from_view(
    const zkboostSharedValidateForestView& view,
    size_t bin_count,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    sharedValidateForestCircuitCompiler compiler(view, bin_count, public_input_mode);
    compiler.compile();
    return compiler.build_circuit();
}

zkboostSharedValidateForestPlonkInstance build_zkboost_shared_validate_forest_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);

    return build_zkboost_shared_validate_forest_plonk_instance_from_view(
        view,
        training.bin_count,
        public_parameters,
        public_input_mode);
}

zkboostSharedValidateForestPlonkInstance build_zkboost_shared_validate_forest_plonk_instance_from_view(
    const zkboostSharedValidateForestView& view,
    size_t bin_count,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    const zkboostGenericCircuit circuit =
        build_zkboost_shared_validate_forest_generic_circuit_from_view(
            view, bin_count, public_input_mode);
    const zkboostCompiledCircuitStatement statement =
        compile_zkboost_generic_circuit_statement(
            "shared_validate_forest", circuit, public_parameters);

    zkboostSharedValidateForestPlonkInstance out;
    out.setup = statement.setup;
    out.witness = statement.witness;
    out.public_inputs = statement.public_inputs;
    out.linked_input_digest = digest_zkboost_shared_validate_forest_public_inputs(view);
    out.public_input_mode = public_input_mode;
    out.summary = view.summary;
    out.committed_table =
        build_zkboost_shared_validate_forest_committed_table_metadata(
            bin_count,
            circuit.lookup_table_rows.size(),
            statement.setup.verification_key);
    return out;
}

size_t required_zkboost_shared_validate_forest_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    const zkboostSharedValidateForestView view =
        build_zkboost_shared_validate_forest_view(training, inference);
    return required_zkboost_shared_validate_forest_plonk_srs_size_from_view(
        view,
        training.bin_count,
        public_input_mode);
}

size_t required_zkboost_shared_validate_forest_plonk_srs_size_from_view(
    const zkboostSharedValidateForestView& view,
    size_t bin_count,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    return required_zkboost_generic_circuit_plonk_srs_size(
        build_zkboost_shared_validate_forest_generic_circuit_from_view(
            view, bin_count, public_input_mode));
}

bool verify_zkboost_shared_validate_forest_committed_table(
    const plonkVerificationKey& verification_key,
    const zkboostSharedValidateForestCommittedTable& committed_table)
{
    if (committed_table.comparison_row_count > committed_table.total_lookup_row_count) {
        return false;
    }
    if (committed_table.total_lookup_row_count > committed_table.table_domain_size) {
        return false;
    }
    if (committed_table.table_domain_size != verification_key.n) {
        return false;
    }
    return committed_table.left_commitment == verification_key.table_1_commitment &&
           committed_table.right_commitment == verification_key.table_2_commitment &&
           committed_table.output_commitment == verification_key.table_3_commitment;
}

plonkProof prove_zkboost_shared_validate_forest_plonk(
    const zkboostSharedValidateForestPlonkInstance& instance)
{
    return prove_plonk(instance.setup.proving_key, instance.witness, instance.public_inputs);
}

bool verify_zkboost_shared_validate_forest_plonk(
    const zkboostSharedValidateForestPlonkInstance& instance,
    const plonkProof& proof)
{
    if (!verify_zkboost_shared_validate_forest_committed_table(
            instance.setup.verification_key,
            instance.committed_table)) {
        return false;
    }
    return verify_zkboost_shared_validate_forest_plonk(
        instance.setup.verification_key,
        instance.public_inputs,
        proof);
}

bool verify_zkboost_shared_validate_forest_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof)
{
    return verify_plonk(verification_key, public_inputs, proof);
}
