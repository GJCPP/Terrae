#include "zkboost/shared_trace/validate_histograms_plonk.hpp"

#include "algebra_utils.hpp"
#include "zkboost/compiler/generic_circuit.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_histogram_plonk: " + message);
    }
}

std::int64_t checked_i64_from_fr(const Fr& value, const char* context)
{
    return canonical_signed_int64_from_fr(value, context);
}

size_t checked_size(std::int64_t value, const std::string& name)
{
    require_condition(value >= 0, name + " must be non-negative");
    return static_cast<size_t>(value);
}

void validate_view_shapes(const forestTrainingSemanticTrace& training,
                          const zkboostSharedHistogramView& view)
{
    require_condition(view.summary.tree_count == training.tree_count, "tree_count mismatch");
    require_condition(view.summary.node_count == training.node_count, "node_count mismatch");
    require_condition(view.summary.feature_count == training.feature_count, "feature_count mismatch");
    require_condition(view.summary.bin_count == training.bin_count, "bin_count mismatch");
    require_condition(view.summary.sample_count == training.query_count, "sample_count mismatch");
    require_condition(view.summary.query_row_count == training.tree_count * training.query_count,
                      "query_row_count mismatch");
    require_condition(view.summary.histogram_width == training.feature_count * training.bin_count,
                      "histogram_width mismatch");
    require_condition(view.summary.histogram_size == training.node_count * view.summary.histogram_width,
                      "histogram_size mismatch");
    require_condition(view.summary.channel_count == 3, "channel_count must be three");
    require_condition(training.leaf_indices.size() == view.summary.query_row_count,
                      "leaf_indices length mismatch");
    require_condition(training.training_inputs.size() == training.query_count * training.feature_count,
                      "training_inputs length mismatch");
    require_condition(training.split_node_mask.size() == training.node_count,
                      "split_node_mask length mismatch");
    require_condition(training.left_child_indices.size() == training.node_count,
                      "left_child_indices length mismatch");
    require_condition(training.right_child_indices.size() == training.node_count,
                      "right_child_indices length mismatch");
    require_condition(view.query_counts.size() == view.summary.query_row_count,
                      "query_counts length mismatch");
    require_condition(view.query_pseudo_residuals.size() == view.summary.query_row_count,
                      "query_pseudo_residuals length mismatch");
    require_condition(view.query_diagonal_hessians.size() == view.summary.query_row_count,
                      "query_diagonal_hessians length mismatch");
    require_condition(view.lifted_leaf_indices.size() ==
                          view.summary.query_row_count * training.feature_count,
                      "lifted_leaf_indices length mismatch");
    require_condition(view.flattened_bucket_indices.size() ==
                          view.summary.query_row_count * training.feature_count,
                      "flattened_bucket_indices length mismatch");
    require_condition(view.leaf_histograms.size() == 3, "leaf_histograms channel mismatch");
    require_condition(view.full_histograms.size() == 3, "full_histograms channel mismatch");
    require_condition(view.left_statistics.size() == 3, "left_statistics channel mismatch");
    require_condition(view.right_statistics.size() == 3, "right_statistics channel mismatch");
    for (size_t channel = 0; channel < 3; ++channel) {
        require_condition(view.leaf_histograms.at(channel).size() == view.summary.histogram_size,
                          "leaf histogram length mismatch");
        require_condition(view.full_histograms.at(channel).size() == view.summary.histogram_size,
                          "full histogram length mismatch");
        require_condition(view.left_statistics.at(channel).size() == view.summary.histogram_size,
                          "left statistics length mismatch");
        require_condition(view.right_statistics.at(channel).size() == view.summary.histogram_size,
                          "right statistics length mismatch");
    }
}

class sharedHistogramCircuitCompiler : public zkboostGenericCircuitCompilerBase {
public:
    sharedHistogramCircuitCompiler(const forestTrainingSemanticTrace& training,
                                   const zkboostSharedHistogramView& view)
        : training_(training)
        , view_(view)
    {
        validate_view_shapes(training_, view_);
        zero_ = constant(0);
        add_input_lookup_table();
        initialize_public_inputs();
    }

    void compile()
    {
        constrain_bucket_indices();
        // Leaf histogram construction is proved by the KZG weighted-LogUp
        // component. This PLONK keeps only the tree-local propagation and
        // prefix/suffix statistics that consume the committed leaf histogram.
        compile_full_histograms();
        compile_left_right_statistics();
    }

private:

    static constexpr std::int64_t kTrainingInputLookupTag = 3200000011LL;

    void add_input_lookup_table()
    {
        // The input-bin alphabet is fixed by the public bin_count.
        for (size_t value = 0; value < training_.bin_count; ++value) {
            add_lookup_table_row(static_cast<std::int64_t>(value),
                                 kTrainingInputLookupTag,
                                 0);
        }
    }

    void initialize_public_inputs()
    {
        public_leaf_indices_.reserve(training_.leaf_indices.size());
        for (std::int64_t value : training_.leaf_indices) {
            public_leaf_indices_.push_back(public_value(value));
        }

        public_query_pseudo_residuals_.reserve(view_.query_pseudo_residuals.size());
        for (const Fr& value : view_.query_pseudo_residuals) {
            public_query_pseudo_residuals_.push_back(
                public_value(checked_i64_from_fr(value, "histogram query_pseudo_residuals")));
        }

        public_query_diagonal_hessians_.reserve(view_.query_diagonal_hessians.size());
        for (const Fr& value : view_.query_diagonal_hessians) {
            public_query_diagonal_hessians_.push_back(
                public_value(checked_i64_from_fr(value, "histogram query_diagonal_hessians")));
        }

        public_leaf_histograms_.resize(3);
        public_full_histograms_.resize(3);
        public_left_statistics_.resize(3);
        public_right_statistics_.resize(3);
        for (size_t channel = 0; channel < 3; ++channel) {
            public_leaf_histograms_.at(channel).reserve(view_.summary.histogram_size);
            public_full_histograms_.at(channel).reserve(view_.summary.histogram_size);
            public_left_statistics_.at(channel).reserve(view_.summary.histogram_size);
            public_right_statistics_.at(channel).reserve(view_.summary.histogram_size);
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            for (size_t i = 0; i < view_.summary.histogram_size; ++i) {
                public_leaf_histograms_.at(channel).push_back(
                    public_value(checked_i64_from_fr(view_.leaf_histograms.at(channel).at(i),
                                                     "histogram leaf_histograms")));
            }
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            for (size_t i = 0; i < view_.summary.histogram_size; ++i) {
                public_full_histograms_.at(channel).push_back(
                    public_value(checked_i64_from_fr(view_.full_histograms.at(channel).at(i),
                                                     "histogram full_histograms")));
            }
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            for (size_t i = 0; i < view_.summary.histogram_size; ++i) {
                public_left_statistics_.at(channel).push_back(
                    public_value(checked_i64_from_fr(view_.left_statistics.at(channel).at(i),
                                                     "histogram left_statistics")));
            }
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            for (size_t i = 0; i < view_.summary.histogram_size; ++i) {
                public_right_statistics_.at(channel).push_back(
                    public_value(checked_i64_from_fr(view_.right_statistics.at(channel).at(i),
                                                     "histogram right_statistics")));
            }
        }

        public_training_inputs_.reserve(training_.training_inputs.size());
        for (std::int64_t value : training_.training_inputs) {
            public_training_inputs_.push_back(public_value(value));
        }
        public_flattened_bucket_indices_.reserve(view_.flattened_bucket_indices.size());
        for (const Fr& value : view_.flattened_bucket_indices) {
            public_flattened_bucket_indices_.push_back(
                public_value(checked_i64_from_fr(value, "histogram flattened_bucket_indices")));
        }
    }

    void constrain_bucket_indices()
    {
        const std::int64_t max_node = static_cast<std::int64_t>(training_.node_count - 1);
        for (size_t query_row = 0; query_row < view_.summary.query_row_count; ++query_row) {
            const std::int64_t leaf = training_.leaf_indices.at(query_row);
            constrain_nonnegative_at_most_with_limbs(public_leaf_indices_.at(query_row),
                                                      leaf,
                                                      max_node);
            const size_t sample = query_row % training_.query_count;
            for (size_t feature = 0; feature < training_.feature_count; ++feature) {
                const size_t input_index = sample * training_.feature_count + feature;
                const size_t bucket_index = query_row * training_.feature_count + feature;
                add_lookup(public_training_inputs_.at(input_index),
                           constant(kTrainingInputLookupTag),
                           zero_);
                const zkboostAssignedCircuitValue leaf_base =
                    affine(public_leaf_indices_.at(query_row),
                           Fr(static_cast<std::int64_t>(view_.summary.histogram_width)),
                           Fr(static_cast<std::int64_t>(feature * training_.bin_count)));
                const zkboostAssignedCircuitValue expected =
                    add_scaled(leaf_base,
                               Fr(1),
                               public_training_inputs_.at(input_index),
                               Fr(1),
                               Fr(0));
                constrain_equal(public_flattened_bucket_indices_.at(bucket_index), expected);
            }
        }
    }

    void compile_full_histograms()
    {
        for (size_t channel = 0; channel < 3; ++channel) {
            for (size_t node = 0; node < training_.node_count; ++node) {
                const bool is_split = checked_size(training_.split_node_mask.at(node), "split_node_mask") != 0;
                for (size_t col = 0; col < view_.summary.histogram_width; ++col) {
                    const size_t index = node * view_.summary.histogram_width + col;
                    zkboostAssignedCircuitValue expected = public_leaf_histograms_.at(channel).at(index);
                    if (is_split) {
                        const size_t left = checked_size(training_.left_child_indices.at(node),
                                                         "left_child_indices");
                        const size_t right = checked_size(training_.right_child_indices.at(node),
                                                          "right_child_indices");
                        require_condition(left < training_.node_count, "left child index out of range");
                        require_condition(right < training_.node_count, "right child index out of range");
                        const size_t left_index = left * view_.summary.histogram_width + col;
                        const size_t right_index = right * view_.summary.histogram_width + col;
                        expected = add_scaled(expected,
                                              Fr(1),
                                              public_full_histograms_.at(channel).at(left_index),
                                              Fr(1),
                                              Fr(0));
                        expected = add_scaled(expected,
                                              Fr(1),
                                              public_full_histograms_.at(channel).at(right_index),
                                              Fr(1),
                                              Fr(0));
                    }
                    constrain_equal(public_full_histograms_.at(channel).at(index), expected);
                }
            }
        }
    }

    void compile_left_right_statistics()
    {
        for (size_t channel = 0; channel < 3; ++channel) {
            for (size_t node = 0; node < training_.node_count; ++node) {
                for (size_t feature = 0; feature < training_.feature_count; ++feature) {
                    const size_t total_index =
                        node * view_.summary.histogram_width +
                        feature * training_.bin_count +
                        (training_.bin_count - 1);
                    for (size_t bin = 0; bin < training_.bin_count; ++bin) {
                        const size_t index =
                            node * view_.summary.histogram_width +
                            feature * training_.bin_count +
                            bin;
                        if (bin == 0) {
                            constrain_equal(public_left_statistics_.at(channel).at(index),
                                            public_full_histograms_.at(channel).at(index));
                        } else {
                            const size_t previous = index - 1;
                            builder_.add_linear_constraint(
                                public_left_statistics_.at(channel).at(previous).variable,
                                Fr(1),
                                public_full_histograms_.at(channel).at(index).variable,
                                Fr(1),
                                public_left_statistics_.at(channel).at(index).variable,
                                Fr(-1),
                                Fr(0));
                        }

                        builder_.add_linear_constraint(
                            public_left_statistics_.at(channel).at(index).variable,
                            Fr(1),
                            public_right_statistics_.at(channel).at(index).variable,
                            Fr(1),
                            public_left_statistics_.at(channel).at(total_index).variable,
                            Fr(-1),
                            Fr(0));
                    }
                }
            }
        }
    }

    const forestTrainingSemanticTrace& training_;
    const zkboostSharedHistogramView& view_;

    zkboostAssignedCircuitValue zero_;
    std::vector<zkboostAssignedCircuitValue> public_leaf_indices_;
    std::vector<zkboostAssignedCircuitValue> public_query_pseudo_residuals_;
    std::vector<zkboostAssignedCircuitValue> public_query_diagonal_hessians_;
    std::vector<std::vector<zkboostAssignedCircuitValue>> public_leaf_histograms_;
    std::vector<std::vector<zkboostAssignedCircuitValue>> public_full_histograms_;
    std::vector<std::vector<zkboostAssignedCircuitValue>> public_left_statistics_;
    std::vector<std::vector<zkboostAssignedCircuitValue>> public_right_statistics_;
    std::vector<zkboostAssignedCircuitValue> public_training_inputs_;
    std::vector<zkboostAssignedCircuitValue> public_flattened_bucket_indices_;
};

zkboostNamedPublicSpans histogram_named_public_spans(const zkboostSharedHistogramView& view)
{
    const size_t q = view.summary.query_row_count;
    const size_t h = view.summary.histogram_size;
    size_t offset = 0;
    zkboostNamedPublicSpans spans;
    spans["leaf_indices"] = {offset, q};
    offset += q;
    spans["pseudo_residuals"] = {offset, q};
    offset += q;
    spans["diagonal_hessians"] = {offset, q};
    offset += q;
    spans["leaf_histograms"] = {offset, 3 * h};
    offset += 3 * h;
    spans["full_histograms"] = {offset, 3 * h};
    offset += 3 * h;
    spans["left_statistics"] = {offset, 3 * h};
    offset += 3 * h;
    spans["right_statistics"] = {offset, 3 * h};
    offset += 3 * h;
    spans["training_inputs"] = {offset, view.summary.sample_count * view.summary.feature_count};
    offset += view.summary.sample_count * view.summary.feature_count;
    spans["flattened_bucket_indices"] = {
        offset, view.summary.query_row_count * view.summary.feature_count};
    return spans;
}

} // namespace

std::vector<Fr> encode_zkboost_shared_histogram_public_inputs(
    const zkboostSharedHistogramView& view,
    const forestTrainingSemanticTrace& training)
{
    std::vector<Fr> out;
    out.reserve(view.summary.query_row_count * 3 + view.summary.histogram_size * 12 +
                training.training_inputs.size() + view.flattened_bucket_indices.size());
    for (std::int64_t leaf : training.leaf_indices) {
        out.push_back(fr_from_dec(std::to_string(leaf)));
    }
    out.insert(out.end(), view.query_pseudo_residuals.begin(), view.query_pseudo_residuals.end());
    out.insert(out.end(), view.query_diagonal_hessians.begin(), view.query_diagonal_hessians.end());
    for (size_t channel = 0; channel < 3; ++channel) {
        out.insert(out.end(), view.leaf_histograms.at(channel).begin(), view.leaf_histograms.at(channel).end());
    }
    for (size_t channel = 0; channel < 3; ++channel) {
        out.insert(out.end(), view.full_histograms.at(channel).begin(), view.full_histograms.at(channel).end());
    }
    for (size_t channel = 0; channel < 3; ++channel) {
        out.insert(out.end(), view.left_statistics.at(channel).begin(), view.left_statistics.at(channel).end());
    }
    for (size_t channel = 0; channel < 3; ++channel) {
        out.insert(out.end(), view.right_statistics.at(channel).begin(), view.right_statistics.at(channel).end());
    }
    for (std::int64_t input : training.training_inputs) {
        out.push_back(fr_from_dec(std::to_string(input)));
    }
    out.insert(out.end(),
               view.flattened_bucket_indices.begin(),
               view.flattened_bucket_indices.end());
    return out;
}

zkboostSharedHistogramPlonkInstance build_zkboost_shared_histogram_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const kzgPublicParameters& public_parameters)
{
    return build_zkboost_shared_histogram_plonk_instance(
        training,
        build_zkboost_shared_boosting_state_view(training),
        public_parameters);
}

zkboostSharedHistogramPlonkInstance build_zkboost_shared_histogram_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& boosting,
    const kzgPublicParameters& public_parameters)
{
    return build_zkboost_shared_histogram_plonk_instance_from_view(
        training,
        build_zkboost_shared_histogram_view(training, boosting),
        public_parameters);
}

zkboostSharedHistogramPlonkInstance build_zkboost_shared_histogram_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& view,
    const kzgPublicParameters& public_parameters)
{
    sharedHistogramCircuitCompiler compiler(training, view);
    compiler.compile();
    const zkboostCompiledCircuitStatement statement =
        compile_zkboost_generic_circuit_statement("shared_validate_histograms",
                                                  compiler.build_circuit(),
                                                  public_parameters,
                                                  histogram_named_public_spans(view));

    zkboostSharedHistogramPlonkInstance out;
    out.setup = statement.setup;
    out.witness = statement.witness;
    out.public_inputs = statement.public_inputs;
    out.summary = view.summary;
    return out;
}

size_t required_zkboost_shared_histogram_plonk_srs_size(
    const forestTrainingSemanticTrace& training)
{
    return required_zkboost_shared_histogram_plonk_srs_size(
        training,
        build_zkboost_shared_boosting_state_view(training));
}

size_t required_zkboost_shared_histogram_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedBoostingStateView& boosting)
{
    return required_zkboost_shared_histogram_plonk_srs_size_from_view(
        training,
        build_zkboost_shared_histogram_view(training, boosting));
}

size_t required_zkboost_shared_histogram_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedHistogramView& view)
{
    sharedHistogramCircuitCompiler compiler(training, view);
    compiler.compile();
    return required_zkboost_generic_circuit_plonk_srs_size(compiler.build_circuit());
}

plonkProof prove_zkboost_shared_histogram_plonk(
    const zkboostSharedHistogramPlonkInstance& instance)
{
    return prove_plonk(instance.setup.proving_key, instance.witness, instance.public_inputs);
}

bool verify_zkboost_shared_histogram_plonk(
    const zkboostSharedHistogramPlonkInstance& instance,
    const plonkProof& proof)
{
    return verify_zkboost_shared_histogram_plonk(
        instance.setup.verification_key,
        instance.public_inputs,
        proof);
}

bool verify_zkboost_shared_histogram_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof)
{
    return verify_plonk(verification_key, public_inputs, proof);
}

bool link_zkboost_shared_validate_forest_to_histograms(
    const zkboostSharedValidateForestPlonkInstance& forest_instance,
    const zkboostSharedHistogramPlonkInstance& histogram_instance)
{
    const size_t count = histogram_instance.summary.query_row_count;
    if (forest_instance.summary.checked_paths != count) {
        return false;
    }
    if (forest_instance.public_inputs.size() < count || histogram_instance.public_inputs.size() < count) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (forest_instance.public_inputs.at(i) != histogram_instance.public_inputs.at(i)) {
            return false;
        }
    }
    return true;
}

bool link_zkboost_shared_boosting_state_to_histograms(
    const zkboostSharedBoostingStatePlonkInstance& boosting_instance,
    const zkboostSharedHistogramPlonkInstance& histogram_instance)
{
    const size_t count = histogram_instance.summary.query_row_count;
    if (boosting_instance.summary.cube_size != count) {
        return false;
    }
    if (boosting_instance.public_inputs.size() < 3 * count ||
        histogram_instance.public_inputs.size() < 3 * count) {
        return false;
    }
    const size_t boosting_residual_offset = count;
    const size_t boosting_hessian_offset = 2 * count;
    const size_t histogram_residual_offset = count;
    const size_t histogram_hessian_offset = 2 * count;
    for (size_t i = 0; i < count; ++i) {
        if (boosting_instance.public_inputs.at(boosting_residual_offset + i) !=
            histogram_instance.public_inputs.at(histogram_residual_offset + i)) {
            return false;
        }
        if (boosting_instance.public_inputs.at(boosting_hessian_offset + i) !=
            histogram_instance.public_inputs.at(histogram_hessian_offset + i)) {
            return false;
        }
    }
    return true;
}
