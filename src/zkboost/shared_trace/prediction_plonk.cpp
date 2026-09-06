#include "zkboost/shared_trace/prediction_plonk.hpp"

#include "fixed_point.hpp"
#include "zkboost/compiler/generic_circuit.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_condition(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error("zkboost_shared_prediction_plonk: " + message);
    }
}

std::int64_t checked_i64(size_t value, const std::string& context)
{
    require_condition(value <= static_cast<size_t>(std::numeric_limits<std::int64_t>::max()),
                      context + " exceeds int64 range");
    return static_cast<std::int64_t>(value);
}

std::int64_t checked_rank_max(size_t score_bit_bound, size_t class_count)
{
    require_condition(score_bit_bound > 0, "score_bit_bound must be non-zero");
    require_condition(score_bit_bound < 63, "score_bit_bound must be smaller than 63");
    require_condition(class_count > 0, "class_count must be non-zero");

    require_condition(
        class_count <= static_cast<size_t>(std::numeric_limits<std::int64_t>::max()),
        "class_count exceeds int64");
    const std::uint64_t encoded_max =
        (std::uint64_t(1) << score_bit_bound) - 1;
    const std::uint64_t class_count_u64 =
        static_cast<std::uint64_t>(class_count);
    const std::uint64_t tie_break_max = class_count_u64 - 1;
    const std::uint64_t int64_max =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    require_condition(
        encoded_max <= (int64_max - tie_break_max) / class_count_u64,
        "rank score range exceeds int64");
    return static_cast<std::int64_t>(
        encoded_max * class_count_u64 + tie_break_max);
}

size_t tree_score_index(const zkboostSharedPredictionSummary& summary,
                        size_t iteration,
                        size_t cls,
                        size_t sample)
{
    return (iteration * summary.class_count + cls) * summary.sample_count + sample;
}

size_t class_score_index(const zkboostSharedPredictionSummary& summary,
                         size_t cls,
                         size_t sample)
{
    return cls * summary.sample_count + sample;
}

Fr fr_from_i64(std::int64_t value)
{
    Fr out;
    out.setStr(std::to_string(value), 10);
    return out;
}

void validate_prediction_view(const forestTrainingSemanticTrace& training,
                              const zkboostSharedPredictionView& view)
{
    require_condition(training.iteration_count > 0, "iteration_count must be non-zero");
    require_condition(training.class_count > 0, "class_count must be non-zero");
    require_condition(training.tree_count == training.iteration_count * training.class_count,
                      "tree_count must equal iteration_count * class_count");
    require_condition(view.summary.iteration_count == training.iteration_count,
                      "iteration_count mismatch");
    require_condition(view.summary.class_count == training.class_count,
                      "class_count mismatch");
    require_condition(view.summary.sample_count > 0, "sample_count must be non-zero");
    require_condition(view.summary.tree_score_count ==
                          training.tree_count * view.summary.sample_count,
                      "tree_score_count mismatch");
    validate_fixed_point_encoding(training.score_encoding,
                                  "zkboost_shared_prediction_plonk score");
    require_condition(training.score_encoding.signed_values,
                      "score encoding must be signed");
    require_condition(view.summary.score_bit_bound == training.score_encoding.total_bits,
                      "score_bit_bound mismatch");
    require_condition(view.tree_scores.size() == view.summary.tree_score_count,
                      "tree_scores length mismatch");
    require_condition(view.class_scores.size() ==
                          training.class_count * view.summary.sample_count,
                      "class_scores length mismatch");
    require_condition(view.predicted_scores.size() == view.summary.sample_count,
                      "predicted_scores length mismatch");
    require_condition(view.class_selector.size() ==
                          training.class_count * view.summary.sample_count,
                      "class_selector length mismatch");
    require_condition(view.predicted_labels.size() == view.summary.sample_count,
                      "predicted_labels length mismatch");
}

class sharedPredictionCircuitCompiler : public zkboostGenericCircuitCompilerBase {
public:
    sharedPredictionCircuitCompiler(
        const forestTrainingSemanticTrace& training,
        const zkboostSharedPredictionView& view,
        zkboostSharedValidateForestPublicInputMode public_input_mode)
        : training_(training)
        , view_(view)
        , public_input_mode_(public_input_mode)
    {
        validate_prediction_view(training_, view_);
        zero_ = constant(0);
        one_ = constant(1);
        score_offset_ = checked_i64(
            fixed_point_signed_offset(training_.score_encoding),
            "score offset");
        rank_max_ = checked_rank_max(
            training_.score_encoding.total_bits,
            training_.class_count);
        initialize_bound_values();
    }

    void compile()
    {
        compile_score_aggregation();
        compile_argmax();
    }

private:
    zkboostAssignedCircuitValue bound_value(std::int64_t value)
    {
        if (public_input_mode_ ==
            zkboostSharedValidateForestPublicInputMode::PreprocessedConstants) {
            return constant(value);
        }
        return public_value(value);
    }

    void initialize_bound_values()
    {
        tree_scores_.reserve(view_.tree_scores.size());
        for (std::int64_t value : view_.tree_scores) {
            tree_scores_.push_back(bound_value(value));
        }

        class_scores_.reserve(view_.class_scores.size());
        for (std::int64_t value : view_.class_scores) {
            class_scores_.push_back(bound_value(value));
        }

        predicted_scores_.reserve(view_.predicted_scores.size());
        for (std::int64_t value : view_.predicted_scores) {
            predicted_scores_.push_back(bound_value(value));
        }

        class_selector_.reserve(view_.class_selector.size());
        for (std::int64_t value : view_.class_selector) {
            class_selector_.push_back(bound_value(value));
        }

        predicted_labels_.reserve(view_.predicted_labels.size());
        for (std::int64_t value : view_.predicted_labels) {
            predicted_labels_.push_back(bound_value(value));
        }
    }

    zkboostAssignedCircuitValue constrain_signed_score(
        const zkboostAssignedCircuitValue& value,
        std::int64_t witness_value)
    {
        require_condition(
            witness_value >= fixed_point_min_scaled_value(training_.score_encoding) &&
                witness_value <= fixed_point_max_scaled_value(training_.score_encoding),
            "score exceeds configured signed score range");
        const std::int64_t encoded_i64 = witness_value + score_offset_;
        const zkboostAssignedCircuitValue encoded_value =
            affine(value, Fr(1), Fr(score_offset_));
        constrain_nonnegative_with_limbs(
            encoded_value,
            encoded_i64,
            training_.score_encoding.total_bits);
        return encoded_value;
    }

    void constrain_nonnegative_at_most(
        const zkboostAssignedCircuitValue& value,
        std::int64_t witness_value,
        std::int64_t max_value)
    {
        require_condition(witness_value >= 0 && witness_value <= max_value,
                          "argmax gap is outside its configured range");
        constrain_nonnegative_at_most_with_limbs(value, witness_value, max_value);
    }

    void compile_score_aggregation()
    {
        for (size_t index = 0; index < tree_scores_.size(); ++index) {
            constrain_signed_score(
                tree_scores_.at(index),
                view_.tree_scores.at(index));
        }

        encoded_class_scores_.resize(view_.class_scores.size());
        for (size_t cls = 0; cls < view_.summary.class_count; ++cls) {
            for (size_t sample = 0; sample < view_.summary.sample_count; ++sample) {
                zkboostAssignedCircuitValue sum =
                    tree_scores_.at(tree_score_index(view_.summary, 0, cls, sample));
                for (size_t iteration = 1;
                     iteration < view_.summary.iteration_count;
                     ++iteration) {
                    sum = add_scaled(
                        sum,
                        Fr(1),
                        tree_scores_.at(
                            tree_score_index(view_.summary, iteration, cls, sample)),
                        Fr(1),
                        Fr(0));
                }
                const size_t index =
                    class_score_index(view_.summary, cls, sample);
                constrain_equal(class_scores_.at(index), sum);
                encoded_class_scores_.at(index) =
                    constrain_signed_score(
                        class_scores_.at(index),
                        view_.class_scores.at(index));
            }
        }
    }

    void compile_argmax()
    {
        const std::int64_t class_count =
            checked_i64(view_.summary.class_count, "class_count");
        for (size_t sample = 0; sample < view_.summary.sample_count; ++sample) {
            zkboostAssignedCircuitValue selector_sum = zero_;
            zkboostAssignedCircuitValue selected_score = zero_;
            zkboostAssignedCircuitValue selected_rank = zero_;
            zkboostAssignedCircuitValue selected_label = zero_;
            std::vector<zkboostAssignedCircuitValue> ranks;
            std::vector<std::int64_t> rank_witnesses;
            ranks.reserve(view_.summary.class_count);
            rank_witnesses.reserve(view_.summary.class_count);

            std::int64_t selected_rank_witness = 0;
            size_t selected_class_count = 0;
            for (size_t cls = 0; cls < view_.summary.class_count; ++cls) {
                const size_t index =
                    class_score_index(view_.summary, cls, sample);
                const zkboostAssignedCircuitValue selector =
                    class_selector_.at(index);
                constrain_boolean(selector);
                const zkboostAssignedCircuitValue selected_score_term =
                    multiply(selector, class_scores_.at(index));

                const std::int64_t tie_break =
                    checked_i64(view_.summary.class_count - 1 - cls,
                                "class tie break");
                const zkboostAssignedCircuitValue rank =
                    affine(encoded_class_scores_.at(index),
                           Fr(class_count),
                           Fr(tie_break));
                ranks.push_back(rank);
                const std::int64_t encoded_score =
                    view_.class_scores.at(index) + score_offset_;
                const std::int64_t rank_witness =
                    encoded_score * class_count + tie_break;
                require_condition(rank_witness >= 0 && rank_witness <= rank_max_,
                                  "rank witness exceeds configured range");
                rank_witnesses.push_back(rank_witness);
                const zkboostAssignedCircuitValue selected_rank_term = multiply(selector, rank);
                if (cls == 0) {
                    selector_sum = selector;
                    selected_score = selected_score_term;
                    selected_rank = selected_rank_term;
                } else {
                    selector_sum =
                        add_scaled(selector_sum, Fr(1), selector, Fr(1), Fr(0));
                    selected_score =
                        add_scaled(selected_score, Fr(1), selected_score_term, Fr(1), Fr(0));
                    selected_rank =
                        add_scaled(selected_rank, Fr(1), selected_rank_term, Fr(1), Fr(0));
                    selected_label =
                        add_scaled(selected_label,
                                   Fr(1),
                                   selector,
                                   Fr(checked_i64(cls, "class index")),
                                   Fr(0));
                }

                require_condition(
                    view_.class_selector.at(index) == 0 ||
                        view_.class_selector.at(index) == 1,
                    "class_selector must be binary");
                if (view_.class_selector.at(index) == 1) {
                    selected_rank_witness = rank_witnesses.back();
                    ++selected_class_count;
                }
            }

            require_condition(selected_class_count == 1,
                              "class_selector must select exactly one class");
            constrain_equal(selector_sum, one_);
            constrain_equal(predicted_scores_.at(sample), selected_score);
            constrain_equal(predicted_labels_.at(sample), selected_label);

            for (size_t cls = 0; cls < view_.summary.class_count; ++cls) {
                const zkboostAssignedCircuitValue gap =
                    add_scaled(selected_rank,
                               Fr(1),
                               ranks.at(cls),
                               Fr(-1),
                               Fr(0));
                constrain_nonnegative_at_most(
                    gap,
                    selected_rank_witness - rank_witnesses.at(cls),
                    rank_max_);
            }
        }
    }

    const forestTrainingSemanticTrace& training_;
    const zkboostSharedPredictionView& view_;
    zkboostSharedValidateForestPublicInputMode public_input_mode_;

    zkboostAssignedCircuitValue zero_;
    zkboostAssignedCircuitValue one_;
    std::int64_t score_offset_ = 0;
    std::int64_t rank_max_ = 0;

    std::vector<zkboostAssignedCircuitValue> tree_scores_;
    std::vector<zkboostAssignedCircuitValue> class_scores_;
    std::vector<zkboostAssignedCircuitValue> encoded_class_scores_;
    std::vector<zkboostAssignedCircuitValue> predicted_scores_;
    std::vector<zkboostAssignedCircuitValue> class_selector_;
    std::vector<zkboostAssignedCircuitValue> predicted_labels_;
};

zkboostNamedPublicSpans prediction_named_public_spans(
    const zkboostSharedPredictionView& view,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    if (public_input_mode ==
        zkboostSharedValidateForestPublicInputMode::PreprocessedConstants) {
        return {};
    }

    const size_t tree_scores = view.tree_scores.size();
    const size_t class_scores = view.class_scores.size();
    const size_t predicted_scores = view.predicted_scores.size();
    const size_t class_selector = view.class_selector.size();
    return {
        {"tree_scores", {0, tree_scores}},
        {"class_scores", {tree_scores, class_scores}},
        {"predicted_scores", {tree_scores + class_scores, predicted_scores}},
        {"class_selector",
         {tree_scores + class_scores + predicted_scores, class_selector}},
        {"predicted_labels",
         {tree_scores + class_scores + predicted_scores + class_selector,
          view.predicted_labels.size()}},
    };
}

} // namespace

zkboostSharedPredictionView build_zkboost_shared_prediction_view(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference)
{
    return build_zkboost_shared_prediction_view(
        training,
        derive_zkboost_shared_trace_prediction(training, inference));
}

zkboostSharedPredictionView build_zkboost_shared_prediction_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedTracePrediction& prediction)
{
    require_condition(training.tree_count > 0, "tree_count must be non-zero");
    require_condition(training.iteration_count > 0,
                      "iteration_count must be non-zero");
    require_condition(training.class_count > 0, "class_count must be non-zero");
    require_condition(
        training.tree_count == training.iteration_count * training.class_count,
        "tree_count must equal iteration_count * class_count");
    const size_t sample_count = prediction.predicted_labels.size();
    require_condition(sample_count > 0, "prediction sample_count must be non-zero");

    zkboostSharedPredictionView out;
    out.summary.iteration_count = training.iteration_count;
    out.summary.class_count = training.class_count;
    out.summary.sample_count = sample_count;
    out.summary.tree_score_count = training.tree_count * sample_count;
    out.summary.score_bit_bound = training.score_encoding.total_bits;
    out.tree_scores = prediction.tree_scores;
    out.class_scores = prediction.class_scores;
    out.predicted_scores = prediction.predicted_scores;
    out.class_selector = prediction.class_selector;
    out.predicted_labels = prediction.predicted_labels;
    validate_prediction_view(training, out);
    return out;
}

std::vector<Fr> encode_zkboost_shared_prediction_public_inputs(
    const zkboostSharedPredictionView& view)
{
    std::vector<Fr> out;
    out.reserve(view.tree_scores.size() +
                view.class_scores.size() +
                view.predicted_scores.size() +
                view.class_selector.size() +
                view.predicted_labels.size());
    for (std::int64_t value : view.tree_scores) {
        out.push_back(fr_from_i64(value));
    }
    for (std::int64_t value : view.class_scores) {
        out.push_back(fr_from_i64(value));
    }
    for (std::int64_t value : view.predicted_scores) {
        out.push_back(fr_from_i64(value));
    }
    for (std::int64_t value : view.class_selector) {
        out.push_back(fr_from_i64(value));
    }
    for (std::int64_t value : view.predicted_labels) {
        out.push_back(fr_from_i64(value));
    }
    return out;
}

zkboostGenericCircuit build_zkboost_shared_prediction_generic_circuit_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedPredictionView& view,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    sharedPredictionCircuitCompiler compiler(training, view, public_input_mode);
    compiler.compile();
    return compiler.build_circuit();
}

zkboostSharedPredictionPlonkInstance build_zkboost_shared_prediction_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    return build_zkboost_shared_prediction_plonk_instance_from_view(
        training,
        build_zkboost_shared_prediction_view(training, inference),
        public_parameters,
        public_input_mode);
}

zkboostSharedPredictionPlonkInstance build_zkboost_shared_prediction_plonk_instance(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedTracePrediction& prediction,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    return build_zkboost_shared_prediction_plonk_instance_from_view(
        training,
        build_zkboost_shared_prediction_view(training, prediction),
        public_parameters,
        public_input_mode);
}

zkboostSharedPredictionPlonkInstance
build_zkboost_shared_prediction_plonk_instance_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedPredictionView& view,
    const kzgPublicParameters& public_parameters,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    const zkboostGenericCircuit circuit =
        build_zkboost_shared_prediction_generic_circuit_from_view(
            training, view, public_input_mode);
    const zkboostCompiledCircuitStatement statement =
        compile_zkboost_generic_circuit_statement(
            "shared_prediction",
            circuit,
            public_parameters,
            prediction_named_public_spans(view, public_input_mode));

    zkboostSharedPredictionPlonkInstance out;
    out.setup = statement.setup;
    out.witness = statement.witness;
    out.public_inputs = statement.public_inputs;
    out.view = view;
    out.public_input_mode = public_input_mode;
    return out;
}

size_t required_zkboost_shared_prediction_plonk_srs_size(
    const forestTrainingSemanticTrace& training,
    const forestSemanticInferenceQueryBatch& inference,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    return required_zkboost_shared_prediction_plonk_srs_size_from_view(
        training,
        build_zkboost_shared_prediction_view(training, inference),
        public_input_mode);
}

size_t required_zkboost_shared_prediction_plonk_srs_size_from_view(
    const forestTrainingSemanticTrace& training,
    const zkboostSharedPredictionView& view,
    zkboostSharedValidateForestPublicInputMode public_input_mode)
{
    return required_zkboost_generic_circuit_plonk_srs_size(
        build_zkboost_shared_prediction_generic_circuit_from_view(
            training, view, public_input_mode));
}

plonkProof prove_zkboost_shared_prediction_plonk(
    const zkboostSharedPredictionPlonkInstance& instance)
{
    return prove_plonk(
        instance.setup.proving_key,
        instance.witness,
        instance.public_inputs);
}

bool verify_zkboost_shared_prediction_plonk(
    const zkboostSharedPredictionPlonkInstance& instance,
    const plonkProof& proof)
{
    return verify_zkboost_shared_prediction_plonk(
        instance.setup.verification_key,
        instance.public_inputs,
        proof);
}

bool verify_zkboost_shared_prediction_plonk(
    const plonkVerificationKey& verification_key,
    const std::vector<Fr>& public_inputs,
    const plonkProof& proof)
{
    return verify_plonk(verification_key, public_inputs, proof);
}

bool link_zkboost_shared_validate_forest_to_prediction(
    const zkboostSharedValidateForestFetchInstance& forest_fetch_instance,
    const zkboostSharedPredictionPlonkInstance& prediction_instance)
{
    const zkboostSharedPredictionSummary& prediction =
        prediction_instance.view.summary;
    if (forest_fetch_instance.summary.tree_count !=
            prediction.iteration_count * prediction.class_count ||
        forest_fetch_instance.summary.sample_count != prediction.sample_count ||
        forest_fetch_instance.summary.checked_paths !=
            prediction.tree_score_count) {
        return false;
    }
    return forest_fetch_instance.view.routing.tree_scores ==
           prediction_instance.view.tree_scores;
}
