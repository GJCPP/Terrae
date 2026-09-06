from dataclasses import dataclass
from extractor import (
    TrainerForest,
    TrainerInferenceBatch,
    TrainerNode,
    TrainerTrainingData,
    TrainerTrainingSemantics,
    TrainingTraceParameters,
    extract_inference_trace,
    extract_training_trace,
)
from schema import FixedPointEncoding, SemanticInferenceQueryBatch, TrainingSemanticTrace


GRADIENT_MODE_SECOND_ORDER_XGBOOST = "second-order-xgboost"
GRADIENT_MODE_FIRST_ORDER_RSS = "first-order-rss"
GRADIENT_MODES = (
    GRADIENT_MODE_FIRST_ORDER_RSS,
    GRADIENT_MODE_SECOND_ORDER_XGBOOST,
)
LOSS_MODE_FIRST_ORDER_RSS = 0
LOSS_MODE_SECOND_ORDER_XGBOOST = 1


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _next_power_of_two(value: int) -> int:
    _require(value > 0, "value must be positive")
    if _is_power_of_two(value):
        return value
    return 1 << (value - 1).bit_length()


def _nonnegative_bit_bound(max_value: int) -> int:
    _require(max_value >= 0, "bit bound input must be non-negative")
    if max_value == 0:
        return 1
    return max_value.bit_length()


def _signed_total_bits(max_abs_value: int, fractional_bits: int) -> int:
    _require(max_abs_value >= 0, "signed bit width input must be non-negative")
    integer_bits = max_abs_value.bit_length() + 1
    return max(2, integer_bits + fractional_bits)


def _first_order_rss_gain_fraction(
    left_gradient: int,
    right_gradient: int,
    left_count: int,
    right_count: int,
    softmax_scale: int,
) -> tuple[int, int]:
    """Return the exact RSS gain after cancelling the common Hessian scale."""
    _require(left_count > 0 and right_count > 0, "RSS split counts must be positive")
    _require(softmax_scale > 0, "softmax scale must be positive")
    parent_gradient = left_gradient + right_gradient
    parent_count = left_count + right_count
    numerator = (
        (left_gradient * left_gradient) * right_count * parent_count
        + (right_gradient * right_gradient) * left_count * parent_count
        - (parent_gradient * parent_gradient) * left_count * right_count
    )
    denominator = softmax_scale * left_count * right_count * parent_count
    return numerator, denominator


def _quantized_exp_output(scaled_input: int, fractional_bits: int) -> int:
    # Keep the Python trainer aligned with the C++ proof-side quantized exp rule.
    import math

    scale = 1 << fractional_bits
    return int(math.floor(math.exp(scaled_input / scale) * scale))


def _candidate_column(feature: int, threshold_bin: int, bin_count: int) -> int:
    return feature * bin_count + threshold_bin


@dataclass(frozen=True)
class FixedPointTrainerConfig:
    feature_count: int
    bin_count: int
    full_upper_bound: int
    input_bit_bound: int
    structure_bit_bound: int
    iteration_count: int = 1
    class_count: int = 1
    max_depth: int = 1
    learning_rate_numerator: int = 4
    learning_rate_denominator: int = 1
    l2_regularization: int = 1
    split_penalty: int = 0
    split_gain_threshold: int = 1
    gain_fractional_bits: int = 2
    leaf_fractional_bits: int = 0
    score_fractional_bits: int = 0
    gradient_fractional_bits: int = 2
    hessian_fractional_bits: int = 4
    gradient_mode: str = GRADIENT_MODE_FIRST_ORDER_RSS


@dataclass(frozen=True)
class FixedPointTrainerModel:
    params: TrainingTraceParameters
    forest: TrainerForest
    max_depth: int
    nodes_per_tree: int
    real_class_count: int


@dataclass(frozen=True)
class FixedPointTrainingRun:
    model: FixedPointTrainerModel
    training_data: TrainerTrainingData
    semantics: TrainerTrainingSemantics
    training_trace: TrainingSemanticTrace | None
    predicted_labels: list[int]
    accuracy_numerator: int
    accuracy_denominator: int


@dataclass(frozen=True)
class FixedPointInferenceResult:
    inference_batch: TrainerInferenceBatch
    inference_trace: SemanticInferenceQueryBatch | None
    leaf_indices: list[int]
    tree_scores: list[int]
    predicted_labels: list[int]
    accuracy_numerator: int
    accuracy_denominator: int


class MultiClassHistogramFixedPointTrainer:
    def __init__(self, config: FixedPointTrainerConfig):
        _require(config.feature_count > 0 and _is_power_of_two(config.feature_count), "feature_count must be a power of two")
        _require(config.bin_count > 1 and _is_power_of_two(config.bin_count), "bin_count must be a power of two greater than one")
        _require(config.full_upper_bound >= config.bin_count, "full_upper_bound must cover all feature bins")
        _require(config.input_bit_bound > 0, "input_bit_bound must be positive")
        _require(config.structure_bit_bound > 0, "structure_bit_bound must be positive")
        _require(config.iteration_count > 0 and _is_power_of_two(config.iteration_count), "iteration_count must be a power of two")
        _require(config.class_count > 0 and _is_power_of_two(config.class_count), "class_count must be a power of two")
        _require(config.max_depth > 0, "max_depth must be positive because the root has depth one")
        _require(config.split_gain_threshold >= 0, "split_gain_threshold must be non-negative")
        _require(config.learning_rate_numerator > 0, "learning_rate_numerator must be positive")
        _require(config.learning_rate_denominator > 0, "learning_rate_denominator must be positive")
        _require(config.l2_regularization > 0, "l2_regularization must be positive")
        _require(config.gain_fractional_bits >= 0, "gain_fractional_bits must be non-negative")
        _require(config.leaf_fractional_bits >= 0, "leaf_fractional_bits must be non-negative")
        _require(config.score_fractional_bits >= 0, "score_fractional_bits must be non-negative")
        _require(config.gradient_fractional_bits >= 0, "gradient_fractional_bits must be non-negative")
        _require(config.hessian_fractional_bits >= 0, "hessian_fractional_bits must be non-negative")
        _require(
            config.gradient_mode in GRADIENT_MODES,
            f"gradient_mode must be one of: {', '.join(GRADIENT_MODES)}",
        )
        self.config = config

    def _nodes_per_tree(self) -> int:
        return 1 << self.config.max_depth

    def _real_nodes_per_tree(self) -> int:
        return self._nodes_per_tree() - 1

    @staticmethod
    def _child_indices(local_index: int) -> tuple[int, int]:
        return 2 * local_index + 1, 2 * local_index + 2

    def fit(
        self,
        training_inputs: list[int],
        labels: list[int],
        *,
        build_trace: bool = True,
    ) -> FixedPointTrainingRun:
        query_count = len(labels)
        _require(query_count > 0 and _is_power_of_two(query_count), "query_count must be a non-zero power of two")
        _require(
            len(training_inputs) == query_count * self.config.feature_count,
            "training_inputs must contain query_count * feature_count entries",
        )
        for index, value in enumerate(training_inputs):
            _require(0 <= value < self.config.bin_count, f"training_inputs[{index}] must be in [0, bin_count)")

        real_class_count = max(labels) + 1 if labels else 0
        _require(real_class_count > 0, "labels must be non-empty")
        _require(real_class_count <= self.config.class_count, "labels exceed configured class_count")
        for index, label in enumerate(labels):
            _require(0 <= label < real_class_count, f"labels[{index}] out of range")

        trace_class_count = self.config.class_count
        tree_count = self.config.iteration_count * trace_class_count
        _require(_is_power_of_two(tree_count), "tree_count must be a power of two")
        nodes_per_tree = self._nodes_per_tree()
        real_nodes_per_tree = self._real_nodes_per_tree()
        node_count = tree_count * nodes_per_tree

        base_labels = [
            1 if labels[query] == cls else 0
            for cls in range(trace_class_count)
            for query in range(query_count)
        ]

        candidate_column_count = self.config.feature_count * self.config.bin_count
        forest_nodes: list[TrainerNode] = []
        root_indices: list[int] = []
        leaf_indices: list[int] = []
        candidate_mask = [0] * (node_count * candidate_column_count)
        selected_candidate_indices = [0] * node_count

        cumulative_scores = [[0] * query_count for _ in range(trace_class_count)]

        max_abs_leaf_value = 0
        max_abs_quantized_gain = 0
        max_abs_gain_numerator = 0
        max_abs_gradient = 0
        max_abs_score_channel = 0
        max_probability_scaled = 0
        max_hessian_scaled = 0
        max_aggregate_hessian_scaled = 0
        softmax_scale = 1 << self.config.gradient_fractional_bits
        first_order_rss = self.config.gradient_mode == GRADIENT_MODE_FIRST_ORDER_RSS
        lambda_scaled = 0 if first_order_rss else self.config.l2_regularization * softmax_scale
        split_penalty_scaled = 0 if first_order_rss else self.config.split_penalty * softmax_scale
        max_possible_aggregate_gradient_scaled = query_count * softmax_scale
        max_possible_aggregate_hessian_scaled = query_count * softmax_scale

        for iteration in range(self.config.iteration_count):
            for query in range(query_count):
                row_max = cumulative_scores[0][query]
                for cls in range(trace_class_count):
                    row_max = max(row_max, cumulative_scores[cls][query])
                max_abs_score_channel = max(max_abs_score_channel, abs(row_max))
                for cls in range(trace_class_count):
                    score = cumulative_scores[cls][query]
                    shifted_score = score - row_max
                    max_abs_score_channel = max(max_abs_score_channel, abs(score), abs(shifted_score))

            residuals_scaled: list[list[int]] = [[0] * query_count for _ in range(trace_class_count)]
            hessians_scaled: list[list[int]] = [[0] * query_count for _ in range(trace_class_count)]
            exp_outputs = [[0] * query_count for _ in range(trace_class_count)]
            for query in range(query_count):
                row_max = cumulative_scores[0][query]
                for cls in range(trace_class_count):
                    row_max = max(row_max, cumulative_scores[cls][query])
                denominator = 0
                for cls in range(trace_class_count):
                    shifted_score = cumulative_scores[cls][query] - row_max
                    exp_value = _quantized_exp_output(shifted_score, self.config.score_fractional_bits)
                    exp_outputs[cls][query] = exp_value
                    denominator += exp_value
                _require(denominator > 0, "softmax denominator must be positive")
                for cls in range(trace_class_count):
                    probability_scaled = (exp_outputs[cls][query] * softmax_scale) // denominator
                    label_scaled = (1 if labels[query] == cls else 0) * softmax_scale
                    if self.config.gradient_mode == GRADIENT_MODE_FIRST_ORDER_RSS:
                        # Paper Equations (2.3)--(2.4): fit the negative gradient Y - P.
                        residual_scaled = label_scaled - probability_scaled
                        # Reuse the third histogram channel as a count surrogate:
                        # one fixed-point unit per sample.
                        hessian_scaled = softmax_scale
                    else:
                        # Appendix G.1: the proof-compatible XGBoost variant uses P - Y
                        # together with the diagonal Hessian.
                        residual_scaled = probability_scaled - label_scaled
                        hessian_scaled = (
                            probability_scaled * (softmax_scale - probability_scaled)
                        ) // softmax_scale
                    max_probability_scaled = max(max_probability_scaled, probability_scaled)
                    residuals_scaled[cls][query] = residual_scaled
                    hessians_scaled[cls][query] = hessian_scaled
                    max_abs_gradient = max(max_abs_gradient, abs(residual_scaled))
                    max_hessian_scaled = max(max_hessian_scaled, hessian_scaled)

            for cls in range(trace_class_count):
                tree_id = iteration * trace_class_count + cls
                node_base = tree_id * nodes_per_tree
                root_indices.append(node_base)

                visited = [False] * nodes_per_tree
                local_split_mask = [False] * nodes_per_tree
                local_leaf_mask = [False] * nodes_per_tree
                local_selected_features: list[int | None] = [None] * nodes_per_tree
                local_thresholds = [0] * nodes_per_tree
                local_leaf_values = [0] * nodes_per_tree
                local_left_children = list(range(nodes_per_tree))
                local_right_children = list(range(nodes_per_tree))

                def candidate_row_offset(local_index: int) -> int:
                    return (node_base + local_index) * candidate_column_count

                def make_leaf(local_index: int, sample_indices: list[int]) -> None:
                    nonlocal max_abs_leaf_value, max_abs_gradient, max_aggregate_hessian_scaled
                    visited[local_index] = True
                    local_leaf_mask[local_index] = True
                    leaf_gradient = sum(residuals_scaled[cls][sample] for sample in sample_indices)
                    leaf_hessian = sum(hessians_scaled[cls][sample] for sample in sample_indices)
                    max_abs_gradient = max(max_abs_gradient, abs(leaf_gradient))
                    max_aggregate_hessian_scaled = max(max_aggregate_hessian_scaled, leaf_hessian)
                    if self.config.gradient_mode == GRADIENT_MODE_FIRST_ORDER_RSS:
                        # Paper Equation (2.9), with the learning rate folded into
                        # the stored fixed-point leaf contribution.
                        local_leaf_values[local_index] = (
                            leaf_gradient
                            * (1 << self.config.leaf_fractional_bits)
                            * self.config.learning_rate_numerator
                        ) // (
                            len(sample_indices)
                            * softmax_scale
                            * self.config.learning_rate_denominator
                        )
                    else:
                        local_leaf_values[local_index] = (
                            (-leaf_gradient)
                            * (1 << self.config.leaf_fractional_bits)
                            * self.config.learning_rate_numerator
                        ) // (
                            (leaf_hessian + lambda_scaled)
                            * self.config.learning_rate_denominator
                        )
                    max_abs_leaf_value = max(max_abs_leaf_value, abs(local_leaf_values[local_index]))

                def grow_node(local_index: int, depth: int, sample_indices: list[int]) -> None:
                    nonlocal max_abs_gradient, max_abs_gain_numerator, max_abs_quantized_gain, max_aggregate_hessian_scaled
                    visited[local_index] = True
                    best_column: int | None = None
                    best_feature = 0
                    best_threshold = 0
                    best_gain = None
                    best_left_samples: list[int] = []
                    best_right_samples: list[int] = []
                    best_quantized_gain = 0
                    row_offset = candidate_row_offset(local_index)

                    for feature in range(self.config.feature_count):
                        for threshold_bin in range(0, self.config.bin_count - 1):
                            column = _candidate_column(feature, threshold_bin, self.config.bin_count)
                            left_samples = [
                                sample
                                for sample in sample_indices
                                if training_inputs[sample * self.config.feature_count + feature] <= threshold_bin
                            ]
                            right_samples = [
                                sample
                                for sample in sample_indices
                                if training_inputs[sample * self.config.feature_count + feature] > threshold_bin
                            ]
                            if not left_samples or not right_samples:
                                continue

                            candidate_mask[row_offset + column] = 1

                            left_gradient = sum(residuals_scaled[cls][sample] for sample in left_samples)
                            right_gradient = sum(residuals_scaled[cls][sample] for sample in right_samples)
                            left_hessian = sum(hessians_scaled[cls][sample] for sample in left_samples)
                            right_hessian = sum(hessians_scaled[cls][sample] for sample in right_samples)
                            parent_gradient = left_gradient + right_gradient
                            parent_hessian = left_hessian + right_hessian
                            max_abs_gradient = max(
                                max_abs_gradient,
                                abs(left_gradient),
                                abs(right_gradient),
                                abs(parent_gradient),
                            )
                            max_aggregate_hessian_scaled = max(
                                max_aggregate_hessian_scaled,
                                left_hessian,
                                right_hessian,
                                parent_hessian,
                            )

                            if self.config.gradient_mode == GRADIENT_MODE_FIRST_ORDER_RSS:
                                gain_numerator, gain_denominator = _first_order_rss_gain_fraction(
                                    left_gradient,
                                    right_gradient,
                                    len(left_samples),
                                    len(right_samples),
                                    softmax_scale,
                                )
                                quantized_gain = (
                                    gain_numerator * (1 << self.config.gain_fractional_bits)
                                ) // gain_denominator
                            else:
                                left_denominator = left_hessian + lambda_scaled
                                right_denominator = right_hessian + lambda_scaled
                                parent_denominator = parent_hessian + lambda_scaled
                                gain_denominator = left_denominator * right_denominator * parent_denominator
                                gain_numerator = (
                                    (left_gradient * left_gradient) * right_denominator * parent_denominator
                                    + (right_gradient * right_gradient) * left_denominator * parent_denominator
                                    - (parent_gradient * parent_gradient) * left_denominator * right_denominator
                                    - split_penalty_scaled * gain_denominator
                                )
                                quantized_gain = (
                                    gain_numerator * (1 << self.config.gain_fractional_bits)
                                ) // gain_denominator
                            tie_break = quantized_gain * candidate_column_count + (
                                candidate_column_count - 1 - column
                            )
                            if best_gain is None or tie_break > best_gain:
                                best_gain = tie_break
                                best_column = column
                                best_feature = feature
                                best_threshold = threshold_bin
                                best_left_samples = left_samples
                                best_right_samples = right_samples
                                best_quantized_gain = quantized_gain

                            max_abs_gain_numerator = max(max_abs_gain_numerator, abs(gain_numerator))
                            max_abs_quantized_gain = max(max_abs_quantized_gain, abs(quantized_gain))

                    if best_column is None:
                        make_leaf(local_index, sample_indices)
                        return

                    selected_candidate_indices[node_base + local_index] = best_column
                    if best_quantized_gain < self.config.split_gain_threshold:
                        make_leaf(local_index, sample_indices)
                        return
                    if depth >= self.config.max_depth:
                        make_leaf(local_index, sample_indices)
                        return

                    left_child, right_child = self._child_indices(local_index)
                    local_split_mask[local_index] = True
                    local_selected_features[local_index] = best_feature
                    local_thresholds[local_index] = best_threshold + 1
                    local_left_children[local_index] = left_child
                    local_right_children[local_index] = right_child
                    grow_node(left_child, depth + 1, best_left_samples)
                    grow_node(right_child, depth + 1, best_right_samples)

                grow_node(0, 1, list(range(query_count)))

                for sample in range(query_count):
                    local_index = 0
                    while local_index < real_nodes_per_tree and local_split_mask[local_index]:
                        selected_feature = local_selected_features[local_index]
                        _require(selected_feature is not None, "split node must have a selected feature")
                        threshold = local_thresholds[local_index]
                        feature_value = training_inputs[sample * self.config.feature_count + selected_feature]
                        if feature_value < threshold:
                            local_index = local_left_children[local_index]
                        else:
                            local_index = local_right_children[local_index]

                    leaf_indices.append(node_base + local_index)
                    cumulative_scores[cls][sample] += local_leaf_values[local_index]
                    max_abs_score_channel = max(max_abs_score_channel, abs(cumulative_scores[cls][sample]))

                for local_index in range(nodes_per_tree):
                    global_index = node_base + local_index
                    if local_index == nodes_per_tree - 1 or not visited[local_index]:
                        forest_nodes.append(
                            TrainerNode(
                                tree_id=0,
                                is_split_node=False,
                                is_leaf_node=False,
                                left_child_index=0,
                                right_child_index=0,
                                selected_feature_index=None,
                                threshold=0,
                                leaf_value=0,
                            )
                        )
                    elif local_split_mask[local_index]:
                        forest_nodes.append(
                            TrainerNode(
                                tree_id=tree_id,
                                is_split_node=True,
                                is_leaf_node=False,
                                left_child_index=node_base + local_left_children[local_index],
                                right_child_index=node_base + local_right_children[local_index],
                                selected_feature_index=local_selected_features[local_index],
                                threshold=local_thresholds[local_index],
                                leaf_value=0,
                            )
                        )
                    else:
                        forest_nodes.append(
                            TrainerNode(
                                tree_id=tree_id,
                                is_split_node=False,
                                is_leaf_node=True,
                                left_child_index=0,
                                right_child_index=0,
                                selected_feature_index=None,
                                threshold=0,
                                leaf_value=local_leaf_values[local_index],
                            )
                        )

        for query in range(query_count):
            row_max = cumulative_scores[0][query]
            for cls in range(trace_class_count):
                row_max = max(row_max, cumulative_scores[cls][query])
            max_abs_score_channel = max(max_abs_score_channel, abs(row_max))
            for cls in range(trace_class_count):
                score = cumulative_scores[cls][query]
                shifted_score = score - row_max
                max_abs_score_channel = max(max_abs_score_channel, abs(score), abs(shifted_score))

        training_data = TrainerTrainingData(
            query_count=query_count,
            training_inputs=list(training_inputs),
            base_labels=base_labels,
        )
        semantics = TrainerTrainingSemantics(
            leaf_indices=leaf_indices,
            candidate_mask=candidate_mask,
            selected_candidate_indices=selected_candidate_indices,
        )

        max_possible_gain_denominator = (
            softmax_scale * (query_count + 1) ** 3
            if first_order_rss
            else (max(max_aggregate_hessian_scaled, max_possible_aggregate_hessian_scaled) + lambda_scaled) ** 3
        )

        params = TrainingTraceParameters(
            loss_mode=(
                LOSS_MODE_FIRST_ORDER_RSS
                if first_order_rss
                else LOSS_MODE_SECOND_ORDER_XGBOOST
            ),
            tree_count=tree_count,
            feature_count=self.config.feature_count,
            bin_count=self.config.bin_count,
            iteration_count=self.config.iteration_count,
            class_count=trace_class_count,
            input_bit_bound=self.config.input_bit_bound,
            full_upper_bound=self.config.full_upper_bound,
            structure_bit_bound=self.config.structure_bit_bound,
            gradient_encoding=FixedPointEncoding(
                signed_values=True,
                total_bits=_signed_total_bits(
                    max(max_abs_gradient, max_possible_aggregate_gradient_scaled)
                    * self.config.learning_rate_numerator,
                    0,
                ),
                fractional_bits=0,
            ),
            gain_numerator_encoding=FixedPointEncoding(
                signed_values=True,
                total_bits=_signed_total_bits(max_abs_gain_numerator, 0),
                fractional_bits=0,
            ),
            quantized_gain_encoding=FixedPointEncoding(
                signed_values=True,
                total_bits=_signed_total_bits(max_abs_quantized_gain, self.config.gain_fractional_bits),
                fractional_bits=self.config.gain_fractional_bits,
            ),
            leaf_value_encoding=FixedPointEncoding(
                signed_values=True,
                total_bits=_signed_total_bits(max_abs_leaf_value, self.config.leaf_fractional_bits),
                fractional_bits=self.config.leaf_fractional_bits,
            ),
            score_encoding=FixedPointEncoding(
                signed_values=True,
                total_bits=_signed_total_bits(max_abs_score_channel, self.config.score_fractional_bits),
                fractional_bits=self.config.score_fractional_bits,
            ),
            softmax_encoding=FixedPointEncoding(
                signed_values=True,
                total_bits=_signed_total_bits(max_probability_scaled, self.config.gradient_fractional_bits),
                fractional_bits=self.config.gradient_fractional_bits,
            ),
            hessian_bit_bound=max(
                1,
                _nonnegative_bit_bound(
                    max(max_hessian_scaled, max_aggregate_hessian_scaled, max_possible_aggregate_hessian_scaled)
                ),
            ),
            gain_denominator_bit_bound=max(
                1,
                _nonnegative_bit_bound(max_possible_gain_denominator),
            ),
            leaf_denominator_bit_bound=max(
                1,
                _nonnegative_bit_bound(
                    (max(max_aggregate_hessian_scaled, max_possible_aggregate_hessian_scaled) + lambda_scaled)
                    * self.config.learning_rate_denominator
                ),
            ),
            gain_gap_bit_bound=max(1, _nonnegative_bit_bound(max_abs_quantized_gain)),
            l2_regularization=lambda_scaled,
            learning_rate_numerator=self.config.learning_rate_numerator,
            learning_rate_denominator=self.config.learning_rate_denominator,
            split_penalty=split_penalty_scaled,
            split_gain_threshold=self.config.split_gain_threshold,
            max_depth=self.config.max_depth,
        )

        forest = TrainerForest(root_indices=root_indices, nodes=forest_nodes)
        model = FixedPointTrainerModel(
            params=params,
            forest=forest,
            max_depth=self.config.max_depth,
            nodes_per_tree=nodes_per_tree,
            real_class_count=real_class_count,
        )
        training_trace = (
            extract_training_trace(params, forest, training_data, semantics)
            if build_trace
            else None
        )
        predicted_labels = self._predict_labels_from_scores(cumulative_scores, real_class_count)
        accuracy_numerator = sum(1 for actual, predicted in zip(labels, predicted_labels) if actual == predicted)
        return FixedPointTrainingRun(
            model=model,
            training_data=training_data,
            semantics=semantics,
            training_trace=training_trace,
            predicted_labels=predicted_labels,
            accuracy_numerator=accuracy_numerator,
            accuracy_denominator=len(labels),
        )

    def predict(
        self,
        model: FixedPointTrainerModel,
        inference_inputs: list[int],
        labels: list[int] | None = None,
        *,
        build_trace: bool = True,
    ) -> FixedPointInferenceResult:
        _require(
            len(inference_inputs) % self.config.feature_count == 0,
            "inference_inputs must be divisible by feature_count",
        )
        sample_count = len(inference_inputs) // self.config.feature_count
        _require(sample_count > 0 and _is_power_of_two(sample_count), "inference query_count must be a non-zero power of two")
        for index, value in enumerate(inference_inputs):
            _require(0 <= value < self.config.bin_count, f"inference_inputs[{index}] must be in [0, bin_count)")

        tree_count = model.params.tree_count
        leaf_indices: list[int] = []
        tree_scores: list[int] = []
        cumulative_scores = [[0] * sample_count for _ in range(model.params.class_count)]

        for tree_id in range(tree_count):
            cls = tree_id % model.params.class_count
            local_index = 0
            node_base = tree_id * model.nodes_per_tree
            for sample in range(sample_count):
                local_index = 0
                while True:
                    node = model.forest.nodes[node_base + local_index]
                    if not node.is_split_node:
                        break
                    _require(node.selected_feature_index is not None, "split nodes must provide selected_feature_index")
                    feature_value = inference_inputs[sample * self.config.feature_count + node.selected_feature_index]
                    if feature_value < node.threshold:
                        local_index = node.left_child_index - node_base
                    else:
                        local_index = node.right_child_index - node_base

                leaf_index = node_base + local_index
                leaf_value = model.forest.nodes[leaf_index].leaf_value
                leaf_indices.append(leaf_index)
                tree_scores.append(leaf_value)
                cumulative_scores[cls][sample] += leaf_value

        query_tree_ids = [tree_id + 1 for tree_id in range(tree_count) for _ in range(sample_count)]
        lifted_inputs: list[int] = []
        for tree_id in range(tree_count):
            (void := tree_id)
            lifted_inputs.extend(inference_inputs)

        inference_batch = TrainerInferenceBatch(
            query_count=tree_count * sample_count,
            feature_count=self.config.feature_count,
            query_tree_ids=query_tree_ids,
            lifted_inputs=lifted_inputs,
        )
        inference_trace = (
            extract_inference_trace(model.params, inference_batch)
            if build_trace
            else None
        )
        predicted_labels = self._predict_labels_from_scores(cumulative_scores, model.real_class_count)
        accuracy_numerator = 0
        accuracy_denominator = 0
        if labels is not None:
            _require(len(labels) == sample_count, "labels length must match inference sample count")
            accuracy_numerator = sum(1 for actual, predicted in zip(labels, predicted_labels) if actual == predicted)
            accuracy_denominator = len(labels)
        return FixedPointInferenceResult(
            inference_batch=inference_batch,
            inference_trace=inference_trace,
            leaf_indices=leaf_indices,
            tree_scores=tree_scores,
            predicted_labels=predicted_labels,
            accuracy_numerator=accuracy_numerator,
            accuracy_denominator=accuracy_denominator,
        )

    @staticmethod
    def _predict_labels_from_scores(cumulative_scores: list[list[int]], real_class_count: int) -> list[int]:
        sample_count = len(cumulative_scores[0]) if cumulative_scores else 0
        predictions: list[int] = []
        for sample in range(sample_count):
            best_class = 0
            best_score = cumulative_scores[0][sample]
            for cls in range(1, real_class_count):
                if cumulative_scores[cls][sample] > best_score:
                    best_class = cls
                    best_score = cumulative_scores[cls][sample]
            predictions.append(best_class)
        return predictions


class SingleClassHistogramFixedPointTrainer(MultiClassHistogramFixedPointTrainer):
    def __init__(self, config: FixedPointTrainerConfig):
        _require(config.iteration_count == 1, "reference single-class trainer requires iteration_count = 1")
        _require(config.class_count == 1, "reference single-class trainer requires class_count = 1")
        super().__init__(config)
