from dataclasses import dataclass

from schema import FixedPointEncoding, SemanticInferenceQueryBatch, SemanticTraceCase, TrainingSemanticTrace
from validate import validate_inference_trace, validate_trace_case, validate_training_trace


@dataclass(frozen=True)
class TrainingTraceParameters:
    loss_mode: int
    tree_count: int
    feature_count: int
    bin_count: int
    iteration_count: int
    class_count: int
    input_bit_bound: int
    full_upper_bound: int
    structure_bit_bound: int
    gradient_encoding: FixedPointEncoding
    gain_numerator_encoding: FixedPointEncoding
    quantized_gain_encoding: FixedPointEncoding
    leaf_value_encoding: FixedPointEncoding
    score_encoding: FixedPointEncoding
    softmax_encoding: FixedPointEncoding
    hessian_bit_bound: int
    gain_denominator_bit_bound: int
    leaf_denominator_bit_bound: int
    gain_gap_bit_bound: int
    l2_regularization: int
    learning_rate_numerator: int
    learning_rate_denominator: int
    split_penalty: int
    split_gain_threshold: int
    max_depth: int


@dataclass(frozen=True)
class TrainerNode:
    tree_id: int
    is_split_node: bool
    is_leaf_node: bool
    left_child_index: int
    right_child_index: int
    selected_feature_index: int | None
    threshold: int
    leaf_value: int


@dataclass(frozen=True)
class TrainerForest:
    root_indices: list[int]
    nodes: list[TrainerNode]


@dataclass(frozen=True)
class TrainerTrainingData:
    query_count: int
    training_inputs: list[int]
    base_labels: list[int]


@dataclass(frozen=True)
class TrainerTrainingSemantics:
    leaf_indices: list[int]
    candidate_mask: list[int]
    selected_candidate_indices: list[int]


@dataclass(frozen=True)
class TrainerInferenceBatch:
    query_count: int
    feature_count: int
    query_tree_ids: list[int]
    lifted_inputs: list[int]


def _feature_selector_row(feature_count: int, selected_feature_index: int | None, is_split_node: bool) -> list[int]:
    if not is_split_node:
        return [0] * feature_count
    if selected_feature_index is None:
        raise ValueError("split nodes must provide selected_feature_index")
    if selected_feature_index < 0 or selected_feature_index >= feature_count:
        raise ValueError("selected_feature_index out of range")
    row = [0] * feature_count
    row[selected_feature_index] = 1
    return row


def extract_training_trace(
    params: TrainingTraceParameters,
    forest: TrainerForest,
    training_data: TrainerTrainingData,
    semantics: TrainerTrainingSemantics,
) -> TrainingSemanticTrace:
    node_count = len(forest.nodes)
    if node_count == 0:
        raise ValueError("forest.nodes must be non-empty")
    if len(forest.root_indices) != params.tree_count:
        raise ValueError("forest.root_indices size must match tree_count")
    if params.tree_count != params.iteration_count * params.class_count:
        raise ValueError("tree_count must equal iteration_count * class_count")

    feature_selector: list[int] = []
    split_node_mask: list[int] = []
    left_child_indices: list[int] = []
    right_child_indices: list[int] = []
    node_tree_ids: list[int] = []
    thresholds: list[int] = []
    node_leaf_mask: list[int] = []
    node_leaf_values: list[int] = []

    for node_index, node in enumerate(forest.nodes):
        is_dummy = not node.is_split_node and not node.is_leaf_node
        if not is_dummy and (node.tree_id < 0 or node.tree_id >= params.tree_count):
            raise ValueError(f"node {node_index}: tree_id out of range")
        if node.left_child_index < 0 or node.left_child_index >= node_count:
            raise ValueError(f"node {node_index}: left_child_index out of range")
        if node.right_child_index < 0 or node.right_child_index >= node_count:
            raise ValueError(f"node {node_index}: right_child_index out of range")

        # Tree IDs are public protocol values: 0 is reserved for an unused
        # padded node, while real trees are numbered 1..tree_count.
        node_tree_ids.append(0 if is_dummy else node.tree_id + 1)
        split_node_mask.append(1 if node.is_split_node else 0)
        left_child_indices.append(node.left_child_index)
        right_child_indices.append(node.right_child_index)
        thresholds.append(node.threshold)
        node_leaf_mask.append(1 if node.is_leaf_node else 0)
        node_leaf_values.append(node.leaf_value)
        feature_selector.extend(
            _feature_selector_row(params.feature_count, node.selected_feature_index, node.is_split_node)
        )

    trace = TrainingSemanticTrace(
        loss_mode=params.loss_mode,
        tree_count=params.tree_count,
        node_count=node_count,
        feature_count=params.feature_count,
        bin_count=params.bin_count,
        iteration_count=params.iteration_count,
        class_count=params.class_count,
        query_count=training_data.query_count,
        input_bit_bound=params.input_bit_bound,
        full_upper_bound=params.full_upper_bound,
        structure_bit_bound=params.structure_bit_bound,
        gradient_encoding=params.gradient_encoding,
        gain_numerator_encoding=params.gain_numerator_encoding,
        quantized_gain_encoding=params.quantized_gain_encoding,
        leaf_value_encoding=params.leaf_value_encoding,
        score_encoding=params.score_encoding,
        softmax_encoding=params.softmax_encoding,
        hessian_bit_bound=params.hessian_bit_bound,
        gain_denominator_bit_bound=params.gain_denominator_bit_bound,
        leaf_denominator_bit_bound=params.leaf_denominator_bit_bound,
        gain_gap_bit_bound=params.gain_gap_bit_bound,
        l2_regularization=params.l2_regularization,
        learning_rate_numerator=params.learning_rate_numerator,
        learning_rate_denominator=params.learning_rate_denominator,
        split_penalty=params.split_penalty,
        split_gain_threshold=params.split_gain_threshold,
        max_depth=params.max_depth,
        root_indices=list(forest.root_indices),
        split_node_mask=split_node_mask,
        left_child_indices=left_child_indices,
        right_child_indices=right_child_indices,
        node_tree_ids=node_tree_ids,
        feature_selector=feature_selector,
        thresholds=thresholds,
        node_leaf_mask=node_leaf_mask,
        node_leaf_values=node_leaf_values,
        training_inputs=list(training_data.training_inputs),
        base_labels=list(training_data.base_labels),
        leaf_indices=list(semantics.leaf_indices),
        candidate_mask=list(semantics.candidate_mask),
        selected_candidate_indices=list(semantics.selected_candidate_indices),
    )
    validate_training_trace(trace)
    return trace


def extract_inference_trace(
    params: TrainingTraceParameters,
    inference_batch: TrainerInferenceBatch,
) -> SemanticInferenceQueryBatch:
    if inference_batch.feature_count != params.feature_count:
        raise ValueError("inference feature_count must match training feature_count")
    trace = SemanticInferenceQueryBatch(
        query_count=inference_batch.query_count,
        feature_count=inference_batch.feature_count,
        query_tree_ids=list(inference_batch.query_tree_ids),
        lifted_inputs=list(inference_batch.lifted_inputs),
    )
    validate_inference_trace(trace, tree_count=params.tree_count, full_upper_bound=params.full_upper_bound)
    return trace


def extract_trace_case(
    params: TrainingTraceParameters,
    forest: TrainerForest,
    training_data: TrainerTrainingData,
    semantics: TrainerTrainingSemantics,
    inference_batch: TrainerInferenceBatch,
) -> SemanticTraceCase:
    trace_case = SemanticTraceCase(
        training_trace=extract_training_trace(params, forest, training_data, semantics),
        inference_queries=extract_inference_trace(params, inference_batch),
    )
    validate_trace_case(trace_case)
    return trace_case
