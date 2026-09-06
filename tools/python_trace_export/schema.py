from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FixedPointEncoding:
    signed_values: bool
    total_bits: int
    fractional_bits: int


@dataclass
class TrainingSemanticTrace:
    loss_mode: int
    tree_count: int
    node_count: int
    feature_count: int
    bin_count: int
    iteration_count: int
    class_count: int
    query_count: int
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
    root_indices: List[int]
    split_node_mask: List[int]
    left_child_indices: List[int]
    right_child_indices: List[int]
    node_tree_ids: List[int]
    feature_selector: List[int]
    thresholds: List[int]
    node_leaf_mask: List[int]
    node_leaf_values: List[int]
    training_inputs: List[int]
    base_labels: List[int]
    leaf_indices: List[int]
    candidate_mask: List[int]
    selected_candidate_indices: List[int]


@dataclass
class SemanticInferenceQueryBatch:
    query_count: int
    feature_count: int
    query_tree_ids: List[int]
    lifted_inputs: List[int]


@dataclass
class SemanticTraceCase:
    training_trace: TrainingSemanticTrace
    inference_queries: SemanticInferenceQueryBatch
