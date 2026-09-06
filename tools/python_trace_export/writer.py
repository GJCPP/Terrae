from pathlib import Path
from typing import Iterable, TextIO

from schema import FixedPointEncoding, SemanticInferenceQueryBatch, SemanticTraceCase, TrainingSemanticTrace
from validate import validate_inference_trace, validate_trace_case, validate_training_trace


CASE_FORMAT_TAG = "forest_training_semantic_trace_case_v1"
TRAINING_FORMAT_TAG = "forest_training_semantic_trace_v1"
INFERENCE_FORMAT_TAG = "forest_inference_query_trace_v1"


def _write_scalar_field(out: TextIO, name: str, value: int) -> None:
    out.write(f"scalar {name} {value}\n")


def _write_fr_field(out: TextIO, name: str, value: int) -> None:
    out.write(f"fr {name} {value}\n")


def _write_encoding_field(out: TextIO, name: str, encoding: FixedPointEncoding) -> None:
    signed = 1 if encoding.signed_values else 0
    out.write(
        f"encoding {name} offset_binary {signed} {encoding.total_bits} {encoding.fractional_bits}\n"
    )


def _write_i64vec_field(out: TextIO, name: str, values: Iterable[int]) -> None:
    serialized = [str(value) for value in values]
    out.write(f"i64vec {name} {len(serialized)}")
    if serialized:
        out.write(" ")
        out.write(" ".join(serialized))
    out.write("\n")


def _write_training_trace(out: TextIO, trace: TrainingSemanticTrace) -> None:
    _write_scalar_field(out, "training.loss_mode", trace.loss_mode)
    _write_scalar_field(out, "training.tree_count", trace.tree_count)
    _write_scalar_field(out, "training.node_count", trace.node_count)
    _write_scalar_field(out, "training.feature_count", trace.feature_count)
    _write_scalar_field(out, "training.bin_count", trace.bin_count)
    _write_scalar_field(out, "training.iteration_count", trace.iteration_count)
    _write_scalar_field(out, "training.class_count", trace.class_count)
    _write_scalar_field(out, "training.query_count", trace.query_count)
    _write_scalar_field(out, "training.input_bit_bound", trace.input_bit_bound)
    _write_scalar_field(out, "training.full_upper_bound", trace.full_upper_bound)
    _write_scalar_field(out, "training.structure_bit_bound", trace.structure_bit_bound)
    _write_scalar_field(out, "training.hessian_bit_bound", trace.hessian_bit_bound)
    _write_scalar_field(out, "training.gain_denominator_bit_bound", trace.gain_denominator_bit_bound)
    _write_scalar_field(out, "training.leaf_denominator_bit_bound", trace.leaf_denominator_bit_bound)
    _write_scalar_field(out, "training.gain_gap_bit_bound", trace.gain_gap_bit_bound)
    _write_scalar_field(out, "training.l2_regularization", trace.l2_regularization)
    _write_scalar_field(out, "training.learning_rate_numerator", trace.learning_rate_numerator)
    _write_scalar_field(out, "training.learning_rate_denominator", trace.learning_rate_denominator)
    _write_fr_field(out, "training.split_penalty", trace.split_penalty)
    _write_fr_field(out, "training.split_gain_threshold", trace.split_gain_threshold)
    _write_scalar_field(out, "training.max_depth", trace.max_depth)

    _write_encoding_field(out, "training.gradient_encoding", trace.gradient_encoding)
    _write_encoding_field(out, "training.gain_numerator_encoding", trace.gain_numerator_encoding)
    _write_encoding_field(out, "training.quantized_gain_encoding", trace.quantized_gain_encoding)
    _write_encoding_field(out, "training.leaf_value_encoding", trace.leaf_value_encoding)
    _write_encoding_field(out, "training.score_encoding", trace.score_encoding)
    _write_encoding_field(out, "training.softmax_encoding", trace.softmax_encoding)

    _write_i64vec_field(out, "training.root_indices", trace.root_indices)
    _write_i64vec_field(out, "training.split_node_mask", trace.split_node_mask)
    _write_i64vec_field(out, "training.left_child_indices", trace.left_child_indices)
    _write_i64vec_field(out, "training.right_child_indices", trace.right_child_indices)
    _write_i64vec_field(out, "training.node_tree_ids", trace.node_tree_ids)
    _write_i64vec_field(out, "training.feature_selector", trace.feature_selector)
    _write_i64vec_field(out, "training.thresholds", trace.thresholds)
    _write_i64vec_field(out, "training.node_leaf_mask", trace.node_leaf_mask)
    _write_i64vec_field(out, "training.node_leaf_values", trace.node_leaf_values)
    _write_i64vec_field(out, "training.training_inputs", trace.training_inputs)
    _write_i64vec_field(out, "training.base_labels", trace.base_labels)
    _write_i64vec_field(out, "training.leaf_indices", trace.leaf_indices)
    _write_i64vec_field(out, "training.candidate_mask", trace.candidate_mask)
    _write_i64vec_field(out, "training.selected_candidate_indices", trace.selected_candidate_indices)


def _write_inference_queries(out: TextIO, queries: SemanticInferenceQueryBatch) -> None:
    _write_scalar_field(out, "inference.query_count", queries.query_count)
    _write_scalar_field(out, "inference.feature_count", queries.feature_count)
    _write_i64vec_field(out, "inference.query_tree_ids", queries.query_tree_ids)
    _write_i64vec_field(out, "inference.lifted_inputs", queries.lifted_inputs)


def write_training_trace(path: str, trace: TrainingSemanticTrace) -> None:
    validate_training_trace(trace)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as out:
        out.write(f"format {TRAINING_FORMAT_TAG}\n")
        _write_training_trace(out, trace)


def write_inference_trace(path: str, queries: SemanticInferenceQueryBatch) -> None:
    validate_inference_trace(queries)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as out:
        out.write(f"format {INFERENCE_FORMAT_TAG}\n")
        _write_inference_queries(out, queries)


def write_semantic_trace_case(path: str, trace_case: SemanticTraceCase) -> None:
    validate_trace_case(trace_case)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as out:
        out.write(f"format {CASE_FORMAT_TAG}\n")
        _write_training_trace(out, trace_case.training_trace)
        _write_inference_queries(out, trace_case.inference_queries)
