from schema import FixedPointEncoding, SemanticInferenceQueryBatch, SemanticTraceCase, TrainingSemanticTrace


class TraceValidationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise TraceValidationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _require_int(name: str, value: int, *, minimum: int | None = None, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(f"{name} must be <= {maximum}")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        _fail(f"{name} must be a bool")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _require_power_of_two(name: str, value: int) -> None:
    _require(_is_power_of_two(value), f"{name} must be a power of two")


def _require_length(name: str, values: list[int], expected: int) -> None:
    _require(len(values) == expected, f"{name} must contain exactly {expected} entries")


def _require_non_negative_vector(name: str, values: list[int]) -> None:
    for index, value in enumerate(values):
        _require_int(f"{name}[{index}]", value, minimum=0)


def _require_binary_vector(name: str, values: list[int]) -> None:
    for index, value in enumerate(values):
        _require_int(f"{name}[{index}]", value, minimum=0, maximum=1)


def _validate_encoding(name: str, encoding: FixedPointEncoding) -> None:
    _require_bool(f"{name}.signed_values", encoding.signed_values)
    _require_int(f"{name}.total_bits", encoding.total_bits, minimum=1)
    _require_int(f"{name}.fractional_bits", encoding.fractional_bits, minimum=0)
    _require(
        encoding.fractional_bits <= encoding.total_bits,
        f"{name}.fractional_bits must be <= {name}.total_bits",
    )


def validate_training_trace(trace: TrainingSemanticTrace) -> None:
    scalar_fields = {
        "loss_mode": trace.loss_mode,
        "tree_count": trace.tree_count,
        "node_count": trace.node_count,
        "feature_count": trace.feature_count,
        "bin_count": trace.bin_count,
        "iteration_count": trace.iteration_count,
        "class_count": trace.class_count,
        "query_count": trace.query_count,
        "input_bit_bound": trace.input_bit_bound,
        "full_upper_bound": trace.full_upper_bound,
        "structure_bit_bound": trace.structure_bit_bound,
        "hessian_bit_bound": trace.hessian_bit_bound,
        "gain_denominator_bit_bound": trace.gain_denominator_bit_bound,
        "leaf_denominator_bit_bound": trace.leaf_denominator_bit_bound,
        "gain_gap_bit_bound": trace.gain_gap_bit_bound,
        "l2_regularization": trace.l2_regularization,
        "learning_rate_numerator": trace.learning_rate_numerator,
        "learning_rate_denominator": trace.learning_rate_denominator,
        "split_penalty": trace.split_penalty,
        "split_gain_threshold": trace.split_gain_threshold,
        "max_depth": trace.max_depth,
    }
    for name, value in scalar_fields.items():
        _require_int(name, value, minimum=0)
    _require(trace.loss_mode in (0, 1), "loss_mode must be 0 (first-order RSS) or 1 (second-order XGBoost)")
    if trace.loss_mode == 0:
        _require(trace.l2_regularization == 0, "first-order RSS requires l2_regularization = 0")
        _require(trace.split_penalty == 0, "first-order RSS requires split_penalty = 0")

    for name in (
        "tree_count",
        "node_count",
        "feature_count",
        "bin_count",
        "iteration_count",
        "class_count",
        "query_count",
        "max_depth",
    ):
        _require(scalar_fields[name] > 0, f"{name} must be non-zero")

    for name in ("tree_count", "node_count", "feature_count", "bin_count", "query_count"):
        _require_power_of_two(name, scalar_fields[name])

    _require(
        trace.tree_count == trace.iteration_count * trace.class_count,
        "tree_count must equal iteration_count * class_count",
    )

    _require(trace.learning_rate_numerator > 0, "learning_rate_numerator must be non-zero")
    _require(trace.learning_rate_denominator > 0, "learning_rate_denominator must be non-zero")
    _validate_encoding("gradient_encoding", trace.gradient_encoding)
    _validate_encoding("gain_numerator_encoding", trace.gain_numerator_encoding)
    _validate_encoding("quantized_gain_encoding", trace.quantized_gain_encoding)
    _validate_encoding("leaf_value_encoding", trace.leaf_value_encoding)
    _validate_encoding("score_encoding", trace.score_encoding)
    _validate_encoding("softmax_encoding", trace.softmax_encoding)

    node_feature_count = trace.node_count * trace.feature_count
    query_row_count = trace.tree_count * trace.query_count
    training_input_count = trace.query_count * trace.feature_count
    base_label_count = trace.class_count * trace.query_count
    candidate_count = trace.node_count * trace.feature_count * trace.bin_count
    candidate_column_count = trace.feature_count * trace.bin_count

    _require_length("root_indices", trace.root_indices, trace.tree_count)
    _require_length("split_node_mask", trace.split_node_mask, trace.node_count)
    _require_length("left_child_indices", trace.left_child_indices, trace.node_count)
    _require_length("right_child_indices", trace.right_child_indices, trace.node_count)
    _require_length("node_tree_ids", trace.node_tree_ids, trace.node_count)
    _require_length("feature_selector", trace.feature_selector, node_feature_count)
    _require_length("thresholds", trace.thresholds, trace.node_count)
    _require_length("node_leaf_mask", trace.node_leaf_mask, trace.node_count)
    _require_length("node_leaf_values", trace.node_leaf_values, trace.node_count)
    _require_length("training_inputs", trace.training_inputs, training_input_count)
    _require_length("base_labels", trace.base_labels, base_label_count)
    _require_length("leaf_indices", trace.leaf_indices, query_row_count)
    _require_length("candidate_mask", trace.candidate_mask, candidate_count)
    _require_length("selected_candidate_indices", trace.selected_candidate_indices, trace.node_count)

    _require_non_negative_vector("root_indices", trace.root_indices)
    _require_binary_vector("split_node_mask", trace.split_node_mask)
    _require_non_negative_vector("left_child_indices", trace.left_child_indices)
    _require_non_negative_vector("right_child_indices", trace.right_child_indices)
    _require_non_negative_vector("node_tree_ids", trace.node_tree_ids)
    _require_non_negative_vector("thresholds", trace.thresholds)
    _require_binary_vector("node_leaf_mask", trace.node_leaf_mask)
    _require_non_negative_vector("training_inputs", trace.training_inputs)
    _require_non_negative_vector("base_labels", trace.base_labels)
    _require_non_negative_vector("leaf_indices", trace.leaf_indices)
    _require_binary_vector("candidate_mask", trace.candidate_mask)
    _require_non_negative_vector("selected_candidate_indices", trace.selected_candidate_indices)

    for index, root in enumerate(trace.root_indices):
        _require(root < trace.node_count, f"root_indices[{index}] must be < node_count")
    for index, child in enumerate(trace.left_child_indices):
        _require(child < trace.node_count, f"left_child_indices[{index}] must be < node_count")
    for index, child in enumerate(trace.right_child_indices):
        _require(child < trace.node_count, f"right_child_indices[{index}] must be < node_count")
    for index, tree_id in enumerate(trace.node_tree_ids):
        is_dummy = trace.split_node_mask[index] == 0 and trace.node_leaf_mask[index] == 0
        if is_dummy:
            _require(tree_id == 0, f"dummy node_tree_ids[{index}] must be 0")
        else:
            _require(1 <= tree_id <= trace.tree_count,
                     f"real node_tree_ids[{index}] must be in [1, tree_count]")
    for index, threshold in enumerate(trace.thresholds):
        _require(
            threshold <= trace.full_upper_bound,
            f"thresholds[{index}] must be <= full_upper_bound",
        )
    for index, value in enumerate(trace.training_inputs):
        _require(value < trace.bin_count, f"training_inputs[{index}] must be < bin_count")
    for index, leaf_index in enumerate(trace.leaf_indices):
        _require(leaf_index < trace.node_count, f"leaf_indices[{index}] must be < node_count")

    for node in range(trace.node_count):
        split_mask = trace.split_node_mask[node]
        leaf_mask = trace.node_leaf_mask[node]
        _require(
            not (split_mask == 1 and leaf_mask == 1),
            f"node {node} cannot be both a split node and a leaf node",
        )

        row_offset = node * trace.feature_count
        row = trace.feature_selector[row_offset : row_offset + trace.feature_count]
        _require_binary_vector(f"feature_selector[{node}]", row)
        selected_features = sum(row)
        if split_mask == 1:
            _require(selected_features == 1, f"feature_selector row {node} must be one-hot for split nodes")
        else:
            _require(selected_features == 0, f"feature_selector row {node} must be all-zero for non-split nodes")

        selected_candidate = trace.selected_candidate_indices[node]
        if split_mask == 1:
            _require(
                selected_candidate < candidate_column_count,
                f"selected_candidate_indices[{node}] must be < feature_count * bin_count for split nodes",
            )

    for tree, root in enumerate(trace.root_indices):
        _require(
            trace.node_tree_ids[root] == tree + 1,
            f"root_indices[{tree}] must point to a node in the same tree",
        )

    for node in range(trace.node_count):
        if trace.split_node_mask[node] == 0:
            continue
        node_tree = trace.node_tree_ids[node]
        left_child = trace.left_child_indices[node]
        right_child = trace.right_child_indices[node]
        _require(
            trace.node_tree_ids[left_child] == node_tree,
            f"left_child_indices[{node}] must stay within one tree",
        )
        _require(
            trace.node_tree_ids[right_child] == node_tree,
            f"right_child_indices[{node}] must stay within one tree",
        )

    for query_row, leaf_index in enumerate(trace.leaf_indices):
        tree = query_row // trace.query_count
        _require(
            trace.node_tree_ids[leaf_index] == tree + 1,
            f"leaf_indices[{query_row}] must point to a node in tree {tree}",
        )
        _require(
            trace.node_leaf_mask[leaf_index] == 1,
            f"leaf_indices[{query_row}] must point to a leaf node",
        )

    for value in trace.base_labels:
        _require(value in (0, 1), "base_labels must be binary")
    for query in range(trace.query_count):
        column_sum = 0
        for cls in range(trace.class_count):
            column_sum += trace.base_labels[cls * trace.query_count + query]
        _require(column_sum == 1, f"base_labels column {query} must sum to 1")


def validate_inference_trace(
    queries: SemanticInferenceQueryBatch,
    *,
    tree_count: int | None = None,
    full_upper_bound: int | None = None,
) -> None:
    _require_int("query_count", queries.query_count, minimum=1)
    _require_int("feature_count", queries.feature_count, minimum=1)
    _require_power_of_two("query_count", queries.query_count)
    _require_power_of_two("feature_count", queries.feature_count)

    _require_length("query_tree_ids", queries.query_tree_ids, queries.query_count)
    _require_length("lifted_inputs", queries.lifted_inputs, queries.query_count * queries.feature_count)
    _require_non_negative_vector("query_tree_ids", queries.query_tree_ids)
    _require_non_negative_vector("lifted_inputs", queries.lifted_inputs)

    if tree_count is not None:
        _require_int("tree_count", tree_count, minimum=1)
        for index, tree_id in enumerate(queries.query_tree_ids):
            _require(1 <= tree_id <= tree_count, f"query_tree_ids[{index}] must be in [1, tree_count]")

    if full_upper_bound is not None:
        _require_int("full_upper_bound", full_upper_bound, minimum=0)
        for index, value in enumerate(queries.lifted_inputs):
            _require(
                value <= full_upper_bound,
                f"lifted_inputs[{index}] must be <= full_upper_bound",
            )


def validate_trace_case(trace_case: SemanticTraceCase) -> None:
    validate_training_trace(trace_case.training_trace)
    validate_inference_trace(
        trace_case.inference_queries,
        tree_count=trace_case.training_trace.tree_count,
        full_upper_bound=trace_case.training_trace.full_upper_bound,
    )
    _require(
        trace_case.inference_queries.feature_count == trace_case.training_trace.feature_count,
        "inference feature_count must match training feature_count",
    )
