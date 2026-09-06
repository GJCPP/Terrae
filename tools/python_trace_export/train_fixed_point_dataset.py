#!/usr/bin/env python3

import argparse
from pathlib import Path

from datasets import load_binned_evaluation_split, load_binned_reference_dataset
from datasets import BinnedClassificationDataset
from fixed_point_trainer import (
    GRADIENT_MODES,
    GRADIENT_MODE_FIRST_ORDER_RSS,
    FixedPointTrainerConfig,
    MultiClassHistogramFixedPointTrainer,
)
from trainer_config import load_trainer_external_config
from writer import write_inference_trace, write_training_trace
from zkboost.datasets import load_binned_binary_reference_dataset


def _feature_bit_bound(full_upper_bound: int) -> int:
    return max(1, full_upper_bound.bit_length())


def _next_power_of_two(value: int) -> int:
    out = 1
    while out < value:
        out <<= 1
    return out


def _load_dataset(name: str, bin_count: int) -> BinnedClassificationDataset:
    if name in ("iris-binary", "wine-binary", "breast-cancer-binary", "optdigits-parity-train"):
        binary = load_binned_binary_reference_dataset(name, bin_count)
        return BinnedClassificationDataset(
            name=binary.name,
            feature_count=binary.feature_count,
            class_count=(max(binary.labels) + 1) if binary.labels else 0,
            query_count=binary.query_count,
            original_query_count=binary.original_query_count,
            bin_count=binary.bin_count,
            flattened_inputs=list(binary.flattened_inputs),
            labels=list(binary.labels),
        )
    return load_binned_reference_dataset(name, bin_count)


def _truncate_dataset(dataset: BinnedClassificationDataset,
                      max_original_queries: int | None) -> BinnedClassificationDataset:
    if max_original_queries is None or max_original_queries >= dataset.original_query_count:
        return dataset
    if max_original_queries <= 0:
        raise ValueError("max_original_queries must be positive")

    kept = max_original_queries
    rows = [
        dataset.flattened_inputs[query * dataset.feature_count:(query + 1) * dataset.feature_count]
        for query in range(kept)
    ]
    labels = list(dataset.labels[:kept])
    padded_query_count = _next_power_of_two(kept)
    padded_rows = [list(row) for row in rows]
    padded_labels = list(labels)
    while len(padded_labels) < padded_query_count:
        source_index = len(padded_labels) % kept
        padded_rows.append(list(rows[source_index]))
        padded_labels.append(labels[source_index])

    flattened_inputs: list[int] = []
    for row in padded_rows:
        flattened_inputs.extend(row)

    class_count = (max(padded_labels) + 1) if padded_labels else 0
    return BinnedClassificationDataset(
        name=dataset.name,
        feature_count=dataset.feature_count,
        class_count=class_count,
        query_count=padded_query_count,
        original_query_count=kept,
        bin_count=dataset.bin_count,
        flattened_inputs=flattened_inputs,
        labels=padded_labels,
    )


def _node_counts_for_real_classes(
    nodes: list[object],
    *,
    trace_class_count: int,
    real_class_count: int,
) -> tuple[int, int]:
    """Return active trace-node and active model-node counts.

    The trace pads the class axis to a power of two. Nodes in added class
    channels are required by the proof representation, but are not part of
    the classifier trained for the dataset's real labels.
    """
    active_trace_node_count = 0
    active_node_count = 0
    for node in nodes:
        is_active = node.is_split_node or node.is_leaf_node
        if not is_active:
            continue
        active_trace_node_count += 1
        if node.tree_id % trace_class_count < real_class_count:
            active_node_count += 1
    return active_trace_node_count, active_node_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a multi-class, multi-round fixed-point histogram-tree model on a local reference dataset."
    )
    parser.add_argument("--config", type=Path, help="Path to an external JSON trainer config.")
    parser.add_argument(
        "--dataset",
        choices=[
            "iris",
            "wine",
            "wdbc",
            "optdigits-train",
            "optdigits-7class-train",
            "dry-bean",
            "iris-binary",
            "wine-binary",
            "breast-cancer-binary",
            "optdigits-parity-train",
        ],
        help="Dataset name.",
    )
    parser.add_argument("--iterations", type=int, help="Power-of-two boosting rounds.")
    parser.add_argument("--max-depth", type=int, help="Maximum per-tree depth.")
    parser.add_argument("--bin-count", type=int, help="Power-of-two feature bin count.")
    parser.add_argument(
        "--loss-mode",
        "--gradient-mode",
        dest="gradient_mode",
        choices=GRADIENT_MODES,
        help=(
            "Training objective. first-order-rss matches the paper's main GBDT "
            "trainer and is the default."
        ),
    )
    parser.add_argument("--max-original-queries", type=int, help="Optional cap before power-of-two padding.")
    parser.add_argument(
        "--leakage-free-evaluation-training-split",
        action="store_true",
        help=(
            "For Dry Bean and Optdigits-7, select the proportional stratified training subset "
            "before fitting the quantizer and report accuracy on the held-out test split. "
            "The held-out data is evaluated in plaintext and is not exported into either proof trace."
        ),
    )
    parser.add_argument("--training-trace", type=Path, help="Path to write the training trace.")
    parser.add_argument("--inference-trace", type=Path, help="Path to write the inference trace.")
    args = parser.parse_args()

    external_config = load_trainer_external_config(args.config) if args.config is not None else None

    dataset_name = (
        args.dataset
        if args.dataset is not None
        else (external_config.dataset if external_config is not None else "wine")
    )
    iterations = (
        args.iterations
        if args.iterations is not None
        else (external_config.iterations if external_config is not None else 2)
    )
    max_depth = (
        args.max_depth
        if args.max_depth is not None
        else (external_config.max_depth if external_config is not None else 1)
    )
    bin_count = (
        args.bin_count
        if args.bin_count is not None
        else (external_config.bin_count if external_config is not None else 16)
    )
    l2_regularization = external_config.l2_regularization if external_config is not None else 2
    split_penalty = external_config.split_penalty if external_config is not None else 0
    split_gain_threshold = external_config.split_gain_threshold if external_config is not None else 1
    gain_fractional_bits = external_config.gain_fractional_bits if external_config is not None else 2
    leaf_fractional_bits = external_config.leaf_fractional_bits if external_config is not None else 4
    score_fractional_bits = external_config.score_fractional_bits if external_config is not None else 4
    gradient_fractional_bits = external_config.gradient_fractional_bits if external_config is not None else 6
    hessian_fractional_bits = external_config.hessian_fractional_bits if external_config is not None else 8
    gradient_mode = (
        args.gradient_mode
        if args.gradient_mode is not None
        else (
            external_config.gradient_mode
            if external_config is not None
            else GRADIENT_MODE_FIRST_ORDER_RSS
        )
    )

    evaluation_dataset: BinnedClassificationDataset | None = None
    accuracy_split = "exported-inference"
    if args.leakage_free_evaluation_training_split:
        if dataset_name not in ("dry-bean", "optdigits-7class-train"):
            raise ValueError(
                "--leakage-free-evaluation-training-split supports only dry-bean and "
                "optdigits-7class-train"
            )
        if args.max_original_queries is None:
            raise ValueError(
                "--leakage-free-evaluation-training-split requires --max-original-queries"
            )
        evaluation_split = load_binned_evaluation_split(
            dataset_name,
            bin_count,
            args.max_original_queries,
            split_seed=42,
        )
        dataset = evaluation_split.training
        evaluation_dataset = evaluation_split.test
        accuracy_split = evaluation_split.split_name
    else:
        dataset = _truncate_dataset(_load_dataset(dataset_name, bin_count), args.max_original_queries)
    padded_class_count = (
        external_config.trace_class_count
        if external_config is not None and external_config.trace_class_count is not None
        else 1 << max(0, (dataset.class_count - 1).bit_length())
    )
    full_upper_bound = (
        external_config.full_upper_bound
        if external_config is not None and external_config.full_upper_bound is not None
        else bin_count
    )
    input_bit_bound = (
        external_config.input_bit_bound
        if external_config is not None and external_config.input_bit_bound is not None
        else _feature_bit_bound(full_upper_bound)
    )
    structure_bit_bound = (
        external_config.structure_bit_bound
        if external_config is not None and external_config.structure_bit_bound is not None
        else max(
            2,
            max(
                dataset.query_count * padded_class_count * iterations,
                padded_class_count * iterations * (1 << max_depth),
            ).bit_length(),
        )
    )
    config = FixedPointTrainerConfig(
        feature_count=dataset.feature_count,
        bin_count=bin_count,
        full_upper_bound=full_upper_bound,
        input_bit_bound=input_bit_bound,
        structure_bit_bound=structure_bit_bound,
        iteration_count=iterations,
        class_count=padded_class_count,
        max_depth=max_depth,
        learning_rate_numerator=external_config.learning_rate_numerator if external_config is not None else 4,
        learning_rate_denominator=external_config.learning_rate_denominator if external_config is not None else 1,
        l2_regularization=l2_regularization,
        split_penalty=split_penalty,
        split_gain_threshold=split_gain_threshold,
        gain_fractional_bits=gain_fractional_bits,
        leaf_fractional_bits=leaf_fractional_bits,
        score_fractional_bits=score_fractional_bits,
        gradient_fractional_bits=gradient_fractional_bits,
        hessian_fractional_bits=hessian_fractional_bits,
        gradient_mode=gradient_mode,
    )

    trainer = MultiClassHistogramFixedPointTrainer(config)
    run = trainer.fit(
        dataset.flattened_inputs,
        dataset.labels,
        build_trace=args.training_trace is not None,
    )
    inference = trainer.predict(
        run.model,
        dataset.flattened_inputs,
        dataset.labels,
        build_trace=args.inference_trace is not None,
    )
    accuracy_numerator = inference.accuracy_numerator
    accuracy_denominator = inference.accuracy_denominator
    if evaluation_dataset is not None:
        evaluation = trainer.predict(
            run.model,
            evaluation_dataset.flattened_inputs,
            build_trace=False,
        )
        accuracy_denominator = evaluation_dataset.original_query_count
        accuracy_numerator = sum(
            actual == predicted
            for actual, predicted in zip(
                evaluation_dataset.labels[:accuracy_denominator],
                evaluation.predicted_labels[:accuracy_denominator],
            )
        )

    if args.training_trace is not None:
        write_training_trace(str(args.training_trace), run.training_trace)
    if args.inference_trace is not None:
        write_inference_trace(str(args.inference_trace), inference.inference_trace)

    print(f"Dataset: {dataset.name}")
    if args.config is not None:
        print(f"  config: {args.config}")
    print(f"  original samples: {dataset.original_query_count}")
    print(f"  padded samples: {dataset.query_count}")
    print(f"  feature_count: {dataset.feature_count}")
    print(f"  real_class_count: {dataset.class_count}")
    print(f"  trace_class_count: {config.class_count}")
    print(f"  iteration_count: {config.iteration_count}")
    print(f"  max_depth: {config.max_depth}")
    print(f"  gradient_mode: {config.gradient_mode}")
    print(
        f"  learning_rate: {config.learning_rate_numerator}/{config.learning_rate_denominator}"
    )
    print(f"  split_gain_threshold: {config.split_gain_threshold}")
    active_trace_node_count, active_node_count = _node_counts_for_real_classes(
        run.model.forest.nodes,
        trace_class_count=config.class_count,
        real_class_count=dataset.class_count,
    )
    print(f"  tree_count: {run.model.params.tree_count}")
    print(f"  allocated_node_slot_count: {len(run.model.forest.nodes)}")
    print(f"  active_trace_node_count: {active_trace_node_count}")
    # Compatibility field consumed by benchmark summaries; this now uses the
    # manuscript meaning: nodes in trees for real dataset classes only.
    print(f"  active_node_count: {active_node_count}")
    print(f"  node_count: {active_node_count}")
    print(f"  training accuracy: {run.accuracy_numerator}/{run.accuracy_denominator}")
    print(f"  inference accuracy: {inference.accuracy_numerator}/{inference.accuracy_denominator}")
    print(f"  accuracy: {accuracy_numerator}/{accuracy_denominator}")
    print(f"  accuracy split: {accuracy_split}")
    print(f"  accuracy samples: {accuracy_denominator}")
    if args.training_trace is not None:
        print(f"  training trace: {args.training_trace}")
    if args.inference_trace is not None:
        print(f"  inference trace: {args.inference_trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
