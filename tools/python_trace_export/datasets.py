from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable


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


@dataclass(frozen=True)
class RawClassificationDataset:
    name: str
    feature_count: int
    class_count: int
    features: list[list[float]]
    labels: list[int]


@dataclass(frozen=True)
class BinnedClassificationDataset:
    name: str
    feature_count: int
    class_count: int
    query_count: int
    original_query_count: int
    bin_count: int
    flattened_inputs: list[int]
    labels: list[int]


@dataclass(frozen=True)
class UniformBinQuantizer:
    bin_count: int
    minima: tuple[float, ...]
    maxima: tuple[float, ...]


@dataclass(frozen=True)
class BinnedClassificationSplit:
    training: BinnedClassificationDataset
    test: BinnedClassificationDataset
    split_name: str
    quantizer_fit_sample_count: int
    quantizer: UniformBinQuantizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _raw_dataset_path(*parts: str) -> Path:
    return _repo_root() / "data" / "datasets" / "raw" / Path(*parts)


def _parse_csv_lines(lines: Iterable[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        rows.append([field.strip() for field in stripped.split(",")])
    return rows


def _round_robin_reorder_by_label(features: list[list[float]],
                                  labels: list[int],
                                  class_count: int) -> tuple[list[list[float]], list[int]]:
    _require(len(features) == len(labels), "features and labels must have the same length")
    _require(class_count > 0, "class_count must be positive")

    buckets: list[list[int]] = [[] for _ in range(class_count)]
    for index, label in enumerate(labels):
        if label < 0 or label >= class_count:
            raise ValueError(f"label {label} is outside class range 0..{class_count - 1}")
        buckets[label].append(index)

    reordered_indices: list[int] = []
    offset = 0
    while len(reordered_indices) < len(labels):
        appended = False
        for bucket in buckets:
            if offset < len(bucket):
                reordered_indices.append(bucket[offset])
                appended = True
        if not appended:
            break
        offset += 1

    return (
        [features[index] for index in reordered_indices],
        [labels[index] for index in reordered_indices],
    )


def load_iris_dataset() -> RawClassificationDataset:
    path = _raw_dataset_path("iris.data")
    rows = _parse_csv_lines(path.read_text(encoding="utf-8").splitlines())
    label_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    features: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        if len(row) != 5:
            raise ValueError("iris.data rows must have 5 columns")
        features.append([float(value) for value in row[:4]])
        labels.append(label_map[row[4]])
    return RawClassificationDataset(
        name="iris",
        feature_count=4,
        class_count=3,
        features=features,
        labels=labels,
    )


def load_wine_dataset() -> RawClassificationDataset:
    path = _raw_dataset_path("wine.data")
    rows = _parse_csv_lines(path.read_text(encoding="utf-8").splitlines())
    features: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        if len(row) != 14:
            raise ValueError("wine.data rows must have 14 columns")
        labels.append(int(row[0]) - 1)
        features.append([float(value) for value in row[1:]])
    return RawClassificationDataset(
        name="wine",
        feature_count=13,
        class_count=3,
        features=features,
        labels=labels,
    )


def load_wdbc_dataset() -> RawClassificationDataset:
    path = _raw_dataset_path("wdbc.data")
    rows = _parse_csv_lines(path.read_text(encoding="utf-8").splitlines())
    features: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        if len(row) != 32:
            raise ValueError("wdbc.data rows must have 32 columns")
        diagnosis = row[1]
        if diagnosis == "M":
            labels.append(1)
        elif diagnosis == "B":
            labels.append(0)
        else:
            raise ValueError(f"unexpected WDBC diagnosis label: {diagnosis}")
        features.append([float(value) for value in row[2:]])
    return RawClassificationDataset(
        name="wdbc",
        feature_count=30,
        class_count=2,
        features=features,
        labels=labels,
    )


def load_optdigits_dataset(split: str) -> RawClassificationDataset:
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    filename = "optdigits.tra" if split == "train" else "optdigits.tes"
    path = _raw_dataset_path(filename)
    rows = _parse_csv_lines(path.read_text(encoding="utf-8").splitlines())
    features: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        if len(row) != 65:
            raise ValueError(f"{filename} rows must have 65 columns")
        features.append([float(value) for value in row[:64]])
        labels.append(int(row[64]))
    return RawClassificationDataset(
        name=f"optdigits-{split}",
        feature_count=64,
        class_count=10,
        features=features,
        labels=labels,
    )


def load_optdigits_7class_train_dataset() -> RawClassificationDataset:
    raw = load_optdigits_dataset("train")
    features: list[list[float]] = []
    labels: list[int] = []
    for row, label in zip(raw.features, raw.labels):
        if 0 <= label <= 6:
            features.append(list(row))
            labels.append(label)
    features, labels = _round_robin_reorder_by_label(features, labels, 7)
    return RawClassificationDataset(
        name="optdigits-7class-train",
        feature_count=64,
        class_count=7,
        features=features,
        labels=labels,
    )


def load_optdigits_7class_test_dataset() -> RawClassificationDataset:
    raw = load_optdigits_dataset("test")
    features: list[list[float]] = []
    labels: list[int] = []
    for row, label in zip(raw.features, raw.labels):
        if 0 <= label <= 6:
            features.append(list(row))
            labels.append(label)
    return RawClassificationDataset(
        name="optdigits-7class-test",
        feature_count=64,
        class_count=7,
        features=features,
        labels=labels,
    )


def load_dry_bean_dataset() -> RawClassificationDataset:
    path = _raw_dataset_path("Dry_Bean_Dataset.arff")
    label_map = {
        "SEKER": 0,
        "BARBUNYA": 1,
        "BOMBAY": 2,
        "CALI": 3,
        "HOROZ": 4,
        "SIRA": 5,
        "DERMASON": 6,
    }
    features: list[list[float]] = []
    labels: list[int] = []
    in_data = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if not in_data:
            if stripped.upper() == "@DATA":
                in_data = True
            continue
        row = [field.strip() for field in stripped.split(",")]
        if len(row) != 17:
            raise ValueError("Dry_Bean_Dataset.arff rows must have 17 columns")
        label = row[16].upper()
        if label not in label_map:
            raise ValueError(f"unexpected Dry Bean class label: {row[16]}")
        features.append([float(value) for value in row[:16]])
        labels.append(label_map[label])
    features, labels = _round_robin_reorder_by_label(features, labels, len(label_map))
    return RawClassificationDataset(
        name="dry-bean",
        feature_count=16,
        class_count=7,
        features=features,
        labels=labels,
    )


def load_reference_dataset(name: str) -> RawClassificationDataset:
    if name == "iris":
        return load_iris_dataset()
    if name == "wine":
        return load_wine_dataset()
    if name == "wdbc":
        return load_wdbc_dataset()
    if name == "optdigits-train":
        return load_optdigits_dataset("train")
    if name == "optdigits-test":
        return load_optdigits_dataset("test")
    if name == "optdigits-7class-train":
        return load_optdigits_7class_train_dataset()
    if name == "dry-bean":
        return load_dry_bean_dataset()
    raise ValueError(f"unknown dataset: {name}")


def fit_uniform_bin_quantizer(
    features: list[list[float]],
    bin_count: int,
) -> UniformBinQuantizer:
    _require(features, "features must be non-empty")
    _require(
        bin_count > 1 and _is_power_of_two(bin_count),
        "bin_count must be a power of two greater than one",
    )
    feature_count = len(features[0])
    for row in features:
        _require(len(row) == feature_count, "all feature rows must have the same width")
    return UniformBinQuantizer(
        bin_count=bin_count,
        minima=tuple(min(row[feature] for row in features) for feature in range(feature_count)),
        maxima=tuple(max(row[feature] for row in features) for feature in range(feature_count)),
    )


def transform_uniform_bins(
    features: list[list[float]],
    quantizer: UniformBinQuantizer,
) -> list[list[int]]:
    feature_count = len(quantizer.minima)
    _require(feature_count > 0, "quantizer must contain at least one feature")
    _require(len(quantizer.maxima) == feature_count, "quantizer bounds must have the same width")
    quantized: list[list[int]] = []
    for row in features:
        _require(len(row) == feature_count, "all feature rows must match the quantizer width")
        encoded_row: list[int] = []
        for feature, value in enumerate(row):
            lower = quantizer.minima[feature]
            upper = quantizer.maxima[feature]
            if upper <= lower:
                encoded_row.append(0)
                continue
            normalized = (value - lower) / (upper - lower)
            bin_index = int(normalized * quantizer.bin_count)
            encoded_row.append(max(0, min(quantizer.bin_count - 1, bin_index)))
        quantized.append(encoded_row)
    return quantized


def quantize_uniform_bins(features: list[list[float]], bin_count: int) -> list[list[int]]:
    quantizer = fit_uniform_bin_quantizer(features, bin_count)
    return transform_uniform_bins(features, quantizer)


def pad_dataset_to_power_of_two(name: str,
                                quantized_features: list[list[int]],
                                labels: list[int],
                                minimum_query_count: int | None = None) -> BinnedClassificationDataset:
    _require(len(quantized_features) == len(labels), "features and labels must have the same length")
    _require(quantized_features, "quantized_features must be non-empty")
    original_feature_count = len(quantized_features[0])
    for row in quantized_features:
        _require(len(row) == original_feature_count, "all quantized feature rows must have the same width")

    padded_feature_count = _next_power_of_two(original_feature_count)
    padded_quantized_features: list[list[int]] = []
    for row in quantized_features:
        padded_row = list(row)
        padded_row.extend([0] * (padded_feature_count - original_feature_count))
        padded_quantized_features.append(padded_row)

    original_query_count = len(labels)
    if minimum_query_count is not None:
        _require(minimum_query_count > 0, "minimum_query_count must be positive")
    padded_query_count = _next_power_of_two(
        max(original_query_count, minimum_query_count or original_query_count)
    )
    padded_features = [list(row) for row in padded_quantized_features]
    padded_labels = list(labels)
    while len(padded_labels) < padded_query_count:
        source_index = len(padded_labels) % original_query_count
        padded_features.append(list(padded_quantized_features[source_index]))
        padded_labels.append(labels[source_index])

    flattened_inputs: list[int] = []
    for row in padded_features:
        flattened_inputs.extend(row)

    class_count = (max(padded_labels) + 1) if padded_labels else 0
    bin_count = max(max(row) for row in padded_features) + 1
    return BinnedClassificationDataset(
        name=name,
        feature_count=padded_feature_count,
        class_count=class_count,
        query_count=padded_query_count,
        original_query_count=original_query_count,
        bin_count=bin_count,
        flattened_inputs=flattened_inputs,
        labels=padded_labels,
    )


def load_binned_reference_dataset(name: str, bin_count: int) -> BinnedClassificationDataset:
    raw = load_reference_dataset(name)
    quantized = quantize_uniform_bins(raw.features, bin_count)
    return pad_dataset_to_power_of_two(raw.name, quantized, raw.labels)


def _raw_subset(
    dataset: RawClassificationDataset,
    indices: list[int],
    name: str,
) -> RawClassificationDataset:
    features = [list(dataset.features[index]) for index in indices]
    labels = [dataset.labels[index] for index in indices]
    features, labels = _round_robin_reorder_by_label(
        features,
        labels,
        dataset.class_count,
    )
    return RawClassificationDataset(
        name=name,
        feature_count=dataset.feature_count,
        class_count=dataset.class_count,
        features=features,
        labels=labels,
    )


def stratified_holdout(
    dataset: RawClassificationDataset,
    test_denominator: int = 5,
    seed: int = 42,
) -> tuple[RawClassificationDataset, RawClassificationDataset]:
    _require(test_denominator > 1, "test_denominator must be greater than one")
    buckets: list[list[int]] = [[] for _ in range(dataset.class_count)]
    for index, label in enumerate(dataset.labels):
        _require(0 <= label < dataset.class_count, "label is outside the dataset class range")
        buckets[label].append(index)

    training_indices: list[int] = []
    test_indices: list[int] = []
    generator = random.Random(seed)
    for bucket in buckets:
        generator.shuffle(bucket)
        _require(len(bucket) >= test_denominator, "each class must contain enough samples for a holdout")
        test_count = max(1, len(bucket) // test_denominator)
        training_indices.extend(bucket[:-test_count])
        test_indices.extend(bucket[-test_count:])

    return (
        _raw_subset(dataset, training_indices, f"{dataset.name}-train"),
        _raw_subset(dataset, test_indices, f"{dataset.name}-test"),
    )


def stratified_subsample(
    dataset: RawClassificationDataset,
    sample_count: int,
    seed: int = 42,
) -> RawClassificationDataset:
    _require(sample_count > 0, "sample_count must be positive")
    _require(
        sample_count <= len(dataset.labels),
        "sample_count must not exceed the dataset size",
    )

    buckets: list[list[int]] = [[] for _ in range(dataset.class_count)]
    for index, label in enumerate(dataset.labels):
        _require(0 <= label < dataset.class_count, "label is outside the dataset class range")
        buckets[label].append(index)

    generator = random.Random(seed)
    for bucket in buckets:
        generator.shuffle(bucket)

    total_count = len(dataset.labels)
    exact_counts = [
        sample_count * len(bucket) / total_count
        for bucket in buckets
    ]
    selected_counts = [int(exact_count) for exact_count in exact_counts]
    remainder = sample_count - sum(selected_counts)
    remainder_order = sorted(
        range(dataset.class_count),
        key=lambda cls: (
            exact_counts[cls] - selected_counts[cls],
            len(buckets[cls]),
            -cls,
        ),
        reverse=True,
    )
    for cls in remainder_order[:remainder]:
        selected_counts[cls] += 1

    selected_indices: list[int] = []
    for cls, selected_count in enumerate(selected_counts):
        _require(
            selected_count <= len(buckets[cls]),
            "stratified allocation exceeds a class bucket",
        )
        selected_indices.extend(buckets[cls][:selected_count])

    return _raw_subset(
        dataset,
        selected_indices,
        f"{dataset.name}-stratified-N{sample_count}-seed-{seed}",
    )


def load_binned_evaluation_split(
    name: str,
    bin_count: int,
    training_samples: int,
    split_seed: int = 42,
) -> BinnedClassificationSplit:
    _require(training_samples > 0, "training_samples must be positive")
    if name == "dry-bean":
        training_pool, test_raw = stratified_holdout(
            load_dry_bean_dataset(),
            seed=split_seed,
        )
        split_name = (
            f"dry-bean-stratified-holdout-80-20-seed-{split_seed}"
            f"-proportional-training-subset-seed-{split_seed}"
        )
    elif name == "optdigits-7class-train":
        training_pool = load_optdigits_7class_train_dataset()
        test_raw = load_optdigits_7class_test_dataset()
        split_name = (
            "optdigits-official-test-labels-0-6"
            f"-proportional-training-subset-seed-{split_seed}"
        )
    else:
        raise ValueError(f"evaluation split is not defined for dataset: {name}")

    # `training_samples` is the requested proof-trace capacity. When the clean
    # training pool is smaller, use every real training sample and fill the
    # remaining rows only through the normal deterministic padding rule. The
    # held-out test pool remains disjoint and is never used for padding.
    selected_training_count = min(training_samples, len(training_pool.labels))
    selected_training = stratified_subsample(
        training_pool,
        selected_training_count,
        seed=split_seed,
    )
    quantizer = fit_uniform_bin_quantizer(selected_training.features, bin_count)
    training = pad_dataset_to_power_of_two(
        selected_training.name,
        transform_uniform_bins(selected_training.features, quantizer),
        selected_training.labels,
        minimum_query_count=training_samples,
    )
    test = pad_dataset_to_power_of_two(
        test_raw.name,
        transform_uniform_bins(test_raw.features, quantizer),
        test_raw.labels,
    )
    return BinnedClassificationSplit(
        training=training,
        test=test,
        split_name=split_name,
        quantizer_fit_sample_count=selected_training_count,
        quantizer=quantizer,
    )
