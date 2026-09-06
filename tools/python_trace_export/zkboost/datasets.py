from dataclasses import dataclass

from datasets import BinnedClassificationDataset, load_binned_reference_dataset


@dataclass(frozen=True)
class BinnedBinaryClassificationDataset:
    name: str
    feature_count: int
    query_count: int
    original_query_count: int
    bin_count: int
    flattened_inputs: list[int]
    labels: list[int]


def _binarize_labels(name: str, labels: list[int]) -> list[int]:
    if name == "iris-binary":
        return [1 if label == 0 else 0 for label in labels]
    if name == "wine-binary":
        return [1 if label == 0 else 0 for label in labels]
    if name == "breast-cancer-binary":
        return list(labels)
    if name == "optdigits-parity-train":
        return [label % 2 for label in labels]
    raise ValueError(f"unknown binary dataset: {name}")


def _base_dataset_name(name: str) -> str:
    if name == "iris-binary":
        return "iris"
    if name == "wine-binary":
        return "wine"
    if name == "breast-cancer-binary":
        return "wdbc"
    if name == "optdigits-parity-train":
        return "optdigits-train"
    raise ValueError(f"unknown binary dataset: {name}")


def _normalize_feature_bins_to_full_range(
    flattened_inputs: list[int],
    *,
    query_count: int,
    feature_count: int,
    bin_count: int,
) -> list[int]:
    out = list(flattened_inputs)
    for feature in range(feature_count):
        values = [flattened_inputs[query * feature_count + feature] for query in range(query_count)]
        lo = min(values)
        hi = max(values)
        if hi == lo:
            if query_count > 1:
                out[feature] = 0
                out[feature_count + feature] = bin_count - 1
            continue
        for query in range(query_count):
            offset = query * feature_count + feature
            out[offset] = ((flattened_inputs[offset] - lo) * (bin_count - 1)) // (hi - lo)
    return out


def load_binned_binary_reference_dataset(name: str, bin_count: int) -> BinnedBinaryClassificationDataset:
    base_name = _base_dataset_name(name)
    dataset: BinnedClassificationDataset = load_binned_reference_dataset(base_name, bin_count)
    labels = _binarize_labels(name, dataset.labels)
    flattened_inputs = _normalize_feature_bins_to_full_range(
        dataset.flattened_inputs,
        query_count=dataset.query_count,
        feature_count=dataset.feature_count,
        bin_count=dataset.bin_count,
    )
    return BinnedBinaryClassificationDataset(
        name=name,
        feature_count=dataset.feature_count,
        query_count=dataset.query_count,
        original_query_count=dataset.original_query_count,
        bin_count=dataset.bin_count,
        flattened_inputs=flattened_inputs,
        labels=labels,
    )
