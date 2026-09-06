#!/usr/bin/env python3

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GRADIENT_MODES = ("first-order-rss", "second-order-xgboost")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_int(mapping: dict[str, Any], key: str, context: str | None = None) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"{context or key} must be an integer")
    return value


def _default_if_none(value: int | None, default: int) -> int:
    return default if value is None else value


@dataclass(frozen=True)
class TrainerExternalConfig:
    dataset: str
    iterations: int
    max_depth: int
    bin_count: int
    learning_rate_numerator: int
    learning_rate_denominator: int
    l2_regularization: int
    split_penalty: int
    split_gain_threshold: int
    gain_fractional_bits: int
    leaf_fractional_bits: int
    score_fractional_bits: int
    gradient_fractional_bits: int
    hessian_fractional_bits: int
    gradient_mode: str
    full_upper_bound: int | None
    input_bit_bound: int | None
    structure_bit_bound: int | None
    trace_class_count: int | None


def trainer_external_config_defaults() -> TrainerExternalConfig:
    return TrainerExternalConfig(
        dataset="wine",
        iterations=2,
        max_depth=1,
        bin_count=16,
        learning_rate_numerator=4,
        learning_rate_denominator=1,
        l2_regularization=2,
        split_penalty=0,
        split_gain_threshold=1,
        gain_fractional_bits=2,
        leaf_fractional_bits=4,
        score_fractional_bits=4,
        gradient_fractional_bits=6,
        hessian_fractional_bits=8,
        gradient_mode="first-order-rss",
        full_upper_bound=None,
        input_bit_bound=None,
        structure_bit_bound=None,
        trace_class_count=None,
    )


def load_trainer_external_config(path: Path) -> TrainerExternalConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("trainer config root must be an object")

    defaults = trainer_external_config_defaults()

    dataset_section = raw.get("dataset", {})
    model_section = raw.get("model", {})
    regularization_section = raw.get("regularization", {})
    quantization_section = raw.get("quantization", {})
    trace_section = raw.get("trace", {})

    for name, section in (
        ("dataset", dataset_section),
        ("model", model_section),
        ("regularization", regularization_section),
        ("quantization", quantization_section),
        ("trace", trace_section),
    ):
        if not isinstance(section, dict):
            raise ValueError(f"{name} section must be an object")

    dataset = dataset_section.get("name", defaults.dataset)
    if not isinstance(dataset, str):
        raise ValueError("dataset.name must be a string")
    gradient_mode = model_section.get("gradient_mode", defaults.gradient_mode)
    if not isinstance(gradient_mode, str):
        raise ValueError("model.gradient_mode must be a string")

    config = TrainerExternalConfig(
        dataset=dataset,
        iterations=_default_if_none(_as_int(model_section, "iterations", "model.iterations"), defaults.iterations),
        max_depth=_default_if_none(_as_int(model_section, "max_depth", "model.max_depth"), defaults.max_depth),
        bin_count=_default_if_none(_as_int(model_section, "bin_count", "model.bin_count"), defaults.bin_count),
        learning_rate_numerator=_default_if_none(
            _as_int(model_section, "learning_rate_numerator", "model.learning_rate_numerator"),
            defaults.learning_rate_numerator,
        ),
        learning_rate_denominator=_default_if_none(
            _as_int(model_section, "learning_rate_denominator", "model.learning_rate_denominator"),
            defaults.learning_rate_denominator,
        ),
        l2_regularization=_default_if_none(
            _as_int(regularization_section, "l2_regularization", "regularization.l2_regularization"),
            defaults.l2_regularization,
        ),
        split_penalty=_default_if_none(
            _as_int(regularization_section, "split_penalty", "regularization.split_penalty"),
            defaults.split_penalty,
        ),
        split_gain_threshold=_default_if_none(
            _as_int(regularization_section, "split_gain_threshold", "regularization.split_gain_threshold"),
            defaults.split_gain_threshold,
        ),
        gain_fractional_bits=_default_if_none(
            _as_int(quantization_section, "gain_fractional_bits", "quantization.gain_fractional_bits"),
            defaults.gain_fractional_bits,
        ),
        leaf_fractional_bits=_default_if_none(
            _as_int(quantization_section, "leaf_fractional_bits", "quantization.leaf_fractional_bits"),
            defaults.leaf_fractional_bits,
        ),
        score_fractional_bits=_default_if_none(
            _as_int(quantization_section, "score_fractional_bits", "quantization.score_fractional_bits"),
            defaults.score_fractional_bits,
        ),
        gradient_fractional_bits=_default_if_none(
            _as_int(quantization_section, "gradient_fractional_bits", "quantization.gradient_fractional_bits"),
            defaults.gradient_fractional_bits,
        ),
        hessian_fractional_bits=_default_if_none(
            _as_int(quantization_section, "hessian_fractional_bits", "quantization.hessian_fractional_bits"),
            defaults.hessian_fractional_bits,
        ),
        gradient_mode=gradient_mode,
        full_upper_bound=_as_int(quantization_section, "full_upper_bound", "quantization.full_upper_bound"),
        input_bit_bound=_as_int(quantization_section, "input_bit_bound", "quantization.input_bit_bound"),
        structure_bit_bound=_as_int(trace_section, "structure_bit_bound", "trace.structure_bit_bound"),
        trace_class_count=_as_int(trace_section, "class_count", "trace.class_count"),
    )

    _require(
        config.dataset in (
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
        ),
        "unsupported dataset",
    )
    _require(config.iterations > 0, "model.iterations must be positive")
    _require(config.max_depth > 0, "model.max_depth must be positive because the root has depth one")
    _require(config.bin_count > 1, "model.bin_count must be greater than one")
    _require(config.learning_rate_numerator > 0, "model.learning_rate_numerator must be positive")
    _require(config.learning_rate_denominator > 0, "model.learning_rate_denominator must be positive")
    _require(config.l2_regularization > 0, "regularization.l2_regularization must be positive")
    _require(config.split_penalty >= 0, "regularization.split_penalty must be non-negative")
    _require(config.split_gain_threshold >= 0, "regularization.split_gain_threshold must be non-negative")
    _require(config.gain_fractional_bits >= 0, "quantization.gain_fractional_bits must be non-negative")
    _require(config.leaf_fractional_bits >= 0, "quantization.leaf_fractional_bits must be non-negative")
    _require(config.score_fractional_bits >= 0, "quantization.score_fractional_bits must be non-negative")
    _require(config.gradient_fractional_bits >= 0, "quantization.gradient_fractional_bits must be non-negative")
    _require(config.hessian_fractional_bits >= 0, "quantization.hessian_fractional_bits must be non-negative")
    _require(
        config.gradient_mode in GRADIENT_MODES,
        f"model.gradient_mode must be one of: {', '.join(GRADIENT_MODES)}",
    )
    if config.full_upper_bound is not None:
        _require(config.full_upper_bound >= config.bin_count, "quantization.full_upper_bound must cover bins")
    if config.input_bit_bound is not None:
        _require(config.input_bit_bound > 0, "quantization.input_bit_bound must be positive")
    if config.structure_bit_bound is not None:
        _require(config.structure_bit_bound > 0, "trace.structure_bit_bound must be positive")
    if config.trace_class_count is not None:
        _require(config.trace_class_count > 0, "trace.class_count must be positive")

    return config
