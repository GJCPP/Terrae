#!/usr/bin/env python3

import unittest
from dataclasses import replace
from unittest.mock import patch

from fixed_point_trainer import (
    GRADIENT_MODE_FIRST_ORDER_RSS,
    GRADIENT_MODE_SECOND_ORDER_XGBOOST,
    FixedPointTrainerConfig,
    MultiClassHistogramFixedPointTrainer,
    _first_order_rss_gain_fraction,
)

from benchmark_focus_extract import (
    BASE_SUMMARY_FIELDNAMES,
    _accuracy,
    _accuracy_samples,
    _accuracy_split,
)
from datasets import (
    RawClassificationDataset,
    fit_uniform_bin_quantizer,
    load_binned_evaluation_split,
    stratified_holdout,
    stratified_subsample,
    transform_uniform_bins,
)


class UniformBinQuantizerTest(unittest.TestCase):
    def test_transform_clamps_values_outside_training_range(self) -> None:
        quantizer = fit_uniform_bin_quantizer([[0.0], [10.0]], 8)
        self.assertEqual(
            transform_uniform_bins([[-5.0], [0.0], [5.0], [10.0], [15.0]], quantizer),
            [[0], [0], [4], [7], [7]],
        )

    def test_stratified_holdout_is_disjoint_and_deterministic(self) -> None:
        dataset = RawClassificationDataset(
            name="synthetic",
            feature_count=1,
            class_count=2,
            features=[[float(index)] for index in range(20)],
            labels=[index % 2 for index in range(20)],
        )
        training, test = stratified_holdout(dataset, test_denominator=5)
        training_again, test_again = stratified_holdout(dataset, test_denominator=5)

        training_ids = {int(row[0]) for row in training.features}
        test_ids = {int(row[0]) for row in test.features}
        self.assertEqual(len(training_ids), 16)
        self.assertEqual(len(test_ids), 4)
        self.assertFalse(training_ids & test_ids)
        self.assertEqual(training, training_again)
        self.assertEqual(test, test_again)
        training_other_seed, test_other_seed = stratified_holdout(
            dataset,
            test_denominator=5,
            seed=7,
        )
        self.assertNotEqual(training, training_other_seed)
        self.assertNotEqual(test, test_other_seed)
        self.assertEqual(training.labels.count(0), 8)
        self.assertEqual(training.labels.count(1), 8)
        self.assertEqual(test.labels.count(0), 2)
        self.assertEqual(test.labels.count(1), 2)


    def test_stratified_subsample_preserves_source_proportions(self) -> None:
        dataset = RawClassificationDataset(
            name="imbalanced",
            feature_count=1,
            class_count=3,
            features=[[float(index)] for index in range(100)],
            labels=[0] * 50 + [1] * 30 + [2] * 20,
        )

        selected = stratified_subsample(dataset, 11, seed=42)
        selected_again = stratified_subsample(dataset, 11, seed=42)
        selected_other_seed = stratified_subsample(dataset, 11, seed=7)

        self.assertEqual(selected.labels.count(0), 6)
        self.assertEqual(selected.labels.count(1), 3)
        self.assertEqual(selected.labels.count(2), 2)
        self.assertEqual(len({int(row[0]) for row in selected.features}), 11)
        self.assertEqual(selected, selected_again)
        self.assertNotEqual(selected, selected_other_seed)

    def test_leakage_free_split_pads_beyond_clean_training_pool(self) -> None:
        training_pool = RawClassificationDataset(
            name="synthetic-optdigits-train",
            feature_count=1,
            class_count=2,
            features=[[float(index)] for index in range(6)],
            labels=[0, 1, 0, 1, 0, 1],
        )
        test_pool = RawClassificationDataset(
            name="synthetic-optdigits-test",
            feature_count=1,
            class_count=2,
            features=[[100.0], [101.0]],
            labels=[0, 1],
        )

        with (
            patch("datasets.load_optdigits_7class_train_dataset", return_value=training_pool),
            patch("datasets.load_optdigits_7class_test_dataset", return_value=test_pool),
        ):
            split = load_binned_evaluation_split(
                "optdigits-7class-train",
                bin_count=4,
                training_samples=8,
            )

        self.assertEqual(split.training.original_query_count, 6)
        self.assertEqual(split.training.query_count, 8)
        self.assertEqual(split.quantizer_fit_sample_count, 6)
        self.assertEqual(split.test.original_query_count, 2)
        self.assertEqual(split.test.query_count, 2)

        rows = [
            split.training.flattened_inputs[
                index * split.training.feature_count :
                (index + 1) * split.training.feature_count
            ]
            for index in range(split.training.query_count)
        ]
        self.assertEqual(rows[6:], rows[:2])
        self.assertEqual(split.training.labels[6:], split.training.labels[:2])

    def test_requested_trace_capacity_can_exceed_next_power_of_two_of_pool(self) -> None:
        training_pool = RawClassificationDataset(
            name="synthetic-optdigits-train",
            feature_count=1,
            class_count=2,
            features=[[float(index)] for index in range(6)],
            labels=[0, 1, 0, 1, 0, 1],
        )
        test_pool = RawClassificationDataset(
            name="synthetic-optdigits-test",
            feature_count=1,
            class_count=2,
            features=[[100.0], [101.0]],
            labels=[0, 1],
        )

        with (
            patch("datasets.load_optdigits_7class_train_dataset", return_value=training_pool),
            patch("datasets.load_optdigits_7class_test_dataset", return_value=test_pool),
        ):
            split = load_binned_evaluation_split(
                "optdigits-7class-train",
                bin_count=4,
                training_samples=16,
            )

        self.assertEqual(split.training.original_query_count, 6)
        self.assertEqual(split.training.query_count, 16)
        self.assertEqual(split.quantizer_fit_sample_count, 6)


class FixedPointLearningRateTest(unittest.TestCase):
    def test_first_order_rss_gain_normalization_is_exact(self) -> None:
        for left_gradient, right_gradient, left_count, right_count, scale in [
            (17, -9, 3, 5, 64),
            (-31, 7, 11, 2, 64),
            (0, 19, 1, 13, 16),
        ]:
            numerator, denominator = _first_order_rss_gain_fraction(
                left_gradient, right_gradient, left_count, right_count, scale
            )
            parent_gradient = left_gradient + right_gradient
            left_hessian = left_count * scale
            right_hessian = right_count * scale
            parent_hessian = (left_count + right_count) * scale
            old_numerator = (
                left_gradient**2 * right_hessian * parent_hessian
                + right_gradient**2 * left_hessian * parent_hessian
                - parent_gradient**2 * left_hessian * right_hessian
            )
            old_denominator = left_hessian * right_hessian * parent_hessian
            self.assertEqual(old_numerator, numerator * scale**2)
            self.assertEqual(old_denominator, denominator * scale**2)
            self.assertEqual(
                (old_numerator << 2) // old_denominator,
                (numerator << 2) // denominator,
            )

    def test_half_learning_rate_is_stored_in_trace_and_leaf_values(self) -> None:
        config = FixedPointTrainerConfig(
            feature_count=1,
            bin_count=2,
            full_upper_bound=2,
            input_bit_bound=2,
            structure_bit_bound=4,
            iteration_count=1,
            class_count=2,
            max_depth=1,
            learning_rate_numerator=1,
            learning_rate_denominator=2,
            l2_regularization=1,
            leaf_fractional_bits=4,
            score_fractional_bits=4,
            gradient_fractional_bits=2,
            gradient_mode=GRADIENT_MODE_SECOND_ORDER_XGBOOST,
        )
        run = MultiClassHistogramFixedPointTrainer(config).fit(
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        )

        self.assertEqual(run.model.params.learning_rate_numerator, 1)
        self.assertEqual(run.model.params.learning_rate_denominator, 2)
        self.assertEqual(run.model.forest.nodes[0].leaf_value, 4)
        self.assertEqual(run.model.forest.nodes[2].leaf_value, -4)

    def test_first_order_rss_uses_negative_gradient_mean_for_leaf_value(self) -> None:
        config = FixedPointTrainerConfig(
            feature_count=1,
            bin_count=2,
            full_upper_bound=2,
            input_bit_bound=2,
            structure_bit_bound=4,
            iteration_count=1,
            class_count=2,
            max_depth=1,
            learning_rate_numerator=1,
            learning_rate_denominator=2,
            leaf_fractional_bits=4,
            score_fractional_bits=4,
            gradient_fractional_bits=2,
            gradient_mode=GRADIENT_MODE_FIRST_ORDER_RSS,
        )
        trainer = MultiClassHistogramFixedPointTrainer(config)
        run = trainer.fit(
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            build_trace=False,
        )

        self.assertEqual(run.model.forest.nodes[0].leaf_value, 2)
        self.assertEqual(run.model.forest.nodes[2].leaf_value, -2)
        traced = trainer.fit([0, 0, 0, 0], [0, 0, 0, 1])
        self.assertIsNotNone(traced.training_trace)
        self.assertEqual(traced.training_trace.loss_mode, 0)

    def test_first_order_rss_split_uses_quantized_rss_reduction(self) -> None:
        config = FixedPointTrainerConfig(
            feature_count=1,
            bin_count=2,
            full_upper_bound=2,
            input_bit_bound=2,
            structure_bit_bound=5,
            iteration_count=1,
            class_count=2,
            max_depth=2,
            learning_rate_numerator=1,
            learning_rate_denominator=2,
            split_gain_threshold=4,
            gain_fractional_bits=2,
            leaf_fractional_bits=4,
            score_fractional_bits=4,
            gradient_fractional_bits=2,
            gradient_mode=GRADIENT_MODE_FIRST_ORDER_RSS,
        )
        run = MultiClassHistogramFixedPointTrainer(config).fit(
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            build_trace=False,
        )

        self.assertEqual(run.model.params.max_depth, 2)
        self.assertEqual(run.model.nodes_per_tree, 4)
        self.assertTrue(run.model.forest.nodes[0].is_split_node)
        self.assertEqual(run.model.forest.nodes[0].threshold, 1)

        root_only = MultiClassHistogramFixedPointTrainer(
            replace(config, max_depth=1)
        ).fit(
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            build_trace=False,
        )
        self.assertEqual(root_only.model.nodes_per_tree, 2)
        self.assertFalse(root_only.model.forest.nodes[0].is_split_node)


class BenchmarkAccuracySchemaTest(unittest.TestCase):
    def test_summary_uses_explicit_held_out_accuracy_fields(self) -> None:
        case_result = {
            "benchmark_payload": {
                "train_export": {
                    "fields": {
                        "training_accuracy": "63/64",
                        "accuracy": "740/1264",
                        "accuracy_split": "optdigits-official-test-labels-0-6",
                        "accuracy_samples": "1264",
                    }
                }
            }
        }

        self.assertIn("accuracy", BASE_SUMMARY_FIELDNAMES)
        self.assertNotIn("training_accuracy", BASE_SUMMARY_FIELDNAMES)
        self.assertEqual(_accuracy(case_result), "740/1264")
        self.assertEqual(
            _accuracy_split(case_result),
            "optdigits-official-test-labels-0-6",
        )
        self.assertEqual(_accuracy_samples(case_result), "1264")


if __name__ == "__main__":
    unittest.main()
