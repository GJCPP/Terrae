#pragma once

#include "fixed_point.hpp"

#include <cstddef>
#include <cstdint>

class zkboostBinaryTreeShape {
public:
    size_t max_depth = 0;
    size_t leaf_count = 0;
    size_t internal_node_count = 0;
    size_t node_count = 0;
};

class zkboostTrainingHyperparameters {
public:
    size_t round_count = 0;
    size_t tree_count = 0;
    size_t max_depth = 0;
    size_t bin_count = 0;
    std::int64_t learning_rate = 0;
    std::int64_t l2_regularization = 0;
    std::int64_t split_penalty = 0;
};

class zkboostFixedPointParameters {
public:
    size_t input_bit_bound = 0;
    size_t full_upper_bound = 0;
    size_t structure_bit_bound = 0;
    fixedPointEncoding score_encoding;
    fixedPointEncoding probability_encoding;
    fixedPointEncoding gradient_encoding;
    fixedPointEncoding hessian_encoding;
    fixedPointEncoding gain_encoding;
    fixedPointEncoding leaf_value_encoding;
};
