#include "forest_prediction.hpp"

#include "bls12_381_wrapper.hpp"

#include <stdexcept>

forestPredictionPublicTable build_forest_prediction_public_table(
    size_t iteration_count,
    size_t class_count,
    size_t score_bit_bound)
{
    if (iteration_count == 0 || !isPowerOfTwo(iteration_count)) {
        throw std::runtime_error("build_forest_prediction_public_table: iteration_count must be a non-zero power of two");
    }
    if (class_count == 0 || !isPowerOfTwo(class_count)) {
        throw std::runtime_error("build_forest_prediction_public_table: class_count must be a non-zero power of two");
    }
    if (score_bit_bound == 0) {
        throw std::runtime_error("build_forest_prediction_public_table: score_bit_bound must be non-zero");
    }

    forestPredictionPublicTable out;
    out.iteration_count = iteration_count;
    out.class_count = class_count;
    out.score_bit_bound = score_bit_bound;
    return out;
}
