#pragma once

#include <cstddef>
#include <stdexcept>

// Public training-objective selector shared by the semantic trace, Terrae,
// and CertXGB. Numeric codes are part of the serialized trace format.
enum class forestTrainingLoss : size_t {
    FirstOrderRss = 0,
    SecondOrderXgboost = 1,
};

inline size_t forest_training_loss_code(forestTrainingLoss loss)
{
    return static_cast<size_t>(loss);
}

inline forestTrainingLoss forest_training_loss_from_code(size_t code)
{
    switch (code) {
    case 0:
        return forestTrainingLoss::FirstOrderRss;
    case 1:
        return forestTrainingLoss::SecondOrderXgboost;
    default:
        throw std::runtime_error("forest training loss mode must be 0 (first-order RSS) or 1 (second-order XGBoost)");
    }
}

inline const char* forest_training_loss_name(forestTrainingLoss loss)
{
    switch (loss) {
    case forestTrainingLoss::FirstOrderRss:
        return "first-order-rss";
    case forestTrainingLoss::SecondOrderXgboost:
        return "second-order-xgboost";
    }
    throw std::runtime_error("invalid forest training loss mode");
}
