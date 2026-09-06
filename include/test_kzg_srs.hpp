#pragma once

#include "kzg.hpp"

#include <map>
#include <utility>

inline const kzgPublicParameters& cached_testing_kzg_public_parameters(size_t deg1_plus_1, size_t deg2_plus_1)
{
    static std::map<std::pair<size_t, size_t>, kzgPublicParameters> cache;
    const auto key = std::make_pair(deg1_plus_1, deg2_plus_1);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(key, generate_kzg_public_parameters(deg1_plus_1, deg2_plus_1)).first->second;
}

inline kzgPublicParameters testing_kzg_public_parameters(size_t deg1_plus_1, size_t deg2_plus_1)
{
    return cached_testing_kzg_public_parameters(deg1_plus_1, deg2_plus_1);
}
