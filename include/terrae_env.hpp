#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

inline const char* terrae_getenv(const char* name)
{
    if (const char* value = std::getenv(name); value != nullptr) {
        return value;
    }

    constexpr std::string_view terrae_prefix = "TERRAE_";
    const std::string_view key(name);
    if (key.rfind(terrae_prefix, 0) != 0) {
        return nullptr;
    }

    static std::string legacy_name;
    legacy_name = "ZKGBDT_";
    legacy_name.append(key.substr(terrae_prefix.size()));
    return std::getenv(legacy_name.c_str());
}

inline bool terrae_env_flag_enabled(const char* name)
{
    const char* flag = terrae_getenv(name);
    return flag != nullptr && flag[0] != '\0' && std::string(flag) != "0" && std::string(flag) != "false";
}
