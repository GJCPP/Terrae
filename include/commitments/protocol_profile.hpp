#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "terrae_env.hpp"

namespace structured_protocol_profile {

#if (defined(TERRAE_ENABLE_STRUCTURED_PROTOCOL_PROFILE) && TERRAE_ENABLE_STRUCTURED_PROTOCOL_PROFILE) || \
    (defined(ZKGBDT_ENABLE_STRUCTURED_PROTOCOL_PROFILE) && ZKGBDT_ENABLE_STRUCTURED_PROTOCOL_PROFILE)

inline bool compile_enabled()
{
    return true;
}

inline const std::string& output_path()
{
    static const std::string path = []() {
        const char* env = terrae_getenv("TERRAE_PROTOCOL_PROFILE_PATH");
        return env == nullptr ? std::string() : std::string(env);
    }();
    return path;
}

inline bool enabled()
{
    return !output_path().empty();
}

inline std::string sanitize_field(const std::string& value)
{
    std::string out = value;
    for (char& ch : out) {
        if (ch == '\t' || ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    return out;
}

inline void append_row(const std::string& category,
                       const std::string& name,
                       const std::string& params,
                       double seconds,
                       const std::string& status)
{
    if (!enabled()) {
        return;
    }

    static uint64_t sequence = 0;
    static bool header_initialized = false;

    if (!header_initialized) {
        std::ifstream existing(output_path(), std::ios::binary | std::ios::ate);
        const bool write_header = !existing.good() || existing.tellg() == 0;
        existing.close();

        if (write_header) {
            std::ofstream header_stream(output_path(), std::ios::app);
            header_stream << "seq\tcategory\tname\tparams\tseconds\tstatus\n";
        }
        header_initialized = true;
    }

    std::ofstream out(output_path(), std::ios::app);
    out << sequence++ << "\t" << sanitize_field(category) << "\t"
        << sanitize_field(name) << "\t" << sanitize_field(params) << "\t" << seconds << "\t"
        << sanitize_field(status) << "\n";
}

template <class Fn>
decltype(auto) timed_void(const std::string& category,
                          const std::string& name,
                          const std::string& params,
                          Fn&& fn)
{
    if (!enabled()) {
        fn();
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    try {
        fn();
    } catch (const std::exception& ex) {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        append_row(category, name, params, elapsed.count(), std::string("throw:") + ex.what());
        throw;
    } catch (...) {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        append_row(category, name, params, elapsed.count(), "throw");
        throw;
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    append_row(category, name, params, elapsed.count(), "ok");
}

#else

inline bool compile_enabled()
{
    return false;
}

inline bool enabled()
{
    return false;
}

inline void append_row(const std::string&,
                       const std::string&,
                       const std::string&,
                       double,
                       const std::string&)
{
}

template <class Fn>
decltype(auto) timed_void(const std::string&, const std::string&, const std::string&, Fn&& fn)
{
    if constexpr (std::is_void_v<std::invoke_result_t<Fn>>) {
        fn();
        return;
    } else {
        return fn();
    }
}

#endif

} // namespace structured_protocol_profile
