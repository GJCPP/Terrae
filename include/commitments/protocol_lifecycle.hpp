#pragma once

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

class ProtocolLifecycle {
protected:
    void mark_setup() noexcept { is_setup = true; }

    void begin_finalize(std::string_view protocol_name, bool require_setup = true)
    {
        if (require_setup && !is_setup) {
            throw std::runtime_error(std::string(protocol_name) + " not setup");
        }
        if (is_finalized) {
            throw std::runtime_error(std::string(protocol_name) + " already finalized");
        }
        is_finalized = true;
    }

    template <class FinalizeFn>
    void auto_finalize(std::string_view protocol_name, FinalizeFn&& finalize_fn) noexcept
    {
        if (!is_setup || is_finalized || std::uncaught_exceptions() > 0) {
            return;
        }
        try {
            if (!std::forward<FinalizeFn>(finalize_fn)()) {
                std::cerr << "Error: " << protocol_name << " finalize failed during destruction\n";
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error: " << protocol_name << " finalize threw during destruction: "
                      << ex.what() << "\n";
        } catch (...) {
            std::cerr << "Error: " << protocol_name << " finalize threw during destruction\n";
        }
    }

    bool is_setup = false;
    bool is_finalized = false;
};
