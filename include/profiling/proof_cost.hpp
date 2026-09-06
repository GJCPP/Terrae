#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace proof_cost {

class Report {
public:
    void add(const std::string& name, std::uint64_t value = 1)
    {
        counters_[name] += value;
    }

    void set_max(const std::string& name, std::uint64_t value)
    {
        auto [it, inserted] = counters_.emplace(name, value);
        if (!inserted) {
            it->second = std::max(it->second, value);
        }
    }

    const std::map<std::string, std::uint64_t>& counters() const noexcept
    {
        return counters_;
    }

private:
    std::map<std::string, std::uint64_t> counters_;
};

inline thread_local Report* active_report = nullptr;
inline thread_local std::string active_prefix;

inline std::string sanitize_name(std::string value)
{
    for (char& ch : value) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch)) {
            ch = static_cast<char>(95);
        } else {
            ch = static_cast<char>(std::tolower(uch));
        }
    }
    return value;
}

class Scope {
public:
    explicit Scope(Report& report)
        : previous_report_(active_report), previous_prefix_(active_prefix)
    {
        active_report = &report;
        active_prefix.clear();
    }

    ~Scope()
    {
        active_report = previous_report_;
        active_prefix = previous_prefix_;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Report* previous_report_ = nullptr;
    std::string previous_prefix_;
};

class PrefixScope {
public:
    explicit PrefixScope(std::string prefix)
        : previous_prefix_(active_prefix)
    {
        const std::string sanitized = sanitize_name(std::move(prefix));
        active_prefix = previous_prefix_.empty() ? sanitized : previous_prefix_ + "_" + sanitized;
    }

    ~PrefixScope()
    {
        active_prefix = previous_prefix_;
    }

    PrefixScope(const PrefixScope&) = delete;
    PrefixScope& operator=(const PrefixScope&) = delete;

private:
    std::string previous_prefix_;
};

inline bool enabled()
{
    return active_report != nullptr;
}

inline void add(const std::string& name, std::uint64_t value = 1)
{
    if (active_report == nullptr) {
        return;
    }
    active_report->add(name, value);
    if (!active_prefix.empty()) {
        active_report->add(active_prefix + "_" + name, value);
    }
}

inline void set_max(const std::string& name, std::uint64_t value)
{
    if (active_report == nullptr) {
        return;
    }
    active_report->set_max(name, value);
    if (!active_prefix.empty()) {
        active_report->set_max(active_prefix + "_" + name, value);
    }
}

inline void record_relation(const std::string& family,
                            std::size_t logical_slots,
                            std::size_t padded_slots = 0)
{
    if (active_report == nullptr) {
        return;
    }
    const std::string base = "terrae_" + sanitize_name(family);
    const std::size_t padded = padded_slots == 0 ? logical_slots : padded_slots;
    add(base + "_instance_count");
    add(base + "_logical_relation_slots", logical_slots);
    add(base + "_padded_relation_slots", padded);
    add(base + "_domain_size_sum", padded);
    set_max(base + "_max_domain_size", padded);
}

inline void print_metrics(const Report& report, std::ostream& out = std::cout)
{
    for (const auto& [name, value] : report.counters()) {
        out << "metric cost_" << name << " " << value << "\n";
    }
}

} // namespace proof_cost
