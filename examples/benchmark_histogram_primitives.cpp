#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "matrix_utils.hpp"
#include "polynomial.hpp"
#include "proof_runtime.hpp"
#include "zkboost/primitives/histogram_ram.hpp"
#include "zkboost/shared_trace/weighted_logup.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

enum class Route {
    Terrae,
    Ram,
    WeightedLogup,
};

class Scenario {
public:
    size_t n = 0;
    size_t m = 0;
};

class HistogramWitness {
public:
    poly indices;
    poly values;
    poly sums;
};

class BenchmarkResult {
public:
    double prover_seconds = 0.0;
    double verifier_prepare_seconds = 0.0;
    double verifier_seconds = 0.0;
    size_t proof_bytes = 0;
    bool accepted = false;
    bool proof_fully_consumed = false;
};

std::string route_name(Route route)
{
    switch (route) {
    case Route::Terrae:
        return "terrae_hist_agg";
    case Route::Ram:
        return "zkboost_ram";
    case Route::WeightedLogup:
        return "zkboost_weighted_logup";
    }
    throw std::runtime_error("benchmark_histogram_primitives: unknown route");
}

template <class Fn>
double seconds_for(Fn&& fn)
{
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

bool is_power_of_two(size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

size_t checked_mul(size_t left, size_t right, const char* context)
{
    if (right != 0 && left > static_cast<size_t>(-1) / right) {
        throw std::runtime_error(std::string(context) + ": size overflow");
    }
    return left * right;
}

poly poly_from_evaluations(const std::vector<Fr>& evals)
{
    return poly_from_evals(evals);
}

bool file_exists(const std::string& path)
{
    std::ifstream in(path);
    return in.good();
}

std::string join_path(const std::string& dir, const std::string& filename)
{
    if (dir.empty()) {
        return filename;
    }
    const char last = dir[dir.size() - 1];
    if (last == '/') {
        return dir + filename;
    }
    return dir + "/" + filename;
}

std::string proof_filename(Route route, const Scenario& scenario, size_t proof_n, size_t proof_m)
{
    return route_name(route) +
           "_n" + std::to_string(scenario.n) +
           "_m" + std::to_string(scenario.m) +
           "_proofn" + std::to_string(proof_n) +
           "_proofm" + std::to_string(proof_m) +
           ".proof";
}

void write_proof_file(const std::string& path, const SerializedProof& proof)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("benchmark_histogram_primitives: failed to open proof output: " + path);
    }
    out.write(reinterpret_cast<const char*>(proof.bytes.data()),
              static_cast<std::streamsize>(proof.bytes.size()));
    if (!out) {
        throw std::runtime_error("benchmark_histogram_primitives: failed to write proof: " + path);
    }
}

SerializedProof read_proof_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("benchmark_histogram_primitives: failed to open proof input: " + path);
    }

    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) {
        throw std::runtime_error("benchmark_histogram_primitives: failed to read proof: " + path);
    }

    SerializedProof proof;
    proof.bytes.assign(bytes.begin(), bytes.end());
    return proof;
}

HistogramWitness make_histogram_witness(size_t n, size_t m, size_t proof_n, size_t proof_m)
{
    if (!is_power_of_two(n) || !is_power_of_two(m) ||
        !is_power_of_two(proof_n) || !is_power_of_two(proof_m)) {
        throw std::runtime_error("benchmark_histogram_primitives: n, m, proof_n, and proof_m must be powers of two");
    }
    if (proof_n < n || proof_m < m) {
        throw std::runtime_error("benchmark_histogram_primitives: proof shape must cover logical n and m");
    }

    std::vector<Fr> indices(proof_n, Fr(0));
    std::vector<Fr> values(proof_n, Fr(0));
    std::vector<Fr> sums(proof_m, Fr(0));

    for (size_t row = 0; row < n; ++row) {
        const size_t bucket = (row * 131 + row / 7 + 19) % m;
        const size_t value = 1 + ((row * 17 + 5) % 23);
        indices.at(row) = field_from_size_t(bucket);
        values.at(row) = field_from_size_t(value);
        sums.at(bucket) += field_from_size_t(value);
    }

    HistogramWitness witness;
    witness.indices = poly_from_evaluations(indices);
    witness.values = poly_from_evaluations(values);
    witness.sums = poly_from_evaluations(sums);
    return witness;
}

void add_histogram_check(ProofProgram& program,
                         Route route,
                         const virCommitment& indices,
                         const virCommitment& values,
                         const virCommitment& sums,
                         size_t n,
                         size_t m)
{
    if (route == Route::Terrae) {
        program.hist_agg().add_check(indices, values, sums, n, m);
        return;
    }
    if (route == Route::Ram) {
        program.zkboost_histogram_ram().add_check(indices, values, sums, n, m);
        return;
    }
    add_zkboost_weighted_logup_histogram_to_program(program, indices, values, sums, n, m);
}

BenchmarkResult run_once(const kzgPublicParameters& public_params,
                         Route route,
                         const HistogramWitness& witness,
                         const Scenario& scenario,
                         size_t proof_n,
                         size_t proof_m,
                         const std::string& proof_dir,
                         bool verifier_only)
{
    BenchmarkResult result;
    SerializedProof proof;

    const std::string proof_path =
        proof_dir.empty() ? std::string() : join_path(proof_dir, proof_filename(route, scenario, proof_n, proof_m));
    if (verifier_only) {
        if (proof_path.empty()) {
            throw std::runtime_error("benchmark_histogram_primitives: --verifier-only requires --proof-dir");
        }
        proof = read_proof_file(proof_path);
        result.prover_seconds = 0.0;
    } else {
        ProofRuntime prover_runtime = ProofRuntime::prover(public_params);
        ProofProgram prover_program(prover_runtime);
        result.prover_seconds = seconds_for([&] {
            const virCommitment indices = prover_program.bind_virtual(
                WitnessCommitmentInput::prover("histogram.indices", witness.indices, proof_n - 1));
            const virCommitment values = prover_program.bind_virtual(
                WitnessCommitmentInput::prover("histogram.values", witness.values, proof_n - 1));
            const virCommitment sums = prover_program.bind_virtual(
                WitnessCommitmentInput::prover("histogram.sums", witness.sums, proof_m - 1));
            add_histogram_check(prover_program, route, indices, values, sums, proof_n, proof_m);
            if (!prover_program.finalize()) {
                throw std::runtime_error("benchmark_histogram_primitives: prover proof rejected");
            }
            proof = prover_runtime.take_proof();
        });
        if (!proof_path.empty()) {
            write_proof_file(proof_path, proof);
            std::cerr << "proof_write\t" << proof_path << "\tbytes\t" << proof.bytes.size() << "\n";
        }
    }
    result.proof_bytes = proof.bytes.size();

    ProofRuntime verifier_runtime = ProofRuntime::verifier(public_params, proof);
    ProofProgram verifier_program(verifier_runtime);
    result.verifier_prepare_seconds = seconds_for([&] {
        const virCommitment indices = verifier_program.bind_virtual(
            WitnessCommitmentInput::verifier("histogram.indices", proof_n - 1));
        const virCommitment values = verifier_program.bind_virtual(
            WitnessCommitmentInput::verifier("histogram.values", proof_n - 1));
        const virCommitment sums = verifier_program.bind_virtual(
            WitnessCommitmentInput::verifier("histogram.sums", proof_m - 1));
        add_histogram_check(verifier_program, route, indices, values, sums, proof_n, proof_m);
        if (!verifier_program.prepare_verifier_preprocessing()) {
            throw std::runtime_error("benchmark_histogram_primitives: verifier preprocessing rejected proof");
        }
    });
    result.verifier_seconds = seconds_for([&] {
        result.accepted = verifier_program.verify_preprocessed();
        result.proof_fully_consumed = verifier_runtime.proof_fully_consumed();
    });

    return result;
}

BenchmarkResult run_average(const kzgPublicParameters& public_params,
                            Route route,
                            const HistogramWitness& witness,
                            const Scenario& scenario,
                            size_t proof_n,
                            size_t proof_m,
                            size_t repeats,
                            const std::string& proof_dir,
                            bool verifier_only)
{
    BenchmarkResult total;
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        const BenchmarkResult next =
            run_once(public_params, route, witness, scenario, proof_n, proof_m, proof_dir, verifier_only);
        total.prover_seconds += next.prover_seconds;
        total.verifier_prepare_seconds += next.verifier_prepare_seconds;
        total.verifier_seconds += next.verifier_seconds;
        total.proof_bytes = next.proof_bytes;
        total.accepted = next.accepted;
        total.proof_fully_consumed = next.proof_fully_consumed;
        if (!next.accepted || !next.proof_fully_consumed) {
            throw std::runtime_error("benchmark_histogram_primitives: verifier rejected proof");
        }
    }
    total.prover_seconds /= static_cast<double>(repeats);
    total.verifier_prepare_seconds /= static_cast<double>(repeats);
    total.verifier_seconds /= static_cast<double>(repeats);
    return total;
}

std::vector<Route> parse_route(const std::string& value)
{
    if (value == "both") {
        return {Route::Terrae, Route::Ram};
    }
    if (value == "terrae") {
        return {Route::Terrae};
    }
    if (value == "ram") {
        return {Route::Ram};
    }
    if (value == "weighted-logup") {
        return {Route::WeightedLogup};
    }
    if (value == "all") {
        return {Route::Terrae, Route::Ram, Route::WeightedLogup};
    }
    throw std::runtime_error(
        "benchmark_histogram_primitives: --route must be both, all, terrae, ram, or weighted-logup");
}

size_t parse_size_arg(const std::string& text, const char* flag)
{
    size_t parsed = 0;
    try {
        parsed = static_cast<size_t>(std::stoull(text));
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("benchmark_histogram_primitives: invalid ") + flag);
    }
    if (parsed == 0) {
        throw std::runtime_error(std::string("benchmark_histogram_primitives: ") + flag + " must be non-zero");
    }
    return parsed;
}

Scenario parse_case_arg(const std::string& text)
{
    const size_t separator = text.find(':');
    if (separator == std::string::npos || separator == 0 || separator + 1 == text.size()) {
        throw std::runtime_error("benchmark_histogram_primitives: --case must have form n:m");
    }

    Scenario scenario;
    scenario.n = parse_size_arg(text.substr(0, separator), "--case n");
    scenario.m = parse_size_arg(text.substr(separator + 1), "--case m");
    if (!is_power_of_two(scenario.n) || !is_power_of_two(scenario.m)) {
        throw std::runtime_error("benchmark_histogram_primitives: --case n and m must be powers of two");
    }
    return scenario;
}

void print_usage(const char* argv0)
{
    std::cerr << "usage: " << argv0
              << " [--route both|all|terrae|ram|weighted-logup] [--repeats N] [--srs PATH] [--srs-size N]"
              << " [--srs-cache PATH] [--proof-n N --proof-m M] [--proof-dir DIR]"
              << " [--verifier-only] [--case n:m ...]\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::string route_arg = "both";
        std::string srs_path;
        std::string srs_cache_path;
        std::string proof_dir;
        size_t repeats = 1;
        size_t srs_size = 131072;
        size_t fixed_proof_n = 0;
        size_t fixed_proof_m = 0;
        bool verifier_only = false;
        std::vector<Scenario> scenarios;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&](const char* flag) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("benchmark_histogram_primitives: missing value for ") + flag);
                }
                return argv[++i];
            };

            if (arg == "--route") {
                route_arg = require_value("--route");
            } else if (arg == "--repeats") {
                repeats = parse_size_arg(require_value("--repeats"), "--repeats");
            } else if (arg == "--srs") {
                srs_path = require_value("--srs");
            } else if (arg == "--srs-cache") {
                srs_cache_path = require_value("--srs-cache");
            } else if (arg == "--srs-size") {
                srs_size = parse_size_arg(require_value("--srs-size"), "--srs-size");
            } else if (arg == "--proof-n") {
                fixed_proof_n = parse_size_arg(require_value("--proof-n"), "--proof-n");
            } else if (arg == "--proof-m") {
                fixed_proof_m = parse_size_arg(require_value("--proof-m"), "--proof-m");
            } else if (arg == "--proof-dir") {
                proof_dir = require_value("--proof-dir");
            } else if (arg == "--verifier-only") {
                verifier_only = true;
            } else if (arg == "--case") {
                scenarios.push_back(parse_case_arg(require_value("--case")));
            } else if (arg == "--help") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("benchmark_histogram_primitives: unknown argument " + arg);
            }
        }

        init_groups();

        if (scenarios.empty()) {
            scenarios = {
                {512, 256},
                {1024, 256},
                {2048, 256},
                {1024, 512},
                {1024, 1024},
            };
        }
        const std::vector<Route> routes = parse_route(route_arg);
        if (!srs_path.empty() && !srs_cache_path.empty()) {
            throw std::runtime_error("benchmark_histogram_primitives: --srs and --srs-cache are mutually exclusive");
        }
        if (verifier_only && proof_dir.empty()) {
            throw std::runtime_error("benchmark_histogram_primitives: --verifier-only requires --proof-dir");
        }

        if ((fixed_proof_n == 0) != (fixed_proof_m == 0)) {
            throw std::runtime_error("benchmark_histogram_primitives: --proof-n and --proof-m must be provided together");
        }
        if (fixed_proof_n != 0 &&
            (!is_power_of_two(fixed_proof_n) || !is_power_of_two(fixed_proof_m))) {
            throw std::runtime_error("benchmark_histogram_primitives: proof shape must use powers of two");
        }

        size_t max_required_domain = 1;
        for (const Scenario& scenario : scenarios) {
            const size_t proof_n = fixed_proof_n == 0 ? scenario.n : fixed_proof_n;
            const size_t proof_m = fixed_proof_m == 0 ? scenario.m : fixed_proof_m;
            if (proof_n < scenario.n || proof_m < scenario.m) {
                throw std::runtime_error("benchmark_histogram_primitives: fixed proof shape does not cover all cases");
            }
            max_required_domain = std::max(max_required_domain, proof_n);
            max_required_domain = std::max(max_required_domain, proof_m);
            if (std::find(routes.begin(), routes.end(), Route::Ram) != routes.end()) {
                max_required_domain =
                    std::max(max_required_domain,
                             zkboost_histogram_ram_shape(proof_n, proof_m).trace_domain_size);
            }
        }
        max_required_domain = std::max(max_required_domain, size_t(1) << 16);
        srs_size = std::max(srs_size, checked_mul(max_required_domain, 4, "benchmark_histogram_primitives"));

        kzgPublicParameters public_params;
        const double srs_seconds = seconds_for([&] {
            if (!srs_path.empty()) {
                public_params = load_kzg_public_parameters(srs_path, srs_size, srs_size);
                return;
            }
            if (!srs_cache_path.empty() && file_exists(srs_cache_path)) {
                public_params = load_kzg_public_parameters(srs_cache_path, srs_size, srs_size);
                return;
            }
            public_params = generate_kzg_public_parameters(srs_size, srs_size);
            if (!srs_cache_path.empty()) {
                write_kzg_public_parameters(public_params, srs_cache_path);
                std::cerr << "srs_write\t" << srs_cache_path << "\tsrs_size\t" << srs_size << "\n";
            }
        });

        std::cerr << "srs_seconds\t" << std::fixed << std::setprecision(6) << srs_seconds
                  << "\tsrs_size\t" << srs_size
                  << "\tverifier_only\t" << (verifier_only ? 1 : 0) << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "route\tn\tm\tproof_n\tproof_m\tram_trace_domain\tproof_bytes\tprover_seconds\tverifier_seconds\taccepted\tconsumed\n";
        std::cout.flush();

        const size_t total_jobs = checked_mul(scenarios.size(), routes.size(), "benchmark_histogram_primitives");
        size_t current_job = 0;
        for (const Scenario& scenario : scenarios) {
            const size_t proof_n = fixed_proof_n == 0 ? scenario.n : fixed_proof_n;
            const size_t proof_m = fixed_proof_m == 0 ? scenario.m : fixed_proof_m;
            const HistogramWitness witness = make_histogram_witness(scenario.n, scenario.m, proof_n, proof_m);
            const size_t ram_trace_domain =
                zkboost_histogram_ram_shape(proof_n, proof_m).trace_domain_size;

            for (Route route : routes) {
                ++current_job;
                std::cerr << "progress_start\t" << current_job << '/' << total_jobs
                          << "\troute\t" << route_name(route)
                          << "\tn\t" << scenario.n
                          << "\tm\t" << scenario.m
                          << "\tproof_n\t" << proof_n
                          << "\tproof_m\t" << proof_m
                          << "\tram_trace_domain\t" << ram_trace_domain
                          << "\trepeats\t" << repeats << "\n";
                const BenchmarkResult result =
                    run_average(public_params, route, witness, scenario, proof_n, proof_m, repeats, proof_dir, verifier_only);
                std::cout << route_name(route) << '\t'
                          << scenario.n << '\t'
                          << scenario.m << '\t'
                          << proof_n << '\t'
                          << proof_m << '\t'
                          << ram_trace_domain << '\t'
                          << result.proof_bytes << '\t'
                          << result.prover_seconds << '\t'
                          << result.verifier_seconds << '\t'
                          << (result.accepted ? 1 : 0) << '\t'
                          << (result.proof_fully_consumed ? 1 : 0) << '\n';
                std::cout.flush();
                std::cerr << "progress_done\t" << current_job << '/' << total_jobs
                          << "\troute\t" << route_name(route)
                          << "\tn\t" << scenario.n
                          << "\tm\t" << scenario.m
                          << "\tproof_n\t" << proof_n
                          << "\tproof_m\t" << proof_m
                          << "\tram_trace_domain\t" << ram_trace_domain
                          << "\tprover_seconds\t" << result.prover_seconds
                          << "\tverifier_prepare_seconds\t" << result.verifier_prepare_seconds
                          << "\tverifier_seconds\t" << result.verifier_seconds
                          << "\taccepted\t" << (result.accepted ? 1 : 0)
                          << "\tconsumed\t" << (result.proof_fully_consumed ? 1 : 0)
                          << "\n";
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
