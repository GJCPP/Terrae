#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "proof_runtime.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

poly poly_from_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

struct Inputs {
    WitnessCommitmentInput private_table;
    WitnessCommitmentInput first_query;
    WitnessCommitmentInput second_query;
};

struct Bundle {
    SerializedProof proof;
};

bool run_private_table_lookup_program(ProofRuntime& runtime, const Inputs& inputs)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        // The table is private, so both sides work from its commitment rather than raw values.
        const virCommitment table_commitment = program.bind_virtual(inputs.private_table);

        // Each query is also private. We can prove multiple lookups against the same table
        // inside one proof by reusing the committed table expression.
        program.lookup().add_check(program.bind_virtual(inputs.first_query), table_commitment, 4, 8);
        program.lookup().add_check(program.bind_virtual(inputs.second_query), table_commitment, 4, 8);
    });
}

std::string bytes_to_hex(const std::vector<uint8_t>& bytes)
{
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (uint8_t byte : bytes) {
        out.push_back(kHex[(byte >> 4u) & 0x0fu]);
        out.push_back(kHex[byte & 0x0fu]);
    }
    return out;
}

uint8_t hex_nibble(char c)
{
    if (c >= '0' && c <= '9') {
        return static_cast<uint8_t>(c - '0');
    }
    if (c >= 'a' && c <= 'f') {
        return static_cast<uint8_t>(10 + c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return static_cast<uint8_t>(10 + c - 'A');
    }
    throw std::runtime_error("invalid hex digit");
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex)
{
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("hex string must have even length");
    }

    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        bytes.push_back(static_cast<uint8_t>((hex_nibble(hex[i]) << 4u) | hex_nibble(hex[i + 1])));
    }
    return bytes;
}

void write_bundle(const std::string& path, const Bundle& bundle)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    out << "proof " << bytes_to_hex(bundle.proof.bytes) << "\n";
}

Bundle read_bundle(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path);
    }

    std::string proof_hex;

    std::string key;
    std::string value;
    while (in >> key >> value) {
        if (key == "proof") {
            proof_hex = value;
        }
    }

    if (proof_hex.empty()) {
        throw std::runtime_error("invalid bundle");
    }

    Bundle bundle;
    bundle.proof.bytes = hex_to_bytes(proof_hex);
    return bundle;
}

void print_usage(const char* argv0)
{
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv0 << " prover <srs-path> <bundle-path>\n";
    std::cerr << "  " << argv0 << " verifier <srs-path> <bundle-path>\n";
}

int run_prover(const std::string& srs_path, const std::string& bundle_path)
{
    // The prover keeps the table secret.
    const poly private_table = poly_from_ints({3, 5, 7, 11, 13, 17, 19, 23});
    const poly first_query = poly_from_ints({5, 7, 7, 11});
    const poly second_query = poly_from_ints({3, 13, 19, 23});

    const Inputs prover_inputs{
        WitnessCommitmentInput::prover("private.table", private_table, 7),
        WitnessCommitmentInput::prover("query.one", first_query, 3),
        WitnessCommitmentInput::prover("query.two", second_query, 3),
    };

    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    ProofRuntime prover = ProofRuntime::prover(public_params);
    const bool prover_ok = run_private_table_lookup_program(prover, prover_inputs);

    const Bundle bundle{
        prover.take_proof(),
    };
    write_bundle(bundle_path, bundle);

    std::cout << "Private table lookup example\n";
    std::cout << "  mode: prover\n";
    std::cout << "  result: " << (prover_ok ? "success" : "failure") << "\n";
    std::cout << "  proof size: " << bundle.proof.bytes.size() << " bytes\n";
    return prover_ok ? 0 : 1;
}

int run_verifier(const std::string& srs_path, const std::string& bundle_path)
{
    const Bundle bundle = read_bundle(bundle_path);
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);

    const Inputs verifier_inputs{
        WitnessCommitmentInput::verifier("private.table", 7),
        WitnessCommitmentInput::verifier("query.one", 3),
        WitnessCommitmentInput::verifier("query.two", 3),
    };

    ProofRuntime verifier = ProofRuntime::verifier(public_params, bundle.proof);
    const bool verifier_ok = run_private_table_lookup_program(verifier, verifier_inputs);

    std::cout << "Private table lookup example\n";
    std::cout << "  mode: verifier\n";
    std::cout << "  result: " << (verifier_ok ? "success" : "failure") << "\n";
    std::cout << "  proof fully consumed: " << (verifier.proof_fully_consumed() ? "yes" : "no") << "\n";
    return (verifier_ok && verifier.proof_fully_consumed()) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    init_groups();

    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::string mode = argv[1];
        const std::string srs_path = argv[2];
        const std::string bundle_path = argv[3];
        if (mode == "prover") {
            return run_prover(srs_path, bundle_path);
        }
        if (mode == "verifier") {
            return run_verifier(srs_path, bundle_path);
        }
        print_usage(argv[0]);
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
