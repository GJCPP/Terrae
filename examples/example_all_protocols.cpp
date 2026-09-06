#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "cube_sum.hpp"
#include "matrix_sum.hpp"
#include "proof_runtime.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

constexpr size_t kDeg1Plus1 = 1024;
constexpr size_t kDeg2Plus1 = 1024;

constexpr size_t kZeroDomainSize = 4;
constexpr size_t kSumDomainSize = 4;
constexpr size_t kMatrixRows = 2;
constexpr size_t kMatrixCols = 4;
constexpr size_t kLookupQuerySize = 4;
constexpr size_t kLookupTableSize = 8;
constexpr size_t kTableRows = 4;
constexpr size_t kTableCols = 2;
constexpr size_t kSelectedRows = 2;
constexpr size_t kCubeOuter = 2;
constexpr size_t kCubeMiddle = 2;
constexpr size_t kCubeInner = 2;

const char* const kRowPrefixKey = "all.matrix.row_prefix";
const char* const kColumnPrefixKey = "all.matrix.column_prefix";
const char* const kRowSumKey = "all.matrix.row_sum";
const char* const kColumnSumKey = "all.matrix.column_sum";
const char* const kLookupTableKey = "all.lookup.table";
const char* const kMatrixTableKey = "all.matrix.table";
const char* const kCqLookupTableKey = "all.cq.lookup.table";
const char* const kCqMatrixTableKey = "all.cq.matrix.table";
const char* const kCubeSumKey = "all.cube.sum";

poly poly_from_domain_ints(const std::vector<int>& values)
{
    std::vector<Fr> evals;
    evals.reserve(values.size());
    for (int value : values) {
        evals.push_back(fr_from_int(value));
    }
    return poly_from_evals(evals);
}

poly matrix_poly_from_rows(const std::vector<std::vector<int>>& rows)
{
    std::vector<Fr> evals;
    for (const auto& row : rows) {
        for (int value : row) {
            evals.push_back(fr_from_int(value));
        }
    }
    return poly_from_evals(evals);
}

poly cube_poly_from_slices(const std::vector<std::vector<std::vector<int>>>& cube)
{
    std::vector<Fr> evals;
    for (const auto& outer : cube) {
        for (const auto& middle : outer) {
            for (int value : middle) {
                evals.push_back(fr_from_int(value));
            }
        }
    }
    return poly_from_evals(evals);
}

Fr example_sum_claim()
{
    return fr_from_int(10);
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

struct ExampleScenario {
    poly zero_witness;
    poly sum_witness;
    poly matrix;
    poly row_prefix_claim;
    poly column_prefix_claim;
    poly row_sum_claim;
    poly column_sum_claim;
    poly lookup_query;
    poly lookup_table;
    poly selected_rows;
    poly matrix_table;
    poly cube;
    poly cube_sum_claim;
};

struct ExamplePublicArtifacts {
    Group1 row_prefix_commitment;
    Group1 column_prefix_commitment;
    Group1 row_sum_commitment;
    Group1 column_sum_commitment;
    Group1 lookup_table_commitment;
    Group1 matrix_table_commitment;
    Group1 cube_sum_commitment;
    Group2 cq_lookup_table_g2;
    Group2 cq_matrix_table_g2;
};

struct ExampleBundle {
    ExamplePublicArtifacts public_artifacts;
    SerializedProof proof;
};

ExampleScenario build_scenario()
{
    ExampleScenario scenario;
    scenario.zero_witness = vanishing_poly(kZeroDomainSize);
    scenario.sum_witness = poly_from_domain_ints({1, 2, 3, 4});
    scenario.matrix = matrix_poly_from_rows({
        {1, 2, 3, 4},
        {5, 6, 7, 8},
    });
    scenario.row_prefix_claim = matrix_row_prefix_sum_polynomial(scenario.matrix, kMatrixRows, kMatrixCols);
    scenario.column_prefix_claim = matrix_column_prefix_sum_polynomial(scenario.matrix, kMatrixRows, kMatrixCols);
    scenario.row_sum_claim = matrix_row_sum_polynomial(scenario.matrix, kMatrixRows, kMatrixCols);
    scenario.column_sum_claim = matrix_column_sum_polynomial(scenario.matrix, kMatrixRows, kMatrixCols);
    scenario.lookup_query = poly_from_domain_ints({5, 7, 7, 11});
    scenario.lookup_table = poly_from_domain_ints({3, 5, 7, 11, 13, 17, 19, 23});
    scenario.selected_rows = matrix_poly_from_rows({
        {7, 11},
        {19, 23},
    });
    scenario.matrix_table = matrix_poly_from_rows({
        {3, 5},
        {7, 11},
        {13, 17},
        {19, 23},
    });
    scenario.cube = cube_poly_from_slices({
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8},
        },
    });
    scenario.cube_sum_claim = cube_middle_sum_polynomial(scenario.cube, kCubeOuter, kCubeMiddle, kCubeInner);
    return scenario;
}

ExamplePublicArtifacts build_public_artifacts(const kzgPublicParameters& public_params, const ExampleScenario& scenario)
{
    kzgProtocol kzg;
    kzg.setup(public_params);

    ExamplePublicArtifacts artifacts;
    artifacts.row_prefix_commitment = kzg.commit_public(kRowPrefixKey, scenario.row_prefix_claim, 7).com;
    artifacts.column_prefix_commitment = kzg.commit_public(kColumnPrefixKey, scenario.column_prefix_claim, 7).com;
    artifacts.row_sum_commitment = kzg.commit_public(kRowSumKey, scenario.row_sum_claim, 1).com;
    artifacts.column_sum_commitment = kzg.commit_public(kColumnSumKey, scenario.column_sum_claim, 3).com;
    artifacts.lookup_table_commitment = kzg.commit_public(kLookupTableKey, scenario.lookup_table, 7).com;
    artifacts.matrix_table_commitment = kzg.commit_public(kMatrixTableKey, scenario.matrix_table, 7).com;
    artifacts.cube_sum_commitment = kzg.commit_public(kCubeSumKey, scenario.cube_sum_claim, 3).com;
    artifacts.cq_lookup_table_g2 = kzg.commit_g2(scenario.lookup_table);
    artifacts.cq_matrix_table_g2 = kzg.commit_g2(scenario.matrix_table);
    return artifacts;
}

void write_bundle(const std::string& path, const ExampleBundle& bundle)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    out << "row_prefix " << proof_encoding(bundle.public_artifacts.row_prefix_commitment) << "\n";
    out << "column_prefix " << proof_encoding(bundle.public_artifacts.column_prefix_commitment) << "\n";
    out << "row_sum " << proof_encoding(bundle.public_artifacts.row_sum_commitment) << "\n";
    out << "column_sum " << proof_encoding(bundle.public_artifacts.column_sum_commitment) << "\n";
    out << "lookup_table " << proof_encoding(bundle.public_artifacts.lookup_table_commitment) << "\n";
    out << "matrix_table " << proof_encoding(bundle.public_artifacts.matrix_table_commitment) << "\n";
    out << "cube_sum " << proof_encoding(bundle.public_artifacts.cube_sum_commitment) << "\n";
    out << "cq_lookup_table_g2 " << proof_encoding(bundle.public_artifacts.cq_lookup_table_g2) << "\n";
    out << "cq_matrix_table_g2 " << proof_encoding(bundle.public_artifacts.cq_matrix_table_g2) << "\n";
    out << "proof " << bytes_to_hex(bundle.proof.bytes) << "\n";
}

std::string required_field(const std::map<std::string, std::string>& fields, const std::string& key)
{
    const auto it = fields.find(key);
    if (it == fields.end()) {
        throw std::runtime_error("missing field in bundle: " + key);
    }
    return it->second;
}

ExampleBundle read_bundle(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path);
    }

    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream line_stream(line);
        std::string key;
        std::string value;
        line_stream >> key >> value;
        if (key.empty() || value.empty()) {
            throw std::runtime_error("invalid bundle line: " + line);
        }
        fields[key] = value;
    }

    ExampleBundle bundle;
    bundle.public_artifacts.row_prefix_commitment = decode_group1(required_field(fields, "row_prefix"));
    bundle.public_artifacts.column_prefix_commitment = decode_group1(required_field(fields, "column_prefix"));
    bundle.public_artifacts.row_sum_commitment = decode_group1(required_field(fields, "row_sum"));
    bundle.public_artifacts.column_sum_commitment = decode_group1(required_field(fields, "column_sum"));
    bundle.public_artifacts.lookup_table_commitment = decode_group1(required_field(fields, "lookup_table"));
    bundle.public_artifacts.matrix_table_commitment = decode_group1(required_field(fields, "matrix_table"));
    bundle.public_artifacts.cube_sum_commitment = decode_group1(required_field(fields, "cube_sum"));
    bundle.public_artifacts.cq_lookup_table_g2 = decode_group2(required_field(fields, "cq_lookup_table_g2"));
    bundle.public_artifacts.cq_matrix_table_g2 = decode_group2(required_field(fields, "cq_matrix_table_g2"));

    bundle.proof.bytes = hex_to_bytes(required_field(fields, "proof"));
    return bundle;
}

struct ExampleProtocolInputs {
    WitnessCommitmentInput zero_witness;
    WitnessCommitmentInput sum_witness;
    WitnessCommitmentInput prefix_matrix;
    ProtocolPolynomialInput row_prefix_claim;
    ProtocolPolynomialInput column_prefix_claim;
    WitnessCommitmentInput lookup_query;
    ProtocolPolynomialInput lookup_table;
    WitnessCommitmentInput cq_lookup_query;
    CqLookupPublicInput cq_lookup_table;
    WitnessCommitmentInput table_lookup_query;
    ProtocolPolynomialInput matrix_table;
    WitnessCommitmentInput cq_table_lookup_query;
    CqTableLookupPublicInput cq_matrix_table;
    WitnessCommitmentInput row_sum_matrix;
    ProtocolPolynomialInput row_sum_claim;
    WitnessCommitmentInput column_sum_matrix;
    ProtocolPolynomialInput column_sum_claim;
    WitnessCommitmentInput cube_input;
    ProtocolPolynomialInput cube_sum_claim;
};

ExampleProtocolInputs build_prover_inputs(const ExampleScenario& scenario, const kzgPublicParameters& public_params)
{
    return ExampleProtocolInputs{
        WitnessCommitmentInput::prover("zero.witness", scenario.zero_witness, 4),
        WitnessCommitmentInput::prover("sum.witness", scenario.sum_witness, 3),
        WitnessCommitmentInput::prover("prefix.matrix", scenario.matrix, 7),
        PublicCommitmentInput::prover(kRowPrefixKey, scenario.row_prefix_claim, 7),
        PublicCommitmentInput::prover(kColumnPrefixKey, scenario.column_prefix_claim, 7),
        WitnessCommitmentInput::prover("lookup.query", scenario.lookup_query, 3),
        PublicCommitmentInput::prover(kLookupTableKey, scenario.lookup_table, 7),
        WitnessCommitmentInput::prover("cq.lookup.query", scenario.lookup_query, 3),
        build_cq_lookup_public_table(public_params, kCqLookupTableKey, scenario.lookup_table, kLookupTableSize, 7),
        WitnessCommitmentInput::prover("table_lookup.query", scenario.selected_rows, 3),
        PublicCommitmentInput::prover(kMatrixTableKey, scenario.matrix_table, 7),
        WitnessCommitmentInput::prover("cq.table_lookup.query", scenario.selected_rows, 3),
        build_cq_table_lookup_public_table(public_params, kCqMatrixTableKey, scenario.matrix_table, kTableRows, kTableCols, 7),
        WitnessCommitmentInput::prover("row_sum.matrix", scenario.matrix, 7),
        PublicCommitmentInput::prover(kRowSumKey, scenario.row_sum_claim, 1),
        WitnessCommitmentInput::prover("column_sum.matrix", scenario.matrix, 7),
        PublicCommitmentInput::prover(kColumnSumKey, scenario.column_sum_claim, 3),
        WitnessCommitmentInput::prover("cube.input", scenario.cube, 7),
        PublicCommitmentInput::prover(kCubeSumKey, scenario.cube_sum_claim, 3)
    };
}

ExampleProtocolInputs build_verifier_inputs(const ExampleBundle& bundle)
{
    cqLookupPublicTable cq_lookup_table;
    cq_lookup_table.key = kCqLookupTableKey;
    cq_lookup_table.domain_size = kLookupTableSize;
    cq_lookup_table.degree_bound = 7;
    cq_lookup_table.commitment = bundle.public_artifacts.lookup_table_commitment;
    cq_lookup_table.table_g2 = bundle.public_artifacts.cq_lookup_table_g2;

    cqTableLookupPublicTable cq_table_lookup_table;
    cq_table_lookup_table.key = kCqMatrixTableKey;
    cq_table_lookup_table.row_count = kTableRows;
    cq_table_lookup_table.column_count = kTableCols;
    cq_table_lookup_table.domain_size = kTableRows * kTableCols;
    cq_table_lookup_table.degree_bound = 7;
    cq_table_lookup_table.commitment = bundle.public_artifacts.matrix_table_commitment;
    cq_table_lookup_table.table_g2 = bundle.public_artifacts.cq_matrix_table_g2;
    return ExampleProtocolInputs{
        WitnessCommitmentInput::verifier("zero.witness", 4),
        WitnessCommitmentInput::verifier("sum.witness", 3),
        WitnessCommitmentInput::verifier("prefix.matrix", 7),
        PublicCommitmentInput::verifier(kRowPrefixKey, bundle.public_artifacts.row_prefix_commitment, 7),
        PublicCommitmentInput::verifier(kColumnPrefixKey, bundle.public_artifacts.column_prefix_commitment, 7),
        WitnessCommitmentInput::verifier("lookup.query", 3),
        PublicCommitmentInput::verifier(kLookupTableKey, bundle.public_artifacts.lookup_table_commitment, 7),
        WitnessCommitmentInput::verifier("cq.lookup.query", 3),
        cq_lookup_table,
        WitnessCommitmentInput::verifier("table_lookup.query", 3),
        PublicCommitmentInput::verifier(kMatrixTableKey, bundle.public_artifacts.matrix_table_commitment, 7),
        WitnessCommitmentInput::verifier("cq.table_lookup.query", 3),
        cq_table_lookup_table,
        WitnessCommitmentInput::verifier("row_sum.matrix", 7),
        PublicCommitmentInput::verifier(kRowSumKey, bundle.public_artifacts.row_sum_commitment, 1),
        WitnessCommitmentInput::verifier("column_sum.matrix", 7),
        PublicCommitmentInput::verifier(kColumnSumKey, bundle.public_artifacts.column_sum_commitment, 3),
        WitnessCommitmentInput::verifier("cube.input", 7),
        PublicCommitmentInput::verifier(kCubeSumKey, bundle.public_artifacts.cube_sum_commitment, 3)
    };
}

bool run_all_protocols_program(ProofRuntime& runtime, const ExampleProtocolInputs& input)
{
    return run_proof_program(runtime, [&](ProofProgram& program) {
        virCommitment zero_witness = program.bind_virtual(input.zero_witness);
        program.zero().add_check(zero_witness, kZeroDomainSize);

        virCommitment sum_witness = program.bind_virtual(input.sum_witness);
        program.sum().add_check(sum_witness, example_sum_claim(), kSumDomainSize);

        program.row_prefix().add_check(program.bind_virtual(input.prefix_matrix),
                                       program.bind_virtual(input.row_prefix_claim),
                                       kMatrixRows,
                                       kMatrixCols);

        program.column_prefix().add_check(program.bind_virtual(input.prefix_matrix),
                                          program.bind_virtual(input.column_prefix_claim),
                                          kMatrixRows,
                                          kMatrixCols);

        program.lookup().add_check(
            program.bind_virtual(input.lookup_query),
            program.bind_virtual(input.lookup_table),
            kLookupQuerySize,
            kLookupTableSize);

        std::visit(
            [&](const auto& table_input) {
                using TableInput = std::decay_t<decltype(table_input)>;
                auto& context = [&]() -> auto& {
                    if constexpr (std::is_same_v<TableInput, PublicPolynomialInput>) {
                        return program.create_cq_lookup_context(table_input, kLookupTableSize);
                    } else {
                        return program.create_cq_lookup_context(table_input);
                    }
                }();
                context.add_check(input.cq_lookup_query, kLookupQuerySize);
            },
            input.cq_lookup_table);

        program.table_lookup().add_check(
            program.bind_virtual(input.table_lookup_query),
            program.bind_virtual(input.matrix_table),
            kSelectedRows,
            kTableRows,
            kTableCols);

        std::visit(
            [&](const auto& table_input) {
                using TableInput = std::decay_t<decltype(table_input)>;
                auto& context = [&]() -> auto& {
                    if constexpr (std::is_same_v<TableInput, PublicPolynomialInput>) {
                        return program.create_cq_table_lookup_context(table_input, kTableRows, kTableCols);
                    } else {
                        return program.create_cq_table_lookup_context(table_input);
                    }
                }();
                context.add_check(input.cq_table_lookup_query, kSelectedRows);
            },
            input.cq_matrix_table);

        program.row_sum().add_check(
            program.bind_virtual(input.row_sum_matrix),
            program.bind_virtual(input.row_sum_claim),
            kMatrixRows,
            kMatrixCols);

        program.column_sum().add_check(
            program.bind_virtual(input.column_sum_matrix),
            program.bind_virtual(input.column_sum_claim),
            kMatrixRows,
            kMatrixCols);

        program.cube_sum().add_check(
            program.bind_virtual(input.cube_input),
            program.bind_virtual(input.cube_sum_claim),
            kCubeOuter,
            kCubeMiddle,
            kCubeInner);
    });
}

void print_usage(const char* argv0)
{
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv0 << " prover <srs-path> <bundle-path>\n";
    std::cerr << "  " << argv0 << " verifier <srs-path> <bundle-path>\n";
}

int run_prover(const std::string& srs_path, const std::string& bundle_path)
{
    const ExampleScenario scenario = build_scenario();

    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);
    const ExamplePublicArtifacts public_artifacts = build_public_artifacts(public_params, scenario);
    const ExampleProtocolInputs input = build_prover_inputs(scenario, public_params);

    ExampleBundle bundle;
    bundle.public_artifacts = public_artifacts;
    bundle.proof =
        prove_with_runtime(public_params, [&](ProofRuntime& runtime) { return run_all_protocols_program(runtime, input); })
            .proof;
    write_bundle(bundle_path, bundle);

    std::cout << "All-protocol prover example\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << "  unified proof bytes: " << bundle.proof.bytes.size() << "\n";
    std::cout << "  verifier public data: commitments/artifacts only\n";
    return 0;
}

int run_verifier(const std::string& srs_path, const std::string& bundle_path)
{
    const ExampleBundle bundle = read_bundle(bundle_path);
    const kzgPublicParameters public_params = load_kzg_public_parameters(srs_path);

    const ExampleProtocolInputs input = build_verifier_inputs(bundle);
    const VerificationResult result =
        verify_with_runtime(public_params,
                            make_proof_artifact(bundle.proof),
                            [&](ProofRuntime& runtime) { return run_all_protocols_program(runtime, input); });

    std::cout << "All-protocol verifier example\n";
    std::cout << "  bundle: " << bundle_path << "\n";
    std::cout << "  unified program: " << (result.accepted ? "ok" : "fail") << "\n";
    std::cout << "  proof fully consumed: " << (result.proof_fully_consumed ? "yes" : "no") << "\n";
    std::cout << "  verifier loaded raw public polynomials: no\n";
    return (result.accepted && result.proof_fully_consumed) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    init_groups();

    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    const std::string srs_path = argv[2];
    const std::string bundle_path = argv[3];

    try {
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
