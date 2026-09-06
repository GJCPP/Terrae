#include "bls12_381_wrapper.hpp"
#include "kzg.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* argv0)
{
    std::cerr << "Usage:\n";
    std::cerr << "  " << argv0 << " <output-path> <deg1-plus-1> <deg2-plus-1>\n";
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
        const std::string output_path = argv[1];
        const size_t deg1_plus_1 = static_cast<size_t>(std::stoull(argv[2]));
        const size_t deg2_plus_1 = static_cast<size_t>(std::stoull(argv[3]));

        const kzgPublicParameters public_params = generate_kzg_public_parameters(deg1_plus_1, deg2_plus_1);
        write_kzg_public_parameters(public_params, output_path);

        std::cout << "KZG setup helper\n";
        std::cout << "  output: " << output_path << "\n";
        std::cout << "  deg1-plus-1: " << deg1_plus_1 << "\n";
        std::cout << "  deg2-plus-1: " << deg2_plus_1 << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
