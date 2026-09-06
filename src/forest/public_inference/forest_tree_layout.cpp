#include "forest_tree_layout.hpp"

#include "bls12_381_wrapper.hpp"
#include "matrix_utils.hpp"

#include <stdexcept>
#include <vector>

namespace {

poly canonical_tree_layout_polynomial(size_t tree_count)
{
    std::vector<Fr> evals;
    evals.reserve(tree_count);
    for (size_t tree = 0; tree < tree_count; ++tree) {
        evals.push_back(field_from_size_t(tree + 1));
    }
    return poly_from_evals(evals);
}

}

forestTreeLayoutPublicTable build_forest_tree_layout_public_table(std::string key_prefix,
                                                                  size_t iteration_count,
                                                                  size_t class_count)
{
    if (iteration_count == 0 || !isPowerOfTwo(iteration_count)) {
        throw std::runtime_error(
            "build_forest_tree_layout_public_table: iteration_count must be a non-zero power of two");
    }
    if (class_count == 0 || !isPowerOfTwo(class_count)) {
        throw std::runtime_error(
            "build_forest_tree_layout_public_table: class_count must be a non-zero power of two");
    }
    if (key_prefix.empty()) {
        throw std::runtime_error("build_forest_tree_layout_public_table: key_prefix must be non-empty");
    }

    forestTreeLayoutPublicTable out;
    out.key_prefix = std::move(key_prefix);
    out.iteration_count = iteration_count;
    out.class_count = class_count;
    return out;
}

forestTreeLayoutPublicTable build_forest_tree_layout_public_table(const kzgPublicParameters& public_params,
                                                                  std::string key_prefix,
                                                                  size_t iteration_count,
                                                                  size_t class_count)
{
    forestTreeLayoutPublicTable out =
        build_forest_tree_layout_public_table(std::move(key_prefix), iteration_count, class_count);
    kzgProtocol kzg;
    kzg.setup(public_params);
    out.base_layout_commitment =
        kzg.commit_public(out.key_prefix + ".base_layout",
                          canonical_tree_layout_polynomial(out.tree_count()),
                          out.tree_count() - 1)
            .com;
    return out;
}
