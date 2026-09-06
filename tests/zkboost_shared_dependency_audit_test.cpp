#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TERRAE_SOURCE_DIR
#define TERRAE_SOURCE_DIR "."
#endif

namespace {

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool is_checked_source_file(const std::filesystem::path& path)
{
    return path.extension() == ".hpp" || path.extension() == ".cpp";
}

} // namespace

TEST(ZkboostSharedDependencyAuditTest, ProductionCodeDoesNotUseGBDTProofPrimitives)
{
    const std::filesystem::path root(TERRAE_SOURCE_DIR);
    const std::vector<std::filesystem::path> checked_roots = {
        root / "include" / "zkboost" / "shared_trace",
        root / "src" / "zkboost" / "shared_trace",
    };
    const std::vector<std::string> forbidden_tokens = {
        "forest_training_proof",
        "forest_training_inference",
        "forest_structure",
        "forest_boosting_state",
        "forest_histogram_refresh",
        "forest_leaf_value",
        "forest_split_statistics",
        "forest_split_gain",
        "forest_quantized_split_gain",
        "forest_split_selection",
        "forest_split_update",
        "forest_training_then_inference_workflow",
    };

    for (const std::filesystem::path& checked_root : checked_roots) {
        ASSERT_TRUE(std::filesystem::exists(checked_root)) << checked_root;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(checked_root)) {
            if (!entry.is_regular_file() || !is_checked_source_file(entry.path())) {
                continue;
            }
            const std::string text = read_file(entry.path());
            for (const std::string& token : forbidden_tokens) {
                EXPECT_EQ(text.find(token), std::string::npos)
                    << "Forbidden dependency token `" << token << "` found in "
                    << entry.path();
            }
        }
    }
}
