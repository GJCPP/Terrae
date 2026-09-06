#pragma once

#include "box_check.hpp"
#include "concatenated_table_lookup.hpp"
#include "less_check.hpp"
#include "protocol_lifecycle.hpp"

using namespace mcl::bn;

class forestTrainingInferenceWitness {
public:
    virCommitment query_tree_ids;
    virCommitment lifted_inputs;
    virCommitment leaf_indices;
    virCommitment leaf_lower_bounds;
    virCommitment leaf_upper_bounds;
    virCommitment tree_scores;

    virCommitment node_tree_ids;
    virCommitment node_lower_bounds;
    virCommitment node_upper_bounds;
    virCommitment node_leaf_mask;
    virCommitment node_leaf_values;
};

class forestTrainingInferenceCheckRecord {
public:
    size_t query_row_count = 0;
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t input_bit_bound = 0;
};

class forestTrainingInferenceParam {
public:
    kzgProtocol* kzg = nullptr;
    lessCheckProtocol* less_check = nullptr;
    boxCheckProtocol* box_check = nullptr;
    concatenatedTableLookupProtocol* concatenated_table_lookup = nullptr;
};

// Proves one training-side witness-forest inference step over a committed node
// table.
//
// Query layout:
// - one row per `(tree, sample)` pair
// - each row outputs:
//   - `leaf_index`
//   - the fetched leaf box
//   - and the fetched tree score / leaf value
//
// Statement:
// - those outputs all come from the same committed node row,
// - the fetched row belongs to the requested tree,
// - the fetched row is marked as a leaf,
// - and the lifted sample lies inside the fetched half-open box.
class forestTrainingInferenceProtocol : protected ProtocolLifecycle {
public:
    ~forestTrainingInferenceProtocol() noexcept
    {
        auto_finalize("forestTrainingInferenceProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               lessCheckProtocol& less_check,
               boxCheckProtocol& box_check,
               concatenatedTableLookupProtocol& concatenated_table_lookup)
    {
        params_.kzg = &kzg;
        params_.less_check = &less_check;
        params_.box_check = &box_check;
        params_.concatenated_table_lookup = &concatenated_table_lookup;
        mark_setup();
    }

    void add_check(const forestTrainingInferenceWitness& witness,
                   size_t query_row_count,
                   size_t node_count,
                   size_t feature_count,
                   size_t input_bit_bound);

    bool finalize();

private:
    forestTrainingInferenceParam params_;
    std::vector<forestTrainingInferenceWitness> witnesses_;
    std::vector<forestTrainingInferenceCheckRecord> checks_;
};
