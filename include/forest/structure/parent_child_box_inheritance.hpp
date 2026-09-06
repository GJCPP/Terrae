#pragma once

#include <vector>

#include "column_replication.hpp"
#include "index_routing.hpp"
#include "kzg.hpp"
#include "less_check.hpp"
#include "protocol_lifecycle.hpp"
#include "selector_check.hpp"
#include "zero_check.hpp"

using namespace mcl::bn;

class parentChildBoxInheritanceCheckRecord {
public:
    size_t node_count = 0;
    size_t feature_count = 0;
    size_t bit_bound = 0;
    bool uses_mask = false;
    kzgCommitment lifted_threshold_commitment;
    kzgCommitment lifted_mask_commitment;
    kzgCommitment left_child_lower_commitment;
    kzgCommitment left_child_upper_commitment;
    kzgCommitment right_child_lower_commitment;
    kzgCommitment right_child_upper_commitment;
};

class parentChildBoxInheritanceParam {
public:
    kzgProtocol* kzg = nullptr;
    zeroProtocol* zero_check = nullptr;
    selectorCheckProtocol* selector_check = nullptr;
    columnReplicationProtocol* column_replication = nullptr;
    indexRoutingProtocol* index_routing = nullptr;
    lessCheckProtocol* less_check = nullptr;
};

class parentChildBoxInheritanceVerifier {
public:
    void add_check(const parentChildBoxInheritanceCheckRecord& record);
    bool finalize() const;

    std::vector<parentChildBoxInheritanceCheckRecord> checks;
    const parentChildBoxInheritanceParam* params = nullptr;
};

class parentChildBoxInheritanceProver {
public:
    parentChildBoxInheritanceCheckRecord add_check(const virCommitment& left_child_indices,
                                                   const virCommitment& right_child_indices,
                                                   const virCommitment& parent_lower,
                                                   const virCommitment& parent_upper,
                                                   const virCommitment& thresholds,
                                                   const virCommitment* mask,
                                                   size_t node_count,
                                                   size_t feature_count,
                                                   size_t bit_bound);

    std::vector<parentChildBoxInheritanceCheckRecord> checks;
    const parentChildBoxInheritanceParam* params = nullptr;
};

// Proves that left/right child boxes are derived from a parent box by splitting
// exactly one selected feature at a threshold.
//
// For each active row:
// - `feature_selector` is one-hot across the feature axis
// - `left_lower = parent_lower`
// - `right_upper = parent_upper`
// - on non-selected features:
//     `left_upper = parent_upper` and `right_lower = parent_lower`
// - on the selected feature:
//     `left_upper = threshold` and `right_lower = threshold`
// - the selected threshold is strictly inside the parent interval
//
// Example:
//   parent = [ [1, 10], [5, 20] )
//   select feature 0, threshold 3
//   left  = [ [1, 10], [3, 20] )   // row-major lower/upper split across feature 0
//   right = [ [3, 10], [5, 20] )
//
// Masked rows are ignored structurally and only need a masked one-hot selector.
class parentChildBoxInheritanceProtocol : protected ProtocolLifecycle {
public:
    ~parentChildBoxInheritanceProtocol() noexcept
    {
        auto_finalize("parentChildBoxInheritanceProtocol", [this] { return finalize(); });
    }

    void setup(kzgProtocol& kzg,
               zeroProtocol& zero_check,
               selectorCheckProtocol& selector_check,
               columnReplicationProtocol& column_replication,
               indexRoutingProtocol& index_routing,
               lessCheckProtocol& less_check)
    {
        params_.kzg = &kzg;
        params_.zero_check = &zero_check;
        params_.selector_check = &selector_check;
        params_.column_replication = &column_replication;
        params_.index_routing = &index_routing;
        params_.less_check = &less_check;
        prover_.params = &params_;
        verifier_.params = &params_;
        mark_setup();
    }

    void add_check(const virCommitment& feature_selector,
                   const virCommitment& left_child_indices,
                   const virCommitment& right_child_indices,
                   const virCommitment& parent_lower,
                   const virCommitment& parent_upper,
                   const virCommitment& left_lower,
                   const virCommitment& left_upper,
                   const virCommitment& right_lower,
                   const virCommitment& right_upper,
                   const virCommitment& thresholds,
                   size_t node_count,
                   size_t feature_count,
                   size_t bit_bound);

    void add_masked_check(const virCommitment& feature_selector,
                          const virCommitment& mask,
                          const virCommitment& left_child_indices,
                          const virCommitment& right_child_indices,
                          const virCommitment& parent_lower,
                          const virCommitment& parent_upper,
                          const virCommitment& left_lower,
                          const virCommitment& left_upper,
                          const virCommitment& right_lower,
                          const virCommitment& right_upper,
                          const virCommitment& thresholds,
                          size_t node_count,
                          size_t feature_count,
                          size_t bit_bound);

    bool finalize();

private:
    void add_check_internal(const virCommitment& feature_selector,
                            const virCommitment* mask,
                            const virCommitment& left_child_indices,
                            const virCommitment& right_child_indices,
                            const virCommitment& parent_lower,
                            const virCommitment& parent_upper,
                            const virCommitment& left_lower,
                            const virCommitment& left_upper,
                            const virCommitment& right_lower,
                            const virCommitment& right_upper,
                            const virCommitment& thresholds,
                            size_t node_count,
                            size_t feature_count,
                            size_t bit_bound);

    parentChildBoxInheritanceParam params_;
    parentChildBoxInheritanceProver prover_;
    parentChildBoxInheritanceVerifier verifier_;
    std::vector<virCommitment> feature_selectors_;
    std::vector<virCommitment> masks_;
    std::vector<virCommitment> left_child_indices_;
    std::vector<virCommitment> right_child_indices_;
    std::vector<virCommitment> parent_lowers_;
    std::vector<virCommitment> parent_uppers_;
    std::vector<virCommitment> left_lowers_;
    std::vector<virCommitment> left_uppers_;
    std::vector<virCommitment> right_lowers_;
    std::vector<virCommitment> right_uppers_;
    std::vector<virCommitment> thresholds_;
};
