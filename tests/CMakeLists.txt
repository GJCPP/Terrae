# Tests for Terrae

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

function(add_terrae_gtest target)
  add_executable(${target} ${target}.cpp)
  target_link_libraries(${target} PRIVATE GTest::gtest GTest::gtest_main terrae_core)
  gtest_discover_tests(${target})
endfunction()

set(TERRAE_TEST_TARGETS
  polynomial_test
  algebra_utils_test
  sparse_polynomial_test
  lagrange_utils_test
  transcript_test
  proof_session_test
  proof_runtime_test
  soundness_counter_test
  cq_concatenated_table_lookup_test
  forest_row_fetch_test
  forest_inference_test
  forest_prediction_test
  forest_tree_layout_test
  forest_box_consistency_test
  forest_legality_test
  parent_child_box_inheritance_test
  root_selector_consistency_test
  parent_child_tree_consistency_test
  node_role_consistency_test
  root_box_range_test
  tree_histogram_consistency_test
  forest_histogram_test
  forest_histogram_bucket_address_test
  forest_training_inference_test
  forest_histogram_refresh_test
  forest_leaf_value_test
  forest_boosting_state_test
  forest_split_statistics_test
  forest_split_gain_test
  forest_quantized_split_gain_test
  forest_split_selection_test
  forest_split_update_test
  forest_training_proof_test
  forest_training_workflow_test
  forest_training_export_test
  forest_well_formedness_workflow_test
  forest_committed_inference_workflow_test
  forest_training_then_inference_workflow_test
  forest_training_then_inference_io_test
  forest_training_trace_adapter_test
  forest_training_trace_io_test
  forest_structure_test
  forest_classification_test
  box_check_test
  concatenated_table_lookup_test
  less_check_test
  div_check_test
  rational_check_test
  exp_check_test
  fixed_point_division_test
  kzg_test
  zero_check_test
  column_replication_test
  index_routing_test
  sum_check_test
  lookup_test
  shuffle_test
  cq_lookup_test
  table_lookup_test
  cq_table_lookup_test
  matrix_sum_test
  cube_sum_test
  cube_softmax_test
  cube_argmax_test
  hist_agg_test
  zkboost_histogram_ram_test
  signed_range_check_test
  masked_argmax_test
  argmax_test
  selector_check_test
  plonk_circuit_test
  plonk_blinding_test
  plonk_preprocess_test
  plonk_permutation_test
  plonk_quotient_test
  plonk_linearisation_test
  plonk_verifier_test
  plonk_opening_test
  plonk_proof_test
  plonk_bundle_test
  plonk_lookup_test
  zkboost_trace_io_test
  zkboost_shared_dependency_audit_test
  zkboost_shared_forest_commitments_test
  zkboost_shared_forest_well_formedness_plonk_test
  zkboost_shared_prediction_test
  zkboost_shared_prediction_plonk_test
  zkboost_shared_cert_gbdt_test
  zkboost_shared_inference_workflow_test
  zkboost_shared_validate_forest_test
  zkboost_shared_validate_forest_fetch_test
  zkboost_shared_validate_boosting_state_test
  zkboost_shared_validate_histograms_test
  zkboost_shared_validate_leaf_values_test
  zkboost_shared_validate_leaf_values_plonk_test
  zkboost_shared_validate_splits_test
  zkboost_shared_validate_splits_plonk_test
)

foreach(target IN LISTS TERRAE_TEST_TARGETS)
  add_terrae_gtest(${target})
endforeach()

target_compile_definitions(zkboost_shared_dependency_audit_test
  PRIVATE TERRAE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_custom_target(test_all DEPENDS ${TERRAE_TEST_TARGETS})
