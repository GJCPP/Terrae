function(add_terrae_example target source_file)
  add_executable(${target} ${source_file})
  target_link_libraries(${target} PRIVATE terrae_core)
endfunction()

add_terrae_example(kzg_setup_helper kzg_setup_helper.cpp)

set(TERRAE_DEFAULT_SRS_DEG1_PLUS_1 131072 CACHE STRING "Default KZG SRS G1 size for generate_default_kzg_srs")
set(TERRAE_DEFAULT_SRS_DEG2_PLUS_1 ${TERRAE_DEFAULT_SRS_DEG1_PLUS_1} CACHE STRING "Default KZG SRS G2 size for generate_default_kzg_srs")

add_custom_target(generate_default_kzg_srs
  COMMAND $<TARGET_FILE:kzg_setup_helper> ${CMAKE_BINARY_DIR}/kzg_srs.params ${TERRAE_DEFAULT_SRS_DEG1_PLUS_1} ${TERRAE_DEFAULT_SRS_DEG2_PLUS_1}
  DEPENDS kzg_setup_helper
  COMMENT "Generating default KZG SRS at ${CMAKE_BINARY_DIR}/kzg_srs.params"
)

add_terrae_example(example_custom_program example_custom_program.cpp)
add_terrae_example(example_private_table_lookups example_private_table_lookups.cpp)
add_terrae_example(example_all_protocols example_all_protocols.cpp)
add_terrae_example(example_forest_training_proof example_forest_training_proof.cpp)
add_terrae_example(example_forest_committed_inference example_forest_committed_inference.cpp)
add_terrae_example(example_training_then_inference example_training_then_inference.cpp)
add_terrae_example(example_training_then_inference_bundle example_training_then_inference_bundle.cpp)
add_terrae_example(example_zkboost_shared_trace example_zkboost_shared_trace.cpp)
add_terrae_example(benchmark_histogram_primitives benchmark_histogram_primitives.cpp)
