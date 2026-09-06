# Shared core library
file(GLOB TERRAE_ROOT_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE TERRAE_SUBDIR_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp"
)

set(TERRAE_CORE_SOURCES
  ${TERRAE_ROOT_SOURCES}
  ${TERRAE_SUBDIR_SOURCES}
)

add_library(terrae_core STATIC ${TERRAE_CORE_SOURCES})
add_library(zkgbdt_core ALIAS terrae_core)
target_include_directories(terrae_core PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/algebra
  ${PROJECT_SOURCE_DIR}/include/commitments
  ${PROJECT_SOURCE_DIR}/include/protocols
  ${PROJECT_SOURCE_DIR}/include/plonk
  ${PROJECT_SOURCE_DIR}/include/forest
  ${PROJECT_SOURCE_DIR}/include/forest/public_inference
  ${PROJECT_SOURCE_DIR}/include/forest/structure
  ${PROJECT_SOURCE_DIR}/include/forest/training
)
target_compile_definitions(terrae_core PUBLIC
  $<$<BOOL:${TERRAE_ENABLE_STRUCTURED_PROTOCOL_PROFILE}>:TERRAE_ENABLE_STRUCTURED_PROTOCOL_PROFILE=1>
  $<$<BOOL:${TERRAE_ENABLE_STRUCTURED_PROTOCOL_PROFILE}>:ZKGBDT_ENABLE_STRUCTURED_PROTOCOL_PROFILE=1>
)
target_link_libraries(terrae_core PUBLIC mcl)
