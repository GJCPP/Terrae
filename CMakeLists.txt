cmake_minimum_required(VERSION 3.14)
project(Terrae VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TERRAE_ENABLE_STRUCTURED_PROTOCOL_PROFILE
       "Compile in structured protocol profiling hooks (default OFF for benchmark builds)"
       OFF)
option(ZKGBDT_ENABLE_STRUCTURED_PROTOCOL_PROFILE
       "Legacy alias for TERRAE_ENABLE_STRUCTURED_PROTOCOL_PROFILE"
       OFF)
if(ZKGBDT_ENABLE_STRUCTURED_PROTOCOL_PROFILE)
    set(TERRAE_ENABLE_STRUCTURED_PROTOCOL_PROFILE ON)
endif()

set(TERRAE_MCL_GIT_REPOSITORY
    "https://github.com/herumi/mcl.git"
    CACHE STRING "Git repository used to fetch herumi/mcl")
set(TERRAE_MCL_GIT_TAG
    "v3.04"
    CACHE STRING "Git tag or commit used to fetch herumi/mcl")
set(TERRAE_MCL_SOURCE_DIR
    ""
    CACHE PATH "Optional local herumi/mcl source directory; bypasses the Git fetch when set")

# Add compile options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Include FetchContent module for dependency management
include(FetchContent)
option(BUILD_TESTING "Build Terrae tests" OFF)
include(CTest)

# Fetch mcl from GitHub by default, or use an explicitly supplied local source
# tree for offline/source-archive builds.
if(TERRAE_MCL_SOURCE_DIR)
    get_filename_component(TERRAE_MCL_SOURCE_DIR
                           "${TERRAE_MCL_SOURCE_DIR}"
                           ABSOLUTE
                           BASE_DIR "${CMAKE_SOURCE_DIR}")
    if(NOT EXISTS "${TERRAE_MCL_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
                "TERRAE_MCL_SOURCE_DIR must point to an mcl source tree containing CMakeLists.txt")
    endif()
    message(STATUS "Using local herumi/mcl source: ${TERRAE_MCL_SOURCE_DIR}")
    FetchContent_Declare(mcl SOURCE_DIR "${TERRAE_MCL_SOURCE_DIR}")
else()
    FetchContent_Declare(
        mcl
        GIT_REPOSITORY ${TERRAE_MCL_GIT_REPOSITORY}
        GIT_TAG        ${TERRAE_MCL_GIT_TAG}
    )
endif()

# Make mcl available
FetchContent_MakeAvailable(mcl)

# Add subdirectories
add_subdirectory(src)

# Optional: Add examples and tests if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    add_subdirectory(examples)
endif()

if(BUILD_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_subdirectory(tests)
endif()
