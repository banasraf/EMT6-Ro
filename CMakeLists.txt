cmake_minimum_required(VERSION 3.10)
project(emt6ro-simulation LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-dc>)
option(CUDA_DEBUG OFF)
add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${CUDA_DEBUG}>>:-G>)
if(CUDA_DEBUG)
    message("-- CUDA DEBUG ENABLED")
endif(CUDA_DEBUG)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options($<$<CONFIG:DEBUG>:-O0>)

add_custom_target(lint
        COMMAND ${PROJECT_SOURCE_DIR}/tools/cpplint.py --root=${PROJECT_SOURCE_DIR}/src --recursive ${PROJECT_SOURCE_DIR}/src)

add_custom_target(docs COMMAND mkdir -p docs
                       COMMAND doxygen ${PROJECT_SOURCE_DIR}/doxygen/Doxyfile)

include(cmake/CollectSources.cmake)

set(EMT6RO_BUILD_TESTS TRUE)
set(EMT6RO_BUILD_BENCH TRUE)
set(EMT6RO_BUILD_EXAMPLES TRUE)
if (${EMT6RO_BUILD_ONLY_LIB})
    set(EMT6RO_BUILD_TESTS FALSE)
    set(EMT6RO_BUILD_BENCH FALSE)
    set(EMT6RO_BUILD_EXAMPLES FALSE)
endif()

# BUILD GOOGLE TEST
if (${EMT6RO_BUILD_TESTS})
    set(BUILD_GMOCK OFF CACHE INTERNAL "")
    add_subdirectory(third-party/googletest)
endif()


# BUILD GOOGLE BENCHMARK
if (${EMT6RO_BUILD_BENCH})
    set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
    add_subdirectory(third-party/benchmark)
endif()

add_subdirectory(src)
if (${EMT6RO_BUILD_EXAMPLES})
    add_subdirectory(example)
endif()

add_subdirectory(third-party/pybind11)
add_subdirectory(python/emt6ro/simulation)
