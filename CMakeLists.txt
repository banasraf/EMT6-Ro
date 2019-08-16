cmake_minimum_required(VERSION 3.10)
project(emt6ro-simulation LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-dc>)

add_custom_target(lint
        COMMAND ${PROJECT_SOURCE_DIR}/tools/cpplint.py --recursive ${PROJECT_SOURCE_DIR}/src)

include(cmake/CollectSources.cmake)

# BUILD GOOGLE TEST
set(BUILD_GMOCK OFF CACHE INTERNAL "")
add_subdirectory(third-party/googletest)

# BUILD GOOGLE BENCHMARK
set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
add_subdirectory(third-party/benchmark)

# BUILD NLOHMANN JSON
set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(third-party/json)

add_subdirectory(src)
add_subdirectory(example)

