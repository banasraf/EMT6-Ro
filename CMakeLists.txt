cmake_minimum_required(VERSION 3.10)
project(emt6ro-simulation LANGUAGES CUDA CXX)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-dc>)
option(CUDA_DEBUG OFF)
add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${CUDA_DEBUG}>>:-G>)
if(CUDA_DEBUG)
    message("-- CUDA DEBUG ENABLED")
endif(CUDA_DEBUG)

# Per-event instrumentation counters (repair decisions, kills, divisions,
# irradiation events, and per-decision irradiation/time-in-repair histograms).
# Off by default — atomicAdd in hot device code carries a small but real cost.
# When OFF, the Python API still exposes read_instr_counters / read_instr_histograms;
# they just return zeros. Enable for calibration / diagnostic runs.
option(EMT6RO_INSTRUMENT "Enable per-event instrumentation counters" OFF)
if(EMT6RO_INSTRUMENT)
    add_compile_definitions(EMT6RO_INSTRUMENT)
    message("-- EMT6RO instrumentation ENABLED")
endif()

# NVTX ranges around per-step kernel launches (findOccupied, updateROIs, diffuse,
# simulateCells, countLiving). Cost is zero when no profiler is attached; turn
# this on for builds that will be captured with `nsys profile -t cuda,nvtx ...`.
# Default OFF so untouched builds stay byte-equivalent to current source.
option(EMT6RO_NVTX "Emit NVTX ranges around per-step kernels" OFF)
if(EMT6RO_NVTX)
    add_compile_definitions(EMT6RO_NVTX)
    link_libraries(nvToolsExt)
    message("-- EMT6RO NVTX ranges ENABLED")
endif()

# Per-kernel CUDA-event timers. Accumulates ms spent in each kernel of the
# inner loop and exposes totals via Experiment.get_kernel_timers() in Python.
# Adds ~2 cudaEventRecord calls per kernel per step — negligible on Pascal+
# but kept opt-in so default builds match the reference source byte-for-byte.
option(EMT6RO_TIMING "Per-kernel CUDA-event timers" OFF)
if(EMT6RO_TIMING)
    add_compile_definitions(EMT6RO_TIMING)
    message("-- EMT6RO per-kernel timing ENABLED")
endif()

# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:
#                     -allow-unsuppported-compiler
#                     >)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options($<$<CONFIG:DEBUG>:-O0>)

add_custom_target(lint
        COMMAND ${PROJECT_SOURCE_DIR}/tools/cpplint.py --root=${PROJECT_SOURCE_DIR}/src --recursive ${PROJECT_SOURCE_DIR}/src)

add_custom_target(docs COMMAND mkdir -p docs
                       COMMAND doxygen ${PROJECT_SOURCE_DIR}/doxygen/Doxyfile)

include(cmake/CollectSources.cmake)

set(EMT6RO_BUILD_TESTS TRUE)
set(EMT6RO_BUILD_BENCH OFF)
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
