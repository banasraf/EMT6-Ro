add_subdirectory(site)
add_subdirectory(cell)
add_subdirectory(parameters)
add_subdirectory(common)
add_subdirectory(diffusion)
add_subdirectory(simulation)
add_subdirectory(state)
add_subdirectory(statistics)

add_sources(EMT6RO_SOURCES)
add_test_sources(EMT6RO_TESTS_SOURCES)
add_headers(EMT6RO_HEADERS)
add_bench_sources(EMT6RO_BENCH_SOURCES)

add_library(emt6ro STATIC ${EMT6RO_SOURCES} ${EMT6RO_HEADERS})

set_target_properties(emt6ro
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
       INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:...>
)

target_include_directories(emt6ro PRIVATE ${PROJECT_SOURCE_DIR}/third-party/json)
target_include_directories(emt6ro PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(emt6ro INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# set_property(TARGET emt6ro PROPERTY INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:...>)
