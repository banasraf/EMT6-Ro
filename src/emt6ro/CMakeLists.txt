add_subdirectory(site)
add_subdirectory(cell)
add_subdirectory(parameters)
add_subdirectory(common)
add_subdirectory(diffusion)
add_subdirectory(division)
add_subdirectory(simulation)
add_subdirectory(state)

add_sources(EMT6RO_SOURCES)
add_test_sources(EMT6RO_TESTS_SOURCES)
add_headers(EMT6RO_HEADERS)
add_bench_sources(EMT6RO_BENCH_SOURCES)

add_library(emt6ro STATIC ${EMT6RO_SOURCES} ${EMT6RO_HEADERS})
target_link_libraries(emt6ro PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(emt6ro PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include/emt6ro>)
set_property(TARGET emt6ro PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)