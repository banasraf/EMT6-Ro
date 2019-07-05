set(EMT6RO_TESTS_SOURCES "")
set(EMT6RO_BENCH_SOURCES "")
set(EMT6RO_SOURCES "")
set(EMT6RO_HEADERS "")

add_subdirectory(emt6ro)
add_subdirectory(tests)
add_subdirectory(bench)

add_executable(emt6ro-tests ${EMT6RO_TESTS_SOURCES} ${EMT6RO_SOURCES})
target_include_directories(emt6ro-tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(emt6ro-tests PRIVATE gtest nlohmann_json::nlohmann_json)

add_executable(emt6ro-bench ${EMT6RO_BENCH_SOURCES} ${EMT6RO_SOURCES})
target_include_directories(emt6ro-bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(emt6ro-bench PRIVATE benchmark nlohmann_json::nlohmann_json)