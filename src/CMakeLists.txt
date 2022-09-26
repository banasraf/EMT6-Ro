set(EMT6RO_TESTS_SOURCES "")
set(EMT6RO_BENCH_SOURCES "")
set(EMT6RO_SOURCES "")
set(EMT6RO_HEADERS "")

add_subdirectory(emt6ro)
if (${EMT6RO_BUILD_TESTS})
    add_subdirectory(tests)
endif()
if (${EMT6RO_BUILD_BENCH})
    add_subdirectory(bench)
endif()

if (${EMT6RO_BUILD_TESTS})
    add_executable(emt6ro-tests ${EMT6RO_TESTS_SOURCES} ${EMT6RO_SOURCES})
    target_include_directories(emt6ro-tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(emt6ro-tests PRIVATE gtest)
    target_include_directories(emt6ro-tests PRIVATE ${PROJECT_SOURCE_DIR}/third-party/json)
    set_target_properties(emt6ro-tests
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
       INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:...>
)

endif()

if (${EMT6RO_BUILD_BENCH})
    add_executable(emt6ro-bench ${EMT6RO_BENCH_SOURCES} ${EMT6RO_SOURCES})
    target_include_directories(emt6ro-bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(emt6ro-bench PRIVATE benchmark)
    target_include_directories(emt6ro-bench PRIVATE ${PROJECT_SOURCE_DIR}/third-party/json)
endif()
