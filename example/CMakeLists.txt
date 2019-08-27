add_executable(example main.cc)
target_link_libraries(example emt6ro)
set_property(TARGET example PROPERTY INTERPROCEDURAL_OPTIMIZATIO TRUE)