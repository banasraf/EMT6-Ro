pybind11_add_module(py_emt6ro MODULE py_emt6ro.cc)
target_link_libraries(py_emt6ro PUBLIC emt6ro)
set_target_properties(py_emt6ro PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")