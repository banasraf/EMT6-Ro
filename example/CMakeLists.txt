add_executable(example main.cc)
target_link_libraries(example emt6ro)

add_executable(timelapse timelapse.cc)
target_link_libraries(timelapse emt6ro)
