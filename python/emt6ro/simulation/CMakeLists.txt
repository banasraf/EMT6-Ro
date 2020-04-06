pybind11_add_module(backend MODULE backend.cc)
target_link_libraries(backend PUBLIC emt6ro)
#set_target_properties(backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/emt6ro/simulation")
add_custom_command(TARGET backend PRE_BUILD
		   COMMAND ${CMAKE_COMMAND} -E copy_directory
	           ${CMAKE_SOURCE_DIR}/python/
		   ${CMAKE_BINARY_DIR}/python/)	   
	   #add_custom_command(TARGET simulation POST_BUILD
	   #	   COMMAND ${CMAKE_COMMAND} -E copy
	   #       ${CMAKE_SOURCE_DIR}/python/__init__.py
	   #	   ${CMAKE_BINARY_DIR}/emt6ro/__init__.py)
	   #add_custom_command(TARGET simulation POST_BUILD
	   #	   COMMAND ${CMAKE_COMMAND} -E copy
	   #       ${CMAKE_SOURCE_DIR}/python/setup.py
	   #	   ${CMAKE_BINARY_DIR}/emt6ro/setup.py)
