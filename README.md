# EMT6-Ro
## EMT6/Ro multicellular tumor spheroids simulation.
A CUDA implementation of the cellular automaton proposed and described in the following papers:
1. **Simon D. Angus** and **Monika Joanna Piotrowska**.  *A quantitative cellular automaton model of in vitro multicellular spheroid tumour growth.*
Journal of Theoretical Biology, 258:165–178, 5 2009.
2. **Simon D. Angus** and **Monika Joanna Piotrowska**. *The onset of necrosis in a 3d cellular automaton model of emt6 multi-cellular spheroids.*
Applicationes Mathematicae, 37:69–88, 01 2010.
3.  **Simon D. Angus** and **Monika Joanna Piotrowska**. *A matter of timing: Identifying significant multi-dose radiotherapy improvements by numerical simulation and genetic algorithm search.*
PLOS ONE, 9(12):1–28, 12 2014.

---
**Building:**
```bash
/$ git clone https://github.com/banasraf/EMT6-Ro.git --recursive; cd EMT6-Ro
EMT6-Ro/$ mkdir build; cd build
EMT6-Ro/build/$ cmake -DCMAKE_BUILD_TYPE=Release ..
EMT6-Ro/build/$ make 
```
It should produce `EMT6-Ro/build/example/example` executable.

**Running tests:**
```bash
EMT6-Ro/build/$ TEST_DATA_DIR=../data/test src/emt6ro-tests
```

**Using the library**

The most convenient way to add the library to the other project is to add a subdirectory with the 
contents of this repository and add the following lines to your `CMakeLists.txt`
```cmake
set(EMT6RO_BUILD_ONLY_LIB TRUE)
add_subdirectory(EMT6-Ro)
target_link_libraries(your-target PRIVATE emt6ro)
``` 
---
