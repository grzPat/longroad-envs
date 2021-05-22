cmake_minimum_required(VERSION 3.0.0)
project(longroad)

find_package(OpenMP)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -O3")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -O3 ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories( ${EIGEN3_INCLUDE_DIRS} )

include(CTest)
enable_testing()

add_subdirectory(pybind11)
pybind11_add_module(world src/pywrapper.cpp src/world.cpp)
set_target_properties(world
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR/longroad/}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR/longroad/}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR/longroad/}"
)

add_executable(longroad src/world.cpp)
target_compile_options(longroad PRIVATE -msse3 )
#target_link_libraries (world PRIVATE longroad)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
