cmake_minimum_required(VERSION 3.18)
option(BUILD_WITH_CUDA "Build the CUDA functionaility" ON)

if (BUILD_WITH_CUDA)
    project(_roshambo2 LANGUAGES CXX CUDA)
else (BUILD_WITH_CUDA)
    project(_roshambo2 LANGUAGES CXX)
endif (BUILD_WITH_CUDA)
# Find OpenMP package
find_package(OpenMP REQUIRED)

# Find CUDA
if (BUILD_WITH_CUDA)
    find_package(CUDA REQUIRED)
endif (BUILD_WITH_CUDA)

# Find pybind
find_package(pybind11)

if (BUILD_WITH_CUDA)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    # change here for newer or older cards
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
endif (BUILD_WITH_CUDA)

# TODO: optional GPU build
pybind11_add_module(_roshambo2_cpp roshambo2/backends/cpp_src/cpp_functions.cpp roshambo2/backends/cpp_src/cpp_helper_functions.cpp)
if (BUILD_WITH_CUDA)
    pybind11_add_module(_roshambo2_cuda roshambo2/backends/cuda_src/cuda_functions.cu roshambo2/backends/cuda_src/cuda_interface.cpp)
    target_link_libraries(_roshambo2_cuda PRIVATE OpenMP::OpenMP_CXX)
endif (BUILD_WITH_CUDA)

target_link_libraries(_roshambo2_cpp PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS _roshambo2_cpp
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})

if (BUILD_WITH_CUDA)
install(TARGETS _roshambo2_cuda
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
endif (BUILD_WITH_CUDA)
