#written by Qiqi Wang
# CMakeLists.txt
project(CatMap)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)

if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH sm_61)
endif(NOT DEFINED CUDA_ARCH)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=${CUDA_ARCH}; --std=c++14;-Wno-deprecated-gpu-targets; -lineinfo; -use_fast_math; ${MPI_COMPILE_FLAGS})

set_source_files_properties(
    lorenz_density.cpp
    PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

cuda_add_executable(lorenz_density lorenz_density.cpp)
