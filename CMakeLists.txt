# Copyright (c) 2021-2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Try to detect the right SYCL compiler if one is not explicitly specified:
if (NOT CMAKE_CXX_COMPILER)
    if (WIN32)
        set(CMAKE_CXX_COMPILER icx)
    else()
        find_program(HAS_ICPX "icpx" NO_CACHE)
        if (HAS_ICPX)
            set(CMAKE_CXX_COMPILER icpx)
        else()
            set(CMAKE_CXX_COMPILER clang++)
        endif()
    endif()
endif()

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE PATH "Build Type" FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(SimpleSYCLSamples VERSION 1.0)
option(WITHCUDA "Enable CUDA device support.")

set(CUDA_GPU_ARCH "sm_60" CACHE STRING "CUDA GPUs to compile for.")
if (WITHCUDA)
    mark_as_advanced(CLEAR FORCE CUDA_GPU_ARCH)
else()
    mark_as_advanced(FORCE CUDA_GPU_ARCH)
endif()

option(SAMPLES_BUILD_CUDA "Enable support for the nvptx64-nvidia-cuda SYCL target")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()
endif()

add_subdirectory(samples)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install Path" FORCE)
endif()
