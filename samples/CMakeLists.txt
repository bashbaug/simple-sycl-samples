# Copyright (c) 2021 Ben Ashbaugh
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function(add_sycl_sample)
    set(options TEST)
    set(one_value_args NUMBER TARGET CATEGORY)
    set(multi_value_args SOURCES KERNELS INCLUDES LIBS)
    cmake_parse_arguments(SYCL_SAMPLE
        "${options}" "${one_value_args}" "${multi_value_args}"
        ${ARGN}
    )

    if(NOT SYCL_SAMPLE_NUMBER)
        message(STATUS "No sample number specified for sample ${SYCL_SAMPLE_TARGET}, using 99.")
        set(SYCL_SAMPLE_NUMBER 99)
    endif()

    add_executable(${SYCL_SAMPLE_TARGET} ${SYCL_SAMPLE_SOURCES})

    target_include_directories(${SYCL_SAMPLE_TARGET} PRIVATE ${SYCL_SAMPLE_INCLUDES})
    target_link_libraries(${SYCL_SAMPLE_TARGET} ${SYCL_SAMPLE_LIBS})

    target_compile_options(${SYCL_SAMPLE_TARGET} PRIVATE -fsycl -fsycl-unnamed-lambda)
    # Passing -fsycl via target_link_libraries for older versions of CMake:
    #target_link_options(${SYCL_SAMPLE_TARGET} PRIVATE -fsycl )
    target_link_libraries(${SYCL_SAMPLE_TARGET} PRIVATE sycl -fsycl)

    if (WIN32)
        target_compile_definitions(${SYCL_SAMPLE_TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(${SYCL_SAMPLE_TARGET} PRIVATE /EHa)
    endif()

    set_target_properties(${SYCL_SAMPLE_TARGET} PROPERTIES FOLDER "Samples/${SYCL_SAMPLE_CATEGORY}/${SYCL_SAMPLE_NUMBER}_${SYCL_SAMPLE_TARGET}")

    if(CMAKE_CONFIGURATION_TYPES)
        set(SYCL_SAMPLE_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(SYCL_SAMPLE_CONFIGS ${CMAKE_BUILD_TYPE})
    endif()
    foreach(CONFIG ${SYCL_SAMPLE_CONFIGS})
        install(TARGETS ${SYCL_SAMPLE_TARGET} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
        install(FILES ${SYCL_SAMPLE_KERNELS} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
    endforeach()
    if(SYCL_SAMPLE_TEST)
        add_test(NAME ${SYCL_SAMPLE_TARGET} COMMAND ${SYCL_SAMPLE_TARGET})
    endif()
endfunction()

add_subdirectory( 00_enumsycl )
add_subdirectory( 00_hellosycl )
add_subdirectory( 04_julia )

add_subdirectory( dpcpp )
