# Copyright (c) 2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_sycl_sample(
    TEST
    NUMBER 10
    TARGET bigrange
    SOURCES main.cpp
    ADDITIONAL_COMPILE_OPTIONS -fno-sycl-id-queries-fit-in-int )
