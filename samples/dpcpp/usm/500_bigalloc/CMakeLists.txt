# Copyright (c) 2021-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_sycl_sample(
    TEST
    NUMBER 500
    TARGET bigalloc
    CATEGORY usm
    SOURCES main.cpp)
