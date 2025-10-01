# Copyright (c) 2021-2025 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_sycl_sample(
    TEST
    NUMBER 300
    TARGET smemhelloworld
    CATEGORY usm
    SOURCES main.cpp)
