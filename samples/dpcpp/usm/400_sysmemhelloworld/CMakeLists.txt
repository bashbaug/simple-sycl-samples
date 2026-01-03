# Copyright (c) 2021-2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_sycl_sample(
    #TEST
    NUMBER 400
    TARGET sysmemhelloworld
    CATEGORY usm
    SOURCES main.cpp)
