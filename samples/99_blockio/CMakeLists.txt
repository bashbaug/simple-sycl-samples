# Copyright (c) 2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_sycl_sample(
    TEST
    NUMBER 99
    TARGET blockio
    ADDITIONAL_LINK_OPTIONS -Xspirv-translator=spir64 -spirv-ext=+SPV_INTEL_2d_block_io
    SOURCES main.cpp )
