# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

add_sycl_sample(
    TEST
    NUMBER 99
    TARGET invokesimd
    SOURCES main.cpp
    ADDITIONAL_COMPILE_OPTIONS -Xclang -fsycl-allow-func-ptr
    ADDITIONAL_LINK_OPTIONS -fno-sycl-device-code-split-esimd )
