/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>

using L1_UC = sycl::ext::intel::experimental::cache_control<sycl::ext::intel::experimental::cache_mode::uncached, sycl::ext::intel::experimental::cache_level::L1>;     // R/W
using L1_ST = sycl::ext::intel::experimental::cache_control<sycl::ext::intel::experimental::cache_mode::streaming, sycl::ext::intel::experimental::cache_level::L1>;    // R/W
using L1_CA = sycl::ext::intel::experimental::cache_control<sycl::ext::intel::experimental::cache_mode::cached, sycl::ext::intel::experimental::cache_level::L1>;       // R
using L1_WB = sycl::ext::intel::experimental::cache_control<sycl::ext::intel::experimental::cache_mode::write_back, sycl::ext::intel::experimental::cache_level::L1>;   // W
using L1_WT = sycl::ext::intel::experimental::cache_control<sycl::ext::intel::experimental::cache_mode::write_through, sycl::ext::intel::experimental::cache_level::L1>;// W

using L2_UC = sycl::ext::intel::experimental::cache_control<sycl::ext::intel::experimental::cache_mode::uncached, sycl::ext::intel::experimental::cache_level::L2>;     // R/W
using L2_CA = sycl::ext::intel::experimental::cache_control<sycl::ext::intel::experimental::cache_mode::cached, sycl::ext::intel::experimental::cache_level::L2>;       // R
using L2_WB = sycl::ext::intel::experimental::cache_control<sycl::ext::intel::experimental::cache_mode::write_back, sycl::ext::intel::experimental::cache_level::L2>;   // W

constexpr auto READ_UC_UC = sycl::ext::intel::experimental::read_hint<L1_UC, L2_UC>; // L1UC_L3UC
constexpr auto READ_UC_CA = sycl::ext::intel::experimental::read_hint<L1_UC, L2_CA>; // L1UC_L3C
constexpr auto READ_CA_UC = sycl::ext::intel::experimental::read_hint<L1_CA, L2_UC>; // L1C_L3UC
constexpr auto READ_CA_CA = sycl::ext::intel::experimental::read_hint<L1_CA, L2_CA>; // L1C_L3C
constexpr auto READ_ST_UC = sycl::ext::intel::experimental::read_hint<L1_ST, L2_UC>; // L1S_L3UC
constexpr auto READ_ST_CA = sycl::ext::intel::experimental::read_hint<L1_ST, L2_CA>; // L1S_L3C

constexpr auto WRITE_UC_UC = sycl::ext::intel::experimental::write_hint<L1_UC, L2_UC>; // L1UC_L3UC
constexpr auto WRITE_UC_CA = sycl::ext::intel::experimental::write_hint<L1_UC, L2_WB>; // L1UC_L3WB
constexpr auto WRITE_WT_UC = sycl::ext::intel::experimental::write_hint<L1_WT, L2_UC>; // L1WT_L3UC
constexpr auto WRITE_WT_CA = sycl::ext::intel::experimental::write_hint<L1_WT, L2_WB>; // L1WT_L3WB
constexpr auto WRITE_ST_UC = sycl::ext::intel::experimental::write_hint<L1_ST, L2_UC>; // L1S_L3UC
constexpr auto WRITE_ST_CA = sycl::ext::intel::experimental::write_hint<L1_ST, L2_WB>; // L1S_L3WB
constexpr auto WRITE_WB_CA = sycl::ext::intel::experimental::write_hint<L1_WB, L2_WB>; // L1WB_L3WB

template <typename T>
using ptr_ca =
    sycl::ext::oneapi::experimental::annotated_ptr<T,
    decltype(sycl::ext::oneapi::experimental::properties(READ_CA_CA, WRITE_WB_CA))>;

template <typename T>
using ptr_uc =
    sycl::ext::oneapi::experimental::annotated_ptr<T,
    decltype(sycl::ext::oneapi::experimental::properties(READ_UC_UC, WRITE_UC_UC))>;

int main()
{
    const size_t N = 1024;

    sycl::queue q;

    std::cout << "Running on SYCL device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    int* input = sycl::malloc_shared<int>(N, q);
    int* output = sycl::malloc_shared<int>(N, q);
    std::iota(input, input + N, 1);
    std::fill(output, output + N, 0);

    q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> id) {
        ptr_ca<int> a_input(input);
        ptr_uc<int> a_output(output);
        a_output[id] = a_input[id] * 2;
    }).wait();

    for( int i = 0; i < 16; i++ )
    {
        std::cout << "output[" << i << "] = " << output[i] << std::endl;
    }

    sycl::free(input, q);
    sycl::free(output, q);

    return 0;
}
