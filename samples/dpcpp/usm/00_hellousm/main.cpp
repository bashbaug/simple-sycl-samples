/*
// Copyright (c) 2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <iostream>

int main()
{
    const size_t array_size = 16;

    sycl::queue q{sycl::property::queue::in_order()};

    std::cout << "Hello from SYCL, with USM!\n";
    std::cout << "Running on default SYCL device " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    if (q.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cout << "This device supports aspect::usm_device_allocations.\n";
    } else {
        std::cerr << "This device does not support aspect::usm_device_allocations. Exiting.\n";
        return -1;
    }

    int* d_data = sycl::malloc_device<int>(array_size, q);
    q.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> i) {
        d_data[i] = i.get(0);
    });

    int h_data[array_size];
    q.copy(d_data, h_data, array_size);
    q.wait();

    for( int i = 0; i < array_size; i++ )
    {
        std::cout << "data[" << i << "] = " << h_data[i] << std::endl;
    }

    return 0;
}
