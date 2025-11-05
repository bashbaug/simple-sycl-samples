/*
// Copyright (c) 2025 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <iostream>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace sycl;

int main()
{
    const size_t array_size = 16;

    queue q;

    std::cout << "Hello from SYCL!\n";
    std::cout << "Running on default SYCL device " << q.get_device().get_info<info::device::name>() << std::endl;

    int* data = sycl::malloc_device<int>(array_size, q);

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    syclex::properties kernel_props {
        intelex::grf_size<256>
    };

    q.parallel_for(range<1>{array_size}, kernel_props,
                   [=](id<1> i) { data[i] = i.get(0); });
    int *host_data = new int[array_size];
    q.memcpy(host_data, data, array_size * sizeof(int)).wait();

    for( int i = 0; i < array_size; i++ )
    {
        std::cout << "data[" << i << "] = " << host_data[i] << std::endl;
    }

    sycl::free(data, q);
    delete[] host_data;

    std::cout << "Done.\n";

    return 0;
}
