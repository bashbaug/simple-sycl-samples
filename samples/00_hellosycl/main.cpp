/*
// Copyright (c) 2020 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main()
{
    const size_t array_size = 16;
    int data[array_size];

    queue q;

    std::cout << "Hello from SYCL!\n";
    std::cout << "Running on default SYCL device " << q.get_device().get_info<info::device::name>() << std::endl;

    {
        buffer<int, 1>  resultBuf{ data, range<1>{array_size} };
        q.submit([&](handler& cgh) {
            auto resultAcc = resultBuf.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class Hello>(range<1>{array_size}, [=](id<1> i) {
                resultAcc[i] = i.get(0);
            });
        });
    }

    for( int i = 0; i < array_size; i++ )
    {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    return 0;
}