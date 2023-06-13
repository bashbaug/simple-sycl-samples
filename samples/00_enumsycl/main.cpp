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
    for( auto& p : platform::get_platforms() )
    {
        std::cout << std::endl << "SYCL Platform: " << p.get_info<info::platform::name>() << std::endl;

        for( auto& d : p.get_devices() )
        {
            std::cout << "SYCL Device: " << d.get_info<info::device::name>() << std::endl;
        }
    }

    return 0;
}