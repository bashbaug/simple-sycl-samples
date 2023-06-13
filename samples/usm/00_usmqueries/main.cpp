/*
// Copyright (c) 2020-2022 Ben Ashbaugh
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
            std::cout << "\tSYCL Device: " << d.get_info<info::device::name>() << std::endl;

            std::cout << std::boolalpha;
            std::cout << "\t\tSupports usm_device_allocations: "
                << d.has(aspect::usm_device_allocations) << std::endl;
            std::cout << "\t\tSupports usm_host_allocations: "
                << d.has(aspect::usm_host_allocations) << std::endl;
            std::cout << "\t\tSupports usm_atomic_host_allocations: "
                << d.has(aspect::usm_atomic_host_allocations) << std::endl;
            std::cout << "\t\tSupports usm_restricted_shared_allocations: "
                << d.has(aspect::usm_restricted_shared_allocations) << std::endl;
            std::cout << "\t\tSupports usm_shared_allocations: "
                << d.has(aspect::usm_shared_allocations) << std::endl;
            std::cout << "\t\tSupports usm_atomic_shared_allocations: "
                << d.has(aspect::usm_atomic_shared_allocations) << std::endl;
            std::cout << "\t\tSupports usm_system_allocations: "
                << d.has(aspect::usm_system_allocations) << std::endl;
        }
    }

    return 0;
}
