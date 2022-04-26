/*
// Copyright (c) 2020 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
                << d.get_info<info::device::usm_device_allocations>() << std::endl;
            std::cout << "\t\tSupports usm_host_allocations: "
                << d.get_info<info::device::usm_host_allocations>() << std::endl;
            std::cout << "\t\tSupports usm_restricted_shared_allocations: "
                << d.get_info<info::device::usm_restricted_shared_allocations>() << std::endl;
            std::cout << "\t\tSupports usm_shared_allocations: "
                << d.get_info<info::device::usm_shared_allocations>() << std::endl;
            std::cout << "\t\tSupports usm_system_allocations: "
                << d.get_info<info::device::usm_system_allocations>() << std::endl;
        }
    }

    return 0;
}
