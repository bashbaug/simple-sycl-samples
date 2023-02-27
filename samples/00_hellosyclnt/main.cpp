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
#include <popl/popl.hpp>

#include <iostream>

using namespace cl::sycl;
class Hello;

#ifndef __has_builtin         // Optional of course.
  #define __has_builtin(x) 0  // Compatibility with non-clang compilers.
#endif

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: hellosyclnt [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    const size_t array_size = 16;
    int data[array_size];

    sycl::platform platform = sycl::platform::get_platforms()[platformIndex];
    printf("Running on SYCL platform: %s\n", platform.get_info<sycl::info::platform::name>().c_str());

    sycl::device device = platform.get_devices()[deviceIndex];
    printf("Running on SYCL device: %s\n", device.get_info<sycl::info::device::name>().c_str());

    sycl::context context = sycl::context{ device };
    sycl::queue queue = sycl::queue{ context, device, sycl::property::queue::in_order() };

    {
        buffer<int, 1>  resultBuf{ data, range<1>{array_size} };
        queue.submit([&](handler& cgh) {
            auto resultAcc = resultBuf.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class Hello>(range<1>{array_size}, [=](id<1> i) {
                int* ptr = resultAcc.get_pointer() + i;
                int value = i.get(0);
                #if __has_builtin(__builtin_nontemporal_store)
                    __builtin_nontemporal_store(value, ptr);
                #else
                    *ptr = value;
                #endif
            });
        });
    }

    for( int i = 0; i < array_size; i++ )
    {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    return 0;
}