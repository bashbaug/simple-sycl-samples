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
class Hello;

int main()
{
    const size_t array_size = 16;
    int data[array_size];

    queue q{ default_selector{} };

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