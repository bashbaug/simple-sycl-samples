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
#include <algorithm>
#include <iostream>

using namespace cl::sycl;

template <typename T>
static T rand_uniform_01() {
    return T(std::rand()) / T(RAND_MAX);
}

int main()
{
    const size_t array_size = 16;
    std::vector<float> src(array_size);
    std::vector<float> dst(array_size);
    std::generate_n(src.data(), src.size(), rand_uniform_01<float>);
    std::fill(dst.begin(), dst.end(), 0.0f);

    queue q{ default_selector{} };

    std::cout << "Hello from SYCL!\n";
    std::cout << "Running on default SYCL device " << q.get_device().get_info<info::device::name>() << std::endl;

    std::cout << "sqrt:\n";
    {
        buffer<float, 1>  srcBuf{ src.data(), range<1>{array_size} };
        buffer<float, 1>  dstBuf{ dst.data(), range<1>{array_size} };
        q.submit([&](handler& cgh) {
            auto srcAcc = srcBuf.get_access<access::mode::read>(cgh);
            auto dstAcc = dstBuf.get_access<access::mode::write>(cgh);
            cgh.parallel_for(range<1>{array_size}, [=](id<1> i) {
                dstAcc[i] = sqrt(srcAcc[i]);
            });
        });
    }

    for( int i = 0; i < array_size; i++ ) {
        std::cout << i
            << ": src = " << src[i]
            << ", sqrt(src) = " << sqrtf(src[i])
            << ", dst = " << dst[i] << std::endl;
    }

    std::cout << "sin:\n";
    {
        buffer<float, 1>  srcBuf{ src.data(), range<1>{array_size} };
        buffer<float, 1>  dstBuf{ dst.data(), range<1>{array_size} };
        q.submit([&](handler& cgh) {
            auto srcAcc = srcBuf.get_access<access::mode::read>(cgh);
            auto dstAcc = dstBuf.get_access<access::mode::write>(cgh);
            cgh.parallel_for(range<1>{array_size}, [=](id<1> i) {
                dstAcc[i] = sin(srcAcc[i]);
            });
        });
    }

    for( int i = 0; i < array_size; i++ ) {
        std::cout << i
            << ": src = " << src[i]
            << ", sin(src) = " << sinf(src[i])
            << ", dst = " << dst[i] << std::endl;
    }

    std::cout << "cos:\n";
    {
        buffer<float, 1>  srcBuf{ src.data(), range<1>{array_size} };
        buffer<float, 1>  dstBuf{ dst.data(), range<1>{array_size} };
        q.submit([&](handler& cgh) {
            auto srcAcc = srcBuf.get_access<access::mode::read>(cgh);
            auto dstAcc = dstBuf.get_access<access::mode::write>(cgh);
            cgh.parallel_for(range<1>{array_size}, [=](id<1> i) {
                dstAcc[i] = cos(srcAcc[i]);
            });
        });
    }

    for( int i = 0; i < array_size; i++ ) {
        std::cout << i
            << ": src = " << src[i]
            << ", cos(src) = " << cosf(src[i])
            << ", dst = " << dst[i] << std::endl;
    }

    return 0;
}