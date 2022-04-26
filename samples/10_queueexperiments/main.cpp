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

#include <stdio.h>

#include <chrono>

using namespace cl;

using test_clock = std::chrono::high_resolution_clock;

constexpr int maxKernels = 64;
constexpr int testIterations = 32;

struct Params
{
    sycl::context context;
    int numKernels = 8;
    int numIterations = 1;
    size_t numElements = 1;
};

int main(int argc, char** argv)
{
    Params params;

    int platformIndex = 0;
    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<int>>("k", "kernels", "Kernel to Execute", params.numKernels, &params.numKernels);
        op.add<popl::Value<int>>("i", "iterations", "Kernel Iterations", params.numIterations, &params.numIterations);
        op.add<popl::Value<size_t>>("e", "elements", "Number of ND-Range Elements", params.numElements, &params.numElements);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: queueexperiments [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    sycl::platform p = sycl::platform::get_platforms()[platformIndex];
    printf("Running on SYCL platform: %s\n", p.get_info<sycl::info::platform::name>().c_str());

    sycl::device d = p.get_devices()[deviceIndex];
    printf("Running on SYCL device: %s\n", d.get_info<sycl::info::device::name>().c_str());

    params.context = sycl::context{ d };

    return 0;
}