/*
// Copyright (c) 2023 Ben Ashbaugh
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

#include <sycl/sycl.hpp>
#include <popl/popl.hpp>

#include <algorithm>

class TestKernel {
public:
    TestKernel(int* dst0, int* dst1, int* dst2, int* dst3, int* dst4, int* dst5, int* dst6, int* dst7) :
        dst0(dst0), dst1(dst1), dst2(dst2), dst3(dst3), dst4(dst4), dst5(dst5), dst6(dst6), dst7(dst7) {}
    void operator()(sycl::id<1> id) const {
        dst0[id] = dst1[id] = dst2[id] = dst3[id] = dst4[id] = dst5[id] = dst6[id] = dst7[id] = id;
    }
private:
    int *dst0, *dst1, *dst2, *dst3, *dst4, *dst5, *dst6, *dst7;
};

int main(int argc, char** argv)
{
    int deviceIndex = 0;
    bool useHost = false;
    bool useManaged = false;

    size_t allocSize = 2;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("s", "sz", "Size per Allocation (GB)", allocSize, &allocSize);
        op.add<popl::Switch>("h", "host", "Use Host Allocations", &useHost);
        op.add<popl::Switch>("m", "managed", "Use Managed Allocations", &useManaged);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: oversubscribe [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    allocSize *= (1024UL * 1024UL * 1024UL);

    auto devices = sycl::device::get_devices();
    if (deviceIndex > devices.size()) {
        fprintf(stderr, "Error: device index %d is unavailable, only %zu devices found.\n",
            deviceIndex, devices.size());
        return -1;
    }

    auto device = devices[deviceIndex];
    auto platform = device.get_platform();

    printf("Running on SYCL platform: %s\n", platform.get_info<sycl::info::platform::name>().c_str());
    printf("Running on SYCL device: %s\n", device.get_info<sycl::info::device::name>().c_str());

    auto deviceGlobalMemSize = device.get_info<sycl::info::device::global_mem_size>(); 
    printf("Total global memory for device: %zu (%f GB)\n",
        deviceGlobalMemSize,
        deviceGlobalMemSize / 1024.0 / 1024.0 / 1024.0);
    if (useHost) {
        printf("Using host allocations.\n");
    } else if (useManaged) {
        printf("Using managed allocations.\n");
    } else {
        printf("Using device allocations.\n");
    }

    auto queue = sycl::queue{ device, sycl::property::queue::in_order() };

    std::vector<int*> allocs(8);

    size_t total = 0;

    for (int i = 0; i < 8; i++) {
        int* buffer = NULL;

        if (useHost) {
            buffer = sycl::malloc_host<int>(allocSize / sizeof(int), queue);
        } else if (useManaged) {
            buffer = sycl::malloc_shared<int>(allocSize / sizeof(int), queue);
        } else {
            buffer = sycl::malloc_device<int>(allocSize / sizeof(int), queue);
        }

        queue.memset(buffer, 0, allocSize);

        total += allocSize;

        printf("Total allocations so far: %zu bytes (%.1f GB).\n",
            total,
            total / 1024.0 / 1024.0 / 1024.0);

        //size_t freeMem = 0, totalMem = 0;
        //checkCudaErrors(cudaMemGetInfo(&freeMem, &totalMem));

        //printf("Free memory: %zu bytes (%.1f GB).\n",
        //    freeMem,
        //    freeMem / 1024.0 / 1024.0 / 1024.0);

        for (int b = i; b < allocs.size(); b++) {
            allocs[b] = buffer;
        }

        queue.parallel_for(256, TestKernel(
            allocs[0], allocs[1], allocs[2], allocs[3],
            allocs[4], allocs[5], allocs[6], allocs[7]));
        queue.wait();

        printf("Kernel ran successfully.\n");
    }

    printf("Freeing memory...\n");

    for (int b = 0; b < allocs.size(); b++) {
        sycl::free(allocs[b], queue);
        allocs[b] = nullptr;
    }

    printf("Done.\n");

    return 0;
}
