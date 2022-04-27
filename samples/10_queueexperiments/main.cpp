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
    sycl::platform platform;
    sycl::device device;

    sycl::context context;
    sycl::queue queue;

    std::vector<sycl::buffer<float>> buffers;
    std::vector<float*> dptrs;

    int numIterations = 1;
    size_t numElements = 1;
};

class TimeSink {
public:
    TimeSink(sycl::accessor<float> _dst, int _iterations) : dst(_dst), iterations(_iterations) {}
    void operator()(sycl::id<1> i) const {
        float result;
        for (int i = 0; i < iterations; i++) {
            result = 0.0f;
            while (result < 1.0f) result += 1e-6f;
        }
        dst[i] = result;
    }
private:
    sycl::accessor<float> dst;
    int iterations;
};

class TimeSinkUSM {
public:
    TimeSinkUSM(float* _dst, int _iterations) : dst(_dst), iterations(_iterations) {}
    void operator()(sycl::id<1> i) const {
        float result;
        for (int i = 0; i < iterations; i++) {
            result = 0.0f;
            while (result < 1.0f) result += 1e-6f;
        }
        dst[i] = result;
    }
private:
    float* dst;
    int iterations;
};

static void init(Params& params)
{
    for (auto& buffer : params.buffers) {
        params.queue.submit([&](sycl::handler& h) {
            sycl::accessor acc{buffer, h};
            h.fill(acc, 0.0f);
        });
    }
    params.queue.wait();
}

static void go_in_order_queue(Params& params, const int numKernels)
{
    init(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    sycl::queue queue(params.context, params.device, sycl::property::queue::in_order());

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queue.submit([&](sycl::handler& h) {
                sycl::accessor acc{params.buffers[i], h};
                h.parallel_for(params.numElements, TimeSink(acc, params.numIterations));
            });
        }
        queue.wait();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_out_of_order_queue(Params& params, const int numKernels)
{
    init(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    sycl::queue queue(params.context, params.device);

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queue.submit([&](sycl::handler& h) {
                sycl::accessor acc{params.buffers[i], h};
                h.parallel_for(params.numElements, TimeSink(acc, params.numIterations));
            });
        }
        queue.wait();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_multiple_in_order_queues(Params& params, const int numKernels)
{
    init(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    std::vector<sycl::queue> queues;
    for (int i = 0; i < numKernels; i++) {
        queues.push_back(sycl::queue{params.context, params.device, sycl::property::queue::in_order()});
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].submit([&](sycl::handler& h) {
                sycl::accessor acc{params.buffers[i], h};
                h.parallel_for(params.numElements, TimeSink(acc, params.numIterations));
            });
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].wait();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_multiple_out_of_order_queues(Params& params, const int numKernels)
{
    init(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    std::vector<sycl::queue> queues;
    for (int i = 0; i < numKernels; i++) {
        queues.push_back(sycl::queue{params.context, params.device});
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].submit([&](sycl::handler& h) {
                sycl::accessor acc{params.buffers[i], h};
                h.parallel_for(params.numElements, TimeSink(acc, params.numIterations));
            });
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].wait();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_multiple_context_in_order_queues(Params& params, const int numKernels)
{
    init(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    std::vector<sycl::queue> queues;
    for (int i = 0; i < numKernels; i++) {
        queues.push_back(sycl::queue{params.device, sycl::property::queue::in_order()});
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].submit([&](sycl::handler& h) {
                sycl::accessor acc{params.buffers[i], h};
                h.parallel_for(params.numElements, TimeSink(acc, params.numIterations));
            });
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].wait();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void init_usm(Params& params)
{
    for (auto& dptr : params.dptrs) {
        params.queue.fill(dptr, 0.0f, params.numElements);
    }
    params.queue.wait();
}

static void go_in_order_queue_usm(Params& params, const int numKernels)
{
    init_usm(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    sycl::queue queue(params.context, params.device, sycl::property::queue::in_order());

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queue.parallel_for(params.numElements, TimeSinkUSM(params.dptrs[i], params.numIterations));
        }
        queue.wait();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_out_of_order_queue_usm(Params& params, const int numKernels)
{
    init_usm(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    sycl::queue queue(params.context, params.device);

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queue.parallel_for(params.numElements, TimeSinkUSM(params.dptrs[i], params.numIterations));
        }
        queue.wait();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_multiple_in_order_queues_usm(Params& params, const int numKernels)
{
    init_usm(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    std::vector<sycl::queue> queues;
    for (int i = 0; i < numKernels; i++) {
        queues.push_back(sycl::queue{params.context, params.device, sycl::property::queue::in_order()});
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].parallel_for(params.numElements, TimeSinkUSM(params.dptrs[i], params.numIterations));
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].wait();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

static void go_multiple_out_of_order_queues_usm(Params& params, const int numKernels)
{
    init_usm(params);

    printf("%40s (n=%3d): ", __FUNCTION__, numKernels); fflush(stdout);

    std::vector<sycl::queue> queues;
    for (int i = 0; i < numKernels; i++) {
        queues.push_back(sycl::queue{params.context, params.device});
    }

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        for (int i = 0; i < numKernels; i++) {
            queues[i].parallel_for(params.numElements, TimeSinkUSM(params.dptrs[i], params.numIterations));
        }
        for (int i = 0; i < numKernels; i++) {
            queues[i].wait();
        }

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());
    }
    printf("Finished in %f seconds\n", best);
}

int main(int argc, char** argv)
{
    Params params;

    int platformIndex = 0;
    int deviceIndex = 0;
    int numKernels = -1;
    bool testMultipleContext = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<int>>("k", "kernels", "Kernels to Execute (-1 for all)", numKernels, &numKernels);
        op.add<popl::Value<int>>("i", "iterations", "Iterations in Each Kernel", params.numIterations, &params.numIterations);
        op.add<popl::Value<size_t>>("e", "elements", "Number of ND-Range Elements", params.numElements, &params.numElements);
        op.add<popl::Switch>("", "multicontext", "Run the Multiple Context Tests", &testMultipleContext);
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

    if (numKernels > maxKernels) {
        printf("Number of kernels is %d, which exceeds the maximum of %d.\n", numKernels, maxKernels);
        printf("The number of kernels will be set to %d instead.\n", maxKernels);
        numKernels = maxKernels;
    }

    params.platform = sycl::platform::get_platforms()[platformIndex];
    printf("Running on SYCL platform: %s\n", params.platform.get_info<sycl::info::platform::name>().c_str());

    params.device = params.platform.get_devices()[deviceIndex];
    printf("Running on SYCL device: %s\n", params.device.get_info<sycl::info::device::name>().c_str());

    printf("Initializing tests...\n");

    params.context = sycl::context{ params.device };
    params.queue = sycl::queue{ params.context, params.device };

    for (int i = 0; i < maxKernels; i++) {
        params.buffers.push_back(sycl::buffer<float>{sycl::range{params.numElements}});
    }
    if (params.device.get_info<sycl::info::device::usm_device_allocations>()) {
        for (int i = 0; i < maxKernels; i++) {
            params.dptrs.push_back(sycl::malloc_device<float>(params.numElements, params.device, params.context));
        }
    } else {
        printf("Skipping USM tests - device does not support USM.\n");
    }

    printf("... done!\n");

    std::vector<int> counts;
    if (numKernels < 0) {
        counts.assign({1, 2, 4, 8, 16});
    } else {
        counts.assign({numKernels});
    }

    for (auto& count : counts) {
        go_in_order_queue(params, count);
    }
    for (auto& count : counts) {
        go_out_of_order_queue(params, count);
    }
    for (auto& count : counts) {
        go_multiple_in_order_queues(params, count);
    }
    for (auto& count : counts) {
        go_multiple_out_of_order_queues(params, count);
    }
    if (params.device.get_info<sycl::info::device::usm_device_allocations>()) {
        for (auto& count : counts) {
            go_in_order_queue_usm(params, count);
        }
        for (auto& count : counts) {
            go_out_of_order_queue_usm(params, count);
        }
        for (auto& count : counts) {
            go_multiple_in_order_queues_usm(params, count);
        }
        for (auto& count : counts) {
            go_multiple_out_of_order_queues_usm(params, count);
        }
    }

    if (testMultipleContext) {
        for (auto& count : counts) {
            go_multiple_context_in_order_queues(params, count);
        }
    }

    printf("Cleaning up...\n");

    for (auto& dptr : params.dptrs) {
        sycl::free(dptr, params.context);
    }

    printf("... done!\n");

    return 0;
}
