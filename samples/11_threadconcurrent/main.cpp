/*
// Copyright (c) 2022 Ben Ashbaugh & Nico Galoppo
//
// SPDX-License-Identifier: MIT
*/

#include <CL/sycl.hpp>
#include <popl/popl.hpp>

#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <thread>

using namespace cl;
using test_clock = std::chrono::high_resolution_clock;

constexpr int maxThreads = 2;
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
        dst[i] += result;
    }
private:
    sycl::accessor<float> dst;
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

static void go(Params& params, const int kernelNum)
{
    init(params);

    sycl::queue queue(params.context, params.device);

    float total = .0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();

        queue.submit([&](sycl::handler& h) {
            sycl::accessor acc{params.buffers[kernelNum], h};
            h.parallel_for(params.numElements, TimeSink(acc, params.numIterations));
        });

        queue.wait();

        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        total += elapsed_seconds.count();
    }
    printf("%40s (i=%3d): ", __FUNCTION__, kernelNum); fflush(stdout);
    printf("Average time: %f seconds\n", total / testIterations);
}

static void go2(Params& params, const int kernelNum)
{
    init(params);

    sycl::queue queue(params.context, params.device);

    float total = .0f;
    auto start = test_clock::now();
    for (int test = 0; test < testIterations; test++) {
        queue.submit([&](sycl::handler& h) {
            sycl::accessor acc{params.buffers[kernelNum], h};
            h.parallel_for(params.numElements, TimeSink(acc, params.numIterations));
        });
    }
    queue.wait();

    auto end = test_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    printf("%40s (i=%3d): ", __FUNCTION__, kernelNum); fflush(stdout);
    printf("Average time: %f seconds\n", elapsed_seconds.count() / testIterations);
}

int main(int argc, char** argv)
{
    Params params;

    int platformIndex = 0;
    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        //op.add<popl::Value<int>>("t", "threads", "Threads to Execute", numThreads, &numThreads);
        op.add<popl::Value<int>>("i", "iterations", "Iterations in Each Kernel", params.numIterations, &params.numIterations);
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
                "Usage: thread_concurrency [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    //if (numThreads > maxThreads) {
        //printf("Number of kernels is %d, which exceeds the maximum of %d.\n", numKernels, maxKernels);
        //printf("The number of kernels will be set to %d instead.\n", maxKernels);
        //numKernels = maxKernels;
    //}

    params.platform = sycl::platform::get_platforms()[platformIndex];
    printf("Running on SYCL platform: %s\n", params.platform.get_info<sycl::info::platform::name>().c_str());

    params.device = params.platform.get_devices()[deviceIndex];
    printf("Running on SYCL device: %s\n", params.device.get_info<sycl::info::device::name>().c_str());

    printf("Initializing tests...\n");

    params.context = sycl::context{ params.device };
    params.queue = sycl::queue{ params.context, params.device };

    for (int i = 0; i < maxThreads; i++) {
        params.buffers.push_back(sycl::buffer<float>{sycl::range{params.numElements}});
    }

    printf("... done!\n");

    printf("Testing without threads\n");
    go(params, 0);

    printf("Testing with threads\n");
    {
        std::thread t([params]() mutable {
            go(params, 0);
        });

        //usleep( 100000 );

        go(params, 1);

        t.join();
    }

    printf("Testing with threads 2\n");
    {
        std::thread t([params]() mutable {
            go2(params, 0);
        });

        //usleep( 100000 );

        go2(params, 1);

        t.join();
    }

    printf("Cleaning up...\n");

    for (auto& dptr : params.dptrs) {
        sycl::free(dptr, params.context);
    }

    printf("... done!\n");

    return 0;
}
