/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <popl/popl.hpp>

#include <stdio.h>

#include <chrono>

using test_clock = std::chrono::high_resolution_clock;

constexpr int testIterations = 32;
constexpr int groupSize = 256;
constexpr int numShuffles = 512;

struct Params
{
    sycl::platform platform;
    sycl::device device;

    sycl::context context;
    sycl::queue queue;

    size_t numGroups = 64 * 1024;
};

class BroadcastUniform {
public:
    BroadcastUniform(sycl::accessor<float> _dst) : dst(_dst) {}
    void operator()(sycl::nd_item<1> item) const {
        auto index = item.get_global_id(0);
        auto sg = item.get_sub_group();

        int sglid = sg.get_local_id();
        int gid = item.get_group(0);
        int mask = gid > 10000000 ? 0xFFFF : 0; // should always be zero, uniform

        float f = sglid;

        #pragma unroll
        for (int i = 5; i < 5 + numShuffles; i++) {
            // varying = varying + uniform
            f = f + sycl::group_broadcast(sg, f, i & mask);
        }

        dst[index] += f;
    }
private:
    sycl::accessor<float> dst;
};

class Broadcast {
public:
    Broadcast(sycl::accessor<float> _dst) : dst(_dst) {}
    void operator()(sycl::nd_item<1> item) const {
        auto index = item.get_global_id(0);
        auto sg = item.get_sub_group();

        int sglid = sg.get_local_id();
        int mask = index > 10000000 ? 0xFFFF : 0;   // should always be zero, varying

        float f = sglid;

        #pragma unroll
        for (int i = 5; i < 5 + numShuffles; i++) {
            // varying = varying + uniform?
            f = f + sycl::group_broadcast(sg, f, i & mask);
        }

        dst[index] += f;
    }
private:
    sycl::accessor<float> dst;
};

class ShuffleUniform {
public:
    ShuffleUniform(sycl::accessor<float> _dst) : dst(_dst) {}
    void operator()(sycl::nd_item<1> item) const {
        auto index = item.get_global_id(0);
        auto sg = item.get_sub_group();

        int sglid = sg.get_local_id();
        int gid = item.get_group(0);
        int mask = gid > 10000000 ? 0xFFFF : 0; // should always be zero, uniform

        float f = sglid;

        #pragma unroll
        for (int i = 5; i < 5 + numShuffles; i++) {
            // varying = varying + uniform
            f = f + sycl::select_from_group(sg, f, i & mask);
        }

        dst[index] += f;
    }
private:
    sycl::accessor<float> dst;
};

class ShuffleNonUniform {
public:
    ShuffleNonUniform(sycl::accessor<float> _dst) : dst(_dst) {}
    void operator()(sycl::nd_item<1> item) const {
        auto index = item.get_global_id(0);
        auto sg = item.get_sub_group();

        int sglid = sg.get_local_id();
        int mask = index > 10000000 ? 0xFFFF : 0;   // should always be zero, varying

        float f = sglid;

        #pragma unroll
        for (int i = 5; i < 5 + numShuffles; i++) {
            // varying = varying + varying
            f += sycl::select_from_group(sg, f, i & mask);
        }

        dst[index] += f;
    }
private:
    sycl::accessor<float> dst;
};

template<typename Func>
void run_test(Params& params, sycl::buffer<float>& buffer)
{
    params.queue.submit([&](sycl::handler& h) {
        sycl::accessor acc{buffer, h};
        h.fill(acc, 0.0f);
    }).wait();

    printf("%20s: ", typeid(Func).name()); fflush(stdout);

    float best = 999.9f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        params.queue.submit([&](sycl::handler& h) {
            sycl::accessor acc{buffer, h};
            h.parallel_for(
                sycl::nd_range<1>{params.numGroups * groupSize, groupSize},
                Func(acc));
        }).wait();
        
        auto end = test_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        best = std::min(best, elapsed_seconds.count());

        if (test == 0) {
            sycl::host_accessor hp{buffer};
            printf("First few values: %f, %f, %f: ", hp[0], hp[1], hp[2]);
        }
    }
    printf("Finished in %f seconds\n", best);
}

int main(int argc, char** argv)
{
    Params params;

    int deviceIndex = 0;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("n", "groups", "Number of ND-Range Groups", params.numGroups, &params.numGroups);        
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: shuffleexperiments [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    auto devices = sycl::device::get_devices();

    if (deviceIndex > devices.size()) {
        fprintf(stderr, "Error: device index %d is unavailable, only %zu devices found.\n",
            deviceIndex, devices.size());
        return -1;
    }

    params.device = sycl::device::get_devices()[deviceIndex];
    params.platform = params.device.get_platform();

    printf("Running on SYCL platform: %s\n", params.platform.get_info<sycl::info::platform::name>().c_str());
    printf("Running on SYCL device: %s\n", params.device.get_info<sycl::info::device::name>().c_str());

    printf("Initializing tests...\n");

    params.context = sycl::context{ params.device };
    params.queue = sycl::queue{ params.context, params.device, sycl::property::queue::in_order() };

    sycl::buffer<float> buffer{ sycl::range{params.numGroups * groupSize} };

    printf("... done!\n");

    run_test<BroadcastUniform>(params, buffer);
    run_test<Broadcast>(params, buffer);
    run_test<ShuffleUniform>(params, buffer);
    run_test<ShuffleNonUniform>(params, buffer);

    printf("Cleaning up...\n");
    printf("... done!\n");

    return 0;
}
