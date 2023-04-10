/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <popl/popl.hpp>

#include <numeric>
#include <stdio.h>

using namespace sycl;

[[intel::device_indirectly_callable]]
ext::oneapi::experimental::simd<int, 8> my_inc(ext::oneapi::experimental::simd<int, 8> x, int n)
{
    //return x + n;
    return x;
}

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
                "Usage: invokesimd [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    sycl::platform platform = sycl::platform::get_platforms()[platformIndex];
    printf("Running on SYCL platform: %s\n", platform.get_info<sycl::info::platform::name>().c_str());

    sycl::device device = platform.get_devices()[deviceIndex];
    printf("Running on SYCL device: %s\n", device.get_info<sycl::info::device::name>().c_str());

    sycl::context context = sycl::context{ device };
    sycl::queue queue = sycl::queue{ context, device, sycl::property::queue::in_order() };

    const size_t size = 256;
    std::array<int, size> data;
    std::iota(data.begin(), data.end(), 0);

    {
        buffer buf{data};

        queue.submit([&](handler& h) {
            accessor acc{buf, h};
            h.parallel_for(nd_range{{size}, {32}}, [=](auto& item) [[sycl::reqd_sub_group_size(8)]] {
                auto sg = item.get_sub_group();
                auto i = item.get_global_id(0);
                auto value = acc[i];
                value = ext::oneapi::experimental::invoke_simd(sg, my_inc, value, ext::oneapi::experimental::uniform(1));
                acc[i] = value;
            });
        }).wait();
    }

    for (int i = 0; i < size; i++) {
        if (data[i] != i + 1) {
            printf("Mismatch at index %d!  Got %d, wanted %d.\n", i, data[i], i + 1);
            return -1;
        }
    }

    std::cout << "Success!\n";
    return 0;
}
