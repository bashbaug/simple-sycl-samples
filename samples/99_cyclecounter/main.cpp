/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <popl/popl.hpp>

#include <stdio.h>

#ifdef __SYCL_DEVICE_ONLY__
extern SYCL_EXTERNAL ulong __attribute__((overloadable)) intel_get_cycle_counter(void);
#endif

ulong get_cycle_counter(void)
{
#ifdef __SYCL_DEVICE_ONLY__
    return intel_get_cycle_counter();
#else
    return 999;
#endif
}

int main(int argc, char** argv )
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
                "Usage: cyclecounter [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    sycl::platform platform = sycl::platform::get_platforms()[platformIndex];
    printf("Running on SYCL platform: %s\n", platform.get_info<sycl::info::platform::name>().c_str());

    sycl::device device = platform.get_devices()[deviceIndex];
    printf("Running on SYCL device: %s\n", device.get_info<sycl::info::device::name>().c_str());

    sycl::queue queue{ device, sycl::property::queue::in_order() };

    auto f = sycl::malloc_host<float>(1, queue);
    auto t = sycl::malloc_host<float>(2, queue);

    if (f && t) {
        *t = 77;
        queue.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> id) {
            ulong start = get_cycle_counter();

            // waste a fair bit of time:
            float reg;
            for (int i = 0; i < 10; i++) {
                reg = 0.f;
                while (reg < 1.f)
                    reg += 1e-7f;
            }
            *f = reg;

            ulong end = get_cycle_counter();
            t[0] = start;
            t[1] = end;
        }).wait();

        std::cout << "Success: start: " << t[0] << ", end: " << t[1] << ", delta: " << t[1] - t[0] << "\n";
    }

    sycl::free(f, queue);
    sycl::free(t, queue);

    return 0;
}
