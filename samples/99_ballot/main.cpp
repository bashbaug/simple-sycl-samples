/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <popl/popl.hpp>

#include <stdio.h>

using namespace sycl;

int main(int argc, char** argv)
{
    const size_t sz = 1024;

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
                "Usage: ballot [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    platform platform = platform::get_platforms()[platformIndex];
    printf("Running on SYCL platform: %s\n", platform.get_info<info::platform::name>().c_str());

    device device = platform.get_devices()[deviceIndex];
    printf("Running on SYCL device: %s\n", device.get_info<info::device::name>().c_str());

    context context{ device };
    queue q{ context, device };

    uint32_t* ptr = malloc<uint32_t>(sz, q, usm::alloc::host);
    q.parallel_for(nd_range<1>{sz, 256}, [=](auto item) {
        auto index = item.get_global_id(0);
        // Compute some predicate to have something to get the ballot for:
        bool p = index & 1 || index < 8;
        // Get the ballot as a mask:
        auto sg = item.get_sub_group();
        auto mask = ext::oneapi::group_ballot(sg, p);
        // Extract the raw bits from the ballot.
        // We can use a uint32_t here if we know our sub-group size is <= 32.
        uint32_t val = 0;
        mask.extract_bits(val);
        // Write the ballot bits to memory.
        ptr[index] = val;
    }).wait();

    printf("First few values: %08X %08X %08X %08X\n", ptr[0], ptr[1], ptr[2], ptr[3]);

    free(ptr, q);
    ptr = nullptr;

    printf("... done!\n");

    return 0;
}
