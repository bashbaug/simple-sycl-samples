/*
// Copyright (c) 2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <popl/popl.hpp>

auto handle_async_error = [](sycl::exception_list elist) {
    for (auto &e : elist) {
        try {
            std::rethrow_exception(e);
        } catch (...) {
            std::cout << "Caught async SYCL exception!!\n";
        }
    }
};

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t gwx = 1 << 20;
    size_t lwx = 16;
    size_t divisor = 1 << 10;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size", gwx, &gwx);
        op.add<popl::Value<size_t>>("", "lwx", "Local Work Size", lwx, &lwx);
        op.add<popl::Value<size_t>>("", "divisor", "Divisor", divisor, &divisor);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: bigrange [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    try
    {
        sycl::platform platform = sycl::platform::get_platforms()[platformIndex];
        printf("Running on SYCL platform: %s\n", platform.get_info<sycl::info::platform::name>().c_str());

        sycl::device device = platform.get_devices()[deviceIndex];
        printf("Running on SYCL device: %s\n", device.get_info<sycl::info::device::name>().c_str());

        #if defined(SYCL_KHR_MAX_WORK_GROUP_QUERIES)
            printf("For this device, max_work_group_range_size is: %zu\n",
                device.get_info<sycl::khr::info::device::max_work_group_range_size>());
            printf("    max_work_group_range is: %zu, %zu, %zu\n",
                device.get_info<sycl::khr::info::device::max_work_group_range>()[0],
                device.get_info<sycl::khr::info::device::max_work_group_range>()[1],
                device.get_info<sycl::khr::info::device::max_work_group_range>()[2]);
        #elif defined(SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY)
            printf("For this device, max_global_work_groups is: %zu\n",
                device.get_info<sycl::ext::oneapi::experimental::info::device::max_global_work_groups>());
            printf("    max_work_groups is: %zu, %zu, %zu\n",
                device.get_info<sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>()[0],
                device.get_info<sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>()[1],
                device.get_info<sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>()[2]);
        #else
            printf("No max work group query support is available.\n");
        #endif

        sycl::context context = sycl::context{ device };
        sycl::queue queue = sycl::queue{ context, device, handle_async_error, sycl::property::queue::in_order() };

        printf("Allocating and initializing memory...\n");

        const size_t sz = gwx / divisor;

        int* pDevTotal = sycl::malloc_device<int>(1, queue);
        int* pDevCheck = sycl::malloc_device<int>(sz, queue);

        queue.fill(pDevTotal, 0, 1).wait_and_throw();
        queue.fill(pDevCheck, 0, sz).wait_and_throw();

        printf("Launching a kernel with global work size %zu and local work size %zu...\n", gwx, lwx);
        printf("Note: This is %zu work-groups.\n", gwx / lwx);

        queue.parallel_for(sycl::nd_range<1>{gwx, lwx}, [=](sycl::nd_item<1> item) {
            const size_t global_id = item.get_global_id(0);
            if (global_id % divisor == 0) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> atomicTotal(pDevTotal[0]);
                atomicTotal.fetch_add(1);
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> atomicCheck(pDevCheck[global_id / divisor]);
                atomicCheck.fetch_add(1);
            }
        }).wait_and_throw();

        printf("Reading results...\n");

        int hostTotal = 0;
        std::vector<int> hostCheck(sz);
        queue.copy(pDevTotal, &hostTotal, 1).wait();
        queue.copy(pDevCheck, hostCheck.data(), sz).wait();

        printf("total = %d (should be %zu)\n", hostTotal, sz);

        bool match = true;
        for (size_t i = 0; i < sz; i++) {
            if (hostCheck[i] != 1) {
                printf("check[%zu] = %d (should be 1)\n", i, hostCheck[i]);
                match = false;
            }
        }

        if (match) {
            printf("All checks passed.\n");
        }

        printf("Freeing memory...\n");
        sycl::free(pDevTotal, queue);
        sycl::free(pDevCheck, queue);

        printf("Done.\n");
    }
    catch (sycl::exception &e)
    {
        std::cout << "Caught SYCL exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
