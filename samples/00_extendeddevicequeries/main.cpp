/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main()
{
    for( auto& p : platform::get_platforms() )
    {
        std::cout << std::endl << "SYCL Platform: " << p.get_info<info::platform::name>() << std::endl;

        for( auto& d : p.get_devices() )
        {
            std::cout << "SYCL Device: " << d.get_info<info::device::name>() << std::endl;
            printf("\tVendor ID:       %04X\n", d.get_info<info::device::vendor_id>());
#if SYCL_EXT_INTEL_DEVICE_INFO >= 1
            if (d.has(aspect::ext_intel_device_id)) {
                printf("\tDevice ID:       %04X\n", d.get_info<ext::intel::info::device::device_id>());
            }
#endif
            printf("\tMax Sub-Devices: %u\n", d.get_info<info::device::partition_max_sub_devices>());
#if SYCL_EXT_INTEL_DEVICE_INFO >= 3
            if (d.has(aspect::ext_intel_gpu_slices)) {
                printf("\tNum Slices:      %u\n", d.get_info<ext::intel::info::device::gpu_slices>());
            }
            if (d.has(aspect::ext_intel_gpu_slices) &&
                d.has(aspect::ext_intel_gpu_subslices_per_slice)) {
                printf("\tNum Sub-Slices:  %u\n",
                    d.get_info<ext::intel::info::device::gpu_slices>() *
                    d.get_info<ext::intel::info::device::gpu_subslices_per_slice>());
            }
            if (d.has(aspect::ext_intel_gpu_slices) && 
                d.has(aspect::ext_intel_gpu_subslices_per_slice) &&
                d.has(aspect::ext_intel_gpu_eu_count_per_subslice)) {
                printf("\tNum EUs:         %u\n",
                    d.get_info<ext::intel::info::device::gpu_slices>() *
                    d.get_info<ext::intel::info::device::gpu_subslices_per_slice>() *
                    d.get_info<ext::intel::info::device::gpu_eu_count_per_subslice>());
            }
#endif
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
            if (d.has(aspect::ext_intel_device_info_uuid)) {
                auto deviceUUID = d.get_info<ext::intel::info::device::uuid>();
                printf("\tDevice UUID:     %02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X\n",
                    deviceUUID[0], deviceUUID[1], deviceUUID[2], deviceUUID[3],
                    deviceUUID[4], deviceUUID[5], deviceUUID[6], deviceUUID[7],
                    deviceUUID[8], deviceUUID[9], deviceUUID[10], deviceUUID[11],
                    deviceUUID[12], deviceUUID[13], deviceUUID[14], deviceUUID[15]);
            }
#endif
        }
    }

    return 0;
}
