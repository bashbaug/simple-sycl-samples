/*
// Copyright (c) 2022 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/sycl.hpp>
#include <popl/popl.hpp>

#include <stdio.h>
#include <chrono>

#include "bmp.hpp"

const char* filename = "julia.bmp";

using namespace cl;
using test_clock = std::chrono::high_resolution_clock;

class Julia {
public:
    Julia(sycl::uchar4* _dst, float _cr, float _ci) : dst(_dst), cr(_cr), ci(_ci) {}
    void operator()(sycl::item<2> item) const {
        const float cMinX = -1.5f;
        const float cMaxX =  1.5f;
        const float cMinY = -1.5f;
        const float cMaxY =  1.5f;

        const int cWidth = item.get_range().get(1);
        const int cIterations = 16;

        int x = item.get_id().get(1);
        int y = item.get_id().get(0);

        float a = x * ( cMaxX - cMinX ) / cWidth + cMinX;
        float b = y * ( cMaxY - cMinY ) / cWidth + cMinY;

        float result = 0.0f;
        const float thresholdSquared = cIterations * cIterations / 64.0f;

        for( int i = 0; i < cIterations; i++ ) {
            float aa = a * a;
            float bb = b * b;

            float magnitudeSquared = aa + bb;
            if( magnitudeSquared >= thresholdSquared ) {
                break;
            }

            result += 1.0f / cIterations;
            b = 2 * a * b + ci;
            a = aa - bb + cr;
        }

        result = sycl::max( result, 0.0f );
        result = sycl::min( result, 1.0f );

        // BGRA
        sycl::float4 color( 1.0f, sycl::sqrt(result), result, 1.0f );

        color *= 255.0f;

        dst[ y * cWidth + x ] = color.convert<sycl::uchar>();
    }
private:
    sycl::uchar4* dst;
    float cr;
    float ci;
};

int main(int argc, char** argv)
{
    int platformIndex = 0;
    int deviceIndex = 0;

    size_t outer = 4;
    size_t iterations = 16;
    size_t scale = 1;
    size_t gwx = 512;
    size_t gwy = 512;
    size_t lwx = 0;
    size_t lwy = 0;
    bool useHostUSM = false;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("o", "outer", "Outer Iterations", outer, &outer);
        op.add<popl::Value<size_t>>("i", "iterations", "Inner Iterations", iterations, &iterations);
        op.add<popl::Value<size_t>>("m", "memscale", "Memory allocation Scale", scale, &scale);
        op.add<popl::Value<size_t>>("", "gwx", "Global Work Size X AKA Image Width", gwx, &gwx);
        op.add<popl::Value<size_t>>("", "gwy", "Global Work Size Y AKA Image Height", gwy, &gwy);
        op.add<popl::Value<size_t>>("", "lwx", "Local Work Size X", lwx, &lwx);
        op.add<popl::Value<size_t>>("", "lwy", "Local Work Size Y", lwy, &lwy);
        op.add<popl::Switch>("h", "hostmem", "Use Host USM", &useHostUSM);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: julia [options]\n"
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

    sycl::uchar4* ptr = sycl::malloc<sycl::uchar4>(
        scale * gwx * gwy,
        device,
        context,
        useHostUSM ? sycl::usm::alloc::host : sycl::usm::alloc::shared);

    // Touch the allocation on the host to cause a transfer.
    ptr[0] = 1;

    auto start = test_clock::now();
    for (int o = 0; o < outer; o++) {
        if (lwx == 0 && lwy == 0) {
            for (int i = 0; i < iterations; i++) {
                queue.parallel_for({gwx, gwy}, Julia(ptr, -0.123f, 0.745f));
            }
        }
        else {
            for (int i = 0; i < iterations; i++) {
                queue.parallel_for(sycl::nd_range<2>{{gwx, gwy}, {lwx, lwy}}, Julia(ptr, -0.123f, 0.745f));
            }
        }
        queue.wait();
        ptr[0] = 1;
    }
    auto end = test_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    printf("Finished in %f seconds\n", elapsed_seconds.count());

    BMP::save_image(reinterpret_cast<const uint32_t*>(ptr), gwx, gwy, filename);
    printf("Wrote image file %s\n", filename);

    printf("... done!\n");

    return 0;
}
