/*
// Copyright (c) 2020 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

const size_t  gwx = 1024*1024;

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;
    int pi = 0;
    int di = 0;

    if (argc < 1) {
        printUsage = true;
    }
    else {
        for (size_t i = 1; i < argc; i++) {
            if (!strcmp( argv[i], "-d" )) {
                if (++i < argc) {
                    di = strtol(argv[i], NULL, 10);
                }
            }
            else if (!strcmp( argv[i], "-p")) {
                if (++i < argc) {
                    pi = strtol(argv[i], NULL, 10);
                }
            }
            else {
                printUsage = true;
            }
        }
    }
    if (printUsage) {
        std::cerr <<
            "Usage: sysmemhelloworld  [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            ;
        return -1;
    }

    // setup
    queue q{ platform::get_platforms()[pi].get_devices()[di], property::queue::in_order() };

    auto d = q.get_device();
    auto c = q.get_context();

    std::cout << "Running on SYCL platform: " << 
        d.get_platform().get_info<info::platform::name>() << std::endl;
    std::cout << "Running on SYCL device: " << 
        d.get_info<info::device::name>() << std::endl;

    auto s_src = (uint32_t*)malloc(gwx * sizeof(uint32_t));
    auto s_dst = (uint32_t*)malloc(gwx * sizeof(uint32_t));

    if (s_src && s_dst) {
        // init

        for( size_t i = 0; i < gwx; i++ ) {
            s_src[i] = (uint32_t)i;
        }
        memset(s_dst, 0, gwx * sizeof(uint32_t));

        // go

        q.parallel_for(range<1>{gwx}, [=](id<1> id) {
            s_dst[id] = s_src[id];
        });

        // check results

        q.wait();

        unsigned int    mismatches = 0;
        for( size_t i = 0; i < gwx; i++ ) {
            if( s_dst[i] != i ) {
                if( mismatches < 16 ) {
                    std::cerr << "MisMatch!  dst[" << i << "] == "
                        << s_dst[i] << ", want "
                        << i << "\n";
                }
                mismatches++;
            }
        }

        if( mismatches ) {
            std::cerr << "Error: Found "
                << mismatches << " mismatches / " << gwx << " values!!!\n";
        }
        else {
            std::cout << "Success.\n";
        }
    }

    // clean up
    free(s_src);
    free(s_dst);

    return 0;
}
