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
#include <iostream>

using namespace cl::sycl;

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
            "Usage: hmemhelloworld  [options]\n"
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

    auto h_src = (uint32_t*)malloc_host(gwx * sizeof(uint32_t), c);
    auto h_dst = (uint32_t*)malloc_host(gwx * sizeof(uint32_t), c);

    if (h_src && h_dst) {
        // init

        for( size_t i = 0; i < gwx; i++ ) {
            h_src[i] = (uint32_t)i;
        }
        memset(h_dst, 0, gwx * sizeof(uint32_t));

        // go

        q.parallel_for(range<1>{gwx}, [=](id<1> id) {
            h_dst[id] = h_src[id];
        });

        // check results

        q.wait();

        unsigned int    mismatches = 0;
        for( size_t i = 0; i < gwx; i++ ) {
            if( h_dst[i] != i ) {
                if( mismatches < 16 ) {
                    std::cerr << "MisMatch!  dst[" << i << "] == "
                        << h_dst[i] << ", want "
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
    free(h_src, c);
    free(h_dst, c);

    return 0;
}
