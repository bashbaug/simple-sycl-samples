/*
// Copyright (c) 2021-2022 Ben Ashbaugh
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

#include <sycl.hpp>
#include <iostream>

using namespace sycl;

enum AllocType {
    Device,
    Host,
    Shared
};

int main(
    int argc,
    char** argv )
{
    bool printUsage = false;
    AllocType allocType = Device;
    int pi = 0;
    int di = 0;
    int sz = 2;

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
            else if (!strcmp( argv[i], "-s")) {
                if (++i < argc) {
                    sz = strtol(argv[i], NULL, 10);
                }
            }
            else if (!strcmp( argv[i], "-device")) {
                allocType = Device;
            }
            else if (!strcmp( argv[i], "-host")) {
                allocType = Host;
            }
            else if (!strcmp( argv[i], "-shared")) {
                allocType = Shared;
            }
            else {
                printUsage = true;
            }
        }
    }
    if (printUsage) {
        std::cerr <<
            "Usage: bigalloc  [options]\n"
            "Options:\n"
            "      -d: Device Index (default = 0)\n"
            "      -p: Platform Index (default = 0)\n"
            "      -s: Size to Allocate in GB (default = 2)\n"
            "      -device: Test Device Allocations (default)\n"
            "      -host: Test Host Allocations\n"
            "      -shared: Test Shared Allocations\n"
            ;
        return -1;
    }

    // setup
    queue q{ platform::get_platforms()[pi].get_devices()[di], property::queue::in_order() };

    constexpr float GB = 1024.0f * 1024.0f * 1024.0f;

    auto d = q.get_device();
    auto c = q.get_context();

    std::cout << "Running on SYCL platform: " << 
        d.get_platform().get_info<info::platform::name>() << "\n";
    std::cout << "Running on SYCL device: " << 
        d.get_info<info::device::name>() << "\n";
    std::cout << "For this device:\n";
    std::cout << "\tinfo::device::global_mem_size is " <<
        d.get_info<info::device::global_mem_size>() << " (" <<
        d.get_info<info::device::global_mem_size>() / GB << "GB)\n";
    std::cout << "\tinfo::device::max_mem_alloc_size is " <<
        d.get_info<info::device::max_mem_alloc_size>() << " (" <<
        d.get_info<info::device::max_mem_alloc_size>() / GB << "GB)\n";

    size_t allocSize = (size_t)sz * 1024 * 1024 * 1024 / sizeof(uint32_t);
    size_t gwx = allocSize / 1024;

    std::cout << "Testing allocation size " << sz << " GB (" << allocSize << " uint32_t values).\n";

    auto h_buf = new uint32_t[allocSize];
    auto d_buf = 
        allocType == Device ? (uint32_t*)malloc_device(allocSize * sizeof(uint32_t), d, c) :
        allocType == Host ? (uint32_t*)malloc_host(allocSize * sizeof(uint32_t), c) :
        allocType == Shared ? (uint32_t*)malloc_shared(allocSize * sizeof(uint32_t), d, c) :
        nullptr;

    if (h_buf && d_buf) {
        // init

        for( size_t i = 0; i < allocSize; i++ ) {
            h_buf[i] = (uint32_t)i;
        }

        q.memcpy(d_buf, h_buf, allocSize * sizeof(uint32_t));

        // go

        q.parallel_for(range<1>{gwx}, [=](id<1> id) {
            for(size_t i = 0; i < 1024; i++) {
                d_buf[id * 1024 + i] += 2;
            }
        });

        // check results

        q.memcpy(h_buf, d_buf, allocSize * sizeof(uint32_t)).wait();    // blocking

        unsigned int    mismatches = 0;
        for( size_t i = 0; i < allocSize; i++ ) {
            uint32_t want = i + 2;
            if( h_buf[i] != want ) {
                if( mismatches < 16 ) {
                    std::cerr << "MisMatch!  dst[" << i << "] == "
                        << h_buf[i] << ", want "
                        << want << "\n";
                }
                mismatches++;
            }
        }

        if( mismatches ) {
            std::cerr << "Error: Found " << mismatches << " mismatches / " << allocSize << " values!!!\n";
        }
        else {
            std::cout << "Success.\n";
        }
    }
    else {
        std::cerr << "Allocation failed!  h_buf = " << h_buf << ", d_buf == " << d_buf << "\n";
    }

    // clean up
    delete [] h_buf;
    free(d_buf, c);

    return 0;
}
