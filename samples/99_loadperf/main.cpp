/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <popl/popl.hpp>

#include <assert.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using test_clock = std::chrono::steady_clock;

constexpr size_t num_wis = 16 * 1024 * 1024;
constexpr size_t per_wi = 32;

constexpr size_t iterations = 16;

class CopyScalar {
public:
    CopyScalar(int* dst_, int* src_) : dst(dst_), src(src_) {}
    void operator() [[intel::kernel_args_restrict]] (sycl::nd_item<1> item) const {
        auto gid = ( item.get_group(0) * item.get_local_range(0) + item.get_local_id(0) ) * per_wi;

        #pragma unroll
        for (size_t i = 0; i < per_wi; i++) {
            dst[gid] = src[gid];
            gid++;
        }
    }
private:
    int* dst;
    int* src;
};

class CopyCoalesced {
public:
    CopyCoalesced(int* dst_, int* src_) : dst(dst_), src(src_) {}
    void operator() [[intel::kernel_args_restrict]] (sycl::nd_item<1> item) const {
        const auto inc = item.get_local_range(0);
        auto gid = item.get_group(0) * item.get_local_range(0) * per_wi + item.get_local_id(0);

        #pragma unroll
        for (size_t i = 0; i < per_wi; i++) {
            dst[gid] = src[gid];
            gid += inc;
        }
    }
private:
    int* dst;
    int* src;
};

bool checkCopyResults(const std::vector<int>& dst, const std::vector<int>& src) {
    assert(dst.size() == src.size());

    for (size_t i = 0; i < src.size(); i++) {
        auto got = dst[i];
        auto want = src[i];
        if (got != want) {
            std::cout << "Mismatch at index " << i << ", got "
                        << got << ", wanted " << want << "\n";
            return false;
        }
    }

    return true;
}

class PartialReductionScalar {
public:
    PartialReductionScalar(int* dst_, int* src_) : dst(dst_), src(src_) {}
    void operator() [[intel::kernel_args_restrict]] (sycl::nd_item<1> item) const {
        auto srcid = ( item.get_group(0) * item.get_local_range(0) + item.get_local_id(0) ) * per_wi;
        int sum = 0;

        auto rsrc = src + srcid;

        #pragma unroll
        for (size_t i = 0; i < per_wi; i++) {
            sum += rsrc[0];
            rsrc++;
        }

        auto dstid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
        dst[dstid] = sum;
    }
private:
    int* dst;
    int* src;
};

class PartialReductionCoalesced {
public:
    PartialReductionCoalesced(int* dst_, int* src_) : dst(dst_), src(src_) {}
    void operator() [[intel::kernel_args_restrict]] (sycl::nd_item<1> item) const {
        const auto inc = item.get_local_range(0);
        auto srcid = item.get_group(0) * item.get_local_range(0) * per_wi + item.get_local_id(0);
        int sum = 0;

        auto rsrc = src + srcid;

        #pragma unroll
        for (size_t i = 0; i < per_wi; i++) {
            sum += rsrc[0];
            rsrc += inc;
        }

        auto dstid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
        dst[dstid] = sum;
    }
private:
    int* dst;
    int* src;
};

class PartialReductionCoalescedx4 {
public:
    PartialReductionCoalescedx4(int* dst_, int* src_) : dst(dst_), src(src_) {}
    void operator() [[intel::kernel_args_restrict]] (sycl::nd_item<1> item) const {
        const auto inc = item.get_local_range(0) * 4;
        auto srcid = item.get_group(0) * item.get_local_range(0) * per_wi + item.get_local_id(0) * 4;
        int sum = 0;

        auto rsrc = src + srcid;

        #pragma unroll
        for (size_t i = 0; i < per_wi / 4; i++) {
            sum += rsrc[0];
            sum += rsrc[1];
            sum += rsrc[2];
            sum += rsrc[3];
            rsrc += inc;
        }

        auto dstid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
        dst[dstid] = sum;
    }
private:
    int* dst;
    int* src;
};

class PartialReductionBlockRead {
public:
    PartialReductionBlockRead(int* dst_, int* src_) : dst(dst_), src(src_) {}
    void operator() [[intel::kernel_args_restrict]] (sycl::nd_item<1> item) const {
        auto sg = item.get_sub_group();
        const auto inc = sg.get_max_local_range()[0] * 8;
        auto srcoffset = ( item.get_group(0) * item.get_local_range(0) + sg.get_group_linear_id() * sg.get_max_local_range()[0] ) * per_wi;
        int sum = 0;

        auto rsrc = src + srcoffset;

        #pragma unroll
        for (size_t i = 0; i < per_wi / 8; i++) {
            using int8 = sycl::vec<int, 8>;
            int8 v = item.get_sub_group().load<8>(sycl::global_ptr<int>(rsrc));
            sum += v.s0();
            sum += v.s1();
            sum += v.s2();
            sum += v.s3();
            sum += v.s4();
            sum += v.s5();
            sum += v.s6();
            sum += v.s7();
            rsrc += inc;
        }

        auto dstid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
        dst[dstid] = sum;
    }
private:
    int* dst;
    int* src;
};

bool checkPartialReductionResults(const std::vector<int>& dst, const std::vector<int>& src) {
    int src_result = 0; std::for_each(src.cbegin(), src.end(), [&](int n){ src_result += n; });
    int dst_result = 0; std::for_each(dst.cbegin(), dst.end(), [&](int n){ dst_result += n; });

    if (src_result != dst_result) {
        std::cout << "Full reduction: " << src_result << ", check: " << dst_result << "\n";
        return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    size_t mask = ~0;
    size_t num_wgs = 65536;
    size_t wg_size = 256;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<size_t>>("n", "num_wgs", "Number of Work-Groups", num_wgs, &num_wgs);
        op.add<popl::Value<size_t>>("g", "wg_size", "Work-Group Size", wg_size, &wg_size);
        op.add<popl::Value<size_t>, popl::Attribute::advanced>("m", "mask", "Test Mask", mask, &mask);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            std::cerr <<  "Usage: loadperf [options]\n" << op.help();
            return -1;
        }
    }

    const size_t num_wis = num_wgs * wg_size;

    sycl::queue q{sycl::property::queue::in_order()};
    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Number of work-groups: " << num_wgs
              << ", Work-group size: " << wg_size
              << ", Buffer size: " << num_wis * per_wi * sizeof(int) << " ("
              << num_wis * per_wi * sizeof(int) / 1024.0 / 1024.0 / 1024.0
              << "GB)\n";

    std::vector<int> h_src(num_wis * per_wi);
#if 0
    std::iota(h_src.begin(), h_src.end(), 0);
#else
    std::random_device rdev;
    std::mt19937 rng(rdev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
    std::generate(h_src.begin(), h_src.end(), [&]() { return dist(rng); });
#endif

    int* src = sycl::malloc_device<int>(h_src.size(), q);
    q.copy(h_src.data(), src, h_src.size()).wait();

    int* dst = sycl::malloc_device<int>(h_src.size(), q);

    if (mask & 0x1)
    {
        float best = 999.0f;
        for (size_t i = 0; i < iterations; i++) {
            q.fill(dst, 0, h_src.size()).wait();

            auto start = test_clock::now();
            q.parallel_for(sycl::nd_range<1>{num_wis, 256}, CopyScalar(dst, src)).wait();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        std::vector<int> h_dst(num_wis * per_wi);
        q.copy(dst, h_dst.data(), h_dst.size()).wait();
        checkCopyResults(h_dst, h_src);
        auto gbps = num_wis * per_wi * sizeof(int) / best / 1024 / 1024 / 1024;
        std::cout << "Finished scalar copy in " << best << " seconds (" << gbps << " GB/s).\n";
    }

    if (mask & 0x2)
    {
        float best = 999.0f;
        for (size_t i = 0; i < iterations; i++) {
            q.fill(dst, 0, h_src.size()).wait();

            auto start = test_clock::now();
            q.parallel_for(sycl::nd_range<1>{num_wis, 256}, CopyCoalesced(dst, src)).wait();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        std::vector<int> h_dst(num_wis * per_wi);
        q.copy(dst, h_dst.data(), h_dst.size()).wait();
        checkCopyResults(h_dst, h_src);
        auto gbps = num_wis * per_wi * sizeof(int) / best / 1024 / 1024 / 1024;
        std::cout << "Finished coalesced copy in " << best << " seconds (" << gbps << " GB/s).\n";
    }

    if (mask & 0x4)
    {
        float best = 999.0f;
        for (size_t i = 0; i < iterations; i++) {
            q.fill(dst, 0, h_src.size()).wait();

            auto start = test_clock::now();
            q.parallel_for(sycl::nd_range<1>{num_wis, 256}, PartialReductionScalar(dst, src)).wait();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        std::vector<int> h_dst(num_wis);
        q.copy(dst, h_dst.data(), h_dst.size()).wait();
        checkPartialReductionResults(h_dst, h_src);
        auto gbps = num_wis * per_wi * sizeof(int) / best / 1024 / 1024 / 1024;
        std::cout << "Finished scalar partial reduction in " << best << " seconds (" << gbps << " GB/s).\n";
    }

    if (mask & 0x8)
    {
        float best = 999.0f;
        for (size_t i = 0; i < iterations; i++) {
            q.fill(dst, 0, h_src.size()).wait();

            auto start = test_clock::now();
            q.parallel_for(sycl::nd_range<1>{num_wis, 256}, PartialReductionCoalesced(dst, src)).wait();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        std::vector<int> h_dst(num_wis);
        q.copy(dst, h_dst.data(), h_dst.size()).wait();
        checkPartialReductionResults(h_dst, h_src);
        auto gbps = num_wis * per_wi * sizeof(int) / best / 1024 / 1024 / 1024;
        std::cout << "Finished coalesced partial reduction in " << best << " seconds (" << gbps << " GB/s).\n";
    }

    if (mask & 0x10)
    {
        float best = 999.0f;
        for (size_t i = 0; i < iterations; i++) {
            q.fill(dst, 0, h_src.size()).wait();

            auto start = test_clock::now();
            q.parallel_for(sycl::nd_range<1>{num_wis, 256}, PartialReductionCoalescedx4(dst, src)).wait();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        std::vector<int> h_dst(num_wis);
        q.copy(dst, h_dst.data(), h_dst.size()).wait();
        checkPartialReductionResults(h_dst, h_src);
        auto gbps = num_wis * per_wi * sizeof(int) / best / 1024 / 1024 / 1024;
        std::cout << "Finished coalesced partial reduction x4 in " << best << " seconds (" << gbps << " GB/s).\n";
    }

    if (mask & 0x20)
    {
        float best = 999.0f;
        for (size_t i = 0; i < iterations; i++) {
            q.fill(dst, 0, h_src.size()).wait();

            auto start = test_clock::now();
            q.parallel_for(sycl::nd_range<1>{num_wis, 256}, PartialReductionBlockRead(dst, src)).wait();
            auto end = test_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            best = std::min(best, elapsed_seconds.count());
        }
        std::vector<int> h_dst(num_wis);
        q.copy(dst, h_dst.data(), h_dst.size()).wait();
        checkPartialReductionResults(h_dst, h_src);
        auto gbps = num_wis * per_wi * sizeof(int) / best / 1024 / 1024 / 1024;
        std::cout << "Finished coalesced partial reduction block read in " << best << " seconds (" << gbps << " GB/s).\n";
    }

    sycl::free(src, q);
    sycl::free(dst, q);

    std::cout << "Success.\n";
    return 0;
}
