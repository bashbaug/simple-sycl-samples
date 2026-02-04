/*
// Copyright (c) 2026 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>

#include <iostream>

auto handle_async_error = [](sycl::exception_list elist) {
    for (auto &e : elist) {
        try {
            std::rethrow_exception(e);
        } catch (...) {
            std::cout << "Caught SYCL ASYNC exception!!\n";
        }
    }
};

template <typename T>
void fill_matrix(std::vector<T>& M, size_t numRows, size_t numCols)
{
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            T value = static_cast<T>(((r % 256) * 65536) + (c % 256));
            M[r * numCols + c] = value;
        }
    }
}

template <>
void fill_matrix(std::vector<uint8_t>& M, size_t numRows, size_t numCols)
{
    uint8_t value = 0;
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            M[r * numCols + c] = value++;
        }
    }
}

template <typename T>
void print_matrix(std::vector<T>& M, size_t numRows, size_t numCols)
{
    for (size_t r = 0; r < numRows; r++) {
        for (size_t c = 0; c < numCols; c++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<uint32_t>(M[r * numCols + c]) << " ";
        }
        std::cout << std::endl;
    }
}

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N>
using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N>
using vector_t = sycl::vec<T, N>;
#endif

using coord_t = vector_t<int, 2>;

SYCL_EXTERNAL void __attribute__((convergent)) __spirv_Subgroup2DBlockLoadINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_base_pointer, int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate,
    void* dst_pointer);
SYCL_EXTERNAL void __attribute__((convergent)) __spirv_Subgroup2DBlockLoadTransposeINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_base_pointer, int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate,
    void* dst_pointer);
SYCL_EXTERNAL void __attribute__((convergent)) __spirv_Subgroup2DBlockLoadTransformINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_base_pointer, int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate,
    void* dst_pointer);
SYCL_EXTERNAL void __attribute__((convergent)) __spirv_Subgroup2DBlockPrefetchINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_base_pointer, int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate);
SYCL_EXTERNAL void __attribute__((convergent)) __spirv_Subgroup2DBlockStoreINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_pointer,
    void* dst_base_pointer,
    int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate);

#ifndef __SYCL_DEVICE_ONLY__
void __spirv_Subgroup2DBlockLoadINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_base_pointer, int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate,
    void* dst_pointer) { __builtin_unreachable(); }
void __spirv_Subgroup2DBlockLoadTransposeINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_base_pointer, int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate,
    void* dst_pointer) { __builtin_unreachable(); }
void __spirv_Subgroup2DBlockLoadTransformINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_base_pointer, int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate,
    void* dst_pointer) { __builtin_unreachable(); }
void __spirv_Subgroup2DBlockPrefetchINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_base_pointer, int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate) { __builtin_unreachable(); }
void __spirv_Subgroup2DBlockStoreINTEL(
    int element_size, int block_width, int block_height, int block_count,
    const void* src_pointer,
    void* dst_base_pointer,
    int memory_width, int memory_height, int memory_pitch,
    coord_t coordinate) { __builtin_unreachable(); }
#endif

int main(int argc, char** argv)
{
    sycl::queue queue(handle_async_error, sycl::property::queue::in_order());

    sycl::platform platform = queue.get_device().get_platform();
    sycl::device device = queue.get_device();

    printf("Running on SYCL platform: %s\n", platform.get_info<sycl::info::platform::name>().c_str());
    printf("Running on SYCL device: %s\n", device.get_info<sycl::info::device::name>().c_str());

    using MatrixType = uint8_t;
    constexpr size_t numRows = 64;
    constexpr size_t numCols = 64;

    std::vector<MatrixType> srcMatrix(numRows * numCols), dstMatrix(numRows * numCols);
    fill_matrix(srcMatrix, numRows, numCols);
    std::fill(dstMatrix.begin(), dstMatrix.end(), 0);

    MatrixType* pSrcData = sycl::malloc_device<MatrixType>(numRows * numCols, queue);
    MatrixType* pDstData = sycl::malloc_device<MatrixType>(numRows * numCols, queue);

    std::cout << "Populating source matrix...\n";
    queue.copy(srcMatrix.data(), pSrcData, numRows * numCols).wait();

    std::cout << "Running kernel...\n";
    const int byteWidth = static_cast<int>(numCols * sizeof(MatrixType));
    const int height = static_cast<int>(numRows);
    queue.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        coord_t coordinate{0, 0};
        int bytePitch = byteWidth;
        MatrixType data[2];
        __spirv_Subgroup2DBlockLoadINTEL(sizeof(MatrixType), 16, 2, 1, pSrcData, byteWidth, height, bytePitch, coordinate, data);
        __spirv_Subgroup2DBlockStoreINTEL(sizeof(MatrixType), 16, 2, 1, data, pDstData, byteWidth, height, bytePitch, coordinate);
    }).wait();

    std::cout << "Retrieving destination matrix...\n";
    queue.copy(pDstData, dstMatrix.data(), numRows * numCols).wait();

    std::cout << "Destination matrix is:\n";
    print_matrix(dstMatrix, numRows, numCols);

    std::cout << "Freeing pointers...\n";
    sycl::free(pSrcData, queue);
    sycl::free(pDstData, queue);

    std::cout << "Done.\n";
    return 0;
}
