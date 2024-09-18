// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <stdio.h>

#include <chrono>
#include <numeric>
#include <vector>

#include <popl/popl.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>

using test_clock = std::chrono::high_resolution_clock;

constexpr auto vec_size = 16;
constexpr auto sg_size = 16;
constexpr auto wg_size = 256;
constexpr auto sg_per_wg = wg_size / sg_size;

using fvec = sycl::vec<float, vec_size>;

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL x
#else
#define SYCL_DEVICE_OCL(x)                                                     \
  inline x { assert(false); return 0; }
#endif

SYCL_DEVICE_OCL(float sub_group_reduce_add(float i));
SYCL_DEVICE_OCL(float sub_group_reduce_max(float i));

template <typename AllocT> auto *local_mem() {
  auto item = sycl::ext::oneapi::experimental::this_nd_item<1>();
  sycl::multi_ptr<AllocT, sycl::access::address_space::local_space>
      As_multi_ptr =
          sycl::ext::oneapi::group_local_memory_for_overwrite<AllocT>(
              item.get_group());
  auto *As = *As_multi_ptr;
  return As;
}

template <class T, uint32_t N>
decltype(auto) sg_reduce_max(sycl::vec<T, N> &vec) {
  sycl::vec<T, N> ret;
  for (int i = 0; i < N; i++) {
    ret[i] = sub_group_reduce_max(vec[i]);
  }

  return ret;
}

template <uint32_t sg_num, class T, uint32_t N, class mem_t>
decltype(auto) inline wg_reduce_max(mem_t &mem, sycl::vec<T, N> &vec) {

  if constexpr (sg_num == 0) {
    return;
  } else {
    auto item = sycl::ext::oneapi::experimental::this_nd_item<1>();
    auto sg = item.get_sub_group();
    auto group = item.get_group();

    sycl::group_barrier(group);

    if constexpr (sg_num == 1) {
#pragma unroll
      for (int i = 0; i < N; i++) {
        vec[i] = mem[i];
      }
    } else {
      auto sg_group_id = sg.get_group_id();
      auto sg_local_id = sg.get_local_id()[0];
      if (sg_group_id < (sg_num / 2)) {
        auto tmp = sg_group_id * N;
        static constexpr auto base = sg_num * N / 2;
        if constexpr (N < sg_size) {
          if (sg_local_id < N) {
            auto offset = (tmp + sg_local_id);
            mem[offset] = max(mem[offset], mem[base + offset]);
          }
        } else {
          static constexpr auto step = N / sg_size;
#pragma unroll
          for (int i = 0; i < step; i++) {
            auto offset = (tmp + sg_local_id * step + i);
            mem[offset] = sycl::max(mem[offset], mem[base + offset]);
          }
        }
      }
    }
    return vec;
  }
}

template <class T, uint32_t N>
decltype(auto) inline group_reduce_max(sycl::vec<T, N> &vec) {
  auto item = sycl::ext::oneapi::experimental::this_nd_item<1>();
  auto sg = item.get_sub_group();

  auto sg_max = sg_reduce_max<T, N>(vec);

  auto sg_group_id = sg.get_group_id();
  auto sg_local_id = sg.get_local_id()[0];

  auto smem = local_mem<float[sg_per_wg * N]>();

  if constexpr (N < sg_size) {
    if (sg_local_id < N) {
      smem[sg_group_id * N + sg_local_id] = sg_max[sg_local_id];
    }
  } else {
    static constexpr auto step = N / sg_size;
    auto base = sg_local_id * step;
#pragma unroll
    for (int i = 0; i < step; i++) {
      smem[sg_group_id * N + base + i] = sg_max[base + i];
    }
  }

  sycl::vec<T, N> group_vec;

  wg_reduce_max<sg_per_wg, float, N, decltype(smem)>(smem, group_vec);
  wg_reduce_max<sg_per_wg / 2, float, N, decltype(smem)>(smem, group_vec);
  wg_reduce_max<sg_per_wg / 4, float, N, decltype(smem)>(smem, group_vec);
  wg_reduce_max<sg_per_wg / 8, float, N, decltype(smem)>(smem, group_vec);
  wg_reduce_max<sg_per_wg / 16, float, N, decltype(smem)>(smem, group_vec);
  wg_reduce_max<sg_per_wg / 32, float, N, decltype(smem)>(smem, group_vec);
  wg_reduce_max<sg_per_wg / 64, float, N, decltype(smem)>(smem, group_vec);

  return group_vec;
}

template <typename T>
void checkResults(size_t gws, const std::vector<T>& output)
{
  for (size_t g = 0; g < gws / wg_size; g++) {
    for (size_t i = 0; i < wg_size; i++) {
      for (size_t v = 0; v < vec_size; v++) {
        const float want = (g + 1) * wg_size * vec_size - vec_size + v + 1;
        if (output[(g * wg_size + i) * vec_size + v] != want) {
          printf("mismatch!  group = %zu, index = %zu, output[%zu] = %f vs. %f\n", g, i, v, output[(g * wg_size + i) * vec_size + v], want);
        }
      }
    }
  }
}

int main(int argc, char** argv)
{
  int platformIndex = 0;
  int deviceIndex = 0;

  size_t iterations = 16;
  size_t gws = 1024 * 1024;

  {
    popl::OptionParser op("Supported Options");
    op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
    op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
    op.add<popl::Value<size_t>>("i", "iterations", "Iterations", iterations, &iterations);
    op.add<popl::Value<size_t>>("", "gws", "Global Work Size", gws, &gws);
    //op.add<popl::Value<size_t>>("", "lws", "Local Work Size", lws, &lws);

    bool printUsage = false;
    try {
      op.parse(argc, argv);
    } catch (std::exception &e) {
      fprintf(stderr, "Error: %s\n\n", e.what());
      printUsage = true;
    }

    if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
      fprintf(stderr,
          "Usage: reductions [options]\n"
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

  std::vector<float> input(gws * vec_size);
  std::iota(input.begin(), input.end(), 1);

  std::vector<float> output(gws * vec_size);

  fvec* dinput = sycl::malloc_device<fvec>(gws, queue);
  fvec* doutput = sycl::malloc_device<fvec>(gws, queue);

  queue.memcpy(dinput, input.data(), gws * sizeof(fvec));
  queue.fill(doutput, 0, gws);
  queue.wait();

  {
    auto start = test_clock::now();
    for (size_t i = 0; i < iterations; i++) {
      queue.parallel_for(sycl::nd_range<1>{gws, wg_size}, [=](sycl::nd_item<1> it) {
          auto gid = it.get_global_id(0);
          auto g = it.get_group();
          fvec value = dinput[gid];
          auto group_max = sycl::reduce_over_group(g, value, sycl::maximum<>());
          doutput[gid] = group_max;
      });
    }
    queue.wait();
    auto end = test_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    printf("SYCL version: Finished %zu iterations in %f seconds\n", iterations, elapsed_seconds.count());
  }

  queue.memcpy(output.data(), doutput, gws * sizeof(fvec));
  queue.wait();

  checkResults(gws, output);

  {
    auto start = test_clock::now();
    for (size_t i = 0; i < iterations; i++) {
      queue.parallel_for(sycl::nd_range<1>{gws, wg_size}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
          auto gid = it.get_global_id(0);
          fvec value = dinput[gid];
          auto group_max = group_reduce_max<float, vec_size>(value);
          doutput[gid] = group_max;
      });
    }
    queue.wait();
    auto end = test_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    printf("Alternate version: Finished %zu iterations in %f seconds\n", iterations, elapsed_seconds.count());
  }

  queue.memcpy(output.data(), doutput, gws * sizeof(fvec));
  queue.wait();

  checkResults(gws, output);

  sycl::free(dinput, queue);
  sycl::free(doutput, queue);
  return 0;
}
