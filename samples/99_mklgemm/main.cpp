/*
// Copyright (c) 2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <popl/popl.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <random>
#include <vector>

#if WITHMKL
#include "oneapi/mkl.hpp"
#endif

using bfloat16 = sycl::ext::oneapi::bfloat16;

using test_clock = std::chrono::high_resolution_clock;

bool fixedData = false;
bool validate = false;
bool wallclock = false;
int testIterations = 16;
float threshold = 0.01f;

std::string makeTestName(const std::string &func, int tM, int tN, int tK,
                         size_t M, size_t N, size_t K)
{
    std::ostringstream ret;
    ret << func;
    ret << "<tM:" << tM << ", tN:" << tN << ", tK:" << tK << ">";
    ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
    return ret.str();
}

std::string makeTestName(const std::string &func,
                         size_t M, size_t N, size_t K)
{
    std::ostringstream ret;
    ret << func;
    ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
    return ret.str();
}

template <typename T>
static void fill_matrix(std::vector<T>& M, size_t numRows, size_t numCols)
{
    if (fixedData) {
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                M[r * numCols + c] = r + c;
            }
        }
    } else {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> dist(-1.0, 1.0);
        std::generate(std::begin(M), std::end(M), [&]{ return dist(rng); });
    }
}

template <typename T>
static void vnni_matrix(
    std::vector<T> &dst, const std::vector<T> &src,
    size_t numRows, size_t numCols, size_t factor)
{
    for (size_t r = 0; r < numRows / factor; r++) {
        for (size_t c = 0; c < numCols; c++) {
            for (size_t k = 0; k < factor; k++) {
                dst[r * numCols * factor + c * factor + k] =
                    src[(r * factor + k) * numCols + c];
            }
        }
    }
}

template <typename DstT, typename SrcT>
static void compute_reference(
    std::vector<DstT>& C,
    const std::vector<SrcT>& A, const std::vector<SrcT>& B,
    size_t M, size_t N, size_t K)
{
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            DstT sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum = std::fma(static_cast<DstT>(A[m * K + k]),
                               static_cast<DstT>(B[k * N + n]), sum);
            }
            C[m * N + n] = sum;
        }
    }
}

template <typename T>
int check_results(const std::vector<T>& C,
                  const std::vector<T>& C_ref)
{
    float err = 0.f;
    for (int i = 0; i < C.size(); ++i) {
        auto localErr = std::fabs(C[i] - C_ref[i]) /
                        std::max(std::fabs(C[i]),
                                 std::fabs(C_ref[i]));
        err = std::max(localErr, err);
        if (localErr >= threshold) {
            std::cerr << "Error at index " << i << " (local error " << localErr
                      << "): Wanted " << C_ref[i] << ", got " << C[i]
                      << std::endl;
            break;
        }
    }

    return err < 0.001f;
}

static float hw_time(sycl::event& event)
{
    auto ns = event.get_profiling_info<sycl::info::event_profiling::command_end>() -
              event.get_profiling_info<sycl::info::event_profiling::command_start>();
    return ns / 1e9f;
}

class kernel_bfloat16_naive {
public:
    kernel_bfloat16_naive(float* C_, bfloat16* A_, bfloat16* B_, int K_):
        C(C_), A(A_), B(B_), K(K_) {}
    void operator()(sycl::nd_item<2> item) const {
        const int N = item.get_global_range(1);
        int m = item.get_global_id(0);
        int n = item.get_global_id(1);

        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum = sycl::fma(static_cast<float>(A[m * K + k]), static_cast<float>(B[k * N + n]), sum);
        }

        C[m * N + n] = sum;
    }
private:
    float* C;
    bfloat16* A;
    bfloat16* B;
    int K;
};

void bfloat16_naive(sycl::queue q,
    float* C, bfloat16* A, bfloat16* B,
    size_t M, size_t N, size_t K,
    const std::vector<float>& C_ref)
{
    std::cout << std::setw(80) << makeTestName(__FUNCTION__, M, N, K) << ": " << std::flush;

    q.fill(C, 0.0f, C_ref.size()).wait();

    auto lws = std::min<size_t>(K, 32);

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        auto event = q.parallel_for(sycl::nd_range<2>{{M, N}, {1, lws}},
            kernel_bfloat16_naive(C, A, B, K));
        q.wait();
        auto end = test_clock::now();
        std::chrono::duration<float> sw_time = end - start;
        auto elapsed = wallclock ? sw_time.count() : hw_time(event);
        best = std::min(best, elapsed);
    }
    auto gops = 2.0 * M * N * K / best / 1e9;
    printf("Best in %f seconds (%f gops)\n", best, gops);

    if (validate) {
        printf("Checking results... "); fflush(stdout);
        std::vector<float> C_check(C_ref.size());
        q.copy(C, C_check.data(), C_check.size()).wait();
        check_results(C_check, C_ref);
        printf(" done!\n");
    }
}

#if WITHMKL
void bfloat16_mkl(sycl::queue q,
    float* C, bfloat16* A, bfloat16* B,
    size_t M, size_t N, size_t K,
    const std::vector<float>& C_ref)
{
    std::cout << std::setw(80) << makeTestName(__FUNCTION__, M, N, K) << ": " << std::flush;

    q.fill(C, 0.0f, C_ref.size()).wait();

    auto lws = std::min<size_t>(K, 32);

    float best = 999.0f;
    for (int test = 0; test < testIterations; test++) {
        auto start = test_clock::now();
        auto event = oneapi::mkl::blas::row_major::gemm(q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            M, N, K,
            1.0f,   // alpha
            A, M,
            B, N,
            0.0f,   // beta
            C, M);
        q.wait();
        auto end = test_clock::now();
        std::chrono::duration<float> sw_time = end - start;
        auto elapsed = wallclock ? sw_time.count() : hw_time(event);
        best = std::min(best, elapsed);
    }
    auto gops = 2.0 * M * N * K / best / 1e9;
    printf("Best in %f seconds (%f gops)\n", best, gops);

    if (validate) {
        printf("Checking results... "); fflush(stdout);
        std::vector<float> C_check(C_ref.size());
        q.copy(C, C_check.data(), C_check.size()).wait();
        check_results(C_check, C_ref);
        printf(" done!\n");
    }
}
#endif

int main(int argc, char** argv)
{
    size_t matrixSize = 512;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<size_t>>("m", "matrixsize", "Matrix Size", matrixSize, &matrixSize);
        op.add<popl::Value<int>>("i", "iterations", "Test Iterations", testIterations, &testIterations);
        op.add<popl::Switch>("", "validate", "Validate Results", &validate);
        op.add<popl::Switch>("", "fixed", "Use Fixed Data", &fixedData);
        op.add<popl::Switch>("", "wallclock", "Measure Wallclock Time", &wallclock);
        op.add<popl::Value<float>>("", "threshold", "Local Error Threshold", threshold, &threshold);
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n\n";
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            std::cerr << "Usage: matrixexperiments [options]\n" << op.help();
            return -1;
        }
    }

    sycl::queue q{{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
    std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    std::cout << "Config:\n";
    std::cout << "\tTest Iterations: " << testIterations << "\n";
    std::cout << "\tValidating data?: " << std::boolalpha << validate << "\n";
    std::cout << "\tFixed data?: " << std::boolalpha << fixedData << "\n";
    std::cout << "\tWallclock time?:" << std::boolalpha << wallclock << "\n";

    const auto M = matrixSize;
    const auto N = matrixSize;
    const auto K = matrixSize;

    std::vector<bfloat16> A_vec(M * K);
    std::vector<bfloat16> B_vec(K * N);
    std::vector<bfloat16> Bvnni_vec(K * N);

    std::vector<float> C_ref(M * N);

    std::cout << "Initializing source matrices...\n";
    fill_matrix(A_vec, M, K);
    fill_matrix(B_vec, K, N);

    vnni_matrix(Bvnni_vec, B_vec, K, N, 2);

    bfloat16* A = sycl::malloc_device<bfloat16>(A_vec.size(), q);
    bfloat16* B = sycl::malloc_device<bfloat16>(B_vec.size(), q);
    bfloat16* Bvnni = sycl::malloc_device<bfloat16>(Bvnni_vec.size(), q);

    float* C = sycl::malloc_device<float>(C_ref.size(), q);

    q.copy(A_vec.data(), A, A_vec.size()).wait();
    q.copy(B_vec.data(), B, B_vec.size()).wait();
    q.copy(Bvnni_vec.data(), Bvnni, Bvnni_vec.size()).wait();

    if (validate) {
        std::cout << "Computing reference...\n";
        compute_reference(C_ref, A_vec, B_vec, M, N, K);
    }

#if WITHMKL
    MKLVersion mkl_version;
    mkl_get_version(&mkl_version);
    std::cout << "Using MKL version: "
        << mkl_version.MajorVersion << "." << mkl_version.UpdateVersion << "\n";
    bfloat16_mkl(q, C, A, B, M, N, K, C_ref);
#endif

    std::cout << "Running tests...\n";

    bfloat16_naive(q, C, A, B, M, N, K, C_ref);

    std::cout << "Success.\n";
    return 0;
}
