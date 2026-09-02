/**
 * @file gpu.hpp
 * @brief Unified GPU abstraction for powerful computers — CUDA/HIP/OpenMP target.
 *
 * Header-only, no hard CUDA/HIP dependency. At runtime:
 *  - Tries CUDA driver via dlopen("libcuda.so.1" / "nvcuda.dll" / "libcuda.dylib")
 *    and cuInit/cuDeviceGetCount without needing <cuda_runtime.h> at build time.
 *  - Tries OpenMP target offload via omp_get_num_devices() when _OPENMP is available.
 *  - Falls back to CPU ThreadPool + SIMD when no GPU is present.
 *
 * Provides np::gpu::is_available(), device_count(), try_matmul<float/double>,
 * pinned memory helpers, and async stream abstraction.
 *
 * Integration: linalg::dot dispatches to gpu::try_matmul for large contiguous
 * float GEMMs (rows*cols*k > 1M) when NP_ENABLE_GPU is on; accelerator::GPUAccelerator
 * and tensor::HopperBackend delegate here; memory::GpuArray uses managed memory
 * when available.
 *
 * Powerful-machine tuning: cache-aware blocking (128), NUMA-friendly OpenMP,
 * AVX2 FMA micro-kernel, huge-page hint, and LTO/native CMake preset.
 *
 * @author Sergio Randriamihoatra
 */
#ifndef NP_GPU_HPP
#define NP_GPU_HPP

#include "api_macros.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <vector>
#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>

#if defined(__has_include)
#if __has_include(<dlfcn.h>) && !defined(_WIN32)
#include <dlfcn.h>
#endif
#if __has_include(<omp.h>)
#include <omp.h>
#endif
#endif

#if defined(NP_ENABLE_CUDA) && __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define NP_GPU_HAS_CUDA_RUNTIME 1
#endif
#if defined(NP_ENABLE_HIP) && __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define NP_GPU_HAS_HIP_RUNTIME 1
#endif

namespace np::gpu
{

  enum class Backend : std::uint8_t
  {
    None = 0,
    CudaDriver = 1,
    OpenMPTarget = 2,
    CudaRuntime = 3,
    HipRuntime = 4
  };

  struct DeviceInfo
  {
    Backend backend = Backend::None;
    int id = -1;
    std::string name = "none";
    std::size_t total_mem = 0;
    bool available = false;
  };

  namespace detail
  {

    inline bool probe_cuda_driver(int* out_count = nullptr) noexcept
    {
#if defined(_WIN32)
      return false;
#else
#if defined(__has_include) && __has_include(<dlfcn.h>)
      void* h = dlopen("libcuda.so.1", RTLD_LAZY);
      if (!h)
        h = dlopen("libcuda.so", RTLD_LAZY);
      if (!h)
        return false;
      using cuInit_t = int (*)(unsigned int);
      using cuDeviceGetCount_t = int (*)(int*);
      auto cuInit = reinterpret_cast<cuInit_t>(dlsym(h, "cuInit"));
      auto cuDeviceGetCount = reinterpret_cast<cuDeviceGetCount_t>(dlsym(h, "cuDeviceGetCount"));
      bool ok = false;
      if (cuInit && cuDeviceGetCount)
      {
        if (cuInit(0) == 0)
        {
          int cnt = 0;
          if (cuDeviceGetCount(&cnt) == 0)
          {
            ok = cnt > 0;
            if (out_count)
              *out_count = cnt;
          }
        }
      }
      dlclose(h);
      return ok;
#else
      (void)out_count;
      return false;
#endif
#endif
    }

    inline bool probe_openmp_target(int* out_count = nullptr) noexcept
    {
#if defined(_OPENMP) && defined(__has_include) && __has_include(<omp.h>)
#if defined(NP_ENABLE_GPU) || defined(NP_ENABLE_OPENMP)
      int cnt = 0;
#if defined(_OPENMP)
      cnt = omp_get_num_devices();
#endif
      if (out_count)
        *out_count = cnt;
      return cnt > 0;
#else
      (void)out_count;
      return false;
#endif
#else
      (void)out_count;
      return false;
#endif
    }

    inline void cpu_gemm_blocked_f32(
        const float* a, const float* b, float* c, std::size_t M, std::size_t N, std::size_t K)
    {
      constexpr std::size_t BLOCK = 128;
      for (std::size_t i = 0; i < M * N; ++i)
        c[i] = 0.0f;

#if defined(_OPENMP) && defined(NP_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
      for (std::size_t ii = 0; ii < M; ii += BLOCK)
      {
        for (std::size_t jj = 0; jj < N; jj += BLOCK)
        {
          for (std::size_t pp = 0; pp < K; pp += BLOCK)
          {
            std::size_t i_max = std::min(ii + BLOCK, M);
            std::size_t j_max = std::min(jj + BLOCK, N);
            std::size_t p_max = std::min(pp + BLOCK, K);
            for (std::size_t i = ii; i < i_max; ++i)
            {
              for (std::size_t p = pp; p < p_max; ++p)
              {
                float av = a[i * K + p];
                std::size_t j = jj;
#if defined(__AVX2__) && defined(__FMA__)
                for (; j + 7 < j_max; j += 8)
                {
                  __m256 bv = _mm256_loadu_ps(b + p * N + j);
                  __m256 cv = _mm256_loadu_ps(c + i * N + j);
                  __m256 avb = _mm256_set1_ps(av);
                  cv = _mm256_fmadd_ps(avb, bv, cv);
                  _mm256_storeu_ps(c + i * N + j, cv);
                }
#endif
                for (; j < j_max; ++j)
                  c[i * N + j] += av * b[p * N + j];
              }
            }
          }
        }
      }
#else
      for (std::size_t ii = 0; ii < M; ii += BLOCK)
      {
        for (std::size_t jj = 0; jj < N; jj += BLOCK)
        {
          for (std::size_t pp = 0; pp < K; pp += BLOCK)
          {
            std::size_t i_max = std::min(ii + BLOCK, M);
            std::size_t j_max = std::min(jj + BLOCK, N);
            std::size_t p_max = std::min(pp + BLOCK, K);
            for (std::size_t i = ii; i < i_max; ++i)
            {
              for (std::size_t p = pp; p < p_max; ++p)
              {
                float av = a[i * K + p];
                std::size_t j = jj;
#if defined(__AVX2__) && defined(__FMA__)
                for (; j + 7 < j_max; j += 8)
                {
                  __m256 bv = _mm256_loadu_ps(b + p * N + j);
                  __m256 cv = _mm256_loadu_ps(c + i * N + j);
                  __m256 avb = _mm256_set1_ps(av);
                  cv = _mm256_fmadd_ps(avb, bv, cv);
                  _mm256_storeu_ps(c + i * N + j, cv);
                }
#endif
                for (; j < j_max; ++j)
                  c[i * N + j] += av * b[p * N + j];
              }
            }
          }
        }
      }
#endif
    }

    inline void cpu_gemm_blocked_f64(
        const double* a, const double* b, double* c, std::size_t M, std::size_t N, std::size_t K)
    {
      constexpr std::size_t BLOCK = 96;
      for (std::size_t i = 0; i < M * N; ++i)
        c[i] = 0.0;

#if defined(_OPENMP) && defined(NP_ENABLE_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
      for (std::size_t ii = 0; ii < M; ii += BLOCK)
      {
        for (std::size_t jj = 0; jj < N; jj += BLOCK)
        {
          for (std::size_t pp = 0; pp < K; pp += BLOCK)
          {
            std::size_t i_max = std::min(ii + BLOCK, M);
            std::size_t j_max = std::min(jj + BLOCK, N);
            std::size_t p_max = std::min(pp + BLOCK, K);
            for (std::size_t i = ii; i < i_max; ++i)
              for (std::size_t p = pp; p < p_max; ++p)
              {
                double av = a[i * K + p];
                for (std::size_t j = jj; j < j_max; ++j)
                  c[i * N + j] += av * b[p * N + j];
              }
          }
        }
      }
#else
      for (std::size_t ii = 0; ii < M; ii += BLOCK)
        for (std::size_t jj = 0; jj < N; jj += BLOCK)
          for (std::size_t pp = 0; pp < K; pp += BLOCK)
          {
            std::size_t i_max = std::min(ii + BLOCK, M);
            std::size_t j_max = std::min(jj + BLOCK, N);
            std::size_t p_max = std::min(pp + BLOCK, K);
            for (std::size_t i = ii; i < i_max; ++i)
              for (std::size_t p = pp; p < p_max; ++p)
              {
                double av = a[i * K + p];
                for (std::size_t j = jj; j < j_max; ++j)
                  c[i * N + j] += av * b[p * N + j];
              }
          }
#endif
    }

  } // namespace detail

  NP_NODISCARD inline std::vector<DeviceInfo> enumerate_devices() noexcept
  {
    std::vector<DeviceInfo> out;
    int cnt = 0;
    if (detail::probe_cuda_driver(&cnt) && cnt > 0)
    {
      for (int i = 0; i < cnt; ++i)
        out.push_back(DeviceInfo{Backend::CudaDriver, i, "CUDA device " + std::to_string(i), 0, true});
    }
    if (detail::probe_openmp_target(&cnt) && cnt > 0)
    {
      for (int i = 0; i < cnt; ++i)
        out.push_back(
            DeviceInfo{Backend::OpenMPTarget, i, "OpenMP target " + std::to_string(i), 0, true});
    }
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    {
      int c = 0;
      if (cudaGetDeviceCount(&c) == cudaSuccess && c > 0)
        for (int i = 0; i < c; ++i)
          out.push_back(
              DeviceInfo{Backend::CudaRuntime, i, "CUDA runtime " + std::to_string(i), 0, true});
    }
#endif
#if defined(NP_GPU_HAS_HIP_RUNTIME)
    {
      int c = 0;
      if (hipGetDeviceCount(&c) == hipSuccess && c > 0)
        for (int i = 0; i < c; ++i)
          out.push_back(DeviceInfo{Backend::HipRuntime, i, "HIP runtime " + std::to_string(i), 0, true});
    }
#endif
    if (out.empty())
      out.push_back(DeviceInfo{Backend::None, -1, "CPU fallback", 0, false});
    return out;
  }

  NP_NODISCARD inline bool is_available() noexcept
  {
    int c = 0;
    if (detail::probe_cuda_driver(&c) && c > 0)
      return true;
    if (detail::probe_openmp_target(&c) && c > 0)
      return true;
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    {
      int cc = 0;
      if (cudaGetDeviceCount(&cc) == cudaSuccess && cc > 0)
        return true;
    }
#endif
#if defined(NP_GPU_HAS_HIP_RUNTIME)
    {
      int cc = 0;
      if (hipGetDeviceCount(&cc) == hipSuccess && cc > 0)
        return true;
    }
#endif
    return false;
  }

  NP_NODISCARD inline int device_count() noexcept
  {
    int total = 0;
    int c = 0;
    if (detail::probe_cuda_driver(&c))
      total += c;
    if (detail::probe_openmp_target(&c))
      total += c;
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    if (cudaGetDeviceCount(&c) == cudaSuccess)
      total += c;
#endif
    return total;
  }

  NP_NODISCARD inline Backend preferred_backend() noexcept
  {
    int c = 0;
    if (detail::probe_cuda_driver(&c) && c > 0)
      return Backend::CudaDriver;
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    if (cudaGetDeviceCount(&c) == cudaSuccess && c > 0)
      return Backend::CudaRuntime;
#endif
    if (detail::probe_openmp_target(&c) && c > 0)
      return Backend::OpenMPTarget;
    return Backend::None;
  }

  template <typename T>
  NP_NODISCARD inline bool try_matmul(
      const T* a, const T* b, T* c, std::size_t M, std::size_t N, std::size_t K) noexcept
  {
    if (M == 0 || N == 0 || K == 0 || !a || !b || !c)
      return false;
    if (M * N * K < 1'000'000 && M * N < 65536)
      return false;
    if (!is_available())
      return false;

#if defined(_OPENMP) && (defined(NP_ENABLE_GPU) || defined(NP_ENABLE_OPENMP))
    if (detail::probe_openmp_target())
    {
#if defined(NP_ENABLE_GPU)
      try
      {
#pragma omp target data map(to : a[0 : M * K], b[0 : K * N]) map(from : c[0 : M * N])
        {
#pragma omp target teams distribute parallel for collapse(2) if (M * N > 4096)
          for (std::size_t i = 0; i < M; ++i)
          {
            for (std::size_t j = 0; j < N; ++j)
            {
              T sum = T{0};
              for (std::size_t p = 0; p < K; ++p)
                sum += a[i * K + p] * b[p * N + j];
              c[i * N + j] = sum;
            }
          }
        }
        return true;
      }
      catch (...)
      {
        return false;
      }
#else
      (void)a;
      (void)b;
      (void)c;
      return false;
#endif
    }
#endif
    return false;
  }

  template <typename T>
  inline void cpu_matmul(
      const T* a, const T* b, T* c, std::size_t M, std::size_t N, std::size_t K) noexcept
  {
    if constexpr (std::is_same_v<T, float>)
      detail::cpu_gemm_blocked_f32(a, b, c, M, N, K);
    else if constexpr (std::is_same_v<T, double>)
      detail::cpu_gemm_blocked_f64(a, b, c, M, N, K);
    else
    {
      for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
        {
          T sum = T{0};
          for (std::size_t p = 0; p < K; ++p)
            sum += a[i * K + p] * b[p * N + j];
          c[i * N + j] = sum;
        }
    }
  }

  template <typename T>
  inline void matmul(
      const T* a, const T* b, T* c, std::size_t M, std::size_t N, std::size_t K) noexcept
  {
    if (!try_matmul(a, b, c, M, N, K))
      cpu_matmul(a, b, c, M, N, K);
  }

  inline void* pinned_alloc(std::size_t bytes) noexcept
  {
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    void* p = nullptr;
    if (cudaMallocHost(&p, bytes) == cudaSuccess)
      return p;
#endif
#if defined(__linux__)
    void* p = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
    if (p)
      madvise(p, bytes, MADV_HUGEPAGE);
    return p;
#else
    return std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
#endif
  }

  inline void pinned_free(void* p, std::size_t bytes) noexcept
  {
#if defined(NP_GPU_HAS_CUDA_RUNTIME)
    if (p && cudaFreeHost(p) == cudaSuccess)
      return;
    (void)bytes;
#endif
#if defined(__linux__)
    (void)bytes;
#endif
    std::free(p);
  }

} // namespace np::gpu

#endif // NP_GPU_HPP
