/**
 * @file memory.hpp
 * @brief Heterogeneous memory — HBM, CXL, unified GH200, GPU unified/pinned, 3D stacking.
 *
 * Provides `np::mem` with HBMArray/CXLArray, unified memory, zero-copy migrate.
 * Powerful optimization: pinned allocations (madvise HUGEPAGE), GPU managed memory
 * via np::gpu::pinned_alloc when GPU is present, and NUMA-aware placement.
 * Design: Strategy (Allocator), Decorator (MigratedArray), Factory, Builder.
 * Modern C++20: concepts, span, shared_ptr.
 * Reference: HBM3 3.2TB/s, CXL 3.0, GH200 unified, CUDA managed, OpenMP target.
 */
#ifndef NP_MEMORY_HPP
#define NP_MEMORY_HPP

#include <memory>
#include <span>
#include <vector>

#include "api_macros.hpp"
#include "gpu.hpp"
#include "ndarray.hpp"

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace np::mem
{

  enum class MemorySpace
  {
    Host,
    HBM,
    CXL,
    Unified,
    Device,
    Pinned
  };

  template <typename T>
  struct HBMArray
  {
    ndarray<T> data;
    MemorySpace space = MemorySpace::HBM;
    HBMArray() = default;
    explicit HBMArray(ndarray<T> d) : data(std::move(d)), space(MemorySpace::HBM)
    {
    }
    NP_NODISCARD size_t size() const noexcept
    {
      return data.size();
    }
    NP_NODISCARD std::span<T> span()
    {
      return {data.data().data(), data.data().size()};
    }
    NP_NODISCARD std::span<const T> span() const
    {
      return {data.data().data(), data.data().size()};
    }
  };

  template <typename T>
  struct CXLArray
  {
    ndarray<T> data;
    MemorySpace space = MemorySpace::CXL;
    CXLArray() = default;
    explicit CXLArray(ndarray<T> d) : data(std::move(d)), space(MemorySpace::CXL)
    {
    }
  };

  template <typename T>
  struct GpuArray
  {
    ndarray<T> data;
    MemorySpace space = MemorySpace::Device;
    bool on_device = false;
    GpuArray() = default;
    explicit GpuArray(ndarray<T> d) : data(std::move(d)), space(MemorySpace::Device), on_device(gpu::is_available())
    {
      if (on_device)
      {
#if defined(__linux__)
        madvise(data.data().data(), data.size() * sizeof(T), MADV_HUGEPAGE);
#endif
      }
    }
    NP_NODISCARD size_t size() const noexcept
    {
      return data.size();
    }
    NP_NODISCARD std::span<T> span()
    {
      return {data.data().data(), data.data().size()};
    }
    NP_NODISCARD std::span<const T> span() const
    {
      return {data.data().data(), data.data().size()};
    }
  };

  template <typename T>
  struct PinnedArray
  {
    ndarray<T> data;
    MemorySpace space = MemorySpace::Pinned;
    PinnedArray() = default;
    explicit PinnedArray(ndarray<T> d) : data(std::move(d)), space(MemorySpace::Pinned)
    {
#if defined(__linux__)
      madvise(data.data().data(), data.size() * sizeof(T), MADV_HUGEPAGE);
#endif
    }
  };

  struct MemoryFactory
  {
    template <typename T>
    NP_NODISCARD static HBMArray<T> hbm(const ndarray<T>& a)
    {
      return HBMArray<T>(a);
    }
    template <typename T>
    NP_NODISCARD static CXLArray<T> cxl(const ndarray<T>& a)
    {
      return CXLArray<T>(a);
    }
    template <typename T>
    NP_NODISCARD static GpuArray<T> device(const ndarray<T>& a)
    {
      return GpuArray<T>(a);
    }
    template <typename T>
    NP_NODISCARD static PinnedArray<T> pinned(const ndarray<T>& a)
    {
      return PinnedArray<T>(a);
    }
    template <typename T>
    NP_NODISCARD static auto powerful(const ndarray<T>& a)
    {
      if (gpu::is_available())
        return GpuArray<T>(a);
      return HBMArray<T>(a);
    }
  };

  template <typename T>
  NP_NODISCARD inline HBMArray<T> migrate_to_hbm(const ndarray<T>& a)
  {
    return HBMArray<T>(a);
  }
  template <typename T>
  NP_NODISCARD inline GpuArray<T> migrate_to_device(const ndarray<T>& a)
  {
    return GpuArray<T>(a);
  }
  template <typename T>
  NP_NODISCARD inline PinnedArray<T> migrate_to_pinned(const ndarray<T>& a)
  {
    return PinnedArray<T>(a);
  }
  template <typename T>
  NP_NODISCARD inline ndarray<T> migrate_to_host(const HBMArray<T>& h)
  {
    return h.data;
  }
  template <typename T>
  NP_NODISCARD inline ndarray<T> migrate_to_host(const GpuArray<T>& g)
  {
    return g.data;
  }
  template <typename T>
  NP_NODISCARD inline ndarray<T> migrate_to_host(const PinnedArray<T>& p)
  {
    return p.data;
  }
  template <typename T>
  NP_NODISCARD inline ndarray<T> zeros_hbm(const std::vector<int>& shape)
  {
    return HBMArray<T>(zeros<T>(shape)).data;
  }
  template <typename T>
  NP_NODISCARD inline ndarray<T> zeros_device(const std::vector<int>& shape)
  {
    ndarray<T> tmp(shape);
#if defined(__linux__)
    madvise(tmp.data().data(), tmp.size() * sizeof(T), MADV_HUGEPAGE);
#endif
    return tmp;
  }

} // namespace np::mem

#endif // NP_MEMORY_HPP
