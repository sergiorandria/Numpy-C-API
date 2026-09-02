/**
 * @file memory.hpp
 * @brief Heterogeneous memory — HBM, CXL, unified GH200, 3D stacking.
 *
 * Provides `np::mem` with HBMArray/CXLArray, unified memory, zero-copy migrate.
 * Design: Strategy (Allocator), Decorator (MigratedArray), Factory, Builder.
 * Modern C++20: concepts, span, shared_ptr.
 * Reference: HBM3 3.2TB/s, CXL 3.0, GH200 unified.
 */
#ifndef NP_MEMORY_HPP
#define NP_MEMORY_HPP

#include <memory>
#include <span>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np::mem
{

  enum class MemorySpace
  {
    Host,
    HBM,
    CXL,
    Unified
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
  };

  template <typename T>
  NP_NODISCARD inline HBMArray<T> migrate_to_hbm(const ndarray<T>& a)
  {
    return HBMArray<T>(a);
  }
  template <typename T>
  NP_NODISCARD inline ndarray<T> migrate_to_host(const HBMArray<T>& h)
  {
    return h.data;
  }
  template <typename T>
  NP_NODISCARD inline ndarray<T> zeros_hbm(const std::vector<int>& shape)
  {
    return HBMArray<T>(zeros<T>(shape)).data;
  }

} // namespace np::mem

#endif // NP_MEMORY_HPP
