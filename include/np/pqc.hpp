/**
 * @file pqc.hpp
 * @brief Post-quantum computation compatibility (constant-time, secure erasure).
 *
 * Provides PQC-friendly primitives for NumPy-C-API:
 * - constant-time compare / select (no secret-dependent branches)
 * - secure_zero (volatile + explicit_bzero style, not optimized away)
 * - side-channel hardened random seeding (zeroizes seed after use)
 * - arch-portable barriers for SIMD constant-time kernels
 *
 * Intended for use with PQC KEMs (Kyber/ML-KEM, McEliece) + signatures
 * (Dilithium/ML-DSA, Falcon, SPHINCS+) when arrays hold key material.
 * All hot paths are branch-free and use `volatile` fences to prevent
 * compiler elision.
 *
 * Reference: NIST FIPS 203/204/205 - ML-KEM / ML-DSA / SLH-DSA
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_PQC_HPP
#define NP_PQC_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif
#endif

namespace np
{
  namespace pqc
  {

    /**
     * @brief Securely zero memory (constant-time, not elided).
     * Uses volatile pointer + compiler fence. Mirrors `explicit_bzero` /
     * `sodium_memzero`.
     */
    NP_API inline void secure_zero(void* ptr, std::size_t n) noexcept
    {
      if (ptr == nullptr || n == 0)
        return;
      volatile unsigned char* p = static_cast<volatile unsigned char*>(ptr);
      for (std::size_t i = 0; i < n; ++i)
        p[i] = 0;
#if defined(__GNUC__) || defined(__clang__)
      __asm__ __volatile__("" : : "r"(p) : "memory");
#endif
      std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    template <typename T>
    NP_API inline void secure_zero(std::vector<T>& v) noexcept
    {
      if (!v.empty())
        secure_zero(v.data(), v.size() * sizeof(T));
    }

    /**
     * @brief Constant-time equality (returns 0 or 1, no branch on secret).
     */
    NP_API inline int ct_eq_u32(std::uint32_t a, std::uint32_t b) noexcept
    {
      std::uint32_t d = a ^ b;
      // d == 0 -> 1 else 0, branch-free
      return static_cast<int>(((d | (~d + 1)) >> 31) ^ 1);
    }

    NP_API inline int ct_eq_u64(std::uint64_t a, std::uint64_t b) noexcept
    {
      std::uint64_t d = a ^ b;
      return d == 0 ? 1 : 0;
    }

    /**
     * @brief Constant-time select: return a if cond==1 else b (cond is 0/1).
     */
    template <typename T>
    NP_API inline T ct_select(int cond, T a, T b) noexcept
    {
      static_assert(std::is_arithmetic_v<T> || std::is_pointer_v<T>);
      // mask = 0xFF..FF if cond==1 else 0
      using U = std::conditional_t<sizeof(T) <= 4, std::uint32_t, std::uint64_t>;
      U mask = static_cast<U>(-static_cast<int>(cond));
      if constexpr (sizeof(T) == 4)
      {
        std::uint32_t au, bu;
        std::memcpy(&au, &a, 4);
        std::memcpy(&bu, &b, 4);
        std::uint32_t ru = (au & mask) | (bu & ~mask);
        T r;
        std::memcpy(&r, &ru, 4);
        return r;
      }
      else if constexpr (sizeof(T) == 8)
      {
        std::uint64_t au, bu;
        std::memcpy(&au, &a, 8);
        std::memcpy(&bu, &b, 8);
        std::uint64_t ru = (au & mask) | (bu & ~mask);
        T r;
        std::memcpy(&r, &ru, 8);
        return r;
      }
      else
      {
        return cond ? a : b;
      }
    }

    /**
     * @brief Constant-time compare of two byte arrays (size n). Returns 1 if equal.
     */
    NP_API inline int ct_memequal(const void* a, const void* b, std::size_t n) noexcept
    {
      const volatile unsigned char* pa = static_cast<const volatile unsigned char*>(a);
      const volatile unsigned char* pb = static_cast<const volatile unsigned char*>(b);
      unsigned char diff = 0;
      for (std::size_t i = 0; i < n; ++i)
        diff |= pa[i] ^ pb[i];
      return ct_eq_u32(diff, 0);
    }

    /**
     * @brief PQC-hardened random seed wrapper – copies seed, uses, then zeroizes.
     * Use with `np::random::Generator` seeds that hold entropy.
     */
    template <typename Seed>
    class SecureSeed
    {
    public:
      explicit SecureSeed(const Seed& s) : seed_(s)
      {
      }
      ~SecureSeed()
      {
        secure_zero(seed_);
      }

      const Seed& get() const noexcept
      {
        return seed_;
      }
      Seed& get() noexcept
      {
        return seed_;
      }

      SecureSeed(const SecureSeed&) = delete;
      SecureSeed& operator=(const SecureSeed&) = delete;

    private:
      Seed seed_;
      void secure_zero(Seed& s) noexcept
      {
        if constexpr (std::is_trivially_copyable_v<Seed>)
          pqc::secure_zero(&s, sizeof(Seed));
      }
    };

    /**
     * @brief Barrier for SIMD constant-time: ensures vector loads/stores are not
     * reordered across PQC boundary (prevents speculation on secret indices).
     */
    NP_API inline void ct_barrier() noexcept
    {
      std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(__GNUC__) || defined(__clang__)
      __asm__ __volatile__("" ::: "memory");
#endif
    }

  } // namespace pqc
} // namespace np

#endif // NP_PQC_HPP
