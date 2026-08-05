/**
 * @file simd.hpp
 * @brief SIMD optimizations for array operations.
 *
 * Provides CPU-specific SIMD optimizations for common array operations.
 * Supports multiple instruction sets: SSE, AVX, AVX2, AVX-512, NEON.
 *
 * Reference: numpy-reference/reference/simd/
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_SIMD_HPP
#define NP_SIMD_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Detect SIMD capabilities at compile time
#if defined(__AVX512F__)
    #define NP_SIMD_AVX512
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define NP_SIMD_AVX2
    #include <immintrin.h>
#elif defined(__AVX__)
    #define NP_SIMD_AVX
    #include <immintrin.h>
#elif defined(__SSE4_2__)
    #define NP_SIMD_SSE42
    #include <nmmintrin.h>
#elif defined(__SSE4_1__)
    #define NP_SIMD_SSE41
    #include <smmintrin.h>
#elif defined(__SSSE3__)
    #define NP_SIMD_SSSE3
    #include <tmmintrin.h>
#elif defined(__SSE3__)
    #define NP_SIMD_SSE3
    #include <pmmintrin.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #define NP_SIMD_SSE2
    #include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define NP_SIMD_NEON
    #include <arm_neon.h>
#endif

namespace np {
namespace simd {

    // =================================================================
    // SIMD Trait Detection
    // =================================================================

    /**
     * @brief Compile-time detection of available SIMD features.
     */
    struct Features {
        static constexpr bool has_sse2 = 
            #ifdef NP_SIMD_SSE2
            true;
            #else
            false;
            #endif

        static constexpr bool has_sse3 = 
            #ifdef NP_SIMD_SSE3
            true;
            #else
            false;
            #endif

        static constexpr bool has_ssse3 = 
            #ifdef NP_SIMD_SSSE3
            true;
            #else
            false;
            #endif

        static constexpr bool has_sse41 = 
            #ifdef NP_SIMD_SSE41
            true;
            #else
            false;
            #endif

        static constexpr bool has_sse42 = 
            #ifdef NP_SIMD_SSE42
            true;
            #else
            false;
            #endif

        static constexpr bool has_avx = 
            #ifdef NP_SIMD_AVX
            true;
            #else
            false;
            #endif

        static constexpr bool has_avx2 = 
            #ifdef NP_SIMD_AVX2
            true;
            #else
            false;
            #endif

        static constexpr bool has_avx512 = 
            #ifdef NP_SIMD_AVX512
            true;
            #else
            false;
            #endif

        static constexpr bool has_neon = 
            #ifdef NP_SIMD_NEON
            true;
            #else
            false;
            #endif
    };

    // =================================================================
    // Vectorized Operations
    // =================================================================

    /**
     * @brief Vector width for different types and instruction sets.
     */
    template <typename T>
    struct VectorWidth {
        #if defined(NP_SIMD_AVX512)
            static constexpr std::size_t value = 64 / sizeof(T);  // 512 bits
        #elif defined(NP_SIMD_AVX) || defined(NP_SIMD_AVX2)
            static constexpr std::size_t value = 32 / sizeof(T);  // 256 bits
        #elif defined(NP_SIMD_SSE2)
            static constexpr std::size_t value = 16 / sizeof(T);  // 128 bits
        #elif defined(NP_SIMD_NEON)
            static constexpr std::size_t value = 16 / sizeof(T);  // 128 bits
        #else
            static constexpr std::size_t value = 1;  // Scalar fallback
        #endif
    };

    // =================================================================
    // SSE2/SSE4.1 Optimizations (x86-64)
    // =================================================================

    #ifdef NP_SIMD_SSE2

    /**
     * @brief Vectorized addition for double arrays (SSE2).
     */
    inline void add_f64_sse2(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 2);
        
        for (; i < vec_end; i += 2) {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d vout = _mm_add_pd(va, vb);
            _mm_storeu_pd(out + i, vout);
        }
        
        // Handle remainder
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    /**
     * @brief Vectorized addition for float arrays (SSE).
     */
    inline void add_f32_sse(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vout = _mm_add_ps(va, vb);
            _mm_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    /**
     * @brief Vectorized multiplication for double arrays (SSE2).
     */
    inline void mul_f64_sse2(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 2);
        
        for (; i < vec_end; i += 2) {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d vout = _mm_mul_pd(va, vb);
            _mm_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    /**
     * @brief Vectorized multiplication for float arrays (SSE).
     */
    inline void mul_f32_sse(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vout = _mm_mul_ps(va, vb);
            _mm_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    /**
     * @brief Vectorized sum reduction for double arrays (SSE2).
     */
    inline double sum_f64_sse2(const double* data, std::size_t n) {
        __m128d vsum = _mm_setzero_pd();
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 2);
        
        for (; i < vec_end; i += 2) {
            __m128d v = _mm_loadu_pd(data + i);
            vsum = _mm_add_pd(vsum, v);
        }
        
        // Horizontal sum
        double temp[2];
        _mm_storeu_pd(temp, vsum);
        double sum = temp[0] + temp[1];
        
        // Handle remainder
        for (; i < n; ++i) {
            sum += data[i];
        }
        
        return sum;
    }

    /**
     * @brief Vectorized sum reduction for float arrays (SSE).
     */
    inline float sum_f32_sse(const float* data, std::size_t n) {
        __m128 vsum = _mm_setzero_ps();
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m128 v = _mm_loadu_ps(data + i);
            vsum = _mm_add_ps(vsum, v);
        }
        
        // Horizontal sum
        float temp[4];
        _mm_storeu_ps(temp, vsum);
        float sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        for (; i < n; ++i) {
            sum += data[i];
        }
        
        return sum;
    }

    /**
     * @brief Vectorized subtraction for double arrays (SSE2).
     */
    inline void sub_f64_sse2(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 2);
        
        for (; i < vec_end; i += 2) {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d vout = _mm_sub_pd(va, vb);
            _mm_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    /**
     * @brief Vectorized subtraction for float arrays (SSE).
     */
    inline void sub_f32_sse(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vout = _mm_sub_ps(va, vb);
            _mm_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    /**
     * @brief Vectorized division for double arrays (SSE2).
     */
    inline void div_f64_sse2(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 2);
        
        for (; i < vec_end; i += 2) {
            __m128d va = _mm_loadu_pd(a + i);
            __m128d vb = _mm_loadu_pd(b + i);
            __m128d vout = _mm_div_pd(va, vb);
            _mm_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    /**
     * @brief Vectorized division for float arrays (SSE).
     */
    inline void div_f32_sse(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vout = _mm_div_ps(va, vb);
            _mm_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    #endif // NP_SIMD_SSE2

    // =================================================================
    // AVX/AVX2 Optimizations (x86-64)
    // =================================================================

    #ifdef NP_SIMD_AVX

    /**
     * @brief Vectorized addition for double arrays (AVX).
     */
    inline void add_f64_avx(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d vout = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    /**
     * @brief Vectorized addition for float arrays (AVX).
     */
    inline void add_f32_avx(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vout = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    /**
     * @brief Vectorized multiplication for double arrays (AVX).
     */
    inline void mul_f64_avx(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d vout = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    /**
     * @brief Vectorized multiplication for float arrays (AVX).
     */
    inline void mul_f32_avx(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vout = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    /**
     * @brief Vectorized sum reduction for double arrays (AVX).
     */
    inline double sum_f64_avx(const double* data, std::size_t n) {
        __m256d vsum = _mm256_setzero_pd();
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m256d v = _mm256_loadu_pd(data + i);
            vsum = _mm256_add_pd(vsum, v);
        }
        
        // Horizontal sum
        double temp[4];
        _mm256_storeu_pd(temp, vsum);
        double sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        for (; i < n; ++i) {
            sum += data[i];
        }
        
        return sum;
    }

    /**
     * @brief Vectorized sum reduction for float arrays (AVX).
     */
    inline float sum_f32_avx(const float* data, std::size_t n) {
        __m256 vsum = _mm256_setzero_ps();
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            vsum = _mm256_add_ps(vsum, v);
        }
        
        // Horizontal sum
        float temp[8];
        _mm256_storeu_ps(temp, vsum);
        float sum = 0.0f;
        for (int j = 0; j < 8; ++j) {
            sum += temp[j];
        }
        
        for (; i < n; ++i) {
            sum += data[i];
        }
        
        return sum;
    }

    /**
     * @brief Vectorized subtraction for double arrays (AVX).
     */
    inline void sub_f64_avx(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d vout = _mm256_sub_pd(va, vb);
            _mm256_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    /**
     * @brief Vectorized subtraction for float arrays (AVX).
     */
    inline void sub_f32_avx(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vout = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    /**
     * @brief Vectorized division for double arrays (AVX).
     */
    inline void div_f64_avx(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            __m256d va = _mm256_loadu_pd(a + i);
            __m256d vb = _mm256_loadu_pd(b + i);
            __m256d vout = _mm256_div_pd(va, vb);
            _mm256_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    /**
     * @brief Vectorized division for float arrays (AVX).
     */
    inline void div_f32_avx(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vout = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    #endif // NP_SIMD_AVX

    // =================================================================
    // AVX-512 Optimizations (x86-64)
    // =================================================================

    #ifdef NP_SIMD_AVX512

    /**
     * @brief Vectorized addition for double arrays (AVX-512).
     */
    inline void add_f64_avx512(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d vout = _mm512_add_pd(va, vb);
            _mm512_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    /**
     * @brief Vectorized addition for float arrays (AVX-512).
     */
    inline void add_f32_avx512(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 16);
        
        for (; i < vec_end; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 vout = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    /**
     * @brief Vectorized sum reduction for double arrays (AVX-512).
     */
    inline double sum_f64_avx512(const double* data, std::size_t n) {
        __m512d vsum = _mm512_setzero_pd();
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m512d v = _mm512_loadu_pd(data + i);
            vsum = _mm512_add_pd(vsum, v);
        }
        
        double sum = _mm512_reduce_add_pd(vsum);
        
        for (; i < n; ++i) {
            sum += data[i];
        }
        
        return sum;
    }

    /**
     * @brief Vectorized subtraction for double arrays (AVX-512).
     */
    inline void sub_f64_avx512(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d vout = _mm512_sub_pd(va, vb);
            _mm512_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    /**
     * @brief Vectorized subtraction for float arrays (AVX-512).
     */
    inline void sub_f32_avx512(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 16);
        
        for (; i < vec_end; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 vout = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    /**
     * @brief Vectorized division for double arrays (AVX-512).
     */
    inline void div_f64_avx512(const double* a, const double* b, double* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 8);
        
        for (; i < vec_end; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            __m512d vb = _mm512_loadu_pd(b + i);
            __m512d vout = _mm512_div_pd(va, vb);
            _mm512_storeu_pd(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    /**
     * @brief Vectorized division for float arrays (AVX-512).
     */
    inline void div_f32_avx512(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 16);
        
        for (; i < vec_end; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 vout = _mm512_div_ps(va, vb);
            _mm512_storeu_ps(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    #endif // NP_SIMD_AVX512

    // =================================================================
    // ARM NEON Optimizations
    // =================================================================

    #ifdef NP_SIMD_NEON

    /**
     * @brief Vectorized addition for float arrays (NEON).
     */
    inline void add_f32_neon(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vout = vaddq_f32(va, vb);
            vst1q_f32(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    /**
     * @brief Vectorized multiplication for float arrays (NEON).
     */
    inline void mul_f32_neon(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vout = vmulq_f32(va, vb);
            vst1q_f32(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    /**
     * @brief Vectorized sum reduction for float arrays (NEON).
     */
    inline float sum_f32_neon(const float* data, std::size_t n) {
        float32x4_t vsum = vdupq_n_f32(0.0f);
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            float32x4_t v = vld1q_f32(data + i);
            vsum = vaddq_f32(vsum, v);
        }
        
        // Horizontal sum
        float32x2_t vsum_low = vget_low_f32(vsum);
        float32x2_t vsum_high = vget_high_f32(vsum);
        float32x2_t vsum_pair = vadd_f32(vsum_low, vsum_high);
        float sum = vget_lane_f32(vsum_pair, 0) + vget_lane_f32(vsum_pair, 1);
        
        for (; i < n; ++i) {
            sum += data[i];
        }
        
        return sum;
    }

    /**
     * @brief Vectorized subtraction for float arrays (NEON).
     */
    inline void sub_f32_neon(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vout = vsubq_f32(va, vb);
            vst1q_f32(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    /**
     * @brief Vectorized division for float arrays (NEON).
     */
    inline void div_f32_neon(const float* a, const float* b, float* out, std::size_t n) {
        std::size_t i = 0;
        const std::size_t vec_end = n - (n % 4);
        
        for (; i < vec_end; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t vout = vdivq_f32(va, vb);
            vst1q_f32(out + i, vout);
        }
        
        for (; i < n; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    #endif // NP_SIMD_NEON

    // =================================================================
    // Generic Dispatch Functions (Runtime Selection)
    // =================================================================

    /**
     * @brief Vectorized addition with automatic dispatch.
     */
    template <typename T>
    inline void add_vectorized(const T* a, const T* b, T* out, std::size_t n) {
        if constexpr (std::is_same_v<T, double>) {
            #if defined(NP_SIMD_AVX512)
                add_f64_avx512(a, b, out, n);
            #elif defined(NP_SIMD_AVX)
                add_f64_avx(a, b, out, n);
            #elif defined(NP_SIMD_SSE2)
                add_f64_sse2(a, b, out, n);
            #else
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = a[i] + b[i];
                }
            #endif
        } else if constexpr (std::is_same_v<T, float>) {
            #if defined(NP_SIMD_AVX512)
                add_f32_avx512(a, b, out, n);
            #elif defined(NP_SIMD_AVX)
                add_f32_avx(a, b, out, n);
            #elif defined(NP_SIMD_SSE2)
                add_f32_sse(a, b, out, n);
            #elif defined(NP_SIMD_NEON)
                add_f32_neon(a, b, out, n);
            #else
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = a[i] + b[i];
                }
            #endif
        } else {
            // Scalar fallback
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = a[i] + b[i];
            }
        }
    }

    /**
     * @brief Vectorized multiplication with automatic dispatch.
     */
    template <typename T>
    inline void mul_vectorized(const T* a, const T* b, T* out, std::size_t n) {
        if constexpr (std::is_same_v<T, double>) {
            #if defined(NP_SIMD_AVX)
                mul_f64_avx(a, b, out, n);
            #elif defined(NP_SIMD_SSE2)
                mul_f64_sse2(a, b, out, n);
            #else
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = a[i] * b[i];
                }
            #endif
        } else if constexpr (std::is_same_v<T, float>) {
            #if defined(NP_SIMD_AVX)
                mul_f32_avx(a, b, out, n);
            #elif defined(NP_SIMD_SSE2)
                mul_f32_sse(a, b, out, n);
            #elif defined(NP_SIMD_NEON)
                mul_f32_neon(a, b, out, n);
            #else
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = a[i] * b[i];
                }
            #endif
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = a[i] * b[i];
            }
        }
    }

    /**
     * @brief Vectorized sum reduction with automatic dispatch.
     */
    template <typename T>
    inline T sum_vectorized(const T* data, std::size_t n) {
        if constexpr (std::is_same_v<T, double>) {
            #if defined(NP_SIMD_AVX512)
                return sum_f64_avx512(data, n);
            #elif defined(NP_SIMD_AVX)
                return sum_f64_avx(data, n);
            #elif defined(NP_SIMD_SSE2)
                return sum_f64_sse2(data, n);
            #else
                T sum = T{0};
                for (std::size_t i = 0; i < n; ++i) {
                    sum += data[i];
                }
                return sum;
            #endif
        } else if constexpr (std::is_same_v<T, float>) {
            #if defined(NP_SIMD_AVX512)
                return sum_f64_avx512(data, n);
            #elif defined(NP_SIMD_AVX)
                return sum_f32_avx(data, n);
            #elif defined(NP_SIMD_SSE2)
                return sum_f32_sse(data, n);
            #elif defined(NP_SIMD_NEON)
                return sum_f32_neon(data, n);
            #else
                T sum = T{0};
                for (std::size_t i = 0; i < n; ++i) {
                    sum += data[i];
                }
                return sum;
            #endif
        } else {
            T sum = T{0};
            for (std::size_t i = 0; i < n; ++i) {
                sum += data[i];
            }
            return sum;
        }
    }

    /**
     * @brief Vectorized subtraction with automatic dispatch.
     */
    template <typename T>
    inline void sub_vectorized(const T* a, const T* b, T* out, std::size_t n) {
        if constexpr (std::is_same_v<T, double>) {
            #if defined(NP_SIMD_AVX512)
                sub_f64_avx512(a, b, out, n);
            #elif defined(NP_SIMD_AVX)
                sub_f64_avx(a, b, out, n);
            #elif defined(NP_SIMD_SSE2)
                sub_f64_sse2(a, b, out, n);
            #else
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = a[i] - b[i];
                }
            #endif
        } else if constexpr (std::is_same_v<T, float>) {
            #if defined(NP_SIMD_AVX512)
                sub_f32_avx512(a, b, out, n);
            #elif defined(NP_SIMD_AVX)
                sub_f32_avx(a, b, out, n);
            #elif defined(NP_SIMD_SSE2)
                sub_f32_sse(a, b, out, n);
            #elif defined(NP_SIMD_NEON)
                sub_f32_neon(a, b, out, n);
            #else
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = a[i] - b[i];
                }
            #endif
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = a[i] - b[i];
            }
        }
    }

    /**
     * @brief Vectorized division with automatic dispatch.
     */
    template <typename T>
    inline void div_vectorized(const T* a, const T* b, T* out, std::size_t n) {
        if constexpr (std::is_same_v<T, double>) {
            #if defined(NP_SIMD_AVX512)
                div_f64_avx512(a, b, out, n);
            #elif defined(NP_SIMD_AVX)
                div_f64_avx(a, b, out, n);
            #elif defined(NP_SIMD_SSE2)
                div_f64_sse2(a, b, out, n);
            #else
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = a[i] / b[i];
                }
            #endif
        } else if constexpr (std::is_same_v<T, float>) {
            #if defined(NP_SIMD_AVX512)
                div_f32_avx512(a, b, out, n);
            #elif defined(NP_SIMD_AVX)
                div_f32_avx(a, b, out, n);
            #elif defined(NP_SIMD_SSE2)
                div_f32_sse(a, b, out, n);
            #elif defined(NP_SIMD_NEON)
                div_f32_neon(a, b, out, n);
            #else
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = a[i] / b[i];
                }
            #endif
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = a[i] / b[i];
            }
        }
    }

} // namespace simd
} // namespace np

#endif // NP_SIMD_HPP
