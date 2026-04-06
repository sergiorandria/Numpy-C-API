/**
 * @file __numpy_fft_internal.h
 * @brief Internal FFT (Fast Fourier Transform) infrastructure
 *
 * This file defines the core interfaces and enumerations for Fast Fourier
 * Transform operations in the NumPy-like library. It provides a unified
 * interface for multiple FFT backends (FFTW, MKL, CuFFT, etc.) and supports
 * various transform types (complex-to-complex, real-to-complex, etc.).
 *
 * The design follows the strategy pattern, allowing different FFT
 * implementations to be swapped at runtime while maintaining a consistent API.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 * @version 1.0
 * @date 2026
 *
 * @see https://www.fftw.org/ FFTW library
 * @see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html Intel MKL
 */

#ifndef __NUMPY_FFT_INTERNAL_H
#define __NUMPY_FFT_INTERNAL_H

#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>
#include <optional>

#include "../../dtype.hpp"
#include "../../__dtype_conversion.h"

/**
 * @namespace np
 * @brief Main namespace for the NumPy-like library
 */
namespace np
{
    /**
     * @brief FFT direction enumeration
     *
     * Specifies the direction of the FFT transformation.
     *
     * @note Forward transforms typically go from time/space domain to frequency domain.
     * @note Inverse transforms go from frequency domain back to time/space domain.
     *
     * @example
     * @code
     * auto forward = np::fft::fft(data, np::FFTDirection::Forward);
     * auto inverse = np::fft::fft(forward, np::FFTDirection::Inverse);
     * @endcode
     */
    enum class FFTDirection : std::uint8_t
    {
        Forward,  ///< Forward FFT: time/space domain → frequency domain
        Inverse   ///< Inverse FFT: frequency domain → time/space domain
    };

    /**
     * @brief FFT normalization mode enumeration
     *
     * Specifies how the transform is normalized. Different applications
     * require different normalization conventions.
     *
     * @see https://en.wikipedia.org/wiki/DFT_matrix#Unitary_transform
     *
     * @example
     * @code
     * // Unitary transform (preserves energy)
     * auto result = np::fft::fft(data, np::FFTNorm::Ortho);
     *
     * // Standard NumPy convention
     * auto result = np::fft::fft(data, np::FFTNorm::Backward);
     * @endcode
     */
    enum class FFTNorm : std::uint8_t
    {
        None,       ///< No normalization: forward = 1, inverse = 1/N
        Ortho,      ///< Orthogonal normalization: 1/sqrt(N) for both directions
        Backward    ///< Backward normalization: forward = 1, inverse = 1/N (NumPy default)
    };

    /**
     * @brief FFT implementation backend enumeration
     *
     * Available FFT implementations that can be used as backends.
     *
     * @note Not all backends may be available on all platforms.
     * @note Use `FftInternal::is_backend_available()` to check availability.
     *
     * @see FftInternal::create()
     * @see FftInternal::get_available_backends()
     */
    enum class FFTBackend : std::uint8_t
    {
        Auto,       ///< Automatically select best available backend
        FFTW,       ///< FFTW library (Fastest Fourier Transform in the West)
        MKL,        ///< Intel Math Kernel Library
        CuFFT,      ///< NVIDIA CUDA FFT (GPU acceleration)
        Stockham,   ///< Stockham auto-sort algorithm (in-place, cache-friendly)
        CooleyTukey,///< Classic Cooley-Tukey radix-2 algorithm
        PrimeFactor ///< Prime factor algorithm for non-power-of-2 sizes
    };

    /**
     * @brief FFT plan cache policy enumeration
     *
     * Determines how FFT plans are cached for repeated transforms of the
     * same size and type. Caching can significantly improve performance
     * when performing many transforms of identical dimensions.
     *
     * @note Plan creation can be expensive; caching is recommended for
     *       repeated transforms.
     * @note Memory usage increases with more aggressive caching policies.
     *
     * @example
     * @code
     * auto fft = FftFactory<float>::create();
     * fft->set_cache_policy(np::FFTCachePolicy::CacheFull);
     * // Subsequent transforms of same size will reuse plans
     * @endcode
     */
    enum class FFTCachePolicy : std::uint8_t
    {
        NoCache,    ///< Don't cache plans (recompute each time - memory efficient)
        CacheSize,  ///< Cache by size only (size_t key - good for 1D transforms)
        CacheFull   ///< Cache by full parameters (size, type, direction - most performant)
    };
}

/**
 * @namespace np::Interface
 * @brief Internal interfaces for FFT implementations
 *
 * This namespace contains abstract base classes that define the contract
 * for all FFT implementations. Users should not use these classes directly.
 */
namespace np::Interface
{
    /**
     * @brief Base interface for all FFT implementations
     *
     * This template class defines the common interface for all FFT variants.
     * Concrete implementations should inherit from this class and implement
     * the pure virtual methods. The interface supports multiple transform
     * types (complex-to-complex, real-to-complex, complex-to-real) and
     * both 1D and multi-dimensional transforms.
     *
     * @tparam T Data type (must be a valid np::dtype enumeration value)
     *
     * @note This class follows the Interface Segregation Principle,
     *       providing a comprehensive but focused API for FFT operations.
     *
     * @example
     * @code
     * // Create FFT instance for double-precision floating point
     * auto fft = FftInternal<double>::create(FFTBackend::FFTW);
     *
     * // Prepare input data
     * std::vector<std::complex<double>> data(1024);
     * // ... fill data ...
     *
     * // Perform forward FFT
     * auto result = fft->fft1d(data, FFTDirection::Forward, FFTNorm::Backward);
     * @endcode
     */
    template <np::dtype T>
    class FftInternal_
    {
      public:
        /**
         * @brief Type aliases for convenience
         *
         * These aliases provide a consistent type interface across
         * different FFT implementations.
         */
        using ValueType =
            np_type_to_cxx<T>::type;                           ///< Value type (float/double/etc)
        using ComplexType = std::complex<ValueType>;            ///< Complex type
        using RealVector =
            std::vector<ValueType>;                             ///< Vector of real values
        using ComplexVector =
            std::vector<ComplexType>;                           ///< Vector of complex values
        using SizeType =
            std::size_t;                           ///< Size type for dimensions

        /**
         * @brief Virtual destructor for proper cleanup of derived classes
         *
         * Ensures that derived class destructors are called correctly
         * when deleting through base class pointers.
         */
        virtual ~FftInternal_() = default;

        /**
         * @brief Performs 1D Complex-to-Complex FFT
         *
         * Transforms a complex-valued sequence from the time/space domain
         * to the frequency domain (or vice versa).
         *
         * @param input Input complex array (size must be valid for the implementation)
         * @param direction Forward or inverse transform
         * @param norm Normalization mode (None, Ortho, or Backward)
         * @return ComplexVector Transformed complex array (same size as input)
         *
         * @throws std::invalid_argument if input is empty or size not supported
         * @throws std::runtime_error if the transform fails
         *
         * @complexity O(N log N) for supported sizes
         * @memory    O(N) for output (in-place transforms also available)
         *
         * @example
         * @code
         * std::vector<std::complex<double>> signal = {{1,0}, {2,0}, {3,0}, {4,0}};
         * auto spectrum = fft->fft1d(signal, FFTDirection::Forward, FFTNorm::Backward);
         * @endcode
         */
        virtual auto fft1d(const ComplexVector& input,
                           FFTDirection direction,
                           FFTNorm norm) -> decltype(ComplexVector()) = 0;

        /**
         * @brief Performs 1D Real-to-Complex FFT
         *
         * Transforms a real-valued sequence to its complex frequency domain
         * representation. Due to Hermitian symmetry, only the non-redundant
         * half of the output is returned (n/2 + 1 elements).
         *
         * @param input Input real array (size N)
         * @param direction Forward or inverse transform
         * @param norm Normalization mode
         * @return ComplexVector Transformed complex array (size floor(N/2)+1)
         *
         * @throws std::invalid_argument if input is empty
         *
         * @note For real inputs, X[k] = conj(X[N-k]), so only the first
         *       floor(N/2)+1 elements are needed.
         * @note This is more memory-efficient than complex-to-complex FFT.
         *
         * @example
         * @code
         * std::vector<double> real_signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
         * auto spectrum = fft->rfft1d(real_signal, FFTDirection::Forward, FFTNorm::Backward);
         * // spectrum has size 5 (8/2 + 1)
         * @endcode
         */
        virtual auto rfft1d(const RealVector& input,
                            FFTDirection direction,
                            FFTNorm norm) -> decltype(ComplexVector()) = 0;

        /**
         * @brief Performs 1D Complex-to-Real FFT (inverse of rfft)
         *
         * Transforms a complex frequency-domain representation (packed format)
         * back to the real-valued time/space domain.
         *
         * @param input Input complex array (packed format from rfft)
         * @param n Original real size (must match the original rfft input size)
         * @param direction Forward or inverse transform
         * @param norm Normalization mode
         * @return RealVector Reconstructed real array (size n)
         *
         * @throws std::invalid_argument if input size is inconsistent with n
         *
         * @note This is the inverse operation of rfft1d.
         * @note For perfect reconstruction, use the same norm and direction.
         *
         * @example
         * @code
         * auto spectrum = fft->rfft1d(real_signal, FFTDirection::Forward, FFTNorm::Backward);
         * auto reconstructed = fft->irfft1d(spectrum, real_signal.size(),
         *                                   FFTDirection::Inverse, FFTNorm::Backward);
         * // reconstructed should approximate real_signal
         * @endcode
         */
        virtual auto irfft1d(const ComplexVector& input,
                             SizeType n,
                             FFTDirection direction,
                             FFTNorm norm) -> decltype(RealVector()) = 0;

        /**
         * @brief Performs multi-dimensional Complex-to-Complex FFT
         *
         * Transforms a multi-dimensional complex array along all dimensions
         * using a row-major (C-order) layout.
         *
         * @param input Input complex array (flattened in row-major order)
         * @param shape Dimensions of the array (e.g., {rows, cols, depth})
         * @param direction Forward or inverse transform
         * @param norm Normalization mode
         * @return ComplexVector Transformed complex array (same shape and layout)
         *
         * @throws std::invalid_argument if shape is empty or product doesn't match input size
         *
         * @complexity O(N log N) where N is total number of elements
         *
         * @example
         * @code
         * // 2D FFT on a 256x256 image
         * std::vector<std::complex<float>> image(256 * 256);
         * auto spectrum = fft->fftnd(image, {256, 256},
         *                            FFTDirection::Forward, FFTNorm::Ortho);
         * @endcode
         */
        virtual auto fftnd(const ComplexVector& input,
                           const std::vector<SizeType> &shape,
                           FFTDirection direction,
                           FFTNorm norm) -> decltype(ComplexVector()) = 0;

        /**
         * @brief Performs multi-dimensional Real-to-Complex FFT
         *
         * Transforms a multi-dimensional real array to its complex frequency
         * domain representation, exploiting Hermitian symmetry for efficiency.
         *
         * @param input Input real array (flattened in row-major order)
         * @param shape Dimensions of the array
         * @param axes Axes to transform (empty = transform all axes)
         * @param norm Normalization mode
         * @return ComplexVector Transformed complex array (packed format)
         *
         * @throws std::invalid_argument if axes are out of range
         *
         * @note The output is packed to remove redundant Hermitian-symmetric elements.
         * @note For 2D transforms, the last axis is the one that gets truncated.
         *
         * @example
         * @code
         * // 2D real FFT on a grayscale image
         * std::vector<float> image(256 * 256);
         * auto spectrum = fft->rfftnd(image, {256, 256}, {}, FFTNorm::Backward);
         * // spectrum size = 256 * (256/2 + 1)
         * @endcode
         */
        virtual auto rfftnd(const RealVector& input,
                            const std::vector<SizeType> &shape,
                            const std::vector<int> &axes,
                            FFTNorm norm) -> decltype(ComplexVector()) = 0;

        /**
         * @brief Performs in-place Complex-to-Complex FFT
         *
         * Transforms a complex array in-place, modifying the input array
         * directly to save memory.
         *
         * @param input Output array (modified in-place)
         * @param direction Forward or inverse transform
         * @param norm Normalization mode
         *
         * @throws std::invalid_argument if input is empty
         *
         * @note In-place transforms are more memory efficient but require
         *       that the input array be modifiable.
         *
         * @example
         * @code
         * std::vector<std::complex<double>> data(1024);
         * fft->fft1d_inplace(data, FFTDirection::Forward, FFTNorm::Backward);
         * // data now contains the transformed result
         * @endcode
         */
        virtual void fft1d_inplace(ComplexVector& input,
                                   FFTDirection direction,
                                   FFTNorm norm) = 0;

        /**
         * @brief Performs in-place Real-to-Complex FFT
         *
         * Transforms a real array in-place, reusing the memory for the
         * complex output. The input array must have extra capacity to
         * hold the complex results.
         *
         * @param input Input real array, output complex array (reinterpreted)
         * @param direction Forward or inverse transform
         * @param norm Normalization mode
         *
         * @note The input array must have space for at least (n/2+1) complex values.
         * @warning This operation reinterprets the memory; use with care.
         *
         * @example
         * @code
         * // Need extra space for complex output
         * std::vector<double> data(1024 + sizeof(std::complex<double>) * (1024/2+1));
         * fft->rfft1d_inplace(data, FFTDirection::Forward, FFTNorm::Backward);
         * @endcode
         */
        virtual void rfft1d_inplace(RealVector& input,
                                    FFTDirection direction,
                                    FFTNorm norm) = 0;

        /**
         * @brief Creates/retrieves a plan for repeated transforms of the same size
         *
         * Precomputes the optimal transform plan for a given size and type,
         * which can be reused for multiple transforms to avoid repeated
         * setup overhead.
         *
         * @param n Size of the transform
         * @param direction Forward or inverse transform
         * @param is_real Whether this is a real-to-complex transform
         * @return SizeType Plan ID (implementation-specific handle)
         *
         * @note Plans are cached according to the current cache policy.
         * @note Plan creation is thread-safe for read operations.
         *
         * @example
         * @code
         * auto plan = fft->create_plan(1024, FFTDirection::Forward, false);
         * for (int i = 0; i < 1000; ++i) {
         *     fft->execute_plan(plan, input[i], output[i]);
         * }
         * fft->destroy_plan(plan);
         * @endcode
         */
        virtual auto create_plan(SizeType n,
                                 FFTDirection direction,
                                 bool is_real) -> decltype(SizeType()) = 0;

        /**
         * @brief Executes a precomputed plan
         *
         * Performs an FFT using a previously created plan. This is faster
         * than creating a new plan for each transform.
         *
         * @param plan_id Plan ID returned by create_plan()
         * @param input Input array
         * @param output Output array
         *
         * @throws std::invalid_argument if plan_id is invalid
         * @throws std::runtime_error if execution fails
         *
         * @note The input and output sizes must match the plan's configuration.
         */
        virtual void execute_plan(SizeType plan_id,
                                  const ComplexVector& input,
                                  ComplexVector& output) = 0;

        /**
         * @brief Destroys a previously created plan
         *
         * Releases resources associated with a plan. The plan ID becomes
         * invalid after this call.
         *
         * @param plan_id Plan ID to destroy
         *
         * @note It's safe to call destroy_plan on the same ID multiple times
         *       (subsequent calls will have no effect).
         */
        virtual void destroy_plan(SizeType plan_id) = 0;

        /**
         * @brief Clears all cached plans
         *
         * Releases all plan resources and empties the plan cache.
         * This is useful when memory needs to be freed or when the
         * cache policy is changed.
         */
        virtual void clear_plans() = 0;

        /**
         * @brief Sets the cache policy for FFT plans
         *
         * Configures how aggressively FFT plans are cached. Different
         * policies trade off memory usage against plan creation time.
         *
         * @param policy Cache policy to use (NoCache, CacheSize, or CacheFull)
         *
         * @note Changing the policy clears existing cached plans.
         *
         * @see FFTCachePolicy
         */
        virtual void set_cache_policy(FFTCachePolicy policy) = 0;

        /**
         * @brief Gets the current cache policy
         *
         * @return FFTCachePolicy Current cache policy
         */
        [[nodiscard]] virtual auto get_cache_policy() const -> decltype(
            FFTCachePolicy()) = 0;

        /**
         * @brief Sets the number of threads to use (if supported)
         *
         * Configures multi-threading for FFT operations. This is only
         * effective for backends that support threading (e.g., FFTW with
         * OpenMP, MKL with TBB).
         *
         * @param num_threads Number of threads (0 = auto/use system default)
         *
         * @note Setting num_threads=1 disables multi-threading.
         * @note This setting may be ignored if the backend doesn't support threading.
         *
         * @example
         * @code
         * fft->set_num_threads(4);  // Use 4 threads
         * @endcode
         */
        virtual void set_num_threads(int num_threads) = 0;

        /**
         * @brief Gets the current number of threads
         *
         * @return int Number of threads (0 = auto)
         */
        [[nodiscard]] virtual auto get_num_threads() const -> decltype(int {}) = 0;

        /**
         * @brief Checks if a size is supported by this implementation
         *
         * Different FFT backends have different size limitations.
         * Some only support powers of two, while others support arbitrary sizes.
         *
         * @param n Size to check
         * @return true if the size is supported, false otherwise
         *
         * @example
         * @code
         * if (!fft->is_size_supported(1000)) {
         *     // Use zero-padding or a different backend
         *     size_t recommended = fft->get_recommended_size(1000);
         * }
         * @endcode
         */
        [[nodiscard]] virtual auto is_size_supported(SizeType n) const -> decltype(
        bool {}) = 0;

        /**
         * @brief Gets the recommended size for optimal performance
         *
         * Returns the next size that is optimal for the current backend
         * (e.g., next power of two, next smooth number, etc.).
         *
         * @param n Desired size
         * @return SizeType Recommended size (>= n)
         *
         * @note The recommended size is always >= the requested size.
         * @note For optimal performance, zero-pad inputs to the recommended size.
         *
         * @example
         * @code
         * size_t optimal = fft->get_recommended_size(1000);
         * // optimal might be 1024 (next power of two)
         * @endcode
         */
        [[nodiscard]] virtual auto get_recommended_size(SizeType n) const -> decltype(
        SizeType {}) = 0;

        /**
         * @brief Gets the alignment requirement for optimal performance
         *
         * Returns the memory alignment required for optimal performance.
         * Aligning input/output arrays to this boundary can improve speed.
         *
         * @return SizeType Required alignment in bytes (0 = no requirement)
         *
         * @example
         * @code
         * size_t alignment = fft->get_alignment();
         * if (alignment > 0) {
         *     // Use aligned_alloc or similar
         *     auto* data = static_cast<std::complex<double>*>(
         *         aligned_alloc(alignment, size * sizeof(std::complex<double>)));
         * }
         * @endcode
         */
        [[nodiscard]] virtual auto get_alignment() const -> decltype(SizeType {}) = 0;

        /**
         * @brief Returns the name of the FFT implementation
         *
         * @return const char* Implementation name (e.g., "FFTW", "MKL", "Stockham")
         *
         * @note The returned string is static and should not be freed.
         */
        [[nodiscard]] virtual auto get_implementation_name() const -> decltype(
            static_cast<const char *>(nullptr)) = 0;

        /**
         * @brief Returns the backend being used
         *
         * @return FFTBackend Enum value identifying the active backend
         */
        [[nodiscard]] virtual auto get_backend() const -> decltype(FFTBackend()) = 0;

        /**
         * @brief Checks if the implementation supports double precision
         *
         * @return true if double precision is supported
         *
         * @note Some backends (e.g., GPU implementations) may only support
         *       single precision.
         */
        [[nodiscard]] virtual auto supports_double_precision() const -> decltype(
        bool {}) = 0;

        /**
         * @brief Checks if the implementation supports single precision
         *
         * @return true if single precision is supported
         */
        [[nodiscard]] virtual auto supports_single_precision() const -> decltype(
        bool {}) = 0;


        /**
         * @brief Creates an FFT instance of the specified type
         *
         * Factory method that creates and returns an FFT implementation
         * for the given data type and backend preference.
         *
         * @tparam U C++ data type (float, double, etc.)
         * @param backend Preferred backend (Auto = automatic selection)
         * @return std::unique_ptr<FftInternal_<cxx_to_np_type<U>::value>>
         *         Unique pointer to the FFT instance
         *
         * @throws std::runtime_error if no suitable backend is available
         *
         * @example
         * @code
         * // Create double-precision FFT with automatic backend
         * auto fft = FftInternal<double>::create(FFTBackend::Auto);
         *
         * // Create single-precision FFT using FFTW
         * auto fft_float = FftInternal<float>::create(FFTBackend::FFTW);
         * @endcode
         */
        template <typename U>
        static auto create(FFTBackend backend) -> decltype(
            std::unique_ptr < FftInternal_<cxx_to_np_type<U>::value>>());

        /**
         * @brief Checks if a particular backend is available
         *
         * Determines whether a specific FFT implementation is available
         * on the current system (library installed, hardware support, etc.).
         *
         * @param backend Backend to check
         * @return true if the backend is available, false otherwise
         *
         * @example
         * @code
         * if (FftInternal<double>::is_backend_available(FFTBackend::CuFFT)) {
         *     auto gpu_fft = FftInternal<double>::create(FFTBackend::CuFFT);
         * }
         * @endcode
         */
        static auto is_backend_available(FFTBackend backend) -> decltype(bool {});

        /**
         * @brief Gets the list of available backends
         *
         * Returns a vector of all FFT backends that are available on the
         * current system.
         *
         * @return std::vector<FFTBackend> List of available backends
         *
         * @example
         * @code
         * auto backends = FftInternal<double>::get_available_backends();
         * std::cout << "Available backends: ";
         * for (auto b : backends) {
         *     std::cout << static_cast<int>(b) << " ";
         * }
         * @endcode
         */
        static auto get_available_backends() -> decltype(std::vector<FFTBackend> {});

      protected:
        /**
         * @brief Protected constructor to prevent direct instantiation
         *
         * This class is abstract and cannot be instantiated directly.
         * Use the factory method create() instead.
         */
        FftInternal_() = default;

        /**
         * @brief Validates input parameters
         *
         * Performs common validation checks on input parameters.
         * Derived classes should call this before performing transforms.
         *
         * @param input Input vector
         * @param n Expected size (if any, empty optional = no size check)
         *
         * @throws std::invalid_argument if validation fails
         *
         * @note This is a convenience method for derived classes.
         */
        virtual void validate_input(const ComplexVector& input,
                                    std::optional<SizeType> n) const
        {
            if (input.empty())
            {
                throw std::invalid_argument("FFT input cannot be empty");
            }
            if (n.has_value() && input.size() != n.value())
            {
                throw std::invalid_argument("FFT input size mismatch");
            }
        }
    };
}

/**
 * @namespace np
 * @brief Type aliases for common FFT instantiations
 *
 * These aliases provide convenient access to FFT implementations for
 * specific data types.
 */
namespace np
{
#define FftInternal FftInternal_
    using FftInternal_int16_   = np::Interface::FftInternal<np::int16>;
    using FftInternal_int32_   = np::Interface::FftInternal<np::int32>;
    using FftInternal_int64_   = np::Interface::FftInternal<np::int64>;
    using FftInternal_uint8_    = np::Interface::FftInternal<np::uint8>;
    using FftInternal_uint16_   = np::Interface::FftInternal<np::uint16>;
    using FftInternal_uint32_   = np::Interface::FftInternal<np::uint32>;
    using FftInternal_uint64_   = np::Interface::FftInternal<np::uint64>;
    using FftInternal_complex64_  = np::Interface::FftInternal<np::complex64>;
    using FftInternal_complex128_ = np::Interface::FftInternal<np::complex128>;

#define FftInternal_int16   FftInternal_int16_
#define FftInternal_int32   FftInternal_int32_
#define FftInternal_int64   FftInternal_int64_
#define FftInternal_uint8   FftInternal_int8_
#define FftInternal_uint16  FftInternal_int16_
#define FftInternal_uint32  FftInternal_int32_
#define FftInternal_uint64  FftInternal_int64_
}

#endif // __NUMPY_FFT_INTERNAL_H