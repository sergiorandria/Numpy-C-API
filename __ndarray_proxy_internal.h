/**
 * @file ndarray_proxy.h
 * @brief Proxy classes for multidimensional array subscript access
 *
 * This file provides stack-based proxy classes that enable intuitive
 * chained subscript syntax (arr[i][j][k]) for multidimensional arrays.
 * The implementation uses compile-time fixed-size stacks to avoid heap
 * allocations, making it suitable for high-performance computing.
 *
 * @author Sergio Randriamihoatra
 * @version 1.0
 * @date 2026
 */

#ifndef NDARRAY_PROXY_H
#define NDARRAY_PROXY_H

#include <array>
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <utility>

namespace np
{
    // Forward declaration
    template <typename T> class Ndarray;

    /**
     * @brief Stack-based index storage that avoids heap allocations
     *
     * This structure replaces std::vector<size_t> for storing indices during
     * proxy chaining. It lives entirely on the stack, providing O(1) operations
     * with no dynamic memory overhead.
     *
     * @tparam MaxDims Maximum number of dimensions supported (default 8)
     *
     * The default MaxDims=8 covers all practical ndarray depths while keeping
     * the structure small enough to be efficiently copied by value.
     */
    template <std::size_t MaxDims = 8>
    struct IndexStack
    {
        std::array<std::size_t, MaxDims> m_data{};  ///< Fixed-size index storage
        std::size_t m_count = 0;                     ///< Number of valid indices

        /**
         * @brief Appends an index to the stack
         * @param v The index value to append
         * @throws None (noexcept)
         */
        constexpr void push_back(std::size_t v) noexcept
        {
            m_data[m_count++] = v;
        }

        /**
         * @brief Returns the number of indices currently stored
         * @return Current count of indices
         */
        [[nodiscard]] constexpr std::size_t size() const noexcept
        {
            return m_count;
        }

        /**
         * @brief Accesses an index by position
         * @param i Position of the index (0-based)
         * @return The index value at position i
         * @pre i < size()
         */
        constexpr auto operator[](std::size_t index) const noexcept -> std::size_t
        {
            return m_data[index];
        }

        /**
         * @brief Returns iterator to the beginning of the index sequence
         * @return Pointer to the first index
         */
        [[nodiscard]] constexpr const std::size_t *begin() const noexcept
        {
            return m_data.data();
        }

        /**
         * @brief Returns iterator to the end of the index sequence
         * @return Pointer to one past the last valid index
         */
        [[nodiscard]] constexpr const std::size_t *end() const noexcept
        {
            return m_data.data() + m_count;
        }
    };

    /**
     * @brief Base class for multidimensional array subscript proxies
     *
     * This template provides the core functionality for chained subscript
     * operations. It maintains a stack of indices and a reference to the
     * underlying Ndarray, forwarding operations to the appropriate array
     * methods when the final dimension is reached.
     *
     * @tparam T Element type of the array
     * @tparam IsConst Whether this is a const proxy (read-only access)
     * @tparam MaxDims Maximum number of dimensions supported
     *
     * When IsConst is true, the proxy binds to a const Ndarray and provides
     * only read operations. When false, it binds to a non-const reference
     * and supports assignment.
     */
    template <typename T, bool IsConst, std::size_t MaxDims = 8>
    class ProxyBase
    {
        /**
         * @brief The appropriate array type based on const-ness
         */
        using Array = std::conditional_t<IsConst, const Ndarray<T>, Ndarray<T>>;

        /**
         * @brief The stack type for index storage
         */
        using Stack = IndexStack<MaxDims>;

        /**
         * @brief Self type alias for return type deduction
         */
        using Self = ProxyBase<T, IsConst, MaxDims>;

        Array &m_array;    ///< Reference to the underlying array
        Stack m_indices;   ///< Stack of accumulated indices (stack-allocated)

      public:
        /**
         * @brief Constructs a proxy for a given array and index stack
         * @param arr Reference to the underlying Ndarray
         * @param idx Initial index stack (may be empty or contain initial indices)
         */
        constexpr ProxyBase(Array &arr, Stack idx) noexcept
            : m_array(arr), m_indices(idx) {}

        /**
         * @brief Assigns a value to the array element (non-const only)
         * @param v Value to assign
         * @return Reference to this proxy for chaining
         * @requires IsConst == false
         *
         * This operator is only available for non-const proxies. It forwards
         * the assignment to the underlying array's set method using the
         * accumulated indices.
         */
        template <bool C = IsConst>
        constexpr auto operator=(const T &v) noexcept -> Self &
        requires(!C)
        {
            m_array.set(m_indices, v);
            return *this;
        }

        /**
         * @brief Move-assigns a value to the array element (non-const only)
         * @param v Value to move-assign
         * @return Reference to this proxy for chaining
         * @requires IsConst == false
         */
        template <bool C = IsConst>
        inline constexpr auto operator=(T && v) noexcept -> Self &
        requires(!C)
        {
            m_array.set(m_indices, std::move(v));
            return *this;
        }

        /**
         * @brief Converts the proxy to the element value (read operation)
         * @return The value stored at the indexed position
         *
         * This conversion operator allows the proxy to be used directly
         * where a value of type T is expected, such as in expressions or
         * function arguments.
         */
        [[nodiscard]]
        constexpr operator T() const noexcept
        {
            return m_array.get(m_indices);
        }

        /**
         * @brief Equality comparison with value
         * @param v Value to compare against
         * @return true if the array element equals v
         */
        [[nodiscard]]
        constexpr auto operator==(const T& v) const noexcept -> bool
        {
            return static_cast<T>(*this) == v;
        }

        /**
         * @brief Inequality comparison with value
         * @param v Value to compare against
         * @return true if the array element does not equal v
         */
        [[nodiscard]]
        constexpr auto operator!=(const T& v) const noexcept -> bool
        {
            return static_cast<T>(*this) != v;
        }

        /**
         * @brief Equality comparison with another proxy
         * @param other Another proxy to compare against
         * @return true if both proxies refer to the same element with equal value
         */
        [[nodiscard]]
        constexpr auto operator==(const Self& other) const noexcept -> bool
        {
            return static_cast<T>(*this) == static_cast<T>(other);
        }

        /**
         * @brief Inequality comparison with another proxy
         * @param other Another proxy to compare against
         * @return true if proxies refer to different elements or have different values
         */
        [[nodiscard]]
        constexpr auto operator!=(const Self& other) const noexcept -> bool
        {
            return !(*this == other);
        }

        /**
         * @brief Equality comparison with other arithmetic types
         * @tparam U Other numeric type
         * @param v Value to compare against
         * @return true if the array element equals v
         */
        template<typename U>
        [[nodiscard]]
        constexpr auto operator==(const U& v) const noexcept -> bool
        {
            return static_cast<T>(*this) == static_cast<T>(v);
        }

        /**
         * @brief Inequality comparison with other arithmetic types
         * @tparam U Other numeric type
         * @param v Value to compare against
         * @return true if the array element does not equal v
         */
        template<typename U>
        [[nodiscard]]
        constexpr auto operator!=(const U& v) const noexcept -> bool
        {
            return static_cast<T>(*this) != static_cast<T>(v);
        }

        /**
         * @brief Assigns from another proxy (copies the value)
         * @param other Source proxy to copy from
         * @return Reference to this proxy
         *
         * This enables expressions like arr[i][j] = arr[k][l] by extracting
         * the value from the source proxy and assigning it to this proxy's
         * position.
         */
        constexpr auto operator=(const Self& other) -> Self &
        {
            if (*this != other)
            {
                T value = static_cast<T>(other);
                m_array.set(m_indices, value);
            }
            return *this;
        }

        /**
         * @brief Stream insertion operator for proxy
         * @param os Output stream
         * @param proxy Proxy to output
         * @return Reference to the output stream
         */
        [[nodiscard]]
        friend auto operator<<(std::ostream& os, const Self& proxy) -> std::ostream &
        {
            os << static_cast<T>(proxy);
            return os;
        }

        /**
         * @brief Advances the proxy to the next dimension
         * @param idx Index for the current dimension
         * @return A new proxy for the next dimension (or final reference)
         *
         * This operator creates a new proxy by copying the current index
         * stack and appending the new index. The operation is cheap because
         * the stack is fixed-size and stack-allocated.
         */
        [[nodiscard]]
        constexpr auto operator[](std::size_t idx) const noexcept -> Self
        {
            Stack next = m_indices; // trivial copy — no heap touch
            next.push_back(idx);
            return Self(m_array, next);
        }
    };

    /**
     * @brief Convenience alias for non-const proxy (read-write access)
     * @tparam T Element type of the array
     * @tparam MaxDims Maximum number of dimensions supported
     */
    template <typename T, std::size_t MaxDims = 8>
    using Proxy = ProxyBase<T, false, MaxDims>;

    /**
     * @brief Convenience alias for const proxy (read-only access)
     * @tparam T Element type of the array
     * @tparam MaxDims Maximum number of dimensions supported
     */
    template <typename T, std::size_t MaxDims = 8>
    using ConstProxy = ProxyBase<T, true, MaxDims>;

} // namespace np

#endif // NDARRAY_PROXY_H