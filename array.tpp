/**
 * @file array.tpp
 * @brief Template implementation of the Array class
 * @author Your Name
 * @version 1.0
 * @date 2025
 *
 * This file contains the template implementations for the np::Array class.
 * It provides a 1D container with NumPy-like semantics and lazy allocation.
 */

#ifndef NP_ARRAY_TPP
#define NP_ARRAY_TPP

#include "array.h"
#include "numpy/exceptions"

#include <optional>
#include <ranges>
#include <utility>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <type_traits>

namespace np
{

    /**
     * @brief Constructs an Array from any range
     * @tparam Container Type of the input range (must satisfy std::ranges::range)
     * @param c The input range to copy elements from
     * @param t Optional data type override
     *
     * Design notes:
     * - Lazy allocation: data is emplaced only during construction
     * - If the range is sized, pre-reserves capacity to avoid reallocation
     * - dtype can be optionally overridden; otherwise inferred/defaulted
     */
    template<typename _Tp>
    template<typename Container>
    requires std::ranges::range<Container>
    Array<_Tp>::Array(const Container& c,
                      std::optional<dtype> t /* = std::nullopt */)
    {
        // Lazily initialize storage for performance improvements
        this->data.emplace();
        // Reserve capacity when size information is available
        if constexpr(std::ranges::sized_range<Container>)
        {
            this->data->reserve(std::ranges::size(c));
        }
        // Copy elements from the input range
        std::ranges::copy(c, std::back_inserter(*this->data));
        if (t)
        {
            this->dtype = *t;
        }
    }

    /**
     * @brief Constructs an Array from an initializer list
     * @param il Initializer list to copy elements from
     * @param t Optional data type override
     *
     * This mirrors std::vector behavior and provides a convenient
     * syntax for small arrays.
     */
    template<typename _Tp>
    Array<_Tp>::Array(std::initializer_list<_Tp> initList, std::optional<dtype> t)
    {
        this->data.emplace();
        this->data->reserve(il.size());
        this->data->assign(il.begin(), il.end());
        if (t)
        {
            this->dtype = *t;
        }
    }

    /**
     * @brief Unchecked element access (non-const)
     * @param i Index of the element to access
     * @return Reference to the element at position i
     *
     * Notes:
     * - If the array has not been initialized yet, this triggers allocation
     * - Bounds are NOT checked (std::vector semantics)
     * - noexcept by design: caller is responsible for correctness
     */
    template<typename _Tp>
    typename Array<_Tp>::reference Array<_Tp>::operator[](size_type i) noexcept
    {
        if (!this->data) [[unlikely]]
        {
            this->data.emplace();
        }
        return (*this->data)[i];
    }

    /**
     * @brief Unchecked element access (const)
     * @param i Index of the element to access
     * @return Const reference to the element at position i
     *
     * @pre The array must be initialized (data engaged)
     *
     * Accessing an uninitialized const Array is undefined behavior by design,
     * matching STL philosophy.
     */
    template<typename _Tp>
    auto Array<_Tp>::operator[](
        size_type i) const noexcept -> typename Array<_Tp>::const_reference
    {
        return (*this->data)[i];
    }

    /**
     * @brief Move assignment operator
     * @param other Array to move from
     * @return Reference to this array
     *
     * Fully transfers ownership of internal state.
     * noexcept enables optimizations (e.g., vector reallocation).
     */
    template<typename _Tp>
    auto Array<_Tp>::operator=(Array<_Tp>&& other) noexcept -> Array<_Tp> &
    {
        if (this != &other)
        {
            this->T        = std::move(other.T);
            this->data     = std::move(other.data);
            this->dtype    = std::move(other.dtype);
            this->flags    = std::move(other.flags);
            this->imag     = std::move(other.imag);
            this->real     = std::move(other.real);
            this->size     = std::move(other.size);
            this->itemsize = std::move(other.itemsize);
            this->nbytes   = std::move(other.nbytes);
            this->ndim     = std::move(other.ndim);
            this->strides  = std::move(other.strides);
        }
        return *this;
    }

    /**
     * @brief Assigns from an initializer list
     * @param init_list Initializer list to assign from
     * @return Reference to this array
     *
     * Guarantees storage exists and replaces contents.
     */
    template<typename _Tp>
    auto Array<_Tp>::operator=(std::initializer_list<_Tp> init_list) -> Array<_Tp> &
    {
        if (!this->data)
        {
            this->data.emplace();
        }
        this->data->assign(init_list.begin(), init_list.end());
        return *this;
    }

    /**
     * @brief Swaps the contents of two arrays
     * @param other Array to swap with
     *
     * Strong exception-safe swap.
     * Enables ADL and interoperates with std::swap.
     */
    template<typename _Tp>
    void Array<_Tp>::swap(Array<_Tp> &other) noexcept
    {
        using std::swap;
        swap(this->T,        other.T);
        swap(this->data,     other.data);
        swap(this->dtype,    other.dtype);
        swap(this->flags,    other.flags);
        swap(this->imag,     other.imag);
        swap(this->real,     other.real);
        swap(this->size,     other.size);
        swap(this->itemsize, other.itemsize);
        swap(this->nbytes,   other.nbytes);
        swap(this->ndim,     other.ndim);
        swap(this->strides,  other.strides);
    }

    /**
     * @brief Returns an iterator to the beginning (non-const)
     * @return Iterator to the first element
     *
     * If storage is uninitialized, returns a default-constructed iterator.
     * This preserves noexcept guarantees and avoids implicit allocation.
     */
    template<typename _Tp>
    auto Array<_Tp>::begin() noexcept
    {
        return this->data ? this->data->begin() : typename std::vector<_Tp>::iterator{};
    }

    /**
     * @brief Returns an iterator to the end (non-const)
     * @return Iterator to one past the last element
     */
    template<typename _Tp>
    auto Array<_Tp>::end() noexcept
    {
        return this->data ? this->data->end() : typename std::vector<_Tp>::iterator{};
    }

    /**
     * @brief Returns a const iterator to the beginning
     * @return Const iterator to the first element
     */
    template<typename _Tp>
    auto Array<_Tp>::begin() const noexcept
    {
        return this->data ? this->data->begin() : typename
               std::vector<_Tp>::const_iterator{};
    }

    /**
     * @brief Returns a const iterator to the end
     * @return Const iterator to one past the last element
     */
    template<typename _Tp>
    auto Array<_Tp>::end() const noexcept
    {
        return this->data ? this->data->end() : typename
               std::vector<_Tp>::const_iterator{};
    }

    /**
     * @brief Returns a const iterator to the beginning
     * @return Const iterator to the first element
     */
    template<typename _Tp>
    auto Array<_Tp>::cbegin() const noexcept
    {
        return begin();
    }

    /**
     * @brief Returns a const iterator to the end
     * @return Const iterator to one past the last element
     */
    template<typename _Tp>
    auto Array<_Tp>::cend() const noexcept
    {
        return end();
    }

    /**
     * @brief Returns a reverse iterator to the beginning (non-const)
     * @return Reverse iterator to the last element
     */
    template<typename _Tp>
    auto Array<_Tp>::rbegin() noexcept
    {
        return this->data ? this->data->rbegin() : typename
               std::vector<_Tp>::reverse_iterator{};
    }

    /**
     * @brief Returns a reverse iterator to the end (non-const)
     * @return Reverse iterator to one before the first element
     */
    template<typename _Tp>
    auto Array<_Tp>::rend() noexcept
    {
        return this->data ? this->data->rend() : typename
               std::vector<_Tp>::reverse_iterator{};
    }

    /**
     * @brief Returns a const reverse iterator to the beginning
     * @return Const reverse iterator to the last element
     */
    template<typename _Tp>
    auto Array<_Tp>::rbegin() const noexcept
    {
        return this->data ? this->data->rbegin() : typename
               std::vector<_Tp>::const_reverse_iterator{};
    }

    /**
     * @brief Returns a const reverse iterator to the end
     * @return Const reverse iterator to one before the first element
     */
    template<typename _Tp>
    auto Array<_Tp>::rend() const noexcept
    {
        return this->data ? this->data->rend() : typename
               std::vector<_Tp>::const_reverse_iterator{};
    }

    /**
     * @brief Returns a const reverse iterator to the beginning
     * @return Const reverse iterator to the last element
     */
    template<typename _Tp>
    auto Array<_Tp>::crbegin() const noexcept
    {
        return rbegin();
    }

    /**
     * @brief Returns a const reverse iterator to the end
     * @return Const reverse iterator to one before the first element
     */
    template<typename _Tp>
    auto Array<_Tp>::crend() const noexcept
    {
        return rend();
    }

    /**
     * @brief Bounds-checked element access (non-const)
     * @param i Index of the element to access
     * @return Reference to the element at position i
     * @throws std::logic_error if storage is uninitialized
     * @throws std::out_of_range if index is out of bounds
     */
    template<typename _Tp>
    typename Array<_Tp>::reference Array<_Tp>::at(size_type i)
    {
        if (!this->data)
        {
            throw std::logic_error("Cannot access element of uninitialized Array");
        }
        return this->data->at(i);
    }

    /**
     * @brief Bounds-checked element access (const)
     * @param i Index of the element to access
     * @return Const reference to the element at position i
     * @throws std::logic_error if storage is uninitialized
     * @throws std::out_of_range if index is out of bounds
     */
    template<typename _Tp>
    typename Array<_Tp>::const_reference Array<_Tp>::at(size_type i) const
    {
        if (!this->data)
        {
            throw std::logic_error("Cannot access element of uninitialized Array");
        }
        return this->data->at(i);
    }

    /**
     * @brief Access the first element (non-const)
     * @return Reference to the first element
     * @pre The array must be non-empty and initialized
     */
    template<typename _Tp>
    typename Array<_Tp>::reference Array<_Tp>::front() noexcept
    {
        return (*this->data).front();
    }

    /**
     * @brief Access the first element (const)
     * @return Const reference to the first element
     * @pre The array must be non-empty and initialized
     */
    template<typename _Tp>
    typename Array<_Tp>::const_reference Array<_Tp>::front() const noexcept
    {
        return (*this->data).front();
    }

    /**
     * @brief Access the last element (non-const)
     * @return Reference to the last element
     * @pre The array must be non-empty and initialized
     */
    template<typename _Tp>
    typename Array<_Tp>::reference Array<_Tp>::back() noexcept
    {
        return (*this->data).back();
    }

    /**
     * @brief Access the last element (const)
     * @return Const reference to the last element
     * @pre The array must be non-empty and initialized
     */
    template<typename _Tp>
    typename Array<_Tp>::const_reference Array<_Tp>::back() const noexcept
    {
        return (*this->data).back();
    }

    /**
     * @brief Checks if the array is empty
     * @return true if the array is empty or uninitialized, false otherwise
     *
     * An uninitialized Array is considered empty with zero capacity.
     */
    template<typename _Tp>
    bool Array<_Tp>::empty() const noexcept
    {
        return !this->data || this->data->empty();
    }

    /**
     * @brief Returns the number of elements in the array
     * @return Number of elements (0 if uninitialized)
     */
    template<typename _Tp>
    typename Array<_Tp>::size_type Array<_Tp>::size() const noexcept
    {
        return this->data ? this->data->size() : 0;
    }

    /**
     * @brief Returns the maximum possible number of elements
     * @return Maximum size the array can theoretically hold
     */
    template<typename _Tp>
    typename Array<_Tp>::size_type Array<_Tp>::max_size() const noexcept
    {
        return std::vector<_Tp>().max_size();
    }

    /**
     * @brief Reserves storage capacity
     * @param new_cap New capacity to reserve
     *
     * Pre-allocates memory for at least new_cap elements.
     * If storage is uninitialized, it will be created.
     */
    template<typename _Tp>
    void Array<_Tp>::reserve(size_type new_cap)
    {
        if (!this->data)
        {
            this->data.emplace();
        }
        this->data->reserve(new_cap);
    }

    /**
     * @brief Returns the current storage capacity
     * @return Current capacity (0 if uninitialized)
     */
    template<typename _Tp>
    typename Array<_Tp>::size_type Array<_Tp>::capacity() const noexcept
    {
        return this->data ? this->data->capacity() : 0;
    }

    /**
     * @brief Reduces memory usage by freeing unused capacity
     */
    template<typename _Tp>
    void Array<_Tp>::shrink_to_fit()
    {
        if (this->data)
        {
            this->data->shrink_to_fit();
        }
    }

    /**
     * @brief Removes all elements from the array
     *
     * Does not affect storage capacity.
     */
    template<typename _Tp>
    void Array<_Tp>::clear() noexcept
    {
        if (this->data)
        {
            this->data->clear();
        }
    }

    /**
     * @brief Resizes the array to contain count elements
     * @param count New size of the array
     * @param value Value to initialize new elements with
     *
     * If storage is uninitialized, it will be created.
     */
    template<typename _Tp>
    void Array<_Tp>::resize(size_type count, const value_type& value)
    {
        if (!this->data)
        {
            this->data.emplace();
        }
        this->data->resize(count, value);
    }

    /**
     * @brief Equality operator for Array objects
     * @tparam _Tp Element type
     * @param lhs Left-hand side array
     * @param rhs Right-hand side array
     * @return true if arrays have the same size and elements
     *
     * Lexicographical comparison semantics,
     * consistent with std::vector and NumPy array ordering.
     */
    template<typename _Tp>
    auto operator==(const Array<_Tp> &lhs, const Array<_Tp> &rhs) -> bool
    {
        if (lhs.size() != rhs.size())
        {
            return false;
        }
        return std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    /**
     * @brief Inequality operator for Array objects
     * @tparam _Tp Element type
     * @param lhs Left-hand side array
     * @param rhs Right-hand side array
     * @return true if arrays are not equal
     */
    template<typename _Tp>
    auto operator!=(const Array<_Tp> &lhs, const Array<_Tp> &rhs) -> bool
    {
        return !(lhs == rhs);
    }

    /**
     * @brief Less-than operator for Array objects
     * @tparam _Tp Element type
     * @param lhs Left-hand side array
     * @param rhs Right-hand side array
     * @return true if lhs is lexicographically less than rhs
     */
    template<typename _Tp>
    auto operator<(const Array<_Tp> &lhs, const Array<_Tp> &rhs) -> bool
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                            rhs.end());
    }

    /**
     * @brief Less-than-or-equal operator for Array objects
     * @tparam _Tp Element type
     * @param lhs Left-hand side array
     * @param rhs Right-hand side array
     * @return true if lhs is lexicographically less than or equal to rhs
     */
    template<typename _Tp>
    auto operator<=(const Array<_Tp> &lhs, const Array<_Tp> &rhs) -> bool
    {
        return !(rhs < lhs);
    }

    /**
     * @brief Greater-than operator for Array objects
     * @tparam _Tp Element type
     * @param lhs Left-hand side array
     * @param rhs Right-hand side array
     * @return true if lhs is lexicographically greater than rhs
     */
    template<typename _Tp>
    auto operator>(const Array<_Tp> &lhs, const Array<_Tp> &rhs) -> bool
    {
        return rhs < lhs;
    }

    /**
     * @brief Greater-than-or-equal operator for Array objects
     * @tparam _Tp Element type
     * @param lhs Left-hand side array
     * @param rhs Right-hand side array
     * @return true if lhs is lexicographically greater than or equal to rhs
     */
    template<typename _Tp>
    auto operator>=(const Array<_Tp> &lhs, const Array<_Tp> &rhs) -> bool
    {
        return !(lhs < rhs);
    }

} // namespace np

#endif // NP_ARRAY_TPP