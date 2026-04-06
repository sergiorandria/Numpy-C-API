/**
 * @file Array.h
 * @brief Header file for the np::Array class template.
 *
 * This file contains the definition of the np::Array class template, which is a part of the Numpy-like library.
 * The Array class provides a container for numerical data, supporting various types and initialization methods.
 * It inherits from np::_Numpy_ndarray_internal to leverage internal ndarray functionalities.
 *
 * @note This file is part of the Numpy project.
 */

#ifndef NP_ARRAY_H
#define NP_ARRAY_H

#include <initializer_list>
#include <optional>
#include <ranges>
#include <type_traits>
#include <vector>

#include "dtype.h"
#include "__ndarray_internal.h"

namespace np {

/**
 * @brief A NumPy-style multidimensional array container
 * 
 * @tparam _Tp Element type. Must be a numeric type suitable for array operations.
 * 
 * The Array class provides a flexible container for numerical data with support
 * for various initialization methods and NumPy-like operations. It supports
 * initialization from ranges, initializer lists, and provides efficient move
 * semantics.
 * 
 * @par Thread Safety
 * Distinct objects: Safe
 * Shared objects: Unsafe
 * 
 * @par Exception Safety
 * All member functions provide at least the strong exception guarantee unless
 * otherwise specified.
 * 
 * @par Example
 * @code
 * // Construction from initializer list
 * np::Array<int> arr1 = {1, 2, 3, 4, 5};
 * 
 * // Construction from range
 * std::vector<double> vec = {1.0, 2.0, 3.0};
 * np::Array<double> arr2(vec);
 * 
 * // Element access
 * int value = arr1[0];
 * arr1.at(2) = 10; // Bounds-checked access
 * @endcode
 * 
 * @see _Numpy_ndarray_internal
 */
template<typename _Tp>
class Array : public _Numpy_ndarray_internal<_Tp> {
public:
    // Member types (following STL conventions)
    using value_type             = _Tp;
    using reference              = _Tp&;
    using const_reference        = const _Tp&;
    using pointer                = _Tp*;
    using const_pointer          = const _Tp*;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    
    /**
     * @brief Default constructor
     * 
     * Constructs an empty array with no elements.
     * 
     * @par Complexity
     * Constant
     * 
     * @par Exception Safety
     * No-throw guarantee
     */
    Array() noexcept = default;
    
    /**
     * @brief Constructs an array from a range
     * 
     * Initializes the array with elements from the provided range-like container.
     * The container must satisfy the std::ranges::range concept.
     * 
     * @tparam Container Type satisfying std::ranges::range
     * @param container Source container to copy elements from
     * @param type Optional data type specification (default: np::int32)
     * 
     * @par Complexity
     * Linear in container.size()
     * 
     * @par Exception Safety
     * Strong guarantee. If an exception is thrown, this function has no effect.
     * 
     * @throws std::bad_alloc if memory allocation fails
     * 
     * @par Example
     * @code
     * std::vector<int> vec = {1, 2, 3};
     * np::Array<int> arr(vec);
     * @endcode
     */
    template<typename Container>
        requires std::ranges::range<Container>
    explicit Array(const Container& container, 
                   std::optional<dtype> type = dtype::int32);
    
    /**
     * @brief Constructs an array from an initializer list
     * 
     * Enables brace-initialization syntax for convenient array creation.
     * 
     * @param init_list Initializer list containing elements
     * @param type Optional data type specification (default: np::int32)
     * 
     * @par Complexity
     * Linear in init_list.size()
     * 
     * @par Exception Safety
     * Strong guarantee
     * 
     * @throws std::bad_alloc if memory allocation fails
     * 
     * @par Example
     * @code
     * np::Array<double> arr = {1.0, 2.0, 3.0, 4.0};
     * @endcode
     */
    Array(std::initializer_list<_Tp> init_list, 
          std::optional<dtype> type = dtype::int32);
    
    /**
     * @brief Copy constructor
     * 
     * Constructs an array as a copy of another array.
     * 
     * @param other Array to copy from
     * 
     * @par Complexity
     * Linear in other.size()
     * 
     * @par Exception Safety
     * Strong guarantee
     */
    Array(const Array& other) = default;
    
    /**
     * @brief Move constructor
     * 
     * Constructs an array by moving from another array, leaving the source
     * in a valid but unspecified state.
     * 
     * @param other Array to move from
     * 
     * @par Complexity
     * Constant
     * 
     * @par Exception Safety
     * No-throw guarantee
     */
    Array(Array&& other) noexcept = default;
    
    /**
     * @brief Destructor
     * 
     * Destroys the array and all contained elements.
     * 
     * @par Complexity
     * Linear in size()
     * 
     * @par Exception Safety
     * No-throw guarantee
     */
    ~Array() = default;
    
    /**
     * @brief Copy assignment operator
     * 
     * Replaces the contents with a copy of the contents of another array.
     * 
     * @param other Array to copy from
     * @return Reference to *this
     * 
     * @par Complexity
     * Linear in other.size()
     * 
     * @par Exception Safety
     * Strong guarantee
     */
    Array& operator=(const Array& other) = default;
    
    /**
     * @brief Move assignment operator
     * 
     * Replaces the contents with those of another array using move semantics.
     * After the operation, other is in a valid but unspecified state.
     * 
     * @param other Array to move from (rvalue reference)
     * @return Reference to *this
     * 
     * @par Complexity
     * Constant
     * 
     * @par Exception Safety
     * No-throw guarantee
     * 
     * @par Example
     * @code
     * np::Array<int> arr1 = {1, 2, 3};
     * np::Array<int> arr2 = std::move(arr1); // arr1 is now unspecified
     * @endcode
     */
    Array& operator=(Array&& other) noexcept;
    
    /**
     * @brief Initializer list assignment
     * 
     * Replaces the contents with elements from an initializer list.
     * 
     * @param init_list Initializer list to assign from
     * @return Reference to *this
     * 
     * @par Complexity
     * Linear in init_list.size()
     * 
     * @par Exception Safety
     * Strong guarantee
     */
    Array& operator=(std::initializer_list<_Tp> init_list);
    
    /**
     * @brief Accesses element at specified index (unchecked)
     * 
     * Returns a reference to the element at position i. No bounds checking
     * is performed. Accessing out-of-bounds elements is undefined behavior.
     * 
     * @param i Index of element to access
     * @return Reference to the element
     * 
     * @par Complexity
     * Constant
     * 
     * @par Exception Safety
     * No-throw guarantee if i is within bounds
     * 
     * @warning Undefined behavior if i < 0 or i >= size()
     * 
     * @note For bounds-checked access, use at()
     * 
     * @see at()
     */
    reference operator[](size_type i) noexcept;
    
    /**
     * @brief Accesses element at specified index (unchecked, const)
     * 
     * @param i Index of element to access
     * @return Const reference to the element
     * 
     * @par Complexity
     * Constant
     * 
     * @par Exception Safety
     * No-throw guarantee if i is within bounds
     */
    const_reference operator[](size_type i) const noexcept;
    
    /**
     * @brief Accesses element with bounds checking
     * 
     * Returns a reference to the element at position i with bounds checking.
     * 
     * @param i Index of element to access
     * @return Reference to the element
     * 
     * @throws std::out_of_range if i >= size()
     * 
     * @par Complexity
     * Constant
     * 
     * @par Exception Safety
     * Strong guarantee
     * 
     * @par Example
     * @code
     * np::Array<int> arr = {1, 2, 3};
     * try {
     *     int value = arr.at(5); // Throws
     * } catch (const std::out_of_range& e) {
     *     // Handle error
     * }
     * @endcode
     */
    reference at(size_type i);
    
    /**
     * @brief Accesses element with bounds checking (const)
     * @param i Index of element to access
     * @return Const reference to the element
     * 
     * @throws std::out_of_range if i >= size()
     * 
     * @par Complexity
     * Constant
     */
    const_reference at(size_type i) const;
    
    /**
     * @brief Accesses the first element
     * @return Reference to the first element
     * 
     * @par Complexity
     * Constant
     * 
     * @warning Calling front() on an empty array is undefined behavior
     */
    reference front() noexcept;
    const_reference front() const noexcept;
    
    /**
     * @brief Accesses the last element
     * @return Reference to the last element
     * 
     * @par Complexity
     * Constant
     * 
     * @warning Calling back() on an empty array is undefined behavior
     */
    reference back() noexcept;
    const_reference back() const noexcept;
    
    auto begin() noexcept;
    auto begin() const noexcept;
    auto cbegin() const noexcept;
    
    auto end() noexcept;
    auto end() const noexcept;
    auto cend() const noexcept;
    
    auto rbegin() noexcept;
    auto rbegin() const noexcept;
    auto crbegin() const noexcept;
    
    auto rend() noexcept;
    auto rend() const noexcept;
    auto crend() const noexcept;
    
    /**
     * @brief Checks whether the array is empty
     * 
     * @return true if size() == 0, false otherwise
     * 
     * @par Complexity
     * Constant
     */
    [[nodiscard]] bool empty() const noexcept;
    
    /**
     * @brief Returns the number of elements
     * 
     * @return Number of elements in the array
     * 
     * @par Complexity
     * Constant
     */
    [[nodiscard]] size_type size() const noexcept;
    
    /**
     * @brief Returns the maximum possible number of elements
     * 
     * @return Maximum number of elements
     * 
     * @par Complexity
     * Constant
     */
    [[nodiscard]] size_type max_size() const noexcept;
    
    /**
     * @brief Reserves storage
     * 
     * Increases the capacity to at least new_cap.
     * 
     * @param new_cap New capacity
     * 
     * @throws std::length_error if new_cap > max_size()
     * @throws std::bad_alloc if allocation fails
     */
    void reserve(size_type new_cap);
    
    /**
     * @brief Returns the number of elements that can be held in currently allocated storage
     * 
     * @return Capacity of the currently allocated storage
     */
    [[nodiscard]] size_type capacity() const noexcept;
    
    /**
     * @brief Reduces memory usage by freeing unused memory
     * Requests the removal of unused capacity.
     */
    void shrink_to_fit();
    
    /**
     * @brief Clears the contents
     * Removes all elements from the array. Leaves capacity() unchanged.
     * 
     * @par Complexity
     * Linear in size()
     * 
     * @par Exception Safety
     * No-throw guarantee
     */
    void clear() noexcept;
    
    /**
     * @brief Resizes the array
     * Resizes the array to contain count elements.
     * 
     * @param count New size of the array
     * @param value Value to initialize new elements with
     * 
     * @par Complexity
     * Linear in count
     */
    void resize(size_type count, const value_type& value = value_type{});
    
    /**
     * @brief Swaps the contents
     * Exchanges the contents of the array with those of other.
     * 
     * @param other Array to exchange contents with
     * 
     * @par Complexity
     * Constant
     * 
     * @par Exception Safety
     * No-throw guarantee
     */
    void swap(Array& other) noexcept;
    
    // Friend swap for ADL (Argument-Dependent Lookup)
    friend void swap(Array& lhs, Array& rhs) noexcept {
        lhs.swap(rhs);
    }
};

/// Trait to check if a type is an Array
template<typename T>
struct is_array : std::false_type {};

template<typename _Tp>
struct is_array<Array<_Tp>> : std::true_type {};

template<typename T>
inline constexpr bool is_array_v = is_array<T>::value;

/// Concept for numeric array element types
template<typename T>
concept NumericArrayElement = std::is_arithmetic_v<T>;

/**
 * @brief Lexicographically compares the values in the array
 * 
 * @par Complexity
 * Linear in size()
 */
template<typename _Tp>
bool operator==(const Array<_Tp>& lhs, const Array<_Tp>& rhs);

template<typename _Tp>
bool operator!=(const Array<_Tp>& lhs, const Array<_Tp>& rhs);

template<typename _Tp>
bool operator<(const Array<_Tp>& lhs, const Array<_Tp>& rhs);

template<typename _Tp>
bool operator<=(const Array<_Tp>& lhs, const Array<_Tp>& rhs);

template<typename _Tp>
bool operator>(const Array<_Tp>& lhs, const Array<_Tp>& rhs);

template<typename _Tp>
bool operator>=(const Array<_Tp>& lhs, const Array<_Tp>& rhs);

// C++20 three-way comparison
#ifdef __cpp_lib_three_way_comparison
template<typename _Tp>
auto operator<=>(const Array<_Tp>& lhs, const Array<_Tp>& rhs);
#endif

} // namespace np

// Include template implementation
#include "array.tpp"
#include "_array_creation_routine.tpp"

#endif // NP_ARRAY_H
