#ifndef NDARRAY_H
#define NDARRAY_H

#include <algorithm>
#include <atomic>
#include <tuple>

// Include type_traits for type-related utilities
#include <array>
#include <iostream>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>
#include <cstdint>

#include "__dtype_internal.h"
#include "__ndarray_internal.h"
#include "__ndarray_proxy_internal.h"

#define NUMPY_VERSION 202604L

namespace np {
    namespace matrix {
        enum class Order : std::uint8_t
        {
            C, // Row-major C-style
            F  // Column-major Fortran-style
        };
    }
    
    /**
     * @brief A class representing a multidimensional array (ndarray).
     *
     * This class provides various methods to manipulate and operate on
     * multidimensional arrays, similar to the functionality provided by
     * NumPy in Python.
     */
    template <typename T = double>
    class Ndarray : public np::_Numpy_ndarray_internal<T> {
        #if NUMPY_VERSION <= 202603L
#define MAX_BOUND_NDARRAY_DEFINED
        // The variant alternative of the
        // main Proxy design implementation is
        // limited a 8 dimension.
        using ProxyVariant =
            std::variant<NdarrayProxy<T, 1, 1>, NdarrayProxy<_Tp, 1, 2>,
            NdarrayProxy<T, 1, 3>, NdarrayProxy<_Tp, 1, 4>,
            NdarrayProxy<T, 1, 5>, NdarrayProxy<_Tp, 1, 6>>;
            
        #endif // NUMPY_VERSION
            
      public:
      
        #if defined(MAX_BOUND_NDARRAY_DEFINED) && defined(INLINE_PROXY_DEFINITION)
        // Using the inline proxy implementation
        // instead of std::variant (not recommended).
        // Will remove std::variant implementation in future
        // updates. Inline declaration of class Proxy is slightly
        // faster than include declaration.
        class Proxy {
          private:
            Ndarray<T> &m_array;
            std::vector<size_t> m_indices;
            
          public:
            Proxy( Ndarray<T> &arr, const std::vector<size_t> &idx )
                : m_array( arr ), m_indices( idx ) {
                // void implementation
            }
            
            Proxy( const Proxy &other ) : m_array( other.m_array ),
                m_indices( other.m_indices ) {}
                
            // Assignment operator
            Proxy &operator=( const T &__v ) {
                m_array.set( m_indices, __v );
                return *this;
            }
            
            Proxy &operator=( const Proxy& other ) {
                if ( this != &other ) {
                    // Get the value from the other proxy and assign it
                    T value = other;  // Uses conversion operator
                    m_array.set( m_indices, value );
                }
                
                return *this;
            }
            
            // Conversion operator
            operator T() const {
                return m_array.get( m_indices );
            }
            
            Proxy operator[]( size_t index ) {
                std::vector<size_t> newIndices = m_indices;
                newIndices.push_back( index );
                return Proxy( m_array, newIndices );
            }
            
            const Proxy operator[]( size_t index ) const {
                std::vector<size_t> newIndices = m_indices;
                newIndices.push_back( index );
                return Proxy( m_array, newIndices );
            }
        };
        
        #endif
        
        // Old implementation NdarrayProxy, the new implementation
        // uses two distinct class: Proxy and ConstProxy
        //using Proxy = NdarrayProxy<T, 1, 8>;
        //using ConstProxy = NdarrayProxy<const T, 1, 8>;
        
        inline constexpr Ndarray() = default;
        
        template<typename Container>
        requires std::ranges::range<Container>
        constexpr Ndarray( const std::optional<Container> &container = std::nullopt );
        
        template<typename Container>
        requires std::ranges::range<Container>
        constexpr explicit Ndarray( const Container&container );
        
        /**
         * @brief Constructs an Ndarray object from an initializer list.
         *
         * @param initList An initializer list to initialize the ndarray.
         */
        constexpr Ndarray( std::initializer_list<T> initList );
        
        /**
         * @brief Constructs an Ndarray object.
         *
         * @param shape A tuple representing the shape of the ndarray.
         * @param type The data type of the elements in the ndarray.
         * @param buffer An optional 2D vector representing the buffer of the ndarray.
         * @param offset An optional offset for the ndarray.
         * @param strides An optional tuple representing the strides of the ndarray.
         * @param order The memory layout order of the ndarray.
         */
        
        // To avoid including sys/types.h, for
        // cross-platform fidelity
        using off_t = std::uint64_t;
        
        constexpr Ndarray( const std::vector<int> &shape, np::dtype type,
                           std::optional<std::vector<T>> buffer = std::nullopt,
                           std::optional<off_t> offset = std::nullopt,
                           std::optional<std::vector<size_t>> strides = std::nullopt,
                           np::matrix::Order order = np::matrix::Order::C ) noexcept;
                           
        /**
        * @brief Constructs an Ndarray from a nested initializer list (N-dimensional)
        *
        * This constructor recursively processes nested initializer lists to create
        * an N-dimensional Ndarray. All sub-lists at each dimension must have
        * consistent sizes.
        *
        * @tparam T The data type of elements in the array
        * @param nested_list A recursively nested initializer list structure
        *
        * @throws std::invalid_argument If dimensions are inconsistent or list is empty
        *
        * @example
        * @code
        * // 1D array
        * np::Ndarray<int> a = {1, 2, 3, 4};
        *
        * // 2D array (2x3)
        * np::Ndarray<int> b = {{1, 2, 3}, {4, 5, 6}};
        *
        * // 3D array (2x2x3)
        * np::Ndarray<int> c = {{{1,2,3},{4,5,6}}, {{7,8,9},{10,11,12}}};
        * @endcode
        */
        constexpr Ndarray( std::initializer_list<std::initializer_list<T>>
                           nested_list );
                           
        /**
        * @brief Accesses a subarray or element at the specified index (non-const version)
        *
        * This operator provides subscript access to the Ndarray. When used on a multi-dimensional
        * array, it returns a Proxy object that supports chained subscript operations. For a 1D array,
        * this directly accesses the element. For N-dimensional arrays, you can chain multiple
        * operator[] calls to access deeper dimensions.
        *
        * @param index The index along the first dimension to access
        * @return Proxy<T> A proxy object that represents a view into the array at the specified index,
        *                  allowing further subscript operations or element access
        *
        * @throws std::out_of_range If the index is out of bounds (when bounds checking is enabled)
        *
        * @note The returned Proxy object maintains a reference to the original array and a stack
        *       of indices. The actual element access is deferred until the full index chain is
        *       completed or the proxy is converted to a value.
        *
        * @warning The proxy object is temporary and should not be stored long-term as it holds
        *          a reference to the original array.
        *
        * @example
        * @code
        * np::Ndarray<int> arr = {{1, 2, 3}, {4, 5, 6}};  // 2x3 array
        *
        * // Chain subscript operations
        * arr[0][1] = 10;  // Sets element at row 0, column 1 to 10
        *
        * // Get a subarray (row view)
        * auto row = arr[1];  // Proxy representing the second row
        * int value = row[2]; // Access element at row 1, column 2
        *
        * // 1D array access
        * np::Ndarray<int> vec = {1, 2, 3, 4};
        * vec[2] = 42;  // Direct element access (Proxy converts to reference)
        * @endcode
        */
        auto operator[]( std::size_t index ) -> Proxy<T> {
            IndexStack<> idx;
            idx.push_back( index );
            return Proxy<T>( *this, idx );
        }
        
        /**
         * @brief Accesses a subarray or element at the specified index (const version)
         *
         * This const-qualified operator provides read-only subscript access to the Ndarray.
         * When used on a multi-dimensional array, it returns a ConstProxy object that supports
         * chained subscript operations for read-only access. For a 1D array, this directly
         * accesses the element as a const reference.
         *
         * @param index The index along the first dimension to access
         * @return ConstProxy<T> A const proxy object that represents a read-only view into the
         *                       array at the specified index, allowing further subscript operations
         *
         * @throws std::out_of_range If the index is out of bounds (when bounds checking is enabled)
         *
         * @note The returned ConstProxy object maintains a const reference to the original array,
         *       ensuring that elements cannot be modified through this access path.
         *
         * @warning The proxy object is temporary and should not be stored long-term as it holds
         *          a reference to the original array.
         *
         * @example
         * @code
         * const np::Ndarray<int> arr = {{1, 2, 3}, {4, 5, 6}};  // Const 2x3 array
         *
         * // Read values through chained subscript operations
         * int value = arr[0][2];  // Returns 3
         *
         * // Get a const row view
         * auto row = arr[1];  // ConstProxy representing the second row
         * int col_value = row[1];  // Returns 5
         *
         * // arr[0][1] = 10;  // Error: Cannot modify const array
         *
         * // 1D const array access
         * const np::Ndarray<int> vec = {1, 2, 3, 4};
         * int element = vec[2];  // Returns 3 (as const reference)
         * @endcode
         *
         * @see operator[](std::size_t) for non-const version
         * @see Proxy, ConstProxy for detailed proxy behavior
         */
        auto operator[]( std::size_t index ) const -> ConstProxy<T> {
            IndexStack<> idx;
            idx.push_back( index );
            return ConstProxy<T>( *this, idx );
        }
        
        /**
         * @brief Accesses the element at the specified index.
         *
         * @param sizes The index of the element to access.
         * @return Reference to the element at the specified index.
         */
        // Inside ndarray.hpp, replace the old operator[] with:
        
        //T& operator[](std::size_t sizes);
        //T& operator()(std::vector<int> indexes);
        
        template <std::size_t Size>
        void set( const std::array<std::size_t, Size> &indices, T value );
        
        template <typename Container>
        void set( const Container &indices, T value );
        
        #if defined(CONSTEXPR_GETTER_SETTER)
        // This is used for research purpose
        // Not yet fully studied, breaks a lot because of
        // the system design. Some value need to be given at runtime.
        template <typename Container>
        T get( const Container &indices ) const;
        
        template <std::size_t Size>
        T &get( const std::array<std::size_t, Size> &indices );
        #endif
        
        /**
        * @brief Accesses an element using a compile-time fixed-size array of indices (non-const version)
        *
        * This method provides direct element access using a std::array of indices. It is more efficient
        * than chained subscript operators when you have all indices available at once, as it computes
        * the flat index in a single pass. The number of indices must match the array's dimensionality.
        *
        * @tparam Size The number of indices (must equal ndim)
        * @param indices A std::array containing the indices for each dimension
        * @return T& Reference to the element at the specified position
        *
        * @throws std::out_of_range If any index is out of bounds for its dimension
        * @throws std::invalid_argument If the number of indices does not match the array's dimensionality
        *
        * @note This method performs bounds checking when NP_DEBUG is defined (optional)
        * @warning The returned reference becomes invalid if the array is reallocated or destroyed
        *
        * @example
        * @code
        * np::Ndarray<int> arr = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // 2x2x2 array
        *
        * // Access element at [1][0][1]
        * std::array<size_t, 3> indices = {1, 0, 1};
        * int& value = arr.get(indices);
        * value = 42;  // Modify the element
        *
        * // More efficient than arr[1][0][1] when indices are already in an array
        * @endcode
        *
        * @see get(const std::array<std::size_t, Size>&) const for const version
        * @see get(const Container&) const for runtime-sized index containers
        */
        template <std::size_t Size>
        auto get( const std::array<std::size_t, Size> &indices ) -> T &;
        
        /**
         * @brief Accesses an element using a compile-time fixed-size array of indices (const version)
         *
         * This const-qualified method provides read-only element access using a std::array of indices.
         * It computes the flat index in a single pass and returns a const reference to the element.
         * The number of indices must match the array's dimensionality.
         *
         * @tparam Size The number of indices (must equal ndim)
         * @param indices A std::array containing the indices for each dimension
         * @return const T& Const reference to the element at the specified position
         *
         * @throws std::out_of_range If any index is out of bounds for its dimension
         * @throws std::invalid_argument If the number of indices does not match the array's dimensionality
         *
         * @note This method performs bounds checking when NP_DEBUG is defined (optional)
         * @note Prefer this version when you don't need to modify the element
         *
         * @example
         * @code
         * const np::Ndarray<int> arr = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
         *
         * // Read element at [0][1][1]
         * std::array<size_t, 3> indices = {0, 1, 1};
         * int value = arr.get(indices);  // Returns 4
         *
         * // Useful when indices are computed or passed from elsewhere
         * @endcode
         *
         * @see get(const std::array<std::size_t, Size>&) for non-const version
         * @see get(const Container&) const for runtime-sized index containers
         */
        template <std::size_t Size>
        auto get( const std::array<std::size_t, Size> &indices ) const -> const T &;
        
        /**
         * @brief Accesses an element using a runtime-sized container of indices (returns by value)
         *
         * This method provides element access using any container that supports .size() and []
         * operators (e.g., std::vector, std::initializer_list). Unlike the std::array version,
         * this method returns the element by value rather than by reference because the container
         * type and size are determined at runtime, making reference safety difficult to guarantee.
         *
         * @tparam Container A container type that supports size() and operator[] (e.g., std::vector,
         *                   std::initializer_list, std::array with runtime size)
         * @param indices A container holding the indices for each dimension
         * @return T The value of the element at the specified position (returned by copy)
         *
         * @throws std::out_of_range If any index is out of bounds for its dimension
         * @throws std::invalid_argument If the number of indices does not match the array's dimensionality
         *
         * @note Returns by value to avoid dangling references when the index container is temporary
         * @note For performance-critical code, prefer the std::array version when the size is known
         *       at compile time
         * @note This method is particularly useful when indices come from user input or are computed
         *       at runtime
         *
         * @example
         * @code
         * np::Ndarray<int> arr = {{1, 2, 3}, {4, 5, 6}};  // 2x3 array
         *
         * // Using std::vector (runtime-sized)
         * std::vector<size_t> indices = {1, 2};
         * int value = arr.get(indices);  // Returns 6
         *
         * // Using initializer_list (convenient for inline calls)
         * int val = arr.get({0, 1});  // Returns 2
         *
         * // Using std::array (will be treated as Container, returns by value)
         * std::array<size_t, 2> idx = {0, 0};
         * int elem = arr.get(idx);  // Returns 1 (copy, not reference)
         *
         * // Useful when reading from file or user input
         * std::vector<size_t> user_indices = read_indices_from_user();
         * int result = arr.get(user_indices);
         * @endcode
         *
         * @note If you need a reference and have a compile-time known size, use the std::array overload
         * @warning This method returns a copy, which may be expensive for large element types
         *
         * @see get(const std::array<std::size_t, Size>&) for compile-time size and reference return
         * @see operator[] for chained subscript syntax
         */
        template <typename Container>
        auto get( const Container& indices ) const -> decltype( T {} );
        
        /**
         * @brief The data type of the elements in the ndarray.
         */
        np::dtype type;
        
        /**
         * @brief A tuple representing the shape of the ndarray.
         */
        std::vector<int> shape;
        
        /**
         * @brief An optional 2D vector representing the buffer of the ndarray.
         */
        std::optional<std::vector<T>> buffer;
        
        /**
         * @brief An optional offset for the ndarray.
         */
        std::optional<off_t> offset;
        
        /**
         * @brief An optional tuple representing the strides of the ndarray.
         */
        std::vector<size_t> strides;
        
        /**
         * @brief The memory layout order of the ndarray.
         */
        np::matrix::Order order;
        
        /**
         * @brief Checks if all elements along the specified axis evaluate to true.
         *
         * @param axis The axis along which to perform the check.
         * @param out Optional output array to store the result.
         * @param keepdims If true, retains reduced dimensions with size one.
         * @param where Optional condition array to apply the check.
         * @return True if all elements evaluate to true, otherwise false.
         */
        
        auto all( size_t axis, std::optional<Ndarray> out, bool keepdims,
        std::vector<bool> where ) const -> decltype( bool {} );
        
        friend auto operator<<( std::ostream &output,
                                Ndarray &array ) -> std::ostream & {
            output << "ndarray(";
            array._print_recursive( 0, 0, output );
            output << ")" << "\n";
            return output;
        }
        
        /**
         * @brief Checks if any element along the specified axis evaluates to
         * true.
         *
         * @param axis The axis along which to perform the check.
         * @return True if any element evaluates to true, otherwise false.
         */
        [[nodiscard( "Return value not used" )]]
        auto any( size_t axis ) const -> decltype( bool {} );
        
        /**
         * @brief Returns the indices of the maximum values along the specified axis.
         *
         * @param axis The axis along which to find the maximum values.
         * @return Indices of the maximum values.
         */
        [[nodiscard]] auto argmax( size_t axis ) const -> decltype( bool {} );
        
        /**
         * @brief Returns the indices of the minimum values along the specified axis.
         *
         * @return Indices of the minimum values.
         */
        auto argmin() const -> bool;
        
        /**
         * @brief Performs an indirect partition along the specified axis.
         *
         * @return Indices that would partition the array.
         */
        int argpartition() const;
        
        /**
         * @brief Returns the indices that would sort the array.
         *
         * @return Indices that would sort the array.
         */
        int argsort() const;
        
        /**
         * @brief Casts the array to a specified type.
         */
        void astype();
        
        /**
         * @brief Swaps the bytes of the array elements.
         */
        void byteswap();
        
        /**
         * @brief Chooses elements from the array based on given indices.
         */
        void choose();
        
        /**
         * @brief Clips the values of the array to a specified range.
         */
        void clip();
        
        /**
         * @brief Compresses the array along the specified axis.
         */
        void compress();
        
        /**
         * @brief Returns the complex conjugate of the array elements.
         */
        void conj();
        
        /**
         * @brief Returns the complex conjugate of the array elements.
         */
        void conjugate();
        
        /**
         * @brief Returns a copy of the array.
         */
        void copy();
        
        /**
         * @brief Returns the cumulative product of the array elements.
         */
        void cumprod();
        
        /**
         * @brief Returns the cumulative sum of the array elements.
         */
        void cumsum();
        
        /**
         * @brief Returns the diagonal elements of the array.
         */
        void diagonal();
        
        /**
         * @brief Dumps the array to a file.
         */
        void dump();
        
        /**
         * @brief Returns a string representation of the array.
         */
        void dumps();
        
        /**
         * @brief Fills the array with a specified value.
         */
        void fill();
        
        /**
         * @brief Flattens the array.
         */
        void flatten();
        
        /**
         * @brief Returns a field of the array.
         */
        void getfield();
        
        /**
         * @brief Returns a copy of the array element as a standard Python scalar.
         */
        void item();
        
        /**
         * @brief Returns the maximum value of the array elements.
         */
        void _max();
        
        /**
         * @brief Returns the mean value of the array elements.
         */
        void mean();
        
        /**
         * @brief Returns the minimum value of the array elements.
         */
        void _min();
        
        /**
         * @brief Returns the indices of the non-zero elements.
         */
        void nonzero();
        
        /**
         * @brief Partitions the array along the specified axis.
         */
        void partition();
        
        /**
         * @brief Returns the product of the array elements.
         */
        void prod();
        
        /**
         * @brief Puts values into the array at specified indices.
         */
        void put();
        
        /**
         * @brief Returns a flattened array.
         */
        void ravel();
        
        /**
         * @brief Repeats the elements of the array.
         */
        void repeat();
        
        /**
         * @brief Reshapes the array to a specified shape.
         */
        void reshape();
        
        /**
         * @brief Resizes the array to a specified shape.
         */
        void resize();
        
        /**
         * @brief Rounds the array elements to the specified number of decimals.
         */
        void round();
        
        /**
         * @brief Finds indices where elements should be inserted to maintain order.
         */
        void searchsorted();
        
        /**
         * @brief Sets a field of the array.
         */
        void setfield();
        
        /**
         * @brief Sets the flags of the array.
         */
        void setflags();
        
        /**
         * @brief Sorts the array along the specified axis.
         */
        void sort();
        
        /**
         * @brief Removes single-dimensional entries from the shape of the array.
         */
        void squeeze();
        
        /**
         * @brief Returns the standard deviation of the array elements.
         */
        void std();
        
        /**
         * @brief Returns the sum of the array elements.
         */
        void sum();
        
        /**
         * @brief Swaps the axes of the array.
         */
        void swapaxes();
        
        /**
         * @brief Takes elements from the array along the specified axis.
         */
        void take();
        
        /**
         * @brief Returns the array as a byte string.
         */
        void tobytes();
        
        /**
         * @brief Writes the array to a file.
         */
        void tofile();
        
        /**
         * @brief Returns the array as a list.
         */
        void tolist();
        
        /**
         * @brief Returns the array as a string.
         */
        void tostring();
        
        /**
         * @brief Returns the sum along diagonals of the array.
         */
        void trace();
        
        /**
         * @brief Transposes the array.
         */
        void transpose();
        
        /**
         * @brief Returns the variance of the array elements.
         */
        void var();
        
        /**
         * @brief Returns a view of the array with the same data.
         */
        void view();
        
      private:
        /**
        * @brief Recursively constructs N-dimensional array from nested initializer lists
        *
        * @tparam U The current level's initializer list type
        * @param list The initializer list at current depth
        * @param depth Current recursion depth (0-based)
        */
        template<typename U>
        void _construct_from_nested( std::initializer_list<U> list,
                                     std::size_t depth = 0 );
                                     
        /**
         * @brief Computes the strides for the ndarray.
         *
         * @tparam Tuple The type of the tuple.
         * @tparam Is The indices of the tuple elements.
         * @param std::index_sequence<Is...> A sequence of indices.
         * @return The computed strides as a tuple.
         */
        
        [[nodiscard]] auto _compute_strides() const -> decltype(
        std::vector<size_t> {} );
        
        /**
         * @brief Recursively prints the elements of the ndarray.
         *
         * @param dim The current dimension being printed.
         * @param offset The offset to the current element in the buffer.
         */
        void _print_recursive( std::size_t dim, std::size_t offset,
                               std::ostream &output ) const;
                               
        /**
         * @brief Computes the flat index in a multi-dimensional array given the
         * indices for each dimension.
         *
         * @tparam Size The number of dimensions in the array.
         * @param indices An array containing the indices for each dimension.
         * @return std::size_t The flat index corresponding to the provided
         * multi-dimensional indices.
         */
        template <std::size_t Size>
        auto
        _get_flat_index( const std::array<std::size_t, Size> &indices ) const ->
        decltype(
        std::size_t {} );
        
        #ifdef MAX_BOUND_NDARRAY_DEFINED
        // Helper to create proxy with correct dimension count
        template <bool IsConst> auto _M_make_proxy( std::size_t index ) const {
            const std::size_t ndims = shape.size();
            
            // Dispatch based on actual number of dimensions
            switch ( ndims ) {
                case 1:
                    return _M_make_proxy_helper<IsConst, 1>( index );
                    
                case 2:
                    return _make_proxy_helper<IsConst, 2>( index );
                    
                case 3:
                    return _make_proxy_helper<IsConst, 3>( index );
                    
                case 4:
                    return _make_proxy_helper<IsConst, 4>( index );
                    
                case 5:
                    return _make_proxy_helper<IsConst, 5>( index );
                    
                case 6:
                    return _make_proxy_helper<IsConst, 6>( index );
                    
                default:
                    return _make_proxy_helper<IsConst, 4>( index ); // fallback
            }
        }
        
        template <bool IsConst, std::size_t N>
        auto _make_proxy_helper( std::size_t index ) const {
            std::array<std::size_t, N> indices{};
            indices[0] = index;
            
            if constexpr( IsConst ) {
                return NdarrayProxy<const T, 1, N>( *this, indices );
            }
            
            else {
                return NdarrayProxy<T, 1, N>( const_cast<Ndarray &>( *this ), indices );
            }
        }
        
        #endif // MAX_BOUND_PROXY_DEFINED
        template <typename U, std::size_t D, std::size_t M> friend class NdarrayProxy;
    };
} // namespace np

#include "__ndarray_methods.tpp"
#include "__ndarray_representation.tpp"
#include "ndarray.tpp"

#endif
