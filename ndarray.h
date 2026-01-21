#ifndef NDARRAY_H
#define NDARRAY_H

#include <tuple>
#include <string>

// Include type_traits for type-related utilities
#include <type_traits>
#include <string_view>
#include <vector>
#include <optional>

#include "__ndarray_internal.h"
//#include "__ndarray_proxy_internal.h"

#include "dtype.h"
#include "array.h"

namespace np {
    namespace matrix {
        enum class Order {
            C,      // Row-major C-style         
            F       // Column-major Fortran-style
        };
    }

    template<typename _Tp = double>
    /**
     * @brief A class representing a multidimensional array (ndarray).
     * 
     * This class provides various methods to manipulate and operate on 
     * multidimensional arrays, similar to the functionality provided by 
     * NumPy in Python.
     */
    class Ndarray: public np::_Numpy_ndarray_internal<_Tp>
    {
    public:
        Ndarray() = default;
        /**
         * @brief Constructs an Ndarray object from an initializer list.
         * 
         * @param il An initializer list to initialize the ndarray.
         */
        Ndarray(std::initializer_list<_Tp> il);

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
        Ndarray(std::vector<int> shape, 
                np::dtype type, 
                std::optional<std::vector<_Tp>> buffer = std::nullopt,
                std::optional<off_t> offset = std::nullopt,
                std::optional<std::tuple<int>> strides = std::nullopt,
                np::matrix::Order order = np::matrix::Order::C) noexcept;
        
        
        /**
         * @brief Accesses the element at the specified index.
         * 
         * @param sizes The index of the element to access.
         * @return Reference to the element at the specified index.
         */

        //_Tp& operator[](std::size_t sizes);
        //_Tp& operator()(std::vector<int> indexes);
        
        template <std::size_t Size>
        void set(const std::array<std::size_t, Size>& indices, _Tp value);


        template <std::size_t Size>
        _Tp get(const std::array<std::size_t, Size>& indices) const;

        /*np::_Numpy_ndarray_proxy operator[](std::size_t index) {
            return np::_Numpy_ndarray_proxy(*this, index * strides[0], 1);
        }*/
        
        
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
        std::optional<std::vector<_Tp>> buffer;

        /**
         * @brief An optional offset for the ndarray.
         */
        std::optional<off_t> offset;

        /**
         * @brief An optional tuple representing the strides of the ndarray.
         */
        std::vector<_Tp> strides;

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

        bool all(size_t axis, std::optional<Ndarray> out, bool keepdims, std::vector<bool> where) const;

        friend std::ostream& operator<<<>(std::ostream& output, Ndarray<_Tp> array);

        

        /**
         * @brief Checks if any element along the specified axis evaluates to true.
         * 
         * @param axis The axis along which to perform the check.
         * @return True if any element evaluates to true, otherwise false.
         */
        bool any(size_t axis) const;

        /**
         * @brief Returns the indices of the maximum values along the specified axis.
         * 
         * @param axis The axis along which to find the maximum values.
         * @return Indices of the maximum values.
         */
        bool argmax(size_t axis) const;

        /**
         * @brief Returns the indices of the minimum values along the specified axis.
         * 
         * @return Indices of the minimum values.
         */
        bool argmin() const;

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
         * @brief Computes the strides for the ndarray.
         * 
         * @tparam Tuple The type of the tuple.
         * @tparam Is The indices of the tuple elements.
         * @param std::index_sequence<Is...> A sequence of indices.
         * @return The computed strides as a tuple.
         */
  
        std::vector<int> _compute_strides() const;

        /**
         * @brief Recursively prints the elements of the ndarray.
         * 
         * @param dim The current dimension being printed.
         * @param offset The offset to the current element in the buffer.
         */
        void _print_recursive(std::size_t dim, std::size_t offset, std::ostream& output) const;
        
        /**
         * @brief Computes the flat index in a multi-dimensional array given the indices for each dimension.
         * 
         * @tparam Size The number of dimensions in the array.
         * @param indices An array containing the indices for each dimension.
         * @return std::size_t The flat index corresponding to the provided multi-dimensional indices.
         */
        template <std::size_t Size>
        std::size_t _get_flat_index(const std::array<std::size_t, Size>& indices) const;
    };
}

#include "ndarray.tpp"
#include "__ndarray_methods.tpp"
#include "__ndarray_representation.tpp"

#endif