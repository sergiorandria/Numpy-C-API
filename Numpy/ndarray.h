#ifndef NDARRAY_H
#define NDARRAY_H

#include <tuple>
#include <string>
#include <type_traits>
#include <string_view>

#include "__ndarray_internal.h"
#include "dtype.h"

namespace np {
    namespace matrix {
        enum class Order {
            C,      // Row-major C-style         
            F       // Column-major Fortran-style
        }
    };

    template<typename _Tp>
    class Ndarray: public np::_Numpy_ndarray_internal<_Tp>
    {    
        Ndarray() = default;
        Ndarray(std::tuple<int> shape, 
                    np::dtype type, 
                    std::optional<std::vector<std::vector<_Tp>>> buffer,
                    std::optional<off_t> offset,
                    std::optional<std::tuple<int>> strides,
                    np::matrix::Order order);

        /**
         * @brief Check if all elements along the given axis evaluate to true.
         * 
         * @param axis Axis along which to perform the check.
         * @return true if all elements are true, false otherwise.
         */
        bool all(size_t axis, std::optional<Ndarray> out, bool keepdims, std::vector<bool> where) const;

        /**
         * @brief Check if any element along the given axis evaluates to true.
         * 
         * @param axis Axis along which to perform the check.
         * @return true if any element is true, false otherwise.
         */
        
        bool any(size_t axis) const;

        /**
         * @brief Return the indices of the maximum values along the given axis.
         * 
         * @param axis Axis along which to find the maximum values.
         * @return Indices of the maximum values.
         */
        bool argmax(size_t axis) const;

        /**
         * @brief Return the indices of the minimum values.
         * 
         * @return Indices of the minimum values.
         */
        bool argmin() const;

        /**
         * @brief Perform an indirect partition along the given axis.
         * 
         * @return Partitioned indices.
         */
        int argpartition() const;

        /**
         * @brief Return the indices that would sort the array.
         * 
         * @return Sorted indices.
         */
        int argsort() const;

        /**
         * @brief Cast the array to a specified type.
         */
        void astype();

        /**
         * @brief Swap the bytes of the array elements.
         */
        void byteswap();

        /**
         * @brief Use an index array to choose elements from the array.
         */
        void choose();

        /**
         * @brief Clip (limit) the values in the array.
         */
        void clip();

        /**
         * @brief Return selected slices of the array along the given axis.
         */
        void compress();

        /**
         * @brief Return the complex conjugate of the array elements.
         */
        void conj();

        /**
         * @brief Return the complex conjugate of the array elements.
         */
        void conjugate();

        /**
         * @brief Return a copy of the array.
         */
        void copy();

        /**
         * @brief Return the cumulative product of the array elements.
         */
        void cumprod();

        /**
         * @brief Return the cumulative sum of the array elements.
         */
        void cumsum();

        /**
         * @brief Return the diagonal elements of the array.
         */
        void diagonal();

        /**
         * @brief Dump the array to a file in binary format.
         */
        void dump();

        /**
         * @brief Return the array as a string in binary format.
         */
        void dumps();

        /**
         * @brief Fill the array with a scalar value.
         */
        void fill();

        /**
         * @brief Return a flattened copy of the array.
         */
        void flatten();

        /**
         * @brief Return a field of the array as a specified type.
         */
        void getfield();

        /**
         * @brief Copy an element of the array to a standard Python scalar.
         */
        void item();

        /**
         * @brief Return the maximum value of the array.
         */
        void _max();

        /**
         * @brief Return the mean value of the array elements.
         */
        void mean();

        /**
         * @brief Return the minimum value of the array.
         */
        void _min();

        /**
         * @brief Return the indices of the non-zero elements.
         */
        void nonzero();

        /**
         * @brief Perform an indirect partition along the given axis.
         */
        void partition();

        /**
         * @brief Return the product of the array elements.
         */
        void prod();

        /**
         * @brief Set array elements using indices.
         */
        void put();

        /**
         * @brief Return a flattened array.
         */
        void ravel();

        /**
         * @brief Repeat elements of the array.
         */
        void repeat();

        /**
         * @brief Give a new shape to the array without changing its data.
         */
        void reshape();

        /**
         * @brief Change the shape and size of the array in-place.
         */
        void resize();

        /**
         * @brief Return the array with each element rounded to the given number of decimals.
         */
        void round();

        /**
         * @brief Find indices where elements should be inserted to maintain order.
         */
        void searchsorted();

        /**
         * @brief Set a field of the array to a specified type.
         */
        void setfield();

        /**
         * @brief Set array flags.
         */
        void setflags();

        /**
         * @brief Sort the array along the given axis.
         */
        void sort();

        /**
         * @brief Remove single-dimensional entries from the shape of the array.
         */
        void squeeze();

        /**
         * @brief Return the standard deviation of the array elements.
         */
        void std();

        /**
         * @brief Return the sum of the array elements.
         */
        void sum();

        /**
         * @brief Interchange two axes of the array.
         */
        void swapaxes();

        /**
         * @brief Return an array formed from the elements at the given indices.
         */
        void take();

        /**
         * @brief Return the array as a bytes object.
         */
        void tobytes();

        /**
         * @brief Write the array to a file.
         */
        void tofile();

        /**
         * @brief Return the array as a (possibly nested) list.
         */
        void tolist();

        /**
         * @brief Return the array as a string.
         */
        void tostring();

        /**
         * @brief Return the sum along diagonals of the array.
         */
        void trace();

        /**
         * @brief Return a transposed view of the array.
         */
        void transpose();

        /**
         * @brief Return the variance of the array elements.
         */
        void var();

        /**
         * @brief Return a view of the array with the same data.
         */
        void view();
        private:
            np::dtype type;
            std::tuple<int> shape;    
            std::optional<std::vector<std::vector<_Tp>>> buffer;
            std::optional<off_t> offset;
            std::optional<std::tuple<int>> strides;
            np::matrix::Order order;
    };
}

#include "ndarray.tpp"

#endif 