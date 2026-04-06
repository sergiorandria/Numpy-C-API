#ifndef _NDARRAY_INTERNAL_H
#define _NDARRAY_INTERNAL_H

#include <map>
#include <vector>
#include <string_view>

#include "dtype.hpp"
#include "macro.h"

namespace np
{
    /**
     * @class _Numpy_ndarray_internal
     * @brief Internal representation of a Numpy ndarray.
     *
     * This class provides an internal structure for handling Numpy ndarray objects.
     *
     * @attribute T
     * Transpose of the array.
     *
     * @attribute data
     * Optional 2D vector containing the data of the array.
     *
     * @attribute dtype
     * Data type of the array elements.
     *
     * @attribute flags
     * Dictionary of boolean flags.
     *
     * @attribute imag
     * Imaginary part of the array elements.
     *
     * @attribute real
     * Real part of the array elements.
     *
     * @attribute size
     * Number of elements in the array.
     *
     * @attribute itemsize
     * Length of one array element in bytes.
     *
     * @attribute nbytes
     * Total number of bytes consumed by the elements of the array.
     *
     * @attribute ndim
     * Number of array dimensions.
     *
     * @attribute strides
     * Tuple of bytes to step in each dimension when traversing an array.
     */

    template <typename T = double>
    class _Numpy_ndarray_internal
    {
      protected:
        /**
        * @note The following attributes are not yet implemented:
        * - .ctypes: ctypes object
        * - .base: ndarray
        * - .flat: numpy::flatiter objects
        */

        LIBNP_CXX_ATTR_DECL std::string_view sv;
        LIBNP_CXX_ATTR_DECL std::vector<T> data;
        LIBNP_CXX_ATTR_DECL np::dtype dtype;
        LIBNP_CXX_ATTR_DECL std::map<T, T *> flags;
        LIBNP_CXX_ATTR_DECL T imag;
        LIBNP_CXX_ATTR_DECL T real;
        LIBNP_CXX_ATTR_DECL int size;
        LIBNP_CXX_ATTR_DECL int itemsize;
        LIBNP_CXX_ATTR_DECL int nbytes;
        LIBNP_CXX_ATTR_DECL int ndim;
        LIBNP_CXX_ATTR_DECL std::vector<int> strides;
    };
}

#endif