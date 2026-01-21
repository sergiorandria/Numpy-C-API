#ifndef _NDARRAY_INTERNAL_H
#define _NDARRAY_INTERNAL_H

#include <map>
#include <vector>
#include <string_view>

#include "dtype.h"

namespace np {
    template <typename _Tp = double>
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
    class _Numpy_ndarray_internal {
    protected: 
    /** 
    * @note The following attributes are not yet implemented:
    * - .ctypes: ctypes object
    * - .base: ndarray
    * - .flat: numpy::flatiter objects
    */

        std::string_view T;
        std::vector<_Tp> data;
        np::dtype dtype;
        std::map<_Tp, _Tp*> flags;
        _Tp imag;
        _Tp real;
        int size;
        int itemsize;
        int nbytes;
        int ndim;
        std::vector<int> strides;
    };  
}

#endif