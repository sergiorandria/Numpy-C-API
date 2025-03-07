#ifndef _NDARRAY_INTERNAL_H
#define _NDARRAY_INTERNAL_H

#include <map>
#include "dtype.h"

namespace np {
    template <typename _Tp>
    /**
     * @class _Numpy_ndarray_internal
     * @brief Internal representation of a Numpy ndarray.
     * 
     * This class provides an internal structure for handling Numpy ndarray objects.
     * 
     * @note The following attributes are not yet implemented:
     * - .ctypes: ctypes object
     * - .base: ndarray
     * - .flat: numpy::flatiter objects
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
    public: 
/* 
        NOTE: Doesn't have any idea to implement 
            .ctypes: ctypes object 
            .base: ndarray
            .flat: numpy::flatiter objects
*/

        virtual std::string_view T = 0;
        virtual std::optional<std::vector<std::vector<_Tp>> data = 0;
        virtual np::dtype dtype = 0;
        virtual std::map<_Tp, _Tp*> flags = 0;
        virtual _Tp imag = 0;
        virtual _Tp real = 0;
        virtual int size = 0;
        virtual int itemsize = 0;
        virtual int nbytes = 0;
        virtual int ndim = 0;
        virtual std::tuple<int> strides = 0;
    };  
}

#endif