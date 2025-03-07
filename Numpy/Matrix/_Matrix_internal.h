#ifndef _MATRIX_INTERNAL
#define _MATRIX_INTERNAL
/* 
    numpy::Matrix: 
        returns a matrix from an array-like object, 
        or from a string data
*/

#include <optional>
#include <string>
#include <vector>
#include <string_view>

#include "../dtype.h"
#include "../__ndarray_internal.h"

namespace np {
    template<typename _Tp>
    class _Matrix_internal: public np::_Numpy_ndarray_internal<_Tp>
    {

    public: 
        

    private:
        std::vector<vector<_Tp>> data;      // Interpreted as a matrix with commas or spaces separating
        std::optional<np::dtype> type;                     // data type of the output matrix
        std::bool_constant<false> copy;     //determines whether the data is copied or whether a view is constructed
    };
}

#endif