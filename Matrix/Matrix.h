#ifndef MATRIX_H 
#define MATRIX_H 

#include "../ndarray.h"
#include "_Matrix_internal.h"

namespace np {
    template<typename _Tp>
    class Matrix: public _Matrix_internal<_Tp> 
    {
    public: 
        Matrix() = default; 
        Matrix(np::Ndarray<_Tp>) 

        ~Matrix() = default;

    };
}
#endif  