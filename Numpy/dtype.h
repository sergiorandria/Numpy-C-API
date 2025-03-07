#ifndef DTYPE_H
#define DTYPE_H

#include "__dtype_internal.h"

template<np::dtype T, typename _TAlloc>
class Dtype: public _Numpy_dtype_internal<T, std::allocator<_TAlloc> >
{
public: 
    Dtype(dtype type): 
            _Numpy_dtype_internal<T, std::allocator<T>>(type){};
    Dtype(dtype type, std::optional<_TAlloc> align): 
            _Numpy_dtype_internal<T, std::allocator<_TAlloc>>(type, align){};
    Dtype(dtype type, std::optional<_TAlloc> align, std::bool_constant<true> copy):
            _Numpy_dtype_internal<T, std::allocator<_TAlloc>>(type, align, copy){}; 
    Dtype(dtype type, std::optional<_TAlloc> align, std::bool_constant<true> copy, std::unordered_map<T, std::optional<T>>  metadata): 
            _Numpy_dtype_internal<T, std::allocator<_TAlloc>>( type, align, copy, metadata){};

    ~Dtype();
};

#endif