#include "__ndarray_proxy_internal.h"

template <typename _Tp>
np::_Numpy_ndarray_proxy<_Tp>::_Numpy_ndarray_proxy(np::Ndarray<_Tp>& arr, std::size_t off, std::size_t d)
: array(arr), offset(off), dim(d) 
{

}

template <typename _Tp>
np::_Numpy_ndarray_proxy<_Tp>::_Numpy_ndarray_proxy operator[](std::size_t index) 
{
    return _Numpy_ndarray_proxy(array, offset + index * array.strides[dim], dim + 1);
}

template <typename _Tp>
operator _Tp&() const 
{
    return array.data[offset];
}