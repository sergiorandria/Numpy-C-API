#ifndef NDARRAY_PROXY_H
#define NDARRAY_PROXY_H

#include "ndarray.h"

namespace np {
    template <typename _Tp>
    class _Numpy_ndarray_proxy {
    public:
        _Numpy_ndarray_proxy(np::Ndarray arr, std::size_t off, std::size_t d);
        _Numpy_ndarray_proxy operator[](std::size_t index) {
            return _Numpy_ndarray_proxy(this->array, offset + index * this->array.strides[dim], dim + 1);
        }

        operator _Tp&();

    private:
        np::Ndarray<_Tp>& array;
        std::size_t offset;
        std::size_t dim;

    };
}

#include "__ndarray_proxy_internal.tpp"

#endif