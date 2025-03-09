#include "array.h"
#include "numpy/exceptions/__visible_deprecation.h"

#include <ranges>

template<typename _Tp>
template<typename Container>
    requires std::ranges::range<Container>
np::Array<_Tp>::Array(const Container& container, std::optional<np::dtype> type) 
{
    this->data.emplace();
    for (auto x: container)
    {
        this->data->push_back(x);
    }
}

template <typename _Tp>
np::Array<_Tp>::Array(std::initializer_list<_Tp> initList, std::optional<np::dtype> type) {
    *this = Array(std::vector<_Tp>(initList), type);
}

template <typename _Tp>
_Tp& np::Array<_Tp>::operator[](int i)
{
    try {
        if (!this->data) {
            this->data.emplace();
        }

        return (*this->data)[i];
    }

    catch (exceptions::Visible_Deprecation<void>& e) {

    }
}

template <typename _Tp>
np::Array& np::Array<_Tp>::operator=(const Array&& array) 
{
    this->T = array.T;
    this->data = array.data;
    this->dtype = array.dtype;
    this->flags = array.flags; 
    this->imag = array.imag;
    this->real = array.real;
    this->size = array.size;
    this->itemsize = array.itemsize;
    this->nbytes = array.nbytes;
    this->ndim = array.ndim; 
    this->strides = array.strides;
}