#include "array.h"
#include "numpy/exceptions/__visible_deprecation.h"

#include <ranges>
#include <utility>

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
np::Array<_Tp>& np::Array<_Tp>::operator=(const Array<_Tp>&& array) 
{
    this->T        = std::move(array.T);
    this->data     = std::move(array.data);
    this->dtype    = std::move(array.dtype);
    this->flags    = std::move(array.flags); 
    this->imag     = std::move(array.imag);
    this->real     = std::move(array.real);
    this->size     = std::move(array.size);
    this->itemsize = std::move(array.itemsize);
    this->nbytes   = std::move(array.nbytes);
    this->ndim     = std::move(array.ndim); 
    this->strides  = std::move(array.strides);

    return *this;
}