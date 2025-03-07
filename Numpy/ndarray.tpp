#include "ndarray.h"

template<typename _Tp>
np::Ndarray<_Tp>::Ndarray(std::tuple<int> shape, 
                            np::dtype type,
                            std::optional<std::vector<std::vector<_Tp>>> buffer,
                            std::optional<off_t> offset,
                            std::optional<std::tuple<int>> strides,
                            np::matrix::Order order)

{
    this->shape = shape;
    this->type = type; 
    this->buffer = buffer; 
    this->offset = offset;
    this->strides = strides; 
    this->order = order;

    switch (type)
    {
        
    }
}


template<typename _Tp>
bool np::Ndarray<_Tp>::all(size_t axis, 
                            std::optional<Ndarray> out, 
                            bool keepdims,
                            std::vector<bool> where) const
{
    for(std::vector<std::vector<_Tp>>::iterator it = buffer.begin(); it != buffer.end(); ++i)
    {
        if (not *it)
            return false;
    }

    return true;
}