#include "__ndarray_iterator.h"

template <typename _Tp>
np::NdArray_iterator<_Tp>::NdArray_iterator(_Tp* data, const std::vector<size_t>& shape)
: data_(data), shape_(shape), strides_(shape.size(), 1), index_(shape.size(), 0) 
{
    for (int i = shape.size() - 2; i >= 0; --i) 
    {
        _strides_[i] = _strides[i + 1] * _shape[i + 1];
    }
}

template <typename _Tp>
bool np::NdArray_iterator<_Tp>::hasNext() const {
    for (size_t i = 0; i < _shape.size(); ++i) 
    {
        if (_index[i] < _shape[i]) {
            return true;
        }
    }
    return false;
}

template <typename _Tp>
_Tp& np::NdArray_iterator::next() 
{
    size_t flatIndex = 0;
    for (size_t i = 0; i < shape_.size(); ++i) 
    {
        flatIndex += _index[i] * _strides[i];
    }
    incrementIndex();
    return _data[flatIndex];
}

template <_Tp>
void np::NdArray_iterator::_increment_index() 
{
    for (int i = _shape.size() - 1; i >= 0; --i) 
    {
        if (++_index[i] < shape_[i]) 
        {
            break;
        }
        _index[i] = 0;
    }
}