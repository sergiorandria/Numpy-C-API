#include <stdexcept>
#include <iostream>
#include <ranges>
#include <utility>

#include "ndarray.h"
#include "numpy/exceptions/__visible_deprecation.h"

template<typename _Tp>
/**
 * @brief Constructs an Ndarray object with the specified shape, data type, and other optional parameters.
 * 
 * @tparam _Tp The type of the elements in the Ndarray.
 * @param shape A tuple representing the shape of the Ndarray.
 * @param type The data type of the elements in the Ndarray.
 * @param buffer An optional parameter that provides a buffer to initialize the Ndarray.
 * @param offset An optional parameter that specifies the offset in the buffer.
 * @param strides An optional parameter that specifies the strides for each dimension.
 * @param order The memory layout order of the Ndarray (C or Fortran order).
 */

np::Ndarray<_Tp>::Ndarray(std::initializer_list<_Tp> il)
{
    this->shape = il;
    this->ndim = this->shape.size();
    this->itemsize = sizeof(_Tp);

    int n = 1; 
    for(auto _elem: il)
        n *= _elem;

    this->data.resize(n);
    this->strides.resize(this->ndim);
}

template<typename _Tp>
np::Ndarray<_Tp>::Ndarray(std::vector<int> shape, 
                            np::dtype type,
                            std::optional<std::vector<_Tp>> buffer,
                            std::optional<off_t> offset,
                            std::optional<std::tuple<int>> strides,
                            np::matrix::Order order) noexcept 
{

    
    // Assign the shape of the Ndarray
    this->shape = shape;
    
    // Assign the data type of the Ndarray
    this->type = type; 
    
    // Assign the buffer if provided
    this->buffer = buffer; 
    
    // Assign the offset if provided
    this->offset = offset;
    
    // Assign the strides if provided
    this->strides = strides; 
    
    // Assign the memory layout order
    this->order = order;

    /*T 
    data ::emplace()
    dtype
    flags 
    size 
    itemsize = sizeof(_Tp); 
    nbytes 
    ndim 
    strides */

    switch (type)
    {
        case np::int16:
        {
            this->itemsize = 4;
        }
        break;
        case np::int32: 
        {
            this->itemsize = 5;
        }
        break;
        case np::int64:
        {
            this->itemsize = 6;
        } 
        break;
        case np::uint8: 
        {
            this->itemsize = 3;
        }
        break;
        case np::uint16:
        {
            this->itemsize = 4;
        } 
        break;
        case np::uint32: 
        {
            this->itemsize = 5;
        }
        break;
        case np::uint64:
        {
            this->itemsize = 6;
        } 
        break;
        case np::float16: 
        case np::float32: 
        case np::float64: 
        case np::longdouble: 
        {

        }
        break;
        case np::complex64: 
        case np::complex128:
        case np::clongdouble:
        {

        }
        break;
        case np::bool_: 
        {

        }
        break;
        case np::string_: 
        case np::unicode_:
        {

        }
        break;
        case np::datetime64:
        case np::timedelta64: 
        {

        } 
        break;
        case np::void_: 
        {

        }
        break; 
        case np::object_:
        {

        }
        break;
    }

    /*auto size = std::tuple_size<decltype(this->shape)>::value;
    if (size != 0)
    {
        for(int i = 0; i != size; ++i)
        {
            _Tp value = std::get<i>(this->shape);

        
        
        }
    }
    else 
    {
        throw std::invalid_argument("Ndarray");
    }
    
    if (buffer.has_value())
    {

    }
    if (offset.has_value())
    {

    }*/
    /*if (!strides.has_value())
    {
        auto size = std::tuple_size<decltype(this->shape)>::value;
    }*/
}

template <typename _Tp>
std::vector<int> np::Ndarray<_Tp>::_compute_strides() const {
    int n = this->shape.size();
    std::vector<int> computed_strides(n, 1);
    
    int stride = 1;
    for (int i = n - 1; i >= 0; --i) {
        computed_strides[i] = stride;
        stride *= this->shape[i];
    }
    
    return computed_strides;
}

template <typename _Tp>
template <std::size_t Size>
std::size_t np::Ndarray<_Tp>::_get_flat_index(const std::array<std::size_t, Size>& indices) const {
    std::vector<int> computed_strides = _compute_strides();
    
    std::size_t flatIndex = 0;
    for (std::size_t i = 0; i < Size; i++) {
        flatIndex += indices[i] * computed_strides[i];
    }

    return flatIndex;
}

template <typename _Tp> 
template <std::size_t Size>
void np::Ndarray<_Tp>::set(const std::array<std::size_t, Size>& indices, _Tp value) {
    std::size_t index = _get_flat_index(indices);
    this->data[index] = value;
}

template <typename _Tp>
template <std::size_t Size>
_Tp np::Ndarray<_Tp>::get(const std::array<std::size_t, Size>& indices) const {
    return this->data[_get_flat_index(indices)];
}

/*template<typename _Tp>
_Tp& np::Ndarray<_Tp>::operator[](std::size_t sizes)
{
    return this->buffer.value()[sizes];
}*/

/*template<typename _Tp>
_Tp& np::Ndarray<_Tp>::operator()(std::vector<int> indexes) 
{
    try {
        return this->data[0];
    }

    catch (exceptions::Visible_Deprecation<void>& e)
    {
        std::cout << "Exception as e:" << e.what() << std::endl;
    }
}*/

template <typename _Tp>
void np::Ndarray<_Tp>::_print_recursive(std::size_t dim, std::size_t offset, std::ostream& output) const {
    std::vector<int> strides = _compute_strides();

    if (dim == this->shape.size() - 1) 
    {
        output << "[";
        for (std::size_t i = 0; i < this->shape[dim]; i++) 
        {
            if (i != this->shape[dim] - 1)
                output << this->data[offset + i] << ",";
            else output << this->data[offset + i];
            
        }   
        output << "]";
    }

    else 
    {
        output << "[";
        for (std::size_t i = 0; i < this->shape[dim]; i++) 
        {
            _print_recursive(dim + 1, offset + i * strides[dim],output);
            if (i < this->shape[dim] - 1) 
                output << ",";
        }
        output << "]";
    }
}

template <typename _Tp>
std::ostream& operator<<(std::ostream& output, np::Ndarray<_Tp> array)
{
    output << "ndarray(";
    array._print_recursive(0, 0, output);
    output << ")" << std::endl;   
    return output; 
}

/*template <typename _Tp = double>
np::_Numpy_ndarray_proxy np::Ndarray::operator[](std::size_t index) {
    return np::_Numpy_ndarray_proxy(*this, index * strides[0], 1);
}*/