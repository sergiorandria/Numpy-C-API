#include <stdexcept>
#include <iostream>

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
np::Ndarray<_Tp>::Ndarray(std::tuple<int,int,int> shape, 
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
    data 
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
    if (!strides.has_value())
    {
        auto size = std::tuple_size<decltype(this->shape)>::value;
    }
}

template <typename _Tp>
template <typename Tuple, std::size_t... Is>
constexpr Tuple np::Ndarray<_Tp>::_compute_strides(std::index_sequence<Is...>) const {
    Tuple strides{};
    std::size_t stride = 1;

    auto compute_stride = [&](auto I) {
        constexpr std::size_t index = sizeof...(Is) - 1 - I;
        
        std::get<index>(strides) = stride;
        stride *= std::get<index>(shape);
    };

    (compute_stride(std::integral_constant<std::size_t, Is>{}), ...);

    return strides;
}

template<typename _Tp>
bool np::Ndarray<_Tp>::all(size_t axis, 
                            std::optional<Ndarray> out, 
                            bool keepdims,
                            std::vector<bool> where) const
{

    return true;
}

template<typename _Tp>
_Tp& np::Ndarray<_Tp>::operator[](std::size_t sizes)
{
    return this->buffer.value()[sizes];
}

template<typename _Tp>
_Tp& np::Ndarray<_Tp>::operator()(std::vector<int> indexes) 
{
    //auto _strideSize = std::tuple_size<_Tp>(this->strides)::value;

    try {
        return this->data[0];
    }
    catch (exceptions::Visible_Deprecation<void>& e)
    {
        std::cout << "Exception as e:" << e.what() << std::endl;
    }
}