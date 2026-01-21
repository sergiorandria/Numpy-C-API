#include "__struct_type.h"

TpStruct::_dtype_struct()
{

}

TpStruct::_dtype_struct(std::uint32_t size):
        size(size),
        val(0)
{

}

TpStruct::_dtype_struct(std::uint32_t val): 
    val(val),
    size(0)
{

}

TpStruct::_dtype_struct(std::uint32_t size, std::uint32_t val):
    val(val),
    size(size)
{

}