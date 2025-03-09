#include "__struct_type.h"

TpStruct::__dtype_struct()
{

}

TpStruct::__dtype_struct(std::uint32_t size):
        size(size),
        val(0)
{

}

TpStruct::__dtype_struct(std::uint32_t val): 
    val(val),
    size(0)
{

}

TpStruct::__dtype_struct(std::uint32_t size, std::uint32_t val):
    val(val),
    size(size)
{

}