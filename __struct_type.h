#ifndef DTYPE_STRUCT_DEF 
#define DTYPE_STRUCT_DEF

#include <cstdint>

typedef struct _dtype_struct
{
    std::uint32_t size;
    std::uint32_t val;
    
    bool atomic;
    bool align;

    _dtype_struct();
    _dtype_struct(std::uint32_t size);
    _dtype_struct(std::uint32_t val);
    _dtype_struct(std::uint32_t size, std::uint32_t val);
} TpStruct;

#endif