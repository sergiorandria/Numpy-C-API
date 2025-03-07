#ifndef DTYPE_STRUCT_DEF 
#define DTYPE_STRUCT_DEF

#include <cstdint>

typedef struct __dtype_struct
{
    std::uint32_t size;
    std::uint32_t val;
    
    bool atomic;
    bool align;

    __dtype_struct();
    __dtype_struct(std::uint32_t size);
    __dtype_struct(std::uint32_t val);
    __dtype_struct(std::uint32_t size, std::uint32_t val);
} TpStruct;

#endif