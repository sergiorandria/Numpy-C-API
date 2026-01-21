#ifndef _NUMPY_DTYPE_FUND
#define _NUMPY_DTYPE_FUND

// Because C++ doesn't have a built-in dictionary type, 
// we will use unordered_map for a dictionary implementation.

#include <unordered_map>
#include <type_traits>
#include <optional>

namespace np {
/* 
    Numpy supports a much greater variety of numerical 
    types.
*/
#ifndef DTYPE_CLASS 
    enum class dtype 
    {
        int8, //Integer types
        int16, 
        int32, 
        int64, 
        uint8,
        uint16, 
        uint32, 
        uint64,

        float16, //Floating-Point types
        float32,
#ifndef FLOAT32_OTHER_EXTENSION 
#define single float32 

#endif 
        float64, 
#ifndef FLOAT64_OTHER_EXTENSION 
#define double_ float64   
 
#endif
        longdouble, 

        complex64, //Complex Number types;
        complex128, 

#ifndef COMPLEX128_OTHER_EXTENSION 
#define cdouble_ complex128  

#endif  
        clongdouble, 

        bool_, // Boolean Type

        string_, //String and Unicode Types 
        unicode_, 

        datetime64, //Datetime and Timedelta Types
        timedelta64, 
        
        void_, //Special Types 
        object_ // General Object type
    };
#define DTYPE_CLASS 
#endif

    template<class _Tp, class _TAlloc>
/* 
    Numpy numerical types are instances of numpy::Dtypes. Once an array 
    is created you can specify dtype using scalar types: numpy.bool, numpy.float32
*/
    class _Numpy_dtype_internal
    {
    public:     
        _Numpy_dtype_internal(dtype type);
        _Numpy_dtype_internal(dtype type, std::optional<_Tp> align);
        _Numpy_dtype_internal(dtype type, std::optional<_Tp> align, std::bool_constant<true> copy); 
        _Numpy_dtype_internal(dtype type, std::optional<_Tp> align, std::bool_constant<true> copy, std::unordered_map<_Tp, std::optional<_Tp>>  metadata);

         ~_Numpy_dtype_internal();
        
        virtual _Tp getDtype() const;
        virtual std::optional<_Tp> getAlignement() const;
        virtual std::integral_constant<bool, true> getCopy() const;
        virtual std::unordered_map<_Tp, std::optional<_Tp>> getMetadata() const;

    private: 
        _Tp dtype;
        std::optional<_Tp> align;
        std::bool_constant<true> copy;  
        std::unordered_map<_Tp, std::optional<_Tp>> metadata;
    };

constexpr dtype float32 = dtype::float32;
constexpr dtype float64 = dtype::float64;
constexpr dtype int32 = dtype::int32;
constexpr dtype int64 = dtype::int64;
constexpr dtype bool_ = dtype::bool_;

//Integer types
#define int16 dtype::int16
#define uint8 dtype::uint8 
#define uint16 dtype::uint16
#define uint32 dtype::uint32 
#define uint64 dtype::uint64   

//Floating-Point types
#define float16 dtype::float16
#define longdouble dtype::longdouble 

//Complex types
#define complex64 dtype::complex64 
#define complex128 dtype::complex128 
#define clongdouble dtype::clongdouble 

//Boolean types
#define string_ dtype::string_ 
#define unicode_ dtype::unicode_

//Datetime types
#define datetime64 dtype::datetime64
#define timedelta64 dtype::timedelta64 

//Special types
#define void_ dtype::void_ 
#define object_ dtype::object_

}

#include "dtype.tpp"

#endif 