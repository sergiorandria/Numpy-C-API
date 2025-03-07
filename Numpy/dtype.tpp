#include "__dtype_internal.h"
#include <type_traits>

template<class _Tp, class _TAlloc>
using Dtype = np::_Numpy_dtype_internal<_Tp, _TAlloc>; 

template<class _Tp, class _TAlloc>
np::_Numpy_dtype_internal<_Tp, _TAlloc>::_Numpy_dtype_internal(np::dtype type)
{
    //_TAlloc.allocate(this, _Tp);
    
    this->dtype = type;  
    this->align = nullptr;
}

template<typename _Tp, typename _TAlloc>
np::_Numpy_dtype_internal<_Tp, _TAlloc>::_Numpy_dtype_internal(np::dtype type, std::optional<_Tp> copy)
{
    //_TAlloc.allocate(this, _Tp);

    this->dtype = type;
    this->align = nullptr;
}

template<class _Tp, class _TAlloc>
np::_Numpy_dtype_internal<_Tp, _TAlloc>::_Numpy_dtype_internal(np::dtype type, std::optional<_Tp> copy, std::bool_constant<true> align)
{
    //_TAlloc.allocate(this, _Tp);

    this->dtype = type;
    this->align = align;
}

template<class _Tp, class _TAlloc>
np::_Numpy_dtype_internal<_Tp, _TAlloc>::_Numpy_dtype_internal(np::dtype type, std::optional<_Tp> copy, std::bool_constant<true> align, std::unordered_map<_Tp, std::optional<_Tp>> metadata)
{
    //_TAlloc.allocate(this, _Tp);

    this->dtype = type;
    this->align = align;
    this->metadata = metadata;
}

template<class _Tp, class _TAlloc>
np::_Numpy_dtype_internal<_Tp, _TAlloc>::~_Numpy_dtype_internal()
{

}

template<class _Tp, class _TAlloc>
_Tp np::_Numpy_dtype_internal<_Tp, _TAlloc>::getDtype() const
{
    return this->dtype;
}

template<class _Tp, class _TAlloc>
std::optional<_Tp> np::_Numpy_dtype_internal<_Tp, _TAlloc>::getAlignement() const
{
    return this->copy;
}

template<class _Tp, class _TAlloc>
std::integral_constant<bool, true> np::_Numpy_dtype_internal<_Tp, _TAlloc>::getCopy() const
{
    return static_cast<std::integral_constant<bool, true>> (this->copy);
}

template<class _Tp, class _TAlloc>
std::unordered_map<_Tp, std::optional<_Tp>> np::_Numpy_dtype_internal<_Tp, _TAlloc>::getMetadata() const
{
    return this->metadata;
}
