#include "__dtype_internal.h"
#include <type_traits>

template<class T, class TAlloc>
np::Numpy_dtype_internal<T, TAlloc>::Numpy_dtype_internal(np::dtype type)
{
    this->dtype_ = type;
    this->align = nullptr;
}

template<typename T, typename TAlloc>
np::Numpy_dtype_internal<T, TAlloc>::Numpy_dtype_internal(np::dtype type,
        std::optional<T> align)
{
    this->dtype_ = type;
    this->align = align;
}

template<class T, class TAlloc>
np::Numpy_dtype_internal<T, TAlloc>::Numpy_dtype_internal(np::dtype type,
        std::optional<T> copy, std::bool_constant<true> align)
{
    this->dtype_ = type;
    this->align = align;
}

template<class T, class TAlloc>
np::Numpy_dtype_internal<T, TAlloc>::Numpy_dtype_internal(np::dtype type,
        std::optional<T> copy, std::bool_constant<true> align,
        std::unordered_map<T, std::optional<T>> metadata)
{
    this->dtype_ = type;
    this->align = align;
    this->metadata = metadata;
}

template<class T, class _TAlloc>
np::Numpy_dtype_internal<T, _TAlloc>::~Numpy_dtype_internal()
{
}

template<class T, class _TAlloc>
auto np::Numpy_dtype_internal<T, _TAlloc>::getDtype() const -> decltype(T())
{
    return this->dtype_;
}

template<class T, class TAlloc>
auto np::Numpy_dtype_internal<T, TAlloc>::getAlignement()
const -> decltype(std::optional<T>())
{
    return this->copy;
}

template<class T, class TAlloc>
auto
np::Numpy_dtype_internal<T, TAlloc>::getCopy() const ->
std::integral_constant<bool, true>
{
    return static_cast<std::integral_constant<bool, true>> (this->copy);
}

template<class T, class _TAlloc>
auto
np::Numpy_dtype_internal<T, _TAlloc>::getMetadata() const ->
std::unordered_map<T, std::optional<T>>
{
    return this->metadata;
}
