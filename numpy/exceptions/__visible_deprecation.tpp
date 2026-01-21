#include "__visible_deprecation.h"

namespace exceptions {
    template<class _Tp>
    using Visible_Deprecation = exceptions::_Numpy_visible_deprecation<_Tp>;
}

template<class _Tp>
exceptions::_Numpy_visible_deprecation<_Tp>::_Numpy_visible_deprecation(const char*msg): std::runtime_error(msg)
{
    what_msg = msg;
}

template<class _Tp>
const char* exceptions::_Numpy_visible_deprecation<_Tp>::what() const throw()
{
    return what_msg.c_str(); 
}