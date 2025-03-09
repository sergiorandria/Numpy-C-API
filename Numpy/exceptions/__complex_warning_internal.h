#ifndef defined(NUMPY_COMPLEX_WARNING)
#define NUMPY_COMPLEX_WARNING

#ifndef NUMPY_COMPLEX_STATIC_CAST
#define NUMPY_COMPLEX_STATIC_CAST 0x0102

#endif

#ifndef NUMPY_COMPLEX_REINTERPRET_CAST 
#define NUMPY_COMPLEX_REINTERPRET_CAST 0x0201 

#endif 

#pragma region 
#pragma endregion

#include <string>
#include <stdexcept>

/*
    .Exception::ComplexWarning:
        The warning raised when casting a complex dtype to a real dtype. 
        As implemented, casting a complex number to a real discards its imaginary part, but
        this behavior may not be what the user actually wants.
*/

namespace exceptions {
    template <class _Tp>
    class _Numpy_complex_warning: public std::runtime_error
    {
        std::string what_msg;
    public: 
        const char* what() const override;
        //Have to store some dtype values inside, check if the cast 
        //is allowed or not (if reinterpret cast, return a warning message more than
        // a static_cast)
        //The function what() is a common function of all exceptions
    };
}

#endif