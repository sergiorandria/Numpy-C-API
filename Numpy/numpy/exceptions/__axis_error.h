#ifndef AXISERROR  
#define AXISERROR 

#include <optional> 
#include <stdexcept>

/* 
    .Exception::AxisError 
        Axis supplied was invalid. 
        This is raised whenever an axis parameter is specified that
        is larger than the number of array dimensions.
*/

namespace exceptions {
    template<class _Tp>
    class _Numpy_rank_warning: public std::runtime_error 
    {
        std::string what_message; 
    public: 
        const char* what() const override;

        _Tp axis; 
/*
        The out of bounds axis or a custom exception message.
        If an axis is provided, then ndim should be specified as well.  
*/
        std::optional ndim; 
//      The number of array dimensions.      

    };
}

#endif 