#ifndef _DTYPE_PROMOTION_ERROR 
#define _DTYPE_PROMOTION_ERROR

#include "../../__dtype_internal.h"
#include "../../__struct_type.h"

/* 
    .Exception::DtypePromotionError: 
        Multiple Dtypes could not be converted to a common one. 
*/
namespace exception {
    template<class _Tp>
    class _Numpy_dtype_promotion: public std::runtime_error 
    {
        std::string what_message; 
    public: 
        const char* what() const override;
        /* Promotion should not be considered "invalid" between dtypes of two arrays when 
        arr1 == arr2 can safely return all False 
        */
    };
}
#endif   