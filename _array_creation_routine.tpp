#include "array.h"
#include "dtype.h"
#include "ndarray.h"

namespace np {
    template<typename _Tp>
    np::Array<_Tp>& empty(std::tuple<_Tp> shape,
                    np::dtype type, 
                    //np::matrix::Order order,
                    std::optional<std::string> device,
                    std::optional<np::Array<_Tp>> like)
    {
        //auto var = np::Array<_Tp>(, );
        
        
    
        
    }

    template<typename _Tp>
    np::Array<_Tp>& empty_like() 
    {
        
    }
}