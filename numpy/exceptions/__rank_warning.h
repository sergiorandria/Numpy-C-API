#ifndef RANK_WARNING 
#define RANK_WARNING

#include <string>
#include <stdexcept>

/* 
    .Exceptions::RankWarning:
        Matrix rank warning. 
        Issued by polynomial functions when the design matrix is rank deficient.
*/

namespace exceptions {
    template <class _TP>
    class _Numpy_rank_warning : public std::runtime_error 
    {
        std::string what_msg; 
    public: 
        const char* what() const override; 
        //A function which handle polynomial abstract type 
        // and verify their rank if it is defficient or not.

    };
}

#endif 