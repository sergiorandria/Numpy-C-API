#ifndef VISIBLE_DEPRECATION_WARNING 
#define VISIBLE_DEPRECATION_WARNING

#include <string>
#include <stdexcept>

/* 
    .Exceptions::VisibleDeprecation    
        By default, the interpreter doesn't show deprecation warnings, 
        so this class can be used when a very visible warning is helpful, 
        for example because the usage is most likely a user bug    
*/
namespace exceptions {
    template<class _Tp>
    class _Numpy_visible_deprecation: public std::runtime_error 
    {
    private:
        std::string what_msg; 
    public: 
        explicit _Numpy_visible_deprecation(const char* msg);
        const char* what() const throw();
        //Implement a function which can check if the current warning must be 
        // shown to the user 
    };
}

#include "__visible_deprecation.tpp"

#endif