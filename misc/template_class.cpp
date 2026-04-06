#include "template_class.h"

namespace misc
{
    template <class T>
    MyObject<T>::MyObject() {}

    template <class T>
    void MyObject<T>::getValue()
    {
        std::cout << "Getting value: ";
        std::cin >> objectVal;
    }

    template <class T>
    void MyObject<T>::printValue()
    {
        std::cout << objectVal;
    }
} // namespace misc

template class misc::MyObject<int>;
template class misc::MyObject<float>;