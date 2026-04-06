#ifndef TEMPLATE_CLASS_H
#define TEMPLATE_CLASS_H

#include <iostream>
#include <string>

namespace misc
{
    template <class T>
    class MyObject
    {
      public:
        MyObject();

        void getValue();
        void printValue();

      private:
        T objectVal;
    };
}
#endif