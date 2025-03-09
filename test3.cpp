#include <iostream>

#include "array.h"

int main()
{
    int data[] = {1,2,3,4,5,6};
    auto array = np::Array<int>(data);

    std::cout << array[2] << std::endl;
}