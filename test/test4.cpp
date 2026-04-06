#include "../ndarray.hpp"
#include "../dtype.h"
#include <iostream>

int main()
{
    int tmp;
    np::Ndarray arr = np::Ndarray({2, 3, 2, 3});
    // Fill array using chained subscript operator [][][][]
    for (std::size_t l = 0; l < 2; ++l)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 2; ++j)
            {
                for (std::size_t k = 0; k < 3; ++k)
                {
                    std::cout << "Arr[" << l << "][" << i << "][" << j << "][" << k << "] = ";
                    std::cin >> tmp;
                    // NEW: Use chained subscript operator!
                    arr[l][i][j][k] = tmp;
                    // OLD: arr.set(std::array<std::size_t, 4>{l,i,j,k}, tmp);
                }
            }
        }
    }
    // Read back using chained subscript
    std::cout << "\nReading back values:\n";
    for (std::size_t l = 0; l < 2; ++l)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 2; ++j)
            {
                for (std::size_t k = 0; k < 3; ++k)
                {
                    std::cout << "arr[" << l << "][" << i << "][" << j << "][" << k << "] = "
                              << arr[l][i][j][k] << "\n";
                }
            }
        }
    }
    std::cout << arr;
    return 0;
}
