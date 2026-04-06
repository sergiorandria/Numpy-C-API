#include <iostream>
#include "../ndarray.hpp"

using namespace std;
using namespace np;

auto main() -> int
{
    Ndarray array{18, 28, 30, 2};
    for (auto &c : array.shape)
        cout << c << " ";
    cout << endl;
    int x = array.shape[0];
    int y = array.shape[1];
    int z = array.shape[2];
    cout << array << endl;
    return 0;
}