#include <iostream>

int consoleLog(std::string message, int i = 1)
{
    for(int j = 0; j < i; ++j)
        std::cout << message << std::endl;

    return 0;
}

int main()
{
    consoleLog("Hello World", 10);
}

