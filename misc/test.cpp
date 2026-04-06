#include "template_class.h"

using namespace misc;

auto main() -> int
{
    MyObject<int> obj;
    obj.getValue();
    obj.printValue();
    return 0;
}