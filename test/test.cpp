#include "ndarray.h"
#include "numpy/exceptions/__visible_deprecation.h"
#include <iostream>

int main() {
    using Shape = std::tuple<int, int, int>;
    using Strides = std::tuple<std::size_t, std::size_t, std::size_t>;
    auto arr = np::Ndarray({1,2}, np::int32, std::nullopt);

    np::Ndarray<int> ndarray;
    ndarray.shape = Shape{4, 3, 2};

    auto strides = ndarray._compute_strides<Strides>(std::make_index_sequence<3>{});

    std::cout << "Strides: ("
              << std::get<0>(strides) << ", "
              << std::get<1>(strides) << ", "
              << std::get<2>(strides) << ")\n";
    try {
                throw exceptions::Visible_Deprecation<void>("Deprecation warning: This feature is deprecated.");
    }
    catch (exceptions::Visible_Deprecation<void>& e) {
        std::cout << "Exception as e: " << e.what() << std::endl;
    }

    std::cout << "Hello" << std::endl;
    return 0;
}
