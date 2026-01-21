#include <tuple>
#include <array>
#include <iostream>

template <typename Tuple, typename Shape, std::size_t... Is>
constexpr Tuple compute_strides(const Shape& shape, std::index_sequence<Is...>) {
    Tuple strides{};
    size_t stride = 1;
    ((std::get<Is>(strides) = (Is + 1 < sizeof...(Is)) ? stride *= shape[Is + 1] : 1), ...);
    return strides;
}

template <typename Tuple, typename Shape>
constexpr Tuple compute_strides(const Shape& shape) {
    return compute_strides<Tuple>(shape, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

int main() {
    std::array<int, 3> shape = {4, 3, 2}; // Example shape
    using StridesTuple = std::tuple<size_t, size_t, size_t>; // Fixed-size tuple

    auto strides = compute_strides<StridesTuple>(shape);

    std::cout << "Strides: (" 
              << std::get<0>(strides) << ", " 
              << std::get<1>(strides) << ", " 
              << std::get<2>(strides) << ")\n";

    return 0;
}
