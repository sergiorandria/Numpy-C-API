#include <atomic>
#include <concepts>
#include <cstdint>
#include <iostream> 

enum class NewType { NewType };

template <typename T> 
struct nt_to_std;

template <>
struct nt_to_std<std::uint8_t> { using value_type = std::uint8_t; };

template <auto T>
struct storage_classifier 
{
    using value_type = typename nt_to_std<std::uint8_t>::value_type; 
    value_type v {};

    static constexpr NewType type = T; 

    constexpr storage_classifier() noexcept = default; 
    constexpr storage_classifier(const value_type &value): v(value) { }

    constexpr operator value_type&() noexcept { return v; }
    constexpr operator value_type() const noexcept { return v; }

    constexpr storage_classifier& operator=(int other) {
        v = static_cast<value_type>(other);
        return *this; 
    }
};

int main() {
    using bb = storage_classifier<NewType::NewType>;

    bb a = 9;
    std::cout << static_cast<int>(a.v) << std::endl;  
    return 0;
}