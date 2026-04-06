#include <iostream>
#include <vector>
#include <stack>
#include <memory>
#include <type_traits>

// Utility to check if a type is a vector
template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc>> : std::true_type {};

template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

// Iterator class for N-dimensional vectors
template <typename T>
class NDVectorIterator {
    using Container = std::vector<T>;

    struct IteratorState {
        typename Container::iterator current, end;
    };

    std::stack<IteratorState> stack;
    T* currentElement = nullptr;

    void findNext() {
        while (!stack.empty()) {
            auto& top = stack.top();

            if (top.current == top.end) {
                stack.pop();
                continue;
            }

            if constexpr (is_vector_v<T>) {
                stack.push({top.current->begin(), top.current->end()});
            } else {
                currentElement = &(*top.current);
                ++top.current;
                return;
            }

            ++top.current;
        }
        currentElement = nullptr;
    }

public:
    explicit NDVectorIterator(Container& vec) {
        stack.push({vec.begin(), vec.end()});
        findNext();
    }

    bool hasNext() const {
        return currentElement != nullptr;
    }

    T next() {
        if (!currentElement) throw std::out_of_range("No more elements!");
        T* temp = currentElement;
        findNext();
        return *temp;
    }
};

int main() {
    std::vector<std::vector<std::vector<int>>> ndVector = {
        {{1, 2}, {3, 4}}, 
        {{5, 6}, {7, 8}}
    };

    NDVectorIterator<std::vector<std::vector<int>>> iter(ndVector);
    while (iter.hasNext()) {
        std::cout << iter.next() << " ";
    }
    std::cout << std::endl;

    return 0;
}
