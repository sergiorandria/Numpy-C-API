#include <iostream>
#include <vector>

namespace misc {
template <typename T> class NDArray {
private:
  std::vector<T> data;
  std::vector<size_t> dimensions;
  std::vector<size_t> strides;

  size_t computeIndex(const std::vector<size_t> &indices) const {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      index += indices[i] * strides[i];
    }
    return index;
  }

public:
  // Proxy class for element access
  class Proxy {
  private:
    NDArray<T> &array;
    std::vector<size_t> indices;

  public:
    Proxy(NDArray<T> &arr, const std::vector<size_t> &idx)
        : array(arr), indices(idx) {}

    // Assignment operator for writing
    Proxy &operator=(const T &value) {
      array.set(indices, value);
      return *this;
    }

    // Conversion operator for reading
    operator T() const { return array.get(indices); }

    // For chained assignments like a[0][1] = x
    Proxy operator[](size_t index) {
      std::vector<size_t> newIndices = indices;
      newIndices.push_back(index);
      return Proxy(array, newIndices);
    }

    // Const version for reading
    const Proxy operator[](size_t index) const {
      std::vector<size_t> newIndices = indices;
      newIndices.push_back(index);
      return Proxy(array, newIndices);
    }
  };

  // Constructor
  NDArray(const std::vector<size_t> &dims) : dimensions(dims) {
    size_t total = 1;
    for (size_t dim : dims) {
      total *= dim;
    }
    data.resize(total);

    // Compute strides for efficient indexing
    strides.resize(dims.size());
    size_t stride = 1;
    for (int i = dims.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= dims[i];
    }
  }

  // Element access
  T get(const std::vector<size_t> &indices) const {
    return data[computeIndex(indices)];
  }

  void set(const std::vector<size_t> &indices, const T &value) {
    data[computeIndex(indices)] = value;
  }

  // Subscript operator returns proxy
  Proxy operator[](size_t index) { return Proxy(*this, {index}); }

  const Proxy operator[](size_t index) const {
    return Proxy(const_cast<NDArray<T> &>(*this), {index});
  }

  // Print the array
  void print() const { printRecursive(0, {}); }

private:
  void printRecursive(size_t dim, std::vector<size_t> indices) const {
    if (dim == dimensions.size()) {
      std::cout << get(indices);
      return;
    }

    std::cout << "[";
    for (size_t i = 0; i < dimensions[dim]; ++i) {
      std::vector<size_t> newIndices = indices;
      newIndices.push_back(i);
      printRecursive(dim + 1, newIndices);
      if (i < dimensions[dim] - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]";
    if (dim == 0) {
      std::cout << std::endl;
    }
  }
};
} // namespace misc

using namespace misc;

int main() {
  // Create a 2x3x4 3D array
  NDArray<int> arr({2, 3, 4});

  // Fill with some values
  int value = 0;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 4; ++k) {
        arr[i][j][k] = value++;
      }
    }
  }

  // Access and modify elements
  std::cout << "Element at [0][1][2]: " << arr[0][1][2] << std::endl;

  arr[1][2][3] = 100;
  std::cout << "Modified element at [1][2][3]: " << arr[1][2][3] << std::endl;

  // Print the entire array
  std::cout << "\nFull array:" << std::endl;
  arr.print();

  // 2D array example
  NDArray<double> mat({3, 3});

  // Set diagonal elements
  for (size_t i = 0; i < 3; ++i) {
    mat[i][i] = i + 1.0;
  }

  std::cout << "\n2D Matrix:" << std::endl;
  mat.print();

  return 0;
}
