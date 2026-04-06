#ifndef NP_ARRAY_TPP
#define NP_ARRAY_TPP

#include "array.h"
#include "numpy/exceptions/__visible_deprecation.h"

#include <algorithm>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace np {

template <typename T>
template <typename Container>
  requires std::ranges::range<Container>
Array<T>::Array(const Container &container, std::optional<dtype> type) {
  this->data.emplace();

  if constexpr (std::ranges::sized_range<Container>) {
    this->data->reserve(std::ranges::size(container));
  }

  std::ranges::copy(container, std::back_inserter(*this->data));

  if (type) {
    this->dtype = *type;
  }
}

template <typename T>
Array<T>::Array(std::initializer_list<T> initList, std::optional<dtype> type) {
  this->data.emplace();
  this->data->reserve(initList.size());
  this->data->assign(initList.begin(), initList.end());

  if (type) {
    this->dtype = *type;
  }
}

template <typename T>
typename Array<T>::reference Array<T>::operator[](size_type i) noexcept {
  if (!this->data) [[unlikely]] {
    this->data.emplace();
  }

  return (*this->data)[i];
}

template <typename T>
typename Array<T>::const_reference
Array<T>::operator[](size_type i) const noexcept {
  return (*this->data)[i];
}

template <typename T> Array<T> &Array<T>::operator=(Array<T> &&other) noexcept {
  if (this != &other) {
    this->T = std::move(other.T);
    this->data = std::move(other.data);
    this->dtype = std::move(other.dtype);
    this->flags = std::move(other.flags);
    this->imag = std::move(other.imag);
    this->real = std::move(other.real);
    this->size = std::move(other.size);
    this->itemsize = std::move(other.itemsize);
    this->nbytes = std::move(other.nbytes);
    this->ndim = std::move(other.ndim);
    this->strides = std::move(other.strides);
  }
  return *this;
}

template <typename T>
Array<T> &Array<T>::operator=(std::initializer_list<T> init_list) {
  if (!this->data) {
    this->data.emplace();
  }
  this->data->assign(init_list.begin(), init_list.end());
  return *this;
}

template <typename T> void Array<T>::swap(Array<T> &other) noexcept {
  using std::swap;
  swap(this->T, other.T);
  swap(this->data, other.data);
  swap(this->dtype, other.dtype);
  swap(this->flags, other.flags);
  swap(this->imag, other.imag);
  swap(this->real, other.real);
  swap(this->size, other.size);
  swap(this->itemsize, other.itemsize);
  swap(this->nbytes, other.nbytes);
  swap(this->ndim, other.ndim);
  swap(this->strides, other.strides);
}

template <typename T> auto Array<T>::begin() noexcept {
  return this->data ? this->data->begin() : typename std::vector<T>::iterator{};
}

template <typename T> auto Array<T>::end() noexcept {
  return this->data ? this->data->end() : typename std::vector<T>::iterator{};
}

template <typename T> auto Array<T>::begin() const noexcept {
  return this->data ? this->data->begin()
                    : typename std::vector<T>::const_iterator{};
}

template <typename T> auto Array<T>::end() const noexcept {
  return this->data ? this->data->end()
                    : typename std::vector<T>::const_iterator{};
}

template <typename T> auto Array<T>::cbegin() const noexcept { return begin(); }

template <typename T> auto Array<T>::cend() const noexcept { return end(); }

template <typename T> auto Array<T>::rbegin() noexcept {
  return this->data ? this->data->rbegin()
                    : typename std::vector<T>::reverse_iterator{};
}

template <typename T> auto Array<T>::rend() noexcept {
  return this->data ? this->data->rend()
                    : typename std::vector<T>::reverse_iterator{};
}

template <typename T> auto Array<T>::rbegin() const noexcept {
  return this->data ? this->data->rbegin()
                    : typename std::vector<T>::const_reverse_iterator{};
}

template <typename T> auto Array<T>::rend() const noexcept {
  return this->data ? this->data->rend()
                    : typename std::vector<T>::const_reverse_iterator{};
}

template <typename T> auto Array<T>::crbegin() const noexcept {
  return rbegin();
}

template <typename T> auto Array<T>::crend() const noexcept { return rend(); }

template <typename T> typename Array<T>::reference Array<T>::at(size_type i) {
  if (!this->data) {
    throw std::logic_error("Cannot access element of uninitialized Array");
  }
  return this->data->at(i);
}

template <typename T>
typename Array<T>::const_reference Array<T>::at(size_type i) const {
  if (!this->data) {
    throw std::logic_error("Cannot access element of uninitialized Array");
  }
  return this->data->at(i);
}

template <typename T> typename Array<T>::reference Array<T>::front() noexcept {
  return (*this->data).front();
}

template <typename T>
typename Array<T>::const_reference Array<T>::front() const noexcept {
  return (*this->data).front();
}

template <typename T> typename Array<T>::reference Array<T>::back() noexcept {
  return (*this->data).back();
}

template <typename T>
typename Array<T>::const_reference Array<T>::back() const noexcept {
  return (*this->data).back();
}

template <typename T> bool Array<T>::empty() const noexcept {
  return !this->data || this->data->empty();
}

template <typename T>
typename Array<T>::size_type Array<T>::size() const noexcept {
  return this->data ? this->data->size() : 0;
}

template <typename T>
typename Array<T>::size_type Array<T>::max_size() const noexcept {
  return std::vector<T>().max_size();
}

template <typename T> void Array<T>::reserve(size_type new_cap) {
  if (!this->data) {
    this->data.emplace();
  }
  this->data->reserve(new_cap);
}

template <typename T>
typename Array<T>::size_type Array<T>::capacity() const noexcept {
  return this->data ? this->data->capacity() : 0;
}

template <typename T> void Array<T>::shrink_to_fit() {
  if (this->data) {
    this->data->shrink_to_fit();
  }
}

template <typename T> void Array<T>::clear() noexcept {
  if (this->data) {
    this->data->clear();
  }
}

template <typename T>
void Array<T>::resize(size_type count, const value_type &value) {
  if (!this->data) {
    this->data.emplace();
  }
  this->data->resize(count, value);
}

template <typename T>
bool operator==(const Array<T> &lhs, const Array<T> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T>
bool operator!=(const Array<T> &lhs, const Array<T> &rhs) {
  return !(lhs == rhs);
}

template <typename T> bool operator<(const Array<T> &lhs, const Array<T> &rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end());
}

template <typename T>
bool operator<=(const Array<T> &lhs, const Array<T> &rhs) {
  return !(rhs < lhs);
}

template <typename T> bool operator>(const Array<T> &lhs, const Array<T> &rhs) {
  return rhs < lhs;
}

template <typename T>
bool operator>=(const Array<T> &lhs, const Array<T> &rhs) {
  return !(lhs < rhs);
}

} // namespace np

#endif // NP_ARRAY_TPP
