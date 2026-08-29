/**
 * @file indexing.hpp
 * @brief Indexing routines (np.c_, np.r_, np.s_, np.index_exp, np.ix_, np.ndenumerate,
 * np.ndindex).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.indexing.html
 *
 * Provides C++ approximations of NumPy's indexing helper objects:
 *   c_ – column-stack helper (np.c_)
 *   r_ – row-stack/arange helper (np.r_)
 *   s_ – index expression builder (np.s_)
 *   index_exp – same as s_ (np.index_exp)
 *   ix_ – open mesh for indexing (np.ix_)
 *   ndenumerate – enumerate with multi-index
 *   ndindex – iterator over N-D index tuples
 *
 * These cannot replicate Python slice syntax (`0:5`) directly; instead they
 * accept `std::pair<int,int>` ranges and `ndarray` inputs.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_INDEXING_HPP
#define NP_INDEXING_HPP

#include <algorithm>
#include <optional>
#include <variant>
#include <vector>

#include "api_macros.hpp"
#include "creation.hpp"
#include "manipulation.hpp"
#include "ndarray.hpp"

namespace np
{

  // ── s_ / index_exp ────────────────────────────────────────────────
  /**
   * @brief Index expression holder (np.s_ / np.index_exp).
   *
   * In NumPy `np.s_[0:5, ::2]` returns a tuple of slices. Here we model
   * the index as a vector of `std::optional<std::pair<int,int>>` where
   * nullopt means `:` (full range). Users build it as `s_(0,5)` etc.
   *
   * Reference: numpy-reference/reference/generated/numpy.s_.html
   */
  struct IndexExp
  {
    std::vector<std::optional<std::pair<int, int>>> slices;

    IndexExp() = default;

    explicit IndexExp(std::initializer_list<std::optional<std::pair<int, int>>> list)
        : slices(list)
    {
    }

    template <typename... Args>
    explicit IndexExp(Args... args)
    {
      (slices.push_back(args), ...);
    }
  };

  inline constexpr struct SClass
  {
    template <typename... Args>
    IndexExp operator()(Args... args) const
    {
      return IndexExp{args...};
    }

    IndexExp operator[](std::optional<std::pair<int, int>> a) const
    {
      return IndexExp{a};
    }
  } s_{};

  inline constexpr IndexExp index_exp = IndexExp{};

  // ── c_ – column stack ────────────────────────────────────────────
  /**
   * @brief Column-stack helper (np.c_).
   *
   * Reference: numpy-reference/reference/generated/numpy.c_.html
   */
  struct CClass
  {
    template <typename T>
    auto operator()(const std::vector<ndarray<T>>& arrays) const -> ndarray<T>
    {
      return column_stack(arrays);
    }

    // Variadic helper: c_(a,b,c) -> column_stack({a,b,c})
    template <typename T, typename... Rest>
    auto operator()(const ndarray<T>& first, const Rest&... rest) const -> ndarray<T>
    {
      std::vector<ndarray<T>> v{first, rest...};
      return column_stack(v);
    }

    // Range helper: c_[0:5] approximated as arange
    template <typename T = int>
    auto range(int start, int stop, int step = 1) const -> ndarray<T>
    {
      return arange<T>(static_cast<T>(start), static_cast<T>(stop), static_cast<T>(step));
    }
  };

  inline constexpr CClass c_{};

  // ── r_ – row stack / arange ──────────────────────────────────────
  /**
   * @brief Row-stack helper (np.r_).
   *
   * Reference: numpy-reference/reference/generated/numpy.r_.html
   */
  struct RClass
  {
    template <typename T>
    auto operator()(const std::vector<ndarray<T>>& arrays) const -> ndarray<T>
    {
      // Row-wise: concatenate along first axis
      return concatenate(arrays, 0);
    }

    template <typename T, typename... Rest>
    auto operator()(const ndarray<T>& first, const Rest&... rest) const -> ndarray<T>
    {
      std::vector<ndarray<T>> v{first, rest...};
      return concatenate(v, 0);
    }

    template <typename T = int>
    auto range(int start, int stop, int step = 1) const -> ndarray<T>
    {
      return arange<T>(static_cast<T>(start), static_cast<T>(stop), static_cast<T>(step));
    }
  };

  inline constexpr RClass r_{};

  // ── ix_ – open mesh ─────────────────────────────────────────────
  /**
   * @brief Open mesh for indexing (np.ix_).
   *
   * Returns N arrays each with shape (1,...,n_i,...,1) suitable for
   * broadcasting indexing.
   *
   * Reference: numpy-reference/reference/generated/numpy.ix_.html
   */
  NP_API template <typename T>
  NP_NODISCARD inline auto ix_(const std::vector<ndarray<T>>& sequences)
      -> std::vector<ndarray<T>>
  {
    size_t N = sequences.size();
    std::vector<ndarray<T>> out;
    out.reserve(N);
    for (size_t i = 0; i < N; ++i)
    {
      if (sequences[i].ndim() != 1)
        throw std::invalid_argument("ix_: inputs must be 1-D");
      std::vector<int> shape(N, 1);
      shape[i] = static_cast<int>(sequences[i].size());
      out.push_back(sequences[i].reshape(shape));
    }
    return out;
  }

  // ── ndindex – iterator over N-D indices ──────────────────────────
  /**
   * @brief N-dimensional index iterator (np.ndindex).
   *
   * Reference: numpy-reference/reference/generated/numpy.ndindex.html
   */
  class ndindex
  {
  public:
    explicit ndindex(const std::vector<int>& shape)
        : shape_(shape), idx_(shape.size(), 0), done_(shape.empty())
    {
      if (shape.empty())
        done_ = true;
      else
      {
        for (int d : shape)
          if (d <= 0)
            done_ = true;
      }
    }

    bool has_next() const noexcept
    {
      return !done_;
    }

    std::vector<int> next()
    {
      if (done_)
        throw std::out_of_range("ndindex: no more indices");
      std::vector<int> cur = idx_;
      // advance
      for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d)
      {
        idx_[d]++;
        if (idx_[d] < shape_[d])
          break;
        idx_[d] = 0;
        if (d == 0)
          done_ = true;
      }
      return cur;
    }

    std::vector<int> shape() const
    {
      return shape_;
    }

  private:
    std::vector<int> shape_;
    std::vector<int> idx_;
    bool done_ = false;
  };

  // ── ndenumerate – enumerate with multi-index ─────────────────────
  /**
   * @brief Enumerate with multi-index (np.ndenumerate).
   *
   * Reference: numpy-reference/reference/generated/numpy.ndenumerate.html
   */
  template <typename T>
  class ndenumerate
  {
  public:
    explicit ndenumerate(const ndarray<T>& arr)
        : arr_(arr), idx_(arr.ndim(), 0), pos_(0), total_(arr.size()), done_(total_ == 0)
    {
    }

    bool has_next() const noexcept
    {
      return !done_;
    }

    std::pair<std::vector<int>, T> next()
    {
      if (done_)
        throw std::out_of_range("ndenumerate: no more");
      std::vector<int> cur = idx_;
      T val = arr_.get(std::vector<std::size_t>(idx_.begin(), idx_.end()));
      // advance multi-index
      for (int d = static_cast<int>(idx_.size()) - 1; d >= 0; --d)
      {
        idx_[d]++;
        if (idx_[d] < arr_.shape[d])
          break;
        idx_[d] = 0;
        if (d == 0)
        {
          // check if next would be beyond
          // we track pos_
        }
      }
      pos_++;
      if (pos_ >= total_)
        done_ = true;
      return {cur, val};
    }

  private:
    ndarray<T> arr_;
    std::vector<int> idx_;
    size_t pos_;
    size_t total_;
    bool done_;
  };

} // namespace np

#endif // NP_INDEXING_HPP
