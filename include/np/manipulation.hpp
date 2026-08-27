/**
 * @file manipulation.hpp
 * @brief Array manipulation routines (tile, flip, roll, split, delete, insert,
 * etc.).
 *
 * Implements NumPy's array manipulation functions:
 *   - Rearranging: flip, fliplr, flipud, roll, rot90
 *   - Tiling: tile, repeat (already in ndarray.hpp)
 *   - Splitting: split, array_split, hsplit, vsplit, dsplit
 *   - Adding/removing: delete, insert, append, trim_zeros, unique
 *   - Building matrices: diag, diagflat, tri, tril, triu, vander
 *   - Other: where, select, choose
 *
 * Reference: numpy-reference/reference/routines.array-manipulation.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MANIPULATION_HPP
#define NP_MANIPULATION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "creation.hpp"
#include "ndarray.hpp"
#include "api_macros.hpp"
#include "dtype.hpp"

namespace np
{
  // Rearranging Elements
  /**
   * @brief Reverse the order of elements along the given axis.
   *
   * Reference: numpy-reference/reference/generated/numpy.flip.html
   *
   * @tparam T Element type
   * @param arr Input array
   * @param axis Axis along which to flip (if nullopt, flips all axes)
   * @return Flipped array (view with reversed strides)
   */
  NP_API template <typename T>
  NP_NODISCARD auto flip(const ndarray<T>& arr, std::optional<int> axis = std::nullopt)
      -> ndarray<T>
  {
    ndarray<T> result = arr;

    if (!axis.has_value())
    {
      // Flip all axes
      for (std::size_t ax = 0; ax < result.ndim(); ++ax)
      {
        if (result.shape[ax] > 1)
        {
          result.offset += result.strides[ax] * (result.shape[ax] - 1);
          result.strides[ax] = -static_cast<std::ptrdiff_t>(result.strides[ax]);
        }
      }
    }
    else
    {
      // Normalize axis
      int ax = *axis;
      if (ax < 0)
      {
        ax += static_cast<int>(result.ndim());
      }
      if (ax < 0 || ax >= static_cast<int>(result.ndim()))
      {
        throw AxisError("axis " + std::to_string(*axis) + " is out of bounds");
      }

      // Flip specified axis
      if (result.shape[ax] > 1)
      {
        result.offset += result.strides[ax] * (result.shape[ax] - 1);
        result.strides[ax] = -static_cast<std::ptrdiff_t>(result.strides[ax]);
      }
    }

    return result;
  }

  /**
   * @brief Flip array left to right (horizontally).
   *
   * Reference: numpy-reference/reference/generated/numpy.fliplr.html
   *
   * @tparam T Element type
   * @param arr Input array (must be at least 2D)
   * @return Array with columns reversed
   */
  NP_API template <typename T>
  NP_NODISCARD auto fliplr(const ndarray<T>& arr) -> ndarray<T>
  {
    if (arr.ndim() < 2)
    {
      throw std::invalid_argument("fliplr requires at least 2 dimensions");
    }
    return flip(arr, 1);
  }

  /**
   * @brief Flip array up to down (vertically).
   *
   * Reference: numpy-reference/reference/generated/numpy.flipud.html
   *
   * @tparam T Element type
   * @param arr Input array (must be at least 1D)
   * @return Array with rows reversed
   */
  NP_API template <typename T>
  NP_NODISCARD auto flipud(const ndarray<T>& arr) -> ndarray<T>
  {
    if (arr.ndim() < 1)
    {
      throw std::invalid_argument("flipud requires at least 1 dimension");
    }
    return flip(arr, 0);
  }

  /**
   * @brief Roll array elements along a given axis.
   *
   * Reference: numpy-reference/reference/generated/numpy.roll.html
   *
   * @tparam T Element type
   * @param arr Input array
   * @param shift Number of places to shift
   * @param axis Axis along which to roll (if nullopt, flattens first)
   * @return Rolled array
   */
  NP_API template <typename T>
  NP_NODISCARD auto
  roll(const ndarray<T>& arr, int shift, std::optional<int> axis = std::nullopt)
      -> ndarray<T>
  {
    if (!axis.has_value())
    {
      // Flatten, roll, reshape
      auto flat = arr.ravel();
      int n = static_cast<int>(flat.size());
      shift = ((shift % n) + n) % n; // Normalize shift

      // Create result array with original shape
      ndarray<T> result(arr.shape);
      for (int i = 0; i < n; ++i)
      {
        result.data()[i] = flat((i - shift + n) % n);
      }
      return result;
    }

    // Normalize axis
    int ax = *axis;
    if (ax < 0)
    {
      ax += static_cast<int>(arr.ndim());
    }
    if (ax < 0 || ax >= static_cast<int>(arr.ndim()))
    {
      throw AxisError("axis " + std::to_string(*axis) + " is out of bounds");
    }

    ndarray<T> result = arr.copy();
    int n = result.shape[ax];
    shift = ((shift % n) + n) % n; // Normalize shift

    if (shift == 0 || n <= 1)
    {
      return result;
    }

    // Create index arrays for source and destination
    std::vector<std::size_t> idx(arr.ndim(), 0);

    // Helper to iterate through all indices
    auto roll_recursive = [&](auto& self, std::size_t dim) -> void
    {
      if (static_cast<int>(dim) == ax)
      {
        // At the rolling axis, perform the shift
        std::vector<T> temp(n);
        for (int i = 0; i < n; ++i)
        {
          idx[dim] = i;
          temp[i] = arr.get(idx);
        }
        for (int i = 0; i < n; ++i)
        {
          idx[dim] = (i + shift) % n;
          result.set(idx, temp[i]);
        }
        return;
      }

      if (dim >= arr.ndim())
      {
        return;
      }

      for (std::size_t i = 0; i < static_cast<std::size_t>(arr.shape[dim]); ++i)
      {
        idx[dim] = i;
        if (dim + 1 < arr.ndim())
        {
          self(self, dim + 1);
        }
        else
        {
          self(self, ax);
        }
      }
    };

    if (ax == 0)
    {
      roll_recursive(roll_recursive, 0);
    }
    else
    {
      roll_recursive(roll_recursive, 0);
    }

    return result;
  }

  /**
   * @brief Rotate array by 90 degrees.
   *
   * Reference: numpy-reference/reference/generated/numpy.rot90.html
   *
   * @tparam T Element type
   * @param arr Input array (must be at least 2D)
   * @param k Number of times to rotate by 90 degrees
   * @param axes Axes defining the plane of rotation
   * @return Rotated array
   */
  NP_API template <typename T>
  NP_NODISCARD auto
  rot90(const ndarray<T>& arr, int k = 1, const std::vector<int>& axes = {0, 1})
      -> ndarray<T>
  {
    if (arr.ndim() < 2)
    {
      throw std::invalid_argument("rot90 requires at least 2 dimensions");
    }

    if (axes.size() != 2)
    {
      throw std::invalid_argument("axes must have exactly 2 elements");
    }

    // Normalize k to [0, 3]
    k = ((k % 4) + 4) % 4;

    if (k == 0)
    {
      return arr.copy();
    }

    // Normalize axes
    std::vector<int> norm_axes = axes;
    for (auto& ax : norm_axes)
    {
      if (ax < 0)
      {
        ax += static_cast<int>(arr.ndim());
      }
      if (ax < 0 || ax >= static_cast<int>(arr.ndim()))
      {
        throw AxisError("axis out of bounds");
      }
    }

    if (norm_axes[0] == norm_axes[1])
    {
      throw std::invalid_argument("axes must be different");
    }

    ndarray<T> result = arr.copy();

    for (int i = 0; i < k; ++i)
    {
      // Transpose the two axes
      result = result.swapaxes(norm_axes[0], norm_axes[1]);
      // Flip the second axis
      result = flip(result, norm_axes[1]);
    }

    return result;
  }

  // Tiling Arrays
  /**
   * @brief Construct an array by repeating arr the number of times given by reps.
   *
   * Reference: numpy-reference/reference/generated/numpy.tile.html
   *
   * @tparam T Element type
   * @param arr Input array
   * @param reps Number of repetitions along each axis
   * @return Tiled array
   */
  NP_API template <typename T>
  NP_NODISCARD auto tile(const ndarray<T>& arr, const std::vector<int>& reps)
      -> ndarray<T>
  {
    if (reps.empty())
    {
      throw std::invalid_argument("reps cannot be empty");
    }

    // Expand dimensions if necessary
    std::vector<int> arr_shape = arr.shape;
    std::vector<int> tile_reps = reps;

    while (arr_shape.size() < tile_reps.size())
    {
      arr_shape.insert(arr_shape.begin(), 1);
    }
    while (tile_reps.size() < arr_shape.size())
    {
      tile_reps.insert(tile_reps.begin(), 1);
    }

    // Calculate result shape
    std::vector<int> result_shape(arr_shape.size());
    for (std::size_t i = 0; i < arr_shape.size(); ++i)
    {
      result_shape[i] = arr_shape[i] * tile_reps[i];
    }

    ndarray<T> result(result_shape);

    // Reshape input if dimensions were expanded
    ndarray<T> arr_expanded = arr.reshape(arr_shape);

    // Fill result with tiles
    std::vector<std::size_t> idx(result_shape.size(), 0);
    std::vector<std::size_t> src_idx(result_shape.size(), 0);

    auto tile_recursive = [&](auto& self, std::size_t dim) -> void
    {
      if (dim >= result_shape.size())
      {
        result.set(idx, arr_expanded.get(src_idx));
        return;
      }

      for (int i = 0; i < result_shape[dim]; ++i)
      {
        idx[dim] = i;
        src_idx[dim] = i % arr_shape[dim];
        self(self, dim + 1);
      }
    };

    tile_recursive(tile_recursive, 0);

    return result;
  }

  // Building Matrices
  /**
   * @brief Extract or construct a diagonal array.
   *
   * Reference: numpy-reference/reference/generated/numpy.diag.html
   *
   * @tparam T Element type
   * @param v Input array (1D or 2D)
   * @param k Diagonal offset (0=main diagonal, >0 above, <0 below)
   * @return Diagonal array
   */
  NP_API template <typename T>
  NP_NODISCARD auto diag(const ndarray<T>& v, int k = 0) -> ndarray<T>
  {
    if (v.ndim() == 1)
    {
      // Construct 2D array with v on diagonal
      int n = static_cast<int>(v.size());
      int size = n + std::abs(k);

      // Create 2D array with correct shape [size, size]
      std::vector<int> shape_vec = {size, size};
      ndarray<T> result(shape_vec);
      // Fill with zeros
      for (std::size_t idx = 0; idx < result.size(); ++idx)
      {
        result.data()[idx] = T{0};
      }

      // Set diagonal
      for (int i = 0; i < n; ++i)
      {
        if (k >= 0)
        {
          result(i, i + k) = v(i);
        }
        else
        {
          result(i - k, i) = v(i);
        }
      }

      return result;
    }
    else if (v.ndim() == 2)
    {
      // Extract diagonal
      int rows = v.shape[0];
      int cols = v.shape[1];
      int diag_size;

      if (k >= 0)
      {
        // Upper diagonal or main diagonal
        diag_size = std::min(rows, cols - k);
      }
      else
      {
        // Lower diagonal
        diag_size = std::min(rows + k, cols);
      }

      // Ensure non-negative
      diag_size = std::max(0, diag_size);

      if (diag_size == 0)
      {
        std::vector<int> empty_shape = {0};
        return ndarray<T>(empty_shape);
      }

      std::vector<int> result_shape = {diag_size};
      ndarray<T> result(result_shape);

      for (int i = 0; i < diag_size; ++i)
      {
        if (k >= 0)
        {
          result(i) = v(i, i + k);
        }
        else
        {
          result(i) = v(i - k, i);
        }
      }

      return result;
    }
    else
    {
      throw std::invalid_argument("diag requires 1D or 2D array");
    }
  }

  /**
   * @brief Create a 2D array with flattened input on the diagonal.
   *
   * Reference: numpy-reference/reference/generated/numpy.diagflat.html
   *
   * @tparam T Element type
   * @param v Input array (flattened before use)
   * @param k Diagonal offset
   * @return 2D array with v on diagonal
   */
  NP_API template <typename T>
  NP_NODISCARD auto diagflat(const ndarray<T>& v, int k = 0) -> ndarray<T>
  {
    auto flat = v.ravel();
    return diag(flat, k);
  }

  /**
   * @brief Array with ones at and below the given diagonal and zeros elsewhere.
   *
   * Reference: numpy-reference/reference/generated/numpy.tri.html
   *
   * @tparam T Element type
   * @param n Number of rows
   * @param m Number of columns (default: n)
   * @param k Diagonal offset (0=main, >0 above, <0 below)
   * @return Lower triangular array
   */
  NP_API template <typename T = double>
  NP_NODISCARD auto tri(int n, int m = -1, int k = 0) -> ndarray<T>
  {
    if (m < 0)
    {
      m = n;
    }

    ndarray<T> result = zeros<T>({n, m});

    for (int i = 0; i < n; ++i)
    {
      for (int j = 0; j < m; ++j)
      {
        if (j <= i + k)
        {
          result(i, j) = T{1};
        }
      }
    }

    return result;
  }

  /**
   * @brief Lower triangle of an array.
   *
   * Reference: numpy-reference/reference/generated/numpy.tril.html
   *
   * @tparam T Element type
   * @param arr Input array (must be at least 2D)
   * @param k Diagonal offset
   * @return Array with elements above kth diagonal zeroed
   */
  NP_API template <typename T>
  NP_NODISCARD auto tril(const ndarray<T>& arr, int k = 0) -> ndarray<T>
  {
    if (arr.ndim() < 2)
    {
      throw std::invalid_argument("tril requires at least 2 dimensions");
    }

    ndarray<T> result = arr.copy();
    int rows = result.shape[result.ndim() - 2];
    int cols = result.shape[result.ndim() - 1];

    // Create base indices
    std::vector<std::size_t> idx(result.ndim(), 0);

    auto apply_tril = [&](auto& self, std::size_t dim) -> void
    {
      if (dim == result.ndim() - 2)
      {
        // At the row dimension
        for (int i = 0; i < rows; ++i)
        {
          idx[dim] = i;
          for (int j = 0; j < cols; ++j)
          {
            idx[dim + 1] = j;
            if (j > i + k)
            {
              result.set(idx, T{0});
            }
          }
        }
        return;
      }

      for (std::size_t i = 0; i < static_cast<std::size_t>(result.shape[dim]); ++i)
      {
        idx[dim] = i;
        self(self, dim + 1);
      }
    };

    if (result.ndim() == 2)
    {
      apply_tril(apply_tril, 0);
    }
    else
    {
      apply_tril(apply_tril, 0);
    }

    return result;
  }

  /**
   * @brief Upper triangle of an array.
   *
   * Reference: numpy-reference/reference/generated/numpy.triu.html
   *
   * @tparam T Element type
   * @param arr Input array (must be at least 2D)
   * @param k Diagonal offset
   * @return Array with elements below kth diagonal zeroed
   */
  NP_API template <typename T>
  NP_NODISCARD auto triu(const ndarray<T>& arr, int k = 0) -> ndarray<T>
  {
    if (arr.ndim() < 2)
    {
      throw std::invalid_argument("triu requires at least 2 dimensions");
    }

    ndarray<T> result = arr.copy();
    int rows = result.shape[result.ndim() - 2];
    int cols = result.shape[result.ndim() - 1];

    std::vector<std::size_t> idx(result.ndim(), 0);

    auto apply_triu = [&](auto& self, std::size_t dim) -> void
    {
      if (dim == result.ndim() - 2)
      {
        for (int i = 0; i < rows; ++i)
        {
          idx[dim] = i;
          for (int j = 0; j < cols; ++j)
          {
            idx[dim + 1] = j;
            if (j < i + k)
            {
              result.set(idx, T{0});
            }
          }
        }
        return;
      }

      for (std::size_t i = 0; i < static_cast<std::size_t>(result.shape[dim]); ++i)
      {
        idx[dim] = i;
        self(self, dim + 1);
      }
    };

    if (result.ndim() == 2)
    {
      apply_triu(apply_triu, 0);
    }
    else
    {
      apply_triu(apply_triu, 0);
    }

    return result;
  }

  /**
   * @brief Generate a Vandermonde matrix.
   *
   * Reference: numpy-reference/reference/generated/numpy.vander.html
   *
   * @tparam T Element type
   * @param x Input 1D array
   * @param n Number of columns (default: x.size())
   * @param increasing Order of powers (false: decreasing, true: increasing)
   * @return Vandermonde matrix
   */
  NP_API template <typename T>
  NP_NODISCARD auto vander(const ndarray<T>& x, int n = -1, bool increasing = false)
      -> ndarray<T>
  {
    if (x.ndim() != 1)
    {
      throw std::invalid_argument("vander requires 1D array");
    }

    int rows = static_cast<int>(x.size());
    if (n < 0)
    {
      n = rows;
    }

    ndarray<T> result(std::vector<int>{rows, n});

    for (int i = 0; i < rows; ++i)
    {
      T val = x(i);
      for (int j = 0; j < n; ++j)
      {
        int power = increasing ? j : (n - 1 - j);
        T powered = T{1};
        for (int p = 0; p < power; ++p)
        {
          powered *= val;
        }
        result(i, j) = powered;
      }
    }

    return result;
  }

  // Splitting Arrays
  /**
   * @brief Split array into multiple sub-arrays.
   *
   * Reference: numpy-reference/reference/generated/numpy.split.html
   *
   * @tparam T Element type
   * @param arr Input array
   * @param indices_or_sections Indices where to split, or number of equal
   * sections
   * @param axis Axis along which to split
   * @return Vector of sub-arrays
   */
  NP_API template <typename T>
  NP_NODISCARD auto
  split(const ndarray<T>& arr, const std::vector<int>& indices_or_sections, int axis = 0)
      -> std::vector<ndarray<T>>
  {
    // Normalize axis
    if (axis < 0)
    {
      axis += static_cast<int>(arr.ndim());
    }
    if (axis < 0 || axis >= static_cast<int>(arr.ndim()))
    {
      throw AxisError("axis out of bounds");
    }

    int n = arr.shape[axis];
    std::vector<ndarray<T>> result;

    if (indices_or_sections.size() == 1 && indices_or_sections[0] > 0)
    {
      // Split into equal sections
      int sections = indices_or_sections[0];
      if (n % sections != 0)
      {
        throw std::invalid_argument("array split does not result in equal division");
      }

      int section_size = n / sections;
      for (int i = 0; i < sections; ++i)
      {
        std::vector<int> new_shape = arr.shape;
        new_shape[axis] = section_size;
        ndarray<T> sub(new_shape);

        // Copy data
        std::vector<std::size_t> src_idx(arr.ndim(), 0);
        std::vector<std::size_t> dst_idx(arr.ndim(), 0);

        auto copy_section = [&](auto& self, std::size_t dim) -> void
        {
          if (dim >= arr.ndim())
          {
            sub.set(dst_idx, arr.get(src_idx));
            return;
          }

          if (dim == static_cast<std::size_t>(axis))
          {
            for (int j = 0; j < section_size; ++j)
            {
              src_idx[dim] = i * section_size + j;
              dst_idx[dim] = j;
              self(self, dim + 1);
            }
          }
          else
          {
            for (std::size_t j = 0; j < static_cast<std::size_t>(arr.shape[dim]); ++j)
            {
              src_idx[dim] = j;
              dst_idx[dim] = j;
              self(self, dim + 1);
            }
          }
        };

        copy_section(copy_section, 0);
        result.push_back(sub);
      }
    }
    else
    {
      // Split at indices
      std::vector<int> split_points = {0};
      split_points.insert(
          split_points.end(), indices_or_sections.begin(), indices_or_sections.end());
      split_points.push_back(n);

      for (std::size_t i = 0; i < split_points.size() - 1; ++i)
      {
        int start = split_points[i];
        int end = split_points[i + 1];

        if (start < 0 || end > n || start > end)
        {
          throw std::invalid_argument("invalid split indices");
        }

        std::vector<int> new_shape = arr.shape;
        new_shape[axis] = end - start;
        ndarray<T> sub(new_shape);

        std::vector<std::size_t> src_idx(arr.ndim(), 0);
        std::vector<std::size_t> dst_idx(arr.ndim(), 0);

        auto copy_section = [&](auto& self, std::size_t dim) -> void
        {
          if (dim >= arr.ndim())
          {
            sub.set(dst_idx, arr.get(src_idx));
            return;
          }

          if (dim == static_cast<std::size_t>(axis))
          {
            for (int j = start; j < end; ++j)
            {
              src_idx[dim] = j;
              dst_idx[dim] = j - start;
              self(self, dim + 1);
            }
          }
          else
          {
            for (std::size_t j = 0; j < static_cast<std::size_t>(arr.shape[dim]); ++j)
            {
              src_idx[dim] = j;
              dst_idx[dim] = j;
              self(self, dim + 1);
            }
          }
        };

        copy_section(copy_section, 0);
        result.push_back(sub);
      }
    }

    return result;
  }

  /**
   * @brief Split array into approximately equal pieces.
   *
   * Reference: numpy-reference/reference/generated/numpy.array_split.html
   */
  NP_API template <typename T>
  NP_NODISCARD auto array_split(const ndarray<T>& arr, int sections, int axis = 0)
      -> std::vector<ndarray<T>>
  {
    if (axis < 0)
    {
      axis += static_cast<int>(arr.ndim());
    }
    if (axis < 0 || axis >= static_cast<int>(arr.ndim()))
    {
      throw AxisError("axis out of bounds");
    }

    int n = arr.shape[axis];
    int base_size = n / sections;
    int remainder = n % sections;

    std::vector<int> split_points;
    int pos = 0;
    for (int i = 0; i < sections; ++i)
    {
      int size = base_size + (i < remainder ? 1 : 0);
      pos += size;
      if (pos < n)
      {
        split_points.push_back(pos);
      }
    }

    return split(arr, split_points, axis);
  }

  /**
   * @brief Split array into multiple sub-arrays horizontally.
   */
  NP_API template <typename T>
  NP_NODISCARD auto
  hsplit(const ndarray<T>& arr, const std::vector<int>& indices_or_sections)
      -> std::vector<ndarray<T>>
  {
    if (arr.ndim() == 0)
    {
      throw std::invalid_argument("hsplit requires at least 1D array");
    }
    int axis = arr.ndim() == 1 ? 0 : 1;
    return split(arr, indices_or_sections, axis);
  }

  /**
   * @brief Split array into multiple sub-arrays vertically.
   */
  NP_API template <typename T>
  NP_NODISCARD auto
  vsplit(const ndarray<T>& arr, const std::vector<int>& indices_or_sections)
      -> std::vector<ndarray<T>>
  {
    if (arr.ndim() < 2)
    {
      throw std::invalid_argument("vsplit requires at least 2D array");
    }
    return split(arr, indices_or_sections, 0);
  }

  /**
   * @brief Split array into multiple sub-arrays along 3rd axis.
   */
  NP_API template <typename T>
  NP_NODISCARD auto
  dsplit(const ndarray<T>& arr, const std::vector<int>& indices_or_sections)
      -> std::vector<ndarray<T>>
  {
    if (arr.ndim() < 3)
    {
      throw std::invalid_argument("dsplit requires at least 3D array");
    }
    return split(arr, indices_or_sections, 2);
  }

  // Adding/Removing Elements
  /**
   * @brief Return a new array with sub-arrays along an axis deleted.
   *
   * Reference: numpy-reference/reference/generated/numpy.delete.html
   *
   * @tparam T Element type
   * @param arr Input array
   * @param indices Indices of sub-arrays to remove
   * @param axis Axis along which to delete (if nullopt, flattens first)
   * @return Array with deleted elements
   */
  NP_API template <typename T>
  NP_NODISCARD auto delete_arr(
      const ndarray<T>& arr,
      const std::vector<int>& indices,
      std::optional<int> axis = std::nullopt) -> ndarray<T>
  {
    if (indices.empty())
    {
      return arr.copy();
    }

    if (!axis.has_value())
    {
      // Flatten and delete
      auto flat = arr.ravel();
      int n = static_cast<int>(flat.size());

      // Normalize and sort indices
      std::set<int> to_delete;
      for (int idx : indices)
      {
        int norm_idx = idx < 0 ? idx + n : idx;
        if (norm_idx >= 0 && norm_idx < n)
        {
          to_delete.insert(norm_idx);
        }
      }

      ndarray<T> result(std::vector<int>{n - static_cast<int>(to_delete.size())});
      int j = 0;
      for (int i = 0; i < n; ++i)
      {
        if (to_delete.find(i) == to_delete.end())
        {
          result(j++) = flat(i);
        }
      }

      return result;
    }

    // Normalize axis
    int ax = *axis;
    if (ax < 0)
    {
      ax += static_cast<int>(arr.ndim());
    }
    if (ax < 0 || ax >= static_cast<int>(arr.ndim()))
    {
      throw AxisError("axis out of bounds");
    }

    int n = arr.shape[ax];

    // Normalize and sort indices
    std::set<int> to_delete;
    for (int idx : indices)
    {
      int norm_idx = idx < 0 ? idx + n : idx;
      if (norm_idx >= 0 && norm_idx < n)
      {
        to_delete.insert(norm_idx);
      }
    }

    std::vector<int> new_shape = arr.shape;
    new_shape[ax] = n - static_cast<int>(to_delete.size());

    ndarray<T> result(new_shape);

    std::vector<std::size_t> src_idx(arr.ndim(), 0);
    std::vector<std::size_t> dst_idx(arr.ndim(), 0);

    auto copy_without = [&](auto& self, std::size_t dim) -> void
    {
      if (dim >= arr.ndim())
      {
        result.set(dst_idx, arr.get(src_idx));
        return;
      }

      if (dim == static_cast<std::size_t>(ax))
      {
        int dst_i = 0;
        for (int i = 0; i < n; ++i)
        {
          if (to_delete.find(i) == to_delete.end())
          {
            src_idx[dim] = i;
            dst_idx[dim] = dst_i++;
            self(self, dim + 1);
          }
        }
      }
      else
      {
        for (std::size_t i = 0; i < static_cast<std::size_t>(arr.shape[dim]); ++i)
        {
          src_idx[dim] = i;
          dst_idx[dim] = i;
          self(self, dim + 1);
        }
      }
    };

    copy_without(copy_without, 0);

    return result;
  }

  /**
   * @brief Insert values along the given axis before the given indices.
   *
   * Reference: numpy-reference/reference/generated/numpy.insert.html
   *
   * @tparam T Element type
   * @param arr Input array
   * @param indices Indices before which to insert
   * @param values Values to insert
   * @param axis Axis along which to insert (if nullopt, flattens first)
   * @return Array with inserted values
   */
  NP_API template <typename T>
  NP_NODISCARD auto insert(
      const ndarray<T>& arr,
      const std::vector<int>& indices,
      const ndarray<T>& values,
      std::optional<int> axis = std::nullopt) -> ndarray<T>
  {
    if (indices.empty())
    {
      return arr.copy();
    }

    if (!axis.has_value())
    {
      // Flatten and insert
      auto flat = arr.ravel();
      auto val_flat = values.ravel();
      int n = static_cast<int>(flat.size());

      // Build (position, value, order) list: value j goes before
      // indices[j % indices.size()]; equal positions keep value order.
      std::vector<std::tuple<int, T, std::size_t>> inserts;
      for (std::size_t j = 0; j < val_flat.size(); ++j)
      {
        int raw = indices[j % indices.size()];
        int idx = raw < 0 ? raw + n + 1 : raw;
        idx = std::max(0, std::min(n, idx));
        inserts.emplace_back(idx, val_flat(j), j);
      }
      std::sort(
          inserts.begin(),
          inserts.end(),
          [](const auto& x, const auto& y)
          {
            if (std::get<0>(x) != std::get<0>(y))
              return std::get<0>(x) < std::get<0>(y);
            return std::get<2>(x) < std::get<2>(y);
          });

      ndarray<T> result(std::vector<int>{n + static_cast<int>(val_flat.size())});
      int k = 0;
      std::size_t v = 0;
      for (int i = 0; i <= n; ++i)
      {
        while (v < inserts.size() && std::get<0>(inserts[v]) == i)
        {
          result(k++) = std::get<1>(inserts[v++]);
        }
        if (i < n)
        {
          result(k++) = flat(i);
        }
      }

      return result;
    }

    throw std::runtime_error("insert with axis parameter not yet fully implemented");
  }

  /**
   * @brief Append values to the end of an array.
   *
   * Reference: numpy-reference/reference/generated/numpy.append.html
   *
   * @tparam T Element type
   * @param arr Input array
   * @param values Values to append
   * @param axis Axis along which to append (if nullopt, flattens both)
   * @return Array with appended values
   */
  NP_API template <typename T>
  NP_NODISCARD auto append(
      const ndarray<T>& arr,
      const ndarray<T>& values,
      std::optional<int> axis = std::nullopt) -> ndarray<T>
  {
    if (!axis.has_value())
    {
      // Flatten both and concatenate
      auto arr_flat = arr.ravel();
      auto val_flat = values.ravel();

      ndarray<T> result(
          std::vector<int>{static_cast<int>(arr_flat.size() + val_flat.size())});

      for (std::size_t i = 0; i < arr_flat.size(); ++i)
      {
        result(i) = arr_flat(i);
      }
      for (std::size_t i = 0; i < val_flat.size(); ++i)
      {
        result(arr_flat.size() + i) = val_flat(i);
      }

      return result;
    }

    // Use concatenate for axis version (requires concatenate.hpp)
    throw std::runtime_error("append with axis requires concatenate.hpp to be included");
  }

  /**
   * @brief Trim the leading and/or trailing zeros from a 1D array.
   *
   * Reference: numpy-reference/reference/generated/numpy.trim_zeros.html
   *
   * @tparam T Element type
   * @param arr Input 1D array
   * @param trim String with 'f' for leading, 'b' for trailing
   * @return Trimmed array
   */
  NP_API template <typename T>
  NP_NODISCARD auto trim_zeros(const ndarray<T>& arr, const std::string& trim = "fb")
      -> ndarray<T>
  {
    if (arr.ndim() != 1)
    {
      throw std::invalid_argument("trim_zeros requires 1D array");
    }

    int n = static_cast<int>(arr.size());
    int start = 0;
    int end = n;

    if (trim.find('f') != std::string::npos)
    {
      while (start < n && arr(start) == T{0})
      {
        ++start;
      }
    }

    if (trim.find('b') != std::string::npos)
    {
      while (end > start && arr(end - 1) == T{0})
      {
        --end;
      }
    }

    if (start >= end)
    {
      return ndarray<T>(std::vector<int>{0});
    }

    ndarray<T> result(std::vector<int>{end - start});
    for (int i = start; i < end; ++i)
    {
      result(i - start) = arr(i);
    }

    return result;
  }

  /**
   * @brief Find the unique elements of an array.
   *
   * Reference: numpy-reference/reference/generated/numpy.unique.html
   *
   * @tparam T Element type
   * @param arr Input array
   * @param return_index If true, also return indices of first occurrences
   * @param return_inverse If true, also return indices to reconstruct input
   * @param return_counts If true, also return counts of each unique element
   * @return Unique sorted elements (and optionally indices, inverse, counts)
   */
  NP_API template <typename T>
  NP_NODISCARD auto unique(
      const ndarray<T>& arr,
      bool return_index = false,
      bool return_inverse = false,
      bool return_counts = false) -> std::
      tuple<ndarray<T>, ndarray<std::size_t>, ndarray<std::size_t>, ndarray<std::size_t>>
  {
    auto flat = arr.ravel();
    std::size_t n = flat.size();

    // Create vector of (value, original_index) pairs
    std::vector<std::pair<T, std::size_t>> pairs;
    for (std::size_t i = 0; i < n; ++i)
    {
      pairs.push_back({flat(i), i});
    }

    // Sort by value
    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // Find unique values
    std::vector<T> unique_vals;
    std::vector<std::size_t> unique_indices;
    std::vector<std::size_t> inverse_indices(n);
    std::vector<std::size_t> counts;

    if (!pairs.empty())
    {
      unique_vals.push_back(pairs[0].first);
      unique_indices.push_back(pairs[0].second);
      std::size_t count = 1;

      for (std::size_t i = 1; i < pairs.size(); ++i)
      {
        if (pairs[i].first != pairs[i - 1].first)
        {
          counts.push_back(count);
          unique_vals.push_back(pairs[i].first);
          unique_indices.push_back(pairs[i].second);
          count = 1;
        }
        else
        {
          ++count;
        }
      }
      counts.push_back(count);

      // Build inverse mapping
      std::size_t unique_idx = 0;
      for (std::size_t i = 0; i < pairs.size(); ++i)
      {
        if (i > 0 && pairs[i].first != pairs[i - 1].first)
        {
          ++unique_idx;
        }
        inverse_indices[pairs[i].second] = unique_idx;
      }
    }

    // Build result arrays
    ndarray<T> result_vals(std::vector<int>{static_cast<int>(unique_vals.size())});
    for (std::size_t i = 0; i < unique_vals.size(); ++i)
    {
      result_vals(static_cast<int>(i)) = unique_vals[i];
    }

    int idx_size = return_index ? static_cast<int>(unique_indices.size()) : 0;
    int inv_size = return_inverse ? static_cast<int>(inverse_indices.size()) : 0;
    int cnt_size = return_counts ? static_cast<int>(counts.size()) : 0;

    ndarray<std::size_t> result_index(std::vector<int>{idx_size});
    if (return_index)
    {
      for (std::size_t i = 0; i < unique_indices.size(); ++i)
      {
        result_index(static_cast<int>(i)) = unique_indices[i];
      }
    }

    ndarray<std::size_t> result_inverse(std::vector<int>{inv_size});
    if (return_inverse)
    {
      for (std::size_t i = 0; i < inverse_indices.size(); ++i)
      {
        result_inverse(static_cast<int>(i)) = inverse_indices[i];
      }
    }

    ndarray<std::size_t> result_counts(std::vector<int>{cnt_size});
    if (return_counts)
    {
      for (std::size_t i = 0; i < counts.size(); ++i)
      {
        result_counts(static_cast<int>(i)) = counts[i];
      }
    }

    return {result_vals, result_index, result_inverse, result_counts};
  }

  // Conditional Selection
  /**
   * @brief Return elements chosen from x or y depending on condition.
   *
   * Reference: numpy-reference/reference/generated/numpy.where.html
   *
   * @tparam T Element type
   * @param condition Boolean array
   * @param x Values where condition is true
   * @param y Values where condition is false
   * @return Array with elements from x where condition, y elsewhere
   */
  NP_API template <typename T>
  NP_NODISCARD auto
  where(const ndarray<bool>& condition, const ndarray<T>& x, const ndarray<T>& y)
      -> ndarray<T>
  {
    // Broadcasting check (simplified)
    if (condition.shape != x.shape || condition.shape != y.shape)
    {
      throw std::invalid_argument("where: arrays must have compatible shapes");
    }

    ndarray<T> result(x.shape);

    for (std::size_t i = 0; i < x.size(); ++i)
    {
      result.data()[i] = condition.data()[i] ? x.data()[i] : y.data()[i];
    }

    return result;
  }

  /**
   * @brief Return indices where condition is true.
   *
   * Reference: numpy-reference/reference/generated/numpy.where.html (single
   * argument form)
   *
   * @param condition Boolean array
   * @return Tuple of arrays, one for each dimension
   */
  inline auto where(const ndarray<bool>& condition) -> std::vector<ndarray<std::size_t>>
  {
    auto indices = condition.nonzero();
    return indices;
  }

  /**
   * @brief Broadcast array to a new shape.
   *
   * Mirrors `np.broadcast_to`. The new shape must be broadcast-compatible
   * with the input shape.
   *
   * @tparam T Element type.
   * @param arr Input array.
   * @param shape Target shape.
   * @return View/copy broadcast to shape.
   *
   * Reference: numpy-reference/reference/generated/numpy.broadcast_to.html
   */
  NP_API template <typename T>
  NP_NODISCARD auto broadcast_to(const ndarray<T>& arr, const std::vector<int>& shape)
      -> ndarray<T>
  {
    // Validate broadcast compatibility via detail::broadcast_shapes
    (void)detail::broadcast_shapes(arr.shape, shape);
    ndarray<T> out(shape);
    detail::Odometer od(shape);
    while (!od.done())
    {
      const auto& idx = od.idx();
      // Map output index to input via broadcast rules
      std::vector<std::size_t> src_idx(arr.ndim(), 0);
      std::size_t out_dim = shape.size();
      std::size_t in_dim = arr.ndim();
      for (std::size_t d = 0; d < out_dim; ++d)
      {
        std::ptrdiff_t in_d = static_cast<std::ptrdiff_t>(d)
            - static_cast<std::ptrdiff_t>(out_dim - in_dim);
        if (in_d < 0)
          continue;
        if (arr.shape[static_cast<std::size_t>(in_d)] == 1)
          src_idx[static_cast<std::size_t>(in_d)] = 0;
        else
          src_idx[static_cast<std::size_t>(in_d)] = idx[d];
      }
      out.set(idx, arr.get(src_idx));
      od.advance();
    }
    return out;
  }

  /**
   * @brief Expand dimensions of an array.
   *
   * Inserts a new axis at `axis` (mirrors `np.expand_dims`).
   *
   * @tparam T Element type.
   * @param arr Input array.
   * @param axis Position of new axis (may be negative).
   * @return View with expanded shape.
   */
  NP_API template <typename T>
  NP_NODISCARD auto expand_dims(const ndarray<T>& arr, int axis) -> ndarray<T>
  {
    int nd = static_cast<int>(arr.ndim());
    if (axis < 0)
      axis += nd + 1;
    if (axis < 0 || axis > nd)
      throw AxisError("expand_dims: axis out of bounds");
    std::vector<int> new_shape = arr.shape;
    new_shape.insert(new_shape.begin() + axis, 1);
    return arr.reshape(new_shape);
  }

  /**
   * @brief Ensure array is at least 1-D.
   *
   * Mirrors `np.atleast_1d`. 0-D arrays become 1-D with single element.
   */
  NP_API template <typename T>
  NP_NODISCARD auto atleast_1d(const ndarray<T>& arr) -> ndarray<T>
  {
    if (arr.ndim() >= 1)
      return arr.copy();
    ndarray<T> out(std::vector<int>{1});
    out.data()[0] = arr.item();
    return out;
  }

  /**
   * @brief Ensure array is at least 2-D.
   *
   * Mirrors `np.atleast_2d`. 1-D becomes (1, N).
   */
  NP_API template <typename T>
  NP_NODISCARD auto atleast_2d(const ndarray<T>& arr) -> ndarray<T>
  {
    if (arr.ndim() >= 2)
      return arr.copy();
    if (arr.ndim() == 1)
    {
      return arr.reshape(std::vector<int>{1, static_cast<int>(arr.size())});
    }
    // 0-D -> (1,1)
    ndarray<T> out(std::vector<int>{1, 1});
    out.data()[0] = arr.item();
    return out;
  }

  /**
   * @brief Ensure array is at least 3-D.
   *
   * Mirrors `np.atleast_3d`. 1-D (N,) -> (1,N,1), 2-D (M,N) -> (M,N,1).
   */
  NP_API template <typename T>
  NP_NODISCARD auto atleast_3d(const ndarray<T>& arr) -> ndarray<T>
  {
    if (arr.ndim() >= 3)
      return arr.copy();
    if (arr.ndim() == 2)
    {
      return arr.reshape(std::vector<int>{arr.shape[0], arr.shape[1], 1});
    }
    if (arr.ndim() == 1)
    {
      return arr.reshape(std::vector<int>{1, static_cast<int>(arr.size()), 1});
    }
    ndarray<T> out(std::vector<int>{1, 1, 1});
    out.data()[0] = arr.item();
    return out;
  }

  /** @brief Move axis to new position (np.moveaxis). */
  NP_API template <typename T>
  NP_NODISCARD auto moveaxis(const ndarray<T>& a, int source, int destination)
      -> ndarray<T>
  {
    int nd = static_cast<int>(a.ndim());
    if (source < 0)
      source += nd;
    if (destination < 0)
      destination += nd;
    if (source < 0 || source >= nd || destination < 0 || destination >= nd)
      throw AxisError("moveaxis: axis out of bounds");
    std::vector<int> perm(nd);
    for (int i = 0; i < nd; ++i)
      perm[i] = i;
    int val = perm[source];
    perm.erase(perm.begin() + source);
    perm.insert(perm.begin() + destination, val);
    return a.transpose(perm);
  }

  /** @brief Roll axis to start position (np.rollaxis). */
  NP_API template <typename T>
  NP_NODISCARD auto rollaxis(const ndarray<T>& a, int axis, int start = 0) -> ndarray<T>
  {
    int nd = static_cast<int>(a.ndim());
    if (axis < 0)
      axis += nd;
    if (start < 0)
      start += nd;
    if (axis < 0 || axis >= nd || start < 0 || start > nd)
      throw AxisError("rollaxis: axis out of bounds");
    if (axis == start)
      return a.copy();
    // Equivalent to moveaxis with destination handling for start > axis case
    int dest = start;
    if (axis < start)
      dest = start - 1;
    return moveaxis(a, axis, dest);
  }

  /** @brief Stack 1-D arrays as columns (np.column_stack). */
  NP_API template <typename T>
  NP_NODISCARD auto column_stack(const std::vector<ndarray<T>>& tup) -> ndarray<T>
  {
    if (tup.empty())
      throw std::invalid_argument("column_stack: need at least one array");
    // If all 1-D, stack as columns -> shape (N, K)
    bool all1d = true;
    for (auto& arr : tup)
      if (arr.ndim() != 1)
        all1d = false;
    if (all1d)
    {
      int n = tup[0].shape[0];
      for (auto& arr : tup)
        if (arr.shape[0] != n)
          throw std::invalid_argument("column_stack: 1-D arrays must have same length");
      ndarray<T> out(std::vector<int>{n, static_cast<int>(tup.size())});
      for (std::size_t k = 0; k < tup.size(); ++k)
        for (int i = 0; i < n; ++i)
          out.at(static_cast<std::size_t>(i), k) = tup[k].at(static_cast<std::size_t>(i));
      return out;
    }
    // Otherwise hstack
    // Fallback to hstack via concatenate along axis 1 (requires same rows)
    int rows = tup[0].shape[0];
    for (auto& arr : tup)
      if (arr.shape[0] != rows)
        throw std::invalid_argument(
            "column_stack: arrays must have same first dimension");
    // Concatenate along last axis (1 for 2-D)
    int total_cols = 0;
    for (auto& arr : tup)
      total_cols += arr.shape[1];
    ndarray<T> out(std::vector<int>{rows, total_cols});
    int col_off = 0;
    for (auto& arr : tup)
    {
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < arr.shape[1]; ++j)
          out.at(static_cast<std::size_t>(i), static_cast<std::size_t>(col_off + j)) =
              arr.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
      col_off += arr.shape[1];
    }
    return out;
  }

  /** @brief Stack arrays vertically (np.row_stack / vstack). */
  NP_API template <typename T>
  NP_NODISCARD auto row_stack(const std::vector<ndarray<T>>& tup) -> ndarray<T>
  {
    if (tup.empty())
      throw std::invalid_argument("row_stack: need at least one array");
    // Promote 1-D to 2-D row (1, N) then vstack
    std::vector<ndarray<T>> tmp;
    tmp.reserve(tup.size());
    for (auto& arr : tup)
    {
      if (arr.ndim() == 1)
        tmp.push_back(arr.reshape(std::vector<int>{1, arr.shape[0]}));
      else
        tmp.push_back(arr);
    }
    int cols = tmp[0].shape[1];
    for (auto& arr : tmp)
      if (arr.shape[1] != cols)
        throw std::invalid_argument("row_stack: arrays must have same number of columns");
    int total_rows = 0;
    for (auto& arr : tmp)
      total_rows += arr.shape[0];
    ndarray<T> out(std::vector<int>{total_rows, cols});
    int row_off = 0;
    for (auto& arr : tmp)
    {
      for (int i = 0; i < arr.shape[0]; ++i)
        for (int j = 0; j < cols; ++j)
          out.at(static_cast<std::size_t>(row_off + i), static_cast<std::size_t>(j)) =
              arr.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
      row_off += arr.shape[0];
    }
    return out;
  }

  /** @brief Assemble array from blocks (np.block) – 2-D case. */
  NP_API template <typename T>
  NP_NODISCARD auto block(const std::vector<std::vector<ndarray<T>>>& blocks)
      -> ndarray<T>
  {
    if (blocks.empty() || blocks[0].empty())
      throw std::invalid_argument("block: need at least one block");
    // First, hstack each row, then vstack rows
    std::vector<ndarray<T>> row_stacked;
    row_stacked.reserve(blocks.size());
    for (auto& row : blocks)
    {
      if (row.empty())
        throw std::invalid_argument("block: empty row");
      int h = row[0].shape[0];
      for (auto& b : row)
        if (b.shape[0] != h)
          throw std::invalid_argument("block: blocks in row must have same rows");
      int total_w = 0;
      for (auto& b : row)
        total_w += b.shape[1];
      ndarray<T> r(std::vector<int>{h, total_w});
      int col_off = 0;
      for (auto& b : row)
      {
        for (int i = 0; i < h; ++i)
          for (int j = 0; j < b.shape[1]; ++j)
            r.at(static_cast<std::size_t>(i), static_cast<std::size_t>(col_off + j)) =
                b.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        col_off += b.shape[1];
      }
      row_stacked.push_back(std::move(r));
    }
    // vstack
    int total_h = 0;
    int w = row_stacked[0].shape[1];
    for (auto& r : row_stacked)
    {
      if (r.shape[1] != w)
        throw std::invalid_argument("block: rows must have same width");
      total_h += r.shape[0];
    }
    ndarray<T> out(std::vector<int>{total_h, w});
    int row_off = 0;
    for (auto& r : row_stacked)
    {
      for (int i = 0; i < r.shape[0]; ++i)
        for (int j = 0; j < w; ++j)
          out.at(static_cast<std::size_t>(row_off + i), static_cast<std::size_t>(j)) =
              r.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
      row_off += r.shape[0];
    }
    return out;
  }

  /** @brief Broadcast arrays to common shape (np.broadcast_arrays). */
  NP_API template <typename T>
  NP_NODISCARD auto broadcast_arrays(const std::vector<ndarray<T>>& arrays)
      -> std::vector<ndarray<T>>
  {
    if (arrays.empty())
      return {};
    std::vector<int> common = arrays[0].shape;
    for (std::size_t i = 1; i < arrays.size(); ++i)
      common = detail::broadcast_shapes(common, arrays[i].shape);
    std::vector<ndarray<T>> out;
    out.reserve(arrays.size());
    for (auto& arr : arrays)
    {
      out.push_back(broadcast_to(arr, common));
    }
    return out;
  }

  // Normal comment: free wrappers for ndarray shape methods

  NP_API template <typename T>
  NP_NODISCARD auto reshape(const ndarray<T>& a, const std::vector<int>& shape)
      -> ndarray<T>
  {
    return a.reshape(shape);
  }
  NP_API template <typename T>
  NP_NODISCARD auto ravel(const ndarray<T>& a) -> ndarray<T>
  {
    return a.ravel();
  }
  NP_API template <typename T>
  NP_NODISCARD auto squeeze(const ndarray<T>& a) -> ndarray<T>
  {
    return a.squeeze();
  }
  NP_API template <typename T>
  NP_NODISCARD auto squeeze(const ndarray<T>& a, int axis) -> ndarray<T>
  {
    return a.squeeze(axis);
  }
  NP_API template <typename T>
  NP_NODISCARD auto transpose(const ndarray<T>& a) -> ndarray<T>
  {
    return a.transpose();
  }
  NP_API template <typename T>
  NP_NODISCARD auto transpose(const ndarray<T>& a, const std::vector<int>& axes)
      -> ndarray<T>
  {
    return a.transpose(axes);
  }
  NP_API template <typename T>
  NP_NODISCARD auto swapaxes(const ndarray<T>& a, int axis1, int axis2) -> ndarray<T>
  {
    return a.swapaxes(axis1, axis2);
  }

} // namespace np

#endif // NP_MANIPULATION_HPP
