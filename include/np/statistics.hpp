/**
 * @file statistics.hpp
 * @brief Statistical functions (NumPy reference: routines.statistics).
 *
 * Provides scalar and axis-aware reductions on np::ndarray mirroring
 * numpy: median, percentile, quantile, average, ptp, corrcoef, cov,
 * histogram, bincount, digitize and the NaN-skipping nan* family.
 *
 * Axis convention matches np::ndarray's member reductions:
 *  - the no-axis overload reduces over the whole (flattened) array and
 *    returns a scalar;
 *  - the overload taking an `int axis` reduces along that single axis
 *    (negative indices count from the end) and returns an ndarray.
 *
 * All floating-point results are computed in double for double inputs and
 * preserved for float inputs (NumPy semantics: a float32 input keeps
 * float32). Integer and bool inputs promote to double.
 *
 * Reference: numpy-reference/reference/routines.statistics.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_STATISTICS_HPP
#define NP_STATISTICS_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "api_macros.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"

namespace np {

namespace detail {

/** @brief Normalize an axis index, throwing np::AxisError if invalid. */
[[nodiscard]] inline int stat_normalize_axis(int axis, std::size_t ndim,
                                             const char *what) {
  const int nd = static_cast<int>(ndim);
  if (axis < 0) {
    axis += nd;
  }
  if (axis < 0 || axis >= nd) {
    throw np::AxisError(std::string(what) + ": axis " + std::to_string(axis) +
                        " is out of bounds for array of dimension " +
                        std::to_string(nd));
  }
  return axis;
}

/** @brief Row-major flat offset of `idx` within `shape`. */
[[nodiscard]] inline std::size_t
row_major_offset(const std::vector<std::size_t> &idx,
                 const std::vector<int> &shape) {
  std::size_t flat = 0;
  for (std::size_t d = 0; d < shape.size(); ++d) {
    flat = flat * static_cast<std::size_t>(shape[d]) + idx[d];
  }
  return flat;
}

/**
 * @brief Gather the elements of one slice of `arr` along `axis`.
 *
 * `base` is a full-size index into `arr`; the `axis` component is
 * overwritten with 0..shape[axis]-1. Returns the slice as a fresh vector,
 * reading through the public `get()` accessor (argument evaluates strides
 * and negative offsets of views).
 */
template <typename T>
[[nodiscard]] std::vector<T> gather_slice(const ndarray<T> &arr, int axis,
                                          std::vector<std::size_t> base) {
  const std::size_t alen = static_cast<std::size_t>(arr.shape[axis]);
  std::vector<T> slice;
  slice.reserve(alen);
  for (std::size_t a = 0; a < alen; ++a) {
    base[static_cast<std::size_t>(axis)] = a;
    slice.push_back(arr.get(base));
  }
  return slice;
}

/**
 * @brief Apply `fn` to every slice along `axis`, assembling the results.
 *
 * @tparam R   Result element type.
 * @tparam T   Input element type.
 * @tparam Fn  Callable `(const std::vector<T>&) -> R`.
 * @param arr  Input array.
 * @param axis Axis along which to reduce (normalized; may be negative).
 * @param fn   Per-slice function.
 * @return     Array with `axis` removed holding `fn` per slice.
 */
template <typename R, typename T, typename Fn>
[[nodiscard]] ndarray<R> stat_axis_map(const ndarray<T> &arr, int axis,
                                       Fn &&fn) {
  axis = stat_normalize_axis(axis, arr.ndim(), "np::stats");
  std::vector<int> out_shape = arr.shape;
  out_shape.erase(out_shape.begin() + axis);
  const std::size_t nd = static_cast<std::size_t>(arr.ndim());

  ndarray<R> out(out_shape);
  detail::Odometer od(out_shape);
  std::vector<std::size_t> full(nd, 0);
  while (!od.done()) {
    const auto &red = od.idx();
    for (std::size_t d = 0, r = 0; d < nd; ++d) {
      if (static_cast<int>(d) == axis) {
        full[d] = 0;
        continue;
      }
      full[d] = red[r++];
    }
    const std::size_t flat = detail::row_major_offset(red, out_shape);
    out.data()[flat] = fn(detail::gather_slice(arr, axis, full));
    od.advance();
  }
  return out;
}

/** @brief Linear interpolation of `values` at fractional `position`. */
[[nodiscard]] inline double lin_interp(const std::vector<double> &values,
                                       double position) {
  const auto n = values.size();
  if (position <= 0.0)
    return values[0];
  if (position >= static_cast<double>(n - 1))
    return values.back();
  const std::size_t lo = static_cast<std::size_t>(position);
  const double frac = position - static_cast<double>(lo);
  return values[lo] + frac * (values[lo + 1] - values[lo]);
}

/** @brief Promote a (possibly bool/integer) element type for reductions. */
template <typename T>
using stat_real_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;

/** @brief NaN tester: false for integer/bool/complex-safe types. */
template <typename T> constexpr bool is_nan_elem(const T &v) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(v);
  } else {
    (void)v;
    return false;
  }
}

} // namespace detail

// =================================================================
// Median / percentile / quantile
// =================================================================

/**
 * @brief Median of all elements.
 *
 * Sorts a copy of the (flattened) input and returns the middle value;
 * for even-size inputs the mean of the two middle values is returned.
 *
 * Reference: numpy-reference/reference/generated/numpy.median.html
 *
 * @tparam T Element type.
 * @param arr Input array.
 * @return Median of all elements.
 * @throws std::invalid_argument if the array is empty.
 * @complexity O(n log n). Space: O(n).
 */
NP_API template <typename T>
NP_NODISCARD auto median(const ndarray<T> &arr) -> double {
  if (arr.size() == 0) {
    throw std::invalid_argument("median: empty array");
  }
  std::vector<double> values;
  values.reserve(arr.size());
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    values.push_back(static_cast<double>(*it));
  }
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

/**
 * @brief Median along a single axis.
 *
 * Reference: numpy-reference/reference/generated/numpy.median.html
 *
 * @tparam T The element type.
 * @param arr Input array.
 * @param axis Axis along which to reduce (may be negative).
 * @return Array of medians with `axis` removed.
 * @throws std::invalid_argument if a reduced slice is empty.
 * @throws np::AxisError if `axis` is out of bounds.
 * @complexity O(n log axis_len).
 */
NP_API template <typename T>
NP_NODISCARD auto median(const ndarray<T> &arr, int axis) -> ndarray<double> {
  return detail::stat_axis_map<double>(
      arr, axis, [](const std::vector<T> &slice) -> double {
        if (slice.empty()) {
          throw std::invalid_argument("median: empty slice along axis");
        }
        std::vector<double> vals;
        vals.reserve(slice.size());
        for (const auto &v : slice) {
          vals.push_back(static_cast<double>(v));
        }
        std::sort(vals.begin(), vals.end());
        const std::size_t n = vals.size();
        if (n % 2 == 1)
          return vals[n / 2];
        return 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
      });
}

/**
 * @brief Percentile over all elements, NumPy `linear` method.
 *
 * The `linear` interpolation method (NumPy default) is used: the sorted
 * value at position `(n-1) * q/100` with fractional interpolation between
 * the two neighbours.
 *
 * Reference: numpy-reference/reference/generated/numpy.percentile.html
 *
 * @tparam T Element type.
 * @tparam Q Percentile type (typically double).
 * @param arr Input array.
 * @param q Percentile, 0 <= q <= 100.
 * @return The q-th percentile of all elements.
 * @throws std::invalid_argument if q is out of range or arr is empty.
 * @complexity O(n log n).
 */
NP_API template <typename T, typename Q>
NP_NODISCARD auto percentile(const ndarray<T> &arr, const Q &q) -> double {
  if (arr.size() == 0) {
    throw std::invalid_argument("percentile: empty array");
  }
  const double p = static_cast<double>(q);
  if (p < 0.0 || p > 100.0) {
    throw std::invalid_argument("percentile: q must be in [0, 100]");
  }
  std::vector<double> values;
  values.reserve(arr.size());
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    values.push_back(static_cast<double>(*it));
  }
  std::sort(values.begin(), values.end());
  return detail::lin_interp(values, (values.size() - 1) * (p / 100.0));
}

/**
 * @brief Percentile along a single axis (linear method).
 *
 * Reference: numpy-reference/reference/generated/numpy.percentile.html
 *
 * @tparam T Element type.
 * @tparam Q Percentile type (typically double).
 * @param arr Input array.
 * @param q Percentile, 0 <= q <= 100.
 * @param axis Axis along which to reduce.
 * @return Array of percentiles with `axis` removed.
 * @throws std::invalid_argument if q is out of range.
 * @throws np::AxisError if `axis` is out of bounds.
 * @complexity O(n log axis_len).
 */
NP_API template <typename T, typename Q>
NP_NODISCARD auto percentile(const ndarray<T> &arr, const Q &q, int axis)
    -> ndarray<double> {
  const double p = static_cast<double>(q);
  if (p < 0.0 || p > 100.0) {
    throw std::invalid_argument("percentile: q must be in [0, 100]");
  }
  return detail::stat_axis_map<double>(
      arr, axis, [p](const std::vector<T> &slice) -> double {
        if (slice.empty()) {
          throw std::invalid_argument("percentile: empty slice along axis");
        }
        std::vector<double> vals;
        vals.reserve(slice.size());
        for (const auto &v : slice) {
          vals.push_back(static_cast<double>(v));
        }
        std::sort(vals.begin(), vals.end());
        return detail::lin_interp(vals, (vals.size() - 1) * (p / 100.0));
      });
}

/**
 * @brief Quantile of all elements (percentile with q in [0,1]).
 *
 * Reference: numpy-reference/reference/generated/numpy.quantile.html
 *
 * @tparam T Element type.
 * @tparam Q Quantile type (typically double).
 * @param arr Input array.
 * @param q Quantile, 0 <= q <= 1.
 * @return The q-th quantile of all elements.
 * @throws std::invalid_argument if q is out of range or arr is empty.
 */
NP_API template <typename T, typename Q>
NP_NODISCARD auto quantile(const ndarray<T> &arr, const Q &q) -> double {
  const double p = static_cast<double>(q);
  if (p < 0.0 || p > 1.0) {
    throw std::invalid_argument("quantile: q must be in [0, 1]");
  }
  return percentile(arr, p * 100.0);
}

/**
 * @brief Quantile along a single axis (linear method).
 *
 * @tparam T Element type.
 * @tparam Q Quantile type (typically double).
 * @param arr Input array.
 * @param q Quantile, 0 <= q <= 1.
 * @param axis Axis along which to reduce.
 * @return Array of quantiles with `axis` removed.
 * @throws std::invalid_argument if q is out of range.
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T, typename Q>
NP_NODISCARD auto quantile(const ndarray<T> &arr, const Q &q, int axis)
    -> ndarray<double> {
  const double p = static_cast<double>(q);
  if (p < 0.0 || p > 1.0) {
    throw std::invalid_argument("quantile: q must be in [0, 1]");
  }
  return percentile(arr, p * 100.0, axis);
}

// =================================================================
// NaN-skipping reduction family (nanmin/nanmax/nansum/nanprod/
// nanmean/nanvar/nanstd/nanmedian/nanpercentile/nanquantile)
// =================================================================

/**
 * @brief Minimum of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nanmin.html
 *
 * @throws std::invalid_argument if every element is NaN.
 */
NP_API template <typename T>
NP_NODISCARD auto nanmin(const ndarray<T> &arr) -> T {
  bool any = false;
  T best{};
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    if (detail::is_nan_elem(*it))
      continue;
    if (!any || *it < best) {
      best = *it;
      any = true;
    }
  }
  if (!any) {
    throw std::invalid_argument("nanmin: all-NaN slice");
  }
  return best;
}

/** @brief Minimum along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanmin(const ndarray<T> &arr, int axis) -> ndarray<T> {
  return detail::stat_axis_map<T>(
      arr, axis, [](const std::vector<T> &slice) -> T {
        bool any = false;
        T best{};
        for (const auto &v : slice) {
          if (detail::is_nan_elem(v))
            continue;
          if (!any || v < best) {
            best = v;
            any = true;
          }
        }
        if (!any) {
          throw std::invalid_argument("nanmin: all-NaN slice");
        }
        return best;
      });
}

/**
 * @brief Maximum of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nanmax.html
 *
 * @throws std::invalid_argument if every element is NaN.
 */
NP_API template <typename T>
NP_NODISCARD auto nanmax(const ndarray<T> &arr) -> T {
  bool any = false;
  T best{};
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    if (detail::is_nan_elem(*it))
      continue;
    if (!any || *it > best) {
      best = *it;
      any = true;
    }
  }
  if (!any) {
    throw std::invalid_argument("nanmax: all-NaN slice");
  }
  return best;
}

/** @brief Maximum along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanmax(const ndarray<T> &arr, int axis) -> ndarray<T> {
  return detail::stat_axis_map<T>(
      arr, axis, [](const std::vector<T> &slice) -> T {
        bool any = false;
        T best{};
        for (const auto &v : slice) {
          if (detail::is_nan_elem(v))
            continue;
          if (!any || v > best) {
            best = v;
            any = true;
          }
        }
        if (!any) {
          throw std::invalid_argument("nanmax: all-NaN slice");
        }
        return best;
      });
}

/**
 * @brief Sum of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nansum.html
 *
 * An all-NaN (or empty) input sums to zero (NumPy semantics).
 */
NP_API template <typename T>
NP_NODISCARD auto nansum(const ndarray<T> &arr) ->
    typename np::_mean_type<T>::type {
  using R = typename np::_mean_type<T>::type;
  R sum{};
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    if (detail::is_nan_elem(*it))
      continue;
    sum += static_cast<R>(*it);
  }
  return sum;
}

/** @brief Sum along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nansum(const ndarray<T> &arr, int axis)
    -> ndarray<typename np::_mean_type<T>::type> {
  using R = typename np::_mean_type<T>::type;
  return detail::stat_axis_map<R>(arr, axis,
                                  [](const std::vector<T> &slice) -> R {
                                    R sum = R{};
                                    for (const auto &v : slice) {
                                      if (detail::is_nan_elem(v))
                                        continue;
                                      sum += static_cast<R>(v);
                                    }
                                    return sum;
                                  });
}

/**
 * @brief Product of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nanprod.html
 *
 * Empty array product is 1 (NumPy behavior).
 */
NP_API template <typename T>
NP_NODISCARD auto nanprod(const ndarray<T> &arr) ->
    typename np::_mean_type<T>::type {
  using R = typename np::_mean_type<T>::type;
  R prod = static_cast<R>(1);
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    if (detail::is_nan_elem(*it))
      continue;
    prod = static_cast<R>(prod * static_cast<R>(*it));
  }
  return prod;
}

/** @brief Product along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanprod(const ndarray<T> &arr, int axis)
    -> ndarray<typename np::_mean_type<T>::type> {
  using R = typename np::_mean_type<T>::type;
  return detail::stat_axis_map<R>(
      arr, axis, [](const std::vector<T> &slice) -> R {
        R prod = static_cast<R>(1);
        for (const auto &v : slice) {
          if (detail::is_nan_elem(v))
            continue;
          prod = static_cast<R>(prod * static_cast<R>(v));
        }
        return prod;
      });
}

/**
 * @brief Mean of all elements, ignoring NaN.
 *
 * Reference: numpy-reference/reference/generated/numpy.nanmean.html
 *
 * If every element is NaN the reduction yields NaN (NumPy behavior).
 */
NP_API template <typename T>
NP_NODISCARD auto nanmean(const ndarray<T> &arr) ->
    typename np::_mean_type<T>::type {
  using R = typename np::_mean_type<T>::type;
  long double sum = 0.0;
  std::size_t n = 0;
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    if (detail::is_nan_elem(*it))
      continue;
    sum += static_cast<long double>(*it);
    ++n;
  }
  if (n == 0) {
    return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
  }
  return static_cast<R>(sum / static_cast<long double>(n));
}

/** @brief Mean along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanmean(const ndarray<T> &arr, int axis)
    -> ndarray<typename np::_mean_type<T>::type> {
  using R = typename np::_mean_type<T>::type;
  return detail::stat_axis_map<R>(
      arr, axis, [](const std::vector<T> &slice) -> R {
        long double sum = 0.0;
        std::size_t n = 0;
        for (const auto &v : slice) {
          if (detail::is_nan_elem(v))
            continue;
          sum += static_cast<long double>(v);
          ++n;
        }
        return n == 0 ? static_cast<R>(std::numeric_limits<double>::quiet_NaN())
                      : static_cast<R>(sum / static_cast<long double>(n));
      });
}

/**
 * @brief Variance of all elements, ignoring NaN (population, ddof=0).
 *
 * Reference: numpy-reference/reference/generated/numpy.nanvar.html
 *
 * All-NaN input yields NaN.
 */
NP_API template <typename T>
NP_NODISCARD auto nanvar(const ndarray<T> &arr) ->
    typename np::_mean_type<T>::type {
  using R = typename np::_mean_type<T>::type;
  const auto m = nanmean(arr);
  if (detail::is_nan_elem(m)) {
    return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
  }
  long double acc = 0.0;
  std::size_t n = 0;
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    if (detail::is_nan_elem(*it))
      continue;
    const long double d =
        static_cast<long double>(*it) - static_cast<long double>(m);
    acc += d * d;
    ++n;
  }
  if (n == 0)
    return static_cast<R>(std::numeric_limits<double>::quiet_NaN());
  return static_cast<R>(acc / static_cast<long double>(n));
}

/** @brief Standard deviation of all elements, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanstd(const ndarray<T> &arr) ->
    typename np::_mean_type<T>::type {
  using R = typename np::_mean_type<T>::type;
  return static_cast<R>(std::sqrt(static_cast<long double>(nanvar(arr))));
}

/**
 * @brief Median of all elements, ignoring NaN.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html
 *
 * Empty or all-NaN input yields NaN.
 */
NP_API template <typename T>
NP_NODISCARD auto nanmedian(const ndarray<T> &arr) {
  return nanpercentile(arr, 50.0);
}

/** @brief Median along an axis, ignoring NaN. */
NP_API template <typename T>
NP_NODISCARD auto nanmedian(const ndarray<T> &arr, int axis) {
  return nanpercentile(arr, 50.0, axis);
}

/** @brief Sorted, NaN-stripped values of `slice` (percentile w/o NaN). */
namespace detail {
template <typename T>
[[nodiscard]] double nan_percentile_of_slice(const std::vector<T> &slice,
                                             double p) {
  std::vector<double> vals;
  vals.reserve(slice.size());
  for (const auto &v : slice) {
    if (is_nan_elem(v))
      continue;
    vals.push_back(static_cast<double>(v));
  }
  if (vals.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(vals.begin(), vals.end());
  return lin_interp(vals, (vals.size() - 1) * (p / 100.0));
}
} // namespace detail

/**
 * @brief Percentile of all elements, ignoring NaN (linear method).
 *
 * Reference: numpy-reference/reference/generated/numpy.nanpercentile.html
 *
 * All-NaN or empty input yields NaN.
 */
NP_API template <typename T, typename Q>
NP_NODISCARD auto nanpercentile(const ndarray<T> &arr, const Q &q) -> double {
  const double p = static_cast<double>(q);
  if (p < 0.0 || p > 100.0) {
    throw std::invalid_argument("nanpercentile: q must be in [0, 100]");
  }
  std::vector<T> all;
  all.reserve(arr.size());
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    all.push_back(*it);
  }
  return detail::nan_percentile_of_slice(all, p);
}

/** @brief Percentile along an axis, ignoring NaN (linear method). */
NP_API template <typename T, typename Q>
NP_NODISCARD auto nanpercentile(const ndarray<T> &arr, const Q &q, int axis)
    -> ndarray<double> {
  const double p = static_cast<double>(q);
  if (p < 0.0 || p > 100.0) {
    throw std::invalid_argument("nanpercentile: q must be in [0, 100]");
  }
  return detail::stat_axis_map<double>(
      arr, axis, [p](const std::vector<T> &slice) -> double {
        return detail::nan_percentile_of_slice(slice, p);
      });
}

/**
 * @brief Quantile of all elements, ignoring NaN (q in [0,1]).
 *
 * Reference: numpy-reference/reference/generated/numpy.nanquantile.html
 */
NP_API template <typename T, typename Q>
NP_NODISCARD auto nanquantile(const ndarray<T> &arr, const Q &q) -> double {
  const double p = static_cast<double>(q);
  if (p < 0.0 || p > 1.0) {
    throw std::invalid_argument("nanquantile: q must be in [0, 1]");
  }
  return nanpercentile(arr, p * 100.0);
}

/** @brief Quantile along an axis, ignoring NaN (q in [0,1]). */
NP_API template <typename T, typename Q>
NP_NODISCARD auto nanquantile(const ndarray<T> &arr, const Q &q, int axis)
    -> ndarray<double> {
  const double p = static_cast<double>(q);
  if (p < 0.0 || p > 1.0) {
    throw std::invalid_argument("nanquantile: q must be in [0, 1]");
  }
  return nanpercentile(arr, p * 100.0, axis);
}

/**
 * @brief Unweighted mean of all elements (scalar).
 *
 * Reference: numpy-reference/reference/generated/numpy.average.html
 *
 * @tparam T Element type of the input array.
 * @param arr Input array.
 * @return Mean of all elements.
 * @throws std::invalid_argument if the array is empty.
 */
NP_API template <typename T>
NP_NODISCARD auto average(const ndarray<T> &arr) -> double {
  if (arr.size() == 0) {
    throw std::invalid_argument("average: empty array");
  }
  using R = typename np::_mean_type<T>::type;
  R total = R{};
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    total += static_cast<R>(*it);
  }
  return static_cast<double>(total / static_cast<R>(arr.size()));
}

/**
 * @brief Weighted mean of all elements.
 *
 * Reference: numpy-reference/reference/generated/numpy.average.html
 *
 * @tparam T Element type of the input array.
 * @tparam W Element type of the weights.
 * @param arr     Input array.
 * @param weights Weights with the same total number of elements as `arr`.
 * @return Weighted mean of all elements.
 * @throws std::invalid_argument if weights size differs from arr size or
 *         the sum of weights is zero.
 */
NP_API template <typename T, typename W>
NP_NODISCARD double average(const ndarray<T> &arr, const ndarray<W> &weights) {
  if (arr.size() == 0) {
    throw std::invalid_argument("average: empty array");
  }
  if (weights.size() != arr.size()) {
    throw std::invalid_argument(
        "average: weights size does not match array size");
  }
  long double num = 0.0;
  long double den = 0.0;
  auto wi = weights.begin();
  for (auto it = arr.begin(); it != arr.end(); ++it, ++wi) {
    const long double wv = static_cast<long double>(*wi);
    num += static_cast<long double>(*it) * wv;
    den += wv;
  }
  if (den == 0.0) {
    throw std::invalid_argument("average: sum of weights is zero");
  }
  return static_cast<double>(num / den);
}

/**
 * @brief Unweighted mean along a single axis.
 *
 * @tparam T Element type of the input array.
 * @param arr  Input array.
 * @param axis Axis along which to reduce.
 * @return Array of means with `axis` removed.
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T>
NP_NODISCARD auto average(const ndarray<T> &arr, int axis) -> ndarray<double> {
  return detail::stat_axis_map<double>(
      arr, axis, [](const std::vector<T> &slice) -> double {
        if (slice.empty()) {
          throw std::invalid_argument("average: empty slice along axis");
        }
        long double sum = 0.0;
        for (const auto &v : slice) {
          sum += static_cast<long double>(v);
        }
        return static_cast<double>(sum /
                                   static_cast<long double>(slice.size()));
      });
}

/**
 * @brief Weighted mean along a single axis.
 *
 * @tparam T Element type of the input array.
 * @tparam W Element type of the weights.
 * @param arr     Input array.
 * @param axis    Axis along which to reduce.
 * @param weights Weights: one per element of `arr` along `axis`.
 * @return Array of weighted means with `axis` removed.
 * @throws std::invalid_argument if weights.size() != arr.shape[axis].
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T, typename W>
NP_NODISCARD auto average(const ndarray<T> &arr, int axis,
                          const ndarray<W> &weights) -> ndarray<double> {
  const int ax = detail::stat_normalize_axis(axis, arr.ndim(), "np::average");
  if (weights.size() != static_cast<std::size_t>(arr.shape[ax])) {
    throw std::invalid_argument(
        "average: weights size must equal arr.shape[axis]");
  }
  return detail::stat_axis_map<double>(
      arr, axis, [&weights](const std::vector<T> &slice) -> double {
        if (slice.empty()) {
          throw std::invalid_argument("average: empty slice along axis");
        }
        long double num = 0.0;
        long double den = 0.0;
        for (std::size_t i = 0; i < slice.size(); ++i) {
          const long double wv = static_cast<long double>(weights.data()[i]);
          num += static_cast<long double>(slice[i]) * wv;
          den += wv;
        }
        if (den == 0.0) {
          throw std::invalid_argument("average: sum of weights is zero");
        }
        return static_cast<double>(num / den);
      });
}

// =================================================================
// ptp (peak-to-peak) -- free wrapper of ndarray::ptp
// =================================================================

/**
 * @brief Peak-to-peak (max - min) over all elements.
 *
 * Reference: numpy-reference/reference/generated/numpy.ptp.html
 *
 * @tparam T Element type.
 * @param arr Input array.
 * @return max(arr) - min(arr).
 * @throws std::runtime_error if the array is empty.
 * @complexity O(n).
 */
NP_API template <typename T> NP_NODISCARD auto ptp(const ndarray<T> &arr) -> T {
  return arr.max() - arr.min();
}

/**
 * @brief Peak-to-peak (max - min) along a single axis.
 *
 * @tparam T Element type.
 * @param arr Input array.
 * @param axis Axis along which to reduce (may be negative).
 * @return Array of peak-to-peak values with `axis` removed.
 * @throws np::AxisError if `axis` is out of bounds.
 */
NP_API template <typename T>
NP_NODISCARD auto ptp(const ndarray<T> &arr, int axis) -> ndarray<T> {
  return arr.ptp(axis);
}

// =================================================================
// Covariance / correlation
// =================================================================

/**
 * @brief Two internal helpers.
 * @internal
 */
namespace detail {

/** @brief Rows (variables) of `m` as vectors of double. */
template <typename T>
[[nodiscard]] std::vector<std::vector<double>>
rows_as_variables(const ndarray<T> &m) {
  std::vector<std::vector<double>> rows;
  const std::size_t nrows =
      static_cast<std::size_t>(m.ndim() >= 2 ? m.shape[0] : 1);
  const std::size_t ncols =
      static_cast<std::size_t>(m.ndim() == 0 ? 1 : m.shape[m.ndim() - 1]);
  rows.reserve(nrows);
  for (std::size_t r = 0; r < nrows; ++r) {
    std::vector<double> row;
    row.reserve(ncols);
    for (std::size_t c = 0; c < ncols; ++c) {
      if (m.ndim() >= 2) {
        std::vector<std::size_t> base(static_cast<std::size_t>(m.ndim()), 0);
        base.front() = r;
        base.back() = c;
        row.push_back(static_cast<double>(m.get(base)));
      } else {
        row.push_back(static_cast<double>(m((std::size_t)c)));
      }
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

/**
 * @brief Compute the covariance matrix of the given rows (variables).
 *
 * Each inner vector is one variable; the k entries are observations.
 * Returns an nvars-nvars matrix. `ddof` defaults to 1 (NumPy).
 */
[[nodiscard]] inline std::vector<std::vector<double>>
cov_from_rows(const std::vector<std::vector<double>> &rows, int ddof) {
  const std::size_t k = rows.empty() ? 0 : rows.front().size();
  const std::size_t n = rows.size();
  std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));
  if (k == 0 || n == 0) {
    return cov;
  }
  const double normalizer = static_cast<double>(k - ddof);
  std::vector<double> mean(n, 0.0);
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t c = 0; c < k; ++c) {
      mean[r] += rows[r][c];
    }
    mean[r] /= static_cast<double>(k);
  }
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t c = 0; c < n; ++c) {
      double acc = 0.0;
      for (std::size_t t = 0; t < k; ++t) {
        acc += (rows[r][t] - mean[r]) * (rows[c][t] - mean[c]);
      }
      cov[r][c] = acc / normalizer;
    }
  }
  return cov;
}

/** @brief Correlation of the given rows (variables) of observations. */
[[nodiscard]] inline std::vector<std::vector<double>>
corr_from_rows(const std::vector<std::vector<double>> &rows) {
  const std::size_t n = rows.size();
  const std::size_t k = rows.empty() ? 0 : rows.front().size();
  std::vector<std::vector<double>> corr(n, std::vector<double>(n, 0.0));
  if (k == 0 || n == 0) {
    return corr;
  }
  std::vector<double> mean(n, 0.0);
  for (std::size_t r = 0; r < n; ++r) {
    double s = 0.0;
    for (std::size_t t = 0; t < k; ++t) {
      s += rows[r][t];
    }
    mean[r] = s / static_cast<double>(k);
  }
  std::vector<double> ss(n, 0.0);
  for (std::size_t r = 0; r < n; ++r) {
    double s = 0.0;
    for (std::size_t t = 0; t < k; ++t) {
      const double d = rows[r][t] - mean[r];
      s += d * d;
    }
    ss[r] = s;
  }
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t c = 0; c < n; ++c) {
      const double denom = std::sqrt(ss[r] * ss[c]);
      if (denom > 0.0) {
        double acc = 0.0;
        for (std::size_t t = 0; t < k; ++t) {
          acc += (rows[r][t] - mean[r]) * (rows[c][t] - mean[c]);
        }
        corr[r][c] = acc / denom;
      } else {
        corr[r][c] = rows[r][c] == rows[c][r] ? 1.0 : 0.0;
      }
    }
  }
  return corr;
}

/** @brief Copy rows into a fresh square ndarray<double>. */
[[nodiscard]] inline ndarray<double>
matrix_to_ndarray(const std::vector<std::vector<double>> &mat) {
  ndarray<double> out(std::vector<int>{static_cast<int>(mat.size()),
                                       static_cast<int>(mat.size())});
  for (std::size_t r = 0; r < mat.size(); ++r) {
    for (std::size_t c = 0; c < mat[r].size(); ++c) {
      out.data()[r * mat.size() + c] = mat[r][c];
    }
  }
  return out;
}

/**
 * @brief Build the rows (variables) fed to cov/corrcoef.
 *
 *  - If `y` is given, `x` and `y` are treated as two observation vectors
 *    (each element is one sample) and two rows are produced.
 *  - Otherwise a 1-D `x` yields a single row and a 2-D `x` yields one row
 *    per variable (rowvar=True).
 */
template <typename T>
[[nodiscard]] std::vector<std::vector<double>> cov_rows(const ndarray<T> &x,
                                                        const ndarray<T> *y) {
  std::vector<std::vector<double>> rows;
  if (y != nullptr) {
    rows.emplace_back();
    rows.emplace_back();
    auto a = x.begin();
    auto b = y->begin();
    for (; a != x.end() && b != y->end(); ++a, ++b) {
      rows[0].push_back(static_cast<double>(*a));
      rows[1].push_back(static_cast<double>(*b));
    }
    return rows;
  }
  if (x.ndim() == 1) {
    rows.emplace_back();
    for (auto it = x.begin(); it != x.end(); ++it) {
      rows.front().push_back(static_cast<double>(*it));
    }
    return rows;
  }
  return rows_as_variables(x);
}

} // namespace detail

/**
 * @brief Covariance matrix of a single variable set.
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.cov.html
 *
 * If `x` is 2-D, each row is a variable (rowvar=True) and the columns
 * are the observations; a 1-D `x` yields a 1x1 matrix. `ddof` defaults
 * to 1 (NumPy sample covariance).
 *
 * @tparam T Element type.
 * @param x    Variable(s).
 * @param ddof Delta degrees of freedom (default 1, NumPy).
 * @return Covariance matrix (nvars x nvars).
 */
NP_API template <typename T>
NP_NODISCARD auto cov(const ndarray<T> &x, int ddof = 1) -> ndarray<double> {
  const auto cv = detail::cov_from_rows(detail::cov_rows<T>(x, nullptr), ddof);
  return detail::matrix_to_ndarray(cv);
}

/**
 * @brief Cross-covariance matrix of two observation vectors.
 *
 * Formula:
 * \f[
 * C_{uv} = \frac{1}{k-\text{ddof}}\sum_i (u_i-\bar u)(v_i-\bar v)
 * \f]
 *
 * Reference: https://numpy.org/doc/stable/reference/generated/numpy.cov.html
 *
 * @tparam T Element type of `x`.
 * @tparam U Element type of `y`.
 * @param x    First observation vector (1-D).
 * @param y    Second observation vector (1-D), same length as `x`.
 * @param ddof Delta degrees of freedom (default 1, NumPy).
 * @return 2x2 covariance matrix.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto cov(const ndarray<T> &x, const ndarray<U> &y, int ddof = 1)
    -> ndarray<double> {
  if (x.size() != y.size()) {
    throw std::invalid_argument("cov: x and y must have the same length");
  }
  const ndarray<double> yd = y.template astype<double>();
  const auto cv = detail::cov_from_rows(detail::cov_rows(x, &yd), ddof);
  return detail::matrix_to_ndarray(cv);
}

/**
 * @brief Pearson correlation coefficient matrix of a single variable.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
 *
 * @tparam T Element type.
 * @param x 2-D (rows are variables) or 1-D input.
 * @return Correlation matrix (1x1 for 1-D input).
 */
NP_API template <typename T>
NP_NODISCARD auto corrcoef(const ndarray<T> &x) -> ndarray<double> {
  const auto cr = detail::corr_from_rows(detail::cov_rows<T>(x, nullptr));
  return detail::matrix_to_ndarray(cr);
}

/**
 * @brief Pearson correlation coefficient matrix of two vectors.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
 *
 * @tparam T Element type of `x`.
 * @tparam U Element type of `y`.
 * @param x First observation vector (1-D).
 * @param y Second observation vector (1-D), same length as `x`.
 * @return 2x2 correlation matrix.
 */
NP_API template <typename T, typename U>
NP_NODISCARD auto corrcoef(const ndarray<T> &x, const ndarray<U> &y)
    -> ndarray<double> {
  if (x.size() != y.size()) {
    throw std::invalid_argument("corrcoef: x and y must have the same length");
  }
  const ndarray<double> yd = y.template astype<double>();
  const auto cr = detail::corr_from_rows(detail::cov_rows(x, &yd));
  return detail::matrix_to_ndarray(cr);
}

// =================================================================
// Histogram / bincount / digitize
// =================================================================

/**
 * @brief Result of np::histogram.
 */
struct Histogram {
  ndarray<std::size_t> counts; ///< Bin counts.
  ndarray<double> edges;       ///< Bin edges (counts+1 values).
};

/**
 * @brief Compute the histogram of an array.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
 *
 * @tparam T Element type.
 * @param arr  Input array (flattened).
 * @param bins If `ndarray<double>`, the bin edges; if integer, the number
 *             of equal-width bins between the data range (default 10).
 * @param range Optional {low, high} overriding the data range.
 * @return Histogram {counts, edges}.
 * @throws std::invalid_argument if bins<=0 or the range is degenerate.
 */
NP_API template <typename T>
NP_NODISCARD auto
histogram(const ndarray<T> &arr, int bins = 10,
          std::optional<std::pair<double, double>> range = std::nullopt)
    -> Histogram {
  if (arr.size() == 0) {
    throw std::invalid_argument("histogram: empty array");
  }
  if (bins <= 0) {
    throw std::invalid_argument("histogram: bins must be a positive integer");
  }
  double lo = range.has_value() ? range->first : static_cast<double>(arr.min());
  double hi =
      range.has_value() ? range->second : static_cast<double>(arr.max());
  if (!(hi > lo)) {
    throw std::invalid_argument("histogram: empty or degenerate range");
  }
  std::vector<double> edges(static_cast<std::size_t>(bins) + 1);
  const double step = (hi - lo) / static_cast<double>(bins);
  for (int b = 0; b <= bins; ++b) {
    edges[static_cast<std::size_t>(b)] = lo + static_cast<double>(b) * step;
  }
  std::vector<std::size_t> counts(static_cast<std::size_t>(bins), 0);
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    const double v = static_cast<double>(*it);
    if (!(v >= lo && v <= hi))
      continue;
    if (v == hi) {
      ++counts[static_cast<std::size_t>(bins) - 1];
      continue;
    }
    const auto b = static_cast<std::size_t>((v - lo) / step);
    ++counts[b];
  }
  Histogram h;
  h.edges = ndarray<double>(std::vector<int>{bins + 1});
  for (std::size_t i = 0; i < edges.size(); ++i) {
    h.edges.data()[i] = edges[i];
  }
  h.counts = ndarray<std::size_t>(std::vector<int>{bins});
  for (std::size_t i = 0; i < counts.size(); ++i) {
    h.counts.data()[i] = counts[i];
  }
  return h;
}

/**
 * @brief Compute the histogram of an array using explicit edges.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
 *
 * @tparam T Element type.
 * @param arr   Input array (flattened).
 * @param edges Array of monotonic bin edges.
 * @return Histogram {counts, edges}.
 * @throws std::invalid_argument if edges has fewer than 2 elements.
 */
NP_API template <typename T>
NP_NODISCARD auto histogram(const ndarray<T> &arr, const ndarray<double> &edges)
    -> Histogram {
  if (edges.size() < 2) {
    throw std::invalid_argument(
        "histogram: edges must have at least 2 entries");
  }
  const std::size_t nbins = edges.size() - 1;
  std::vector<std::size_t> counts(nbins, 0);
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    const double v = static_cast<double>(*it);
    const double lo = edges.data()[0];
    const double hi = edges.data()[nbins];
    if (v < lo || v > hi)
      continue;
    if (v == hi) {
      ++counts[nbins - 1];
      continue;
    }
    const double w = hi - lo;
    for (std::size_t b = 0; b < nbins; ++b) {
      if (v >= edges.data()[b] && v < edges.data()[b + 1]) {
        ++counts[b];
        break;
      }
    }
    (void)w;
  }
  Histogram h;
  h.edges = edges.copy();
  h.counts = ndarray<std::size_t>(std::vector<int>{static_cast<int>(nbins)});
  for (std::size_t i = 0; i < nbins; ++i) {
    h.counts.data()[i] = counts[i];
  }
  return h;
}

/**
 * @brief Count the occurrences of each non-negative integer value.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
 *
 * @tparam T Integral element type.
 * @param arr Input 1-d integer array (non-negative).
 * @param weights Optional weights (same length as `arr`); when given the
 *        result elements are the sum of the weights of the matching values.
 * @param minlength Minimum length of the returned array.
 * @return Array of counts (or weighted sums).
 * @throws std::invalid_argument if a value is negative.
 */
NP_API template <typename T>
NP_NODISCARD auto
bincount(const ndarray<T> &arr,
         std::optional<ndarray<double>> weights = std::nullopt,
         std::size_t minlength = 0) -> ndarray<double> {
  static_assert(std::is_integral_v<T>,
                "np::bincount requires an integral element type");
  int maxv = -1;
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    if (static_cast<long long>(*it) < 0) {
      throw std::invalid_argument(
          "bincount: input values must be non-negative");
    }
    maxv = std::max(maxv, static_cast<int>(*it));
  }
  std::size_t n = static_cast<std::size_t>(maxv + 1);
  if (weights.has_value() && weights->size() != arr.size()) {
    throw std::invalid_argument("bincount: weights size must match arr size");
  }
  const std::size_t out_n = std::max(n, minlength);
  ndarray<double> out(std::vector<int>{static_cast<int>(out_n)});
  std::fill(out.data().begin(), out.data().end(), 0.0);
  std::size_t i = 0;
  const double *wptr = weights.has_value() ? weights->data().data() : nullptr;
  for (auto it = arr.begin(); it != arr.end(); ++it, ++i) {
    const std::size_t idx = static_cast<std::size_t>(*it);
    out.data()[idx] += wptr ? wptr[i] : 1.0;
  }
  return out;
}

/**
 * @brief Return the indices of the bins to which each value in `arr` belongs.
 *
 * Reference:
 * https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
 *
 * @tparam T Element type of the data.
 * @param arr Input array (1-D).
 * @param bins 1-D monotonic bin edges.
 * @param right Whether the intervals are closed on the left or the right.
 * @return Index array (same shape as `arr`).
 */
NP_API template <typename T>
NP_NODISCARD auto digitize(const ndarray<T> &arr, const ndarray<double> &bins,
                           bool right = false) -> ndarray<std::size_t> {
  if (bins.ndim() != 1) {
    throw std::invalid_argument("digitize: bins must be 1-D");
  }
  std::vector<double> sorted_bins(bins.data().begin(),
                                  bins.data().begin() + bins.size());
  ndarray<std::size_t> out(arr.shape);
  std::size_t i = 0;
  for (auto it = arr.begin(); it != arr.end(); ++it, ++i) {
    const double v = static_cast<double>(*it);
    const auto best = [&]() {
      return right
                 ? std::upper_bound(sorted_bins.begin(), sorted_bins.end(), v)
                 : std::lower_bound(sorted_bins.begin(), sorted_bins.end(), v);
    }();
    out.data()[i] =
        static_cast<std::size_t>(std::distance(sorted_bins.begin(), best));
  }
  return out;
}

} // namespace np

#endif // NP_STATISTICS_HPP