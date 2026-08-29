/**
 * @file testing.hpp
 * @brief Test support (np.testing).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.testing.html
 *
 * Provides C++ equivalents of numpy.testing asserts. Each function
 * throws std::runtime_error with a descriptive message on failure,
 * mirroring Python's AssertionError. Overloads handle scalars and
 * `ndarray<T>`.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_TESTING_HPP
#define NP_TESTING_HPP

#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{
  namespace testing
  {

    namespace detail
    {
      template <typename T>
      inline auto to_string(const T& v) -> std::string
      {
        std::ostringstream oss;
        oss << v;
        return oss.str();
      }

      template <typename T>
      inline auto to_string(const std::complex<T>& v) -> std::string
      {
        std::ostringstream oss;
        oss << "(" << v.real() << "," << v.imag() << ")";
        return oss.str();
      }

      inline void fail(const std::string& msg)
      {
        throw std::runtime_error("AssertionError: " + msg);
      }

      template <typename T>
      inline auto almost_equal_scalar(T a, T b, int decimal) -> bool
      {
        if constexpr (std::is_floating_point_v<T>)
        {
          double tol = std::pow(10.0, -decimal);
          return std::abs(a - b) <= tol;
        }
        else if constexpr (
            std::is_same_v<T, std::complex<float>>
            || std::is_same_v<T, std::complex<double>>
            || std::is_same_v<T, std::complex<long double>>)
        {
          double tol = std::pow(10.0, -decimal);
          return std::abs(a - b) <= tol;
        }
        else
        {
          return a == b;
        }
      }

      inline auto allclose_scalar(double a, double b, double rtol, double atol) -> bool
      {
        return std::abs(a - b) <= atol + rtol * std::abs(b);
      }
    } // namespace detail

    /**
     * @brief Assert value is true (np.testing.assert_).
     *
     * Reference: numpy-reference/reference/generated/numpy.testing.assert_.html
     */
    NP_API inline void assert_(bool val, const std::string& msg = "")
    {
      if (!val)
      {
        detail::fail(msg.empty() ? "assert_ failed" : msg);
      }
    }

    /**
     * @brief Assert two objects are equal (np.testing.assert_equal).
     *
     * Reference: numpy-reference/reference/generated/numpy.testing.assert_equal.html
     */
    NP_API template <typename T, typename U>
    inline void
    assert_equal(const T& actual, const U& desired, const std::string& err_msg = "")
    {
      if (!(actual == desired))
      {
        std::string m = err_msg.empty()
            ? detail::to_string(actual) + " != " + detail::to_string(desired)
            : err_msg;
        detail::fail(m);
      }
    }

    NP_API template <typename T>
    inline void assert_equal(
        const ndarray<T>& actual,
        const ndarray<T>& desired,
        const std::string& err_msg = "")
    {
      if (actual.shape != desired.shape)
      {
        detail::fail(err_msg.empty() ? "shape mismatch" : err_msg);
      }
      for (std::size_t i = 0; i < actual.size(); ++i)
      {
        if (!(actual.data()[actual._flat_logical(i)]
              == desired.data()[desired._flat_logical(i)]))
        {
          detail::fail(
              err_msg.empty() ? "array_equal failed at " + std::to_string(i) : err_msg);
        }
      }
    }

    /**
     * @brief Assert almost equal to decimal (np.testing.assert_almost_equal).
     *
     * Reference:
     * numpy-reference/reference/generated/numpy.testing.assert_almost_equal.html
     */
    NP_API template <typename T, typename U>
    inline void assert_almost_equal(
        const T& actual,
        const U& desired,
        int decimal = 6,
        const std::string& err_msg = "")
    {
      if (!detail::almost_equal_scalar(actual, desired, decimal))
      {
        detail::fail(err_msg.empty() ? "almost_equal failed" : err_msg);
      }
    }

    NP_API template <typename T>
    inline void assert_almost_equal(
        const ndarray<T>& actual,
        const ndarray<T>& desired,
        int decimal = 6,
        const std::string& err_msg = "")
    {
      if (actual.shape != desired.shape)
      {
        detail::fail("shape mismatch");
      }
      for (std::size_t i = 0; i < actual.size(); ++i)
      {
        if (!detail::almost_equal_scalar(
                actual.data()[actual._flat_logical(i)],
                desired.data()[desired._flat_logical(i)],
                decimal))
        {
          detail::fail(
              err_msg.empty() ? "array almost_equal failed at " + std::to_string(i)
                              : err_msg);
        }
      }
    }

    /**
     * @brief Assert approx equal to significant digits (np.testing.assert_approx_equal).
     */
    NP_API template <typename T>
    inline void assert_approx_equal(
        const T& actual,
        const T& desired,
        int significant = 7,
        const std::string& err_msg = "")
    {
      double tol = std::pow(10.0, -significant);
      double diff = std::abs(static_cast<double>(actual) - static_cast<double>(desired));
      double scale = std::max(
          std::abs(static_cast<double>(actual)), std::abs(static_cast<double>(desired)));
      if (scale == 0)
      {
        if (diff > tol)
        {
          detail::fail(err_msg.empty() ? "approx_equal failed" : err_msg);
        }
        return;
      }
      if (diff / scale > tol)
      {
        detail::fail(err_msg.empty() ? "approx_equal failed" : err_msg);
      }
    }

    /**
     * @brief Assert array equal (np.testing.assert_array_equal).
     *
     * Reference:
     * numpy-reference/reference/generated/numpy.testing.assert_array_equal.html
     */
    NP_API template <typename T>
    inline void assert_array_equal(
        const ndarray<T>& actual,
        const ndarray<T>& desired,
        const std::string& err_msg = "")
    {
      assert_equal(actual, desired, err_msg);
    }

    /**
     * @brief Assert array almost equal (np.testing.assert_array_almost_equal).
     *
     * Reference:
     * numpy-reference/reference/generated/numpy.testing.assert_array_almost_equal.html
     */
    NP_API template <typename T>
    inline void assert_array_almost_equal(
        const ndarray<T>& actual,
        const ndarray<T>& desired,
        int decimal = 6,
        const std::string& err_msg = "")
    {
      assert_almost_equal(actual, desired, decimal, err_msg);
    }

    /**
     * @brief Assert array less (np.testing.assert_array_less).
     *
     * Reference: numpy-reference/reference/generated/numpy.testing.assert_array_less.html
     */
    NP_API template <typename T>
    inline void assert_array_less(
        const ndarray<T>& x, const ndarray<T>& y, const std::string& err_msg = "")
    {
      if (x.shape != y.shape)
      {
        detail::fail("shape mismatch");
      }
      for (std::size_t i = 0; i < x.size(); ++i)
      {
        if (!(x.data()[x._flat_logical(i)] < y.data()[y._flat_logical(i)]))
        {
          detail::fail(
              err_msg.empty() ? "array_less failed at " + std::to_string(i) : err_msg);
        }
      }
    }

    /**
     * @brief Assert allclose (np.testing.assert_allclose).
     *
     * Reference: numpy-reference/reference/generated/numpy.testing.assert_allclose.html
     */
    NP_API template <typename T>
    inline void assert_allclose(
        const ndarray<T>& actual,
        const ndarray<T>& desired,
        double rtol = 1e-7,
        double atol = 0,
        const std::string& err_msg = "")
    {
      if (actual.shape != desired.shape)
      {
        detail::fail("shape mismatch");
      }
      for (std::size_t i = 0; i < actual.size(); ++i)
      {
        double a = static_cast<double>(actual.data()[actual._flat_logical(i)]);
        double b = static_cast<double>(desired.data()[desired._flat_logical(i)]);
        if (!detail::allclose_scalar(a, b, rtol, atol))
        {
          detail::fail(
              err_msg.empty() ? "allclose failed at " + std::to_string(i) : err_msg);
        }
      }
    }

    NP_API template <typename T, typename U>
    inline void assert_allclose(
        T actual,
        U desired,
        double rtol = 1e-7,
        double atol = 0,
        const std::string& err_msg = "")
    {
      double a = static_cast<double>(actual);
      double b = static_cast<double>(desired);
      if (!detail::allclose_scalar(a, b, rtol, atol))
      {
        detail::fail(err_msg.empty() ? "allclose scalar failed" : err_msg);
      }
    }

    /**
     * @brief Assert string equal (np.testing.assert_string_equal).
     */
    NP_API inline void assert_string_equal(
        const std::string& actual,
        const std::string& desired,
        const std::string& err_msg = "")
    {
      if (actual != desired)
      {
        detail::fail(err_msg.empty() ? "'" + actual + "' != '" + desired + "'" : err_msg);
      }
    }

    /**
     * @brief Assert raises (np.testing.assert_raises).
     *
     * Reference: numpy-reference/reference/generated/numpy.testing.assert_raises.html
     */
    NP_API template <typename Exc, typename F, typename... Args>
    inline void assert_raises(F&& func, Args&&... args)
    {
      try
      {
        std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
      }
      catch (const Exc&)
      {
        return;
      }
      catch (...)
      {
        detail::fail("assert_raises: wrong exception type");
      }
      detail::fail("assert_raises: no exception thrown");
    }

    /**
     * @brief Assert raises with regex (np.testing.assert_raises_regex).
     */
    NP_API template <typename Exc, typename F, typename... Args>
    inline void assert_raises_regex(const std::string& pattern, F&& func, Args&&... args)
    {
      try
      {
        std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
      }
      catch (const Exc& e)
      {
        std::regex re(pattern);
        if (!std::regex_search(e.what(), re))
        {
          detail::fail(
              "assert_raises_regex: message '" + std::string(e.what())
              + "' does not match '" + pattern + "'");
        }
        return;
      }
      catch (...)
      {
        detail::fail("assert_raises_regex: wrong exception");
      }
      detail::fail("assert_raises_regex: no exception");
    }

    /**
     * @brief Assert warns (np.testing.assert_warns) – checks that callable throws
     * warning.
     *
     * In C++ warnings are exceptions derived from NumpyError; we check type.
     */
    NP_API template <typename Warn, typename F, typename... Args>
    inline void assert_warns(F&& func, Args&&... args)
    {
      assert_raises<Warn>(std::forward<F>(func), std::forward<Args>(args)...);
    }

    /**
     * @brief Print assert equal (np.testing.print_assert_equal).
     */
    NP_API template <typename T, typename U>
    inline void
    print_assert_equal(const std::string& test_string, const T& actual, const U& desired)
    {
      try
      {
        assert_equal(actual, desired);
      }
      catch (const std::exception& e)
      {
        std::cerr << test_string << " failed: " << e.what() << "\n";
        throw;
      }
    }

    // Not recommended but provided for parity
    NP_API inline void assert_array_almost_equal_nulp(
        const ndarray<double>& x, const ndarray<double>& y, int nulp = 1)
    {
      if (x.shape != y.shape)
      {
        detail::fail("shape mismatch");
      }
      for (std::size_t i = 0; i < x.size(); ++i)
      {
        double a = x.data()[x._flat_logical(i)];
        double b = y.data()[y._flat_logical(i)];
        double diff = std::abs(a - b);
        double ulp =
            std::numeric_limits<double>::epsilon() * std::max(std::abs(a), std::abs(b));
        if (diff > nulp * ulp)
        {
          detail::fail("nulp failed at " + std::to_string(i));
        }
      }
    }

    NP_API inline void assert_array_max_ulp(
        const ndarray<double>& a, const ndarray<double>& b, int maxulp = 1)
    {
      assert_array_almost_equal_nulp(a, b, maxulp);
    }

  } // namespace testing
} // namespace np

#endif // NP_TESTING_HPP
