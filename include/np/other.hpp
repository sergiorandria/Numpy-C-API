/**
 * @file other.hpp
 * @brief Miscellaneous routines (np.who, np.disp, np.info, etc.).
 *
 * Reference: https://numpy.org/doc/2.2/reference/routines.other.html
 *
 * Provides lightweight stubs for NumPy's miscellaneous introspection
 * helpers that have no direct C++ analogue but are required for
 * 100% API coverage.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_OTHER_HPP
#define NP_OTHER_HPP

#include <iostream>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np
{

  /**
   * @brief Print the NumPy arrays in the given dictionary (np.who).
   *
   * Reference: numpy-reference/reference/generated/numpy.who.html
   *
   * In Python `who` introspects the caller's namespace; here it is a
   * no-op that prints the passed map keys when available.
   */
  NP_API inline void who(const std::vector<std::string>& names = {})
  {
    std::cout << "who: ";
    for (auto& n : names)
      std::cout << n << " ";
    std::cout << "\n";
  }

  /**
   * @brief Display a message (np.disp).
   *
   * Reference: numpy-reference/reference/generated/numpy.disp.html
   */
  NP_API inline void disp(const std::string& msg)
  {
    std::cout << msg << "\n";
  }

  /**
   * @brief Get help on object (np.info).
   *
   * Reference: numpy-reference/reference/generated/numpy.info.html
   */
  NP_API inline std::string info(const std::string& obj = "")
  {
    if (obj.empty())
      return "NumPy C++ API – see include/np/*.hpp";
    return "info: " + obj + " – NumPy C++ API stub";
  }

  /**
   * @brief Print source of object (np.source).
   */
  NP_API inline void source(const std::string& obj = "")
  {
    std::cout << "source(" << obj << "): C++ header-only implementation\n";
  }

  /**
   * @brief Search docstrings (np.lookfor).
   */
  NP_API inline std::string lookfor(const std::string& what)
  {
    return "lookfor: search for '" + what + "' – use grep on include/np/*.hpp";
  }

  /**
   * @brief Deprecated decorator stub (np.deprecate).
   */
  NP_API inline void deprecate(const std::string& msg = "")
  {
    (void)msg;
  }

  NP_API inline void deprecate_with_doc(const std::string& msg = "")
  {
    (void)msg;
  }

  /**
   * @brief Byte bounds of array (np.byte_bounds).
   *
   * Returns {low, high} byte addresses of the array's data buffer.
   * Reference: numpy-reference/reference/generated/numpy.byte_bounds.html
   */
  template <typename T>
  NP_API inline std::pair<const char*, const char*> byte_bounds(const ndarray<T>& a)
  {
    if (a.size() == 0)
      return {nullptr, nullptr};
    const char* low = reinterpret_cast<const char*>(a.data().data());
    const char* high = low + a.size() * sizeof(T);
    return {low, high};
  }

  /**
   * @brief Show build config (np.show_config).
   *
   * Reference: numpy-reference/reference/generated/numpy.show_config.html
   */
  NP_API inline std::string show_config()
  {
    return "np C++ API: header-only, C++20, SIMD auto-detected";
  }

  /**
   * @brief Einsum path optimizer stub (np.einsum_path).
   *
   * Reference: numpy-reference/reference/generated/numpy.einsum_path.html
   *
   * Returns the einsum contraction path as a string and a dummy list.
   */
  NP_API inline std::pair<std::string, std::vector<std::vector<int>>>
  einsum_path(const std::string& subscripts)
  {
    (void)subscripts;
    return {"einsum_path: optimized (stub)", {}};
  }

} // namespace np

#endif // NP_OTHER_HPP
