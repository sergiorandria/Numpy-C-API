/**
 * @file char.hpp
 * @brief String operations for arrays of std::string (numpy.char module).
 *
 * Implements element-wise string manipulation functions matching numpy.char
 * semantics. All functions operate on Ndarray<std::string> and return either
 * string arrays, boolean arrays, or integer arrays depending on the operation.
 *
 * Function signatures, parameter order, and default values mirror the
 * numpy.char pages in numpy-reference/reference/generated/numpy.char.*.
 *
 * Implementation notes:
 *  - encode() and decode() are no-ops (C++ std::string is already byte-based)
 *  - isdecimal() and isnumeric() use isdigit() approximation (C++ limitation)
 *  - split(), rsplit(), splitlines() return flattened arrays (not object
 * arrays)
 *  - translate() uses simplified 256-character table (not dict-based)
 *  - chararray class is deprecated (per NumPy 2.5) but provided for
 * compatibility
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_CHAR_HPP
#define NP_CHAR_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "creation.hpp"
#include "ndarray.hpp"

namespace np {
namespace ch {

/* Internal helpers - character classification and string utilities */
namespace detail {

/* Validate that two arrays have matching shapes */
inline void validate_shapes(const Ndarray<std::string> &a,
                            const Ndarray<std::string> &b,
                            const char *func_name) {
  if (a.shape != b.shape) {
    throw std::invalid_argument(std::string(func_name) + ": shape mismatch");
  }
}

/* Check if string contains cased characters and all are lowercase */
inline bool str_islower(const std::string &s) {
  bool has_cased = false;
  for (char c : s) {
    if (std::isalpha(static_cast<unsigned char>(c))) {
      has_cased = true;
      if (!std::islower(static_cast<unsigned char>(c))) {
        return false;
      }
    }
  }
  return has_cased;
}

/* Check if string contains cased characters and all are uppercase */
inline bool str_isupper(const std::string &s) {
  bool has_cased = false;
  for (char c : s) {
    if (std::isalpha(static_cast<unsigned char>(c))) {
      has_cased = true;
      if (!std::isupper(static_cast<unsigned char>(c))) {
        return false;
      }
    }
  }
  return has_cased;
}

/* Check if string is titlecased */
inline bool str_istitle(const std::string &s) {
  bool in_word = false;
  bool has_cased = false;

  for (char c : s) {
    bool is_alpha = std::isalpha(static_cast<unsigned char>(c));
    if (is_alpha) {
      has_cased = true;
      if (in_word) {
        if (!std::islower(static_cast<unsigned char>(c))) {
          return false;
        }
      } else {
        if (!std::isupper(static_cast<unsigned char>(c))) {
          return false;
        }
        in_word = true;
      }
    } else {
      in_word = false;
    }
  }
  return has_cased;
}

} /* namespace detail */

/* String Operations */

/**
 * @brief Return element-wise string concatenation.
 *
 * Performs element-wise concatenation: result[i] = x1[i] + x2[i].
 * Both input arrays must have identical shapes.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.add.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Element-wise concatenated array
 * @throws std::invalid_argument if x1.shape != x2.shape
 */
NP_API inline auto add(const Ndarray<std::string> &x1,
                       const Ndarray<std::string> &x2) -> Ndarray<std::string> {
  detail::validate_shapes(x1, x2, "char.add");

  Ndarray<std::string> result = empty<std::string>(x1.shape);
  for (std::size_t i = 0; i < x1.size(); ++i) {
    result.data()[i] = x1.data()[i] + x2.data()[i];
  }
  return result;
}

/**
 * @brief Return (a * i), that is string multiple concatenation, element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.multiply.html
 *
 * @param a String array
 * @param i Integer array (repeat count for each string)
 * @return Element-wise string repetition
 */
NP_API inline auto multiply(const Ndarray<std::string> &a,
                            const Ndarray<int> &i) -> Ndarray<std::string> {
  if (a.shape != i.shape) {
    throw std::invalid_argument("multiply: arrays must have the same shape");
  }
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t idx = 0; idx < a.size(); ++idx) {
    int count = i.data()[idx];
    if (count < 0)
      count = 0;
    std::string repeated;
    repeated.reserve(a.data()[idx].size() * count);
    for (int j = 0; j < count; ++j) {
      repeated += a.data()[idx];
    }
    result.data()[idx] = repeated;
  }
  return result;
}

/**
 * @brief Return (a % i), that is string formatting, element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.mod.html
 *
 * @param a Format string array
 * @param values Values to substitute
 * @return Formatted strings
 */
NP_API inline auto mod(const Ndarray<std::string> &a,
                       const Ndarray<std::string> &values)
    -> Ndarray<std::string> {
  if (a.shape != values.shape) {
    throw std::invalid_argument("mod: arrays must have the same shape");
  }
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    // Simplified: just replace first %s with value
    std::string fmt = a.data()[i];
    std::string val = values.data()[i];
    std::size_t pos = fmt.find("%s");
    if (pos != std::string::npos) {
      fmt.replace(pos, 2, val);
    }
    result.data()[i] = fmt;
  }
  return result;
}

/**
 * @brief Return a copy with only the first character capitalized.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.capitalize.html
 *
 * @param a Input string array
 * @return Array with capitalized strings
 */
NP_API inline auto capitalize(const Ndarray<std::string> &a)
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    if (!s.empty()) {
      s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
      for (std::size_t j = 1; j < s.size(); ++j) {
        s[j] =
            static_cast<char>(std::tolower(static_cast<unsigned char>(s[j])));
      }
    }
    result.data()[i] = s;
  }
  return result;
}

/**
 * @brief Return a copy centered in a string of length width.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.center.html
 *
 * @param a Input string array
 * @param width Minimum width of resulting string
 * @param fillchar Padding character (default: space)
 * @return Array with centered strings
 */
NP_API inline auto center(const Ndarray<std::string> &a, int width,
                          char fillchar = ' ') -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    if (static_cast<int>(s.size()) >= width) {
      result.data()[i] = s;
    } else {
      int total_pad = width - static_cast<int>(s.size());
      int left_pad = total_pad / 2;
      int right_pad = total_pad - left_pad;
      result.data()[i] = std::string(left_pad, fillchar) + s +
                         std::string(right_pad, fillchar);
    }
  }
  return result;
}

/**
 * @brief Calls str.lower() for each element.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.lower.html
 *
 * @param a Input string array
 * @return Array with lowercase strings
 */
NP_API inline auto lower(const Ndarray<std::string> &a)
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    result.data()[i] = s;
  }
  return result;
}

/**
 * @brief Calls str.upper() for each element.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.upper.html
 *
 * @param a Input string array
 * @return Array with uppercase strings
 */
NP_API inline auto upper(const Ndarray<std::string> &a)
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    result.data()[i] = s;
  }
  return result;
}

/**
 * @brief Return a copy with leading and trailing characters removed.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.strip.html
 *
 * @param a Input string array
 * @param chars Characters to remove (default: whitespace)
 * @return Array with stripped strings
 */
NP_API inline auto strip(const Ndarray<std::string> &a,
                         const std::string &chars = " \t\n\r")
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    // Left strip
    std::size_t start = s.find_first_not_of(chars);
    if (start == std::string::npos) {
      result.data()[i] = "";
      continue;
    }
    // Right strip
    std::size_t end = s.find_last_not_of(chars);
    result.data()[i] = s.substr(start, end - start + 1);
  }
  return result;
}

/**
 * @brief Return a copy with leading characters removed.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.lstrip.html
 *
 * @param a Input string array
 * @param chars Characters to remove (default: whitespace)
 * @return Array with left-stripped strings
 */
NP_API inline auto lstrip(const Ndarray<std::string> &a,
                          const std::string &chars = " \t\n\r")
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    std::size_t start = s.find_first_not_of(chars);
    if (start == std::string::npos) {
      result.data()[i] = "";
    } else {
      result.data()[i] = s.substr(start);
    }
  }
  return result;
}

/**
 * @brief Return a copy with trailing characters removed.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rstrip.html
 *
 * @param a Input string array
 * @param chars Characters to remove (default: whitespace)
 * @return Array with right-stripped strings
 */
NP_API inline auto rstrip(const Ndarray<std::string> &a,
                          const std::string &chars = " \t\n\r")
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    std::size_t end = s.find_last_not_of(chars);
    if (end == std::string::npos) {
      result.data()[i] = "";
    } else {
      result.data()[i] = s.substr(0, end + 1);
    }
  }
  return result;
}

/**
 * @brief Return a copy with all case-swapped characters.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.swapcase.html
 *
 * @param a Input string array
 * @return Array with swapped case strings
 */
NP_API inline auto swapcase(const Ndarray<std::string> &a)
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    for (char &c : s) {
      if (std::islower(static_cast<unsigned char>(c))) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      } else if (std::isupper(static_cast<unsigned char>(c))) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      }
    }
    result.data()[i] = s;
  }
  return result;
}

/**
 * @brief Return a titlecased version of the string.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.title.html
 *
 * @param a Input string array
 * @return Array with titlecased strings
 */
NP_API inline auto title(const Ndarray<std::string> &a)
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    bool capitalize_next = true;
    for (char &c : s) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        if (capitalize_next) {
          c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
          capitalize_next = false;
        } else {
          c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
      } else {
        capitalize_next = true;
      }
    }
    result.data()[i] = s;
  }
  return result;
}

/**
 * @brief Pad each string with zeros on the left.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.zfill.html
 *
 * @param a Input string array
 * @param width Minimum width of resulting string
 * @return Array with zero-filled strings
 */
NP_API inline auto zfill(const Ndarray<std::string> &a, int width)
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    if (static_cast<int>(s.size()) >= width) {
      result.data()[i] = s;
    } else {
      int pad = width - static_cast<int>(s.size());
      // Handle sign
      if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
        result.data()[i] = s[0] + std::string(pad, '0') + s.substr(1);
      } else {
        result.data()[i] = std::string(pad, '0') + s;
      }
    }
  }
  return result;
}

/**
 * @brief Left-justify strings.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.ljust.html
 *
 * @param a Input string array
 * @param width Minimum width of resulting string
 * @param fillchar Padding character (default: space)
 * @return Array with left-justified strings
 */
NP_API inline auto ljust(const Ndarray<std::string> &a, int width,
                         char fillchar = ' ') -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    if (static_cast<int>(s.size()) >= width) {
      result.data()[i] = s;
    } else {
      int pad = width - static_cast<int>(s.size());
      result.data()[i] = s + std::string(pad, fillchar);
    }
  }
  return result;
}

/**
 * @brief Right-justify strings.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rjust.html
 *
 * @param a Input string array
 * @param width Minimum width of resulting string
 * @param fillchar Padding character (default: space)
 * @return Array with right-justified strings
 */
NP_API inline auto rjust(const Ndarray<std::string> &a, int width,
                         char fillchar = ' ') -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    if (static_cast<int>(s.size()) >= width) {
      result.data()[i] = s;
    } else {
      int pad = width - static_cast<int>(s.size());
      result.data()[i] = std::string(pad, fillchar) + s;
    }
  }
  return result;
}

/**
 * @brief Replace occurrences of substring with new substring.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.replace.html
 *
 * @param a Input string array
 * @param old Substring to replace
 * @param new_str Replacement substring
 * @param count Maximum number of replacements (-1 = all)
 * @return Array with replaced strings
 */
NP_API inline auto replace(const Ndarray<std::string> &a,
                           const std::string &old, const std::string &new_str,
                           int count = -1) -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    std::size_t pos = 0;
    int replacements = 0;
    while ((pos = s.find(old, pos)) != std::string::npos) {
      s.replace(pos, old.size(), new_str);
      pos += new_str.size();
      ++replacements;
      if (count >= 0 && replacements >= count)
        break;
    }
    result.data()[i] = s;
  }
  return result;
}

/* Comparison Functions */

/**
 * @brief Return (x1 == x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.equal.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto equal(const Ndarray<std::string> &x1,
                         const Ndarray<std::string> &x2) -> Ndarray<bool> {
  if (x1.shape != x2.shape) {
    throw std::invalid_argument("equal: arrays must have the same shape");
  }
  Ndarray<bool> result = empty<bool>(x1.shape);
  for (std::size_t i = 0; i < x1.size(); ++i) {
    result.data()[i] = (x1.data()[i] == x2.data()[i]);
  }
  return result;
}

/**
 * @brief Return (x1 != x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.not_equal.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto not_equal(const Ndarray<std::string> &x1,
                             const Ndarray<std::string> &x2) -> Ndarray<bool> {
  if (x1.shape != x2.shape) {
    throw std::invalid_argument("not_equal: arrays must have the same shape");
  }
  Ndarray<bool> result = empty<bool>(x1.shape);
  for (std::size_t i = 0; i < x1.size(); ++i) {
    result.data()[i] = (x1.data()[i] != x2.data()[i]);
  }
  return result;
}

/**
 * @brief Return (x1 >= x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.greater_equal.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto greater_equal(const Ndarray<std::string> &x1,
                                 const Ndarray<std::string> &x2)
    -> Ndarray<bool> {
  if (x1.shape != x2.shape) {
    throw std::invalid_argument(
        "greater_equal: arrays must have the same shape");
  }
  Ndarray<bool> result = empty<bool>(x1.shape);
  for (std::size_t i = 0; i < x1.size(); ++i) {
    result.data()[i] = (x1.data()[i] >= x2.data()[i]);
  }
  return result;
}

/**
 * @brief Return (x1 <= x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.less_equal.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto less_equal(const Ndarray<std::string> &x1,
                              const Ndarray<std::string> &x2) -> Ndarray<bool> {
  if (x1.shape != x2.shape) {
    throw std::invalid_argument("less_equal: arrays must have the same shape");
  }
  Ndarray<bool> result = empty<bool>(x1.shape);
  for (std::size_t i = 0; i < x1.size(); ++i) {
    result.data()[i] = (x1.data()[i] <= x2.data()[i]);
  }
  return result;
}

/**
 * @brief Return (x1 > x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.greater.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto greater(const Ndarray<std::string> &x1,
                           const Ndarray<std::string> &x2) -> Ndarray<bool> {
  if (x1.shape != x2.shape) {
    throw std::invalid_argument("greater: arrays must have the same shape");
  }
  Ndarray<bool> result = empty<bool>(x1.shape);
  for (std::size_t i = 0; i < x1.size(); ++i) {
    result.data()[i] = (x1.data()[i] > x2.data()[i]);
  }
  return result;
}

/**
 * @brief Return (x1 < x2) element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.less.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Boolean array
 */
NP_API inline auto less(const Ndarray<std::string> &x1,
                        const Ndarray<std::string> &x2) -> Ndarray<bool> {
  if (x1.shape != x2.shape) {
    throw std::invalid_argument("less: arrays must have the same shape");
  }
  Ndarray<bool> result = empty<bool>(x1.shape);
  for (std::size_t i = 0; i < x1.size(); ++i) {
    result.data()[i] = (x1.data()[i] < x2.data()[i]);
  }
  return result;
}

/* Information Functions */

/**
 * @brief Returns the number of non-overlapping occurrences of substring.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.count.html
 *
 * @param a Input string array
 * @param sub Substring to count
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of counts
 */
NP_API inline auto count(const Ndarray<std::string> &a, const std::string &sub,
                         int start = 0, int end = -1) -> Ndarray<int> {
  Ndarray<int> result = empty<int>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    int len = static_cast<int>(s.size());
    int e = (end < 0) ? len : std::min(end, len);
    int st = std::max(0, start);

    int cnt = 0;
    std::size_t pos = st;
    while (pos < static_cast<std::size_t>(e) &&
           (pos = s.find(sub, pos)) != std::string::npos &&
           pos < static_cast<std::size_t>(e)) {
      ++cnt;
      pos += sub.size();
    }
    result.data()[i] = cnt;
  }
  return result;
}

/**
 * @brief Returns a boolean array which is True where the string element ends
 * with suffix.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.endswith.html
 *
 * @param a Input string array
 * @param suffix Suffix to check
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Boolean array
 */
NP_API inline auto endswith(const Ndarray<std::string> &a,
                            const std::string &suffix, int start = 0,
                            int end = -1) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    int len = static_cast<int>(s.size());
    int e = (end < 0) ? len : std::min(end, len);
    int st = std::max(0, start);

    std::string sub = s.substr(st, e - st);
    result.data()[i] =
        (sub.size() >= suffix.size() &&
         sub.compare(sub.size() - suffix.size(), suffix.size(), suffix) == 0);
  }
  return result;
}

/**
 * @brief Returns a boolean array which is True where the string element starts
 * with prefix.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.startswith.html
 *
 * @param a Input string array
 * @param prefix Prefix to check
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Boolean array
 */
NP_API inline auto startswith(const Ndarray<std::string> &a,
                              const std::string &prefix, int start = 0,
                              int end = -1) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    int len = static_cast<int>(s.size());
    int e = (end < 0) ? len : std::min(end, len);
    int st = std::max(0, start);

    std::string sub = s.substr(st, e - st);
    result.data()[i] = (sub.size() >= prefix.size() &&
                        sub.compare(0, prefix.size(), prefix) == 0);
  }
  return result;
}

/**
 * @brief Finds the lowest index of the substring.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.find.html
 *
 * @param a Input string array
 * @param sub Substring to find
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of indices (or -1 if not found)
 */
NP_API inline auto find(const Ndarray<std::string> &a, const std::string &sub,
                        int start = 0, int end = -1) -> Ndarray<int> {
  Ndarray<int> result = empty<int>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    int len = static_cast<int>(s.size());
    int e = (end < 0) ? len : std::min(end, len);
    int st = std::max(0, start);

    std::size_t pos = s.find(sub, st);
    if (pos != std::string::npos && pos < static_cast<std::size_t>(e)) {
      result.data()[i] = static_cast<int>(pos);
    } else {
      result.data()[i] = -1;
    }
  }
  return result;
}

/**
 * @brief Finds the highest index of the substring.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rfind.html
 *
 * @param a Input string array
 * @param sub Substring to find
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of indices (or -1 if not found)
 */
NP_API inline auto rfind(const Ndarray<std::string> &a, const std::string &sub,
                         int start = 0, int end = -1) -> Ndarray<int> {
  Ndarray<int> result = empty<int>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    int len = static_cast<int>(s.size());
    int e = (end < 0) ? len : std::min(end, len);
    int st = std::max(0, start);

    std::size_t pos = s.rfind(sub, e - 1);
    if (pos != std::string::npos && pos >= static_cast<std::size_t>(st)) {
      result.data()[i] = static_cast<int>(pos);
    } else {
      result.data()[i] = -1;
    }
  }
  return result;
}

/**
 * @brief Return the string length element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.str_len.html
 *
 * @param a Input string array
 * @return Array of string lengths
 */
NP_API inline auto str_len(const Ndarray<std::string> &a) -> Ndarray<int> {
  Ndarray<int> result = empty<int>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    result.data()[i] = static_cast<int>(a.data()[i].size());
  }
  return result;
}

/* Testing Functions */

/**
 * @brief Returns true for each element if all characters are alphabetic.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isalpha.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isalpha(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    result.data()[i] =
        !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
          return std::isalpha(c);
        });
  }
  return result;
}

/**
 * @brief Returns true for each element if all characters are alphanumeric.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isalnum.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isalnum(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    result.data()[i] =
        !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
          return std::isalnum(c);
        });
  }
  return result;
}

/**
 * @brief Returns true for each element if all characters are digits.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isdigit.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isdigit(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    result.data()[i] =
        !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
          return std::isdigit(c);
        });
  }
  return result;
}

/**
 * @brief Returns true for each element if all cased characters are lowercase.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.islower.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto islower(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    bool has_cased = false;
    bool all_lower = true;
    for (char c : s) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        has_cased = true;
        if (!std::islower(static_cast<unsigned char>(c))) {
          all_lower = false;
          break;
        }
      }
    }
    result.data()[i] = has_cased && all_lower;
  }
  return result;
}

/**
 * @brief Returns true for each element if all cased characters are uppercase.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isupper.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isupper(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    bool has_cased = false;
    bool all_upper = true;
    for (char c : s) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        has_cased = true;
        if (!std::isupper(static_cast<unsigned char>(c))) {
          all_upper = false;
          break;
        }
      }
    }
    result.data()[i] = has_cased && all_upper;
  }
  return result;
}

/**
 * @brief Returns true for each element if all characters are whitespace.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isspace.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isspace(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    result.data()[i] =
        !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
          return std::isspace(c);
        });
  }
  return result;
}

/**
 * @brief Returns true for each element if string is in titlecase.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.istitle.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto istitle(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    bool in_word = false;
    bool has_cased = false;
    bool is_title = true;

    for (char c : s) {
      bool is_alpha = std::isalpha(static_cast<unsigned char>(c));
      if (is_alpha) {
        has_cased = true;
        if (in_word) {
          if (!std::islower(static_cast<unsigned char>(c))) {
            is_title = false;
            break;
          }
        } else {
          if (!std::isupper(static_cast<unsigned char>(c))) {
            is_title = false;
            break;
          }
          in_word = true;
        }
      } else {
        in_word = false;
      }
    }
    result.data()[i] = has_cased && is_title;
  }
  return result;
}

/**
 * @brief Returns true for each element if all characters are decimal.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isdecimal.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isdecimal(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    // C++ doesn't have direct isdecimal; use isdigit as approximation
    result.data()[i] =
        !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
          return std::isdigit(c);
        });
  }
  return result;
}

/**
 * @brief Returns true for each element if all characters are numeric.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.isnumeric.html
 *
 * @param a Input string array
 * @return Boolean array
 */
NP_API inline auto isnumeric(const Ndarray<std::string> &a) -> Ndarray<bool> {
  Ndarray<bool> result = empty<bool>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    // C++ doesn't have direct isnumeric; use isdigit as approximation
    result.data()[i] =
        !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
          return std::isdigit(c);
        });
  }
  return result;
}

/* Split/Join Functions */

/**
 * @brief Join strings with separator.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.join.html
 *
 * @param sep Separator string array
 * @param seq Sequence string array
 * @return Array with joined strings
 */
NP_API inline auto join(const Ndarray<std::string> &sep,
                        const Ndarray<std::string> &seq)
    -> Ndarray<std::string> {
  if (sep.shape != seq.shape) {
    throw std::invalid_argument("join: arrays must have the same shape");
  }
  Ndarray<std::string> result = empty<std::string>(sep.shape);
  for (std::size_t i = 0; i < sep.size(); ++i) {
    const std::string &s = seq.data()[i];
    const std::string &separator = sep.data()[i];
    std::string joined;
    for (std::size_t j = 0; j < s.size(); ++j) {
      if (j > 0)
        joined += separator;
      joined += s[j];
    }
    result.data()[i] = joined;
  }
  return result;
}

/**
 * @brief Expand tabs in each string element.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.expandtabs.html
 *
 * @param a Input string array
 * @param tabsize Number of spaces per tab (default: 8)
 * @return Array with expanded tabs
 */
NP_API inline auto expandtabs(const Ndarray<std::string> &a, int tabsize = 8)
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    std::string expanded;
    int col = 0;
    for (char c : s) {
      if (c == '\t') {
        int spaces = tabsize - (col % tabsize);
        expanded.append(spaces, ' ');
        col += spaces;
      } else if (c == '\n' || c == '\r') {
        expanded += c;
        col = 0;
      } else {
        expanded += c;
        ++col;
      }
    }
    result.data()[i] = expanded;
  }
  return result;
}

/**
 * @brief Partition each element around sep.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.partition.html
 *
 * @param a Input string array
 * @param sep Separator string
 * @return Array of tuples (before, sep, after) - flattened as 3x larger array
 */
NP_API inline auto partition(const Ndarray<std::string> &a,
                             const std::string &sep) -> Ndarray<std::string> {
  // Returns array with 3x elements: [before0, sep0, after0, before1, sep1,
  // after1, ...]
  std::vector<int> new_shape = a.shape;
  new_shape.back() *= 3;
  Ndarray<std::string> result = empty<std::string>(new_shape);

  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    std::size_t pos = s.find(sep);
    if (pos != std::string::npos) {
      result.data()[i * 3] = s.substr(0, pos);
      result.data()[i * 3 + 1] = sep;
      result.data()[i * 3 + 2] = s.substr(pos + sep.size());
    } else {
      result.data()[i * 3] = s;
      result.data()[i * 3 + 1] = "";
      result.data()[i * 3 + 2] = "";
    }
  }
  return result;
}

/**
 * @brief Partition each element around rightmost sep.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rpartition.html
 *
 * @param a Input string array
 * @param sep Separator string
 * @return Array of tuples (before, sep, after) - flattened as 3x larger array
 */
NP_API inline auto rpartition(const Ndarray<std::string> &a,
                              const std::string &sep) -> Ndarray<std::string> {
  std::vector<int> new_shape = a.shape;
  new_shape.back() *= 3;
  Ndarray<std::string> result = empty<std::string>(new_shape);

  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    std::size_t pos = s.rfind(sep);
    if (pos != std::string::npos) {
      result.data()[i * 3] = s.substr(0, pos);
      result.data()[i * 3 + 1] = sep;
      result.data()[i * 3 + 2] = s.substr(pos + sep.size());
    } else {
      result.data()[i * 3] = "";
      result.data()[i * 3 + 1] = "";
      result.data()[i * 3 + 2] = s;
    }
  }
  return result;
}

/**
 * @brief Like find, but raises ValueError when substring is not found.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.index.html
 *
 * @param a Input string array
 * @param sub Substring to find
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of indices
 * @throws std::invalid_argument if substring not found
 */
NP_API inline auto index(const Ndarray<std::string> &a, const std::string &sub,
                         int start = 0, int end = -1) -> Ndarray<int> {
  Ndarray<int> result = find(a, sub, start, end);
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (result.data()[i] == -1) {
      throw std::invalid_argument("index: substring not found");
    }
  }
  return result;
}

/**
 * @brief Like rfind, but raises ValueError when substring is not found.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rindex.html
 *
 * @param a Input string array
 * @param sub Substring to find
 * @param start Start position (optional)
 * @param end End position (optional)
 * @return Array of indices
 * @throws std::invalid_argument if substring not found
 */
NP_API inline auto rindex(const Ndarray<std::string> &a, const std::string &sub,
                          int start = 0, int end = -1) -> Ndarray<int> {
  Ndarray<int> result = rfind(a, sub, start, end);
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (result.data()[i] == -1) {
      throw std::invalid_argument("rindex: substring not found");
    }
  }
  return result;
}

/**
 * @brief Split strings around given separator.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.split.html
 *
 * Note: Returns flattened array of all split parts. In Python numpy, this
 * returns object arrays of lists. C++ implementation returns concatenated
 * results.
 *
 * @param a Input string array
 * @param sep Separator (if empty, splits on whitespace)
 * @param maxsplit Maximum number of splits (-1 = unlimited)
 * @return Flattened array of split strings
 */
NP_API inline auto split(const Ndarray<std::string> &a,
                         const std::string &sep = "", int maxsplit = -1)
    -> Ndarray<std::string> {
  std::vector<std::string> all_parts;

  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    std::vector<std::string> parts;

    if (sep.empty()) {
      // Split on whitespace
      std::istringstream iss(s);
      std::string word;
      while (iss >> word) {
        parts.push_back(word);
        if (maxsplit >= 0 && static_cast<int>(parts.size()) > maxsplit)
          break;
      }
    } else {
      // Split on separator
      std::size_t start = 0;
      std::size_t pos;
      int count = 0;
      while ((pos = s.find(sep, start)) != std::string::npos) {
        parts.push_back(s.substr(start, pos - start));
        start = pos + sep.size();
        ++count;
        if (maxsplit >= 0 && count >= maxsplit)
          break;
      }
      parts.push_back(s.substr(start));
    }

    all_parts.insert(all_parts.end(), parts.begin(), parts.end());
  }

  Ndarray<std::string> result =
      empty<std::string>(std::vector<int>{static_cast<int>(all_parts.size())});
  result.data() = all_parts;
  return result;
}

/**
 * @brief Split strings around given separator from the right.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.rsplit.html
 *
 * @param a Input string array
 * @param sep Separator (if empty, splits on whitespace)
 * @param maxsplit Maximum number of splits (-1 = unlimited)
 * @return Flattened array of split strings
 */
NP_API inline auto rsplit(const Ndarray<std::string> &a,
                          const std::string &sep = "", int maxsplit = -1)
    -> Ndarray<std::string> {
  std::vector<std::string> all_parts;

  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    std::vector<std::string> parts;

    if (sep.empty()) {
      // Split on whitespace (same as split for simplicity)
      std::istringstream iss(s);
      std::string word;
      while (iss >> word) {
        parts.push_back(word);
      }
    } else {
      // Split from right
      std::size_t end = s.size();
      int count = 0;
      while (end > 0) {
        std::size_t pos = s.rfind(sep, end - 1);
        if (pos == std::string::npos) {
          parts.insert(parts.begin(), s.substr(0, end));
          break;
        }
        parts.insert(parts.begin(),
                     s.substr(pos + sep.size(), end - pos - sep.size()));
        end = pos;
        ++count;
        if (maxsplit >= 0 && count >= maxsplit) {
          parts.insert(parts.begin(), s.substr(0, end));
          break;
        }
      }
    }

    all_parts.insert(all_parts.end(), parts.begin(), parts.end());
  }

  Ndarray<std::string> result =
      empty<std::string>(std::vector<int>{static_cast<int>(all_parts.size())});
  result.data() = all_parts;
  return result;
}

/**
 * @brief Split strings on line boundaries.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.splitlines.html
 *
 * @param a Input string array
 * @param keepends Keep line endings (default: false)
 * @return Flattened array of lines
 */
NP_API inline auto splitlines(const Ndarray<std::string> &a,
                              bool keepends = false) -> Ndarray<std::string> {
  std::vector<std::string> all_lines;

  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::string &s = a.data()[i];
    std::istringstream iss(s);
    std::string line;
    while (std::getline(iss, line)) {
      if (keepends && !line.empty()) {
        line += '\n';
      }
      all_lines.push_back(line);
    }
  }

  Ndarray<std::string> result =
      empty<std::string>(std::vector<int>{static_cast<int>(all_lines.size())});
  result.data() = all_lines;
  return result;
}

/**
 * @brief Apply str.translate() element-wise.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.translate.html
 *
 * Simplified implementation: table is a string of 256 characters for direct
 * mapping.
 *
 * @param a Input string array
 * @param table Translation table (256-char string)
 * @param deletechars Characters to delete
 * @return Array with translated strings
 */
NP_API inline auto translate(const Ndarray<std::string> &a,
                             const std::string &table,
                             const std::string &deletechars = "")
    -> Ndarray<std::string> {
  Ndarray<std::string> result = empty<std::string>(a.shape);
  for (std::size_t i = 0; i < a.size(); ++i) {
    std::string s = a.data()[i];
    std::string translated;
    for (char c : s) {
      if (deletechars.find(c) != std::string::npos) {
        continue;
      }
      unsigned char uc = static_cast<unsigned char>(c);
      if (uc < table.size()) {
        translated += table[uc];
      } else {
        translated += c;
      }
    }
    result.data()[i] = translated;
  }
  return result;
}

/* Encode/Decode Functions (Simplified - C++ strings are already bytes) */

/**
 * @brief Calls str.encode() element-wise (no-op in C++).
 *
 * Reference: numpy-reference/reference/generated/numpy.char.encode.html
 *
 * Note: In Python NumPy, this converts unicode to bytes. C++ std::string
 * already represents bytes, so this is essentially a copy operation.
 *
 * @param a Input string array
 * @param encoding Encoding name (ignored)
 * @param errors Error handling (ignored)
 * @return Copy of input array
 */
NP_API inline auto encode(const Ndarray<std::string> &a,
                          const std::string &encoding = "utf-8",
                          const std::string &errors = "strict")
    -> Ndarray<std::string> {
  (void)encoding;
  (void)errors; // Unused in C++
  return a;     // C++ strings are already byte strings
}

/**
 * @brief Calls str.decode() element-wise (no-op in C++).
 *
 * Reference: numpy-reference/reference/generated/numpy.char.decode.html
 *
 * Note: In Python NumPy, this converts bytes to unicode. C++ std::string
 * can represent both, so this is essentially a copy operation.
 *
 * @param a Input string array
 * @param encoding Encoding name (ignored)
 * @param errors Error handling (ignored)
 * @return Copy of input array
 */
NP_API inline auto decode(const Ndarray<std::string> &a,
                          const std::string &encoding = "utf-8",
                          const std::string &errors = "strict")
    -> Ndarray<std::string> {
  (void)encoding;
  (void)errors; // Unused in C++
  return a;     // C++ strings are already decoded
}

/* Compare Function */

/**
 * @brief Performs element-wise comparison of two string arrays.
 *
 * Reference:
 * numpy-reference/reference/generated/numpy.char.compare_chararrays.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @param cmp Comparison operator ("==", "!=", "<", "<=", ">", ">=")
 * @param rstrip Strip trailing whitespace before comparison
 * @return Integer array: -1 (less), 0 (equal), 1 (greater)
 */
NP_API inline auto compare_chararrays(const Ndarray<std::string> &x1,
                                      const Ndarray<std::string> &x2,
                                      const std::string &cmp,
                                      bool rstrip = false) -> Ndarray<int> {
  if (x1.shape != x2.shape) {
    throw std::invalid_argument(
        "compare_chararrays: arrays must have the same shape");
  }

  Ndarray<int> result = empty<int>(x1.shape);
  for (std::size_t i = 0; i < x1.size(); ++i) {
    std::string s1 = x1.data()[i];
    std::string s2 = x2.data()[i];

    if (rstrip) {
      s1.erase(s1.find_last_not_of(" \t\n\r") + 1);
      s2.erase(s2.find_last_not_of(" \t\n\r") + 1);
    }

    int cmp_result = s1.compare(s2);
    if (cmp == "==") {
      result.data()[i] = (cmp_result == 0) ? 1 : 0;
    } else if (cmp == "!=") {
      result.data()[i] = (cmp_result != 0) ? 1 : 0;
    } else if (cmp == "<") {
      result.data()[i] = (cmp_result < 0) ? 1 : 0;
    } else if (cmp == "<=") {
      result.data()[i] = (cmp_result <= 0) ? 1 : 0;
    } else if (cmp == ">") {
      result.data()[i] = (cmp_result > 0) ? 1 : 0;
    } else if (cmp == ">=") {
      result.data()[i] = (cmp_result >= 0) ? 1 : 0;
    } else {
      throw std::invalid_argument(
          "compare_chararrays: invalid comparison operator");
    }
  }
  return result;
}

/* Creation Functions */

/**
 * @brief Create a character array (Ndarray<std::string>).
 *
 * Reference: numpy-reference/reference/generated/numpy.char.array.html
 *
 * @param object Initializer list or vector of strings
 * @return String array
 */
NP_API inline auto array(const std::vector<std::string> &object)
    -> Ndarray<std::string> {
  Ndarray<std::string> result =
      empty<std::string>(std::vector<int>{static_cast<int>(object.size())});
  result.data() = object;
  return result;
}

/**
 * @brief Convert input to a character array (Ndarray<std::string>).
 *
 * Reference: numpy-reference/reference/generated/numpy.char.asarray.html
 *
 * @param a Input array
 * @return String array (copy if already string array)
 */
NP_API inline auto asarray(const Ndarray<std::string> &a)
    -> Ndarray<std::string> {
  return a; /* Already a string array */
}

/* chararray class - deprecated but provided for compatibility */

/**
 * @brief Provides a convenient view on arrays of string type with all char
 * functions directly available as methods.
 *
 * DEPRECATED: This class is deprecated in NumPy 2.5 and should not be used
 * in new code. Use Ndarray<std::string> with np::ch functions instead.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.chararray.html
 *
 * This class wraps an Ndarray<std::string> and provides all char module
 * functions as convenient methods. It exists for API compatibility but
 * adds no functionality over using free functions directly.
 *
 * Many methods inherited from ndarray (reshape, transpose, etc.) are
 * accessible through the underlying array() method or implicit conversion.
 */
class NP_DEPRECATED(
    "chararray is deprecated; use Ndarray<std::string> with np::ch functions")
    chararray {
private:
  Ndarray<std::string> data_;

public:
  /* Constructors */
  explicit chararray(const Ndarray<std::string> &arr) : data_(arr) {}
  explicit chararray(const std::vector<int> &shape)
      : data_(empty<std::string>(shape)) {}

  /* Access to underlying array */
  Ndarray<std::string> &array() { return data_; }
  const Ndarray<std::string> &array() const { return data_; }

  /* Implicit conversion to Ndarray<std::string> */
  operator Ndarray<std::string> &() { return data_; }
  operator const Ndarray<std::string> &() const { return data_; }

  /* Ndarray properties - expose from wrapped array */
  const std::vector<int> &shape() const { return data_.shape; }
  std::size_t size() const { return data_.size(); }
  int ndim() const { return static_cast<int>(data_.shape.size()); }

  /* String transformation methods (return chararray for chaining) */
  auto capitalize() const -> chararray {
    return chararray(ch::capitalize(data_));
  }
  auto center(int width, char fillchar = ' ') const -> chararray {
    return chararray(ch::center(data_, width, fillchar));
  }
  auto lower() const -> chararray { return chararray(ch::lower(data_)); }
  auto upper() const -> chararray { return chararray(ch::upper(data_)); }
  auto strip(const std::string &chars = " \t\n\r") const -> chararray {
    return chararray(ch::strip(data_, chars));
  }
  auto lstrip(const std::string &chars = " \t\n\r") const -> chararray {
    return chararray(ch::lstrip(data_, chars));
  }
  auto rstrip(const std::string &chars = " \t\n\r") const -> chararray {
    return chararray(ch::rstrip(data_, chars));
  }
  auto swapcase() const -> chararray { return chararray(ch::swapcase(data_)); }
  auto title() const -> chararray { return chararray(ch::title(data_)); }
  auto zfill(int width) const -> chararray {
    return chararray(ch::zfill(data_, width));
  }
  auto ljust(int width, char fillchar = ' ') const -> chararray {
    return chararray(ch::ljust(data_, width, fillchar));
  }
  auto rjust(int width, char fillchar = ' ') const -> chararray {
    return chararray(ch::rjust(data_, width, fillchar));
  }
  auto replace(const std::string &old, const std::string &new_str,
               int count = -1) const -> chararray {
    return chararray(ch::replace(data_, old, new_str, count));
  }
  auto expandtabs(int tabsize = 8) const -> chararray {
    return chararray(ch::expandtabs(data_, tabsize));
  }
  auto encode(const std::string &encoding = "utf-8") const -> chararray {
    return chararray(ch::encode(data_, encoding));
  }
  auto decode(const std::string &encoding = "utf-8") const -> chararray {
    return chararray(ch::decode(data_, encoding));
  }
  auto translate(const std::string &table) const -> chararray {
    return chararray(ch::translate(data_, table));
  }

  /* Splitting methods (return Ndarray<std::string>) */
  auto split(const std::string &sep = "", int maxsplit = -1) const
      -> Ndarray<std::string> {
    return ch::split(data_, sep, maxsplit);
  }
  auto rsplit(const std::string &sep = "", int maxsplit = -1) const
      -> Ndarray<std::string> {
    return ch::rsplit(data_, sep, maxsplit);
  }
  auto splitlines(bool keepends = false) const -> Ndarray<std::string> {
    return ch::splitlines(data_, keepends);
  }
  auto partition(const std::string &sep) const -> Ndarray<std::string> {
    return ch::partition(data_, sep);
  }
  auto rpartition(const std::string &sep) const -> Ndarray<std::string> {
    return ch::rpartition(data_, sep);
  }

  /* Join method */
  auto join(const Ndarray<std::string> &seq) const -> Ndarray<std::string> {
    return ch::join(data_, seq);
  }

  /* Information methods (return int/bool arrays) */
  auto count(const std::string &sub, int start = 0, int end = -1) const
      -> Ndarray<int> {
    return ch::count(data_, sub, start, end);
  }
  auto find(const std::string &sub, int start = 0, int end = -1) const
      -> Ndarray<int> {
    return ch::find(data_, sub, start, end);
  }
  auto rfind(const std::string &sub, int start = 0, int end = -1) const
      -> Ndarray<int> {
    return ch::rfind(data_, sub, start, end);
  }
  auto index(const std::string &sub, int start = 0, int end = -1) const
      -> Ndarray<int> {
    return ch::index(data_, sub, start, end);
  }
  auto rindex(const std::string &sub, int start = 0, int end = -1) const
      -> Ndarray<int> {
    return ch::rindex(data_, sub, start, end);
  }
  auto str_len() const -> Ndarray<int> { return ch::str_len(data_); }

  /* Boolean test methods */
  auto startswith(const std::string &prefix, int start = 0, int end = -1) const
      -> Ndarray<bool> {
    return ch::startswith(data_, prefix, start, end);
  }
  auto endswith(const std::string &suffix, int start = 0, int end = -1) const
      -> Ndarray<bool> {
    return ch::endswith(data_, suffix, start, end);
  }
  auto isalpha() const -> Ndarray<bool> { return ch::isalpha(data_); }
  auto isalnum() const -> Ndarray<bool> { return ch::isalnum(data_); }
  auto isdigit() const -> Ndarray<bool> { return ch::isdigit(data_); }
  auto isdecimal() const -> Ndarray<bool> { return ch::isdecimal(data_); }
  auto isnumeric() const -> Ndarray<bool> { return ch::isnumeric(data_); }
  auto islower() const -> Ndarray<bool> { return ch::islower(data_); }
  auto isupper() const -> Ndarray<bool> { return ch::isupper(data_); }
  auto isspace() const -> Ndarray<bool> { return ch::isspace(data_); }
  auto istitle() const -> Ndarray<bool> { return ch::istitle(data_); }

  /* Note: ndarray methods like reshape, transpose, copy, etc. are accessible
   * through the underlying array: ca.array().reshape(shape) or through
   * implicit conversion. We don't re-expose all 80+ ndarray methods here
   * as they're not string-specific and chararray is deprecated anyway. */
};

} /* namespace ch */
} /* namespace np */

#endif /* NP_CHAR_HPP */
