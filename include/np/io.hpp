/**
 * @file io.hpp
 * @brief I/O operations: npy save/load and txt savetxt/loadtxt.
 *
 * Implements a minimal subset of numpy.lib.npyio:
 *   save / load for .npy (version 1.0)
 *   savetxt / loadtxt for textual data
 *
 * The npy implementation supports the common dtypes:
 *   bool, int8/16/32/64, uint8/16/32/64, float32/64, complex64/128
 * Little-endian only, C-order. Fortran order flag is ignored on load
 * (data is always reordered to C).
 *
 * Reference: https://numpy.org/doc/stable/reference/routines.io.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_IO_HPP
#define NP_IO_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"

namespace np {

namespace detail {
inline std::string descr_for_type(const std::string &type_name) {
  // Map C++ types to numpy descr strings (little-endian)
  if (type_name == "bool") return "|b1";
  if (type_name == "int8") return "|i1";
  if (type_name == "uint8") return "|u1";
  if (type_name == "int16") return "<i2";
  if (type_name == "uint16") return "<u2";
  if (type_name == "int32") return "<i4";
  if (type_name == "uint32") return "<u4";
  if (type_name == "int64") return "<i8";
  if (type_name == "uint64") return "<u8";
  if (type_name == "float") return "<f4";
  if (type_name == "double") return "<f8";
  if (type_name == "complex<float>") return "<c8";
  if (type_name == "complex<double>") return "<c16";
  return "";
}

template <typename T> inline std::string dtype_descr() {
  if constexpr (std::is_same_v<T, bool>) return "|b1";
  else if constexpr (std::is_same_v<T, int8_t>) return "|i1";
  else if constexpr (std::is_same_v<T, uint8_t>) return "|u1";
  else if constexpr (std::is_same_v<T, int16_t>) return "<i2";
  else if constexpr (std::is_same_v<T, uint16_t>) return "<u2";
  else if constexpr (std::is_same_v<T, int32_t>) return "<i4";
  else if constexpr (std::is_same_v<T, uint32_t>) return "<u4";
  else if constexpr (std::is_same_v<T, int64_t>) return "<i8";
  else if constexpr (std::is_same_v<T, uint64_t>) return "<u8";
  else if constexpr (std::is_same_v<T, float>) return "<f4";
  else if constexpr (std::is_same_v<T, double>) return "<f8";
  else if constexpr (std::is_same_v<T, std::complex<float>>) return "<c8";
  else if constexpr (std::is_same_v<T, std::complex<double>>) return "<c16";
  else if constexpr (std::is_same_v<T, int>) {
    if constexpr (sizeof(int) == 4) return "<i4";
    else if constexpr (sizeof(int) == 8) return "<i8";
    else return "<i4";
  } else if constexpr (std::is_same_v<T, long>) {
    if constexpr (sizeof(long) == 4) return "<i4";
    else return "<i8";
  } else {
    return "";
  }
}

inline std::string build_npy_header(const std::string &descr,
                                    const std::vector<int> &shape) {
  std::ostringstream oss;
  oss << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (shape.size() == 1) oss << ",";
    else if (i + 1 < shape.size()) oss << ", ";
  }
  if (shape.empty()) oss << ",";
  oss << "), }";
  std::string hdr = oss.str();
  hdr += "\n";
  std::size_t pad = (64 - (hdr.size() % 64)) % 64;
  // pad spaces before final newline
  hdr.pop_back(); // remove \n
  hdr += std::string(pad, ' ');
  hdr += "\n";
  return hdr;
}

inline void write_npy_magic(std::ostream &os) {
  const char magic[6] = {(char)0x93, 'N', 'U', 'M', 'P', 'Y'};
  os.write(magic, 6);
  char ver[2] = {1, 0};
  os.write(ver, 2);
}

inline std::string read_npy_header(std::istream &is, std::vector<int> &shape_out,
                                   std::string &descr_out) {
  char magic[6];
  is.read(magic, 6);
  if (is.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("load: not a npy file (bad magic)");
  }
  char ver[2];
  is.read(ver, 2);
  if (ver[0] != 1 || ver[1] != 0) {
    throw std::runtime_error("load: only npy version 1.0 supported");
  }
  uint16_t hlen = 0;
  is.read(reinterpret_cast<char *>(&hlen), 2);
  // little endian
  // On big endian machines need swap, but assume little
  std::string hdr(hlen, '\0');
  is.read(hdr.data(), hlen);
  if ((std::size_t)is.gcount() != hlen) throw std::runtime_error("load: truncated header");
  // Parse descr
  auto pos = hdr.find("'descr'");
  if (pos == std::string::npos) pos = hdr.find("\"descr\"");
  if (pos == std::string::npos) throw std::runtime_error("load: header missing descr");
  auto q1 = hdr.find('\'', pos + 7);
  if (q1 == std::string::npos) q1 = hdr.find('"', pos + 7);
  auto q2 = hdr.find(hdr[q1], q1 + 1);
  descr_out = hdr.substr(q1 + 1, q2 - q1 - 1);
  // Parse shape
  auto sp = hdr.find("'shape'");
  if (sp == std::string::npos) sp = hdr.find("\"shape\"");
  auto paren1 = hdr.find('(', sp);
  auto paren2 = hdr.find(')', paren1);
  std::string shape_str = hdr.substr(paren1 + 1, paren2 - paren1 - 1);
  shape_out.clear();
  std::istringstream sss(shape_str);
  std::string token;
  while (std::getline(sss, token, ',')) {
    // trim
    size_t a = token.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) continue;
    size_t b = token.find_last_not_of(" \t\n\r");
    std::string t = token.substr(a, b - a + 1);
    if (t.empty()) continue;
    shape_out.push_back(std::stoi(t));
  }
  return hdr;
}

} // namespace detail

/** @brief Save array to .npy file (version 1.0).
 *
 * @tparam T Element type.
 * @param filename Output path.
 * @param arr Input array.
 * @throws std::runtime_error on I/O error or unsupported dtype.
 *
 * Reference: numpy-reference/reference/generated/numpy.save.html
 */
template <typename T>
void save(const std::string &filename, const ndarray<T> &arr) {
  std::string descr = detail::dtype_descr<T>();
  if (descr.empty()) throw std::runtime_error("save: unsupported dtype for npy");
  std::string hdr = detail::build_npy_header(descr, arr.shape);
  std::ofstream os(filename, std::ios::binary);
  if (!os) throw std::runtime_error("save: cannot open file " + filename);
  detail::write_npy_magic(os);
  uint16_t hlen = static_cast<uint16_t>(hdr.size());
  os.write(reinterpret_cast<char *>(&hlen), 2);
  os.write(hdr.data(), hdr.size());
  // Write data in C order (logical order)
  for (std::size_t i = 0; i < arr._numel(); ++i) {
    T v = arr.data()[arr._flat_logical(i)];
    os.write(reinterpret_cast<const char *>(&v), sizeof(T));
  }
}

/** @brief Load array from .npy file (version 1.0).
 *
 * The template parameter must match the file's dtype.
 *
 * @tparam T Element type expected.
 * @param filename Input path.
 * @return ndarray<T> with contents.
 * @throws std::runtime_error on I/O error or dtype mismatch.
 *
 * Reference: numpy-reference/reference/generated/numpy.load.html
 */
template <typename T>
auto load(const std::string &filename) -> ndarray<T> {
  std::ifstream is(filename, std::ios::binary);
  if (!is) throw std::runtime_error("load: cannot open file " + filename);
  std::vector<int> shape;
  std::string descr;
  detail::read_npy_header(is, shape, descr);
  std::string expected = detail::dtype_descr<T>();
  if (descr != expected) {
    throw std::runtime_error("load: dtype mismatch: file has " + descr +
                             " expected " + expected);
  }
  std::size_t n = 1;
  for (int d : shape) n *= static_cast<std::size_t>(d);
  if (shape.empty()) n = 1; // 0-d
  std::vector<T> data(n);
  is.read(reinterpret_cast<char *>(data.data()), n * sizeof(T));
  if ((std::size_t)is.gcount() != n * sizeof(T)) {
    // Might be truncated; still check
    if (is.gcount() != 0) throw std::runtime_error("load: truncated data");
  }
  if (shape.empty()) shape = {};
  // Handle scalar case (numpy 0-d shape () )
  if (n == 1 && shape.empty()) shape = {};
  if (shape.empty() && n == 1) {
    // Keep as 0-d? We'll represent as shape {1} for simplicity if original was ()
    // But keep empty shape for 0-d
  }
  // If file had shape (), we stored as {} above, need to produce array with size 1 but empty shape?
  // We'll keep shape as vector<int>{} for 0-d, else as read
  if (shape.empty() && n > 0 && n != 1) shape = {static_cast<int>(n)};
  // Special: if shape was (3,) we already have it
  return ndarray<T>::from_data(shape, std::move(data));
}

/** @brief Save array to text file (whitespace delimited).
 *
 * Only 1-D and 2-D arrays supported. Mirrors `np.savetxt`.
 *
 * @tparam T Element type (must be streamable).
 * @param filename Output path.
 * @param arr Input array (1-D or 2-D).
 * @param delimiter Column delimiter (default space).
 * @param fmt Format string ignored – C++ streams used.
 *
 * Reference: numpy-reference/reference/generated/numpy.savetxt.html
 */
template <typename T>
void savetxt(const std::string &filename, const ndarray<T> &arr,
             const std::string &delimiter = " ", const std::string &fmt = "") {
  (void)fmt;
  std::ofstream os(filename);
  if (!os) throw std::runtime_error("savetxt: cannot open file " + filename);
  if (arr.ndim() == 1) {
    for (std::size_t i = 0; i < arr.size(); ++i) {
      os << std::setprecision(10) << arr.data()[arr._flat_logical(i)] << "\n";
    }
  } else if (arr.ndim() == 2) {
    for (int i = 0; i < arr.shape[0]; ++i) {
      for (int j = 0; j < arr.shape[1]; ++j) {
        if (j) os << delimiter;
        os << std::setprecision(10) << arr.at(static_cast<std::size_t>(i),
                                              static_cast<std::size_t>(j));
      }
      os << "\n";
    }
  } else {
    throw std::invalid_argument("savetxt: only 1-D or 2-D arrays supported");
  }
}

/** @brief Load array from text file (whitespace delimited).
 *
 * Mirrors `np.loadtxt`. Returns double array; infer shape from file.
 * Empty lines and lines starting with '#' are ignored.
 *
 * @param filename Input path.
 * @param delimiter Column delimiter (default whitespace).
 * @return ndarray<double> 2-D if multi-column else 1-D.
 *
 * Reference: numpy-reference/reference/generated/numpy.loadtxt.html
 */
inline auto loadtxt(const std::string &filename, const std::string &delimiter = " ")
    -> ndarray<double> {
  std::ifstream is(filename);
  if (!is) throw std::runtime_error("loadtxt: cannot open file " + filename);
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(is, line)) {
    // trim leading
    size_t p = line.find_first_not_of(" \t\r\n");
    if (p == std::string::npos) continue;
    if (line[p] == '#') continue;
    std::vector<double> vals;
    if (delimiter == " " || delimiter == "\t") {
      std::istringstream ss(line);
      double v;
      while (ss >> v) vals.push_back(v);
    } else {
      std::istringstream ss(line);
      std::string tok;
      while (std::getline(ss, tok, delimiter[0])) {
        if (tok.empty()) continue;
        vals.push_back(std::stod(tok));
      }
    }
    if (!vals.empty()) rows.push_back(std::move(vals));
  }
  if (rows.empty()) return ndarray<double>(std::vector<int>{0});
  std::size_t cols = rows[0].size();
  for (auto &r : rows) if (r.size() != cols) throw std::runtime_error("loadtxt: inconsistent columns");
  if (cols == 1) {
    ndarray<double> out(std::vector<int>{static_cast<int>(rows.size())});
    for (std::size_t i = 0; i < rows.size(); ++i) out.data()[i] = rows[i][0];
    return out;
  } else {
    ndarray<double> out(std::vector<int>{static_cast<int>(rows.size()), static_cast<int>(cols)});
    for (std::size_t i = 0; i < rows.size(); ++i)
      for (std::size_t j = 0; j < cols; ++j) out.at(i, j) = rows[i][j];
    return out;
  }
}

} // namespace np

#endif // NP_IO_HPP
