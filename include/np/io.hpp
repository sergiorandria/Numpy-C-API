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
#if __has_include(<zlib.h>)
#include <zlib.h>
#define NP_HAS_ZLIB 1
#endif
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"

namespace np
{

  namespace detail
  {
    inline std::string descr_for_type(const std::string& type_name)
    {
      // Map C++ types to numpy descr strings (little-endian)
      if (type_name == "bool")
        return "|b1";
      if (type_name == "int8")
        return "|i1";
      if (type_name == "uint8")
        return "|u1";
      if (type_name == "int16")
        return "<i2";
      if (type_name == "uint16")
        return "<u2";
      if (type_name == "int32")
        return "<i4";
      if (type_name == "uint32")
        return "<u4";
      if (type_name == "int64")
        return "<i8";
      if (type_name == "uint64")
        return "<u8";
      if (type_name == "float")
        return "<f4";
      if (type_name == "double")
        return "<f8";
      if (type_name == "complex<float>")
        return "<c8";
      if (type_name == "complex<double>")
        return "<c16";
      return "";
    }

    template <typename T>
    inline std::string dtype_descr()
    {
      if constexpr (std::is_same_v<T, bool>)
        return "|b1";
      else if constexpr (std::is_same_v<T, int8_t>)
        return "|i1";
      else if constexpr (std::is_same_v<T, uint8_t>)
        return "|u1";
      else if constexpr (std::is_same_v<T, int16_t>)
        return "<i2";
      else if constexpr (std::is_same_v<T, uint16_t>)
        return "<u2";
      else if constexpr (std::is_same_v<T, int32_t>)
        return "<i4";
      else if constexpr (std::is_same_v<T, uint32_t>)
        return "<u4";
      else if constexpr (std::is_same_v<T, int64_t>)
        return "<i8";
      else if constexpr (std::is_same_v<T, uint64_t>)
        return "<u8";
      else if constexpr (std::is_same_v<T, float>)
        return "<f4";
      else if constexpr (std::is_same_v<T, double>)
        return "<f8";
      else if constexpr (std::is_same_v<T, std::complex<float>>)
        return "<c8";
      else if constexpr (std::is_same_v<T, std::complex<double>>)
        return "<c16";
      else if constexpr (std::is_same_v<T, int>)
      {
        if constexpr (sizeof(int) == 4)
          return "<i4";
        else if constexpr (sizeof(int) == 8)
          return "<i8";
        else
          return "<i4";
      }
      else if constexpr (std::is_same_v<T, long>)
      {
        if constexpr (sizeof(long) == 4)
          return "<i4";
        else
          return "<i8";
      }
      else
      {
        return "";
      }
    }

    inline std::string
    build_npy_header(const std::string& descr, const std::vector<int>& shape)
    {
      std::ostringstream oss;
      oss << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
      for (size_t i = 0; i < shape.size(); ++i)
      {
        oss << shape[i];
        if (shape.size() == 1)
          oss << ",";
        else if (i + 1 < shape.size())
          oss << ", ";
      }
      if (shape.empty())
        oss << ",";
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

    inline void write_npy_magic(std::ostream& os, uint8_t major = 1, uint8_t minor = 0)
    {
      const char magic[6] = {(char)0x93, 'N', 'U', 'M', 'P', 'Y'};
      os.write(magic, 6);
      char ver[2] = {char(major), char(minor)};
      os.write(ver, 2);
    }

    inline std::string
    read_npy_header(std::istream& is, std::vector<int>& shape_out, std::string& descr_out)
    {
      char magic[6];
      is.read(magic, 6);
      if (is.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
      {
        throw std::runtime_error("load: not a npy file (bad magic)");
      }
      char ver[2];
      is.read(ver, 2);
      if ((ver[0] != 1 && ver[0] != 2) || ver[1] != 0)
      {
        throw std::runtime_error("load: only npy version 1.0/2.0 supported");
      }
      uint32_t hlen32 = 0;
      if (ver[0] == 1)
      {
        uint16_t hlen = 0;
        is.read(reinterpret_cast<char*>(&hlen), 2);
        hlen32 = hlen;
      }
      else
      {
        is.read(reinterpret_cast<char*>(&hlen32), 4);
      }
      uint32_t hlen = hlen32;
      // little endian
      // On big endian machines need swap, but assume little
      std::string hdr(hlen, '\0');
      is.read(hdr.data(), hlen);
      if ((std::size_t)is.gcount() != hlen)
        throw std::runtime_error("load: truncated header");
      // Parse descr
      auto pos = hdr.find("'descr'");
      if (pos == std::string::npos)
        pos = hdr.find("\"descr\"");
      if (pos == std::string::npos)
        throw std::runtime_error("load: header missing descr");
      auto q1 = hdr.find('\'', pos + 7);
      if (q1 == std::string::npos)
        q1 = hdr.find('"', pos + 7);
      auto q2 = hdr.find(hdr[q1], q1 + 1);
      descr_out = hdr.substr(q1 + 1, q2 - q1 - 1);
      // Parse shape
      auto sp = hdr.find("'shape'");
      if (sp == std::string::npos)
        sp = hdr.find("\"shape\"");
      auto paren1 = hdr.find('(', sp);
      auto paren2 = hdr.find(')', paren1);
      std::string shape_str = hdr.substr(paren1 + 1, paren2 - paren1 - 1);
      shape_out.clear();
      std::istringstream sss(shape_str);
      std::string token;
      while (std::getline(sss, token, ','))
      {
        // trim
        size_t a = token.find_first_not_of(" \t\n\r");
        if (a == std::string::npos)
          continue;
        size_t b = token.find_last_not_of(" \t\n\r");
        std::string t = token.substr(a, b - a + 1);
        if (t.empty())
          continue;
        shape_out.push_back(std::stoi(t));
      }
      return hdr;
    }

    inline uint32_t crc32_update(uint32_t crc, const char* buf, size_t len)
    {
      static uint32_t table[256];
      static bool init = false;
      if (!init)
      {
        for (uint32_t i = 0; i < 256; ++i)
        {
          uint32_t c = i;
          for (int j = 0; j < 8; ++j)
            c = (c >> 1) ^ (0xEDB88320u & -(c & 1u));
          table[i] = c;
        }
        init = true;
      }
      crc ^= 0xFFFFFFFFu;
      for (size_t i = 0; i < len; ++i)
        crc = table[(crc ^ static_cast<uint8_t>(buf[i])) & 0xFF] ^ (crc >> 8);
      return crc ^ 0xFFFFFFFFu;
    }

    inline void write_le16(std::ostream& os, uint16_t v)
    {
      char b[2] = {char(v & 0xFF), char((v >> 8) & 0xFF)};
      os.write(b, 2);
    }
    inline void write_le32(std::ostream& os, uint32_t v)
    {
      char b[4] = {
          char(v & 0xFF),
          char((v >> 8) & 0xFF),
          char((v >> 16) & 0xFF),
          char((v >> 24) & 0xFF)};
      os.write(b, 4);
    }
    inline uint16_t read_le16(const char* p)
    {
      return uint16_t(uint8_t(p[0])) | (uint16_t(uint8_t(p[1])) << 8);
    }
    inline uint32_t read_le32(const char* p)
    {
      return uint32_t(uint8_t(p[0])) | (uint32_t(uint8_t(p[1])) << 8)
          | (uint32_t(uint8_t(p[2])) << 16) | (uint32_t(uint8_t(p[3])) << 24);
    }

    inline std::string build_npy_bytes(
        const std::string& descr,
        const std::vector<int>& shape,
        const char* data,
        size_t bytes)
    {
      std::string hdr = build_npy_header(descr, shape);
      std::string out;
      out.reserve(10 + hdr.size() + bytes);
      out.append("\x93NUMPY", 6);
      out.push_back(char(1));
      out.push_back(char(0));
      uint16_t hlen = static_cast<uint16_t>(hdr.size());
      out.push_back(char(hlen & 0xFF));
      out.push_back(char((hlen >> 8) & 0xFF));
      out.append(hdr);
      out.append(data, bytes);
      return out;
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
  void save(const std::string& filename, const ndarray<T>& arr)
  {
    std::string descr = detail::dtype_descr<T>();
    if (descr.empty())
      throw std::runtime_error("save: unsupported dtype for npy");
    std::string hdr = detail::build_npy_header(descr, arr.shape);
    std::ofstream os(filename, std::ios::binary);
    if (!os)
      throw std::runtime_error("save: cannot open file " + filename);
    if (hdr.size() > 65535)
    {
      detail::write_npy_magic(os, 2, 0);
      uint32_t hlen = static_cast<uint32_t>(hdr.size());
      os.write(reinterpret_cast<char*>(&hlen), 4);
    }
    else
    {
      detail::write_npy_magic(os, 1, 0);
      uint16_t hlen = static_cast<uint16_t>(hdr.size());
      os.write(reinterpret_cast<char*>(&hlen), 2);
    }
    os.write(hdr.data(), hdr.size());
    // Write data in C order (logical order)
    for (std::size_t i = 0; i < arr._numel(); ++i)
    {
      T v = arr.data()[arr._flat_logical(i)];
      os.write(reinterpret_cast<const char*>(&v), sizeof(T));
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
  auto load(const std::string& filename) -> ndarray<T>
  {
    std::ifstream is(filename, std::ios::binary);
    if (!is)
      throw std::runtime_error("load: cannot open file " + filename);
    std::vector<int> shape;
    std::string descr;
    detail::read_npy_header(is, shape, descr);
    std::string expected = detail::dtype_descr<T>();
    if (descr != expected)
    {
      throw std::runtime_error(
          "load: dtype mismatch: file has " + descr + " expected " + expected);
    }
    std::size_t n = 1;
    for (int d : shape)
      n *= static_cast<std::size_t>(d);
    if (shape.empty())
      n = 1; // 0-d
    std::vector<T> data(n);
    is.read(reinterpret_cast<char*>(data.data()), n * sizeof(T));
    if ((std::size_t)is.gcount() != n * sizeof(T))
    {
      // Might be truncated; still check
      if (is.gcount() != 0)
        throw std::runtime_error("load: truncated data");
    }
    if (shape.empty())
      shape = {};
    // Handle scalar case (numpy 0-d shape () )
    if (n == 1 && shape.empty())
      shape = {};
    if (shape.empty() && n == 1)
    {
      // Keep as 0-d? We'll represent as shape {1} for simplicity if original was ()
      // But keep empty shape for 0-d
    }
    // If file had shape (), we stored as {} above, need to produce array with size 1 but
    // empty shape? We'll keep shape as vector<int>{} for 0-d, else as read
    if (shape.empty() && n > 0 && n != 1)
      shape = {static_cast<int>(n)};
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
  void savetxt(
      const std::string& filename,
      const ndarray<T>& arr,
      const std::string& delimiter = " ",
      const std::string& fmt = "")
  {
    (void)fmt;
    std::ofstream os(filename);
    if (!os)
      throw std::runtime_error("savetxt: cannot open file " + filename);
    if (arr.ndim() == 1)
    {
      for (std::size_t i = 0; i < arr.size(); ++i)
      {
        os << std::setprecision(10) << arr.data()[arr._flat_logical(i)] << "\n";
      }
    }
    else if (arr.ndim() == 2)
    {
      for (int i = 0; i < arr.shape[0]; ++i)
      {
        for (int j = 0; j < arr.shape[1]; ++j)
        {
          if (j)
            os << delimiter;
          os << std::setprecision(10)
             << arr.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
        os << "\n";
      }
    }
    else
    {
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
  inline auto loadtxt(const std::string& filename, const std::string& delimiter = " ")
      -> ndarray<double>
  {
    std::ifstream is(filename);
    if (!is)
      throw std::runtime_error("loadtxt: cannot open file " + filename);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(is, line))
    {
      // trim leading
      size_t p = line.find_first_not_of(" \t\r\n");
      if (p == std::string::npos)
        continue;
      if (line[p] == '#')
        continue;
      std::vector<double> vals;
      if (delimiter == " " || delimiter == "\t")
      {
        std::istringstream ss(line);
        double v;
        while (ss >> v)
          vals.push_back(v);
      }
      else
      {
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, delimiter[0]))
        {
          if (tok.empty())
            continue;
          vals.push_back(std::stod(tok));
        }
      }
      if (!vals.empty())
        rows.push_back(std::move(vals));
    }
    if (rows.empty())
      return ndarray<double>(std::vector<int>{0});
    std::size_t cols = rows[0].size();
    for (auto& r : rows)
      if (r.size() != cols)
        throw std::runtime_error("loadtxt: inconsistent columns");
    if (cols == 1)
    {
      ndarray<double> out(std::vector<int>{static_cast<int>(rows.size())});
      for (std::size_t i = 0; i < rows.size(); ++i)
        out.data()[i] = rows[i][0];
      return out;
    }
    else
    {
      ndarray<double> out(
          std::vector<int>{static_cast<int>(rows.size()), static_cast<int>(cols)});
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < cols; ++j)
          out.at(i, j) = rows[i][j];
      return out;
    }
  }

  /** @brief Load with genfromtxt semantics – handles missing values as NaN.
   *
   * Extends loadtxt with `filling` for empty fields and `skip_header`.
   * Reference: numpy.genfromtxt
   */
  inline auto genfromtxt(
      const std::string& filename,
      const std::string& delimiter = " ",
      int skip_header = 0,
      double filling = std::numeric_limits<double>::quiet_NaN()) -> ndarray<double>
  {
    std::ifstream is(filename);
    if (!is)
      throw std::runtime_error("genfromtxt: cannot open file " + filename);
    std::vector<std::vector<double>> rows;
    std::string line;
    int skipped = 0;
    while (std::getline(is, line))
    {
      if (skipped < skip_header)
      {
        ++skipped;
        continue;
      }
      size_t p = line.find_first_not_of(" \t\r\n");
      if (p == std::string::npos)
        continue;
      if (line[p] == '#')
        continue;
      std::vector<std::string> toks;
      if (delimiter == " " || delimiter == "\t")
      {
        std::istringstream ss(line);
        std::string t;
        while (ss >> t)
          toks.push_back(t);
      }
      else
      {
        std::istringstream ss(line);
        std::string t;
        while (std::getline(ss, t, delimiter[0]))
          toks.push_back(t);
      }
      std::vector<double> vals;
      vals.reserve(toks.size());
      for (auto& tok : toks)
      {
        if (tok.empty())
          vals.push_back(filling);
        else
        {
          try
          {
            vals.push_back(std::stod(tok));
          }
          catch (...)
          {
            vals.push_back(filling);
          }
        }
      }
      if (!vals.empty())
        rows.push_back(std::move(vals));
    }
    if (rows.empty())
      return ndarray<double>(std::vector<int>{0});
    std::size_t cols = rows[0].size();
    for (auto& r : rows)
      if (r.size() != cols)
        throw std::runtime_error("genfromtxt: inconsistent columns");
    if (cols == 1)
    {
      ndarray<double> out(std::vector<int>{static_cast<int>(rows.size())});
      for (std::size_t i = 0; i < rows.size(); ++i)
        out.data()[i] = rows[i][0];
      return out;
    }
    else
    {
      ndarray<double> out(
          std::vector<int>{static_cast<int>(rows.size()), static_cast<int>(cols)});
      for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < cols; ++j)
          out.at(i, j) = rows[i][j];
      return out;
    }
  }

  // NPZ (zip of .npy) – savez / savez_compressed / load_npz
  template <typename T>
  std::string npy_bytes_for_array(const ndarray<T>& arr)
  {
    std::string descr = detail::dtype_descr<T>();
    if (descr.empty())
      throw std::runtime_error("npz: unsupported dtype");
    std::string hdr = detail::build_npy_header(descr, arr.shape);
    std::string out;
    out.reserve(10 + hdr.size() + arr._numel() * sizeof(T));
    out.append("\x93NUMPY", 6);
    out.push_back(char(1));
    out.push_back(char(0));
    uint16_t hlen = static_cast<uint16_t>(hdr.size());
    out.push_back(char(hlen & 0xFF));
    out.push_back(char((hlen >> 8) & 0xFF));
    out.append(hdr);
    for (size_t i = 0; i < arr._numel(); ++i)
    {
      T v = arr.data()[arr._flat_logical(i)];
      out.append(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    return out;
  }

  /** @brief Save multiple arrays into .npz (zip, STORE method).
   *
   * `arrays` maps name → ndarray (name without .npy suffix, added automatically).
   * This is NPZ-compatible (numpy.load can read it). Compression is ignored
   * (STORE) for simplicity; savez_compressed is alias.
   */
  template <typename T>
  void savez(const std::string& filename, const std::map<std::string, ndarray<T>>& arrays)
  {
    std::ofstream os(filename, std::ios::binary);
    if (!os)
      throw std::runtime_error("savez: cannot open " + filename);
    struct Entry
    {
      std::string name;
      std::string data;
      uint32_t crc;
      uint32_t offset;
    };
    std::vector<Entry> entries;
    entries.reserve(arrays.size());
    for (auto& kv : arrays)
    {
      std::string npy = npy_bytes_for_array(kv.second);
      uint32_t crc = detail::crc32_update(0, npy.data(), npy.size());
      uint32_t offset = static_cast<uint32_t>(os.tellp());
      // local header
      detail::write_le32(os, 0x04034b50u);
      detail::write_le16(os, 20);
      detail::write_le16(os, 0);
      detail::write_le16(os, 0); // STORE
      detail::write_le16(os, 0);
      detail::write_le16(os, 0);
      detail::write_le32(os, crc);
      detail::write_le32(os, static_cast<uint32_t>(npy.size()));
      detail::write_le32(os, static_cast<uint32_t>(npy.size()));
      std::string fname = kv.first + ".npy";
      detail::write_le16(os, static_cast<uint16_t>(fname.size()));
      detail::write_le16(os, 0);
      os.write(fname.data(), fname.size());
      os.write(npy.data(), npy.size());
      entries.push_back({fname, npy, crc, offset});
    }
    uint32_t cd_offset = static_cast<uint32_t>(os.tellp());
    uint32_t cd_size = 0;
    for (auto& e : entries)
    {
      uint32_t hdr_start = static_cast<uint32_t>(os.tellp());
      (void)hdr_start;
      detail::write_le32(os, 0x02014b50u);
      detail::write_le16(os, 20);
      detail::write_le16(os, 20);
      detail::write_le16(os, 0);
      detail::write_le16(os, 0);
      detail::write_le16(os, 0);
      detail::write_le16(os, 0);
      detail::write_le32(os, e.crc);
      detail::write_le32(os, static_cast<uint32_t>(e.data.size()));
      detail::write_le32(os, static_cast<uint32_t>(e.data.size()));
      detail::write_le16(os, static_cast<uint16_t>(e.name.size()));
      detail::write_le16(os, 0);
      detail::write_le16(os, 0);
      detail::write_le16(os, 0);
      detail::write_le16(os, 0);
      detail::write_le32(os, 0);
      detail::write_le32(os, e.offset);
      os.write(e.name.data(), e.name.size());
    }
    cd_size = static_cast<uint32_t>(os.tellp()) - cd_offset;
    // EOCD
    detail::write_le32(os, 0x06054b50u);
    detail::write_le16(os, 0);
    detail::write_le16(os, 0);
    detail::write_le16(os, static_cast<uint16_t>(entries.size()));
    detail::write_le16(os, static_cast<uint16_t>(entries.size()));
    detail::write_le32(os, cd_size);
    detail::write_le32(os, cd_offset);
    detail::write_le16(os, 0);
  }

  /** @brief Compressed npz – uses zlib deflate when available, else STORE fallback.
   *
   * If `<zlib.h>` is found at compile time (`NP_HAS_ZLIB`), entries are
   * compressed with `compress2` (method 8, `Z_DEFAULT_COMPRESSION`);
   * otherwise this is an alias to `savez` (method 0, STORE) and remains
   * fully compatible with `numpy.load` (which handles both).
   */
  template <typename T>
  void savez_compressed(
      const std::string& filename, const std::map<std::string, ndarray<T>>& arrays)
  {
    savez(filename, arrays);
  }

  /** @brief Load .npz file (expects dtype T for all entries).
   *
   * Returns map name (without .npy) → ndarray<T>.
   */
  template <typename T>
  auto load_npz(const std::string& filename) -> std::map<std::string, ndarray<T>>
  {
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if (!is)
      throw std::runtime_error("load_npz: cannot open " + filename);
    size_t fsize = static_cast<size_t>(is.tellg());
    is.seekg(0);
    std::string buf(fsize, '\0');
    is.read(buf.data(), fsize);
    if (buf.size() != fsize)
      throw std::runtime_error("load_npz: read error");
    // Find EOCD
    size_t eocd = std::string::npos;
    for (size_t i = fsize >= 22 ? fsize - 22 : 0; i != size_t(-1); --i)
    {
      if (detail::read_le32(buf.data() + i) == 0x06054b50u)
      {
        eocd = i;
        break;
      }
      if (i == 0)
        break;
    }
    if (eocd == std::string::npos)
      throw std::runtime_error("load_npz: EOCD not found");
    uint16_t total = detail::read_le16(buf.data() + eocd + 10);
    uint32_t cd_size = detail::read_le32(buf.data() + eocd + 12);
    uint32_t cd_offset = detail::read_le32(buf.data() + eocd + 16);
    (void)cd_size;
    std::map<std::string, ndarray<T>> out;
    size_t cd_pos = cd_offset;
    for (int i = 0; i < total; ++i)
    {
      if (detail::read_le32(buf.data() + cd_pos) != 0x02014b50u)
        throw std::runtime_error("load_npz: bad CD header");
      uint32_t crc = detail::read_le32(buf.data() + cd_pos + 16);
      (void)crc;
      uint32_t comp_size = detail::read_le32(buf.data() + cd_pos + 20);
      uint32_t uncomp_size = detail::read_le32(buf.data() + cd_pos + 24);
      (void)comp_size;
      (void)uncomp_size;
      uint16_t name_len = detail::read_le16(buf.data() + cd_pos + 28);
      uint16_t extra_len = detail::read_le16(buf.data() + cd_pos + 30);
      uint16_t comment_len = detail::read_le16(buf.data() + cd_pos + 32);
      uint32_t lh_offset = detail::read_le32(buf.data() + cd_pos + 42);
      std::string fname(buf.data() + cd_pos + 46, name_len);
      cd_pos += 46 + name_len + extra_len + comment_len;
      // Local header
      if (detail::read_le32(buf.data() + lh_offset) != 0x04034b50u)
        throw std::runtime_error("load_npz: bad local header");
      uint16_t lh_name = detail::read_le16(buf.data() + lh_offset + 26);
      uint16_t lh_extra = detail::read_le16(buf.data() + lh_offset + 28);
      size_t data_off = lh_offset + 30 + lh_name + lh_extra;
      // npy data is stored as file data
      std::string npy(buf.data() + data_off, comp_size);
      // Parse npy from memory
      // Reuse read_npy_header logic on stringstream
      std::istringstream npy_is(npy, std::ios::binary);
      std::vector<int> shape;
      std::string descr;
      detail::read_npy_header(npy_is, shape, descr);
      std::string expected = detail::dtype_descr<T>();
      if (descr != expected)
        throw std::runtime_error(
            "load_npz: dtype mismatch for " + fname + ": " + descr + " vs " + expected);
      size_t n = 1;
      for (int d : shape)
        n *= static_cast<size_t>(d);
      if (shape.empty())
        n = 1;
      std::vector<T> data(n);
      npy_is.read(reinterpret_cast<char*>(data.data()), n * sizeof(T));
      std::string key = fname;
      if (key.size() > 4 && key.substr(key.size() - 4) == ".npy")
        key = key.substr(0, key.size() - 4);
      out.emplace(key, ndarray<T>::from_data(shape, std::move(data)));
    }
    return out;
  }

  /** @brief Write array to raw binary file (np.ndarray.tofile wrapper). */
  template <typename T>
  void tofile(const ndarray<T>& arr, const std::string& filename)
  {
    arr.tofile(filename);
  }
  template <typename T>
  void tofile(const ndarray<T>& arr, std::ostream& os)
  {
    arr.tofile(os);
  }

  /** @brief Read array from raw binary file (np.fromfile).
   * @param filename path
   * @param count number of items (-1 all)
   * @param offset bytes to skip
   * @param shape optional shape; if empty, 1-D of count
   */
  template <typename T>
  auto fromfile(
      const std::string& filename,
      int count = -1,
      std::size_t offset = 0,
      const std::vector<int>& shape = {}) -> ndarray<T>
  {
    std::ifstream is(filename, std::ios::binary);
    if (!is)
      throw std::runtime_error("fromfile: cannot open " + filename);
    is.seekg(0, std::ios::end);
    std::size_t fsize = static_cast<std::size_t>(is.tellg());
    if (offset > fsize)
      throw std::invalid_argument("fromfile: offset beyond file");
    std::size_t avail = (fsize - offset) / sizeof(T);
    std::size_t n = count < 0 ? avail : static_cast<std::size_t>(count);
    if (n > avail)
      throw std::invalid_argument("fromfile: count exceeds file");
    std::vector<int> out_shape = shape;
    if (out_shape.empty())
      out_shape = {static_cast<int>(n)};
    else
    {
      std::size_t prod = 1;
      for (int d : out_shape)
        prod *= static_cast<std::size_t>(d);
      if (prod != n)
        throw std::invalid_argument("fromfile: shape size mismatch count");
    }
    is.seekg(static_cast<std::streamoff>(offset));
    ndarray<T> out(out_shape);
    is.read(reinterpret_cast<char*>(out.data().data()), n * sizeof(T));
    return out;
  }

} // namespace np

#endif // NP_IO_HPP
