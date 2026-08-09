/**
 * @file fft.hpp
 * @brief Fast Fourier Transform (np::fft::fft, np::fft::ifft).
 *
 * Radix-2 Cooley-Tukey with Bluestein's algorithm for arbitrary sizes,
 * applied along the last axis. Inputs are promoted to std::complex<double>.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_FFT_HPP
#define NP_FFT_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

namespace np::fft {

/** @brief Complex type used by the FFT routines. */
using Cplx = std::complex<double>;

namespace detail {

/** @brief Smallest power of two >= n (and >= 2). */
inline std::size_t next_pow2(std::size_t n) {
  std::size_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

/** @brief Twiddle factor w^k = exp(+-2*pi*i*k/n). */
inline Cplx twiddle(int n, int k, bool inverse) {
  const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> *
                       static_cast<double>(k) / static_cast<double>(n);
  return {std::cos(angle), std::sin(angle)};
}

/**
 * @brief In-place iterative radix-2 FFT (n must be a power of two).
 */
inline void radix2(std::vector<Cplx> &a, bool inverse) {
  const std::size_t n = a.size();
  if (n <= 1) {
    return;
  }
  // Bit-reversal permutation
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(a[i], a[j]);
    }
  }
  // Butterfly stages
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const std::size_t half = len >> 1;
    for (std::size_t i = 0; i < n; i += len) {
      for (std::size_t k = 0; k < half; ++k) {
        const Cplx w =
            twiddle(static_cast<int>(len), static_cast<int>(k), inverse);
        const Cplx u = a[i + k];
        const Cplx v = a[i + k + half] * w;
        a[i + k] = u + v;
        a[i + k + half] = u - v;
      }
    }
  }
  if (inverse) {
    for (auto &x : a) {
      x /= static_cast<double>(n);
    }
  }
}

/**
 * @brief In-place Bluestein FFT for arbitrary n.
 *
 * Computes X[k] = sum_j x[j] * exp(+-2*pi*i*j*k/n) (sign selects
 * forward/inverse); inverse also divides by n.
 */
inline void bluestein(std::vector<Cplx> &x, bool inverse) {
  const std::size_t n = x.size();
  if (n <= 1) {
    return;
  }
  const double s = inverse ? 1.0 : -1.0;
  const double inv_n = 1.0 / static_cast<double>(n);

  auto chirp = [s, inv_n](long m) {
    const double angle =
        s * std::numbers::pi_v<double> * static_cast<double>(m * m) * inv_n;
    return Cplx{std::cos(angle), std::sin(angle)};
  };

  // b[j] = x[j] * a[j],  a[j] = exp(s*pi*i*j^2/n)
  std::vector<Cplx> b(n);
  for (std::size_t j = 0; j < n; ++j) {
    b[j] = x[j] * chirp(static_cast<long>(j));
  }

  // c'[j] = exp(-s*pi*i*(j-(n-1))^2/n)  for j = 0..2n-2
  std::vector<Cplx> c(2 * n - 1);
  for (std::size_t j = 0; j < 2 * n - 1; ++j) {
    const long m = static_cast<long>(j) - static_cast<long>(n - 1);
    const double angle =
        -s * std::numbers::pi_v<double> * static_cast<double>(m * m) * inv_n;
    c[j] = {std::cos(angle), std::sin(angle)};
  }

  const std::size_t N = next_pow2(3 * n - 2);
  std::vector<Cplx> fb(N, Cplx{0.0, 0.0});
  std::vector<Cplx> fc(N, Cplx{0.0, 0.0});
  std::copy(b.begin(), b.end(), fb.begin());
  std::copy(c.begin(), c.end(), fc.begin());

  radix2(fb, false);
  radix2(fc, false);
  for (std::size_t i = 0; i < N; ++i) {
    fb[i] *= fc[i];
  }
  radix2(fb, true);

  for (std::size_t k = 0; k < n; ++k) {
    x[k] = chirp(static_cast<long>(k)) * fb[k + n - 1];
    if (inverse) {
      x[k] *= inv_n;
    }
  }
}

/**
 * @brief FFT of a single 1D sequence (in-place).
 */
inline void transform(std::vector<Cplx> &a, bool inverse) {
  if (a.empty()) {
    return;
  }
  const std::size_t n = a.size();
  if ((n & (n - 1)) == 0) {
    radix2(a, inverse);
  } else {
    bluestein(a, inverse);
  }
}

} // namespace detail

/**
 * @brief 1D discrete Fourier transform of a complex sequence.
 */
NP_API NP_NODISCARD inline auto fft(const std::vector<Cplx> &x) -> std::vector<Cplx> {
  std::vector<Cplx> out(x);
  detail::transform(out, false);
  return out;
}

/**
 * @brief 1D inverse discrete Fourier transform of a complex sequence.
 */
NP_API NP_NODISCARD inline auto ifft(const std::vector<Cplx> &x) -> std::vector<Cplx> {
  std::vector<Cplx> out(x);
  detail::transform(out, true);
  return out;
}

/**
 * @brief Promotes any numeric Ndarray to complex and FFTs along the last
 *        axis.
 */
NP_API template <typename T> NP_NODISCARD auto fft(const Ndarray<T> &x) -> Ndarray<Cplx> {
  const std::size_t nd = x.ndim();
  if (nd == 0) {
    throw std::invalid_argument("fft requires ndim >= 1");
  }
  const std::size_t n = static_cast<std::size_t>(x.shape[nd - 1]);

  Ndarray<Cplx> out = Ndarray<Cplx>::from_data(
      x.shape, std::vector<Cplx>(x._numel(), Cplx{0.0, 0.0}));
  for (std::size_t i = 0; i < x._numel(); ++i) {
    out.data()[i] = static_cast<Cplx>(x.data()[x._flat_logical(i)]);
  }

  if (nd == 1) {
    detail::transform(out.data(), false);
    return out;
  }

  // Apply along the last axis: iterate over all outer indices.
  std::vector<int> outer_shape(x.shape.begin(), x.shape.end() - 1);
  np::detail::Odometer od(outer_shape);
  while (!od.done()) {
    const auto &idx = od.idx();
    std::vector<Cplx> slice(n);
    for (std::size_t p = 0; p < n; ++p) {
      std::vector<std::size_t> full = idx;
      full.push_back(p);
      slice[p] = out.data()[np::detail::flat_index(full, out.strides, 0)];
    }
    detail::transform(slice, false);
    for (std::size_t p = 0; p < n; ++p) {
      std::vector<std::size_t> full = idx;
      full.push_back(p);
      out.data()[np::detail::flat_index(full, out.strides, 0)] = slice[p];
    }
    od.advance();
  }
  return out;
}

/**
 * @brief Inverse FFT along the last axis of a numeric Ndarray.
 */
NP_API template <typename T> NP_NODISCARD auto ifft(const Ndarray<T> &x) -> Ndarray<Cplx> {
  const std::size_t nd = x.ndim();
  if (nd == 0) {
    throw std::invalid_argument("ifft requires ndim >= 1");
  }
  const std::size_t n = static_cast<std::size_t>(x.shape[nd - 1]);

  Ndarray<Cplx> out = Ndarray<Cplx>::from_data(
      x.shape, std::vector<Cplx>(x._numel(), Cplx{0.0, 0.0}));
  for (std::size_t i = 0; i < x._numel(); ++i) {
    out.data()[i] = static_cast<Cplx>(x.data()[x._flat_logical(i)]);
  }

  if (nd == 1) {
    detail::transform(out.data(), true);
    return out;
  }

  std::vector<int> outer_shape(x.shape.begin(), x.shape.end() - 1);
  np::detail::Odometer od(outer_shape);
  while (!od.done()) {
    const auto &idx = od.idx();
    std::vector<Cplx> slice(n);
    for (std::size_t p = 0; p < n; ++p) {
      std::vector<std::size_t> full = idx;
      full.push_back(p);
      slice[p] = out.data()[np::detail::flat_index(full, out.strides, 0)];
    }
    detail::transform(slice, true);
    for (std::size_t p = 0; p < n; ++p) {
      std::vector<std::size_t> full = idx;
      full.push_back(p);
      out.data()[np::detail::flat_index(full, out.strides, 0)] = slice[p];
    }
    od.advance();
  }
  return out;
}

/** @brief Elementwise magnitude |z| as a real Ndarray. */
NP_API template <typename T> NP_NODISCARD auto abs(const Ndarray<T> &x) -> Ndarray<double> {
  Ndarray<double> out(x.shape);
  for (std::size_t i = 0; i < x._numel(); ++i) {
    const Cplx v = static_cast<Cplx>(x.data()[x._flat_logical(i)]);
    out.data()[i] = std::abs(v);
  }
  return out;
}

} // namespace np::fft

#endif // NP_FFT_HPP
