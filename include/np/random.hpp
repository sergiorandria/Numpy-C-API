/**
 * @file random.hpp
 * @brief Random number generation (NumPy random.Generator API).
 *
 * Provides NumPy-compatible random number generation using C++11 <random>.
 * Implements the Generator class with all standard distributions.
 *
 * Reference: numpy-reference/reference/random/generator.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_RANDOM_HPP
#define NP_RANDOM_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "creation.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"

namespace np {
namespace random {

/**
 * @brief Random number generator (NumPy Generator equivalent).
 *
 * Wraps C++ std::mt19937_64 (Mersenne Twister) for NumPy-compatible
 * random number generation.
 *
 * Reference: numpy-reference/reference/random/generator.html
 */
class Generator {
public:
  using engine_type = std::mt19937_64;
  using result_type = engine_type::result_type;

  /**
   * @brief Construct generator with optional seed.
   * @param seed Random seed (uses random_device if not provided)
   */
  explicit Generator(std::optional<std::uint64_t> seed = std::nullopt)
      : engine_(seed.has_value() ? *seed : std::random_device{}()) {}

  // =================================================================
  // Simple Random Data
  // =================================================================

  /**
   * @brief Random integers from low (inclusive) to high (exclusive).
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.integers.html
   */
  template <typename T = std::int64_t>
  auto integers(T low, T high, const std::vector<int> &size = {})
      -> Ndarray<T> {
    if (high <= low) {
      throw std::invalid_argument("high must be greater than low");
    }

    if (size.empty()) {
      std::uniform_int_distribution<T> dist(low, high - 1);
      return Ndarray<T>::from_data({1}, {dist(engine_)});
    }

    std::uniform_int_distribution<T> dist(low, high - 1);
    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      *it = dist(engine_);
    }
    return result;
  }

  /**
   * @brief Random floats in the half-open interval [0.0, 1.0).
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.random.html
   */
  template <typename T = double>
  auto random(const std::vector<int> &size = {}) -> Ndarray<T> {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");

    if (size.empty()) {
      std::uniform_real_distribution<T> dist(T{0}, T{1});
      return Ndarray<T>::from_data({1}, {dist(engine_)});
    }

    std::uniform_real_distribution<T> dist(T{0}, T{1});
    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      *it = dist(engine_);
    }
    return result;
  }

  /**
   * @brief Random bytes.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.bytes.html
   */
  auto bytes(std::size_t length) -> std::vector<std::uint8_t> {
    std::uniform_int_distribution<std::uint16_t> dist(0, 255);
    std::vector<std::uint8_t> result(length);
    for (auto &byte : result) {
      byte = static_cast<std::uint8_t>(dist(engine_));
    }
    return result;
  }

  // =================================================================
  // Permutations
  // =================================================================

  /**
   * @brief Randomly permute a sequence or array.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.permutation.html
   */
  template <typename T> auto permutation(const Ndarray<T> &x) -> Ndarray<T> {
    auto result = x.copy();
    shuffle(result);
    return result;
  }

  /**
   * @brief Return permuted range.
   */
  auto permutation(std::int64_t n) -> Ndarray<std::int64_t> {
    auto arr = arange<std::int64_t>(0, n, 1);
    shuffle(arr);
    return arr;
  }

  /**
   * @brief Modify array in-place by shuffling its contents.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.shuffle.html
   */
  template <typename T> void shuffle(Ndarray<T> &x) {
    if (x.ndim() == 1) {
      std::shuffle(x.data().begin(), x.data().end(), engine_);
    } else {
      // Shuffle along first axis
      const std::size_t n0 = static_cast<std::size_t>(x.shape[0]);
      std::vector<std::size_t> indices(n0);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), engine_);

      // Create permuted copy
      auto result = x.copy();
      for (std::size_t i = 0; i < n0; ++i) {
        // Copy row indices[i] to row i
        for (std::size_t j = 0; j < x.size() / n0; ++j) {
          x.data()[i * (x.size() / n0) + j] =
              result.data()[indices[i] * (x.size() / n0) + j];
        }
      }
    }
  }

  /**
   * @brief Random choice from 1-D array.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.choice.html
   */
  template <typename T>
  auto choice(const Ndarray<T> &a, std::size_t size = 1, bool replace = true)
      -> Ndarray<T> {
    if (a.ndim() != 1) {
      throw std::invalid_argument("choice: array must be 1-D");
    }

    const std::size_t n = a.size();
    if (!replace && size > n) {
      throw std::invalid_argument("choice: Cannot sample more elements than "
                                  "population when replace=False");
    }

    std::vector<T> result_data;
    result_data.reserve(size);

    if (replace) {
      std::uniform_int_distribution<std::size_t> dist(0, n - 1);
      for (std::size_t i = 0; i < size; ++i) {
        result_data.push_back(a.data()[dist(engine_)]);
      }
    } else {
      // Sampling without replacement
      std::vector<std::size_t> indices(n);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), engine_);

      for (std::size_t i = 0; i < size; ++i) {
        result_data.push_back(a.data()[indices[i]]);
      }
    }

    return Ndarray<T>::from_data({static_cast<int>(size)},
                                 std::move(result_data));
  }

  // =================================================================
  // Distributions
  // =================================================================

  /**
   * @brief Draw samples from a uniform distribution over [low, high).
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.uniform.html
   */
  template <typename T = double>
  auto uniform(T low = T{0}, T high = T{1}, const std::vector<int> &size = {})
      -> Ndarray<T> {
    std::uniform_real_distribution<T> dist(low, high);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a standard normal distribution (mean=0, stdev=1).
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.standard_normal.html
   */
  template <typename T = double>
  auto standard_normal(const std::vector<int> &size = {}) -> Ndarray<T> {
    std::normal_distribution<T> dist(T{0}, T{1});
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a normal (Gaussian) distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.normal.html
   */
  template <typename T = double>
  auto normal(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {})
      -> Ndarray<T> {
    std::normal_distribution<T> dist(loc, scale);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from an exponential distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.exponential.html
   */
  template <typename T = double>
  auto exponential(T scale = T{1}, const std::vector<int> &size = {})
      -> Ndarray<T> {
    std::exponential_distribution<T> dist(T{1} / scale);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a standard exponential distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.standard_exponential.html
   */
  template <typename T = double>
  auto standard_exponential(const std::vector<int> &size = {}) -> Ndarray<T> {
    return exponential(T{1}, size);
  }

  /**
   * @brief Draw samples from a gamma distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.gamma.html
   */
  template <typename T = double>
  auto gamma(T shape, T scale = T{1}, const std::vector<int> &size = {})
      -> Ndarray<T> {
    std::gamma_distribution<T> dist(shape, scale);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a standard gamma distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.standard_gamma.html
   */
  template <typename T = double>
  auto standard_gamma(T shape, const std::vector<int> &size = {})
      -> Ndarray<T> {
    return gamma(shape, T{1}, size);
  }

  /**
   * @brief Draw samples from a beta distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.beta.html
   */
  template <typename T = double>
  auto beta(T a, T b, const std::vector<int> &size = {}) -> Ndarray<T> {
    // Beta distribution: X ~ Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
    std::gamma_distribution<T> dist_a(a, T{1});
    std::gamma_distribution<T> dist_b(b, T{1});

    if (size.empty()) {
      T x = dist_a(engine_);
      T y = dist_b(engine_);
      return Ndarray<T>::from_data({1}, {x / (x + y)});
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      T x = dist_a(engine_);
      T y = dist_b(engine_);
      *it = x / (x + y);
    }
    return result;
  }

  /**
   * @brief Draw samples from a chi-square distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.chisquare.html
   */
  template <typename T = double>
  auto chisquare(T df, const std::vector<int> &size = {}) -> Ndarray<T> {
    std::chi_squared_distribution<T> dist(df);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from an F distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.f.html
   */
  template <typename T = double>
  auto f(T dfnum, T dfden, const std::vector<int> &size = {}) -> Ndarray<T> {
    std::fisher_f_distribution<T> dist(dfnum, dfden);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a Student's t distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.standard_t.html
   */
  template <typename T = double>
  auto standard_t(T df, const std::vector<int> &size = {}) -> Ndarray<T> {
    std::student_t_distribution<T> dist(df);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a lognormal distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.lognormal.html
   */
  template <typename T = double>
  auto lognormal(T mean = T{0}, T sigma = T{1},
                 const std::vector<int> &size = {}) -> Ndarray<T> {
    std::lognormal_distribution<T> dist(mean, sigma);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a Cauchy distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.standard_cauchy.html
   */
  template <typename T = double>
  auto standard_cauchy(const std::vector<int> &size = {}) -> Ndarray<T> {
    std::cauchy_distribution<T> dist(T{0}, T{1});
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a Weibull distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.weibull.html
   */
  template <typename T = double>
  auto weibull(T a, const std::vector<int> &size = {}) -> Ndarray<T> {
    std::weibull_distribution<T> dist(a, T{1});
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a Poisson distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.poisson.html
   */
  template <typename T = double>
  auto poisson(T lam = T{1}, const std::vector<int> &size = {})
      -> Ndarray<_Np_dtype::_Np_int64> {
    std::poisson_distribution<std::int64_t> dist(lam);
    return _fill_distribution<_Np_dtype::_Np_int64>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a binomial distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.binomial.html
   */
  auto binomial(std::int64_t n, double p, const std::vector<int> &size = {})
      -> Ndarray<_Np_dtype::_Np_int64> {
    std::binomial_distribution<std::int64_t> dist(n, p);
    return _fill_distribution<_Np_dtype::_Np_int64>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a negative binomial distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.negative_binomial.html
   */
  auto negative_binomial(std::int64_t n, double p,
                         const std::vector<int> &size = {})
      -> Ndarray<_Np_dtype::_Np_int64> {
    std::negative_binomial_distribution<std::int64_t> dist(n, p);
    return _fill_distribution<_Np_dtype::_Np_int64>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a geometric distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.geometric.html
   */
  auto geometric(double p, const std::vector<int> &size = {})
      -> Ndarray<_Np_dtype::_Np_int64> {
    std::geometric_distribution<std::int64_t> dist(p);
    return _fill_distribution<_Np_dtype::_Np_int64>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a Pareto II distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.pareto.html
   */
  template <typename T = double>
  auto pareto(T a, const std::vector<int> &size = {}) -> Ndarray<T> {
    // Pareto: X = (1/U)^(1/a) - 1, where U ~ Uniform(0,1)
    std::uniform_real_distribution<T> dist(T{0}, T{1});

    if (size.empty()) {
      T u = dist(engine_);
      return Ndarray<T>::from_data({1}, {std::pow(T{1} / u, T{1} / a) - T{1}});
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      T u = dist(engine_);
      *it = std::pow(T{1} / u, T{1} / a) - T{1};
    }
    return result;
  }

  /**
   * @brief Draw samples from a power distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.power.html
   */
  template <typename T = double>
  auto power(T a, const std::vector<int> &size = {}) -> Ndarray<T> {
    // Power: X = U^(1/a), where U ~ Uniform(0,1)
    std::uniform_real_distribution<T> dist(T{0}, T{1});

    if (size.empty()) {
      return Ndarray<T>::from_data({1}, {std::pow(dist(engine_), T{1} / a)});
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      *it = std::pow(dist(engine_), T{1} / a);
    }
    return result;
  }

  /**
   * @brief Draw samples from a Laplace distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.laplace.html
   */
  template <typename T = double>
  auto laplace(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {})
      -> Ndarray<T> {
    // Laplace: X = loc - scale*sign(U-0.5)*log(1-2*|U-0.5|)
    std::uniform_real_distribution<T> dist(T{0}, T{1});

    if (size.empty()) {
      T u = dist(engine_);
      T sign = (u < T{0.5}) ? T{-1} : T{1};
      T val = loc - scale * sign * std::log(T{1} - T{2} * std::abs(u - T{0.5}));
      return Ndarray<T>::from_data({1}, {val});
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      T u = dist(engine_);
      T sign = (u < T{0.5}) ? T{-1} : T{1};
      *it = loc - scale * sign * std::log(T{1} - T{2} * std::abs(u - T{0.5}));
    }
    return result;
  }

  /**
   * @brief Draw samples from a Gumbel distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.gumbel.html
   */
  template <typename T = double>
  auto gumbel(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {})
      -> Ndarray<T> {
    std::extreme_value_distribution<T> dist(loc, scale);
    return _fill_distribution<T>(engine_, dist, size);
  }

  /**
   * @brief Draw samples from a logistic distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.logistic.html
   */
  template <typename T = double>
  auto logistic(T loc = T{0}, T scale = T{1}, const std::vector<int> &size = {})
      -> Ndarray<T> {
    // Logistic: X = loc + scale * log(U/(1-U))
    std::uniform_real_distribution<T> dist(T{0}, T{1});

    if (size.empty()) {
      T u = dist(engine_);
      return Ndarray<T>::from_data({1},
                                   {loc + scale * std::log(u / (T{1} - u))});
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      T u = dist(engine_);
      *it = loc + scale * std::log(u / (T{1} - u));
    }
    return result;
  }

  /**
   * @brief Draw samples from a Rayleigh distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.rayleigh.html
   */
  template <typename T = double>
  auto rayleigh(T scale = T{1}, const std::vector<int> &size = {})
      -> Ndarray<T> {
    // Rayleigh: X = scale * sqrt(-2*log(U))
    std::uniform_real_distribution<T> dist(T{0}, T{1});

    if (size.empty()) {
      T u = dist(engine_);
      return Ndarray<T>::from_data({1},
                                   {scale * std::sqrt(-T{2} * std::log(u))});
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      T u = dist(engine_);
      *it = scale * std::sqrt(-T{2} * std::log(u));
    }
    return result;
  }

  /**
   * @brief Draw samples from a triangular distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.triangular.html
   */
  template <typename T = double>
  auto triangular(T left, T mode, T right, const std::vector<int> &size = {})
      -> Ndarray<T> {
    // Use inverse CDF method for triangular distribution
    std::uniform_real_distribution<T> dist(T{0}, T{1});
    const T fc = (mode - left) / (right - left);

    if (size.empty()) {
      T u = dist(engine_);
      T val =
          (u < fc)
              ? left + std::sqrt(u * (right - left) * (mode - left))
              : right - std::sqrt((T{1} - u) * (right - left) * (right - mode));
      return Ndarray<T>::from_data({1}, {val});
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      T u = dist(engine_);
      *it =
          (u < fc)
              ? left + std::sqrt(u * (right - left) * (mode - left))
              : right - std::sqrt((T{1} - u) * (right - left) * (right - mode));
    }
    return result;
  }

  /**
   * @brief Draw samples from a hypergeometric distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.hypergeometric.html
   */
  auto hypergeometric(std::int64_t ngood, std::int64_t nbad,
                      std::int64_t nsample, const std::vector<int> &size = {})
      -> Ndarray<std::int64_t> {
    if (ngood < 0 || nbad < 0 || nsample < 0 || nsample > ngood + nbad) {
      throw std::invalid_argument("hypergeometric: invalid parameters");
    }

    // Draw nsample items without replacement from ngood + nbad and
    // count how many of the drawn items were "good". std has no
    // hypergeometric_distribution before C++26, so sample directly.
    auto sample_one = [&]() -> std::int64_t {
      std::int64_t good = ngood;
      std::int64_t bad = nbad;
      std::int64_t successes = 0;
      std::uniform_real_distribution<double> u(0.0, 1.0);
      for (std::int64_t d = 0; d < nsample; ++d) {
        const double p = static_cast<double>(good) / static_cast<double>(good + bad);
        if (u(engine_) < p) {
          ++successes;
          --good;
        } else {
          --bad;
        }
      }
      return successes;
    };

    if (size.empty()) {
      return Ndarray<std::int64_t>::from_data({1}, {sample_one()});
    }

    Ndarray<std::int64_t> result(size, dtype_of<std::int64_t>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      *it = sample_one();
    }
    return result;
  }

  /**
   * @brief Draw samples from a logarithmic series distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.logseries.html
   */
  template <typename T = double>
  auto logseries(T p, const std::vector<int> &size = {})
      -> Ndarray<std::int64_t> {
    // Log-series: P(X=k) = -p^k / (k * log(1-p))
    std::uniform_real_distribution<T> dist(T{0}, T{1});
    const T log_q = std::log(T{1} - p);

    if (size.empty()) {
      T u = dist(engine_);
      T v = dist(engine_);
      std::int64_t k = 1;
      if (u >= p) {
        k = static_cast<std::int64_t>(std::floor(std::log(v) / log_q)) + 1;
      }
      return Ndarray<std::int64_t>::from_data({1}, {k});
    }

    Ndarray<std::int64_t> result(size, dtype_of<std::int64_t>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      T u = dist(engine_);
      T v = dist(engine_);
      std::int64_t k = 1;
      if (u >= p) {
        k = static_cast<std::int64_t>(std::floor(std::log(v) / log_q)) + 1;
      }
      *it = k;
    }
    return result;
  }

  /**
   * @brief Draw samples from a Wald (inverse Gaussian) distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.wald.html
   */
  template <typename T = double>
  auto wald(T mean, T scale, const std::vector<int> &size = {}) -> Ndarray<T> {
    // Wald/Inverse Gaussian: use transformation method
    std::normal_distribution<T> normal(T{0}, T{1});
    std::uniform_real_distribution<T> uniform(T{0}, T{1});

    if (size.empty()) {
      T nu = normal(engine_);
      T y = nu * nu;
      T x = mean + (mean * mean * y) / (T{2} * scale) -
            (mean / (T{2} * scale)) *
                std::sqrt(T{4} * mean * scale * y + mean * mean * y * y);
      T u = uniform(engine_);
      if (u <= mean / (mean + x)) {
        return Ndarray<T>::from_data({1}, {x});
      } else {
        return Ndarray<T>::from_data({1}, {mean * mean / x});
      }
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      T nu = normal(engine_);
      T y = nu * nu;
      T x = mean + (mean * mean * y) / (T{2} * scale) -
            (mean / (T{2} * scale)) *
                std::sqrt(T{4} * mean * scale * y + mean * mean * y * y);
      T u = uniform(engine_);
      *it = (u <= mean / (mean + x)) ? x : mean * mean / x;
    }
    return result;
  }

  /**
   * @brief Draw samples from a von Mises distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.vonmises.html
   */
  template <typename T = double>
  auto vonmises(T mu, T kappa, const std::vector<int> &size = {})
      -> Ndarray<T> {
    // Von Mises: circular normal distribution
    // Use Best-Fisher algorithm
    std::uniform_real_distribution<T> uniform(T{0}, T{1});

    auto sample_one = [&]() -> T {
      T a = T{1} + std::sqrt(T{1} + T{4} * kappa * kappa);
      T b = (a - std::sqrt(T{2} * a)) / (T{2} * kappa);
      T r = (T{1} + b * b) / (T{2} * b);

      while (true) {
        T u1 = uniform(engine_);
        T z = std::cos(std::numbers::pi_v<T> * u1);
        T f = (T{1} + r * z) / (r + z);
        T c = kappa * (r - f);

        T u2 = uniform(engine_);
        if (u2 < c * (T{2} - c) || u2 <= c * std::exp(T{1} - c)) {
          T u3 = uniform(engine_);
          T theta = (u3 < T{0.5}) ? std::acos(f) : -std::acos(f);
          return mu + theta;
        }
      }
    };

    if (size.empty()) {
      return Ndarray<T>::from_data({1}, {sample_one()});
    }

    Ndarray<T> result(size, dtype_of<T>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      *it = sample_one();
    }
    return result;
  }

  /**
   * @brief Draw samples from a Zipf distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.zipf.html
   */
  template <typename T = double>
  auto zipf(T a, const std::vector<int> &size = {}) -> Ndarray<std::int64_t> {
    // Zipf: P(k) ~ 1/k^a
    // Use rejection sampling
    std::uniform_real_distribution<T> uniform(T{0}, T{1});
    const T am1 = a - T{1};
    const T b = std::pow(T{2}, am1);

    auto sample_one = [&]() -> std::int64_t {
      while (true) {
        T u = uniform(engine_);
        T v = uniform(engine_);
        std::int64_t x =
            static_cast<std::int64_t>(std::floor(std::pow(u, -T{1} / am1)));
        T t = std::pow(T{1} + T{1} / static_cast<T>(x), am1);
        if (v * x * (t - T{1}) / (b - T{1}) <= t / b) {
          return x;
        }
      }
    };

    if (size.empty()) {
      return Ndarray<std::int64_t>::from_data({1}, {sample_one()});
    }

    Ndarray<std::int64_t> result(size, dtype_of<std::int64_t>);
    for (auto it = result.begin(); it != result.end(); ++it) {
      *it = sample_one();
    }
    return result;
  }

  /**
   * @brief Draw samples from a multinomial distribution.
   * Reference:
   * numpy-reference/reference/random/generated/numpy.random.Generator.multinomial.html
   */
  auto multinomial(std::int64_t n, const std::vector<double> &pvals,
                   const std::vector<int> &size = {}) -> Ndarray<std::int64_t> {
    const std::size_t k = pvals.size();

    // Verify probabilities sum to 1
    double sum = 0.0;
    for (double p : pvals) {
      sum += p;
    }
    if (std::abs(sum - 1.0) > 1e-7) {
      throw std::invalid_argument("multinomial: pvals must sum to 1");
    }

    if (size.empty()) {
      // Single sample
      std::vector<std::int64_t> counts(k, 0);
      std::int64_t remaining = n;

      for (std::size_t i = 0; i < k - 1; ++i) {
        double p = pvals[i] / (1.0 - std::accumulate(pvals.begin(),
                                                     pvals.begin() + i, 0.0));
        std::binomial_distribution<std::int64_t> dist(remaining, p);
        counts[i] = dist(engine_);
        remaining -= counts[i];
      }
      counts[k - 1] = remaining;

      return Ndarray<std::int64_t>::from_data({static_cast<int>(k)},
                                              std::move(counts));
    }

    // Multiple samples
    std::vector<int> out_shape = size;
    out_shape.push_back(static_cast<int>(k));
    Ndarray<std::int64_t> result(out_shape, dtype_of<std::int64_t>);

    const std::size_t n_samples = result.size() / k;
    for (std::size_t s = 0; s < n_samples; ++s) {
      std::int64_t remaining = n;
      for (std::size_t i = 0; i < k - 1; ++i) {
        double p_adjusted = pvals[i];
        double sum_prev = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
          sum_prev += pvals[j];
        }
        if (sum_prev < 1.0) {
          p_adjusted = pvals[i] / (1.0 - sum_prev);
        }

        std::binomial_distribution<std::int64_t> dist(remaining, p_adjusted);
        std::int64_t count = dist(engine_);
        result.data()[s * k + i] = count;
        remaining -= count;
      }
      result.data()[s * k + (k - 1)] = remaining;
    }

    return result;
  }

  // =================================================================
  // Helper Methods
  // =================================================================

  /**
   * @brief Get the underlying random engine.
   */
  engine_type &engine() { return engine_; }

  /**
   * @brief Get the underlying random engine (const).
   */
  const engine_type &engine() const { return engine_; }

private:
  engine_type engine_;

  /**
   * @brief Fill an array using a distribution.
   */
  template <typename TargetType, typename Dist, typename Engine>
  auto _fill_distribution(Engine &rng, Dist &dist, const std::vector<int> &size)
      -> Ndarray<TargetType> {
    if (size.empty()) {
      return Ndarray<TargetType>::from_data(
          {1}, {(dist(rng))});
    }

    std::size_t total_elements = 1;
    for (int d : size) {
      total_elements *= static_cast<std::size_t>(d);
    }

    Ndarray<TargetType> result(size, dtype_of<TargetType>);
    for (std::size_t i = 0; i < total_elements; ++i) {
      result[i] = static_cast<TargetType>(dist(rng));
    }
    return result;
  }
};

// =================================================================
// Convenience Functions (Module-Level API)
// =================================================================

namespace {
// Thread-local default generator
thread_local Generator default_generator_;
} // namespace

/**
 * @brief Get or create the default random generator.
 * Reference:
 * numpy-reference/reference/random/generated/numpy.random.default_rng.html
 */
inline Generator &
default_rng(std::optional<std::uint64_t> seed = std::nullopt) {
  if (seed.has_value()) {
    default_generator_ = Generator(*seed);
  }
  return default_generator_;
}

// Convenience wrappers using default generator

/** @brief Random integers using default generator. */
template <typename T = std::int64_t>
inline auto randint(T low, T high, const std::vector<int> &size = {})
    -> Ndarray<T> {
  return default_rng().integers(low, high, size);
}

/** @brief Random floats [0, 1) using default generator. */
template <typename T = double>
inline auto rand(const std::vector<int> &size = {}) -> Ndarray<T> {
  return default_rng().random<T>(size);
}

/** @brief Standard normal using default generator. */
template <typename T = double>
inline auto randn(const std::vector<int> &size = {}) -> Ndarray<T> {
  return default_rng().standard_normal<T>(size);
}

/** @brief Uniform distribution using default generator. */
template <typename T = double>
inline auto uniform(T low = T{0}, T high = T{1},
                    const std::vector<int> &size = {}) -> Ndarray<T> {
  return default_rng().uniform(low, high, size);
}

/** @brief Normal distribution using default generator. */
template <typename T = double>
inline auto normal(T loc = T{0}, T scale = T{1},
                   const std::vector<int> &size = {}) -> Ndarray<T> {
  return default_rng().normal(loc, scale, size);
}

/** @brief Exponential distribution using default generator. */
template <typename T = double>
inline auto exponential(T scale = T{1}, const std::vector<int> &size = {})
    -> Ndarray<T> {
  return default_rng().exponential(scale, size);
}

/** @brief Permutation using default generator. */
template <typename T>
inline auto permutation(const Ndarray<T> &x) -> Ndarray<T> {
  return default_rng().permutation(x);
}

/** @brief Permutation of range using default generator. */
inline auto permutation(std::int64_t n) -> Ndarray<std::int64_t> {
  return default_rng().permutation(n);
}

/** @brief Shuffle using default generator. */
template <typename T> inline void shuffle(Ndarray<T> &x) {
  default_rng().shuffle(x);
}

/** @brief Choice using default generator. */
template <typename T>
inline auto choice(const Ndarray<T> &a, std::size_t size = 1,
                   bool replace = true) -> Ndarray<T> {
  return default_rng().choice(a, size, replace);
}

} // namespace random
} // namespace np

#endif // NP_RANDOM_HPP
